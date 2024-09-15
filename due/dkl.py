import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP

from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

from sklearn import cluster

# 参数:（训练数据, 用于特征转换的神经网络, inducing points的数量）
# 函数: 训练数据 -> 神经网络 -> 转换后的数据特征 +  inducing points 的数据
        # 调用 _get_initial_inducing_points 函数，得到 inducing points 位置坐标
        # 调用 _get_initial_lengthscale 函数，得到形状参数
# return: initial inducing points 的位置坐标 和 形状参数
def initial_values(train_dataset, feature_extractor, n_inducing_points):
    
    steps = 10
    # Generate a random permutation of indices and select the first 1000
    
    # Split the indices into 'steps' number of chunks
    # 1、生成和训练数据一样长度的数值, 把数组随机打乱, 取前1000个值, 分成10份
    # idx 是包括 每100个的indexs的tuple  
    # Eg. torch.arange(11).chunk(6) -- (tensor([0, 1]), tensor([2, 3]), ...)
    # # https://pytorch.org/docs/stable/generated/torch.chunk.html

    idx = torch.randperm( len(train_dataset) )[:1000].chunk(steps) # return a tuple 
    
    # 用来存储 把训练数据 经过神经网络换后的 输出数据
    f_X_samples = []

    # 一共10块数据index， 每一块数据 NN 转换，
    with torch.no_grad():
        for i in range(steps):
            # torch.stack will introduce a new dimension
            # 把每100个数 叠加在一起，组成一个X_sample
            X_sample = torch.stack( [ train_dataset[j][0] for j in idx[i] ]) # axis = 0
            if torch.cuda.is_available():
                X_sample = X_sample.cuda()
                feature_extractor = feature_extractor.cuda()
            # X_sample is like a training batch 
            f_X_samples.append( feature_extractor(X_sample).cpu() )  
            
    # torch.cat joins the tensors along an existing dimension without adding any new dimensions.
    # torch.cat() can be seen as an inverse operation for torch.split() and torch.chunk()
    f_X_samples = torch.cat(f_X_samples) # dim=0
    
    # 获取 经过特征转换后 的 初始inducing points 的位置
    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points
    )
    # 得到 初始化形状参数参数 和 inducing points 的位置
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)
    return initial_inducing_points, initial_lengthscale


#### 初始化的 10个 inducing points 位置（针对已经转换过的特征）
# 参数: (经过神经网络转换后的特数据特征, inducing_points的数量)
# 函数: 数据特征 -> Kmeans方法（分成10份）--> 得到10个centroids的中心点 
# return (num_of_inducing points, 神经网络转换后特征维度) E.g. ( 10 , 128 )
def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    # MiniBatchKMeans -> handle large datasets more efficiently by using mini-batches to update the cluster centroids,
    kmeans = cluster.MiniBatchKMeans(
        n_clusters = n_inducing_points, 
        batch_size = n_inducing_points * 10
    )
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)
    return initial_inducing_points

# 初始化lengthscale形状参数，lengthscale = 所有数据点距离相互之间的平均值 
def _get_initial_lengthscale(f_X_samples):
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()
    # torch.pdist -> Computes the p-norm （2） distance between every pair of row vectors in the input. 
    #Input shape （ N , M ） --> return  (1/2 * N * （N + 1） , ) 每2个点之间的距离组成的1维向量
    initial_lengthscale = torch.pdist(f_X_samples).mean()
    return initial_lengthscale.cpu()

# 定义一个近似的 高斯过程 模型
class GP(ApproximateGP):
    def __init__(
        self,
        num_outputs, 
        initial_lengthscale, 
        initial_inducing_points, # (num_of_inducing_points, 特征维度)
        kernel="RBF",
    ):
        n_inducing_points = initial_inducing_points.shape[0]

        if num_outputs > 1:
            
            batch_shape = torch.Size([num_outputs])
            
        else:
            batch_shape = torch.Size([])

        # # Variational/approximate distribution --> 定义一个后验分布
        # define the form of the approximate inducing value posterior q(u)
        
        # This tells us what form the variational distribution q(u) should take
        # handling multiple independent GPs or mini-batch training.
        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points, batch_shape = batch_shape
        )
        
        # # Variational strategy initialization --> 定义一个如何优化后验q(u)的策略，从而能得到q(f)
        # define how to compute 𝑞( 𝐟(𝐗) ) from 𝑞(𝐮)
        
        # This tells us how to transform a distribution q(u) over the inducing point values to 
        # a distribution q(f) over the latent function values for some input x.
        variational_strategy = VariationalStrategy(
            self, initial_inducing_points, variational_distribution
        )
        
        # The IndependentMultitaskVariationalStrategy wraps around an existing VariationalStrategy,
        # extending it to handle multiple tasks.
        
        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks = num_outputs
            )
            
        super().__init__(variational_strategy)
        
        kwargs = {
            "batch_shape": batch_shape,
        }

        if kernel == "RBF":
            kernel = RBFKernel(**kwargs)
        elif kernel == "Matern12":
            kernel = MaternKernel(nu=1 / 2, **kwargs)
        elif kernel == "Matern32":
            kernel = MaternKernel(nu=3 / 2, **kwargs)
        elif kernel == "Matern52":
            kernel = MaternKernel(nu=5 / 2, **kwargs)
        elif kernel == "RQ":
            kernel = RQKernel(**kwargs)
        else:
            raise ValueError("Specified kernel not known.")

        kernel.lengthscale = initial_lengthscale * torch.ones_like( kernel.lengthscale )
        
        self.mean_module = ConstantMean(batch_shape= batch_shape)
        self.covar_module = ScaleKernel(kernel, batch_shape=batch_shape)
        
        # forward method computes the GP's mean and covariance for input x.
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)
    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param

# 初始化参数：feature_extractor 和 gp
# 过程： x -> 神经网络 self.feature_extractor(x) -> features -> 输入给 定义的高斯随机过程模型 --> 待训练的高斯随机模型
# 得到： 最终需要训练的高斯随机过程模型
class DKL(gpytorch.Module):
    def __init__(self, feature_extractor, gp):
        """
        This wrapper class is necessary because ApproximateGP (above) does some magic
        on the forward method which is not compatible with a feature_extractor.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp = gp
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp(features)