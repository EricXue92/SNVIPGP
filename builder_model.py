
import torch
import torch.nn.functional as F
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood
from due import dkl
from due.convnext import SimpleMLP # SimpleConvNet, SimpleResnet
from lib.evaluate_cp import ConformalTrainingLoss
from sngp_wrapper.covert_utils import replace_layer_with_gaussian, convert_to_sn_my
import os

NUM_WORKERS = os.cpu_count()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(args, num_classes, train_dataset):
    if args.sngp:
        model = SimpleMLP(num_classes=args.number_of_class)
        #model = SimpleResnet(num_classes=args.number_of_class)
        GP_KWARGS = {
            'num_inducing': 256,
            'gp_scale': 1.0,
            'gp_bias': 0.,
            'gp_kernel_type': 'gaussian',
            'gp_input_normalization': True,
            'gp_cov_discount_factor': -1,
            'gp_cov_ridge_penalty': 1.,
            'gp_output_bias_trainable': False,
            'gp_scale_random_features': False,
            'gp_use_custom_random_features': True,
            'gp_random_feature_type': 'orf',
            'gp_output_imagenet_initializer': True,
            'num_classes': 10, ##### 4
        }
        model = convert_to_sn_my(model, args.spec_norm_replace_list, args.coeff)
        replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
        if args.conformal_training:
            loss_fn = ConformalTrainingLoss(alpha=args.alpha, beta=args.beta, temperature=args.temperature, sngp_flag=True, args=args)
        else:
            loss_fn = F.cross_entropy
        likelihood = None
    else:
        #feature_extractor = SimpleResnet(num_classes=None)
        feature_extractor = SimpleMLP(num_classes=None)
        feature_extractor = convert_to_sn_my(feature_extractor, args.spec_norm_replace_list, args.coeff)
        initial_inducing_points, initial_lengthscale = dkl.initial_values(train_dataset, feature_extractor, args.n_inducing_points)
        gp = dkl.GP(
            num_outputs=num_classes,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=args.kernel
        )
        model = dkl.DKL(feature_extractor, gp)
        likelihood = SoftmaxLikelihood(num_features=num_classes, num_classes=num_classes, mixing_weights=False)
        likelihood = likelihood.cuda()
        elbo_fn = VariationalELBO(likelihood, gp, num_data=len(train_dataset))
        loss_fn = lambda x, y: -elbo_fn(x, y)

    model = model.to(device)

    return model, likelihood, loss_fn


