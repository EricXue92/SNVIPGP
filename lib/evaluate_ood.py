import numpy as np
import torch
import torch.nn.functional as F
import gpytorch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from .datasets import get_dataset

# 函数： 组合IID和OOD数据，并给与对应的数据标签[0,0,..,1,1,...]
# 参数： IID 和 OOD 数据
# 返回： 组合的数据, 和对应的0,1标签
def prepare_ood_datasets(true_dataset, ood_dataset):
    
    # All TorchVision datasets have two parameters:
    #   transform to modify the features
    #   target_transform to modify the labels
    
    # setting the transformation of ood_dataset to be the same as true_dataset. 
    # This is useful if you want both datasets to undergo the same preprocessing or augmentation steps.
    ood_dataset.transform = true_dataset.transform
    datasets = [true_dataset, ood_dataset]

    # torch.cat( tensors, dim = 0 )
    # torch.cat 默认axis = 0 的方向 结合数据，不增加维度，保持数据形状不变
    
    anomaly_targets = torch.cat(
        ( torch.zeros(len(true_dataset) ), torch.ones( len(ood_dataset) ) )
    ) # 得到tensor([0., 0..., 1., 1...])

    # concatenate multiple datasets into a single dataset
    concat_datasets = torch.utils.data.ConcatDataset( datasets )
    
    # wraps an iterable around the Dataset to enable easy access to the samples.
    dataloader = torch.utils.data.DataLoader(
        concat_datasets, batch_size = 64, shuffle = False, num_workers = 4, pin_memory = True
    )
    return dataloader, anomaly_targets


# 利用训练好的模型来预测组数据，从而获得预测的精确度和得分（不确定性）
# 高斯不确定性得分 = entropy （ -output* log(output) ）
# sngp 不确定 = num_classes / (belief_mass + num_classes)
# 高斯过程的uncertainty 是预测值的 entropy 

# 函数: 训练好的模型 遍历 数据，来得到精确度和OOD检测得分
# 输入: 训练好的模型, 训练好的likelihood, 组合的数据
# 返回: OOD得分, 精确度
def loop_over_dataloader(model, likelihood, dataloader):
    model.eval()
    if likelihood is not None:
        likelihood.eval()
    with torch.no_grad():
        scores = []
        accuracies = []
        
        for data, target in dataloader:
            data = data.cuda()
            target = target.cuda()
            
            if likelihood is None:
                logits = model(data)
                output = F.softmax(logits, dim=1) 
                # Dempster-Shafer uncertainty for SNGP
                # From: https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/ood_utils.py#L22
                # K/(K + sum( exp(logits) ) )
                num_classes = logits.shape[1]
                belief_mass = logits.exp().sum(1)
                uncertainty = num_classes / (belief_mass + num_classes)
            else:
                # sets the number of samples to draw from the likelihood to 32 
                # approximating the predictive distribution when exact inference is computationally expensive or intractable
                
                # model(test_X) (x*): returns the model posterior distribution p(f* | x*, X, y), 返回高斯函数有一个确定的均值和方差
                # likelihood(model(test_x)) : gives us the posterior predictive distribution p(y* | x*, X, y) 
                # which is the probability distribution over the predicted output value
                
                # This sets the number of samples used to approximate the likelihood to 32 within the context.
                with gpytorch.settings.num_likelihood_samples(32):
                    # Make predictions with the model and convert to a data-independent distribution
                    y_pred = model(data).to_data_independent_dist()
                    # Apply the likelihood to the model's predictions to get the predictive distribution
                    predictive_dist = likelihood(y_pred)
                    # Access the probabilities from the predictive distribution
                    probs = predictive_dist.probs 
                    # Compute the mean of the probabilities across the samples
                    output = probs.mean(0) # (batch_size, 4)
                    
                # entropy -> Higher entropy indicates higher uncertainty.
                # element-wise product of the predicted probabilities and their logarithms
                # in_distribution 标签为0, OOD标签为1
                # uncertainty 越大则是OOD， 越小则是in_distribution
                uncertainty = -( output * output.log() ).sum(1)
                
            # 每个batch计算预测精度和不确定性
            pred = torch.argmax(output, dim=1)
            accuracy = pred.eq(target)

            accuracies.append(accuracy.cpu().numpy())
            scores.append(uncertainty.cpu().numpy())

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)
    return scores, accuracies

# 返回IID数据的平均预测精度, 和OOD 检测表现（auroc,  aupr）
def get_ood_metrics(in_dataset, out_dataset, model, likelihood=None): # , root="./"
    
    # 返回 input_size, num_classes, train_dataset, test_dataset
    _, _, _, _, in_dataset = get_dataset(in_dataset) # , root=root
    _, _, _, _, out_dataset = get_dataset(out_dataset) # , root=root

    # 准备 结合的数据（IID, OOD) 和标签 （0,0..., 1,1...)
    dataloader, anomaly_targets = prepare_ood_datasets(in_dataset, out_dataset)
    # 返回OOD得分和精确度
    scores, accuracies = loop_over_dataloader(model, likelihood, dataloader)

    # in_distribution的预测平均精度 (只需要计算IID数据的精确度)
    accuracy = np.mean( accuracies[: len(in_dataset)] )
    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    # it can sometimes be misleading in the context of highly imbalanced dataset
    # Higher AUROC: Indicates better model performance.
    auroc = roc_auc_score(anomaly_targets, scores) 
    precision, recall, _ = precision_recall_curve(anomaly_targets, scores)
    # (0 for normal, 1 for anomaly)
    # A value closer to 1 indicates better performance.
    # Area Under the Precision-Recall Curve (AUPR)
    aupr = auc(recall, precision) 
    return accuracy, auroc,  aupr

##########？？？？
def get_auroc_classification(data, model, likelihood=None):
    if isinstance(data, torch.utils.data.Dataset):
        dataloader = torch.utils.data.DataLoader(
            data, batch_size= 64, shuffle=False, num_workers=4, pin_memory=True
        )
    else:
        dataloader = data
    # Get scores and accuracies by looping over the DataLoader
    scores, accuracies = loop_over_dataloader(model, likelihood, dataloader)
    # Compute the mean accuracy
    accuracy = np.mean(accuracies)
    # Compute the AUROC
    roc_auc = roc_auc_score(1 - accuracies, scores)
    return accuracy, roc_auc
