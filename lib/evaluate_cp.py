import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from .datasets import get_dataset
from lib.datasets import get_Brain_tumor, get_Alzheimer


class ConformalTrainingLoss(nn.Module):
    # alpha coverage, beta, mixed with cross entropy
    def __init__(self, alpha, beta, temperature, sngp_flag):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.sngp_flag = sngp_flag

    def forward(self, probabilities, y):
        # [ont] important notes 1: probabilities = torch.log(probabilities)
        
        # Extracts probabilities corresponding to the true labels
        conformity_score = probabilities[ torch.arange(len(probabilities)), y]
        
        # Computes the quantile threshold based on alpha
        tau = torch.quantile(conformity_score, self.alpha)
        
        # Uses a sigmoid function to scale probabilities relative to tau
        in_set_prob = F.sigmoid( (probabilities - tau) / self.temperature )
        
        # print("in_set_prob", in_set_prob, in_set_prob.shape)
        
        # Clamping: Ensures that the result is not less than 0, preventing negative values after subtraction
        # Computes a penalty based on the size of prediction sets
        # [ont] important notes 2: as we have relative small num of classes, there fore, consider remove the torch.log
        size_loss = torch.log( torch.clamp( in_set_prob.sum(axis=1) - 1, min = 0).mean(axis=0))
        size_loss = self.beta * size_loss
        
        if self.sngp_flag: 
            fn_loss = F.cross_entropy(probabilities, y)
            total_loss = fn_loss + size_loss 
            print("total loss", (size_loss).item() + fn_loss.item(), "size loss", (size_loss).item(), "ce loss", fn_loss.item())
            return total_loss 
        else:
            print(f"size loss is {size_loss}")
            return size_loss 

# 
def tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    
    cal_scores = 1 - cal_smx[torch.arange(n), cal_labels]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    
    qhat = torch.quantile(cal_scores, q_level, interpolation='midpoint') # 'higher'
    # prediction_sets is a boolean array indicating 
    # whether each class's softmax probability in the validation set is above the threshold (1 - qhat).
    prediction_sets = val_smx >= (1 - qhat)
    
    # coverage 
    cov = prediction_sets[ torch.arange( prediction_sets.shape[0] ), val_labels].float().mean()
    
    # efficiency %
    eff = torch.sum(prediction_sets) / len(prediction_sets)
    
    return prediction_sets.numpy(), cov.item(), eff

# E.g. return [ [permutation(10)], [permutation(10)],...]
def get_multiple_permutations(permutation_size: int = 500, num_permutations: int = 5, permutation_data_dir: str = None):
    # # Ensure the directory exists
    if permutation_data_dir is None:
        raise ValueError("permutation_data_dir must be specified")
    if not os.path.exists(permutation_data_dir):
        os.makedirs(permutation_data_dir)
    # Path to the .npz file
    path_name = os.path.join(permutation_data_dir, f"{permutation_size}_{num_permutations}.npz")
    if not os.path.exists(path_name):
        # # Generate permutations
        permutations = [np.random.permutation(permutation_size) for _ in range(num_permutations)]
        # Save permutations to .npz file
        np.savez(path_name, *permutations)
        return permutations 
    else:
        # print(f"Load permutations: size={permutation_size}, count={num_permutations}")
        # # Load permutations from .npz file
        data = np.load(path_name)
        # When np.savez is used with unnamed arguments, it saves arrays with default keys like arr_0, arr_1, etc.
        return [ data[f'arr_{i}'] for i in range(num_permutations) ]


def conformal_evaluate(model, likelihood = None, alpha = 0.05 ):
    _, _, _, val_dataset, test_dataset = get_Brain_tumor()
    if isinstance(val_dataset, torch.utils.data.Dataset):
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size = 32, shuffle = False, num_workers = 4, pin_memory = True
        )
    else:
        val_dataloader = val_dataset
    if isinstance(test_dataset, torch.utils.data.Dataset):
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size = 32, shuffle = False, num_workers = 4, pin_memory = True
        )
    else:
        test_dataloader = test_dataset

    model.eval()
    if likelihood is not None:
        likelihood.eval()
    
    n = len(val_dataset)
    
    val_prediction_list = []
    test_prediction_list = []
    val_label_list = []
    test_label_list = []
    
    with torch.no_grad():
        # validation data
        for data, target in val_dataloader:
            data = data.cuda()
            target = target.cpu()
            
            if likelihood is None:
                # 1: get conformal scores. n = calib_Y.shape[0]
                logits = model(data)
                val_result = F.softmax(logits, dim=1)
                val_prediction_list.append(val_result)
            else:
                with gpytorch.settings.num_likelihood_samples(32):
                    # converts the model's predictions to a data-independent distribution,
                    y_pred = model(data).to_data_independent_dist()
                    # likelihood( model(data) ) -> obtain the predictive distribution.
                    output = likelihood(y_pred).probs.mean(0)    # (batch_size, 4)
                    val_prediction_list.append(output.cpu())
            val_label_list.append(target)
            
        # test data 
        for data, target in test_dataloader:
            data = data.cuda()
            target = target.cpu()
            if likelihood is None:
                # 1: get conformal scores. n = calib_Y.shape[0]
                logits = model(data)
                test_result = F.softmax(logits, dim=1)
                test_prediction_list.append(test_result)
            else:
                with gpytorch.settings.num_likelihood_samples(32):
                    # converts the model's predictions to a data-independent distribution,
                    y_pred = model(data).to_data_independent_dist()
                    # likelihood( model(data) ) -> obtain the predictive distribution.
                    output = likelihood(y_pred).probs.mean(0)    # (batch_size, 4)
                    test_prediction_list.append(output.cpu()) 
            test_label_list.append(target)
        
        val_prediction_tensor = torch.cat(val_prediction_list, axis=0)
        val_label_tensor = torch.cat(val_label_list, axis=0)
        
        test_prediction_list = torch.cat(val_prediction_list, axis=0)
        test_label_tensor = torch.cat(val_label_list, axis=0)
        
        combined_prediction_tensor = torch.cat([val_prediction_tensor, test_prediction_list], axis=0).cpu()
        combined_prediction_label = torch.cat([val_label_tensor, test_label_tensor], axis=0).cpu()
        
        combined_accuracy = ( (torch.argmax(combined_prediction_tensor, axis=1) == combined_prediction_label).float().mean() )
        print(f"combined_accuracy : {combined_accuracy}")
        
        repeated_times = 100
    
        custom_permutation = get_multiple_permutations( permutation_size = len(combined_prediction_tensor), 
                                                    num_permutations = repeated_times, 
                                                    permutation_data_dir = "permutation_data" )
        coverage_list, ineff_list = [], []
        
        cal_size = len(val_label_tensor)
        
        alpha = 0.05
        
        for i in range(repeated_times):
            permutation_index = custom_permutation[i]
            cal_smx, val_smx = combined_prediction_tensor[permutation_index][:cal_size], combined_prediction_tensor[permutation_index][cal_size:]
            cal_labels, val_labels = combined_prediction_label[permutation_index][:cal_size], combined_prediction_label[permutation_index][cal_size:]
            
            _, coverage, ineff = tps(cal_smx, val_smx, cal_labels, val_labels, cal_size, 0.05)
            coverage_list.append(coverage)
            ineff_list.append(ineff)
        
        coverage_mean = np.mean(coverage_list)
        ineff_list = np.mean(ineff_list)
    
        print(f"coverage mean, {np.mean(coverage_list)} ", f"ineff_list, {np.mean(ineff_list)}")

    return coverage_mean, ineff_list


    
