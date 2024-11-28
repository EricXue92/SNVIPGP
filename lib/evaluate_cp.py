import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from lib.datasets import get_tumors_feature
from torch.utils.data import Dataset, DataLoader
from lib.datasets import get_feature_dataset

NUM_WORKERS = os.cpu_count()

class ConformalTrainingLoss(nn.Module):
    def __init__(self, alpha, beta, temperature, args):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.args = args

    def forward(self, probabilities, y):
        conformity_score = probabilities[torch.arange(len(probabilities)), y]
        tau = torch.quantile(conformity_score, self.alpha)
        in_set_prob = F.sigmoid((probabilities - tau) / self.temperature)
        if self.args.size_loss_form == 'log':
            size_loss = torch.log( torch.clamp(in_set_prob.sum(dim=1), min=1).mean(dim=0) )
        elif self.args.size_loss_form == 'identity':
            size_loss = torch.clamp(in_set_prob.sum(dim=1) - 1, min=0).mean(dim=0)
        else:
            raise ValueError("Invalid size loss form")
        size_loss = self.beta * size_loss

        if self.args.sngp or self.args.snn:
            fn_loss = F.cross_entropy(probabilities, y)
            total_loss = fn_loss + size_loss
            print(f"Total loss : {(size_loss.item() + fn_loss.item()):.4f} | Size loss: {size_loss.item():.4f} | Ce loss : {fn_loss.item():.4f}")
            return total_loss
        else:
            print(f"size loss : {size_loss.item():.4f}")
            return size_loss
    def compute(self):
        return self.eff

def tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_scores = 1 - cal_smx[torch.arange(n), cal_labels]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = torch.quantile(cal_scores, q_level, interpolation='midpoint')  # 'higher'
    prediction_sets = val_smx >= (1 - q_hat)
    coverage = prediction_sets[torch.arange(prediction_sets.shape[0]), val_labels].float().mean()
    efficiency = torch.sum(prediction_sets) / len(prediction_sets)
    prediction_sets = prediction_sets.cpu()
    return prediction_sets.numpy(), coverage.item(), efficiency.item()

def adaptive_tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_smx = cal_smx.cpu().numpy()
    val_smx = val_smx.cpu().numpy()
    cal_pi = np.argsort(-cal_smx, axis=1)
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[range(n), cal_labels]
    q_level = np.ceil( (n + 1) * (1 - alpha) ) / n
    q_hat = np.quantile(cal_scores, q_level, method="midpoint")
    val_pi = np.argsort(-val_smx, axis=1)
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= q_hat, val_pi.argsort(axis=1), axis=1)
    coverage = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
    efficiency = np.sum(prediction_sets) / len(prediction_sets)
    return prediction_sets, coverage, efficiency

def get_multiple_permutations(permutation_size: int = 500, num_permutations: int = 5, permutation_data_dir: str = None):
    if permutation_data_dir is None:
        raise ValueError("permutation_data_dir must be specified")
    if not os.path.exists(permutation_data_dir):
        os.makedirs(permutation_data_dir)
    path_name = os.path.join(permutation_data_dir, f"{permutation_size}_{num_permutations}.npz")
    if not os.path.exists(path_name):
        permutations = [np.random.permutation(permutation_size) for _ in range(num_permutations)]
        np.savez(path_name, *permutations)
        return permutations
    else:
        data = np.load(path_name)
        return [data[f'arr_{i}'] for i in range(num_permutations)]

def conformal_evaluate(model, likelihood, dataset, adaptive_flag, alpha):
    if dataset == 'CIFAR10':
        _, _, _, val_dataset, test_dataset = get_feature_dataset("CIFAR10")()
    elif dataset == 'Brain_tumors':
        _, _, _, val_dataset, test_dataset = get_feature_dataset("Brain_tumors")()
    elif dataset == 'Alzheimer':
        _, _, _, val_dataset, test_dataset = get_feature_dataset("Alzheimer")()
    elif dataset == 'SVHN':
        _, _, _, val_dataset, test_dataset = get_feature_dataset("SVHM")()
    elif dataset == 'CIFAR100':
        _, _, _, val_dataset, test_dataset = get_feature_dataset("CIFAR100")()
    else:
        print("Invalid dataset")
        return None, None
    if isinstance(val_dataset, Dataset):
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        val_dataloader = val_dataset
    if isinstance(test_dataset, Dataset):
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        test_dataloader = test_dataset
    model.eval()
    likelihood.eval() if likelihood is not None else None
    val_prediction_list, val_label_list, test_prediction_list, test_label_list = [], [], [], []
    with torch.no_grad():
        for data, target in val_dataloader:
            data = data.cuda()
            target = target.cpu()
            if likelihood is None:
                logits = model(data)
                val_result = F.softmax(logits, dim=1)
                val_prediction_list.append(val_result)
            else:
                with gpytorch.settings.num_likelihood_samples(32):
                    # converts the model's predictions to a data-independent distribution,
                    y_pred = model(data).to_data_independent_dist()
                    # likelihood( model(data) ) -> obtain the predictive distribution.
                    output = likelihood(y_pred).probs.mean(0)  # (batch_size, 4)
                    val_prediction_list.append(output.cpu())
            val_label_list.append(target)
        for data, target in test_dataloader:
            data = data.cuda()
            target = target.cpu()
            if likelihood is None:
                logits = model(data)
                test_result = F.softmax(logits, dim=1)
                test_prediction_list.append(test_result)
            else:
                with gpytorch.settings.num_likelihood_samples(32):
                    y_pred = model(data).to_data_independent_dist()
                    output = likelihood(y_pred).probs.mean(0)
                    test_prediction_list.append(output.cpu())
            test_label_list.append(target)
        val_prediction_list, test_prediction_list = torch.cat(val_prediction_list, dim=0), torch.cat(test_prediction_list, dim=0)
        val_label_list, test_label_list = torch.cat(val_label_list, dim=0), torch.cat(test_label_list, dim=0)
        combined_prediction_tensor = torch.cat([val_prediction_list, test_prediction_list], dim=0).cpu()
        combined_prediction_label = torch.cat([val_label_list, test_label_list], dim=0).cpu()
        print(f"combined accuracy {torch.argmax(combined_prediction_tensor, dim=1).eq(combined_prediction_label).float().mean().item():.4f}")
        repeated_times = 100
        custom_permutation = get_multiple_permutations(permutation_size=len(combined_prediction_tensor),
                                                       num_permutations=repeated_times,
                                                       permutation_data_dir="permutation_data")
        coverage_list, ineff_list = [], []
        cal_size = len(val_label_list)
        for i in range(repeated_times):
            permutation_index = custom_permutation[i]
            cal_smx, val_smx = combined_prediction_tensor[permutation_index][:cal_size], combined_prediction_tensor[permutation_index][cal_size:]
            cal_labels, val_labels = combined_prediction_label[permutation_index][:cal_size], combined_prediction_label[permutation_index][cal_size:]
            if adaptive_flag:
                _, coverage, ineff = adaptive_tps(cal_smx, val_smx, cal_labels, val_labels, cal_size, alpha)
            else:
                _, coverage, ineff = tps(cal_smx, val_smx, cal_labels, val_labels, cal_size, alpha)
            coverage_list.append(coverage)
            ineff_list.append(ineff)
        coverage_mean = np.mean(coverage_list)
        ineff_list = np.mean(ineff_list)
        print(f"coverage mean, {coverage_mean:.4f} ", f"ineff_list, {ineff_list:.4f}")
    return coverage_mean, ineff_list



