import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from .datasets import get_dataset
from lib.datasets import get_Brain_tumors, get_Alzheimer, get_CIFAR10, get_CIFAR100, get_SVHN
from ignite.metrics import Metric


class ConformalTrainingLoss(nn.Module):
    def __init__(self, alpha, beta, temperature, sngp_flag):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.sngp_flag = sngp_flag

    def forward(self, probabilities, y):

        # probabilities = torch.log(probabilities)
        conformity_score = probabilities[torch.arange(len(probabilities)), y]
        tau = torch.quantile(conformity_score, self.alpha)
        in_set_prob = F.sigmoid((probabilities - tau) / self.temperature)

        # print("in_set_prob", in_set_prob, in_set_prob.shape)

        # Clamping: Ensures that the result is not less than 0, preventing negative values after subtraction
        # Computes a penalty based on the size of prediction sets

        ### [ont] important notes 2: as we have relative small num of classes, therefore, consider remove the torch.log
        # size_loss = torch.log(torch.clamp(in_set_prob.sum(axis=1) - 1, min=0).mean(axis=0))
        size_loss = torch.clamp(in_set_prob.sum(axis=1) - 1, min=0).mean(axis=0)
        size_loss = self.beta * size_loss

        if self.sngp_flag:
            fn_loss = F.cross_entropy(probabilities, y)
            total_loss = fn_loss + size_loss
            print("total loss", (size_loss).item() + fn_loss.item(), "size loss", (size_loss).item(), "ce loss",
                  fn_loss.item())
            return total_loss
        else:
            print(f"size loss is {size_loss}")
            return size_loss


class ConformalInefficiency(Metric):
    def __init__(self, alpha=0.05, output_transform=lambda x: x):
        self.cal_smx = None
        self.cal_labels = None
        self.alpha = alpha
        super(ConformalInefficiency, self).__init__(output_transform=output_transform)

    def reset(self):
        self.eff = 0

    def update(self, output):
        val_smx, val_labels = output
        n = len(val_smx)
        cal_scores = 1 - self.cal_smx[torch.arange(n), self.cal_labels]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        qhat = torch.quantile(cal_scores, q_level, interpolation='midpoint')
        prediction_sets = val_smx >= (1 - qhat)
        self.eff = torch.sum(prediction_sets) / len(prediction_sets)

    def compute(self):
        return self.eff


def tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    # 1: get conformal scores
    cal_scores = 1 - cal_smx[torch.arange(n), cal_labels]
    # 2: get adjust quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = torch.quantile(cal_scores, q_level, interpolation='midpoint')  # 'higher'
    prediction_sets = val_smx >= (1 - qhat)
    # coverage
    coverage = prediction_sets[torch.arange(prediction_sets.shape[0]), val_labels].float().mean()
    # efficiency -- the size of the prediction set
    efficiency = torch.sum(prediction_sets) / len(prediction_sets)
    return prediction_sets.numpy(), coverage.item(), efficiency


def adaptive_tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    # Ensure inputs are NumPy arrays
    cal_smx = np.array(cal_smx)
    val_smx = np.array(val_smx)
    cal_pi = np.argsort(-cal_smx, axis=1)
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[range(n), cal_labels]
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="midpoint" )
    val_pi = np.argsort(-val_smx, axis=1)
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    coverage = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
    efficiency = np.sum(prediction_sets) / len(prediction_sets)
    return prediction_sets, coverage, efficiency


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
        np.savez(path_name, *permutations)
        return permutations
    else:
        data = np.load(path_name)
        # When np.savez is used with unnamed arguments, it saves arrays with default keys like arr_0, arr_1, etc.
        return [data[f'arr_{i}'] for i in range(num_permutations)]


def conformal_evaluate(model, likelihood, dataset, adaptive_flag, alpha=0.05):
    if dataset == 'CIFAR10':
        _, _, _, val_dataset, test_dataset = get_CIFAR10()
    elif dataset == 'Brain_tumors':
        _, _, _, val_dataset, test_dataset = get_Brain_tumors()
    elif dataset == 'Alzheimer':
        _, _, _, val_dataset, test_dataset = get_Alzheimer()
    elif dataset == 'SVHN':
        _, _, _, val_dataset, test_dataset = get_SVHN()
    elif dataset == 'CIFAR100':
        _, _, _, val_dataset, test_dataset = get_CIFAR100()

    if isinstance(val_dataset, torch.utils.data.Dataset):
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
        )
    else:
        val_dataloader = val_dataset

    if isinstance(test_dataset, torch.utils.data.Dataset):
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
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

        val_prediction_tensor = torch.cat(val_prediction_list, axis=0)
        val_label_tensor = torch.cat(val_label_list, axis=0)

        test_prediction_list = torch.cat(val_prediction_list, axis=0)
        test_label_tensor = torch.cat(val_label_list, axis=0)

        combined_prediction_tensor = torch.cat([val_prediction_tensor, test_prediction_list], axis=0).cpu()
        combined_prediction_label = torch.cat([val_label_tensor, test_label_tensor], axis=0).cpu()

        # We do not care much about the accuracy here
        combined_accuracy = (
            (torch.argmax(combined_prediction_tensor, axis=1) == combined_prediction_label).float().mean())
        print(f"Combined_accuracy : {combined_accuracy}")

        repeated_times = 100

        custom_permutation = get_multiple_permutations(permutation_size=len(combined_prediction_tensor),
                                                       num_permutations=repeated_times,
                                                       permutation_data_dir="permutation_data")
        coverage_list, ineff_list = [], []

        cal_size = len(val_label_tensor)

        alpha = 0.05
        for i in range(repeated_times):
            permutation_index = custom_permutation[i]
            cal_smx, val_smx = combined_prediction_tensor[permutation_index][:cal_size], combined_prediction_tensor[
                                                                                             permutation_index][
                                                                                         cal_size:]
            cal_labels, val_labels = combined_prediction_label[permutation_index][:cal_size], combined_prediction_label[
                                                                                                  permutation_index][
                                                                                              cal_size:]
            if adaptive_flag:
                _, coverage, ineff = adaptive_tps(cal_smx, val_smx, cal_labels, val_labels, cal_size, 0.05)
            else:
                _, coverage, ineff = tps(cal_smx, val_smx, cal_labels, val_labels, cal_size, 0.05)
            coverage_list.append(coverage)
            ineff_list.append(ineff)

        coverage_mean = np.mean(coverage_list)
        ineff_list = np.mean(ineff_list)

        print(f"coverage mean, {coverage_mean} ", f"ineff_list, {ineff_list}")

    return coverage_mean, ineff_list



