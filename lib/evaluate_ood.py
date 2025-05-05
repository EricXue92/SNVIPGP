import os
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from .datasets import get_feature_dataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import pickle
from lib.utils import plot_conformal_robustness_vs_percentile

NUM_WORKERS = os.cpu_count()

def prepare_ood_datasets(true_dataset, ood_dataset):
    if hasattr(ood_dataset, 'transform') and hasattr(true_dataset, 'transform'):
        ood_dataset.transform = true_dataset.transform
    datasets = [true_dataset, ood_dataset]
    print(f"True dataset Size: {len(true_dataset)} | OOD dataset Size: {len(ood_dataset)}")
    anomaly_targets = torch.cat(
        (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
    )
    concat_datasets = ConcatDataset(datasets)
    dataloader = DataLoader(
        concat_datasets, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    return dataloader, anomaly_targets

def loop_over_dataloader(model, likelihood, dataloader, return_preds=False):
    model.eval()
    if likelihood is not None:
        likelihood.eval()
    else:
        model.classifier.update_covariance_matrix()

    scores, accuracies = [], []
    all_probs, all_preds, all_uncertainties = [], [], []

    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            if likelihood is None:
                # output: (batch_size, num_of_classes) (64, 4)
                output, uncertainty = model(data, kwargs={"update_precision_matrix": False,
                                                          "return_covariance": True} )
                uncertainty = torch.diag(uncertainty)
                # Dempster-Shafer uncertainty for SNGP
                # From: https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/ood_utils.py#L22
                # num_classes = output.shape[1]
                # belief_mass = output.exp().sum(1)
                # uncertainty = num_classes / (belief_mass + num_classes)
                # uncertainty = 1 - torch.max(output, dim=-1)[0]
            else:
                with gpytorch.settings.num_likelihood_samples(128): # 256
                    y_pred = model(data).to_data_independent_dist()
                    predictive_dist = likelihood(y_pred)
                    probs = predictive_dist.probs
                    output = probs.mean(0)
                uncertainty = -(output * output.log()).sum(1)
                #     uncertainty = probs.var(0)
                # #uncertainty, max_indices = torch.max(uncertainty, dim=1)
                # uncertainty = torch.mean(uncertainty, dim=1)
                # uncertainty = -(output * output.log()).sum(1)
                # Dempster-Shafer uncertainty for IPGP
                # num_classes = output.shape[-1]
                # num_classes = torch.tensor(num_classes, dtype=output.dtype, device=output.device)
                # belief_mass = torch.sum(torch.exp(output), dim=-1)
                # uncertainty = num_classes / (belief_mass + num_classes)
                # Cross Entropy Uncertainty
                # uncertainty = -(output * output.log()).sum(1)
            pred = torch.argmax(output, dim=1)
            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())
            scores.append(uncertainty.cpu().numpy())
            if return_preds:
                all_probs.append(output.detach().cpu())
                all_preds.append(pred.detach().cpu())
                all_uncertainties.append(uncertainty.detach().cpu())
    if len(scores) == 0:
        raise ValueError("Scores are empty. Check the model and data.")
    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)

    if return_preds:
        return (
            torch.cat(all_probs, dim=0).numpy(),
            torch.cat(all_preds, dim=0).numpy(),
            torch.cat(all_uncertainties, dim=0).numpy(),
            scores, accuracies
        )
    else:
        return scores, accuracies

def get_ood_metrics(in_dataset: object, out_dataset: object, model: object, likelihood: object = None, save_path=None) -> object:
    _, _, _, val_in_dataset, in_dataset = get_feature_dataset(in_dataset)()
    _, _, _, val_out_dataset, out_dataset = get_feature_dataset(out_dataset)()
    in_dataset = ConcatDataset([val_in_dataset, in_dataset])
    out_dataset = ConcatDataset([val_out_dataset, out_dataset])
    dataloader, anomaly_targets = prepare_ood_datasets(in_dataset, out_dataset)

    probs, preds, uncertainties, scores, accuracies = loop_over_dataloader(
        model, likelihood, dataloader, return_preds=True
    )
    # scores, accuracies = loop_over_dataloader(model, likelihood, dataloader)
    n_in = len(in_dataset)
    accuracy = np.mean(accuracies[:n_in])
    assert len(anomaly_targets) == len(scores), "Mismatch in lengths of anomaly_targets and scores"
    auroc = roc_auc_score(anomaly_targets, scores)
    precision, recall, _ = precision_recall_curve(anomaly_targets, scores)
    aupr = auc(recall, precision)

    if save_path is not None:
        outputs = {
            'iid': {
                'predictive_probs': probs[:n_in],
                'predictive_labels': preds[:n_in],
                'predictive_uncertainty': uncertainties[:n_in],
            },
            'ood': {
                'predictive_probs': probs[n_in:],
                'predictive_uncertainty': uncertainties[n_in:],
            }
        }
        with open(save_path, 'wb') as f:
            pickle.dump(outputs, f)
        print(f"Saved predictive outputs to {save_path}")
    # Conformal robustness evaluation
    alpha, uncertainty_percentile = 0.01, 90
    scores = np.array(scores)
    is_in = (anomaly_targets == 0).numpy()
    is_ood = (anomaly_targets == 1).numpy()
    pi = np.mean(is_ood)

    threshold = np.percentile(scores, uncertainty_percentile)
    is_accepted = scores < threshold

    delta = 1 - np.mean(is_accepted[is_in])
    gamma = np.mean(is_accepted[is_ood])
    accepted_ood_indices = np.where(is_accepted & is_ood)[0]
    beta = np.mean(accuracies[accepted_ood_indices]) if len(accepted_ood_indices) > 0 else 0.0

    Delta = (1 - alpha) * ((1 - pi) * delta + pi) - pi * gamma * beta
    results = {
        "delta (InD FRR)": delta,
        "gamma (OOD FAR)": gamma,
        "beta (OOD coverage)": beta,
        "pi (OOD proportion)": pi,
        "Delta (coverage gap)": Delta
    }
    print("\n--- Conformal Robustness Evaluation ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print("---------------------------------------\n")
    # Plotting inside the function
    plot_conformal_robustness_vs_percentile(scores, accuracies, anomaly_targets)
    return accuracy, auroc, aupr, results


def get_auroc_classification(data, model, likelihood=None):
    if isinstance(data, Dataset):
        dataloader = DataLoader(
            data, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        )
    else:
        dataloader = data
    scores, accuracies = loop_over_dataloader(model, likelihood, dataloader)
    accuracy = np.mean(accuracies)
    roc_auc = roc_auc_score(1 - accuracies, scores)
    return accuracy, roc_auc


