import os
import numpy as np
import torch
import gpytorch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from .datasets import get_feature_dataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset
NUM_WORKERS = os.cpu_count()

def prepare_ood_datasets(true_dataset, ood_dataset):
    # ood_dataset.transform = true_dataset.transform
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

def loop_over_dataloader(model, likelihood, dataloader):
    model.eval()
    if likelihood is not None:
        likelihood.eval()
    else:
        model.classifier.update_covariance_matrix()

    with torch.no_grad():
        scores = []
        accuracies = []
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
                with gpytorch.settings.num_likelihood_samples(32):
                    y_pred = model(data).to_data_independent_dist()
                    predictive_dist = likelihood(y_pred)
                    probs = predictive_dist.probs
                    output = probs.mean(0)

                # uncertainty = -(output * output.log()).sum(1)
                    uncertainty = probs.var(0)
                # uncertainty, max_indices = torch.max(uncertainty, dim=1)
                uncertainty = torch.mean(uncertainty, dim=1)

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

    # del data, target, output, uncertainty, pred, accuracy
    # torch.cuda.empty_cache()
    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)
    return scores, accuracies

def get_ood_metrics(in_dataset: object, out_dataset: object, model: object, likelihood: object = None) -> object:
    _, _, _, val_in_dataset, in_dataset = get_feature_dataset(in_dataset)()
    _, _, _, val_out_dataset, out_dataset = get_feature_dataset(out_dataset)()
    in_dataset = ConcatDataset([val_in_dataset, in_dataset])
    out_dataset = ConcatDataset([val_out_dataset, out_dataset])

    dataloader, anomaly_targets = prepare_ood_datasets(in_dataset, out_dataset)

    scores, accuracies = loop_over_dataloader(model, likelihood, dataloader)

    accuracy = np.mean(accuracies[:len(in_dataset)])
    assert len(anomaly_targets) == len(scores), "Mismatch in lengths of anomaly_targets and scores"
    auroc = roc_auc_score(anomaly_targets, scores)
    precision, recall, _ = precision_recall_curve(anomaly_targets, scores)
    aupr = auc(recall, precision)
    return accuracy, auroc, aupr

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