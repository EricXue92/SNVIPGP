import os
import numpy as np
import torch
import torch.nn.functional as F
import gpytorch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from .datasets import get_dataset
from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import gc

NUM_WORKERS = os.cpu_count()

def prepare_ood_datasets(true_dataset, ood_dataset):
    ood_dataset.transform = true_dataset.transform
    datasets = [true_dataset, ood_dataset]
    anomaly_targets = torch.cat(
        (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
    )
    concat_datasets = ConcatDataset(datasets)
    dataloader = DataLoader(
        concat_datasets, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    return dataloader, anomaly_targets

# GP Uncertainty = entropy （ -output * log(output) ）
# SNGP Uncertainty = num_classes / (belief_mass + num_classes)

def loop_over_dataloader(model, likelihood, dataloader):
    model.eval()
    if likelihood is not None:
        likelihood.eval()
    else:
        # For SNGP Uncertainty
        model.linear.update_covariance_matrix()

    with torch.no_grad():
        scores = []
        accuracies = []
        for i, (data, target) in enumerate(dataloader):
            # # Move data to GPU
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            if likelihood is None:
                output, uncertainty = model(data, kwargs={"update_precision_matrix": False,
                                                          "return_covariance": True, "nosoftmax": True} )
                # SNGP Uncertainty with variance
                # uncertainty = torch.diagonal(uncertainty, 0)
                # Dempster-Shafer uncertainty for SNGP
                # log(softmax) -> logits
                # logits = output - torch.max(output)
                # From: https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/ood_utils.py#L22
                # K/(K + sum( exp(logits) ) )
                num_classes = output.shape[-1]
                num_classes = torch.tensor(num_classes, dtype=output.dtype, device=output.device)
                belief_mass = torch.sum(torch.exp(output), dim=-1)
                uncertainty = num_classes / (belief_mass + num_classes)
            else:
                with gpytorch.settings.num_likelihood_samples(32):
                    y_pred = model(data).to_data_independent_dist()
                    predictive_dist = likelihood(y_pred)
                    probs = predictive_dist.probs
                    output = probs.mean(0)  # (batch_size, num_of_classes)
                    # predictive_variances = probs.var(0)
                    # Cross Entropy -> Higher entropy indicates higher uncertainty.
                uncertainty = -(output * output.log()).sum(1)

            pred = torch.argmax(output, dim=1)
            accuracy = pred.eq(target)

            # Move accuracy and uncertainty to CPU early to free up GPU memory
            accuracies.append(accuracy.cpu().numpy())
            scores.append(uncertainty.cpu().numpy())

    # # # # Clear GPU memory after processing each batch
    del data, target, output, uncertainty, pred, accuracy
    # GPU memory freed
    torch.cuda.empty_cache()
    # if i % 5 == 0:
    # release unused memory
    gc.collect()
    #
    # # Concatenate results on CPU
    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies

def get_ood_metrics(in_dataset: object, out_dataset: object, model: object, likelihood: object = None) -> object:  # , root="./"
    # return input_size, num_classes, train_dataset, val_dataset, test_dataset
    _, _, _, _, in_dataset = get_dataset(in_dataset)  # , root=root
    _, _, _, _, out_dataset = get_dataset(out_dataset)  # , root=root

    dataloader, anomaly_targets = prepare_ood_datasets(in_dataset, out_dataset)
    scores, accuracies = loop_over_dataloader(model, likelihood, dataloader)

    accuracy = np.mean(accuracies[: len(in_dataset)])

    assert len(anomaly_targets) == len(scores), "Mismatch in lengths of anomaly_targets and scores"

    auroc = roc_auc_score(anomaly_targets, scores)
    precision, recall, _ = precision_recall_curve(anomaly_targets, scores)
    aupr = auc(recall, precision)
    return accuracy, auroc, aupr

def get_auroc_classification(data, model, likelihood=None):
    if isinstance(data, Dataset):
        dataloader = DataLoader(
            data, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        ) #
    else:
        dataloader = data
    scores, accuracies = loop_over_dataloader(model, likelihood, dataloader)
    accuracy = np.mean(accuracies)
    roc_auc = roc_auc_score(1 - accuracies, scores)  # The higher, the better
    return accuracy, roc_auc