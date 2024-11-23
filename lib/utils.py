from datetime import datetime
import json
import pathlib
from pathlib import Path
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
from pathlib import Path
from collections import defaultdict
import pandas as pd
import time
import seaborn as sns
sns.set(style="white", font_scale=2)
from matplotlib.ticker import MaxNLocator

def repeat_experiment(args, seeds, main_fn):
    # run = wandb.init()
    result_dict = defaultdict(list)
    if args.sngp:
        tag_name = f"sngp{int(args.sngp)}_epoch{args.epochs}_{args.dataset}_{args.learning_rate}_{args.batch_size}.csv"
    else:
        tag_name = f"sngp{int(args.sngp)}_epoch{args.epochs}_{args.dataset}_{args.learning_rate}_{args.batch_size}_{args.kernel}_{args.n_inducing_points}.csv"
    # tag_name = f"sngp{int(args.sngp)}_epoch{args.epochs}_dataset{args.dataset}_learning_rate{args.learning_rate}.csv"
    parent_name = "results_conformal" if args.conformal_training else "results_normal"
    start_time = time.time()
    for seed in seeds:
        set_seed(seed)
        one_result = main_fn(args)
        for k, v in one_result.items():
            result_dict[k].append(v)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Run time: {run_time:.2f}s")

    results_file_path = Path(f"{parent_name}/{tag_name}")
    results_file_path.parent.mkdir(exist_ok=True)

    summary_metrics = pd.DataFrame(result_dict)
    statistic_metrics = summary_metrics.agg(["mean", "std"]).transpose()
    statistic_metrics["mean-std"] = (statistic_metrics["mean"].round(4).astype(str) + "±" +
                                     statistic_metrics["std"].round(4).astype(str))
    statistic_metrics = statistic_metrics.drop(columns=["mean", "std"]).transpose()
    summary_metrics = pd.concat([summary_metrics, statistic_metrics])
    if results_file_path.exists():
        existing_data = pd.read_csv(results_file_path)
        summary_metrics = pd.concat([existing_data, summary_metrics], ignore_index=True)
    summary_metrics.to_csv(results_file_path, index=False)
    # wandb.finish()

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def plot_loss_curves(results):
    loss = results["train_loss"]
    test_loss = results["val_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["val_acc"]
    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(15, 7))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="val_accuracy")
    ax = plt.gca()  # Get current axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.tight_layout()
    name = f"learning_curve.png"
    plt.savefig(name, bbox_inches='tight')
    # plt.show(block=True)

def plot_OOD(plot_auroc, plot_aupr):
    epochs = range(1, len(plot_auroc) + 1)
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, plot_auroc, label='AUROC')
    plt.plot(epochs, plot_aupr, label='AUPR')
    plt.title('OOD')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('OOD_process.pdf')

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred))
    return acc

def get_results_directory(name, stamp=True):
    timestamp = datetime.now().strftime("%Y-%m-%d-%A-%H-%M-%S")
    results_dir = pathlib.Path("runs")
    if name is not None:
        results_dir = results_dir / name
    results_dir = results_dir / timestamp if stamp else results_dir
    results_dir.mkdir(parents=True)
    return results_dir
