import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use GPU 1

import argparse
import torch
import torch.nn.functional as F
from lib.datasets import get_feature_dataset
from lib.evaluate_ood import get_ood_metrics
from lib.utils import get_results_directory, accuracy_fn, repeat_experiment, plot_loss_curves
from lib.evaluate_cp import conformal_evaluate, ConformalTrainingLoss, tps
from torch.utils.data import DataLoader
import json
import operator
from builder_model import build_model


import wandb
from functools import partial
NUM_WORKERS = os.cpu_count()


# export CUDA_VISIBLE_DEVICES=1
torch.backends.cudnn.benchmark = True

def main(args):
    results_dir = get_results_directory(args.output_dir)
    print(f"save to results_dir {results_dir}")
    #
    # args.temperature = wandb.config.temperature
    # args.beta = wandb.config.beta
    # args.size_loss_form = wandb.config.size_loss_form
    # # # #
    # args.learning_rate = wandb.config.learning_rate
    # args.n_inducing_points = wandb.config.n_inducing_points
    # # args.kernel = wandb.config.kernel
    # args.epochs = wandb.config.epochs


    ds = get_feature_dataset(args.dataset)()
    input_size, num_classes, train_dataset, val_dataset, test_dataset = ds
    print(f"train_dataset: {len(train_dataset)} | val_dataset: {len(val_dataset)} | test_dataset: {len(test_dataset)}")

    if args.n_inducing_points is None:
        args.n_inducing_points = num_classes

    args_dict = vars(args)
    # json.dumps(): converts a Python object into a JSON string.
    args_json = json.dumps(args_dict, indent=4)
    print(f"Training with:\n{args_json}")

    model, likelihood, loss_fn = build_model(args=args, num_classes=num_classes, train_dataset=train_dataset)
    parameters = [ {'params': model.parameters(), 'lr': args.learning_rate} ]

    if args.snipgp:
        parameters.append({"params": likelihood.parameters(), 'lr': args.learning_rate})

    optimizer = torch.optim.AdamW(parameters,
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay) #For CIFAR10
    # optimizer = torch.optim.AdamW(parameters) # For Brain_tumors
    training_steps = len(train_dataset) // args.batch_size * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=training_steps)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                        T_max=args.epochs,
    #                                                        eta_min=1e-4)
    best_inefficiency, best_auroc, best_aupr = float('inf'), float('-inf'), float('-inf')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    def simple_transform(args, outputs):
        if args.snipgp:
            outputs = outputs.to_data_independent_dist()
            outputs = likelihood(outputs).probs.mean(0)
        return outputs

    def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
        train_loss, train_acc = 0, 0
        model = model.to(device)
        model.train()
        if args.snipgp and likelihood is not None:
            likelihood.train()
        for idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            if args.conformal_training and args.snipgp:
                CP_size_fn = ConformalTrainingLoss(alpha=args.alpha, beta=args.beta,
                                                   temperature=args.temperature, args=args)
                loss_cn = loss_fn(y_pred, y)
                y_temp = y_pred.to_data_independent_dist()
                y_temp = likelihood(y_temp).probs.mean(0)
                loss_size = CP_size_fn(y_temp, y)
                loss = (loss_cn + loss_size)
                print(f"Total loss: {loss.item():.4f} | ELBO: {loss_cn.item():.4f} | Size loss: {loss_size:.4f}")
            else:
                loss = loss_fn(y_pred, y)

            train_loss += loss.item()
            y_pred = simple_transform(args, y_pred)
            _, y_pred = y_pred.max(1)
            train_acc += accuracy_fn(y_true=y, y_pred=y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(data_loader)
        train_acc /= len(data_loader)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        return train_loss, train_acc

    def test_step(mode, model, data_loader, accuracy_fn, device):
        test_loss, test_acc = 0, 0
        model.to(device)
        model.eval()
        if args.snipgp:
            likelihood.eval()
        prob_list, target_list = [], []
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                if args.sngp or args.snn:
                    loss = F.cross_entropy(y_pred, y)
                else:
                    loss = -likelihood.expected_log_prob(y, y_pred).mean()

                test_loss += loss.item()
                y_pred = simple_transform(args, y_pred)

                prob_list.append(y_pred)
                target_list.append(y)
                _, y_pred = y_pred.max(1)
                test_acc += accuracy_fn(y_true=y, y_pred=y_pred)
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"{mode} Loss: {test_loss:.4f} | {mode} accuracy: {test_acc:.2f}%\n")
        prob = torch.cat(prob_list, dim=0)
        target = torch.cat(target_list, dim=0)
        return test_loss, test_acc, prob, target
    learning_curve = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [] }

    for epoch in range(args.epochs):
        if args.sngp:
            model.classifier.reset_covariance_matrix() # if args.sngp else None
        print(f"\nEpoch: {epoch + 1}/{args.epochs}\n {'-' * 40}")
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, accuracy_fn, device)
        scheduler.step()
        learning_curve["train_loss"].append(train_loss)
        learning_curve["train_acc"].append(train_acc)
        val_loss, val_acc, val_smx, val_labels = test_step("Validation", model, val_loader, accuracy_fn, device)
        learning_curve["val_loss"].append(val_loss)
        learning_curve["val_acc"].append(val_acc)

        if not args.snn:
            _, auroc, aupr, results = get_ood_metrics(args.dataset, args.OOD, model, likelihood)

            print(f"Train -- OoD Metrics - AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")
        _, coverage, inefficiency = tps(cal_smx=val_smx, val_smx=val_smx, cal_labels=val_labels,
                                        val_labels=val_labels, n=len(val_labels), alpha=args.alpha)
        print(f"Train -- Coverage: {coverage:.4f} | Inefficiency: {inefficiency:.4f}")

        def save_best_metric(metric_name, metric_value, best_metric):
            compare_fn = operator.gt if metric_name == "auroc" else operator.lt
            if compare_fn(metric_value, best_metric):
                best_metric = metric_value
                model_state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                     metric_name: best_metric,
                    'likelihood': likelihood.state_dict() if args.snipgp else None,
                }
                torch.save(model_state, results_dir / f"best_model_{metric_name}.pth")
                print(f"\nNew best {metric_name}: {best_metric:.4f}, save_path {results_dir / f"best_model_{metric_name}.pth"}")
            return best_metric

        if not args.snn:
            best_auroc = save_best_metric("auroc", auroc, best_auroc)
        best_inefficiency = save_best_metric("inefficiency", inefficiency, best_inefficiency)

    def load_best_state(metric_name, model, likelihood):
        state = torch.load(results_dir / f"best_model_{metric_name}.pth")
        model.load_state_dict(state['model'], strict=False)
        likelihood.load_state_dict(state['likelihood']) if args.snipgp else None


    # Load two best states 1) For calculate AUROC 2) For calculate Inefficiency
    if not args.snn:
        load_best_state("auroc", model, likelihood)
        _, auroc, aupr, results = get_ood_metrics(args.dataset, args.OOD, model, likelihood, save_path="./ood_leakage.pkl")
        print(f"Test --- OoD Metrics - AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")

    load_best_state("inefficiency", model, likelihood)
    model.eval()
    likelihood.eval() if args.snipgp else None

    result = {}
    test_loss, test_acc, test_smx, test_labels = test_step("Test", model, test_loader, accuracy_fn, device)

    _, coverage, inefficiency = tps(cal_smx=test_smx, val_smx=test_smx, cal_labels=test_labels, val_labels=test_labels, n=len(test_labels), alpha=args.alpha)
    print(f"Test -- Coverage: {coverage:.4f} | Inefficiency: {inefficiency:.4f}")

    if not args.snn:
        result["auroc"], result["aupr"], result['acc'], result['loss'], result['ineff'] = auroc, aupr, test_acc, test_loss, inefficiency
    else:
        result['acc'], result['loss'], result['ineff'] = test_acc, test_loss, inefficiency

    coverage_mean, ineff_list = conformal_evaluate(model, likelihood, dataset=args.dataset, adaptive_flag=args.adaptive_conformal, alpha=args.alpha)
    result["coverage_mean"], result["ineff_list"] = coverage_mean, ineff_list



    # SNN
    # wandb.log({"epochs": args.epochs, "test_loss": test_loss, "test_Acc": test_acc,
    #            "test_ineff":inefficiency,  "beta":args.beta, "avg_coverage":coverage_mean, "ineff_list":ineff_list,
    #            "temperature":args.temperature, "size_loss_form":args.size_loss_form})
    #
    # wandb.log({"epochs": args.epochs, "test_loss": test_loss, "test_Acc": test_acc, "test_auroc": auroc, "test_aupr": aupr,
    #            "test_ineff":inefficiency,  "beta":args.beta, "avg_coverage":coverage_mean, "ineff_list":ineff_list,
    #            "temperature":args.temperature, "size_loss_form":args.size_loss_form}) #
    # # # # #
    #
    # wandb.log({"epochs": args.epochs, "test_loss": test_loss, "test_Acc": test_acc,
    #            "test_auroc":auroc, "test_aupr": aupr, "n_inducing_points":args.n_inducing_points,
    #            "test_ineff":inefficiency, "avg_coverage":coverage_mean,
    #            "learning_rate":args.learning_rate, "ineff_list":ineff_list})

    # wandb.log({"epochs": args.epochs, "test_loss": test_loss, "test_Acc": test_acc,
    #            "test_ineff":inefficiency, "avg_coverage":coverage_mean,
    #            "learning_rate":args.learning_rate, "ineff_list":ineff_list})

    plot_loss_curves(learning_curve)
    return result

def parse_arguments():
    parser = argparse.ArgumentParser()
    # [0.005, 0.01]
    # 0.005
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate") # breast 5e-3  #
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use for training") # 128
    parser.add_argument("--alpha", type=float, default=0.01, help="Conformal Rate") #####  0.05 or 0.01
    parser.add_argument("--dataset", default="CIFAR10", choices=["CIFAR100", "Alzheimer",'CIFAR10', "SVHN", "CIFAR100", "Colorectal"])
    parser.add_argument("--OOD", default="SVHN", choices=["Brain_tumors", "Alzheimer", 'CIFAR10', 'CIFAR100', "SVHN", "Colorectal", "Breast"])
    parser.add_argument("--n_inducing_points", type=int, default=15, help="Number of inducing points") # 40
    parser.add_argument("--beta", type=float, default=0.1, help="Weight for conformal training loss")
    parser.add_argument("--temperature", type=float, default=0.01, help="Temperature for conformal training loss")
    parser.add_argument("--snn", action="store_true", help="Use standard NN or not")
    parser.add_argument("--sngp", action="store_true", help="Use SNGP or not")
    parser.add_argument("--snipgp", action="store_false", help="Use SNIPGP or not")
    parser.add_argument("--conformal_training", action="store_false", help="conformal training or not")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay") # 1e-4,   (5e-4 for CIFAR10) (5e-4 for sngp breast)
    parser.add_argument("--kernel", default="RBF", choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"], help="Pick a kernel",)
    parser.add_argument("--no_spectral_conv", action="store_true",  dest="spectral_conv", help="Don't use spectral normalization on the convolutions",)
    parser.add_argument( "--adaptive_conformal", action="store_true", help="adaptive conformal")
    parser.add_argument("--no_spectral_bn", action="store_true", dest="spectral_bn", help="Don't use spectral normalization on the batch normalization layers",)
    parser.add_argument("--coeff", type=float, default=3, help="Spectral normalization coefficient") # 3
    parser.add_argument("--n_power_iterations", default=1, type=int, help="Number of power iterations")
    parser.add_argument("--output_dir", default="./default", type=str, help="Specify output directory")
    parser.add_argument("--size_loss_form", default="identity", type=str, help="identity or log")
    parser.add_argument("--spec_norm_replace_list", nargs='+', default=["Linear", "Conv2D"], type=str, help="List of specifications to replace" )
    parser.add_argument("--spectral_normalization", action="store_false", help="Use spectral normalization or not")
    args = parser.parse_args()
    if sum([args.sngp, args.snipgp, args.snn]) != 1:
        parser.error("Exactly one of --snn, --sngp or --snipgp must be set.")
    return args

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #seeds = [1, 23, 42, 202, 2024]
    repeat_experiment(args, seeds, main)
    # # seeds = [23]
    # wandb.login()
    # # # ### Step 1: Define a sweep
    # sweep_config = {
    #     'method': 'grid',
    #     'metric': {'name': 'loss', 'goal': 'minimize'},
    #     'parameters': {
    #         'temperature': {"values":  [0.01, 0.1, 0.5, 1] },
    #         "beta": {"values": [0.005, 0.1, 0.05, 0.5] }, # 0.005, 0.1, 0.05,
    #         "size_loss_form": {"values": ["log", "identity"]}, #
    #     }
    # }
    #
    # sweep_config = {
    #     'method': 'grid',
    #     'metric': {'name': 'loss', 'goal': 'minimize'},
    #     'parameters': {
    #         'n_inducing_points': {"values": [8, 10, 16, 24, 32, 40, 50] },
    #         "epochs": {"values": [10, 20, 30, 40, 50] },
    #         "learning_rate" :{"values": [0.0001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1] }
    #     }
    # }
    # # # # #
    # # # # # # # #
    # if args.conformal_training:
    #     if args.snipgp:
    #         project_name = f"snipgp_CT_{args.dataset}_{args.OOD}_{args.coeff}_{args.n_inducing_points}_{int(args.conformal_training)}"
    #     elif args.sngp:
    #         project_name = f"sngp_ct_{args.dataset}_{args.OOD}_{int(args.conformal_training)}"
    #     else:
    #         project_name = f"snn_ct_{args.dataset}_{args.OOD}_{int(args.conformal_training)}"
    # else:
    #     if args.snipgp:
    #         project_name = f"snipgp_{args.dataset}_{args.OOD}"
    #     elif args.sngp:
    #         project_name = f"sngp_{args.dataset}_{args.OOD}"
    #     else:
    #         project_name = f"snn_{args.dataset}_{args.OOD}"
    #
    # sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    # wandb.agent(sweep_id, function=partial(repeat_experiment, args, seeds, main), count=32)
