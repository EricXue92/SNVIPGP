import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood
from due import dkl
from due.convnext import ConvNextTinyGP, SimpleMLP, SimpleConvNet
from lib.datasets import get_dataset, get_feature_dataset
from lib.evaluate_ood import get_ood_metrics
from lib.utils import get_results_directory, Hyperparameters, repeat_experiment
from lib.evaluate_cp import conformal_evaluate, ConformalTrainingLoss, tps
from sngp_wrapper.covert_utils import replace_layer_with_gaussian, convert_to_sn_my
from torch.utils.data import DataLoader
from lib.helper_functions import accuracy_fn
import operator

def main(args):
    results_dir = get_results_directory(args.output_dir)
    print(f"save to results_dir {results_dir}")
    writer = SummaryWriter(log_dir=str(results_dir))
    ds = get_feature_dataset(args.dataset)
    input_size, num_classes, train_dataset, val_dataset, test_dataset = ds
    if args.n_inducing_points is None:
        args.n_inducing_points = num_classes
    print(f"Training with {args}")

    spec_norm_replace_list = ["Linear", "Conv2D"]
    spec_norm_bound = 0.95

    if args.sngp:
        model = SimpleMLP(num_classes=args.number_of_class)
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
            'num_classes': 4,
        }
        model = convert_to_sn_my(model, spec_norm_replace_list, spec_norm_bound)
        replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)

        if args.conformal_training:
            loss_fn = ConformalTrainingLoss(alpha=args.alpha, beta=args.beta,
                                            temperature=args.temperature, sngp_flag=True)
        else:
            loss_fn = F.cross_entropy
        likelihood = None

    else:
        feature_extractor = SimpleMLP(num_classes=None)
        feature_extractor = convert_to_sn_my(feature_extractor, spec_norm_replace_list, spec_norm_bound)

        initial_inducing_points, initial_lengthscale = dkl.initial_values(
            train_dataset, feature_extractor, args.n_inducing_points )

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

    model = model.cuda()

    parameters = [
        {'params': model.parameters(), 'lr': args.learning_rate},
    ]
    if not args.sngp:
        parameters.append({"params": likelihood.parameters(), 'lr': args.learning_rate})

    optimizer = torch.optim.AdamW(
        parameters
    )

    training_steps = len(train_dataset) // args.batch_size * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_steps)

    best_inefficiency, best_auroc, best_aupr = float('inf'), float('-inf'), float('-inf')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                             pin_memory=True)

    def simple_transform(args, outputs):
        if not args.sngp:
            outputs = outputs.to_data_independent_dist()
            outputs = likelihood(outputs).probs.mean(0)
        return outputs

    def train(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
        train_loss, train_acc = 0, 0
        model.to(device)
        model.train()
        if not args.sngp and likelihood is not None:
            likelihood.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            if args.conformal_training and not args.sngp:
                CP_size_fn = ConformalTrainingLoss(alpha=args.alpha, beta=args.beta,
                                                   temperature=args.temperature, sngp_flag=False)
                loss_cn = loss_fn(y_pred, y)

                y_temp = y_pred.to_data_independent_dist()
                y_temp = likelihood(y_temp).probs.mean(0)

                loss_size = CP_size_fn(y_temp, y)
                loss = loss_cn + loss_size
                print(f"Total loss: {(loss.item() + loss_size):.4f} | ELBO: {loss_cn.item():.4f} | Size loss: {loss_size:.4f}")
            else:
                loss = loss_fn(y_pred, y)

            train_loss += loss.item()

            y_pred = simple_transform(args, y_pred)
            _, y_pred = y_pred.max(1)

            train_acc += accuracy_fn(y_true=y, y_pred=y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_loss /= len(data_loader)
        train_acc /= len(data_loader)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        return train_loss, train_acc

    def test(mode, model, data_loader, accuracy_fn, device):
        test_loss, test_acc =0, 0
        model.to(device)
        model.eval()
        if not args.sngp:
            likelihood.eval()

        prob_list, target_list = [], []
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                if args.sngp:
                    loss = F.cross_entropy(y_pred, y)
                else:
                    loss = -likelihood.expected_log_prob(y, y_pred).mean()

                test_loss += loss.item()
                y_pred = simple_transform(args, y_pred)

                prob_list.append(y_pred)
                target_list.append(y)

                _, y_pred = y_pred.max(1)
                test_acc += accuracy_fn(y_true=y,
                                        y_pred=y_pred)
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        print(f"{mode} Loss: {test_loss:.4f} | {mode} accuracy: {test_acc:.2f}%\n")

        prob = torch.cat(prob_list, dim=0)
        target = torch.cat(target_list, dim=0)

        return test_loss, test_acc, prob, target

    for epoch in range(args.epochs):
        model.classifier.reset_covariance_matrix() if args.sngp else None
        print(f"\nEpoch: {epoch + 1}/{args.epochs}")
        train(model, train_loader, loss_fn, optimizer, accuracy_fn, device)
        val_loss, val_acc, val_smx, val_labels = test("Validation", model, val_loader, accuracy_fn, device)
        _, auroc, aupr = get_ood_metrics(args.dataset, "Alzheimer", model, likelihood, feature_data=True)
        print(f"OoD Metrics - AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")
        _, coverage, inefficiency = tps(cal_smx=val_smx, val_smx=val_smx, cal_labels=val_labels, val_labels=val_labels,
                                  n=len(val_labels), alpha=args.alpha)
        def save_best_metric(metric_name, metric_value, best_metric):
            compare_fn = operator.gt if metric_name == "auroc" else operator.lt
            if compare_fn(metric_value, best_metric):
                best_metric = metric_value
                model_state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                     metric_name: best_metric,
                    'likelihood': likelihood.state_dict() if not args.sngp else None,
                }
                torch.save(model_state, results_dir / f"best_model_{metric_name}.pth")
                print(f"New best {metric_name}: {best_metric:.4f}, save_path {results_dir / f"best_model_{metric_name}.pth"}")

            return best_metric
        best_auroc = save_best_metric("auroc", auroc, best_auroc)
        best_inefficiency = save_best_metric("inefficiency", inefficiency, best_inefficiency)

    def load_best_state(metric_name, model, likelihood):
        state = torch.load(results_dir / f"best_model_{metric_name}.pth")
        model.load_state_dict(state['model'], strict=False)
        likelihood.load_state_dict(state['likelihood']) if not args.sngp else None

    load_best_state("auroc", model, likelihood)
    _, auroc, aupr = get_ood_metrics(args.dataset, "Alzheimer", model, likelihood, feature_data=True)
    print(f"OoD Metrics - AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")

    load_best_state("inefficiency", model, likelihood)
    model.eval()
    likelihood.eval() if not args.sngp else None

    result = {}
    test_loss, test_acc, test_smx, test_labels = test("Test", model, test_loader, accuracy_fn, device)

    _, coverage, inefficiency = tps(cal_smx=test_smx, val_smx=test_smx, cal_labels=test_labels, val_labels=test_labels,
                                    n=len(test_labels), alpha=args.alpha)

    result["auroc"], result["aupr"], result['acc'], result['loss'], result['ineff'] = auroc, aupr, test_acc, test_loss, inefficiency
    coverage_mean, ineff_list = conformal_evaluate(model, likelihood, dataset=args.dataset,
                                                   adaptive_flag=args.adaptive_conformal, alpha=args.alpha)
    result["coverage_mean"], result["ineff_list"] = coverage_mean, ineff_list
    writer.close()
    return result

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate") # sngp = 0.05
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use for training")
    parser.add_argument("--number_of_class", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05, help="Conformal Rate" )
    parser.add_argument("--dataset", default="Brain_tumors", choices=["Brain_tumors", "Alzheimer",'CIFAR10', 'CIFAR100', "SVHN"])
    parser.add_argument("--n_inducing_points", type=int, default=12, help="Number of inducing points" ) # 12
    parser.add_argument("--beta", type=int, default=0.1, help="Weight for conformal training loss")
    parser.add_argument("--temperature", type=int, default=0.1, help="Temperature for conformal training loss")
    parser.add_argument("--sngp", action="store_true", help="Use SNGP (RFF and Laplace) instead of a DUE (sparse GP)")
    parser.add_argument("--conformal_training", action="store_true", help="conformal training or not" )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay") # 5e-4
    parser.add_argument("--kernel", default="RBF", choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"], help="Pick a kernel",)
    parser.add_argument("--no_spectral_conv", action="store_false",  dest="spectral_conv", help="Don't use spectral normalization on the convolutions",)
    parser.add_argument( "--adaptive_conformal", action="store_true", help="adaptive conformal")
    parser.add_argument("--no_spectral_bn", action="store_false", dest="spectral_bn", help="Don't use spectral normalization on the batch normalization layers",)
    #parser.add_argument("--seed", type=int, default=42, help="Seed to use for training")
    parser.add_argument("--coeff", type=float, default=3., help="Spectral normalization coefficient")
    parser.add_argument("--n_power_iterations", default=1, type=int, help="Number of power iterations")
    parser.add_argument("--output_dir", default="./default", type=str, help="Specify output directory")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_arguments()
    seeds = [1, 23, 42, 202, 2024]
    repeat_experiment(args, seeds=seeds, main_fn=main)



