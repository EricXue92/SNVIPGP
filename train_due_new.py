import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np 
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Events, Engine
from ignite.metrics import Metric, Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, global_step_from_engine

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood

from due import dkl
from due.wide_resnet import WideResNet
from due.sngp import Laplace
from lib.datasets import get_dataset
from lib.evaluate_ood import get_ood_metrics
from lib.utils import get_results_directory, Hyperparameters, set_seed
from lib.evaluate_cp import conformal_evaluate, ConformalTrainingLoss
from pathlib import Path

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
        # print(val_smx.shape, val_labels.shape)
        cal_scores = 1 - self.cal_smx[torch.arange(n), self.cal_labels]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        qhat = torch.quantile(cal_scores, q_level, interpolation='midpoint')
        prediction_sets = val_smx >= (1 - qhat)
        self.eff = torch.sum(prediction_sets) / len(prediction_sets)
    def compute(self):
        return self.eff 

def main(hparams):
    
    if hparams.force_directory is None:
        results_dir = get_results_directory(hparams.output_dir)
    else:
        os.makedirs(hparams.force_directory, exist_ok=True)
        results_dir = Path(hparams.force_directory)
        
    writer = SummaryWriter(log_dir=str(results_dir))

    ds = get_dataset(hparams.dataset)
    input_size, num_classes, train_dataset, val_dataset, test_dataset = ds
    hparams.seed = set_seed(hparams.seed)

    if hparams.n_inducing_points is None:
        hparams.n_inducing_points = num_classes
    print(f"Training with {hparams}")
    hparams.save(results_dir / "hparams.json")
    
    feature_extractor = WideResNet(
        input_size,
        hparams.spectral_conv,
        hparams.spectral_bn,
        dropout_rate=hparams.dropout_rate,
        coeff=hparams.coeff,
        n_power_iterations=hparams.n_power_iterations,
    )
    kwargs = {"num_workers": 4, "pin_memory": True}

    if hparams.sngp:
        num_deep_features = 640
        num_gp_features = 128
        normalize_gp_features = True
        num_random_features = 1024
        num_data = len(train_dataset)
        mean_field_factor = 25
        ridge_penalty = 1
        feature_scale = 2

        model = Laplace(
            feature_extractor,
            num_deep_features,
            num_gp_features,
            normalize_gp_features,
            num_random_features,
            num_classes,
            num_data,
            hparams.batch_size,
            ridge_penalty,
            feature_scale,
            mean_field_factor,
        )
        if hparams.conformal_training:
            loss_fn = ConformalTrainingLoss(alpha=0.05, beta=hparams.beta, temperature=1, sngp_flag=True)
        else:
            loss_fn = F.cross_entropy
        likelihood = None
            
    else:
        initial_inducing_points, initial_lengthscale = dkl.initial_values(
            train_dataset, feature_extractor, hparams.n_inducing_points
        )
        gp = dkl.GP(
            num_outputs=num_classes,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=hparams.kernel,
        )
        model = dkl.DKL(feature_extractor, gp)
        
        likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False) 
        likelihood = likelihood.cuda()
        elbo_fn = VariationalELBO(likelihood, gp, num_data=len(train_dataset))
        loss_fn = lambda x, y: -elbo_fn(x, y)

    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=hparams.learning_rate,
        momentum=0.9,
        weight_decay=hparams.weight_decay,
    )

    milestones = [60, 120, 160]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2
    )

    best_inefficiency = float('inf')
    best_model_state = None

    def step(engine, batch):
        model.train()
        if not hparams.sngp:
            likelihood.train()
        optimizer.zero_grad()
        x, y = batch
        x, y = x.cuda(), y.cuda()
        y_pred = model(x)
        if hparams.conformal_training and not hparams.sngp:
            CP_size_fn = ConformalTrainingLoss(alpha=0.05, beta=hparams.beta, temperature=1, sngp_flag=False)
            loss_cn = loss_fn(y_pred, y)
            y_pred = y_pred.to_data_independent_dist()
            y_pred = likelihood(y_pred).probs.mean(0)
            loss_size = CP_size_fn(y_pred, y)
            loss = loss_cn + loss_size
            print("elbo", loss.item(), "loss_size", loss_size)
        else: 
            loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return y_pred, y, loss.item()

    def eval_step(engine, batch):
        model.eval()
        if not hparams.sngp:
            likelihood.eval()
        x, y = batch
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            y_pred = model(x)
        return y_pred, y
    
    def training_accuracy_transform(output):
        y_pred, y, loss = output 
        if not hparams.sngp:
            y_pred = y_pred.to_data_independent_dist()
            y_pred = likelihood(y_pred).probs.mean(0)
        return y_pred, y
    
    def output_loss_transform(output):
        y_pred, y, loss = output 
        return loss

    def output_transform(output):
        y_pred, y = output
        if not hparams.sngp:
            y_pred = y_pred.to_data_independent_dist()
            y_pred = likelihood(y_pred).probs.mean(0)
        return y_pred, y

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average(output_transform = output_loss_transform)
    metric.attach(trainer, "loss")
    
    metric = Accuracy(output_transform = training_accuracy_transform)
    metric.attach(trainer, "accuracy")

    metric = Accuracy(output_transform = output_transform)
    metric.attach(evaluator, "accuracy")
    
    if hparams.sngp:
        metric = Loss(F.cross_entropy)
        metric.attach(evaluator, "loss")
    else:
        metric = Loss(lambda y_pred, y: -likelihood.expected_log_prob(y, y_pred).mean() )
        metric.attach(evaluator, "loss")

    kwargs = {"num_workers": 4, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = hparams.batch_size, 
                                               shuffle = True, drop_last = True, **kwargs)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, 
                                             shuffle = False, **kwargs)

    test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=32, 
                                              shuffle=False, **kwargs )

    if hparams.sngp:
        @trainer.on(Events.EPOCH_STARTED)
        def reset_precision_matrix(trainer):
            model.reset_precision_matrix()
            
    inefficiency_metric = ConformalInefficiency(alpha = 0.05)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        
        train_loss = metrics["loss"]
        train_acc = metrics["accuracy"]
        
        result = f"Train - Epoch: {trainer.state.epoch} "
        if hparams.sngp:
            result += f"Loss: {train_loss:.2f} "
            result += f"Accuracy: {train_acc :.2f} "
        else:
            result += f"ELBO: {train_loss:.2f} "
            result += f"Accuracy: {train_acc :.2f} "
            
        print(result)
        writer.add_scalar("Train/Loss", train_loss, trainer.state.epoch)
        writer.add_scalar("Train/Accuracy", train_acc, trainer.state.epoch)

        if hparams.spectral_conv:
            for name, layer in model.feature_extractor.named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    writer.add_scalar(f"sigma/{name}", layer.weight_sigma, trainer.state.epoch)

        if trainer.state.epoch > 150 and trainer.state.epoch % 5 == 0:
            _, auroc, aupr = get_ood_metrics(
                hparams.dataset, "Alzheimer", model, likelihood
            )
            print(f"OoD Metrics - AUROC: {auroc}, AUPR: {aupr}")
            writer.add_scalar("OoD/auroc", auroc, trainer.state.epoch)
            writer.add_scalar("OoD/auprc", aupr, trainer.state.epoch)
        
        val_state = evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        val_acc = metrics["accuracy"]
        val_loss = metrics["loss"]
        
        cal_smx, cal_labels = val_state.output
        if not hparams.sngp:
            cal_smx = cal_smx.to_data_independent_dist()
            cal_smx = likelihood(cal_smx).probs.mean(0)
            
        inefficiency_metric.cal_smx = cal_smx
        inefficiency_metric.cal_labels = cal_labels
        inefficiency_metric.update((cal_smx, cal_labels))  # Pass the entire validation set
        inefficiency = inefficiency_metric.compute().item()
        
        print(f"Val - Epoch: {trainer.state.epoch} " 
            f"Val Accuracy: {val_acc:.4f} Val Loss: {val_loss:.2f} "
            f"Val Inefficiency: {inefficiency:.4f}")
        
        writer.add_scalar("Validation/Accuracy", val_acc, trainer.state.epoch)
        writer.add_scalar("Validation/Loss", val_loss, trainer.state.epoch)
        writer.add_scalar("Validation/Inefficiency", inefficiency, trainer.state.epoch)
        
        # test_state = evaluator.run(test_loader)
        # metrics = evaluator.state.metrics
        # test_acc = metrics["accuracy"]
        # test_loss = metrics["loss"]
        
        # test_smx, test_labels = test_state.output
        # if not hparams.sngp:
        #     test_smx = test_smx.to_data_independent_dist()
        #     test_smx = likelihood(test_smx).probs.mean(0)
            
        # inefficiency_metric.test_smx = test_smx
        # inefficiency_metric.test_labels = test_labels
        # inefficiency_metric.update((test_smx, test_labels))  # Pass the entire test set
        # inefficiency = inefficiency_metric.compute().item()
        
        # print(f"Test - Epoch: {trainer.state.epoch} "
        #     f"Test Accuracy: {test_acc:.4f} "
        #     f"Test Loss: {test_loss:.4f} " )
        # writer.add_scalar("Test/Accuracy", test_acc, trainer.state.epoch)
        # writer.add_scalar("Test/Loss", test_loss, trainer.state.epoch)
        # writer.add_scalar("Test/Inefficiency", inefficiency, trainer.state.epoch)

        # Save the best model based on the smallest inefficiency
        nonlocal best_inefficiency, best_model_state
        if inefficiency < best_inefficiency:
            best_inefficiency = inefficiency
            best_model_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
                'inefficiency': best_inefficiency,
            }
            
            model_saved_path = results_dir / "best_model.pth"
            torch.save(best_model_state, model_saved_path)

            if likelihood is not None:
                likelihood_saved_path = results_dir / "likelihood.pth"
                torch.save(likelihood.state_dict(), likelihood_saved_path)
            print(f"Best model saved at epoch {trainer.state.epoch} with inefficiency {best_inefficiency:.4f}")

        
    Results_Saved = {}
        
    @trainer.on(Events.COMPLETED)
    def compute_test_loss_at_last_epoch(trainer):
        print("Training completed. Running evaluation on the test set with the best model.")
        if best_model_state is not None:
            model.load_state_dict(best_model_state['model'])
            model.eval()
            
            test_state = evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            test_accuracy = metrics["accuracy"]
            test_loss = metrics["loss"]
            
            cal_smx, cal_labels = test_state.output
            
            if not hparams.sngp:
                cal_smx = cal_smx.to_data_independent_dist()
                cal_smx = likelihood(cal_smx).probs.mean(0)
                
            inefficiency_metric.cal_smx = cal_smx
            inefficiency_metric.cal_labels = cal_labels

            inefficiency_metric.update( (cal_smx, cal_labels) ) # Pass the entire validation set
            # inefficiency_metric.update((test_state.output[0], test_state.output[1]))  # Pass the entire validation set
            inefficiency = inefficiency_metric.compute().item()

            print(f"Final Test Accuracy: {test_accuracy:.4f}, Final Test Loss: {test_loss:.4f}", 
                f"Test_state Inefficiency: {inefficiency:.4f}")
            

            
            _, auroc, aupr = get_ood_metrics(
                hparams.dataset, "Alzheimer", model, likelihood
            ) # , hparams.data_root
            
            Results_Saved["auroc_ood_Alzheimer"] = auroc
            Results_Saved["aupr_ood_Alzheimer"] = aupr
            Results_Saved['test_accuracy'] = test_accuracy
            Results_Saved['test_loss'] = test_loss
            Results_Saved['inefficiency'] = inefficiency
            
        if hparams.sngp:
            coverage_mean, ineff_list = conformal_evaluate(model, likelihood = None, alpha = 0.05 )
            Results_Saved["coverage_mean_sngp"] = str(coverage_mean)
            Results_Saved["ineff_list_sngp"] = str(ineff_list)
            results_json = json.dumps(Results_Saved, indent=4, sort_keys=True)
            (results_dir / "results_sngp.json").write_text(results_json)
        else:
            coverage_mean, ineff_list = conformal_evaluate(model, likelihood, alpha = 0.05 )
            Results_Saved["coverage_mean_gp"] = str(coverage_mean)
            Results_Saved["ineff_list_gp"] = str(ineff_list)
            
            results_json = json.dumps(Results_Saved, indent=4, sort_keys=True)
            (results_dir / "results_GP.json").write_text(results_json)
            
            writer.add_scalar("Final Test/Accuracy", test_accuracy, trainer.state.epoch)
            writer.add_scalar("Final Test/Loss", test_loss, trainer.state.epoch)
            writer.add_scalar("Final Test/Inefficiency", inefficiency, trainer.state.epoch)

    # Adding a progress bar
    ProgressBar(persist=True).attach(trainer)
    # Start training
    trainer.run(train_loader, max_epochs = hparams.epochs)
    
    # results_saved_json = json.dumps(Results_Saved, indent=4, sort_keys=True)
    # (results_dir / "results.json").write_text(results_saved_json)

            
    # Closing the writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use for training")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate",)
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default= 2)
    parser.add_argument("--dataset", default="Brain_tumors", choices=["Brain_tumors", "Alzheimer"],)
    parser.add_argument("--force_directory", default=None)
    parser.add_argument("--kernel", default="RBF", choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"],help="Pick a kernel",)
    parser.add_argument("--no_spectral_conv", action="store_false",  # "spectral_conv": true,
            dest="spectral_conv", # # Specify the attribute name used in the result namespace
            help="Don't use spectral normalization on the convolutions",
    )
    parser.add_argument("--no_spectral_bn", 
        action="store_false", # # "spectral_bn": true,
        dest="spectral_bn", help="Don't use spectral normalization on the batch normalization layers",
    )
    parser.add_argument(
        "--sngp", 
        #action="store_false", # "sngp": True,
        action="store_true", # "sngp": false,
        help="Use SNGP (RFF and Laplace) instead of a DUE (sparse GP)",)
    parser.add_argument("--n_inducing_points", type=int, help="Number of inducing points" )
    parser.add_argument("--beta", type=int, default=1, help="Number of inducing points")
    parser.add_argument("--seed", type=int, default=23, help="Seed to use for training")
    parser.add_argument("--coeff", type=float, default=3, help="Spectral normalization coefficient")
    parser.add_argument("--n_power_iterations", default=1, type=int, help="Number of power iterations")
    parser.add_argument("--output_dir", default="./default", type=str, help="Specify output directory")
    parser.add_argument(
        "--conformal_training", 
        # action="store_false", # "conformal_training": True,
        action="store_true", # "conformal_training": false,
        help="Specify output directory"
    )
    args = parser.parse_args()
    hparams = Hyperparameters( ** vars(args) )
    
    main(hparams)
