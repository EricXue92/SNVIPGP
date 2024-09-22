import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np 
import argparse
import copy
import json
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, global_step_from_engine, EarlyStopping

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood

from due import dkl
from due.wide_resnet import WideResNet
from due.sngp import Laplace
from lib.datasets import get_dataset
from lib.evaluate_ood import get_ood_metrics
from lib.utils import get_results_directory, Hyperparameters, set_seed, plot_training_history, plot_OOD
from lib.evaluate_cp import conformal_evaluate, ConformalTrainingLoss, ConformalInefficiency
from pathlib import Path
from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian

# For context see: https://github.com/pytorch/pytorch/issues/47908
torch.backends.cudnn.benchmark = True

# https://datascience.stackexchange.com/questions/31113/validation-showing-huge-fluctuations-what-could-be-the-cause

def main(hparams):
    if hparams.force_directory is None:
        results_dir = get_results_directory(hparams.output_dir)
    else:
        os.makedirs(hparams.force_directory, exist_ok=True)
        results_dir = Path(hparams.force_directory)
        
    writer = SummaryWriter(log_dir=str(results_dir))

    # Prepare data 
    ds = get_dataset(hparams.dataset)
    input_size, num_classes, train_dataset, val_dataset, test_dataset = ds
    hparams.seed = set_seed(hparams.seed)
    if hparams.n_inducing_points is None:
        hparams.n_inducing_points = num_classes
    print(f"Training with {hparams}")
    
    # Save parameters
    hparams.save(results_dir / "hparams.json")
    
    # Feature transformation
    feature_extractor = WideResNet(
        input_size,
        hparams.spectral_conv,
        hparams.spectral_bn,
        dropout_rate = hparams.dropout_rate,
        coeff = hparams.coeff,
        n_power_iterations = hparams.n_power_iterations,
    )
    
    if hparams.sngp:
        # model for SNGP 
        model = WideResNet(
            input_size,
            hparams.spectral_conv,
            hparams.spectral_bn,
            dropout_rate = hparams.dropout_rate,
            coeff = hparams.coeff,
            n_power_iterations = hparams.n_power_iterations,
            num_classes = hparams.number_of_class,
        )
        spec_norm_replace_list = ["Linear", "Conv2D"]
        spec_norm_bound = 9.
    
        GP_KWARGS = {
            'num_inducing': 2048,
            'gp_scale': 1.0,
            'gp_bias': 0.,
            'gp_kernel_type': 'gaussian', # linear
            'gp_input_normalization': True,
            'gp_cov_discount_factor': -1,
            'gp_cov_ridge_penalty': 1.,
            'gp_output_bias_trainable': False,
            'gp_scale_random_features': False,
            'gp_use_custom_random_features': True,
            'gp_random_feature_type': 'orf',
            'gp_output_imagenet_initializer': True,
        }
        # Enforcing Spectral-Normalization on each layer 
        model = convert_to_sn_my(model, spec_norm_replace_list, spec_norm_bound)
        
        # Equipping the model with laplace approximation 
        replace_layer_with_gaussian(container = model, signature="linear", **GP_KWARGS) 
        
        ########## Here decide whether conformal training 
        if hparams.conformal_training:
            loss_fn = ConformalTrainingLoss(alpha = hparams.alpha, beta = hparams.beta, temperature = 1, sngp_flag = True)
        else:
            loss_fn = F.cross_entropy
        likelihood = None
        
    else:
        initial_inducing_points, initial_lengthscale = dkl.initial_values(
            train_dataset, feature_extractor, hparams.n_inducing_points )
        
        gp = dkl.GP(
            num_outputs = num_classes,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=hparams.kernel,
        )
        # Model for inducing points 
        model = dkl.DKL(feature_extractor, gp)
        likelihood = SoftmaxLikelihood(num_classes = num_classes, mixing_weights=False) 
        likelihood = likelihood.cuda()
        elbo_fn = VariationalELBO(likelihood, gp, num_data=len(train_dataset))
        loss_fn = lambda x, y: -elbo_fn(x, y)


    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = hparams.learning_rate,
        momentum = 0.9,
        weight_decay = hparams.weight_decay,)
    
#     optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=hparams.learning_rate,
#     weight_decay = hparams.weight_decay,
# )
    milestones = [60, 120, 160, 180]
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones = milestones, gamma = 0.2
    )

    best_inefficiency = float('inf')
    best_auroc = float('-inf')
    best_aupr = float('-inf')
    best_model_state_inefficiency, best_model_state_ood = None, None
    
    # For plotting
    plot_train_acc, plot_val_acc = [], []
    plot_train_loss, plot_val_loss = [], []
    plot_auroc, plot_aupr = [], []
    
    def step(engine, batch):
        model.train()
        if not hparams.sngp:
            likelihood.train()
        optimizer.zero_grad()
        x, y = batch
        x, y = x.cuda(), y.cuda()
        y_pred = model(x)
        
        if hparams.conformal_training and not hparams.sngp:
            ### Conformal training for inducing point GP
            CP_size_fn = ConformalTrainingLoss(alpha = hparams.alpha, beta = hparams.beta, temperature = 1, sngp_flag = False)
    
            loss_cn = loss_fn(y_pred, y)
            
            y_pred_temp = y_pred.to_data_independent_dist()
            y_pred_temp = likelihood(y_pred_temp).probs.mean(0)
        
            loss_size = CP_size_fn(y_pred_temp, y)
            loss = loss_cn + loss_size
            print(f"total loss, {loss.item() + loss_size}", f"elbo, {loss.item()}", f"loss_size, {loss_size}")
            
        else: 
            loss = loss_fn(y_pred, y)
            
        loss.backward()
        ########## 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients
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
                                               shuffle = True,  **kwargs) # drop_last = True,
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = hparams.batch_size, 
                                             shuffle = False, **kwargs)
    test_loader = torch.utils.data.DataLoader( test_dataset, batch_size= hparams.batch_size, 
                                              shuffle=False, **kwargs )
    if hparams.sngp:
        @trainer.on(Events.EPOCH_STARTED)
        def reset_precision_matrix(trainer):
            model.linear.reset_covariance_matrix()
            
    inefficiency_metric = ConformalInefficiency(alpha = hparams.alpha)

    ##### Create EarlyStropping handler 
    # def score_function(engine):
    #     # Return the validation loss (the lower, the better)
    #     val_loss = engine.state.metrics['loss']
    #     return -val_loss   # EarlyStopping requires a higher score for improvement
    # Attach the early stopping handler to the evaluator
    # early_stopping_handler = EarlyStopping(
    #         patience=10,  # Stop after 10 epochs without improvement
    #         score_function = score_function,
    #         trainer = trainer  # The trainer to stop when patience is exceeded
    # )
    # Attach the early stopping handler to the evaluator (after validation)
    # evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    # Attach the handler that logs results after each epoch
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        train_loss = metrics["loss"]
        train_acc = metrics["accuracy"]
        
        plot_train_loss.append(train_loss)
        plot_train_acc.append(train_acc)
        
        result = f"Train - Epoch: {trainer.state.epoch} "
        if hparams.sngp:
            result += f"Loss: {train_loss:.2f} "
            result += f"Accuracy: {train_acc :.2f} "
        else:
            result += f"ELBO: {train_loss:.2f} "
            result += f"Accuracy: {train_acc :.2f} "
            
        print(f"Training...{result}")
        
        # writer.add_scalar("Train/Loss", train_loss, trainer.state.epoch)
        # writer.add_scalar("Train/Accuracy", train_acc, trainer.state.epoch)

        _, auroc, aupr = get_ood_metrics(hparams.dataset, "Alzheimer", model, likelihood)
        
        plot_auroc.append(auroc)
        plot_aupr.append(aupr)
        
        print(f"OoD Metrics - AUROC: {auroc}, AUPR: {aupr}")
        
        # if trainer.state.epoch > 10 and trainer.state.epoch % 5 == 0:
            # writer.add_scalar("OoD/auroc", auroc, trainer.state.epoch)
            # writer.add_scalar("OoD/auprc", aupr, trainer.state.epoch)
        
        all_cal_smx = []
        all_cal_labels = []

        def accumulate_outputs(engine):
            # Access the output of the engine, which is (smx, labels) for each batch
            smx, labels = engine.state.output
            
            ########
            if not hparams.sngp:
                smx = smx.to_data_independent_dist()
                smx = likelihood(smx).probs.mean(0)
            # Append the outputs to the lists
            all_cal_smx.append(smx)
            all_cal_labels.append(labels)

        # Attach the handler to the evaluator
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)

        ######### 
        val_state = evaluator.run(val_loader)
        
        # After evaluation, concatenate all the accumulated outputs
        cal_smx = torch.cat(all_cal_smx, dim = 0)
        cal_labels = torch.cat(all_cal_labels, dim = 0)
        
        # print("cal_smx", cal_smx.shape, "cal_labels", cal_labels.shape)
        
        metrics = evaluator.state.metrics
        val_acc = metrics["accuracy"]
        val_loss = metrics["loss"]
        
        plot_val_loss.append(val_loss)
        plot_val_acc.append(val_acc)
            
        inefficiency_metric.cal_smx = cal_smx
        inefficiency_metric.cal_labels = cal_labels
        
        inefficiency_metric.update((cal_smx, cal_labels))  # Pass the entire validation set
        inefficiency = inefficiency_metric.compute().item()
        
        print(f"Validation..- Epoch: {trainer.state.epoch} " 
            f"Val Accuracy: {val_acc:.4f} Val Loss: {val_loss:.4f} "
            f"Val Inefficiency: {inefficiency:.4f}" )
        
        # writer.add_scalar("Validation/Accuracy", val_acc, trainer.state.epoch)
        # writer.add_scalar("Validation/Loss", val_loss, trainer.state.epoch)
        # writer.add_scalar("Validation/Inefficiency", inefficiency, trainer.state.epoch)
        
        # Save the best model based on the smallest inefficiency on val data 
        nonlocal best_inefficiency, best_model_state_inefficiency, best_model_state_ood
        
        if inefficiency < best_inefficiency:
            best_inefficiency = inefficiency
            best_model_state_inefficiency = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
                'inefficiency': best_inefficiency,
                'likelihood': likelihood.state_dict() if not hparams.sngp else None,
            }

            model_saved_path = results_dir / "best_model_inefficiency.pth"
            torch.save(best_model_state_inefficiency, model_saved_path)

            print(f"Best model saved at epoch {trainer.state.epoch} with inefficiency {best_inefficiency:.4f}")
            
        # Save the best model based on the best OOD detection on val data (best_ood)
        nonlocal best_auroc, best_aupr
        
        if auroc >= best_auroc and aupr >= best_aupr:
            best_auroc, best_aupr = auroc, aupr
            best_model_state_ood = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
                'likelihood': likelihood.state_dict() if not hparams.sngp else None,
            }
                
            model_saved_path = results_dir / "best_model_ood.pth"
            torch.save(best_model_state_ood, model_saved_path)
            
            # if likelihood is not None:
            #     likelihood_saved_path = results_dir / "likelihood_ood.pth"
            #     torch.save(likelihood.state_dict(), likelihood_saved_path)
                
            print(f"Best model saved at epoch {trainer.state.epoch} with best_auroc {best_auroc:.4f} and best_aupr {best_aupr:.4f}")
            
            
    results_to_save = {}
        
    @trainer.on(Events.COMPLETED)
    def compute_test_loss_at_last_epoch(trainer):
        print("Training completed. Running evaluation on the test set with the best model.")
        # if best_model_state_inefficiency is not None and best_model_state_ood is not None:

        best_model_state_inefficiency = torch.load(results_dir / "best_model_inefficiency.pth")
        model.load_state_dict(best_model_state_inefficiency['model'])
        likelihood.load_state_dict(best_model_state_inefficiency['likelihood']) if not hparams.sngp else None

        model.eval()  
        likelihood.eval() if not hparams.sngp else None
        
        ood_model = copy.deepcopy(model)
        ood_likelihood = copy.deepcopy(likelihood) if not hparams.sngp else None
        
        best_model_state_ood = torch.load(results_dir / "best_model_ood.pth")
        ood_model.load_state_dict(best_model_state_ood['model'])
        ood_likelihood.load_state_dict(best_model_state_ood['likelihood']) if not hparams.sngp else None
        
        ood_model.eval()  
        ood_likelihood.eval() if not hparams.sngp else None


        all_cal_smx = []
        all_cal_labels = []

        def accumulate_outputs(engine):
            smx, labels = engine.state.output
            #####
            if not hparams.sngp:
                smx = smx.to_data_independent_dist()
                smx = likelihood(smx).probs.mean(0)
            all_cal_smx.append(smx)
            all_cal_labels.append(labels)

        evaluator.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)

        test_state = evaluator.run(test_loader)

        # After evaluation, concatenate all the accumulated outputs
        cal_smx = torch.cat(all_cal_smx, dim=0)
        cal_labels = torch.cat(all_cal_labels, dim=0)
        
        metrics = evaluator.state.metrics
        test_accuracy = metrics["accuracy"]
        test_loss = metrics["loss"]
        
        cal_smx = torch.cat(all_cal_smx, dim=0)
        cal_labels = torch.cat(all_cal_labels, dim=0)

        inefficiency_metric.cal_smx = cal_smx
        inefficiency_metric.cal_labels = cal_labels

        inefficiency_metric.update( (cal_smx, cal_labels) ) # Pass the entire validation set
        # inefficiency_metric.update((test_state.output[0], test_state.output[1]))  # Pass the entire validation set
        inefficiency = inefficiency_metric.compute().item()

        # if not hparams.sngp:
        #     _, auroc, aupr = get_ood_metrics(
        #         hparams.dataset, "Alzheimer", ood_model, ood_likelihood
        #     ) # , hparams.data_root
        # else:
        #     _, auroc, aupr = get_ood_metrics(
        #         hparams.dataset, "Alzheimer", ood_model, None
        #     ) # , hparams.data_root
        
        _, auroc, aupr = get_ood_metrics(
                hparams.dataset, "Alzheimer", ood_model, ood_likelihood
            ) 
            
        print(f"Final Test Accuracy: {test_accuracy:.4f}, Final Test Loss: {test_loss:.4f}", 
                f"Test_state Inefficiency: {inefficiency:.4f}", f"auroc {auroc} ", f"aupr {aupr} ")
            
        results_to_save["auroc_ood_Alzheimer"] = auroc
        results_to_save["aupr_ood_Alzheimer"] = aupr
        results_to_save['test_accuracy'] = test_accuracy
        results_to_save['test_loss'] = test_loss
        results_to_save['Test_inefficiency'] = inefficiency
            
        if hparams.sngp:
            coverage_mean, ineff_list = conformal_evaluate(model, likelihood = None, alpha = hparams.alpha)
            results_to_save["coverage_mean_sngp"] = str(coverage_mean)
            results_to_save["ineff_list_sngp"] = str(ineff_list)
            results_json = json.dumps(results_to_save, indent=4, sort_keys=True)
            (results_dir / "results_sngp.json").write_text(results_json)
            
        else:
            coverage_mean, ineff_list = conformal_evaluate(model, likelihood, alpha = hparams.alpha )
            results_to_save["coverage_mean_gp"] = str(coverage_mean)
            results_to_save["ineff_list_gp"] = str(ineff_list)
            
            results_json = json.dumps(results_to_save, indent=4, sort_keys=True)
            (results_dir / "results_GP.json").write_text(results_json)
            
            # writer.add_scalar("Final Test/Accuracy", test_accuracy, trainer.state.epoch)
            # writer.add_scalar("Final Test/Loss", test_loss, trainer.state.epoch)
            # writer.add_scalar("Final Test/Inefficiency", inefficiency, trainer.state.epoch)

    # Adding a progress bar
    ProgressBar(persist=True).attach(trainer)
    # Start training
    trainer.run(train_loader, max_epochs = hparams.epochs)
    
    writer.close()
    
    # End timing
    end_event.record()
    # Waits for everything to finish running on the GPU
    torch.cuda.synchronize()
    # Calculate the elapsed time (in milliseconds)
    elapsed_time_ms = start_event.elapsed_time(end_event)
    elapsed_time_min = elapsed_time_ms / 60000
    print(f"Elapsed time on GPU: {elapsed_time_min:.3f} min")
    
    plot_training_history(plot_train_loss, plot_val_loss, plot_train_acc, plot_val_acc)
    plot_OOD(plot_auroc, plot_aupr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default = 64, help="Batch size to use for training")

    parser.add_argument("--learning_rate", type = float, default = 0.1, help = "Learning rate",)
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--dropout_rate", type=float, default = 0.3, help="Dropout rate")
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--number_of_class", type = int, default = 4)
    parser.add_argument("--alpha", type= float, default= 0.05, help="Conformal Rate" )
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
        # action="store_false", # "sngp": True,
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
    
    # Create CUDA events to record time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # Start timing
    start_event.record()
    
    main(hparams)
    

