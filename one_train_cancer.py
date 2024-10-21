import os
# Run on GPU 1
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse
import copy
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import EarlyStopping
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood
from due import dkl
from due.wide_resnet import WideResNet
# from due.sngp import Laplace
from lib.datasets import get_dataset
from lib.evaluate_ood import get_ood_metrics
from lib.utils import get_results_directory, Hyperparameters, set_seed,  calculate_and_save_statistics # plot_training_history, plot_OOD,
from lib.evaluate_cp import conformal_evaluate, ConformalTrainingLoss, ConformalInefficiency # UncertaintyMetric
from pathlib import Path
from sngp_wrapper.covert_utils import replace_layer_with_gaussian # convert_to_sn_my,

from torch.utils.data import DataLoader
import csv

# For context see: https://github.com/pytorch/pytorch/issues/47908
NUM_WORKERS = os.cpu_count()

# import wandb
# from functools import partial

# https://datascience.stackexchange.com/questions/31113/validation-showing-huge-fluctuations-what-could-be-the-cause

def set_saving_file(hparams):
    if hparams.force_directory is None:
        results_dir = get_results_directory(hparams.output_dir)
    else:
        os.makedirs(hparams.force_directory, exist_ok=True)
        results_dir = Path(hparams.force_directory)
    return results_dir

def main(hparams):

    #hparams.n_inducing_points = wandb.config.n_inducing_points
    # hparams.learning_rate = wandb.config.learning_rate
    # hparams.dropout_rate = wandb.config.dropout_rate
    # hparams.temperature = wandb.config.temperature
    # hparams.beta = wandb.config.beta

    results_dir = set_saving_file(hparams)
    print(f"save to results_dir {results_dir}")
    writer = SummaryWriter(log_dir=str(results_dir))

    #set_seed(hparams.seed)

    # Data Preparation
    ds = get_dataset(hparams.dataset)
    input_size, num_classes, train_dataset, val_dataset, test_dataset = ds

    if hparams.n_inducing_points is None:
        hparams.n_inducing_points = num_classes
    print(f"Training with {hparams}")

    # Save parameters
    hparams.save(results_dir / "hparams.json")
    
    feature_extractor = WideResNet(
        input_size,
        hparams.spectral_conv,
        hparams.spectral_bn,
        dropout_rate=hparams.dropout_rate,
        coeff=hparams.coeff,
        n_power_iterations=hparams.n_power_iterations,
    )
        
    if hparams.sngp:
        model = WideResNet(
            input_size,
            hparams.spectral_conv,
            hparams.spectral_bn,
            dropout_rate=hparams.dropout_rate,
            coeff=hparams.coeff,
            n_power_iterations=hparams.n_power_iterations,
            num_classes=hparams.number_of_class,
        )
        # spec_norm_replace_list = ["Linear", "Conv2D"]
        # spec_norm_bound = 9.
        GP_KWARGS = {
            'num_inducing': 2048,
            'gp_scale': 1.0,
            'gp_bias': 0.,
            'gp_kernel_type': 'gaussian', #  linear
            'gp_input_normalization': True,
            'gp_cov_discount_factor': -1,
            'gp_cov_ridge_penalty': 1.,
            'gp_output_bias_trainable': False,
            'gp_scale_random_features': False,
            'gp_use_custom_random_features': True,
            'gp_random_feature_type': 'orf',
            'gp_output_imagenet_initializer': True,
            'num_classes': 4,  # 
        }
        
        # Enforcing Spectral-Normalization on each layer 
        # model = convert_to_sn_my(model, spec_norm_replace_list, spec_norm_bound)
        
        # Equipping the model with laplace approximation 
        replace_layer_with_gaussian(container=model, signature="linear", **GP_KWARGS)
        
        ########## Here decide whether conformal training 
        if hparams.conformal_training:
            loss_fn = ConformalTrainingLoss(alpha=hparams.alpha, beta=hparams.beta, temperature=hparams.temperature, sngp_flag=True)
        else:
            loss_fn = F.cross_entropy
        likelihood = None
        
    else:
        initial_inducing_points, initial_lengthscale = dkl.initial_values(
            train_dataset, feature_extractor, hparams.n_inducing_points )
        
        gp = dkl.GP(
            num_outputs=num_classes,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=hparams.kernel,
        )
        # Model for inducing points 
        model = dkl.DKL(feature_extractor, gp)
        # Print state_dict keys to debug
        print("Model state dict keys before training:", model.state_dict().keys())

        likelihood = SoftmaxLikelihood(num_features=hparams.number_of_class, num_classes=num_classes, mixing_weights=False)
        likelihood = likelihood.cuda()
        elbo_fn = VariationalELBO(likelihood, gp, num_data=len(train_dataset))
        loss_fn = lambda x, y: -elbo_fn(x, y)
        
    model = model.cuda()

    #### Note:
    parameters = [ {"params": model.parameters() } ]

    # if not hparams.sngp:
    #     parameters.append( {"params": likelihood.parameters()} )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay
    )

    training_steps = len(train_dataset) // hparams.batch_size * hparams.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_steps)

    best_inefficiency, best_auroc, best_aupr = float('inf'), float('-inf'), float('-inf')
    best_model_state_inefficiency, best_model_state_ood = None, None
    
    # For plotting
    # plot_train_acc, plot_val_acc = [], []
    # plot_train_loss, plot_val_loss = [], []
    # plot_auroc, plot_aupr = [], []
    # Training and Evaluation Logic
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
            CP_size_fn = ConformalTrainingLoss(alpha=hparams.alpha, beta=hparams.beta, temperature=hparams.temperature, sngp_flag=False)
            loss_cn = loss_fn(y_pred, y)
            y_pred_temp = y_pred.to_data_independent_dist()
            y_pred_temp = likelihood(y_pred_temp).probs.mean(0)
            #### Conformal training loss
            loss_size = CP_size_fn(y_pred_temp, y)
            loss = loss_cn + loss_size
            print(f"Total loss: {(loss.item()+ loss_size):.4f} | ELBO: {loss.item():.4f} | Size loss: {loss_size:.4f}")
        else: 
            loss = loss_fn(y_pred, y)
            
        loss.backward()
        optimizer.step() # Update weights
        scheduler.step() # Adjust learning rate
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
    
    def training_accuracy(output):
        y_pred, y, loss = output
        if not hparams.sngp:
            y_pred = y_pred.to_data_independent_dist()
            y_pred = likelihood(y_pred).probs.mean(0)
        return y_pred, y
    
    def training_loss(output):
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

    metric = Average(output_transform=training_loss)
    metric.attach(trainer, "loss")

    metric = Accuracy(output_transform=training_accuracy)
    metric.attach(trainer, "accuracy")

    metric = Accuracy(output_transform=output_transform)
    metric.attach(evaluator, "accuracy")

    if hparams.sngp:
        metric = Loss(F.cross_entropy)
        metric.attach(evaluator, "loss")
    else:
        metric = Loss(lambda y_pred, y: -likelihood.expected_log_prob(y, y_pred).mean())
        metric.attach(evaluator, "loss")

    kwargs = {"num_workers": NUM_WORKERS, "pin_memory": True}
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size,
                                               shuffle=True,  **kwargs) # drop_last = True,
    val_loader = DataLoader(val_dataset, batch_size=hparams.batch_size,
                                             shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size,
                                              shuffle=False, **kwargs )
    if hparams.sngp:
        @trainer.on(Events.EPOCH_STARTED)
        def reset_precision_matrix(trainer):
            model.linear.reset_covariance_matrix()

    inefficiency_metric = ConformalInefficiency(alpha=hparams.alpha)
    ##inefficiency = inefficiency_metric.compute().item()
    # inefficiency = inefficiency_metric.compute()

    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    early_stopping_handler = EarlyStopping(
            patience=20,  # Stop after 10 epochs without improvement
            score_function=score_function,
            trainer=trainer
    )
    evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    # Define a function to be executed at the end of each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        train_loss = metrics["loss"]
        train_acc = metrics["accuracy"]

        result = f"Train - Epoch: {trainer.state.epoch} | "
        if hparams.sngp:
            result += f"Loss: {train_loss:.2f} | Accuracy: {train_acc:.2f} "
        else:
            result += f"ELBO: {train_loss:.2f} | Accuracy: {train_acc:.2f} "
        print(result)

        _, auroc, aupr = get_ood_metrics(hparams.dataset, "Alzheimer", model, likelihood)
        print(f"OoD Metrics - AUROC: {auroc :.4f} | AUPR: {aupr :.4f}")

        all_cal_smx = []
        all_cal_labels = []

        def accumulate_outputs(engine):
            # Access the output of the engine, which is (smx, labels) for each batch
            smx, labels = engine.state.output
            if not hparams.sngp:
                smx = smx.to_data_independent_dist()
                smx = likelihood(smx).probs.mean(0)
            # Append the outputs to the lists
            all_cal_smx.append(smx)
            all_cal_labels.append(labels)

        # Attach the handler to the evaluator
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)
        val_state = evaluator.run(val_loader)
        
        # After evaluation, concatenate all the accumulated outputs
        cal_smx = torch.cat(all_cal_smx, dim=0)
        cal_labels = torch.cat(all_cal_labels, dim=0)
        
        metrics = evaluator.state.metrics
        val_acc = metrics["accuracy"]
        val_loss = metrics["loss"]
        
        inefficiency_metric.cal_smx = cal_smx
        inefficiency_metric.cal_labels = cal_labels
        
        inefficiency_metric.update((cal_smx, cal_labels))  # Pass the entire validation set
        inefficiency = inefficiency_metric.compute().item()
        
        print(f"Validation - Epoch: {trainer.state.epoch} | " 
            f"Val Accuracy: {val_acc:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Inefficiency: {inefficiency:.4f}" )

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
            # Ensure the results directory exists
            results_dir.mkdir(parents=True, exist_ok=True)
            # Debugging output
            print("model_saved_path", model_saved_path)
            # print(f"Model state dict keys: {model.state_dict().keys()}")
            # print(f"Optimizer state dict keys: {list(optimizer.state_dict().keys())}")
            try:
                # Attempt to save the model
                torch.save(best_model_state_inefficiency, model_saved_path)
                print(f"Best model saved at epoch {trainer.state.epoch} with inefficiency {best_inefficiency:.4f}")
            except Exception as e:
                print(f"Failed to save model due to: {e}")

        
        # # Save the best model based on the best OOD detection on val data (best_ood)
        # nonlocal best_auroc, best_aupr
        #
        # if auroc >= best_auroc and aupr >= best_aupr:
        #     best_auroc, best_aupr = auroc, aupr
        #     best_model_state_ood = {
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'epoch': trainer.state.epoch,
        #         'likelihood': likelihood.state_dict() if not hparams.sngp else None,
        #     }
        #
        #     model_saved_path = results_dir / "best_model_ood.pth"
        #     # Ensure the results directory exists
        #     results_dir.mkdir(parents=True, exist_ok=True)
        #     # Debugging output
        #     print("model_saved_path", model_saved_path)
        #
        #     try:
        #         # Attempt to save the model
        #         torch.save(best_model_state_ood, model_saved_path)
        #         print(f"Best model saved at epoch {trainer.state.epoch} with best_auroc {best_auroc:.4f} and best_aupr {best_aupr:.4f}")
        #     except Exception as e:
        #         print(f"Failed to save model due to: {e}")


    result = {}
    # Define a function to be executed when training is completed
    @trainer.on(Events.COMPLETED)
    def compute_test_loss_at_last_epoch(trainer):
        print("Training completed. Running evaluation on the test set with the best model.")
        # if best_model_state_inefficiency is not None and best_model_state_ood is not None:

        best_model_state_inefficiency = torch.load(results_dir / "best_model_inefficiency.pth")
        model.load_state_dict(best_model_state_inefficiency['model'], strict=False)
        likelihood.load_state_dict(best_model_state_inefficiency['likelihood']) if not hparams.sngp else None

        model.eval()  
        likelihood.eval() if not hparams.sngp else None

        # # ensuring that changes to the new object do not affect the original object
        # ood_model = copy.deepcopy(model)
        # ood_likelihood = copy.deepcopy(likelihood) if not hparams.sngp else None
        #
        # best_model_state_ood = torch.load(results_dir / "best_model_ood.pth")
        # ood_model.load_state_dict(best_model_state_ood['model'], strict=False)
        # ood_likelihood.load_state_dict(best_model_state_ood['likelihood']) if not hparams.sngp else None
        #
        # ood_model.eval()
        # ood_likelihood.eval() if not hparams.sngp else None

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

        _, auroc, aupr = get_ood_metrics(
                hparams.dataset, "Alzheimer", model, likelihood)

        print(f"Test Accuracy: {test_accuracy:.4f} | Test Loss: {test_loss:.4f} |" 
                f"Test_state Inefficiency: {inefficiency:.4f} | "  f"auroc {auroc:.4f} | ", f"aupr {aupr:.4f} ")
            
        result["auroc"] = round(auroc, 4)
        result["aupr"] = round(aupr, 4)
        result['acc'] = round(test_accuracy, 4)
        result['loss'] = round(test_loss, 4)
        result['ineff'] = round(inefficiency, 4)

        if hparams.sngp:
            coverage_mean, ineff_list = conformal_evaluate(model, likelihood=None, dataset=hparams.dataset,
                                                           adaptive_flag=hparams.adaptive_conformal, alpha=hparams.alpha)
            result["coverage_mean"] = round(coverage_mean, 4)
            result["ineff_list"] = round(ineff_list, 4)
        else:
            coverage_mean, ineff_list = conformal_evaluate(model, likelihood, dataset=hparams.dataset,
                                                           adaptive_flag=hparams.adaptive_conformal, alpha=hparams.alpha )
            result["coverage_mean"] = round(coverage_mean,4)
            result["ineff_list"] = round(ineff_list, 4)

        # if trainer.state.epoch > 10 and trainer.state.epoch % 10 == 0:
        #     wandb.log({"Test_Loss": test_loss, "Test_Accuracy": test_accuracy,
        #                "Epoch": trainer.state.epoch, "Test_AUROC": auroc, "Test_AUPR": aupr,
        #                "Test_inefficiency":inefficiency, "coverage_mean":coverage_mean, "ineff_list":ineff_list})

    # Adding a progress bar
    ProgressBar(persist=True).attach(trainer)
    # Start training
    trainer.run(train_loader, max_epochs=hparams.epochs)
    writer.close()
    # plot_training_history(plot_train_loss, plot_val_loss, plot_train_acc, plot_val_acc)
    # plot_OOD(plot_auroc, plot_aupr)
    return result

# Define a function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate") # sngp = 0.05
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use for training")
    parser.add_argument("--number_of_class", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05, help="Conformal Rate" )
    parser.add_argument("--dataset", default="Brain_tumors", choices=["Brain_tumors", "Alzheimer",'CIFAR10', 'CIFAR100', "SVHN"])
    parser.add_argument("--n_inducing_points", type=int, default=12, help="Number of inducing points" ) # 12
    parser.add_argument("--beta", type=int, default=0.1, help="Weight for conformal training loss")
    parser.add_argument("--temperature", type=int, default=1., help="Temperature for conformal training loss")
    parser.add_argument("--sngp", action="store_true", help="Use SNGP (RFF and Laplace) instead of a DUE (sparse GP)")
    parser.add_argument("--conformal_training", action="store_true", help="conformal training or not" )
    parser.add_argument("--force_directory", default="temp")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay") # 5e-4
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--kernel", default="RBF", choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"], help="Pick a kernel",)
    parser.add_argument("--no_spectral_conv", action="store_false",  dest="spectral_conv", help="Don't use spectral normalization on the convolutions",)
    parser.add_argument( "--adaptive_conformal", action="store_true", help="adaptive conformal")
    parser.add_argument("--no_spectral_bn", action="store_false", dest="spectral_bn", help="Don't use spectral normalization on the batch normalization layers",)
    # parser.add_argument("--seed", type=int, nargs='+', default=[23], help="List of seeds to use for training")
    parser.add_argument("--coeff", type=float, default=3., help="Spectral normalization coefficient")
    parser.add_argument("--n_power_iterations", default=1, type=int, help="Number of power iterations")
    parser.add_argument("--output_dir", default="./default", type=str, help="Specify output directory")
    args = parser.parse_args()
    return args 
    
def run_main(args):
    seeds = [1, 7, 23, 42, 56] #[1, 7, 23, 42, 56]
    # For the final results.csv file
    dict_list = []
    #run = wandb.init()
    hparams = Hyperparameters(**vars(args))

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    ## python your_script.py --seed 42 99 123

    for seed in seeds:
        set_seed(seed)
        result = main(hparams)
        dict_list.append(result)

    if hparams.sngp:
        file_name = f"SNGP_{hparams.epochs}.csv"
    else:
        file_name = f"SNGPIP_{hparams.epochs}.csv"

    with open(f'{file_name}', mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_list[0].keys())
        # Write the header (column names)
        writer.writeheader()
        # Write each dictionary as a row in the CSV file
        writer.writerows(dict_list)

    # The file has been saved automatically after the 'with' block
    print(f"Data successfully written to {csv_file}")

    calculate_and_save_statistics(file_name)

    end_event.record()
    torch.cuda.synchronize()
    # Print the elapsed time
    elapsed_time = start_event.elapsed_time(end_event)
    elapsed_time_min = elapsed_time / 60000
    print(f"Elapsed time on GPU: {elapsed_time_min:.3f} min")
    # wandb.finish()
    
if __name__ == "__main__":

    args = parse_arguments()
    #### Run without wandb
    run_main(args)

    # wandb.login()
    # # Step 1: Define a sweep
    # sweep_config = {'method': 'grid'}
    # metric = {'name': 'loss',
    #          'goal': 'minimize' }
    # sweep_config['metric'] = metric

    ### sngp
    # parameters = {'dropout_rate': {'values': [0.3] },
    #               'learning_rate': {'values': [0.001]},
    #      # 'beta':{"values":[0.01]},
    #      # 'temperature': {"values": [1]},
    #              }

    # ## Inducing Points
    # parameters = {'dropout_rate': {'values': [0.3] },
    #               'learning_rate' : {'values':[0.001] },
    #                }

    #
    # sweep_config['parameters'] = parameters
    # ### Step 2: Initialize the Sweep
    # sweep_id = wandb.sweep(sweep=sweep_config, project="One_Model_IPGP")
    # ###Step 4: Activate sweep agents
    # wandb.agent(sweep_id, function=partial(run_main, args=args) , count=1)
