import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np 
import argparse
import json
import torch
import torch.nn.functional as F
# torch.cuda.empty_cache()
# help you visualize metrics like loss, accuracy, and other statistics during training through TensorBoard
from torch.utils.tensorboard.writer import SummaryWriter
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


# Custom metric: efficiency 
class ConformalInefficiency(Metric):
    def __init__(self, cal_smx, cal_labels, n, alpha = 0.05, output_transform = lambda x: x ):
        # self.cal_smx = cal_smx 
        # self.cal_labels = cal_labels 
        self.n = n
        self.alpha = alpha 
        super(ConformalInefficiency, self).__init__(output_transform = output_transform)
        
    def reset(self):
        self.eff = 0
        
    def update(self, output):
        val_smx, val_labels = output
        cal_scores = 1 - self.cal_smx[torch.arange(self.n), self.cal_labels]
        q_level = np.ceil((self.n + 1) * (1 - self.alpha)) / self.n
        qhat = torch.quantile(cal_scores, q_level, interpolation='midpoint')
        prediction_sets = val_smx >= (1 - qhat)
        self.eff = torch.sum(prediction_sets) / len(prediction_sets)
    
    def compute(self):
        return self.eff 
        
    
# hparams是自定义Hyperparameters class的对象, 方便saved, loaded, and updated 参数
# 该对象的属性值 来自于 用户的输入和默认的值，并包括各种自定义的函数 

def main(hparams):
    
    # hparams.output_dir = "./default"
    # results_dir = "runs/default/2024-06-30-Sunday-20-22-46(运行时间)"
    results_dir = get_results_directory(hparams.output_dir)
    
    # To visualize the logs, you need to run TensorBoard pointing to the log directory:
    # tensorboard --logdir=path/to/results_dir

    # SummaryWriter -> to keep track of your training process, visualize metrics, and debug your models.
    # log_dir: Directory where the TensorBoard logs will be saved.
    writer = SummaryWriter(log_dir = str(results_dir) ) # E.g. writer.add_scalar('Loss/train', loss.item(), epoch)

    # 
    ds = get_dataset(hparams.dataset) # , root = hparams.data_root
    input_size, num_classes, train_dataset, val_dataset, test_dataset = ds
    hparams.seed = set_seed(hparams.seed)

    # num_classes 和inducing_points的数量 一样
    if hparams.n_inducing_points is None:
        hparams.n_inducing_points = num_classes
    print(f"Training with {hparams}")
    # 保存 自定义的存储参数的函数
    hparams.save(results_dir / "hparams.json")
    # 
    feature_extractor = WideResNet(
        input_size,
        hparams.spectral_conv,
        hparams.spectral_bn,
        dropout_rate = hparams.dropout_rate,
        coeff = hparams.coeff,
        n_power_iterations = hparams.n_power_iterations,
    )

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
            loss_fn = ConformalTrainingLoss(alpha=0.05, beta = 0.1, temperature=1, sngp_flag = True)
        else:
            loss_fn = F.cross_entropy
        likelihood = None
            
    else:
        # 初始化inducing points, 形状参数
        initial_inducing_points, initial_lengthscale = dkl.initial_values(
            train_dataset, feature_extractor, hparams.n_inducing_points
        )
        gp = dkl.GP(
            num_outputs = num_classes,
            initial_lengthscale = initial_lengthscale,
            initial_inducing_points = initial_inducing_points,
            kernel = hparams.kernel,
        )
        model = dkl.DKL(feature_extractor, gp)
        
        likelihood = SoftmaxLikelihood(num_classes = num_classes, mixing_weights = False ) 
        likelihood = likelihood.cuda()
        elbo_fn = VariationalELBO(likelihood, gp, num_data=len(train_dataset))
        loss_fn = lambda x, y: -elbo_fn(x, y)

    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = hparams.learning_rate,
        momentum = 0.9,
        weight_decay = hparams.weight_decay,
    )

    milestones = [60, 120, 160]

    # torch.optim.lr_scheduler.MultiStepLR -> Learning rate scheduler 
    # Adjusts the learning rate of each parameter group by a specified factor (gamma) 
    # once the number of epochs reaches one of the predefined milestones
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones = milestones, gamma=0.2
    )
    
    # This function defines a single training step within the Ignite engine loop.
    # This function is designed to be flexible, allowing easy integration with different models and loss functions
    
    # Save the best model parameters that generate the best inefficiency
    
    best_inefficiency = 10.0
    best_model_state = None
    
    def step(engine, batch):
        model.train()
        if not hparams.sngp:
            likelihood.train()
        optimizer.zero_grad()
        x, y = batch
        x, y = x.cuda(), y.cuda()
        y_pred = model(x)
        
        if (hparams.conformal_training) and (not hparams.sngp):
            CP_size_fn = ConformalTrainingLoss(alpha=0.05, beta = 0.1, temperature=1, sngp_flag = False)
            loss_cn = loss_fn(y_pred, y)
            y_pred = y_pred.to_data_independent_dist()
            # The mean here is over likelihood samples
            y_pred = likelihood(y_pred).probs.mean(0) # (batch_size, num_of_classes)
            
            loss_size = CP_size_fn(y_pred, y)
            print(f"loss size {loss_size} and loss sn {loss_cn}")
            loss = loss_cn + loss_size
            
            #print(f"CP_size_loss {CP_size_fn(y_pred, y).item()} and gp loss {loss - CP_size_fn(y_pred, y).item()}")
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


    def output_loss_transform(output):
        y_pred, y, loss = output 
        return loss 
    
    def out_eff_transform(output):
        y_pred, y, loss = output
        n = len(y)
        alpha = 0.05 
        cal_scores = 1 - y_pred[torch.arange(n), y]
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        qhat = torch.quantile(cal_scores, q_level, interpolation='midpoint')
        prediction_sets = y_pred >= (1 - qhat)
        eff = torch.sum(prediction_sets) / len(prediction_sets)
        return eff 
        
    # Create the trainer engine
    trainer = Engine(step) #trainer: Manages the training loop with the step function
    
    # # Create the evaluation engine 
    evaluator = Engine(eval_step) # evaluator: Manages the evaluation loop with the eval_step function

    # 
    
    
    # Average: Computes the average of a given metric (e.g., loss).
    metric = Average(output_transform = output_loss_transform)
    
    # attach: Associates the metric with the trainer, computing the average loss during training.
    metric.attach(trainer, "loss")  # "loss": The name under which the metric is stored in the trainer's state.

        
    #  The output variable is a tuple (y_pred, y) returned by the eval_step (process function)
    def output_transform(output):
        y_pred, y = output
        # Sample softmax values independently for classification at test time
        y_pred = y_pred.to_data_independent_dist()
        # The mean here is over likelihood samples
        y_pred = likelihood(y_pred).probs.mean(0) # (batch_size, num_of_classes)
        return y_pred, y

    if hparams.sngp:
        output_transform = lambda x: x  # noqa

    metric = Accuracy(output_transform = output_transform)
    # # Attach the Accuracy metric to the evaluator
    metric.attach(evaluator, "accuracy")

    if hparams.sngp:
        metric = Loss(F.cross_entropy)
    else:
        metric = Loss(lambda y_pred, y: -likelihood.expected_log_prob(y, y_pred).mean())

    ##  Attach the loss metric to the evaluator
    metric.attach(evaluator, "loss")

    kwargs = {"num_workers": 4, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size= 32, shuffle=False, **kwargs
    )

    if hparams.sngp:
        @trainer.on(Events.EPOCH_STARTED)
        # Events.EPOCH_STARTED occurs at the beginning of each epoch during training.
        # This is the function that will be executed when the EPOCH_STARTED event occurs.
        def reset_precision_matrix(trainer):
            model.reset_precision_matrix()
    @trainer.on(Events.EPOCH_COMPLETED)
    
    ## The @trainer.on(Events.EPOCH_COMPLETED) decorator registers the on_epoch_completed function to be called 
    ## after every epoch.
    
    def log_results(trainer):
        metrics = trainer.state.metrics
        train_loss = metrics["loss"]
        result = f"Train - Epoch: {trainer.state.epoch} "
        if hparams.sngp:
            result += f"Loss: {train_loss:.2f} "
        else:
            result += f"ELBO: {train_loss:.2f} "
        print(result)

        # "Loss/train" -- This is the hierarchical name under
        # which the scalar value will be categorized and can be viewed in TensorBoard.
        # Loss: A high-level category.
        # train: A sub-category indicating the scalar pertains to training 
        # (as opposed to validation, for instance).
        writer.add_scalar("Loss/train", train_loss, trainer.state.epoch)
        # tensorboard --logdir=runs
        
        if hparams.spectral_conv:
            for name, layer in model.feature_extractor.named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    writer.add_scalar(
                        f"sigma/{name}", layer.weight_sigma, trainer.state.epoch
                    )

        if trainer.state.epoch > 150 and trainer.state.epoch % 5 == 0:
            _, auroc, aupr = get_ood_metrics(
                hparams.dataset, "Alzheimer", model, likelihood, #hparams.data_root
            )
            print(f"OoD Metrics - AUROC: {auroc}, AUPR: {aupr}")
            writer.add_scalar("OoD/auroc", auroc, trainer.state.epoch)
            writer.add_scalar("OoD/auprc", aupr, trainer.state.epoch)

        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        
        acc = metrics["accuracy"]
        test_loss = metrics["loss"]

        result = f"Test - Epoch: {trainer.state.epoch} "
        if hparams.sngp:
            result += f"Loss: {test_loss:.2f} "
        else:
            result += f"NLL: {test_loss:.2f} "
        result += f"Acc: {acc:.4f} "
        print(result)
        # 
        writer.add_scalar("Loss/test", test_loss, trainer.state.epoch)
        writer.add_scalar("Accuracy/test", acc, trainer.state.epoch)
        scheduler.step()

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)
    trainer.run(train_loader, max_epochs= 5)
    
    ###### Done training - time to evaluate OOD
    results = {}
    evaluator.run(test_loader)
    test_acc = evaluator.state.metrics["accuracy"]
    test_loss = evaluator.state.metrics["loss"]
    results["test_accuracy"] = test_acc
    results["test_loss"] = test_loss

    _, auroc, aupr = get_ood_metrics(
        hparams.dataset, "Alzheimer", model, likelihood, #hparams.data_root
    )
    results["auroc_ood_svhn"] = auroc
    results["aupr_ood_svhn"] = aupr

    print(f"Final accuracy {results['test_accuracy']:.4f}")

    #### save the model and likelohood 
    model_saved_path = results_dir / "model.pt"
    likelihood_saved_path = results_dir / "likelihood.pt"
    torch.save(model.state_dict(), model_saved_path)
    if likelihood is not None:
        torch.save(likelihood.state_dict(), likelihood_saved_path)
    
    ###### Done OOD evaluation - time to evaluate the CP
    if hparams.sngp:
        coverage_mean, ineff_list = conformal_evaluate(model, likelihood = None, alpha = 0.05 )
        results["coverage_mean_sngp"] = str(coverage_mean)
        results["ineff_list_sngp"] = str(ineff_list)
        
        results_json = json.dumps(results, indent=4, sort_keys=True)
        (results_dir / "results_sngp.json").write_text(results_json)
    else:
        coverage_mean, ineff_list = conformal_evaluate(model, likelihood, alpha = 0.05 )
        results["coverage_mean_gp"] = str(coverage_mean)
        results["ineff_list_gp"] = str(ineff_list)
        
        results_json = json.dumps(results, indent=4, sort_keys=True)
        (results_dir / "results_GP.json").write_text(results_json)

    writer.close()
    # 
    # model.load_state_dict(torch.load(PATH, weights_only=True))
    # model.eval()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", 
        type=int, # Automatically convert an argument to the given type
        default= 32, # Default value used when an argument is not provided
        help="Batch size to use for training" # Help message for an argument
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--dataset",
        default="Brain_tumors", 
        choices=["Brain_tumors", "Alzheimer"],
        help="Pick a dataset",
    )
    parser.add_argument(
        "--kernel",
        default="RBF",
        choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"],
        help="Pick a kernel",
    )
    parser.add_argument(
        "--no_spectral_conv",
        #######  if the option is specified, assign the value False to args.no_spectral_conv. 
        #######  Not specifying it implies True.
        action="store_false",  # "spectral_conv": true,
        dest="spectral_conv", # # Specify the attribute name used in the result namespace
        help="Don't use spectral normalization on the convolutions",
        
    )
    parser.add_argument(
        "--no_spectral_bn", 
        action="store_false", # # "spectral_bn": true,
        dest="spectral_bn", 
        help="Don't use spectral normalization on the batch normalization layers",
    )
    parser.add_argument(
        "--sngp",
        action="store_false", # "sngp": True,
        # action="store_true", # "sngp": false,
        help="Use SNGP (RFF and Laplace) instead of a DUE (sparse GP)",
    )
    parser.add_argument(
        "--n_inducing_points", type=int, help="Number of inducing points"
    )
    parser.add_argument("--seed", type=int, default=23, help="Seed to use for training")
    parser.add_argument(
        "--coeff", type=float, default=3, help="Spectral normalization coefficient"
    )
    parser.add_argument(
        "--n_power_iterations", default=1, type=int, help="Number of power iterations"
    )
    parser.add_argument(
        "--output_dir", default="./default", type=str, help="Specify output directory"
    )
    parser.add_argument(
        "--conformal_training", 
        # action="store_false", # "conformal_training": True,
        action="store_true", # "conformal_training": false,
        help="Specify output directory"
    )
    # parser.add_argument(
    #     "--data_root", default="./data", type=str, help="Specify data directory"
    # )
    args = parser.parse_args()
    # 把读取的 输入参数 作为一个字典传入 Hyperparameters 类，构造一个hparams对象
    # vars(object): 对象object的属性和属性值的字典对象
    # Convert args to a dictionary and unpack into Hyperparameters  
    hparams = Hyperparameters( ** vars(args) )
    main(hparams)


# class ExampleClass:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#     example_instance = ExampleClass(10, 20)
#     # var(object) 
#     # Use vars() to get the __dict__ attribute of the instance
#     instance_vars = vars(example_instance) --> {'x': 10, 'y': 20}

# E.g. enroll(name, gender, age=6, city='Beijing') 位置参数在前，默认参数在后，一般变化小的参数就可以作为默认参数 

# 可变参数: 传入的参数个数是可变的, E.g.  nums = [1, 2, 3];  calc(*nums);
# def calc(*numbers):
# sum = 0
# for n in numbers:
#     sum = sum + n * n
# return sum
# calc(1,2) 或 calc（1,2,3）

# Python允许你在list或tuple前面加一个*号，把list或tuple的元素变成可变参数传进去

# *nums表示把nums这个list的所有元素作为可变参数传进去
# *args是可变参数, args接收的是一个tuple

# 关键字参数： 函数的调用者可以传入任意不受限制的关键字参数
# **kw是关键字参数,kw接收的是一个dict
