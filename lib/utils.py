from datetime import datetime
import json
import pathlib
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

def calculate_and_save_statistics(file_name):
    df = pd.read_csv(file_name)
    # Calculate the mean and variance for each column
    mean_values = round(df.mean(), 4)
    variance_values = round(df.var(), 4)

    # Print the results
    print(f"Mean of each column: \n{mean_values}")
    print(f"\nVariance of each column: \n{variance_values}")

    # Optional: Save the mean and variance to another CSV file
    tmp = file_name.split('.')[0]
    mean_variance_file = f"{tmp}_output.csv"

    mean_variance_df = pd.DataFrame({'Mean': mean_values, 'Variance': variance_values})
    mean_variance_df.to_csv(mean_variance_file)

    print(f"\nMean and variance saved to {mean_variance_file}")

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def plot_training_history(train_loss, val_loss, train_acc, val_acc):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Learning_process.pdf')

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


# path = pathlib.Path("runs")
# # Check if the path exists
# if path.exists():
#     print(f"The path {path} exists.")
# else:
#     print(f"The path {path} does not exist.")
#     # Create the directory if it does not exist # 创建路径path.mkdir()
#     path.mkdir()
#     print(f"Directory {path} created.")

# # Create a Path object for the "runs" directory
def get_results_directory(name, stamp=True):
    # 2024-06-30-Sunday-20-22-46
    timestamp = datetime.now().strftime("%Y-%m-%d-%A-%H-%M-%S")
    # Create a Path object for the "runs" directory
    results_dir = pathlib.Path("runs")
    if name is not None:
        results_dir = results_dir / name
    # "runs/2024-06-30-Sunday-20-22-46"
    results_dir = results_dir / timestamp if stamp else results_dir
    # Create the directory and any missing parent directories
    results_dir.mkdir(parents=True)
    return results_dir


# The Hyperparameters class manages hyperparameters for a model,
# allowing them to be easily saved, loaded, and updated.
# It supports initialization from a file and dynamic updates via keyword arguments.
class Hyperparameters:
    def __init__(self, *args, **kwargs):
        # If the first argument is a Path object, load hyperparameters from the file
        if len(args) == 1 and isinstance(args[0], Path):
            self.load(args[0])
        # Otherwise, initialize hyperparameters from keyword arguments
        self.from_dict(kwargs)

    # convert hyperparameters to dictionary
    # class ExampleClass:
    #     def __init__(self, name, value):
    #         self.name = name
    #         self.value = value
    #     def to_dict(self):
    #         return vars(self)

    # example = ExampleClass(name="example", value=42)
    # example_dict = example.to_dict()
    # print(example_dict)  # Output: {'name': 'example', 'value': 42}

    #  The to_dict method converts the attributes of the object to a dictionary.
    def to_dict(self):
        return vars(self)

    # The from_dict method updates the attributes of the object using a dictionary.
    def from_dict(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

    # class MyClass:
    #     def from_dict(self, dictionary):
    #         for k, v in dictionary.items():
    #             setattr(self, k, v)
    # data = {
    #     "name": "Alice",
    #     "age": 30,
    #     "occupation": "Engineer"
    # }
    # obj = MyClass()
    # obj.from_dict(data)

    # print(obj.name)       # Alice
    # print(obj.age)        # 30
    # print(obj.occupation) # Engineer

    # Converts the hyperparameters to a JSON-formatted string for easy readability and storage
    def to_json(self):
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)

    # Saves the hyperparameters to a specified path in JSON format.
    def save(self, path):
        path.write_text(self.to_json())

    # Loads hyperparameters from a JSON file and updates the object's attributes
    def load(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        self.from_dict(json.loads(path.read_text()))

    # Checks if a hyperparameter exists using the hasattr function
    def __contains__(self, k):
        return hasattr(self, k)

    # provides a readable string representation of the hyperparameters in JSON format
    def __str__(self):
        return f"Hyperparameters:\n {self.to_json()}"


''' 
# Create hyperparameters from a dictionary
hyperparams = Hyperparameters(learning_rate=0.01, batch_size=32)

# Save hyperparameters to a file
hyperparams.save( Path('hyperparams.json') ) # 建立一个hyperparams.json的文件, 并保存相关参数

# Load hyperparameters from a file
loaded_hyperparams = Hyperparameters(Path('hyperparams.json'))

# Print hyperparameters
print(loaded_hyperparams)

'''