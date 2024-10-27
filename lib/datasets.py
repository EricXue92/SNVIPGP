import os
import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset, random_split, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import glob

NUM_WORKERS = os.cpu_count()

IMAGENET_CONVNEXT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_CONVNEXT_STD = [0.229, 0.224, 0.225]

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        if self.transform:
            if isinstance(self.transform, tuple):
                raise ValueError("Transform is a tuple, expected a callable function.")
            data = self.transform(data)
        return data, target

class FeatureDataset(Dataset):
    def __init__(self, feature_dir):
        # Initialize the dataset by loading all feature file paths
        self.feature_paths = glob.glob(os.path.join(feature_dir, '**/*.pt'), recursive=True)
        if "Brain_tumors" in feature_dir:
            self.label_map = {"glioma_tumor":0, "meningioma_tumor": 1, "normal": 2, "pituitary_tumor": 3}
        elif "Alzheimer" in feature_dir:
            self.label_map = {"MildDemented":0, "ModerateDemented": 1, "NonDemented": 2, "VeryMildDemented": 3}
        else:
            raise ValueError("Unknown dataset")
    def __len__(self):
        return len(self.feature_paths)
    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        feature = torch.load(feature_path).float()
        # os.path.dirname: Return the directory name of pathname path.
        # os.path.basename: retrieves the last part of a path
        label = os.path.basename(os.path.dirname(feature_path))
        return feature, self.label_map[label]


class CustomDataset(Dataset):
    def __init__(self, representations, labels):
        self.representations = representations
        self.labels = labels

    def __len__(self):
        return len(self.representations)

    def __getitem__(self, idx):
        return self.representations[idx], self.labels[idx]


def get_tumors_feature(image_path: str = "./data/Brain_tumors"):

    full_dataset = FeatureDataset(image_path)

    input_size = 768
    num_classes = 4

    targets = [full_dataset[i][1] for i in range(len(full_dataset))]

    train_indices, temp_indices, _, temp_labels = train_test_split(
        range(len(full_dataset)), targets, test_size=0.2, stratify=targets, random_state=23
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, stratify=temp_labels, random_state=23
    )

    train_dataset = TransformedDataset(Subset(full_dataset, train_indices), transform=None)
    val_dataset = TransformedDataset(Subset(full_dataset, val_indices), transform=None)
    test_dataset = TransformedDataset(Subset(full_dataset, test_indices), transform=None)

    return input_size, num_classes, train_dataset, val_dataset, test_dataset


def get_cifar10(image_path="../data_feature/CIFAR10", train_flag=True):
    file_path = os.path.join(image_path, "train") if train_flag else os.path.join(image_path,"test")
    file_path = os.path.join(file_path, "CIFAR10.pt")
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    input_size, num_classes = 768, 10
    # Load the features and labels
    data = torch.load(file_path)
    representations = data['features']
    labels = data['labels']

    print("representations.shape:", representations.shape, "labels.shape:", labels.shape)

    if train_flag:
        dataset_size = len(representations)
        train_size = int(0.8 * dataset_size)
        indices = list(range(dataset_size))

        train_indices, val_indices = indices[:train_size], indices[train_size:]

        train_dataset = Subset(CustomDataset(representations, labels), train_indices)
        val_dataset = Subset(CustomDataset(representations, labels), val_indices)

        return input_size, num_classes, train_dataset, val_dataset

    else:
        train_data = CustomDataset(representations, labels)

        return input_size, num_classes, train_data


def get_feature_dataset(dataset):
    return all_feature_datasets[dataset]


all_feature_datasets = {
    # "Brain_tumors": get_tumors_feature(image_path="./data_feature/Brain_tumors"),
    # "Alzheimer": get_tumors_feature(image_path="./data_feature/Alzheimer"),
    "CIFAR10": get_cifar10(),
    #"SVHM": get_cifar10_or_svhm_feature(dataset_name="SVHM", train_flag=True),
}


if __name__ == "__main__":

    get_cifar10()


#     feauture_f = FeatureDataset('../data_feature/Brain_tumors')
#     print("len(feauture_f):", len(feauture_f))
#     print("feauture_f[0]:", feauture_f[0][0].shape, feauture_f[0][1])


