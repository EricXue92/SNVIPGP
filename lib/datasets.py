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

def get_SVHN():
    input_size = 32
    num_classes = 10
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.SVHN(
        "data/SVHN", split="train", transform=transform, download=False
    )
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    test_dataset = datasets.SVHN(
        "data/SVHN", split="test", transform=transform, download=False
    )
    return input_size, num_classes, train_dataset, val_dataset, test_dataset

def get_CIFAR10():
    input_size = 32
    num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR10(
        "data/CIFAR10", train=True, transform=train_transform, download=False
    )
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    # Split the dataset
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR10(
        "data/CIFAR10", train=False, transform=test_transform, download=False
    )
    return input_size, num_classes, train_dataset, val_dataset, test_dataset

def get_CIFAR100():
    input_size = 32
    num_classes = 100
    normalize = transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR100(
        "data/CIFAR100", train=True, transform=train_transform, download=True
    )
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR100(
        "data/CIFAR100", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, val_dataset, test_dataset


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


def get_Brain_tumors():
    image_path = "./data/Brain_tumors"
    crop_pct = 224 / 236
    size = int( 224 / crop_pct)
    train_transform = v2.Compose([
                    v2.Resize(size, interpolation=v2.InterpolationMode.BILINEAR),
                    v2.CenterCrop(224),
                    v2.ToTensor(),
                    v2.Resize(64, interpolation=v2.InterpolationMode.BILINEAR),
                    v2.Normalize(IMAGENET_CONVNEXT_MEAN, IMAGENET_CONVNEXT_STD),
                ])
    val_test_transform = v2.Compose([
        v2.Resize(size, interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Resize(64, interpolation=v2.InterpolationMode.BILINEAR),
        v2.Normalize(IMAGENET_CONVNEXT_MEAN, IMAGENET_CONVNEXT_STD),
    ])

    data = datasets.ImageFolder(root=image_path)
    input_size = 64
    num_classes = 4

    targets = [sample[1] for sample in data.imgs]

    train_indices, temp_indices, _, temp_labels = train_test_split(
        range(len(data)), targets, test_size=0.2, stratify=targets, random_state=23
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, stratify=temp_labels, random_state=23
    )

    train_dataset = TransformedDataset(Subset(data, train_indices), transform=train_transform)
    val_dataset = TransformedDataset(Subset(data, val_indices), transform=val_test_transform)
    test_dataset = TransformedDataset(Subset(data, test_indices), transform=val_test_transform)


    return input_size, num_classes, train_dataset, val_dataset, test_dataset


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

def get_Alzheimer():
    image_path = "./data/Alzheimer"
    crop_pct = 224 / 236
    size = int( 224 / crop_pct)
    train_transform = v2.Compose([
                    v2.Resize(size, interpolation=v2.InterpolationMode.BILINEAR),
                    v2.CenterCrop(224),
                    v2.ToTensor(),
                    v2.Resize(64, interpolation=v2.InterpolationMode.BILINEAR),
                    v2.Normalize(IMAGENET_CONVNEXT_MEAN, IMAGENET_CONVNEXT_STD),
                ])
    val_test_transform = v2.Compose([
        v2.Resize(size, interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Resize(64, interpolation=v2.InterpolationMode.BILINEAR),
        v2.Normalize(IMAGENET_CONVNEXT_MEAN, IMAGENET_CONVNEXT_STD),
    ])
    data = datasets.ImageFolder(root=image_path)
    input_size = 64
    num_classes = 4
    targets = [sample[1] for sample in data.imgs]
    train_indices, temp_indices, _, temp_labels = train_test_split(
        range(len(data)), targets, test_size=0.2, stratify=targets, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, stratify=temp_labels, random_state=42
    )

    train_dataset = TransformedDataset(Subset(data, train_indices), transform=train_transform)
    val_dataset = TransformedDataset(Subset(data, val_indices), transform=val_test_transform)
    test_dataset = TransformedDataset(Subset(data, test_indices), transform=val_test_transform)

    return input_size, num_classes, train_dataset, val_dataset, test_dataset

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

all_datasets = {
    "SVHN": get_SVHN,
    "CIFAR10": get_CIFAR10,
    "CIFAR100": get_CIFAR100,
    "Brain_tumors": get_Brain_tumors,
    "Alzheimer": get_Alzheimer
}

all_feature_datasets = {
    "Brain_tumors": get_tumors_feature(image_path="./data_feature/Brain_tumors"),
    "Alzheimer": get_tumors_feature(image_path="./data_feature/Alzheimer"),
}

def get_feature_dataset(dataset):
    return all_feature_datasets[dataset]

def get_dataset(dataset):  # root="./"
    return all_datasets[dataset]()

def get_dataloaders(dataset, train_batch_size=64):  # 128
    ds = all_datasets[dataset]()
    input_size, num_classes, train_dataset, val_dataset, test_dataset = ds
    kwargs = {"num_workers": NUM_WORKERS, "pin_memory": True}
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_batch_size, shuffle=False, **kwargs
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=train_batch_size, shuffle=False, **kwargs  # 1000
    )
    return train_loader, val_loader, test_loader, input_size, num_classes

# if __name__ == "__main__":
#     feauture_f = FeatureDataset('../data_feature/Brain_tumors')
#     print("len(feauture_f):", len(feauture_f))
#     print("feauture_f[0]:", feauture_f[0][0].shape, feauture_f[0][1])


