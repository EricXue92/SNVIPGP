import os
import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset, random_split, DataLoader, Subset
from sklearn.model_selection import train_test_split

NUM_WORKERS = os.cpu_count()

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
        "data/CIFAR100", train=True, transform=train_transform, download=False
    )
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR100(
        "data/CIFAR100", train=False, transform=test_transform, download=False
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

# https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html
# https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256?resource=download
# 3 * 256 * 256
# 2476 310 310
def get_Brain_tumors():
    image_path = "./data/Brain_tumors"

    ### For get the training mean and std
    # transform = transforms.Compose([
    #     transforms.Resize((64, 64)),  # Resize images to input size
    #     transforms.ToTensor(),  # Convert images to tensors
    # ])

    train_transform = transforms.Compose([
        transforms.Resize([64,64]),
        # randomly crops a 64x64 region from the image
        transforms.RandomCrop(64, padding=4),
        # randomly flips the image horizontally
        transforms.RandomHorizontalFlip(),
        # rotates the image randomly within a range of ±15 degrees
        transforms.RandomRotation(degrees=15),
        # converts a PIL Image or NumPy ndarray into a PyTorch tensor.
        # scales the pixel values from a range of [0, 255] to [0.0, 1.0]
        transforms.ToTensor(),
        transforms.Normalize((0.2114, 0.2114, 0.2114), (0.1894, 0.1894, 0.1894) )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize([64,64]),
        transforms.ToTensor(),
        transforms.Normalize((0.2114, 0.2114, 0.2114), (0.1894, 0.1894, 0.1894) )
    ])

    ### For get the training mean and std
    # data = datasets.ImageFolder(root=image_path, transform=transform)

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

    # x = torch.stack([sample[0] for sample in Subset(data, train_indices)])
    # #get the mean of each channel
    # mean = torch.mean(x, dim=(0,2,3))
    # std = torch.std(x, dim=(0,2,3))
    # print(f"mean : {mean}")
    # print(f"std : {std}")

    return input_size, num_classes, train_dataset, val_dataset, test_dataset




## https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset
# 3 * 208 * 176
# 5120 640 640
def get_Alzheimer():
    image_path = "./data/Alzheimer"
    # ### For get the training mean and std
    # transform = transforms.Compose([
    #     transforms.Resize((64, 64)),  # Resize images to input size
    #     transforms.ToTensor(),  # Convert images to tensors
    # ])
    # normalize = transforms.Normalize((0.2815, 0.2815, 0.2815),
    #                                  (0.3206, 0.3206, 0.3206) )

    normalize = transforms.Normalize((0.2819, 0.2819, 0.2819),
                                     (0.3210, 0.3210, 0.3210) )

    train_transform = transforms.Compose(
        [
            transforms.Resize([64,64]),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            normalize
        ])
    val_test_transform = transforms.Compose([
        transforms.Resize([64,64]),
        transforms.ToTensor(),
        normalize
    ])

    #data = datasets.ImageFolder(root=image_path, transform=transform)
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

    # x = torch.stack([sample[0] for sample in Subset(data, train_indices)])
    # #get the mean of each channel
    # mean = torch.mean(x, dim=(0,2,3))
    # std = torch.std(x, dim=(0,2,3))
    # print(f"mean : {mean}")
    # print(f"std : {std}")

    return input_size, num_classes, train_dataset, val_dataset, test_dataset


all_datasets = {
    "SVHN": get_SVHN,
    "CIFAR10": get_CIFAR10,
    "CIFAR100": get_CIFAR100,
    "Brain_tumors": get_Brain_tumors,
    "Alzheimer": get_Alzheimer
}

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


