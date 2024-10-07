import os
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset, random_split
from PIL import Image

# National Cancer Institute -- Cancer Imaging Program (CIP)
# https://www.cancerimagingarchive.net/browse-collections/
NUM_WORKERS = os.cpu_count()

def get_SVHN():
    input_size = 32
    num_classes = 10
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.SVHN(
        "data/SVHN", split="train", transform=transform, download=False,
    )
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    test_dataset = datasets.SVHN(
        "data/SVHN", split="test", transform=transform, download=False,
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
        "data/CIFAR10", train=False, transform=test_transform, download=False,
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
            data = self.transform(data)
        return data, target


# https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html
# 3 * 256 * 256
def get_Brain_tumors():
    image_path = "./data/Brain_tumors"
    normalize = transforms.Normalize((0.2069, 0.2069, 0.2069),
                                     (0.1891, 0.1891, 0.1891))

    train_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        normalize
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        normalize
    ])
    data = datasets.ImageFolder(root=image_path)
    input_size = 64
    num_classes = 4

    # 2476 309 311
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = len(data) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        data, [train_size, val_size, test_size]
    )

    # Apply transformations
    train_dataset = TransformedDataset(train_dataset, transform=train_transform)
    val_dataset = TransformedDataset(val_dataset, transform=val_test_transform)
    test_dataset = TransformedDataset(test_dataset, transform=val_test_transform)

    # x = torch.stack( [ sample[0] for sample in ConcatDataset([ train_dataset ]) ] )
    # #get the mean of each channel
    # mean = torch.mean(x, dim=(0,2,3)) #tensor([0.2069, 0.2069, 0.2069])
    # std = torch.std(x, dim=(0,2,3)) #tensor([0.1891, 0.1891, 0.1891])
    return input_size, num_classes, train_dataset, val_dataset, test_dataset

# 3 * 208 * 176
def get_Alzheimer():
    image_path = "./data/Alzheimer"
    normalize = transforms.Normalize((0.2815, 0.2815, 0.2815),
                                     (0.3207, 0.3207, 0.3207)),
    train_transform = transforms.Compose(
        [
            transforms.Resize(size=(64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize
        ])
    val_test_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        normalize
    ])
    data = datasets.ImageFolder(root=image_path)
    input_size = 64
    num_classes = 4
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = len(data) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data,
                                                                             [train_size, val_size, test_size])

    train_dataset = TransformedDataset(train_dataset, transform=train_transform)
    val_dataset = TransformedDataset(val_dataset, transform=val_test_transform)
    test_dataset = TransformedDataset(test_dataset, transform=val_test_transform)

    # x = torch.stack( [ sample[0] for sample in ConcatDataset([ train_dataset ]) ] )
    # #get the mean of each channel
    # mean = torch.mean(x, dim=(0,2,3)) #tensor([0.2815, 0.2815, 0.2815])
    # std = torch.std(x, dim=(0,2,3)) #tensor([0.3207, 0.3207, 0.3207])

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


def get_dataloaders(dataset, train_batch_size=32):  # 128
    ds = all_datasets[dataset]()
    input_size, num_classes, train_dataset, val_dataset, test_dataset = ds

    kwargs = {"num_workers": NUM_WORKERS, "pin_memory": True}
    train_loader = data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs
    )

    val_loader = data.DataLoader(
        val_dataset, batch_size=train_batch_size, shuffle=True, **kwargs
    )

    test_loader = data.DataLoader(
        test_dataset, batch_size= 32, shuffle=False, **kwargs  # 1000
    )

    return train_loader, val_loader, test_loader, input_size, num_classes


