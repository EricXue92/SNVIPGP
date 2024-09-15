import os 
import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

# For repeated experiment 
seed = 23
np.random.seed(seed)
torch.manual_seed(seed)
''' 
# return input_size, num_classes, train_dataset, test_dataset
# SVHM(Street view house numbers) 10 classes (0-9), 32 * 32
def get_SVHN(root):
    input_size = 32
    num_classes = 10
    # NOTE: these are not correct mean and std for SVHN, but are commonly used
    transform = transforms.Compose(
        [ transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # "./" + "/SVHN" 在当前.py所在的路径 建立文件夹SVHM
    # 如果 root = "data"， 则在当前.py所在路径 建立data文件夹，然后建立SVHN文件夹 （"data/SVHN"）
    
    train_dataset = datasets.SVHN(
        root + "/SVHN", split="train", transform=transform, download=True
    )
    test_dataset = datasets.SVHN(
        root + "/SVHN", split="test", transform=transform, download=True
    )
    return input_size, num_classes, train_dataset, test_dataset
# CIFAR(Canadian Institute for Advanced Research) 加拿大高等研究院 10类 (6个动物 + 4 个交通工具) 
# （50000 training, 10000 test) 
#  32*32 

def get_CIFAR10(root):
    input_size = 32
    num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # Alternative
    # normalize = transforms.Normalize(
    #     (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    # )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset
# 60000 = 100 个类 * 600 
# 32x32 colour images 
# 50000 training, 10000 test
def get_CIFAR100(root):
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
        root + "/CIFAR100", train=True, transform=train_transform, download=True
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR100(
        root + "/CIFAR100", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset
'''

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset  = dataset
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
def get_Brain_tumor():
    image_path = "./data/Brain_tumors"
    normalize = transforms.Normalize( (0.2114, 0.2114, 0.2114), 
                                    (0.1891, 0.1891, 0.1891) ) 
    train_transform = transforms.Compose([ 
                        transforms.Resize(size=(64, 64)),
                        transforms.RandomCrop(64, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize 
                        ] )
    val_test_transform = transforms.Compose([
                        transforms.Resize(size=(64, 64)),
                        transforms.ToTensor(), 
                        normalize
                        ])
    data = datasets.ImageFolder(root = image_path) 
    input_size = 64
    num_classes = 4
    # Split the dataset
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = len(data) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                                                data, [train_size, val_size, test_size]
                                                )
    
    # Apply transformations
    train_dataset = TransformedDataset(train_dataset, transform = train_transform)
    val_dataset = TransformedDataset(val_dataset, transform = val_test_transform )
    test_dataset = TransformedDataset(test_dataset, transform = val_test_transform)
    
    # x = torch.stack( [ sample[0] for sample in ConcatDataset([ train_dataset ]) ] )
    # #get the mean of each channel 
    # mean = torch.mean(x, dim=(0,2,3)) #tensor([0.2114, 0.2114, 0.2114]) 
    # std = torch.std(x, dim=(0,2,3)) #tensor([0.1891, 0.1891, 0.1891])
    print("train, val, test", len(train_dataset), len(val_dataset), len(test_dataset))
    return input_size, num_classes, train_dataset, val_dataset, test_dataset

# 3 * 208 * 176
def get_Alzheimer():
    image_path = "./data/Alzheimer"
    normalize = transforms.Normalize((0.2817, 0.2817, 0.2817), (0.2817, 0.2817, 0.2817)),  
    train_transform = transforms.Compose(
    [ 
        transforms.Resize(size=(64, 64) ),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ] )
    
    val_test_transform = transforms.Compose([
            transforms.Resize(size=(64, 64) ),
            transforms.ToTensor(), 
            normalize
            ])
    
    data = datasets.ImageFolder(root = image_path) 
    input_size = 64
    num_classes = 4
    
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = len(data) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, 
                                                            [ train_size, val_size, test_size])
    # Apply transformations
    train_dataset = TransformedDataset(train_dataset, transform = train_transform)
    val_dataset = TransformedDataset(val_dataset, transform = val_test_transform )
    test_dataset = TransformedDataset(test_dataset, transform = val_test_transform)
    return input_size, num_classes, train_dataset, val_dataset, test_dataset


# 字典 { 数据名称：下载该数据的函数}
all_datasets = {
    # "SVHN": get_SVHN,
    # "CIFAR10": get_CIFAR10,
    # "CIFAR100": get_CIFAR100,
    "Brain_tumors" :get_Brain_tumor,
    "Alzheimer": get_Alzheimer
}

# 生成数据的函数，默认root = "./"

# 输入:数据名称 （ "SVHN" or "CIFAR10" or "CIFAR100")
# return: 下载对应的数据
def get_dataset(dataset):  # root="./"
    return all_datasets[dataset]()

# return train_loader, test_loader, input_size, num_classes
# 输入: 数据名称（ "SVHN" or "CIFAR10" or "CIFAR100"), train_batch_size=128, root="./"
def get_dataloaders(dataset, train_batch_size = 32):
    ds = all_datasets[dataset]()
    input_size, num_classes, train_dataset, val_dataset, test_dataset = ds
    
    kwargs = {"num_workers": 4, "pin_memory": True}
    train_loader = data.DataLoader(
        train_dataset, batch_size = train_batch_size, shuffle = True, **kwargs
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size = train_batch_size, shuffle = True, drop_last=True, **kwargs
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size = 64, shuffle= True, **kwargs
    )
    return train_loader, val_loader, test_loader, input_size, num_classes


