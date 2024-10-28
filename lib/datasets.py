import os
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from sklearn.model_selection import train_test_split
import glob

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

def get_tumors_feature(image_path: str = "./data_feature/Brain_tumors"):
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

def get_cifar10_or_svhm(image_path: str = "./data_feature/CIFAR10"):
    input_size, num_classes = 768, 10
    dataset_name = os.path.basename(image_path)

    train_feature_path = os.path.join(image_path, "train")
    test_feature_path = os.path.join(image_path, "test")

    train_feature_path = os.path.join(train_feature_path, f"{dataset_name}.pt")
    test_feature_path = os.path.join(test_feature_path, f"{dataset_name}.pt")

    if not (os.path.exists(train_feature_path) and os.path.exists(test_feature_path)):
        raise FileNotFoundError(f"One or both files do not exist: {train_feature_path}, {test_feature_path}.")

    train_data = torch.load(train_feature_path)
    train_representations, labels= train_data['features'], train_data['labels']

    X_train, X_val, y_train, y_val = train_test_split(train_representations, labels , test_size=0.1, random_state=42)
    train_dataset, val_dataset = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)

    test_data = torch.load(test_feature_path)
    test_representations, labels = test_data['features'], test_data['labels']
    test_dataset = TensorDataset(test_representations, labels)

    return input_size, num_classes, train_dataset, val_dataset, test_dataset

all_feature_datasets = {
    "Brain_tumors": lambda: get_tumors_feature(image_path="./data_feature/Brain_tumors"),
    "Alzheimer": lambda: get_tumors_feature(image_path="./data_feature/Alzheimer"),
    "CIFAR10": lambda: get_cifar10_or_svhm(image_path="./data_feature/CIFAR10"),
    "SVHN": lambda: get_cifar10_or_svhm(image_path="./data_feature/SVHN")
}

def get_feature_dataset(dataset):
    return all_feature_datasets[dataset]


# if __name__ == "__main__":
#     temp = get_feature_dataset("CIFAR10")()
#     input_size, num_classes, train_dataset, val_dataset, test_dataset = temp
#     print(train_dataset[8])

#     feauture_f = FeatureDataset('../data_feature/Brain_tumors')
#     print("len(feauture_f):", len(feauture_f))
#     print("feauture_f[0]:", feauture_f[0][0].shape, feauture_f[0][1])


