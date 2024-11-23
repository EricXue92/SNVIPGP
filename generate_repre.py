import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
import os
from due.convnext import ConvNextGP
import torchvision
from torchvision.transforms import v2
from sngp_wrapper.covert_utils import convert_to_sn_my
import clip


IMAGENET_CONVNEXT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_CONVNEXT_STD = [0.229, 0.224, 0.225]

# TRANSFORMS = transforms.Compose([
#     transforms.Resize(236, interpolation=transforms.InterpolationMode.BILINEAR),  # Resize to 236
#     transforms.CenterCrop(224),  # Center crop to 224
#     transforms.ToTensor(),  # Convert to tensor
#     transforms.Normalize(mean=IMAGENET_CONVNEXT_MEAN, std=IMAGENET_CONVNEXT_STD)  # Normalize
# ])
TRANSFORMS = transforms.Compose([
    transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),  # Resize to 232x232
    transforms.CenterCrop(224),  # Central crop to 224x224
    transforms.ToTensor(),  # Convert image to tensor and rescale to [0.0, 1.0]
    transforms.Normalize(mean=IMAGENET_CONVNEXT_MEAN, std=IMAGENET_CONVNEXT_STD)  # Normalize with given mean and std
])

# Dataset classes
class DatasetConfig:
    def __init__(self, dataset_name, image_path):
        self.dataset_name = dataset_name
        self.image_path = image_path

# Dataset configurations
datasets_config ={
        "Brain_tumors": DatasetConfig("Brain_tumors", "./data/Brain_tumors"),
        "Alzheimer": DatasetConfig("Alzheimer", "./data/Alzheimer"),
        "CIFAR10": DatasetConfig("CIFAR10", "data/CIFAR10"),
        "SVHN": DatasetConfig("SVHN", "data/SVHN")
}

def get_cifar10_dataset():
    return {
        datasets.CIFAR10("data/CIFAR10", train=True, transform=TRANSFORMS, download=False),
        datasets.CIFAR10("data/CIFAR10", train=False, transform=TRANSFORMS, download=False)
    }

def get_svhm_dataset():
    return (
        datasets.SVHN("data/SVHN", split="train", transform=TRANSFORMS, download=False),
        datasets.SVHN("data/SVHN", split="test", transform=TRANSFORMS, download=False)
    )

def get_transform(model_name: str):
    if model_name == "convnext":
        transform  = v2.Compose([
            v2.Resize(236, interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(384, interpolation=v2.InterpolationMode.BILINEAR),
            v2.Normalize(IMAGENET_CONVNEXT_MEAN, IMAGENET_CONVNEXT_STD),
        ])
    else:
        _, transform = clip.load("ViT-L/14@336px", device=torch.device("cuda"))
    return transform

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path

def retrieve_model(model_name):
    # spec_norm_replace_list = ["Linear", "Conv2D"]
    # coeff = 0.95
    if model_name == "convnext":
        model = ConvNextGP(num_classes=None).cuda()
    else:
        model, _ = clip.load("ViT-L/14@336px", device=torch.device("cuda"))
    # Constraint SN
    # model = convert_to_sn_my(model, spec_norm_replace_list, coeff)
    return model

def save_features(model, dataloader, output_dir, dataset_name):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        if dataset_name == "Brain_tumors" or dataset_name == "Alzheimer":
            for inputs, labels, paths in tqdm(dataloader):
                inputs = inputs.cuda()
                # hasattr(object, attribute) : returns True if the object has the given attribute, otherwise False
                if hasattr(model, "encode_image"):
                    features = model.encode_image(inputs)
                else:
                    features = model(inputs)

                for feature, path in zip(features, paths):
                    # os.path.dirname(...) returns the directory name of the path
                    # os.path.basename(...) returns the last component of the path
                    class_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(path)))
                    os.makedirs(class_dir, exist_ok=True)
                    feature_path = os.path.join(class_dir, os.path.basename(path).replace('.jpg', '.pt'))
                    torch.save(feature.cpu(), feature_path)

        elif dataset_name == "CIFAR10" or dataset_name == "SVHN":
            representations, labels = [], []
            for index, (input, label) in enumerate(tqdm(dataloader)):
                input = input.cuda()
                if hasattr(model, "encode_image"):
                    features = model.encode_image(input)
                else:
                    features = model(input)
                representations.append(features.cpu())
                labels.append(label.cpu())

            representations = torch.concat(representations, dim=0)
            labels = torch.concat(labels, dim=0)
            # Save features and labels
            torch.save({"features": representations, "labels": labels}, os.path.join(output_dir, f"{dataset_name}.pt"))
        else:
            raise ValueError("Unknown dataset")

# For CIFAR10 and SVHN with train and test datasets
def process_dataset(get_dataset_func, model, output_dir, dataset_name):
    train_dataset, test_dataset = get_dataset_func()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    save_features(model, train_loader, os.path.join(output_dir, "train"), dataset_name)
    save_features(model, test_loader, os.path.join(output_dir, "test"), dataset_name)

# if __name__ == "__main__":
#
#     config = datasets_config["SVHN"]  # Use "CIFAR10", "SVHN", "Brain_tumors", or "Alzheimer"
#     model_name = "convnext"
#     model = retrieve_model(model_name=model_name)
#
#     output_dir = f"./data_feature/{config.dataset_name}"
#
#     if config.dataset_name == "Brain_tumors" or config.dataset_name == "Alzheimer":
#         data = ImageFolderWithPaths(root=config.image_path, transform=get_transform(model_name=model_name))
#         dataloader = DataLoader(data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
#         save_features(model, dataloader, output_dir, dataset_name=config.dataset_name)
#
#     elif config.dataset_name == "CIFAR10":
#         process_dataset(get_cifar10_dataset, model, output_dir, config.dataset_name)
#
#     elif config.dataset_name == "SVHN":
#         process_dataset(get_svhm_dataset, model, output_dir, config.dataset_name)
#
#     else:
#         raise ValueError("Unknown dataset")




