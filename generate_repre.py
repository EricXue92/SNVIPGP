import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from due.convnext import ConvNextTinyGP
from PIL import Image
import clip
import torchvision
from torchvision.transforms import v2


IMAGENET_CONVNEXT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_CONVNEXT_STD = [0.229, 0.224, 0.225]

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

class Brain_tumors:
    dataset_name = "Brain_tumors"
    image_path = "./data/Brain_tumors"

class Alzheimer:
    dataset_name = "Alzheimer"
    image_path = "./data/Alzheimer"

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    # return the image path along with the image and label
    def __getitem__(self, index):
        # Call the superclass's __getitem__ method to get the image and label
        original_tuple = super().__getitem__(index)
        # Get the image path from the 'imgs' attribute
        path = self.imgs[index][0]
        # Return the original tuple plus the image path
        return original_tuple + (path,)

def retrieve_model(model_name):
    if model_name == "convnext":
        model = ConvNextTinyGP()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-L/14@336px", device=device)
    return model

def save_features(model, dataloader, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for inputs, labels, paths in tqdm(dataloader):
            inputs = inputs.cuda()
            if hasattr(model, "encode_image"):
                features = model.encode_image(inputs)
            else:
                features = model(inputs)
            for feature, path in zip(features, paths):
                class_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(path)))
                os.makedirs(class_dir, exist_ok=True)
                # os.path.dirname() function takes a full file path as input and returns the directory name of that path.
                # ./data/Alzheimer/class1/image1.jpg -> ./data/Alzheimer/class1

                # os.path.basename(...) returns the last component of the path
                # ./data/Alzheimer/class1 -> class1
                feature_path = os.path.join(class_dir, os.path.basename(path).replace('.jpg', '.pt'))
                torch.save(feature.cpu(), feature_path)

if __name__ == "__main__":
    config = Brain_tumors#  Brain_tumors, Alzheimer
    model_name = "convnext"
    data = ImageFolderWithPaths(root=config.image_path, transform=get_transform(model_name=model_name))
    dataloader = DataLoader(data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model = retrieve_model(model_name=model_name)
    model.cuda()
    output_dir = f"./data_feature/{config.dataset_name}"
    save_features(model, dataloader, output_dir)

