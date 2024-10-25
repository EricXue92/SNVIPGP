import torch.nn as nn
import torchvision
import torch.nn.functional as F

class ConvNextTinyGP(nn.Module):
    def __init__(self):
        super(ConvNextTinyGP, self).__init__()
        feature_extractor = torchvision.models.convnext_tiny(weights="ConvNeXt_Tiny_Weights.IMAGENET1K_V1")
        self.feature_extractor = feature_extractor
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        # feature_extractor.classifier = nn.Identity()
        # self.num_classes = num_classes
        # if self.num_classes:
        #     self.classifier = nn.Linear(768, num_classes) # please determine 768 by the classifier/head of the model

    def forward(self, x):
        features = self.flatten(self.feature_extractor(x))
        return features
        # if self.num_classes is None:
        #     return features
        # else:
        #     out = self.classifier(features, **kwargs)
        #     if isinstance(out, tuple):
        #         logits, uncertainty = out
        #         prob = F.log_softmax(logits, dim=1)
        #         return prob

class SimpleMLP(nn.Module):
    def __init__(self, num_classes: int):
        super(SimpleMLP, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.prelu = nn.PReLU()
        if self.num_classes is not None:
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, kwargs={}):
        x = x.view(x.size(0), -1)
        x = self.prelu(self.fc1(x))
        x = self.prelu(self.fc2(x))
        if self.num_classes is not None:
            x = self.classifier(x, **kwargs)
        return x


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 24, 128)
        self.prelu = nn.PReLU()
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, kwargs={}):
        x = x.view(x.size(0), 1, 32, 24)  # Reshape to (batch_size, 1, 32, 24)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.prelu(self.fc1(x))
        if self.num_classes is not None:
            x = self.classifier(x, **kwargs)
        return x