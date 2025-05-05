import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNextGP(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvNextGP, self).__init__()
        self.feature_extractor = torchvision.models.convnext_tiny(weights="ConvNeXt_Tiny_Weights.DEFAULT").to(device) # 768
        #self.feature_extractor = torchvision.models.convnext_base(weights="ConvNeXt_Base_Weights.DEFAULT").to(device) # 1024
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Replace the classifier with nn.Identity to keep the features unchanged
        self.feature_extractor.classifier = nn.Identity()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.num_classes = num_classes

        if self.num_classes is not None:
            self.classifier = nn.Linear(768, num_classes) # please determine 768 by the classifier/head of the model
        else:
            self.classifier = None

    def forward(self, x, **kwargs):
        features = self.feature_extractor(x)
        features = self.flatten(features)
        if self.classifier is None:
            return features

        logits = self.classifier(features)

        if isinstance(logits, tuple):
            logits, uncertainty = logits
            prob = F.log_softmax(logits, dim=1)
            return prob, uncertainty
        else:
            return F.log_softmax(logits, dim=1)

class EfficientNetGP(nn.Module):
    def __init__(self, num_classes: int):
        super(EfficientNetGP, self).__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        # auto_transform = weights.transforms()
        self.feature_extractor = torchvision.models.efficientnet_b0(weights=weights).to(device)
        self.flatten = nn.Flatten()
        # Replace the classifier with nn.Identity to keep the features unchanged
        self.feature_extractor.classifier = nn.Identity()

        self.num_classes = num_classes
        if self.num_classes is not None:
            self.classifier = nn.Linear(1280, num_classes) # please determine 1280 by the classifier/head of the model
        else:
            self.classifier = None

    def forward(self, x, **kwargs):
        features = self.feature_extractor(x)
        features = self.flatten(features)
        if self.classifier is None:
            return features
        logits = self.classifier(features)

        if isinstance(logits, tuple):
            logits, uncertainty = logits
            prob = F.log_softmax(logits, dim=1)
            return prob, uncertainty
        else:
            return F.log_softmax(logits, dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        # self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.PReLU()
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.fc(x)
        # out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        # Check dimensions match before addition
        assert out.shape == residual.shape, f"Shape mismatch: {out.shape} vs {residual.shape}"
        return out + residual

class SimpleResnet(nn.Module):
    def __init__(self, num_classes: int):
        super(SimpleResnet, self).__init__()
        self.num_classes = num_classes
        # Input layer
        # self.input_layer = nn.Linear(768, 256)
        self.input_layer = nn.Linear(1024, 512)
        self.input_activation = nn.PReLU()  # Add activation after input layer
        # Residual blocks
        self.res1 = ResidualBlock(512, 256)
        self.res2 = ResidualBlock(256, 128)
        #self.res3 = ResidualBlock(256, 128)
        # Classifier
        if self.num_classes is not None:
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, kwargs={}):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # Check input dimension
        assert x.size(1) == 1024, f"Expected input dimension 1024, got {x.size(1)}"

        # Input layer with activation
        x = self.input_layer(x)
        x = self.input_activation(x)
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        # x = self.res3(x)
        # Classification layer
        if self.num_classes is not None:
            x = self.classifier(x, **kwargs)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, num_classes: int):
    #     super(SimpleMLP, self).__init__()
    #     self.num_classes = num_classes
    #     self.fc1 = nn.Linear(768, 256)
    #     self.fc2 = nn.Linear(256, 128)
    #     # self.dropout = nn.Dropout(0.1)  # for cifar10
    #     self.dropout = nn.Dropout(0.1)
    #     self.prelu = nn.PReLU()
    #     if self.num_classes is not None:
    #         self.classifier = nn.Linear(128, num_classes)
    # def forward(self, x, kwargs={}):
    #     x = x.view(x.size(0), -1)
    #     x = self.prelu(self.fc1(x))
    #     x = self.dropout(x)
    #     x = self.prelu(self.fc2(x))
    #     x = self.dropout(x)
    #     if self.num_classes is not None:
    #         x = self.classifier(x, **kwargs)
    #     return x

        super(SimpleMLP, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(768, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        # self.dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.3) # for cifar10
        self.prelu = nn.PReLU()
        if self.num_classes is not None:
            self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, kwargs={}):
        x = x.view(x.size(0), -1)
        x = self.prelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.prelu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        if self.num_classes is not None:
            x = self.classifier(x, **kwargs)
        return x

        # For Breast sngp
        # super(SimpleMLP, self).__init__()
        # self.num_classes = num_classes
    #     self.fc1 = nn.Linear(768, 256)
    #     self.fc2 = nn.Linear(256, 128)
    #     self.bn1 = nn.BatchNorm1d(256)
    #     self.bn2 = nn.BatchNorm1d(128)
    #     self.dropout = nn.Dropout(0.1)
    #     self.relu = nn.ReLU()
    #     self.classifier = nn.Linear(128, num_classes)
    #
    # def forward(self, x, kwargs={}):
    #     x = x.view(x.size(0), -1)
    #     x = self.fc1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.dropout(x)
    #     x = self.fc2(x)
    #     x = self.bn2(x)
    #     x = self.relu(x)
    #     x = self.dropout(x)
    #     if self.num_classes is not None:
    #         x = self.classifier(x, **kwargs)
    #     return x





# class SimpleConvNet(nn.Module):
#     def __init__(self, num_classes: int):
#         super(SimpleConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64 * 32 * 24, 128)
#         self.prelu = nn.PReLU()
#         self.num_classes = num_classes
#         if self.num_classes is not None:
#             self.classifier = nn.Linear(128, num_classes)
#
#     def forward(self, x, kwargs={}):
#         x = x.view(x.size(0), 1, 32, 24)  # Reshape to (batch_size, 1, 32, 24)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = self.prelu(self.fc1(x))
#         if self.num_classes is not None:
#             x = self.classifier(x, **kwargs)
#         return x
#
#
