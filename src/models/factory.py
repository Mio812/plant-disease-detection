"""Classification models: a from-scratch CNN and transfer-learning backbones."""

import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """Four-block convolutional network trained from scratch as our baseline."""

    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


HEADS = {"custom_cnn": "classifier", "resnet18": "fc", "resnet50": "fc",
         "mobilenet_v2": "classifier", "efficientnet_b0": "classifier"}


def freeze_backbone(model, name):
    """Train only the classification head, leaving the pretrained features untouched.

    BatchNorm running statistics are frozen as well: if they were left to re-estimate
    they would absorb PlantVillage's uniform-background statistics, which is exactly
    the domain information this arm is meant to exclude.
    """
    heads = (["crop_head", "disease_head"] if hasattr(model, "crop_head")
             else [HEADS[name.lower()]])
    for param in model.parameters():
        param.requires_grad = False
    for head in heads:
        for param in getattr(model, head).parameters():
            param.requires_grad = True

    original_train = model.train

    def train(mode=True):
        original_train(mode)
        if mode:
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return model

    model.train = train
    return model


def trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def build_model(name, num_classes, pretrained=True):
    name = name.lower()
    if name == "custom_cnn":
        return CustomCNN(num_classes)
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {name!r}")
