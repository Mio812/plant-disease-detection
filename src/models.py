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


def build_model(name, num_classes, pretrained=True):
    name = name.lower()
    if name == "custom_cnn":
        return CustomCNN(num_classes)
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {name!r}")
