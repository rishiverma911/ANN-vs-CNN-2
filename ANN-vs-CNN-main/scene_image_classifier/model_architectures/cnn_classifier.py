"""Convolutional model used for scene classification."""

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """Extract visual features with convolutions before classification."""

    def __init__(
        self,
        num_classes: int = 6,
        input_channels: int = 3,
        image_size: int = 150,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.image_size = image_size

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.feature_map_size = image_size // 8
        self.flatten_size = 128 * self.feature_map_size * self.feature_map_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.input_channels, self.image_size, self.image_size)

    @property
    def output_shape(self) -> tuple[int]:
        return (self.num_classes,)
