"""Dense neural-network baseline for scene classification."""

import torch
import torch.nn as nn


class ANNClassifier(nn.Module):
    """Flatten an RGB image and classify it with fully connected layers."""

    def __init__(
        self,
        num_classes: int = 6,
        input_channels: int = 3,
        image_size: int = 150,
        hidden_sizes: tuple[int, ...] = (160, 32),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.image_size = image_size
        self.flatten_size = input_channels * image_size * image_size

        layers: list[nn.Module] = []
        in_features = self.flatten_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.input_channels, self.image_size, self.image_size)

    @property
    def output_shape(self) -> tuple[int]:
        return (self.num_classes,)
