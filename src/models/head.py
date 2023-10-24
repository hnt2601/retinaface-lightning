import torch
from torch import nn


class ClassHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, num_anchors * 2, kernel_size=(1, 1), stride=(1, 1), padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=(1, 1), stride=(1, 1), padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)
