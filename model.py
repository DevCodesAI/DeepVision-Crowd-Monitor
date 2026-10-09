"""CSRNet-style inference network; weights must be trained separately."""
import torch
from torch import nn
from torchvision.models import vgg16

class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = vgg16(weights=None).features
        self.frontend = nn.Sequential(*list(backbone.children())[:23])
        layers = []
        in_channels = 512
        for out_channels, dilation in zip([512, 512, 512, 256, 128, 64], [2]*6):
            layers.extend([nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation), nn.ReLU(inplace=True)])
            in_channels = out_channels
        self.backend = nn.Sequential(*layers)
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        return self.output_layer(self.backend(self.frontend(x)))
