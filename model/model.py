import os
import sys
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet, resnet50, ResNet50_Weights

from model import module as md


class CustomResNet50(nn.Module):

    def __init__(self,
            pool: bool = False) -> None:

        super(CustomResNet50, self).__init__()

        self.pool = pool
        self.resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)

    def forward(self,
            x: torch.Tensor) -> torch.Tensor:

        x = self._forward(x)

        return x

    def _forward(self,
            x: torch.Tensor) -> torch.Tensor:

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        if self.pool:

            x = self.avgpool(x)
            x = torch.flatten(x, 1)

        return x
