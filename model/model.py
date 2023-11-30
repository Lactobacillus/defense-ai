import os
import sys
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.init as init
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


# class Aggregator(nn.Module):

#     def __init__(self,
#             pool: bool = False) -> None:

#         super(Aggregator, self).__init__()

#         self.linear = nn.Linear(2048, 64)
#         self.layer1 = md.ResBlock3d(64, 64)
#         self.layer2 = md.ResBlock3d(64, 32)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(256, 1)

#     def forward(self,
#             x: torch.Tensor) -> torch.Tensor:

#         x = torch.permute(x, (0, 1, 3, 4, 2))
#         x = F.leaky_relu(self.linear(x)) # (bs, fl, w, h, d)
#         x = torch.permute(x, (0, 4, 1, 2, 3)) # (bs, d, fl, w, h)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         bs, ch, fl, w, h = x.size()
#         x = x.contiguous().view(bs, ch * fl, w, h)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x

#     def reset_fc(self) -> None:

#         init.xavier_uniform_(self.fc.weight)
#         init.zeros_(self.fc.bias)


class Aggregator(nn.Module):

    def __init__(self,
            pool: bool = False) -> None:

        super(Aggregator, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1)

    def forward(self,
            x: torch.Tensor) -> torch.Tensor:

        x = x.max(1) # (bs, w, h, d)
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def reset_fc(self) -> None:

        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
