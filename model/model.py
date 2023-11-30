import os
import sys
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision.models import ResNet, resnet50, ResNet50_Weights
# torchvision: PyTorch의 컴퓨터 비전 모델 라이브러리
# ResNet : 깊은 신경망을 효과적으로 훈련하기 위한 모델, 잔차 학습
from model import module as md


class CustomResNet50(nn.Module):

    def __init__(self,
            pool: bool = False) -> None:
            # 평균 풀링 레이어를 포함할지 
            
        super(CustomResNet50, self).__init__()

        self.pool = pool
        self.resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)

    def forward(self,
            x: torch.Tensor) -> torch.Tensor:

        x = self._forward(x)

        return x

    def _forward(self,
            x: torch.Tensor) -> torch.Tensor:   # 특징 추출

        x = self.resnet.conv1(x)   # Convolution
        x = self.resnet.bn1(x)   # Batch Normalization
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


class Aggregator(nn.Module):

    def __init__(self,
            pool: bool = False) -> None:

        super(Aggregator, self).__init__()

        self.linear = nn.Linear(2048, 64)   # linear layer   
        self.layer1 = md.ResBlock3d(64, 64)     #  3D Residual Block
        self.layer2 = md.ResBlock3d(64, 32)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     # 공간차원은 1x1로 만드는 평균 풀링 레이어
        self.fc = nn.Linear(512, 1)

    def forward(self,
            x: torch.Tensor) -> torch.Tensor:

        x = torch.permute(x, (0, 1, 3, 4, 2))   # 입력데이터 형태 조정
        x = F.leaky_relu(self.linear(x)) # (bs, fl, w, h, d)
        x = torch.permute(x, (0, 4, 1, 2, 3)) # (bs, d, fl, w, h)
        x = self.layer1(x)
        x = self.layer2(x)
        bs, ch, fl, w, h = x.size()
        x = x.contiguous().view(bs, ch * fl, w, h)  # 텐서모양 변경 - 채널*특징, 연속적으로 저장된 형태
        x = self.avgpool(x)
        x = torch.flatten(x, 1)     
        x = self.fc(x)

        return x

    def reset_fc(self) -> None:

        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
