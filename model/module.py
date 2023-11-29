import os
import sys
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):

    def __init__(self,
            in_dim: int,
            out_dim: int,
            act: str = 'tanh') -> None:

        super(LinearLayer, self).__init__()

        assert act in ['relu', 'tanh', 'gelu', 'sigmoid', 'none', 'linear', 'sine', 'norm', 'exp'], \
            '[error] Wrong activation function! Received: {}'.format(act)

        self.act = act
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self,
            x: torch.Tensor) -> torch.Tensor:

        out = x
        out = self.linear(x)

        match self.act:

            case 'relu': out = F.relu(out)
            case 'tanh': out = torch.tanh(out)
            case 'gelu': out = F.gelu(out)
            case 'sigmoid': out = F.sigmoid(out)
            case 'sine': out = torch.sin(out)
            case 'norm': out = F.normalize(out, p = 2, dim = -1)
            case 'exp': out = torch.exp(out)

        return out


# original implementation is available at https://github.com/krrish94/nerf-pytorch/blob/a14357da6cada433d28bf11a45c7bcaace76c06e/nerf/nerf_helpers.py
# note: do not rollback, this implementation is faster than the original
class PosEncoding(nn.Module):

    def __init__(self,
            enc_dim: int = 6,
            include_input: bool = True,
            log_sampling: bool = True) -> None:

        super(PosEncoding, self).__init__()

        self.enc_dim = enc_dim
        self.include_input = include_input
        self.log_sampling = log_sampling

        if log_sampling:

            self.freq_bands = nn.Parameter(2.0 ** torch.linspace(0, enc_dim - 1, enc_dim), requires_grad = False)

        else:

            self.freq_bands = nn.Parameter(torch.linspace(1, 2 ** (enc_dim - 1), enc_dim), requires_grad = False)

    def forward(self,
            x: torch.Tensor) -> torch.Tensor:

        x = x.view(x.size(0), -1)
        
        out_dim = x.size(-1) if self.include_input else 0
        out_dim = out_dim + x.size(-1) * 2 * self.enc_dim
        
        out = torch.zeros(x.size(0), out_dim, device = x.device)
        sin = torch.sin(x.unsqueeze(-1) * self.freq_bands).view(x.size(0), -1)
        cos = torch.cos(x.unsqueeze(-1) * self.freq_bands).view(x.size(0), -1)

        out[:, :sin.size(-1)] = sin
        out[:, sin.size(-1):sin.size(-1) * 2] = cos
        out[:, -x.size(-1):] = x

        return out


class MLP(nn.Module):

    def __init__(self,
            in_dim: int,
            h_dim: int,
            out_dim: int,
            num_h_layer: int = 4,
            h_act: str = 'tanh',
            out_act: str = 'sigmoid') -> None:

        super(MLP, self).__init__()

        self.in_layer = LinearLayer(in_dim, h_dim, h_act)
        self.h_layer = nn.ModuleList([LinearLayer(h_dim, h_dim, h_act) for _ in range(num_h_layer)])
        self.out_layer = LinearLayer(h_dim, out_dim, out_act)

    def forward(self,
            x: torch.Tensor) -> torch.Tensor:

        out = x
        out = self.in_layer(out)

        for idx, m_layer in enumerate(self.h_layer):

            out = m_layer(out)

        out = self.out_layer(out)

        return out


class ResBlock3d(nn.Module):

    def __init__(self,
            in_ch: int,
            out_ch: int) -> None:

        super(ResBlock3d, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv1 = nn.Conv3d(in_ch, in_ch, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(in_ch)
        self.conv2 = nn.Conv3d(in_ch, in_ch, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(in_ch)

        if in_ch != out_ch:

            self.conv3 = nn.conv1d(in_ch, out_ch, kernel_size = 1, padding = 0)

    def forward(self,
            x: torch.Tensor) -> torch.Tensor:

        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = F.relu(out)

        if self.in_ch != self.out_ch:

            out = self.conv3(out)

        return out
