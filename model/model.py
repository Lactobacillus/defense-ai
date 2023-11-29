import os
import sys
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import module as md


class TempModel(nn.Module):

    def __init__(self,
            args: Dict[str, Any]) -> None:

        super(SingleImageModel, self).__init__()

        raise NotImplementedError

    def forward(self,
            x: torch.Tensor) -> torch.Tensor:

        out = x

        return out
