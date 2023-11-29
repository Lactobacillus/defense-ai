import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable, Iterator, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader


class TempData(Dataset):

    def __init__(self,
            filename: str) -> None:

        pass

    def __len__(self) -> int:

        raise NotImplementedError

    def __getitem__(self,
            idx: int) -> None:

        raise NotImplementedError
