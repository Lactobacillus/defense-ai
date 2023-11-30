import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossAccumulator(object):

    def __init__(self) -> None:

        self.loss = defaultdict(list)
        self.num_samples = list()

    def add(self,
            b_loss: Dict[str, Any],
            b_size: int) -> None:

        for key in b_loss.keys(): self.loss[key].append(b_loss[key].detach().cpu() * b_size)
        self.num_samples.append(b_size)

    def get(self) -> Dict[str, Any]:

        accumulated = dict()

        for key in self.loss.keys(): accumulated[key] = sum(self.loss[key]) / sum(self.num_samples)

        return accumulated

class EMA(object):

    def __init__(self,
            model: nn.Module,
            decay: float) -> None:
        
        self.model = model
        self.decay = decay
        self.shadow = dict()
        self.backup = dict()

        self.register()

    def register(self) -> None:

        for name, param in self.model.named_parameters():

            if param.requires_grad:

                self.shadow[name] = param.data.clone()

    def update(self) -> None:

        for name, param in self.model.named_parameters():

            if param.requires_grad:

                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:

        for name, param in self.model.named_parameters():

            if param.requires_grad:

                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self) -> None:

        for name, param in self.model.named_parameters():

            if param.requires_grad:

                param.data = self.backup[name]

        self.backup = dict()

    def state_dict(self):

        return self.shadow

    def load_state_dict(self,
            state_dict: Dict[str, torch.Tensor]) -> None:

        self.shadow = state_dict
