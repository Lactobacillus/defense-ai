import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable


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
