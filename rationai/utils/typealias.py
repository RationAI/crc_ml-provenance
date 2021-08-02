"""
Type aliases for complex types.
"""
from typing import Callable, Iterator

import torch

TorchOptimGenerator = Callable[[Iterator[torch.Tensor]], torch.optim.Optimizer]

TorchRegularizer = Callable[[torch.Tensor], torch.Tensor]
