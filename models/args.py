from typing import NamedTuple

from torch import optim

class Args(NamedTuple):
    clip: float = 1.
    parameters: [torch.Tensor]
    optim: optim.Optimizer


