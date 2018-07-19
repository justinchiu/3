from typing import NamedTuple

from torch import optim

class Args(NamedTuple):
    optimizer: optim.Optimizer
    clip: float = 1.
    report_interval: int = 100

