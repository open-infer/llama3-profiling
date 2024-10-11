from typing import Callable

import torch
from torch import nn

from benchmark import run


def init(device: str) -> tuple[Callable, tuple]:
    return nn.Linear(2048, 2048).to(device), (torch.rand(1, 2048, device=device),)


run(init)
