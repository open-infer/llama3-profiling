from typing import Callable

import torch

from benchmark import run
from llama.model import Attention, ModelArgs


def init(device: str) -> tuple[Callable, tuple]:
    # ModelArgs below replicate conditions of the profiling run described here:
    # https://docs.google.com/document/d/1r3rE2eYpIueytuebXonuc5J2fDIHlGG142sy8hY7LSc/edit
    model_args = ModelArgs(n_heads=32, n_kv_heads=8, dim=2048, max_batch_size=1, max_seq_len=128)
    model = Attention(model_args).to(device)

    return (
        model,
        (torch.rand(1, 2, 2048, device=device), 0, torch.rand(2, 32, device=device), torch.rand(2, 2, device=device))
    )


run(init)
