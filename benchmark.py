from time import perf_counter
from typing import Callable

import torch
from torch.profiler import profile, schedule, ProfilerActivity


def run(initializer: Callable[[str], tuple[Callable, tuple]]) -> None:
    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=100)
        print(f"Profiling summary after step '{p.step_num}':\n{output}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using '{device}'\n=============================================")

    model, args = initializer(device)

    print("\n\nBENCHMARKING\n=============================================")

    n = 1000
    total_run_time = 0
    for i in range(n):
        start_time = perf_counter()
        model(*args)
        run_time = perf_counter() - start_time
        total_run_time += run_time
        print(f"Iteration: '{i}', time spent: '{run_time}'")

    print(f"Total time spent in '{n}' iterations: '{total_run_time}'")

    print("\n\nPROFILING\n=============================================")

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
    ) as prof:
        for _ in range(1 + 1 + 3):
            model(*args)
            prof.step()
