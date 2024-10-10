import torch
from torch import nn
from torch.profiler import profile, schedule, ProfilerActivity


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=100)
    print(f"Profiling summary after step '{p.step_num}':")
    print(output)
    p.export_chrome_trace(f"trace_linear_after_step_{p.step_num}.json.gz")


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using '{device}'")

model = nn.Linear(28 * 28, 512).to(device)
x = torch.rand(1, 28 * 28, device=device)

with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for _ in range(1 + 1 + 3):
        model(x)
        prof.step()
