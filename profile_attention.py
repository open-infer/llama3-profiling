import torch
from torch.profiler import profile, schedule, ProfilerActivity

from llama.model import Attention, ModelArgs


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=100)
    print(f"Profiling summary after step '{p.step_num}':")
    print(output)
    p.export_chrome_trace(f"trace_attention_after_step_{p.step_num}.json.gz")


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using '{device}'")

# {
#   "dim": 2048,
#   "ffn_dim_multiplier": 1.5,
#   "multiple_of": 256,
#   "n_heads": 32,
#   "n_kv_heads": 8,
#   "n_layers": 16,
#   "norm_eps": 1e-05,
#   "rope_theta": 500000.0,
#   "vocab_size": 128256
# }

# torch.Size([1, 2, 2048])
# torch.Size([2, 32])
# torch.Size([2, 2])


# ModelArgs below replicate conditions of the profiling run described here:
# https://docs.google.com/document/d/1r3rE2eYpIueytuebXonuc5J2fDIHlGG142sy8hY7LSc/edit
model_args = ModelArgs(n_heads=32, n_kv_heads=8, dim=2048, max_batch_size=1, max_seq_len=128)
model = Attention(model_args).to(device)

x = torch.rand(1, 2, 2048, device=device)
freqs_cis = torch.rand(2, 32, device=device)
mask = torch.rand(2, 2, device=device)

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
        model(x=x, start_pos=0, freqs_cis=freqs_cis, mask=mask)
        prof.step()
