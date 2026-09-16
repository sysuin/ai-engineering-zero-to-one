# Model size, training compute and memory, from the shape of the network. The formulas are
# standard approximations; the two published configurations check them.

def parameters(layers, d_model, vocab, d_ff=None, gated=False, tied=True):
    d_ff = d_ff or 4 * d_model
    attention = 4 * d_model * d_model                  # query, key, value, output projections
    mlp = (3 if gated else 2) * d_model * d_ff        # two matrices, or three when gated
    embeddings = vocab * d_model * (1 if tied else 2)  # input table (and output, if separate)
    return layers * (attention + mlp) + embeddings


CONFIGS = [
    # name, layers, d_model, vocab, d_ff, gated, tied, published parameter count
    ("a 12-layer model, 2019",  12,  768, 50_257, None,   False, True,  124e6),
    ("a 32-layer model, 2023",  32, 4096, 32_000, 11_008, True,  False, 6.7e9),
]
print(f"  {'configuration':24} {'estimated':>11} {'published':>10}")
for name, L, d, v, ff, gated, tied, published in CONFIGS:
    n = parameters(L, d, v, ff, gated, tied)
    print(f"  {name:24} {n / 1e9:>9.3f} B {published / 1e9:>8.3f} B")

# Training compute: about 6 floating-point operations per parameter per training token
# (2 for the forward pass, 4 for the backward). A widely cited 2022 result suggests about
# 20 training tokens per parameter for the best model at a fixed compute budget.
N = 7e9
D = 20 * N
flops = 6 * N * D
print(f"\n7B parameters x {D / 1e9:,.0f}B tokens -> {flops:.1e} FLOPs to train")
gpu_flops_per_s = 1e14          # an assumption: one accelerator's useful throughput
print(f"at {gpu_flops_per_s:.0e} FLOPs/s per accelerator: "
      f"{flops / gpu_flops_per_s / 86_400 / 365:,.0f} accelerator-years")

# Inference: about 2 FLOPs per parameter per token generated.
print(f"generating one token with 7B parameters: about {2 * N:.1e} FLOPs")

# Memory to hold the weights, at different precisions.
print("\nmemory for the weights of a 70B-parameter model")
for bits in (32, 16, 8, 4):
    print(f"  {bits:>2}-bit   {70e9 * bits / 8 / 1e9:>6,.0f} GB")
