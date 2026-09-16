# LoRA and the memory bill, in arithmetic — and why a low-rank update is enough.
#
# The model is illustrative, not any particular one: a dense transformer of about eight
# billion parameters, 32 layers wide 4,096, with LoRA on the four attention projections.

import numpy as np

PARAMS, LAYERS, WIDTH = 8e9, 32, 4_096
GB = 1e9

# ------------------------------------------------------------------ 1. parameters
print("1. What LoRA trains: W stays frozen, and the update is B x A, of rank r\n")
print(f"  a {WIDTH:,} x {WIDTH:,} projection has {WIDTH * WIDTH:,} weights;")
print(f"  B ({WIDTH:,} x r) and A (r x {WIDTH:,}) have 2 x {WIDTH:,} x r between them\n")
print(f"  {'rank':>5} {'trained parameters':>19} {'share of model':>15} "
      f"{'adapter, 16-bit':>16}")
for rank in (4, 8, 16, 64):
    trained = LAYERS * 4 * 2 * WIDTH * rank
    print(f"  {rank:>5} {trained:>19,} {trained / PARAMS:>15.3%} "
          f"{trained * 2 / 1e6:>13.0f} MB")

# ------------------------------------------------------------------ 2. memory
print("\n2. Memory to train, before activations\n")
# Mixed-precision training with Adam keeps, per trained parameter: 16-bit weights and
# gradients (2 + 2 bytes), and 32-bit master weights and two optimiser moments (4 + 8).
FULL_BYTES = 2 + 2 + 4 + 8
lora_trained = LAYERS * 4 * 2 * WIDTH * 16
rows = [
    ("full fine-tune", PARAMS * FULL_BYTES),
    ("LoRA, rank 16, base in 16-bit", PARAMS * 2 + lora_trained * FULL_BYTES),
    ("LoRA, rank 16, base in 4-bit", PARAMS * 0.5 * 1.06 + lora_trained * FULL_BYTES),
]
for name, total in rows:
    print(f"  {name:<32} {total / GB:>6.1f} GB")
print("  (the 4-bit row adds 6% for quantisation scales; activations come on top of all)")

# ------------------------------------------------------------------ 3. why low rank works
print("\n3. How much of an update a rank-r approximation keeps\n")
rng = np.random.default_rng(30)
n, true_rank = 512, 8
signal = rng.normal(size=(n, true_rank)) @ rng.normal(size=(true_rank, n))
noise = rng.normal(size=(n, n)) * np.linalg.norm(signal) / n / 3
for name, update in (("an update that is mostly low-rank", signal + noise),
                     ("an update with no structure", rng.normal(size=(n, n)))):
    singular = np.linalg.svd(update, compute_uv=False)
    energy = np.cumsum(singular ** 2) / np.sum(singular ** 2)
    kept = "  ".join(f"r={r}: {energy[r - 1]:.0%}" for r in (1, 8, 32, 128))
    print(f"  {name:<34} {kept}")

print("\nLoRA's bet is that the change fine-tuning needs looks like the first row: a few")
print("directions carry nearly all of it. When the change does not — a new language, a")
print("large body of new knowledge — a small rank is more likely to fall short, which is")
print("one more reason facts belong in retrieval.")
