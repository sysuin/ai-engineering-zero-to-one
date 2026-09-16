# One real next-token distribution, and what each sampling strategy does to it.

import math

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
response = client.chat.completions.create(
    model=MODEL_FAST, temperature=1, logprobs=True, top_logprobs=5, max_completion_tokens=10,
    messages=[{"role": "user", "content":
               "One word, lowercase, describing a supplier who is always late:"}])
top = response.choices[0].logprobs.content[0].top_logprobs
logp = {t.token: t.logprob for t in top}
covered = sum(math.exp(v) for v in logp.values())
print(f"the model's top {len(logp)} candidates cover {covered:.3f} of the probability\n")


def reshape(logp: dict, T: float) -> dict:
    """Temperature on log-probabilities: divide, then renormalise (softmax)."""
    scaled = {k: v / T for k, v in logp.items()}
    peak = max(scaled.values())
    z = sum(math.exp(v - peak) for v in scaled.values())
    return {k: math.exp(v - peak) / z for k, v in scaled.items()}


def entropy_bits(p: dict) -> float:
    return -sum(q * math.log2(q) for q in p.values() if q > 0)


def nucleus(p: dict, top_p: float) -> int:
    total, n = 0.0, 0
    for q in sorted(p.values(), reverse=True):
        total += q
        n += 1
        if total >= top_p:
            break
    return n


base = reshape(logp, 1.0)
ranked = sorted(base.items(), key=lambda kv: -kv[1])
print("at T=1, renormalised over the top candidates:")
for token, q in ranked[:6]:
    print(f"  {token!r:14} {q:.3f}  {'█' * round(q * 40)}")

print(f"\n  {'T':>4} {'top token':>9} {'entropy':>8} {'top-p 0.9 keeps':>16}")
for T in (0.3, 0.7, 1.0, 1.5, 2.5):
    p = reshape(logp, T)
    print(f"  {T:>4} {max(p.values()):>9.3f} {entropy_bits(p):>6.2f} b {nucleus(p, 0.9):>12} tokens")

print(f"\ntop-k with k=3 keeps {sum(q for _, q in ranked[:3]):.3f} of the mass at T=1, "
      f"whatever the shape")
