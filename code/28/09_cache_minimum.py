# timeout: 600
# The provider's prefix cache has a minimum length. The same request sent twice, at a range of
# prompt lengths: how much of the second one is served from the cache?

import uuid

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
SENTENCE = "State the figure first, and say plainly when a figure is missing. "


def thrice(repeats: int) -> list[tuple[int, int]]:
    system = f"[{uuid.uuid4().hex}] You are an analyst for Meridian Supply Co. " + SENTENCE * repeats
    usage = []
    for _ in range(3):
        u = client.chat.completions.create(
            model=MODEL_FAST, max_completion_tokens=16,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": "Which quarter had the highest revenue?"}]).usage
        usage.append((u.prompt_tokens, getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0))
    return usage


print("a fresh prefix each time, so nothing can be cached before the first call\n")
print(f"  {'prompt tokens':>13}{'cached on calls 1, 2, 3':>28}")
cached_at, uncached_at = [], []
for repeats in (40, 80, 90, 95, 100, 120, 200):
    usage = thrice(repeats)
    prompt = usage[0][0]
    print(f"  {prompt:>13,}{', '.join(f'{c:,}' for _, c in usage):>28}")
    (cached_at if any(c for _, c in usage[1:]) else uncached_at).append(prompt)

print(f"\nlongest prompt with nothing cached on a repeat: {max(uncached_at):,} tokens")
print(f"shortest prompt with part of it cached:        {min(cached_at):,} tokens")
