# A context budget, enforced before sending: reserve room for the answer, keep what must be
# kept, and fill what is left in priority order — dropping whole items, never half of one.

from dataclasses import dataclass

import tiktoken

from clarity.config import MODEL_FAST

encoder = tiktoken.encoding_for_model(MODEL_FAST)


@dataclass
class Part:
    name: str
    text: str
    priority: int           # lower is more important
    required: bool = False

    @property
    def tokens(self) -> int:
        return len(encoder.encode(self.text))


def fit(parts: list[Part], window: int, answer_reserve: int) -> tuple[list[Part], list[Part]]:
    budget = window - answer_reserve
    kept = [p for p in parts if p.required]
    used = sum(p.tokens for p in kept)
    if used > budget:
        raise ValueError(f"required parts need {used} tokens; only {budget} available")
    dropped = []
    for part in sorted((p for p in parts if not p.required), key=lambda p: p.priority):
        if used + part.tokens <= budget:
            kept.append(part)
            used += part.tokens
        else:
            dropped.append(part)
    order = {id(p): i for i, p in enumerate(parts)}          # send in the original order
    return sorted(kept, key=lambda p: order[id(p)]), dropped


chunk = "Specification item {n} is at revision {r} and approved for food-contact use. " * 6
parts = [
    Part("system prompt", "You are Clarity. Cite every figure. " * 20, 0, required=True),
    Part("conversation state", '{"account": "Halloway Group", "quarter": "2024 Q2"}', 0, required=True),
    Part("question", "Which items changed revision since 2023?", 0, required=True),
    *[Part(f"retrieved chunk {i + 1} (rank {i + 1})", chunk.format(n=i, r=i % 4), 10 + i)
      for i in range(12)],
    *[Part(f"older turn {i + 1}", "Earlier we discussed depot capacity in Dayton. " * 4, 30 + i)
      for i in range(6)],
]

WINDOW, RESERVE = 1_600, 400
kept, dropped = fit(parts, WINDOW, RESERVE)
print(f"window {WINDOW:,} tokens, {RESERVE} reserved for the answer\n")
print(f"  kept    {sum(p.tokens for p in kept):>5} tokens: " + ", ".join(p.name for p in kept[:4]) + ", ...")
print(f"  dropped {sum(p.tokens for p in dropped):>5} tokens: " + ", ".join(p.name for p in dropped))
print(f"\n  {len([p for p in kept if 'chunk' in p.name])} of 12 retrieved chunks fit; "
      f"{len([p for p in kept if 'turn' in p.name])} of 6 older turns fit")
