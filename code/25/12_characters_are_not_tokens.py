# timeout: 300
# Clarity's edge accepts a question of up to 2,000 characters. What is the most that lets through?
# Six questions of exactly 2,000 characters, each counted in bytes, in the tokeniser's tokens,
# and in the prompt tokens the API actually billed for it.

import random
import statistics
import sys
import time
from pathlib import Path

import tiktoken
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                        # noqa: E402
from clarity.evals.runner import load                        # noqa: E402
from clarity.v0_17.service import AskRequest                 # noqa: E402
from clarity.v0_9.agent import Budget                        # noqa: E402

LIMIT = next(m.max_length for m in AskRequest.model_fields["question"].metadata
             if hasattr(m, "max_length"))
client, encoder, rng = OpenAI(), tiktoken.encoding_for_model(MODEL_FAST), random.Random(25)
reviews = " ".join(p.read_text() for p in sorted(Path("data/meridian/documents/quarterly-reviews").glob("*.md")))


def fill(unit, n: int = LIMIT) -> str:
    text = ""
    while len(text) < n:
        text += unit()
    return text[:n]


QUESTIONS = {
    "English prose": " ".join(reviews.split())[:LIMIT],
    "digits": fill(lambda: f"{rng.randint(0, 10**9)} "),
    "Chinese characters": fill(lambda: chr(rng.randint(0x4E00, 0x9FFF))),
    "emoji": fill(lambda: chr(rng.randint(0x1F300, 0x1F5FF))),
    "one accent, repeated": "a" + "\u0301" * (LIMIT - 1),
    "mixed scripts": fill(lambda: chr(rng.choice([rng.randint(0x0400, 0x04FF),
                                                  rng.randint(0x0980, 0x09FF),
                                                  rng.randint(0x10A0, 0x10FF)]))),
}


def billed(text: str) -> int:
    """Prompt tokens the API counted for a message containing only this text."""
    reply = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=16,
        messages=[{"role": "user", "content": text}])
    return reply.usage.prompt_tokens


overhead = billed("a") - len(encoder.encode("a"))
golden = [len(encoder.encode(c["question"])) for c in load()]
print(f"the edge accepts up to {LIMIT:,} characters; the golden set's {len(golden)} questions")
print(f"are {statistics.median(golden):.0f} tokens at the median and {max(golden)} at the longest\n")
print(f"  {'2,000 characters of':<22}{'accepted':>9}{'bytes':>7}{'tokens':>8}{'billed':>8}"
      f"{'encode':>9}")
rows = {}
for name, text in QUESTIONS.items():
    assert len(text) == LIMIT
    AskRequest(question=text)                       # raises if the edge would refuse it
    started = time.perf_counter()
    tokens = len(encoder.encode(text))
    seconds = time.perf_counter() - started
    rows[name] = billed(text) - overhead
    print(f"  {name:<22}{'yes':>9}{len(text.encode()):>7,}{tokens:>8,}{rows[name]:>8,}"
          f"{seconds * 1000:>7.1f}ms")

worst = max(rows, key=rows.get)
budget, calls = Budget().tokens, 12 + 1                 # max_steps=12, then the answer
median = statistics.median(golden)
print(f"\n  billed = the API's prompt tokens, less {overhead} for the message itself")
print(f"\nEvery model call resends the question. Over the {calls} calls a max_steps=12 run")
print(f"can make, the question alone uses this much of the run's {budget:,}-token budget:")
for label, tokens in (("a median golden question", median), (f"2,000 characters of {worst}", rows[worst])):
    print(f"  {label:<32}{calls} x {tokens:>5,.0f} = {calls * tokens:>6,.0f} tokens, "
          f"{calls * tokens / budget:>5.1%}")
