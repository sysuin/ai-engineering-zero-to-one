# timeout: 1800
# The faithfulness checker from 03_rag_metrics, measured the way 01 measured the judge: answers
# whose support is certain by construction. Each is built from a passage that was retrieved,
# then left alone, given an invented claim, given a contradicted figure, or given a true claim
# from a document that was not retrieved.

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                           # noqa: E402
from clarity.evals.runner import load, plain                    # noqa: E402
from clarity.v0_5.rag import Clarity                            # noqa: E402
from meridian_index import load_index                           # noqa: E402
from openai import OpenAI                                       # noqa: E402

K = 6
client = OpenAI()
chunks, vectors = load_index()
system = Clarity(chunks, vectors)
cases = [c for c in load() if c["kind"] == "document" and c.get("span")][:60]
FAITHFUL = """You check whether an answer is supported by the passages it was given.

Reply with JSON only: {"unsupported": ["..."], "supported_count": N}

List every factual claim in the answer that is NOT stated in the passages. A claim
that is stated in different words is supported. General framing is not a claim."""


def squash(text: str) -> str:
    return plain(" ".join(text.split()))


def other_digit(figure: str) -> str:
    """Change the last digit of a figure, so it keeps its shape and loses its truth."""
    i = max(i for i, ch in enumerate(figure) if ch.isdigit())
    return figure[:i] + str((int(figure[i]) + 3) % 10) + figure[i + 1:]


def build(case: dict):
    passages = system.retrieve(case["question"], k=K)
    context = "\n\n".join(f"[{p.source}] {p.text}" for p in passages)
    span = " ".join(case["span"].split())
    if squash(span)[:60] not in squash(context):
        return None                                    # only answers built from retrieved text
    figure = case["answer"]
    if figure not in span or not any(ch.isdigit() for ch in figure):
        return None                                    # a figure is what can be contradicted
    wrong = other_digit(figure)
    elsewhere = [c for c in load() if c.get("span") and c["source"] != case["source"]
                 and squash(c["span"])[:60] not in squash(context)]
    true_absent = " ".join(random.Random(case["id"]).choice(elsewhere)["span"].split())
    invented = "Management expects the figure to improve by 17.3% in the following quarter."
    assert "17.3" not in context and squash(wrong) not in squash(span)
    return context, {"supported": span,
                     "invented claim": f"{span} {invented}",
                     "contradicted figure": span.replace(figure, wrong, 1),
                     "true, not retrieved": f"{span} {true_absent}"}


def flagged(context: str, answer: str) -> bool:
    reply = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=300, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": FAITHFUL},
                  {"role": "user", "content": f"Passages:\n{context[:6000]}\n\nAnswer: {answer}"}])
    try:
        unsupported = json.loads(reply.choices[0].message.content or "{}").get("unsupported")
    except json.JSONDecodeError:
        unsupported = None
    return bool([u for u in (unsupported or []) if isinstance(u, str) and u.strip()])


with ThreadPoolExecutor(max_workers=8) as pool:
    built = [b for b in pool.map(build, cases) if b]
jobs = [(context, kind, answer) for context, variants in built for kind, answer in variants.items()]
with ThreadPoolExecutor(max_workers=8) as pool:
    verdicts = list(pool.map(lambda j: flagged(j[0], j[2]), jobs))

print(f"{len(built)} answers built from retrieved passages, each in four versions\n")
print(f"  {'version':24}{'should be flagged':>19}{'flagged':>10}")
for kind, should in (("supported", "no"), ("invented claim", "yes"),
                     ("contradicted figure", "yes"), ("true, not retrieved", "yes")):
    hits = [v for (_, k, _), v in zip(jobs, verdicts) if k == kind]
    print(f"  {kind:24}{should:>19}{sum(hits):>6} of {len(hits)}")
print("\n'true, not retrieved' is flagged by design: faithfulness asks whether the passages")
print("support a claim, not whether the claim is true")
