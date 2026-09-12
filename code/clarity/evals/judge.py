"""
Clarity's judges: a model grading a model.

Three architectures, and the difference between them is what the judge is given.

    pointwise        one answer, judged alone
    reference-based  one answer, against the expected answer
    pairwise         two answers, asked which is better

Only the second is really usable for regression testing, and Chapter 22 measures why:
a pointwise judge with no reference is grading its own opinion of the domain, and a
pairwise judge has an opinion about which side of the page an answer is on.

The prompt rules here are not style. Each one is a bias that was measured before it was
written down:

    ask for a verdict, not a score   a 1-5 scale produces 4s
    give the reference answer        without one you measure plausibility
    demand the reason first          a verdict written before its reason is a guess
    never mention which system       or you have measured the judge's expectations
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_FAST                          # noqa: E402

POINTWISE = """You grade answers about Meridian Supply Co., a distributor.

Reply with JSON only: {"reason": "...", "verdict": "CORRECT" or "WRONG"}

Write the reason first and the verdict second.
CORRECT means the answer states the fact the question asks for.
Extra context, hedging, or a fuller answer than asked for does not make it wrong.
A confident answer containing a wrong figure or a wrong name is WRONG."""

REFERENCE = POINTWISE + """

You are given the expected answer. Judge only whether the answer conveys it.
A different wording, rounding, or level of precision is still CORRECT if it is the
same fact. A different fact is WRONG."""

PAIRWISE = """You compare two answers to the same question about Meridian Supply Co.

Reply with JSON only: {"reason": "...", "winner": "A" or "B" or "TIE"}

Write the reason first. Judge only which answer better states the fact the question
asks for. Length, formatting and confidence are not quality."""


PAIRWISE_REFERENCE = PAIRWISE + """

You are given the expected answer. The better answer is the one that states it.
If both state it, or neither does, reply TIE."""


@dataclass
class Verdict:
    correct: bool | None
    reason: str
    raw: str


class Judge:
    def __init__(self, model: str = MODEL_FAST, client: OpenAI | None = None,
                 system: str = REFERENCE) -> None:
        self.model, self.system = model, system
        self.client = client or OpenAI()

    def _ask(self, user: str, tokens: int = 220) -> dict:
        reply = self.client.chat.completions.create(
            model=self.model, max_completion_tokens=tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": self.system},
                      {"role": "user", "content": user}])
        text = reply.choices[0].message.content or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"reason": text[:200], "verdict": "", "winner": ""}

    def score(self, question: str, answer: str,
              expected: str | None = None) -> Verdict:
        parts = [f"Question: {question}"]
        if expected is not None:
            parts.append(f"Expected answer: {expected}")
        parts.append(f"Answer: {answer[:1500]}")
        out = self._ask("\n\n".join(parts))
        verdict = str(out.get("verdict", "")).upper()
        return Verdict(True if "CORRECT" in verdict else
                       False if "WRONG" in verdict else None,
                       str(out.get("reason", ""))[:300], json.dumps(out)[:400])

    def compare(self, question: str, a: str, b: str,
                expected: str | None = None) -> str:
        reference = f"Expected answer: {expected}\n\n" if expected else ""
        out = self._ask(f"Question: {question}\n\n{reference}"
                        f"Answer A: {a[:1200]}\n\nAnswer B: {b[:1200]}")
        winner = str(out.get("winner", "")).upper()
        return "A" if winner == "A" else "B" if winner == "B" else "TIE"


def kappa(a: list[int], b: list[int]) -> tuple[float, float]:
    """Raw agreement and Cohen's kappa, as in §21.8."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return observed, (observed - expected) / (1 - expected) if expected < 1 else 0.0
