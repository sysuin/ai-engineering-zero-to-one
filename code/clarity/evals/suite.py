"""
Clarity v0.14 — the six-layer suite.

It began with three. §22.14's mutation test walked three plausible bugs straight through
that version, and the three layers below marked * are what it took to stop them. Layers,
cheapest first, and the order is the whole design:

    1  deterministic   string rules over the golden set. Free, exact, no model.
    2  grounded        did the answer come from a source? computed from the trace.
    3  judge           a reference-anchored model verdict, on a sample.
    4  abstention *    the refusals, scored again on their own.
    5  warehouse  *    the numeric cases, scored again on their own.
    6  retrieval  *    did the verified passage actually come back?

Layer 1 catches most regressions and costs nothing, so it runs on everything. Layer 3
costs a model call per case and catches the things a string cannot, so it runs on a
sample. A suite that starts with the judge is a suite that gets switched off when the
bill arrives.

Layers 4 and 5 are slices of layer 1, scored separately. That is not redundancy: a
population that is a sixth of the set can collapse entirely while the overall figure
moves six points, which is exactly how the warehouse bug got through.

Every layer has a threshold, and the build fails when any layer drops below it. The
thresholds are floors, not targets: they are set just under where the system runs
today, so that a regression trips them and normal variance does not.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.evals.judge import REFERENCE, Judge                # noqa: E402
from clarity.evals.runner import correct, load, plain           # noqa: E402

SOURCES = {"query_warehouse", "search_documents"}


def squeeze(text: str) -> str:
    """
    Every character that carries meaning, and nothing that carries layout.

    Tool results arrive as JSON, so a newline in a document is the two characters
    backslash-n by the time it reaches here. An earlier version of this file compared
    against that and scored 0% retrieval on a healthy system — a suite that fails the
    control is measuring itself, not the code.
    """
    return re.sub(r"[^a-z0-9.%$]", "", plain(text).replace("\\n", " "))

# Six layers, not one. The first version of this file had three, and §22.14's
# mutation test walked three plausible bugs straight through it: an overall floor
# averages a collapse in one population into a healthy majority, and no floor on
# retrieval means a broken retriever is invisible whenever a fallback exists.
#
# The floors below are a backstop, deliberately loose. What actually catches a
# regression is DRIFT: a drop of more than this many points against a recorded
# baseline. Absolute floors set just under today's score look rigorous and produce a
# build that fails on Tuesdays, because every layer here is a sample — the second
# version of this file did exactly that and failed its own control.
# `warehouse` and `abstention` are both slices of `deterministic`, scored again on
# their own. That is not redundancy — it is the point. A regression confined to one
# population is divided by every case outside it, and the twenty warehouse questions
# can collapse entirely while the overall figure moves six points.
THRESHOLDS = {"deterministic": 0.60, "grounded": 0.90, "judge": 0.55,
              "abstention": 0.55, "retrieval": 0.40, "warehouse": 0.55}
DRIFT = 0.08


@dataclass
class Layer:
    name: str
    passed: int = 0
    total: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 1.0

    baseline: float | None = None

    @property
    def ok(self) -> bool:
        if self.score < THRESHOLDS[self.name]:
            return False
        if self.baseline is None:
            return True
        return self.score >= self.baseline - DRIFT

    def record(self, case_id: str, good: bool) -> None:
        self.total += 1
        self.passed += int(good)
        if not good:
            self.failures.append(case_id)


def run(answer_fn, cases: list[dict] | None = None, judge_sample: int = 12,
        judge: Judge | None = None,
        baseline: dict[str, float] | None = None) -> dict[str, Layer]:
    """
    `answer_fn(question) -> (answer, tools_used)`.

    The suite knows nothing about how the answer was produced, which is what lets
    §22.14 swap in a deliberately broken system and see whether anything notices.
    """
    cases = cases if cases is not None else load()
    judge = judge or Judge(system=REFERENCE)
    layers = {name: Layer(name, baseline=(baseline or {}).get(name))
              for name in THRESHOLDS}

    for n, case in enumerate(cases):
        result = answer_fn(case["question"])
        answer, tools = result[0], result[1]
        context = result[2] if len(result) > 2 else ""

        unanswerable = case["kind"] == "unanswerable"

        # Layer 1 scores everything, and layer 4 scores the refusals again on their
        # own. A population that is a sixth of the set can collapse entirely without
        # moving an overall figure by more than the floor's slack.
        right = correct(case, answer)
        layers["deterministic"].record(case["id"], right)
        if unanswerable:
            layers["abstention"].record(case["id"], right)
        if case["kind"] == "warehouse":
            layers["warehouse"].record(case["id"], right)

        layers["grounded"].record(case["id"],
                                  bool(SOURCES & set(tools)) or unanswerable)

        # Layer 5: did the passage the case was verified against actually come back?
        #
        # Only for cases the documents alone can answer. A first version scored every
        # `document` case and reported 0% on a healthy system, because "revenue in
        # 2023 Q1" is answered from the warehouse and no document is ever fetched.
        # That is §22.13's mistake again: where the gold came from is not a claim
        # about which tool should run.
        span = case.get("span")
        if span and case.get("tier") == "tail" and case["kind"] == "document":
            layers["retrieval"].record(case["id"],
                                       squeeze(span)[:48] in squeeze(context))

        if n < judge_sample and not unanswerable:
            verdict = judge.score(case["question"], answer, case["answer"])
            layers["judge"].record(case["id"], bool(verdict.correct))

    return layers


def report(layers: dict[str, Layer]) -> str:
    lines = []
    for name in THRESHOLDS:
        layer = layers[name]
        against = (f"baseline {layer.baseline:.0%}" if layer.baseline is not None
                   else f"floor {THRESHOLDS[name]:.0%}")
        lines.append(f"  {name:<15}{layer.passed:>4}/{layer.total:<5}"
                     f"{layer.score:>6.0%}   {against:<16}"
                     f"{'ok' if layer.ok else 'FAIL'}")
    return "\n".join(lines)


def scores(layers: dict[str, Layer]) -> dict[str, float]:
    return {name: layer.score for name, layer in layers.items()}


def green(layers: dict[str, Layer]) -> bool:
    return all(layer.ok for layer in layers.values())
