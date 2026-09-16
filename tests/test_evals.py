"""
Clarity's evaluations, as tests.

Two things make this different from a normal test file, and both are visible in the
first ten lines.

**The suite is slow and costs money**, so it runs once per session and every test reads
the same result. A fixture that re-runs the system per assertion turns a two-minute
suite into an hour and a bill.

**Assertions are thresholds, not equalities.** `assert score == 0.92` is a test that
fails on Tuesday. `assert score >= 0.80` is a test that fails when something broke.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from clarity.evals.runner import load                           # noqa: E402
from clarity.evals.suite import THRESHOLDS, run                 # noqa: E402
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import build_tools                      # noqa: E402
from clarity.v0_9.agent import Agent, Budget                    # noqa: E402
from meridian_index import load_index                           # noqa: E402

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. If the answer "
          "is not in the documents or the warehouse, say so plainly rather than "
          "guessing.")
# Full run in CI on a schedule; a sample on every pull request. §22.16 is about this line.
SAMPLE = int(os.environ.get("EVAL_SAMPLE", "24"))


def sample(cases: list[dict], n: int) -> list[dict]:
    """The same n cases every run, from every kind and tier in proportion.
    The first n cases of the file are all one kind, and a layer that
    scores another kind would have nothing to score."""
    if not n:
        return cases
    rng, picked = random.Random(22), []
    for group in sorted({(c["kind"], c["tier"]) for c in cases}):
        pool = [c for c in cases if (c["kind"], c["tier"]) == group]
        size = max(1, round(n * len(pool) / len(cases)))
        picked += rng.sample(pool, size)
    return picked


@pytest.fixture(scope="session")
def layers():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("no API key; evaluations are integration tests")
    chunks, vectors = load_index()
    tools = build_tools(Retriever(chunks, vectors), Warehouse())

    def answer(question: str):
        result = Agent(tools, budget=Budget(steps=6)).run(question, system=SYSTEM)
        # What every tool returned goes to the retrieval layer, which checks whether the
        # passage a case needed was ever in the room. Without it that layer scores zero.
        return (result.answer, [s.tool for s in result.steps if s.tool],
                " ".join(s.result for s in result.steps if s.tool))

    cases = sample(load(), SAMPLE)
    return run(answer, cases=cases, judge_sample=10 if SAMPLE else 30)


@pytest.mark.parametrize("layer", list(THRESHOLDS))
def test_layer_above_floor(layers, layer):
    result = layers[layer]
    # An empty layer scores 100%: a sample that never reaches it passes silently.
    assert result.total, f"{layer} scored no cases"
    assert result.score >= THRESHOLDS[layer], (
        f"{layer} scored {result.score:.0%}, "
        f"floor {THRESHOLDS[layer]:.0%}; failing: {result.failures[:8]}")


def test_no_answer_without_a_source(layers):
    """The one rule from §22.11 that is a property rather than a prediction."""
    assert not layers["grounded"].failures, (
        f"answers with no source call: {layers['grounded'].failures[:8]}")


def test_golden_set_is_intact():
    """Cheap, no model, and it catches the corpus changing underneath the set."""
    cases = load()
    assert len(cases) == 120
    assert sum(1 for c in cases if c["kind"] == "unanswerable") == 20
    for case in cases:
        if case["kind"] == "unanswerable":
            assert case["answer"] is None, f"{case['id']} should have no answer"
        else:
            assert case["answer"], f"{case['id']} has no answer"
