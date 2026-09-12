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
# Full run in CI on a schedule; a sample on every push. §22.11 is about this line.
SAMPLE = int(os.environ.get("EVAL_SAMPLE", "24"))


@pytest.fixture(scope="session")
def layers():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("no API key; evaluations are integration tests")
    chunks, vectors = load_index()
    tools = build_tools(Retriever(chunks, vectors), Warehouse())

    def answer(question: str):
        result = Agent(tools, budget=Budget(steps=6)).run(question, system=SYSTEM)
        return result.answer, [s.tool for s in result.steps if s.tool]

    cases = load()[:SAMPLE] if SAMPLE else load()
    return run(answer, cases=cases, judge_sample=min(10, SAMPLE))


@pytest.mark.parametrize("layer", ["deterministic", "grounded", "judge"])
def test_layer_above_floor(layers, layer):
    result = layers[layer]
    assert result.score >= THRESHOLDS[layer], (
        f"{layer} scored {result.score:.0%}, floor is {THRESHOLDS[layer]:.0%}. "
        f"Failing cases: {', '.join(result.failures[:8])}")


def test_no_answer_without_a_source(layers):
    """The one rule from §22.7 that is a property rather than a prediction."""
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
