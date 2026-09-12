"""
Clarity v0.12 — a supervisor, a researcher and an analyst.

The pattern is agents-as-tools: the supervisor has no access to Meridian at all. It can
only ask two colleagues, each of whom has half the tools and a narrower brief, and then
write the answer from what they say.

That is the strongest honest version of the argument for multiple agents:

    each worker sees a smaller tool list, so it chooses better
    each worker has a shorter context, so it costs less per step
    the supervisor sees findings rather than raw rows, so it reasons over less

Chapter 20 measures whether any of that survives contact with the task, against the one
agent from Chapter 17 that already does the whole job.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_FAST                     # noqa: E402
from clarity.v0_8.tools import Tool, _obj                 # noqa: E402
from clarity.v0_9.agent import Agent, Budget              # noqa: E402

RESEARCHER = (
    "You are Meridian's researcher. You read documents and report what they say. "
    "Quote or closely paraphrase, name the source file, and never state a figure the "
    "documents do not contain. If the documents do not answer, say so plainly — an "
    "honest 'not in the documents' is worth more than a guess.")

ANALYST = (
    "You are Meridian's analyst. You compute figures from the sales warehouse. "
    "Report the number, the metric definition you used, and nothing else. If the "
    "warehouse cannot answer, say what it does hold. Never explain *why* a number "
    "moved — that is not in your data.")

SUPERVISOR = (
    "You lead a two-person team at Meridian and you have no data access yourself.\n\n"
    "ask_analyst computes figures from the sales warehouse.\n"
    "ask_researcher finds what the documents say.\n\n"
    "Most questions need both. Ask each colleague one clear, self-contained question "
    "— they cannot see the user's question or each other's answers. Then write the "
    "final answer: the figure first, then the cause, then anything you could not "
    "verify. Never state a figure or a cause a colleague did not give you.")


@dataclass
class TeamRun:
    answer: str = ""
    tokens: int = 0
    seconds: float = 0.0
    calls: int = 0                       # model calls across every agent
    delegations: list[tuple[str, str]] = field(default_factory=list)
    # Full text, untruncated. The transcript below is for reading; this is for
    # measuring, and the two must not be the same list.
    replies: list[tuple[str, str]] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)


class Team:
    def __init__(self, tools: list[Tool], model: str = MODEL_FAST,
                 budget: Budget | None = None) -> None:
        by_name = {t.name: t for t in tools}
        # The split is the design. A worker cannot reach a tool outside its brief, so
        # "the researcher invented a number" stops being a possible bug report.
        self._researcher = [by_name["search_documents"], by_name["today"]]
        self._analyst = [by_name["query_warehouse"], by_name["arithmetic"]]
        self._model = model
        self._budget = budget or Budget(steps=6)

    def run(self, question: str) -> TeamRun:
        import time
        started = time.perf_counter()
        result = TeamRun()

        def delegate(who: str, system: str, tools: list[Tool]):
            def ask(question: str) -> str:
                worker = Agent(tools, budget=self._budget, model=self._model)
                run = worker.run(question, system=system)
                result.tokens += run.tokens
                result.calls += len(run.steps) + 1
                result.delegations.append((who, question))
                result.replies.append((who, run.answer))
                result.transcript.append(f"{who} <- {question}")
                result.transcript.append(f"{who} -> {run.answer.strip()[:160]}")
                return run.answer
            return ask

        tools = [
            Tool("ask_researcher",
                 "Ask the researcher what the documents say. Give one complete "
                 "question; they cannot see anything else.",
                 _obj(question={"type": "string",
                                "description": "A complete, self-contained question."}),
                 delegate("researcher", RESEARCHER, self._researcher), timeout=120),
            Tool("ask_analyst",
                 "Ask the analyst to compute a figure from the sales warehouse. Give "
                 "one complete question; they cannot see anything else.",
                 _obj(question={"type": "string",
                                "description": "A complete, self-contained question."}),
                 delegate("analyst", ANALYST, self._analyst), timeout=120),
        ]

        boss = Agent(tools, budget=Budget(steps=8), model=self._model)
        run = boss.run(question, system=SUPERVISOR)
        result.answer = run.answer
        result.tokens += run.tokens
        result.calls += len(run.steps) + 1
        result.seconds = time.perf_counter() - started
        return result
