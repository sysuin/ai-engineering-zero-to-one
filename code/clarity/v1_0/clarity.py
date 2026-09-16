"""
Clarity v1.0 — every chapter, in one object.

Thirty-one chapters built pieces. This file is the assembly, and the assembly is where
you find out whether the pieces fit. It is deliberately short: if composing a system out
of its own parts takes three hundred lines, the parts have the wrong shape.

What it composes, and the chapter that argued for each:

    retrieval        Ch 11-14   hybrid, filtered, MMR — measured at each step
    the warehouse    Ch 15      numbers come from SQL, not from prose
    tools            Ch 16      two of them, with typed arguments
    the loop         Ch 17      budgeted, with a stop rule (v0.9's agent)
    the guard        Ch 29      active content stripped, tools allowlisted by role
    the cache        Ch 28      exact match only, keyed by tenant, optional
    the bulkhead     Ch 26      a ceiling on concurrent runs; the rest are shed
    tracing          Ch 23      one span per model call, priced, redacted at the boundary

What it does not compose, although earlier chapters built it — Chapter 31's review of this
file finds both, and says what their absence costs:

    the gateway      Ch 27      fallback and a breaker per provider (platform/gateway.py)
    the store        Ch 25      runs, steps and feedback in a database (v0_17/store.py)

One rule governs the composition: **nothing below knows about anything above it.** The
retriever has never heard of HTTP, the agent has never heard of the cache, and the guard
has never heard of the agent. That is what made every one of those layers testable on its
own, and it is the only structural property of this system worth copying.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_FAST                              # noqa: E402
from clarity.evals.runner import abstained                         # noqa: E402
from clarity.platform.cache import ExactCache                      # noqa: E402
from clarity.platform.guard import Allowlist, strip_active         # noqa: E402
from clarity.platform.instrumented import TracedClient, traced_tools  # noqa: E402
from clarity.platform.resilience import Bulkhead                   # noqa: E402
from clarity.platform.tracing import span                          # noqa: E402
from clarity.v0_6.retrieve import Retriever                        # noqa: E402
from clarity.v0_7.warehouse import Warehouse                       # noqa: E402
from clarity.v0_8.tools import build_tools                         # noqa: E402
from clarity.v0_9.agent import Agent, Budget, Run                  # noqa: E402
from meridian_index import load_index                              # noqa: E402

SYSTEM = (
    "You are an analyst for Meridian. Use tools for every fact. Cite the source of "
    "each number. If the answer is not in the documents or the warehouse, say so "
    "plainly rather than estimating."
)

# Who may do what. Ch 29's point: the check belongs beside the tool, not in the prompt,
# because a prompt is a suggestion and an allowlist is a gate.
PERMISSIONS = Allowlist({
    "analyst": {"search_documents", "query_warehouse"},
    "viewer": {"search_documents"},
})


@dataclass
class Answer:
    """What one question produced, and everything needed to defend it."""
    text: str
    tools: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    # What the tools returned, concatenated. Not for display — for §21.9's recall
    # measurement, which asks whether the passage a case was verified against actually
    # came back. Without this an answer's grounding cannot be checked after the fact,
    # and a recall metric computed from filenames alone reports 0% on a healthy system.
    context: str = ""
    seconds: float = 0.0
    tokens: int = 0
    cached: bool = False
    refused: bool = False
    stopped_because: str = "answered"
    trace_id: str = ""

    @property
    def grounded(self) -> bool:
        """An answer with no tool behind it is a guess, whatever it sounds like."""
        return bool(self.tools) or self.refused


class Clarity:
    """
    The whole system, with one method that matters.

    Everything optional is off by default and switched on by an argument, because the
    default configuration is the one that gets deployed by accident and it should be
    the boring one.
    """

    def __init__(self, *, client=None, cache: bool = True, tenant: str = "meridian",
                 role: str = "analyst", max_steps: int = 6,
                 max_concurrent: int = 8, model: str = MODEL_FAST) -> None:
        self.client = TracedClient(client)
        self.tenant, self.role, self.model = tenant, role, model
        self.cache = ExactCache() if cache else None
        # Ch 26: the constraint is concurrent calls to a provider, not CPU. A pool
        # that bounds them is the capacity plan, and it belongs at the entrance.
        self.pool = Bulkhead(limit=max_concurrent)
        self.budget = Budget(steps=max_steps)
        self._engine: list | None = None

    # ------------------------------------------------------------------ lazy start
    def engine(self) -> list:
        """
        The index loads once, on first use, and stays warm.

        Ch 31's sizing is the reason this is a lazily-initialised attribute rather than
        a module import: the index is the system's state, and a system whose state is a
        warm index cannot be serverless.
        """
        if self._engine is None:
            chunks, vectors = load_index()
            self._engine = traced_tools(
                build_tools(Retriever(chunks, vectors, client=self.client),
                            Warehouse(client=self.client)))
        return self._engine

    # ------------------------------------------------------------------ the method
    def ask(self, question: str, *, on_step=None) -> Answer:
        started = time.perf_counter()
        # Ch 29: the question arrives from outside. Strip anything active before it
        # reaches a prompt, and do it here rather than in four places downstream.
        question = strip_active(question.strip())

        if self.cache is not None:
            hit = self.cache.get(question, tenant=self.tenant)
            if hit is not None:
                return Answer(text=hit, cached=True,
                              seconds=time.perf_counter() - started)

        tools = [t for t in self.engine()
                 if PERMISSIONS.permits(self.role, t.name)]

        with span("clarity.ask", **{"clarity.tenant": self.tenant,
                                    "clarity.role": self.role}) as current:
            agent = Agent(tools, client=self.client, budget=self.budget,
                          model=self.model)
            run: Run = self.pool(agent.run, question, system=SYSTEM,
                                 on_step=on_step)
            raw = current.get_span_context().trace_id if current else 0
            # Zero means no exporter was installed — `tracing.start()` was never
            # called. An id of thirty-two zeros in a database is worse than a blank,
            # because it looks like a real one until somebody searches for it.
            trace_id = f"{raw:032x}" if raw else ""

        answer = Answer(
            text=run.answer,
            tools=[s.tool for s in run.steps if s.tool],
            sources=_sources(run),
            context=" ".join(s.result for s in run.steps
                             if s.result and not s.failed),
            seconds=time.perf_counter() - started,
            tokens=run.tokens,
            refused=_is_refusal(run.answer),
            stopped_because=run.stopped_because,
            trace_id=trace_id,
        )
        # A refusal is cached like any other answer. It is a correct result, and
        # re-deriving it costs exactly as much as re-deriving a wrong one.
        if self.cache is not None:
            self.cache.put(question, answer.text, tenant=self.tenant)
        return answer


# One refusal detector, in one place — `clarity.evals.runner.abstained`. A second copy
# of that phrase list is how a system comes to refuse in production and score zero on
# abstention in the suite, or the reverse. Ch 27 spent a re-run of four chapters on a
# detector that missed one phrasing; the lesson was not "add the phrasing" but "have
# one detector".
_is_refusal = abstained


SOURCE_FIELD = re.compile(r'"source"\s*:\s*"([^"]+)"')


def _sources(run: Run) -> list[str]:
    """
    Which documents the answer could have come from.

    Read out of the tool *results*, not the tool arguments. A first version read
    `arguments["source"]` and always returned an empty list, because the arguments of
    `search_documents` are a query and a limit — the sources are what came back.

    A second version parsed the result as JSON and still returned nothing, because
    v0.9's loop truncates a tool result at three thousand characters before handing it
    to the model, which is right for the model and fatal for `json.loads`. So this
    reads the field, not the document.
    """
    found: set[str] = set()
    for step in run.steps:
        if step.failed or not step.result:
            continue
        found.update(SOURCE_FIELD.findall(step.result))
    return sorted(found)
