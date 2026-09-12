"""
Clarity v0.8 — the tools, and the loop that runs them.

Four rules hold this together, and each cost somebody a bad afternoon to learn:

  every tool has a timeout          the silent hang is always a missing one
  every error names the fix         the error is the next thing the model reads
  empty is not an error             `[]` is an answer; treating it as failure loops
  nothing is ever eval'd            the model's output is untrusted input, always
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_FAST                  # noqa: E402
from clarity.v0_6.retrieve import Retriever            # noqa: E402
from clarity.v0_7.warehouse import Warehouse           # noqa: E402


class ToolError(Exception):
    """
    An error the model is expected to read and act on.

    The message is a prompt: name what was wrong, name the valid values, and say what
    to retry with. Chapter 16 measured terse errors recovering 0 times out of 3 and
    instructive ones 2 out of 3.
    """


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., Any]
    timeout: float = 10.0

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


def _obj(**properties) -> dict:
    return {"type": "object", "properties": properties,
            "required": [k for k, v in properties.items() if "default" not in v],
            "additionalProperties": False}


def build_tools(retriever: Retriever, warehouse: Warehouse) -> list[Tool]:
    def search_documents(query: str, limit: int = 5) -> list[dict]:
        passages = retriever.search(query, k=min(limit, 10))
        # An empty list is a valid answer, not a failure. Raising here would send the
        # model looking for a different phrasing forever.
        return [{"source": p.source, "heading": p.heading, "text": p.text[:600]}
                for p in passages]

    def query_warehouse(question: str) -> dict:
        result = warehouse.ask(question)
        if result is None:
            because = getattr(warehouse, "last_refusal", None)
            raise ToolError(
                (f"Refused: {because}. " if because else "") +
                "That cannot be computed from the sales data. It holds revenue, "
                "gross_profit, margin_pct, orders, units, cost and discounts, by "
                "region, category, supplier, segment, channel, customer, sku, year "
                "and quarter. Nothing about headcount, contracts or logistics. "
                "If the answer is written in a document, use search_documents.")
        return {"rows": result.rows, "sql": result.sql,
                "metric": result.spec.metric, "definition": result.note}

    def arithmetic(expression: str) -> float:
        """
        Evaluate a simple expression.

        Never `eval`. The model's output is untrusted input — Chapter 29 shows what
        arrives in it — so this parses a restricted grammar and refuses everything else.
        """
        import ast
        import operator

        allowed = {ast.Add: operator.add, ast.Sub: operator.sub,
                   ast.Mult: operator.mul, ast.Div: operator.truediv,
                   ast.Pow: operator.pow, ast.USub: operator.neg}

        def walk(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in allowed:
                return allowed[type(node.op)](walk(node.left), walk(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
                return allowed[type(node.op)](walk(node.operand))
            raise ToolError(
                f"Only + - * / ** and numbers are allowed. {expression!r} contains "
                f"{type(node).__name__}. Retry with arithmetic only, using "
                "search_documents or query_warehouse to obtain any values first.")

        try:
            tree = ast.parse(expression, mode="eval").body
        except SyntaxError as error:
            raise ToolError(f"{expression!r} is not an expression: {error.msg}. "
                            "Retry with something like '(8461842 - 5661205) / 8461842'.")
        return round(float(walk(tree)), 6)

    def today(offset_days: int = 0) -> str:
        """The model has no clock. Chapter 5, made concrete."""
        return (date.today() + timedelta(days=offset_days)).isoformat()

    return [
        Tool("search_documents",
             "Search Meridian's quarterly reviews, contracts and support tickets for a "
             "passage. Use for what happened, why, or what someone said. Do NOT use "
             "for figures that must be exact — those come from query_warehouse. "
             "Returns an empty list if nothing matches, which is an answer.",
             _obj(query={"type": "string", "description": "What to look for."},
                  limit={"type": "integer", "description": "Max passages, default 5.",
                         "default": 5}),
             search_documents),
        Tool("query_warehouse",
             "Compute an exact figure from the sales database — revenue, margin, "
             "orders, units — optionally by region, category, supplier or quarter. "
             "Returns the number, the query that produced it and the definition used. "
             "Do NOT use for anything outside sales, such as headcount or contract "
             "terms.",
             _obj(question={"type": "string",
                            "description": "The numeric question, in plain English."}),
             query_warehouse),
        Tool("arithmetic",
             "Evaluate an arithmetic expression over numbers you already have. Cannot "
             "look anything up. Use it rather than doing the sum yourself.",
             _obj(expression={"type": "string",
                              "description": "e.g. '(8461842 - 5661205) / 8461842'"}),
             arithmetic),
        Tool("today",
             "Today's date, or a date offset from it. You have no clock; ask.",
             _obj(offset_days={"type": "integer",
                               "description": "Days from today. Default 0.",
                               "default": 0}),
             today),
    ]


@dataclass
class Turn:
    """One pass of the loop, kept so a run can be read afterwards."""
    tool: str
    arguments: dict
    result: str
    seconds: float
    failed: bool = False


class ToolRunner:
    def __init__(self, tools: list[Tool], client: OpenAI | None = None,
                 max_rounds: int = 6):
        self.tools = {t.name: t for t in tools}
        self.client = client or OpenAI()
        self.max_rounds = max_rounds

    def _execute(self, name: str, arguments: dict) -> tuple[str, float, bool]:
        tool = self.tools.get(name)
        if tool is None:
            return (f"No tool called {name!r}. Available: "
                    f"{', '.join(self.tools)}.", 0.0, True)
        started = time.time()
        try:
            # Every tool call runs under a timeout. Without one, a tool that never
            # returns hangs the whole conversation with no error anywhere.
            with ThreadPoolExecutor(max_workers=1) as pool:
                value = pool.submit(tool.run, **arguments).result(timeout=tool.timeout)
            return json.dumps(value, default=str)[:4000], time.time() - started, False
        except FutureTimeout:
            return (f"{name} took longer than {tool.timeout}s and was abandoned. "
                    "Try a narrower request.", time.time() - started, True)
        except ToolError as error:
            return str(error), time.time() - started, True
        except TypeError as error:
            return (f"{name} rejected those arguments: {error}. Check the schema and "
                    "retry.", time.time() - started, True)
        except Exception as error:                                   # noqa: BLE001
            return (f"{name} failed: {type(error).__name__}: {error}",
                    time.time() - started, True)

    def run(self, question: str, system: str = "") -> tuple[str, list[Turn]]:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})
        trace: list[Turn] = []

        for _ in range(self.max_rounds):
            reply = self.client.chat.completions.create(
                model=MODEL_FAST, temperature=0, max_completion_tokens=700,
                messages=messages,
                tools=[t.schema() for t in self.tools.values()],
            ).choices[0].message

            if not reply.tool_calls:
                return (reply.content or "").strip(), trace

            messages.append(reply)
            # Calls in one reply are independent, so they run together.
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs = [(call, pool.submit(self._execute, call.function.name,
                                           json.loads(call.function.arguments)))
                        for call in reply.tool_calls]
                for call, job in jobs:
                    result, seconds, failed = job.result()
                    trace.append(Turn(call.function.name,
                                      json.loads(call.function.arguments),
                                      result[:200], seconds, failed))
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": result})

        return ("I could not finish that within the step budget.", trace)
