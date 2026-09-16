"""
Clarity v0.7 — answering numeric questions without the model writing SQL.

Chapter 8 established the authority boundary: the model reports what it read, and code
decides what it means. This is the same rule applied to a database.

The model fills in a `QuerySpec` — which metric, which dimensions, which filters — and
code turns that into SQL. The model never emits a string that reaches the database, so
there is no query to inject into, no DELETE to refuse, and no dialect for it to get
wrong. What it can do is choose the wrong metric, which is a mistake you can see.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_FAST      # noqa: E402

LAYER = yaml.safe_load((Path(__file__).parent / "semantic.yaml").read_text())
METRICS: dict[str, dict] = LAYER["metrics"]
DIMENSIONS: list[str] = LAYER["dimensions"]
TYPES: dict[str, str] = LAYER["types"]
SOURCE: str = LAYER["source"]

MetricName = Literal[tuple(METRICS)]          # type: ignore[valid-type]
DimensionName = Literal[tuple(DIMENSIONS)]    # type: ignore[valid-type]


class Filter(BaseModel):
    """One WHERE clause, restricted to dimensions the layer allows."""
    dimension: DimensionName
    operator: Literal["=", "!=", ">", ">=", "<", "<="]
    value: str | int | float

    def coerced(self) -> str | int | float:
        """
        Match the value to the column's declared type.

        Without this the model filters `quarter = 'Q3'` against an integer column. The
        query is valid SQL, runs without error, matches nothing, and returns NULL.
        """
        if TYPES.get(self.dimension) != "int":
            return self.value
        if isinstance(self.value, int):
            return self.value
        digits = re.findall(r"-?\d+", str(self.value))
        if not digits:
            raise ValueError(
                f"{self.dimension} is a number and the question gave {self.value!r}")
        return int(digits[0])


class QuerySpec(BaseModel):
    """
    What the model is allowed to decide.

    There is no `sql` field, and that absence is the whole design. The model chooses
    which question to ask; code decides how to ask it.

    The first version of this class also had no `answerable` field, and it was worse
    for it: forced to pick a metric, the model returned confident wrong numbers that
    a score counting refusals as failures rewards (§15.7). Chapter 9's rule applies here
    exactly — "I cannot express this" has to be representable before it can be chosen.
    """
    answerable: bool = Field(
        description="False if the question needs a metric or dimension that is not "
                    "in the lists below, or a comparison between two periods.")
    why_not: str | None = Field(
        default=None, description="If not answerable, what the layer lacks.")
    metric: MetricName = Field(description="Which defined metric to compute.")
    group_by: list[DimensionName] = Field(
        default_factory=list, description="Dimensions to break the metric down by.")
    filters: list[Filter] = Field(
        default_factory=list, description="Restrictions the question states.")
    order: Literal["highest first", "lowest first", "none"] = "none"
    limit: int | None = Field(default=None, ge=1, le=100)


@dataclass
class Result:
    rows: list[tuple]
    sql: str
    spec: QuerySpec
    note: str


class Warehouse:
    def __init__(self, path: str = "data/meridian/warehouse/meridian.db",
                 client: OpenAI | None = None, tenant: str | None = None):
        self.path = path
        self.client = client or OpenAI()
        self.tenant = tenant

    # ------------------------------------------------------------------ build
    def to_sql(self, spec: QuerySpec) -> tuple[str, list]:
        """Turn a spec into parameterised SQL. Nothing here is interpolated from text."""
        metric = METRICS[spec.metric]["sql"]
        select = [*spec.group_by, f"{metric} AS value"]

        where, params = [], []
        for f in spec.filters:
            # `dimension` and `operator` are constrained by the schema, so neither can
            # carry anything unexpected. The value is always a bound parameter.
            where.append(f"{f.dimension} {f.operator} ?")
            params.append(f.coerced())
        if self.tenant:
            where.append("tenant = ?")
            params.append(self.tenant)

        sql = f"SELECT {', '.join(select)} FROM {SOURCE}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        if spec.group_by:
            sql += " GROUP BY " + ", ".join(spec.group_by)
        if spec.order != "none":
            sql += f" ORDER BY value {'DESC' if spec.order == 'highest first' else 'ASC'}"
        if spec.limit:
            sql += f" LIMIT {int(spec.limit)}"
        return sql, params

    # ------------------------------------------------------------------ plan
    def plan(self, question: str) -> QuerySpec | None:
        catalogue = "\n".join(
            f"  {name}: {m['label']} — {m['note']}" for name, m in METRICS.items())
        parsed = self.client.chat.completions.parse(
            model=MODEL_FAST, temperature=0, max_completion_tokens=500,
            response_format=QuerySpec,
            messages=[{"role": "system", "content":
                       "Translate the question into a query specification.\n\n"
                       f"Metrics available:\n{catalogue}\n\n"
                       f"Dimensions available: {', '.join(DIMENSIONS)}\n\n"
                       "Use only these. Include a filter only for something the "
                       "question explicitly states.\n\n"
                       "Set answerable=false if the question needs a metric or "
                       "dimension that is not listed, counts rows of a table other "
                       "than sales, or compares two time periods. Do not substitute "
                       "the nearest available metric."},
                      {"role": "user", "content": question}],
        ).choices[0].message.parsed
        return parsed

    # ------------------------------------------------------------------ answer
    def ask(self, question: str) -> Result | None:
        spec = self.plan(question)
        if spec is None or not spec.answerable:
            # Keep the reason so callers can report it. A refusal that does not say
            # why is indistinguishable from a bug, which Chapter 17 discovered the
            # expensive way.
            self.last_refusal = (spec.why_not if spec and spec.why_not
                                 else "the planner could not express this question")
            return None
        self.last_refusal = None
        sql, params = self.to_sql(spec)
        con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)   # read-only
        try:
            rows = con.execute(sql, params).fetchall()
        finally:
            con.close()
        return Result(rows, sql, spec, METRICS[spec.metric]["note"])
