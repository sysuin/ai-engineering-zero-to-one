"""
Clarity v0.15 — tracing, in the vendor-neutral format.

Three decisions are baked into this file, and each one is a bill somebody paid.

**Traces, not logs.** A log line tells you a thing happened. A trace tells you what it
happened *inside*, which for a system that makes eight nested calls per question is the
only shape of record you can read at two in the morning. §23.2 puts the same incident
through all three formats.

**Every span carries tokens and money.** Cost is not a separate system. It is an
attribute on the span that spent it, because the question is never "what did today
cost" — it is "what did *this* cost, and which part of it".

**Nothing sensitive goes in an attribute.** Spans are shipped to somebody else's
service, kept for months, and read by people who were never granted access to the
documents. §23.5 measures what leaks when you forget that.
"""
from __future__ import annotations

import json
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import rate                                 # noqa: E402

# What a span may never carry, whatever the caller passes. The list is short on
# purpose: a redaction rule nobody can recite is a rule nobody applies.
SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+|password\s*[=:]\s*\S+)",
                    re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
MAX_ATTRIBUTE = 400


def redact(value):
    """
    The boundary. Everything entering a span attribute goes through here.

    Truncation is not a nicety — a span carrying a whole document is how a retrieval
    system quietly copies its corpus into an observability vendor.
    """
    if not isinstance(value, str):
        return value
    value = SECRET.sub("[redacted:secret]", value)
    value = EMAIL.sub("[redacted:email]", value)
    if len(value) > MAX_ATTRIBUTE:
        return value[:MAX_ATTRIBUTE] + f"… [+{len(value) - MAX_ATTRIBUTE} chars]"
    return value


@dataclass
class Collected:
    """An in-memory exporter, so a listing can read its own trace back."""
    spans: list[ReadableSpan] = field(default_factory=list)

    def rows(self) -> list[dict]:
        out = []
        for span in sorted(self.spans, key=lambda s: s.start_time):
            attributes = dict(span.attributes or {})
            out.append({
                "name": span.name,
                "span_id": f"{span.context.span_id:016x}",
                "parent": (f"{span.parent.span_id:016x}" if span.parent else None),
                "start": span.start_time,
                "ms": (span.end_time - span.start_time) / 1e6,
                "tokens_in": attributes.get("gen_ai.usage.input_tokens", 0),
                "tokens_out": attributes.get("gen_ai.usage.output_tokens", 0),
                "cost": attributes.get("clarity.cost_usd", 0.0),
                "attributes": attributes,
            })
        return out


class _Exporter(SpanExporter):
    def __init__(self, sink: Collected) -> None:
        self.sink = sink

    def export(self, spans) -> SpanExportResult:
        self.sink.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


_INSTALLED: dict[str, Collected] = {}


def start(service: str = "clarity") -> Collected:
    """
    Install a provider that keeps every span in memory, and return the sink.

    The tracer provider is process-global and set-once: OpenTelemetry refuses to
    replace it, which is correct and catches people out. So the first call installs
    it and later calls hand back the same sink, emptied. A listing that traces two
    runs needs exactly that, and so does a test suite.
    """
    if "sink" not in _INSTALLED:
        sink = Collected()
        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        provider.add_span_processor(SimpleSpanProcessor(_Exporter(sink)))
        trace.set_tracer_provider(provider)
        _INSTALLED["sink"] = sink
    else:
        _INSTALLED["sink"].spans.clear()
    return _INSTALLED["sink"]


def tracer():
    return trace.get_tracer("clarity")


@contextmanager
def span(name: str, **attributes):
    """A span with the redaction boundary applied to everything that enters it."""
    with tracer().start_as_current_span(name) as current:
        for key, value in attributes.items():
            current.set_attribute(key, redact(value))
        yield current


def record_model_call(current, model: str, usage) -> float:
    """
    Attach the semantic-convention attributes for a model call, and the money.

    The `gen_ai.*` names are OpenTelemetry's, not Clarity's. Using them is the whole
    argument of §23.6: a dashboard built on these keys keeps working when you change
    provider, framework, or observability vendor.
    """
    # `rate` is dollars per *million* tokens. Getting this wrong by a factor of a
    # million produces a dashboard nobody questions, because every number on it is
    # wrong in the same direction.
    # An embeddings response has prompt_tokens and no completion_tokens. Reading
    # attributes off a provider's usage object without a default is how a tracing
    # layer takes down the thing it was added to watch.
    tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
    tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
    rates = rate(model)
    cost = 0.0
    if rates:
        cost = (tokens_in * rates[0] + tokens_out * rates[1]) / 1e6
    current.set_attribute("gen_ai.request.model", model)
    if usage:
        current.set_attribute("gen_ai.usage.input_tokens", tokens_in)
        current.set_attribute("gen_ai.usage.output_tokens", tokens_out)
    current.set_attribute("clarity.cost_usd", round(cost, 6))
    return cost


def waterfall(rows: list[dict], width: int = 18) -> str:
    """
    Render a trace the way a person reads one: nesting, time, and what it cost.

    The whole line stays under 70 characters on purpose. A waterfall that wraps in a
    terminal is a waterfall nobody reads, and the first version of this function
    produced hundred-character rows that folded into noise.
    """
    if not rows:
        return "  (no spans)"
    origin = min(r["start"] for r in rows)
    total = max((r["start"] - origin) / 1e6 + r["ms"] for r in rows) or 1.0
    depth = {}
    for row in rows:
        depth[row["span_id"]] = (depth.get(row["parent"], -1) + 1
                                 if row["parent"] in depth else 0)
    lines = []
    for row in rows:
        offset = (row["start"] - origin) / 1e6
        left = int(offset / total * width)
        bar = max(1, int(row["ms"] / total * width))
        label = " " * depth[row["span_id"]] * 2 + row["name"]
        money = f"${row['cost']:.5f}" if row["cost"] else ""
        tokens = (f"{row['tokens_in']:>4}/{row['tokens_out']:<3}"
                  if row["tokens_in"] else " " * 8)
        lines.append(f"  {label[:23]:<23}{' ' * left}{'#' * bar:<{width - left}}"
                     f"{row['ms']:>6.0f}ms {tokens} {money:>8}")
    return "\n".join(lines)
