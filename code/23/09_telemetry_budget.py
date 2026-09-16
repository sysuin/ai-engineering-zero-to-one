# Traces cost storage, and the bill follows from four numbers: requests, spans per request,
# bytes per span, and how long you keep them. The span shape is Clarity's own trace from 01.

import json
import sys

sys.path.insert(0, "code")
from clarity.platform.tracing import span, start                  # noqa: E402

trace = json.load(open("code/23/_trace.json"))["deep"]
sink = start("clarity")

# Rebuild the same tree of spans with the attributes Clarity records, then measure them.
by_id = {row["span_id"]: row for row in trace}
children: dict[str | None, list[dict]] = {}
for row in trace:
    children.setdefault(row["parent"], []).append(row)


def emit(row: dict) -> None:
    attributes = {}
    if row["name"].startswith("model."):
        attributes = {"gen_ai.request.model": "model-name", "gen_ai.usage.input_tokens":
                      row["tokens_in"], "gen_ai.usage.output_tokens": row["tokens_out"],
                      "clarity.cost_usd": row["cost"]}
    elif row["name"].startswith("tool."):
        attributes = {"clarity.tool": row["name"][5:], "clarity.arguments.chars": 64,
                      "clarity.result.chars": 1800}
    with span(row["name"], **attributes):
        for child in children.get(row["span_id"], []):
            emit(child)


for root in children[None]:
    emit(root)
spans = len(sink.spans)
bytes_per_span = sum(len(s.to_json(indent=None)) for s in sink.spans) / spans
print(f"one Clarity request: {spans} spans, {bytes_per_span:,.0f} bytes each as the SDK "
      "writes them in JSON")
print("(exporters usually send protobuf, which is smaller; the arithmetic is the same)\n")

RETENTION_DAYS = 30
print(f"  {'requests a day':>15} {'keep everything':>16} {'tail-sampled':>13}   "
      f"(stored for {RETENTION_DAYS} days)")
for per_day in (1_000, 50_000, 1_000_000):
    everything = per_day * spans * bytes_per_span * RETENTION_DAYS / 1e9
    # Keep every failure and slow request (say 5%), and one in ten of the rest.
    sampled = everything * (0.05 + 0.95 * 0.10)
    print(f"  {per_day:>15,} {everything:>13,.2f} GB {sampled:>10,.2f} GB")

print("\nThe bill scales with spans per request as much as with traffic: an agent that takes")
print("twice the steps writes twice the telemetry. A retention policy and a tail-sampling")
print("rule are decided with this table in front of you, not after the first invoice.")
