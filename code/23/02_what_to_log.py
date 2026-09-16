# timeout: 600
# Uses clarity/platform/tracing.py — every attribute entering a span passes
# through redact() on the way in.
# What a naive trace carries out of your building, and what the boundary stops.

import json
import sys

sys.path.insert(0, "code")
from clarity.platform.instrumented import TracedClient, traced_tools  # noqa: E402
from clarity.platform.tracing import MAX_ATTRIBUTE, redact, span, start  # noqa: E402
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import build_tools                      # noqa: E402
from meridian_index import load_index                           # noqa: E402

# A support ticket, as a real one arrives: a person, their address, a key somebody
# pasted in frustration, and a whole document quoted underneath.
TICKET = (
    "From: dana.okafor@halloway-group.example — our API key sk-live-9f2b71c4ae8d55 "
    "stopped working after the migration. Escalating because the Q3 review says: "
    + ("Revenue for 2024 Q3 was $8,461,842 across 5,579 orders, up 4.8% on the prior "
       "quarter. Gross margin was 33.1%. We shipped 102,433 units. " * 6))

print("=== the attribute as the caller passes it ===\n")
print(f"  {len(TICKET):,} characters")
print(f"  {TICKET[:96]}…")

clean = redact(TICKET)
print("\n=== the attribute as it reaches the span ===\n")
print(f"  {len(clean):,} characters")
print(f"  {clean[:96]}…")
print(f"  …{clean[-46:]}")

print("\n  removed:")
for label, needle in (("the email address", "dana.okafor@halloway-group.example"),
                      ("the API key", "sk-live-9f2b71c4ae8d55")):
    print(f"    {label:<20}{'gone' if needle not in clean else 'STILL PRESENT'}")
print(f"    {'the document body':<20}truncated at {MAX_ATTRIBUTE} characters "
      f"({len(TICKET) - len(clean):,} characters dropped)")

# --------------------------------------------------------------- at scale
chunks, vectors = load_index()
sink = start()
client = TracedClient()
tools = traced_tools(build_tools(Retriever(chunks, vectors, client=client),
                                 Warehouse(client=client)))
with span("clarity.request", **{"clarity.question": "What was revenue in 2024 Q3?"}):
    tools[0].run(query="Midwest decline", limit=5)

rows = sink.rows()
attributed = sum(len(str(v)) for r in rows for v in r["attributes"].values())
naive = attributed + sum(len(str(r["attributes"].get("clarity.result_chars", 0)))
                         for r in rows)
result_chars = sum(r["attributes"].get("clarity.result_chars", 0) for r in rows)

print("\n=== one retrieval, measured ===\n")
print(f"  passages returned to the model   {result_chars:,} characters")
print(f"  characters stored on spans       {attributed:,}")
print(f"  ratio                            "
      f"{result_chars / max(attributed, 1):.0f}x smaller")
json.dump({"raw": len(TICKET), "clean": len(clean), "result_chars": result_chars,
           "attributed": attributed, "max_attribute": MAX_ATTRIBUTE},
          open("code/23/_redaction.json", "w"), indent=1)

print()
print("The span records that six passages came back and how many characters they")
print("were. It does not record the passages. That is the whole rule, and it is not")
print("about privacy alone — a trace carrying its own retrieved context is a copy of")
print("your corpus, shipped to a third party, kept for ninety days, readable by")
print("everyone with a dashboard login.")
print()
print("Three things that belong on every model span, and nothing else:")
print()
print("  identity     model, tenant, request id, prompt version")
print("  shape        token counts, result counts, latency, cost")
print("  decisions    which tools were chosen, whether it answered or refused")
print()
print("Three that never do:")
print()
print("  the prompt and the completion, in full")
print("  retrieved documents, rows, or any user content")
print("  anything that arrived in a header")
print()
print("If you need the text to debug — and sometimes you do — put it somewhere with")
print("its own retention and its own access list, and put the *pointer* on the span.")
print("§24.3 builds that, because reproducing a bad run needs the input and the trace")
print("is the wrong place to keep it.")
