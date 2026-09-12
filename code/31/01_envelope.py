# timeout: 300
# Back of an envelope: volume, tokens, cost, latency and storage, before any code.

import json
from dataclasses import dataclass, field

# Rates in dollars per million tokens, and embedding per million. Placeholders, said
# out loud: this book prints no price list, and Appendix C says where to find yours.
RATE_IN, RATE_OUT, RATE_EMBED = 0.15, 0.60, 0.02
BYTES_PER_VECTOR = 1536 * 4          # float32, before any quantisation


@dataclass
class Design:
    name: str
    requests_per_day: int
    model_calls_per_request: float
    input_tokens_per_call: int
    output_tokens_per_call: int
    documents: int
    chunks_per_document: int
    seconds_per_call: float
    peak_multiplier: float = 3.0     # a day's traffic does not arrive evenly

    def tokens(self) -> tuple[int, int]:
        calls = self.requests_per_day * self.model_calls_per_request
        return (int(calls * self.input_tokens_per_call),
                int(calls * self.output_tokens_per_call))

    def daily_cost(self) -> float:
        tin, tout = self.tokens()
        return (tin * RATE_IN + tout * RATE_OUT) / 1e6

    def index(self) -> dict:
        chunks = self.documents * self.chunks_per_document
        return {"chunks": chunks,
                "vector_gb": chunks * BYTES_PER_VECTOR / 1e9,
                "embed_cost": chunks * 400 * RATE_EMBED / 1e6}

    def concurrency(self) -> float:
        """Little's law: in-flight work is arrival rate times time in system."""
        per_second = self.requests_per_day / 86_400 * self.peak_multiplier
        return per_second * (self.model_calls_per_request * self.seconds_per_call)

    def latency(self) -> float:
        return self.model_calls_per_request * self.seconds_per_call


DESIGNS = [
    Design("a support copilot over ten years of tickets",
           requests_per_day=4_000, model_calls_per_request=3,
           input_tokens_per_call=3_000, output_tokens_per_call=250,
           documents=900_000, chunks_per_document=3, seconds_per_call=1.8),
    Design("a nightly contract-risk enricher",
           requests_per_day=50_000, model_calls_per_request=2,
           input_tokens_per_call=6_000, output_tokens_per_call=400,
           documents=50_000, chunks_per_document=25, seconds_per_call=2.0,
           peak_multiplier=1.0),
    Design("a regulated advisory assistant",
           requests_per_day=300, model_calls_per_request=6,
           input_tokens_per_call=4_000, output_tokens_per_call=300,
           documents=20_000, chunks_per_document=8, seconds_per_call=2.2),
]

print("Three requirements, sized before anything is built.\n")
rows = {}
for design in DESIGNS:
    tin, tout = design.tokens()
    index = design.index()
    rows[design.name] = {
        "tokens_in": tin, "tokens_out": tout,
        "daily_cost": design.daily_cost(),
        "monthly_cost": design.daily_cost() * 30,
        "per_request": design.daily_cost() / design.requests_per_day,
        "chunks": index["chunks"], "vector_gb": index["vector_gb"],
        "embed_cost": index["embed_cost"],
        "concurrency": design.concurrency(), "latency": design.latency()}
    row = rows[design.name]
    print(f"  {design.name}")
    print(f"    {design.requests_per_day:,} requests/day  x  "
          f"{design.model_calls_per_request:g} calls  =  "
          f"{tin / 1e6:.1f}M in, {tout / 1e6:.2f}M out")
    print(f"    cost      ${row['daily_cost']:,.2f}/day   "
          f"${row['monthly_cost']:,.0f}/month   "
          f"${row['per_request']:.4f}/request")
    print(f"    index     {row['chunks']:,} chunks   "
          f"{row['vector_gb']:.1f} GB of vectors   "
          f"${row['embed_cost']:,.0f} to build once")
    print(f"    shape     {row['latency']:.1f}s per request, "
          f"{row['concurrency']:.1f} concurrent at peak")
    print()

json.dump({"rates": {"in": RATE_IN, "out": RATE_OUT, "embed": RATE_EMBED},
           "designs": rows}, open("code/31/_envelope.json", "w"), indent=1)

support, nightly, regulated = (rows[d.name] for d in DESIGNS)
biggest = max(r["concurrency"] for r in rows.values())
print("Read what each number decides, because that is the point of doing this before")
print("drawing anything.")
print()
print(f"Start with the shape column, because it is the one that surprises people. The")
print(f"busiest of these three needs {biggest:.1f} requests in flight at peak. Not "
      f"thirty. Not a")
print("cluster. One container with a warm index would serve all three of these")
print("systems at once and spend most of its day idle.")
print()
print("That is the normal case for internal software, and sizing is how you find out")
print("before you have designed for a load you will never see. The architecture that")
print("wins here is the boring one, and the argument for it is arithmetic.")
print()
print(f"Then read cost. The nightly enricher is ${nightly['monthly_cost']:,.0f} a "
      f"month and the regulated")
print(f"assistant is ${regulated['monthly_cost']:,.0f}. Those two systems have "
      f"nothing in common: one is a")
print("throughput problem where a batch API and a queue are the whole design, and the")
print("other is a governance problem where cost will never be the argument. Knowing")
print("which you have, before the meeting, is most of what this exercise buys.")
print()
print(f"And read the index separately. The support copilot's "
      f"{support['chunks'] / 1e6:.1f}M chunks are "
      f"{support['vector_gb']:.0f}GB of")
print("vectors — a fixed cost, a storage decision, and usually where the surprise is.")
print("It is also the number that decides whether §12.2's approximate index is")
print("necessary or premature.")
print()
print("Three habits that make an envelope useful rather than decorative:")
print()
print("  size the index separately     it is fixed cost and storage, not throughput")
print("  use Little's law for shape    concurrency is arrival rate times duration,")
print("                                and it selects the deployment target")
print("  put the per-request cost      a monthly total is an argument; a per-request")
print("  beside the monthly one        figure is a design constraint")
print()
print("(The rates above are placeholders. Substitute your provider's and every number")
print("becomes yours — the arithmetic does not change.)")
