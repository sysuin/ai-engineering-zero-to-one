# timeout: 300
# Back of an envelope: volume, tokens, cost, latency and storage, before any code.

import json
from dataclasses import dataclass

# Dollars per million tokens, input and output, for a fast model and a smart one, and for
# embeddings. Placeholders, said out loud: this book prints no price list, and Appendix C
# says where to find yours. What matters is the ratio between the tiers, not the digits.
RATES = {"fast": (0.15, 0.60), "smart": (2.00, 8.00)}
RATE_EMBED = 0.02
TOKENS_PER_CHUNK = 400
BYTES_PER_VECTOR = 1536 * 4          # float32, before any quantisation or graph overhead


@dataclass
class Design:
    name: str
    requests_per_day: int
    model_calls_per_request: float
    input_tokens_per_call: int
    output_tokens_per_call: int
    documents: int                   # 0: nothing is retrieved, each item is read whole
    chunks_per_document: int
    seconds_per_call: float
    tier: str = "fast"
    window_hours: float = 24.0       # the hours the day's requests arrive in
    peak_multiplier: float = 3.0     # and how much busier the busiest hour is

    def tokens(self) -> tuple[int, int]:
        calls = self.requests_per_day * self.model_calls_per_request
        return (int(calls * self.input_tokens_per_call),
                int(calls * self.output_tokens_per_call))

    def daily_cost(self) -> float:
        rate_in, rate_out = RATES[self.tier]
        tin, tout = self.tokens()
        return (tin * rate_in + tout * rate_out) / 1e6

    def index(self) -> dict:
        chunks = self.documents * self.chunks_per_document
        return {"chunks": chunks,
                "vector_gb": chunks * BYTES_PER_VECTOR / 1e9,
                "embed_cost": chunks * TOKENS_PER_CHUNK * RATE_EMBED / 1e6}

    def latency(self) -> float:
        """Calls made one after another. Calls that could run in parallel would not add."""
        return self.model_calls_per_request * self.seconds_per_call

    def concurrency(self) -> float:
        """Little's law: in-flight work is arrival rate times time in system."""
        per_second = (self.requests_per_day / (self.window_hours * 3600)
                      * self.peak_multiplier)
        return per_second * self.latency()


DESIGNS = [
    Design("a support copilot over ten years of tickets",
           requests_per_day=4_000, model_calls_per_request=3,
           input_tokens_per_call=3_000, output_tokens_per_call=250,
           documents=900_000, chunks_per_document=3, seconds_per_call=1.8),
    Design("a nightly contract-risk enricher",
           requests_per_day=50_000, model_calls_per_request=2,
           input_tokens_per_call=6_000, output_tokens_per_call=400,
           documents=0, chunks_per_document=0, seconds_per_call=2.0,
           window_hours=8, peak_multiplier=1.0),
    Design("a regulated advisory assistant",
           requests_per_day=300, model_calls_per_request=6,
           input_tokens_per_call=4_000, output_tokens_per_call=300,
           documents=20_000, chunks_per_document=8, seconds_per_call=2.2,
           tier="smart"),
    Design("an IT helpdesk agent that can act",
           requests_per_day=1_500, model_calls_per_request=5,
           input_tokens_per_call=5_000, output_tokens_per_call=300,
           documents=3_000, chunks_per_document=6, seconds_per_call=2.0),
    Design("meeting summaries inside a video product",
           requests_per_day=200_000, model_calls_per_request=2,
           input_tokens_per_call=12_000, output_tokens_per_call=600,
           documents=0, chunks_per_document=0, seconds_per_call=8.0),
]

print("Five requirements, sized before anything is built.\n")
rows = {}
for design in DESIGNS:
    tin, tout = design.tokens()
    index = design.index()
    rows[design.name] = {
        "tier": design.tier,
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
          f"{design.model_calls_per_request:g} {design.tier} calls  =  "
          f"{tin / 1e6:,.1f}M in, {tout / 1e6:,.2f}M out")
    print(f"    cost      ${row['daily_cost']:,.2f}/day   "
          f"${row['monthly_cost']:,.0f}/month   "
          f"${row['per_request']:.4f}/request")
    if row["chunks"]:
        print(f"    index     {row['chunks']:,} chunks   "
              f"{row['vector_gb']:.1f} GB of vectors   "
              f"${row['embed_cost']:,.2f} to embed once")
    else:
        print("    index     none: each item is read whole")
    print(f"    shape     {row['latency']:.1f}s per request, "
          f"{row['concurrency']:.1f} in flight at peak")
    print()

json.dump({"rates": RATES, "embed": RATE_EMBED, "designs": rows},
          open("code/31/_envelope.json", "w"), indent=1)

support, nightly, regulated, helpdesk, meetings = (rows[d.name] for d in DESIGNS)
ranked = sorted(rows.items(), key=lambda kv: kv[1]["concurrency"])
small = [name for name, row in ranked if row["concurrency"] < 10]
busiest_name, busiest = ranked[-1]
second_name, second = ranked[-2]

print("Read what each number decides.")
print()
print(f"Shape first. {len(small)} of the {len(rows)} designs have fewer than ten requests "
      f"in flight at")
print(f"their busiest; the largest of those is {second['concurrency']:.1f}. One process "
      f"with a warm index")
print("would serve any of them, and spend most of its day idle.")
print(f"The exception is {busiest_name}, at "
      f"{busiest['concurrency']:.0f} in flight:")
print("a hundred calls open at once is a rate-limit question, a pooling question and a")
print("failure-domain question before it is a hardware one. A different architecture,")
print("found by arithmetic rather than by an outage.")
print()
print(f"Then cost. The same fast-model rates give ${helpdesk['monthly_cost']:,.0f} a month "
      f"for the helpdesk and")
print(f"${meetings['monthly_cost']:,.0f} for the meeting summaries. The regulated "
      f"assistant pays")
print(f"{RATES['smart'][0] / RATES['fast'][0]:.0f}x the input rate for a smart model and "
      f"still costs ${regulated['monthly_cost']:,.0f} a month —")
print(f"${regulated['per_request']:.2f} a question, which is not the argument in that "
      f"design.")
print()
changed = 0.01
incremental = nightly["monthly_cost"] * changed
print(f"The enricher's ${nightly['monthly_cost']:,.0f} a month assumes every contract is "
      f"re-read every night.")
print(f"If {changed:.0%} of contracts are new or changed on a given night, reading only "
      f"those costs")
print(f"${incremental:,.0f} a month; at a batch discount of half, "
      f"${incremental / 2:,.0f}. The envelope asked a")
print("question the requirement did not: does anything need re-reading at all?")
print()
print(f"And the index separately. The support copilot's {support['chunks'] / 1e6:.1f}M "
      f"chunks are {support['vector_gb']:.1f} GB")
print("of vectors before any graph overhead — a storage decision, a startup time, and")
print("the number that decides whether §12.2's approximate index is needed.")
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
