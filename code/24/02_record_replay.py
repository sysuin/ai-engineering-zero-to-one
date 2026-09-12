# timeout: 1200
# Uses clarity/platform/replay.py — a cassette keyed by a canonical form of each
# each request, so a replay is deterministic by construction.
# Record a run once. Replay it as often as you like, for nothing.

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "code")
from clarity.platform.replay import Cassette, RecordingClient  # noqa: E402
from clarity.v0_6.retrieve import Retriever                    # noqa: E402
from clarity.v0_7.warehouse import Warehouse                   # noqa: E402
from clarity.v0_8.tools import build_tools                     # noqa: E402
from clarity.v0_9.agent import Agent, Budget                   # noqa: E402
from meridian_index import load_index                          # noqa: E402

TAPE = Path("data/meridian/cassettes/midwest.json")
TAPE.unlink(missing_ok=True)
QUESTION = "What was revenue in 2024 Q3, and what caused the Midwest decline?"
SYSTEM = "You are an analyst for Meridian. Use tools for every fact."

chunks, vectors = load_index()


def run(mode: str) -> tuple[str, float, int, Cassette]:
    cassette = Cassette(TAPE)
    client = RecordingClient(cassette, mode=mode)
    tools = build_tools(Retriever(chunks, vectors, client=client),
                        Warehouse(client=client))
    started = time.perf_counter()
    result = Agent(tools, client=client, budget=Budget(steps=6)).run(QUESTION,
                                                                    system=SYSTEM)
    elapsed = time.perf_counter() - started
    if mode == "record":
        cassette.save()
    return result.answer, elapsed, result.tokens, cassette


print("--- recording ---")
answer, seconds, tokens, tape = run("record")
print(f"  {seconds:5.1f}s   {tokens:,} tokens billed   "
      f"{len(tape)} exchanges written to the cassette")
print(f"  {' '.join(answer.split())[:66]}")
print(f"  (whatever the run did, replay reproduces it — including a "
      f"refusal)")

print("\n--- replaying, three times ---")
replays = []
for n in range(3):
    again, seconds, tokens, tape = run("replay")
    replays.append(again)
    print(f"  {seconds:5.2f}s   0 tokens billed   "
          f"{tape.hits} cassette hits, {tape.misses} misses")

identical = all(r == answer for r in replays)
print(f"\n  byte-identical to the recorded answer: "
      f"{'yes, all three' if identical else 'NO'}")

# A changed input is a cassette miss, and it should be loud.
print("\n--- the same harness, one word of the question changed ---")
cassette = Cassette(TAPE)
client = RecordingClient(cassette, mode="replay")
tools = build_tools(Retriever(chunks, vectors, client=client),
                    Warehouse(client=client))
try:
    Agent(tools, client=client, budget=Budget(steps=6)).run(
        QUESTION.replace("Midwest", "Northeast"), system=SYSTEM)
    print("  replayed anyway — which would be a bug in the cassette")
except Exception as error:                                     # noqa: BLE001
    print(f"  {type(error).__name__}: {str(error).splitlines()[0][:66]}")

json.dump({"exchanges": len(tape), "identical": identical,
           "recorded_tokens": tokens, "answer": answer[:300]},
          open("code/24/_replay.json", "w"), indent=1)

print()
print("That is the whole of it, and it changes what debugging costs.")
print()
print("A failing run recorded once can be replayed on a colleague's laptop, in a")
print("test, in CI, and at three in the morning, forever, without a key and without")
print("a bill. It is also the only way to make a change to your *code* and know that")
print("a difference in behaviour came from the change rather than from the model")
print("having a different afternoon — §24.1 measured how often that happens.")
print()
print("Two limits, and both matter.")
print()
print("A cassette cannot test a prompt change. Change the prompt and the request no")
print("longer matches, which is exactly what the last block shows: a miss, loudly,")
print("rather than a plausible answer to a question you did not ask. That is the")
print("right behaviour and it is worth checking your recording library does it.")
print()
print("And a cassette is a recording of a system's *inputs*, so it goes stale the")
print("moment the corpus changes. Keep them next to the regression cases they")
print("support, and regenerate them when the case stops being about what it was.")
