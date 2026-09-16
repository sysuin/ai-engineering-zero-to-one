# timeout: 600
# "The first call with a new schema can be slower while it is compiled." Ten schemas the provider
# has never seen, each called three times in a row: is the first call slower?

import statistics
import time
import uuid

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
TEXT = "Invoice INV-73451 for 1,240.50 was issued on 3 March to Kestrel Supply, payable in 45 days."


def schema() -> dict:
    """A realistic little schema whose field names nobody has used before."""
    tag = uuid.uuid4().hex[:8]
    properties = {f"invoice_{tag}": {"type": "string"}, f"amount_{tag}": {"type": "number"},
                  f"days_{tag}": {"type": "integer"},
                  f"kind_{tag}": {"type": "string", "enum": ["invoice", "credit_note", "receipt"]},
                  f"lines_{tag}": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                   "required": ["sku", "qty"],
                                   "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}}}}}
    return {"type": "json_schema", "json_schema": {"name": f"s_{tag}", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}}}


def timed(fmt: dict) -> float:
    started = time.perf_counter()
    client.chat.completions.create(model=MODEL_FAST, temperature=0, max_completion_tokens=200,
                                   response_format=fmt, messages=[{"role": "user", "content": TEXT}])
    return time.perf_counter() - started


firsts, seconds, thirds = [], [], []
for _ in range(10):
    fmt = schema()
    firsts.append(timed(fmt))
    seconds.append(timed(fmt))
    thirds.append(timed(fmt))

print("ten new strict schemas, each called three times in a row\n")
for label, values in (("first call", firsts), ("second call", seconds), ("third call", thirds)):
    print(f"  {label:12} median {statistics.median(values):.2f}s   range {min(values):.2f}–{max(values):.2f}s")
gap = statistics.median(firsts) - statistics.median(seconds + thirds)
spread = statistics.median([max(v) - min(v) for v in zip(seconds, thirds)])
print(f"\nfirst minus later, medians: {gap:+.2f}s; typical difference between two later calls: {spread:.2f}s")
print("-> " + ("the first call was measurably slower" if gap > 2 * spread else
               "no first-call penalty larger than the ordinary variation between calls"))
