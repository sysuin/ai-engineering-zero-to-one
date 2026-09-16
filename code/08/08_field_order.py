# timeout: 900
# Strict mode writes fields in the order the schema declares them. Does it matter whether
# the model quotes its evidence before or after it gives the answer?

from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from pydantic import BaseModel, Field

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()
TASK = ("Of all the periods measured in days in this agreement, which is the SHORTEST, and "
        "how many days is it?")


class AnswerFirst(BaseModel):
    days: int
    quote: str = Field(description="The sentence stating that period, copied exactly.")


class QuoteFirst(BaseModel):
    quote: str = Field(description="The sentence stating that period, copied exactly.")
    days: int


def truth(text: str) -> int:
    import re
    return min(int(n) for n in re.findall(r"(\d+) days", text))


def ask(schema, text: str) -> bool:
    parsed = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, response_format=schema, max_completion_tokens=300,
        messages=[{"role": "user", "content": f"{TASK}\n\n<contract>\n{text}\n</contract>"}],
    ).choices[0].message.parsed
    return parsed is not None and parsed.days == truth(text)


print(f"{len(contracts)} contracts, each with five periods in days\n")
results = {}
for schema in (AnswerFirst, QuoteFirst):
    with ThreadPoolExecutor(max_workers=12) as pool:
        right = list(pool.map(lambda c: ask(schema, c["text"]), contracts))
    results[schema.__name__] = sum(right) / len(right)
    print(f"  {schema.__name__:12} {results[schema.__name__]:.1%}")

gap = results["QuoteFirst"] - results["AnswerFirst"]
print(f"\nquote first minus answer first: {100 * gap:+.1f} points on {len(contracts)} contracts")
