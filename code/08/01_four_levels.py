# timeout: 900
# You need a record. The model writes text. Four ways of closing that gap.

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()

WANTED = {"payment_days", "payment_basis", "price_cap_pct", "price_notice_days"}

ASK = ("Extract the payment terms and price-adjustment terms from this supply "
       "agreement.\n\n{text}")

SCHEMA = {
    "type": "object",
    "properties": {
        "payment_days": {"type": "integer"},
        "payment_basis": {"type": "string", "enum": ["invoice", "receipt"]},
        "price_cap_pct": {"type": "integer"},
        "price_notice_days": {"type": "integer"},
    },
    "required": sorted(WANTED),
    "additionalProperties": False,
}


def level_1_just_ask(text: str):
    return client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=300,
        messages=[{"role": "user", "content": ASK.format(text=text)}],
    ).choices[0].message.content


def level_2_ask_for_json(text: str):
    return client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=300,
        messages=[{"role": "user", "content": ASK.format(text=text) +
                   "\n\nReply with JSON only, using the keys "
                   f"{sorted(WANTED)}."}],
    ).choices[0].message.content


def level_3_json_mode(text: str):
    return client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=300,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": ASK.format(text=text) +
                   "\n\nReply with a JSON object."}],
    ).choices[0].message.content


def level_4_strict_schema(text: str):
    return client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=300,
        response_format={"type": "json_schema", "json_schema": {
            "name": "contract_terms", "strict": True, "schema": SCHEMA}},
        messages=[{"role": "user", "content": ASK.format(text=text)}],
    ).choices[0].message.content


LEVELS = {
    "1. just ask":            level_1_just_ask,
    "2. ask for JSON":        level_2_ask_for_json,
    "3. JSON mode":           level_3_json_mode,
    "4. strict schema":       level_4_strict_schema,
}


def inspect(raw: str | None) -> tuple[bool, bool, bool]:
    """Did it parse, are the keys exactly right, are the values the right types?"""
    if raw is None:
        return False, False, False
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False, False, False
    if not isinstance(obj, dict):
        return True, False, False
    keys_ok = set(obj) == WANTED
    types_ok = keys_ok and all(
        isinstance(obj[k], int) for k in
        ("payment_days", "price_cap_pct", "price_notice_days")) and \
        obj.get("payment_basis") in ("invoice", "receipt")
    return True, keys_ok, types_ok


results = {}
print(f"{'':22} {'parses':>8} {'right keys':>12} {'right types':>13}")
for label, fn in LEVELS.items():
    with ThreadPoolExecutor(max_workers=12) as pool:
        raws = list(pool.map(lambda c: fn(c["text"]), contracts))
    flags = [inspect(r) for r in raws]
    n = len(flags)
    row = {"parses": sum(f[0] for f in flags) / n,
           "keys": sum(f[1] for f in flags) / n,
           "types": sum(f[2] for f in flags) / n,
           "example": (raws[0] or "")[:90].replace("\n", " ")}
    results[label] = row
    print(f"{label:22} {row['parses']:>7.0%} {row['keys']:>12.0%} {row['types']:>13.0%}")

print("\nWhat each one actually returned, first contract:")
for label, row in results.items():
    print(f"  {label:22} {row['example']!r}")

Path("code/08/_four_levels.json").write_text(json.dumps(results, indent=2))
print(f"\n{len(contracts)} contracts. Only the last row is safe to hand to a program.")
