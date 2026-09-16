# timeout: 900
# The same four examples, in different orders, and a set that leaves a label out.

from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()
DEMOS, TEST = contracts[:8], contracts[8:]
INSTRUCTION = ("Read the supply agreement and output its payment term in Meridian's "
               "internal shorthand. Reply with the code only.")
by_code = {}
for demo in DEMOS:
    by_code.setdefault(demo["payment_code"].split("-")[1], []).append(demo)
inv, rcp = by_code["INV"], by_code["RCP"]

SETS = {
    "2 INV, 2 RCP, receipt last":  [inv[0], inv[1], rcp[0], rcp[1]],
    "2 INV, 2 RCP, receipt first": [rcp[0], rcp[1], inv[0], inv[1]],
    "2 INV, 2 RCP, alternating":   [inv[0], rcp[0], inv[1], rcp[1]],
    "4 INV, no receipt example":   inv[:4],
}


def ask(demos: list[dict], contract: dict) -> str:
    messages = [{"role": "system", "content": INSTRUCTION}]
    for demo in demos:
        messages += [{"role": "user", "content": demo["text"]},
                     {"role": "assistant", "content": demo["payment_code"]}]
    messages.append({"role": "user", "content": contract["text"]})
    return (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, messages=messages, max_completion_tokens=32,
    ).choices[0].message.content or "").strip()


rcp_tests = [c for c in TEST if c["payment_code"].endswith("RCP")]
print(f"{len(TEST)} held-out contracts, {len(rcp_tests)} of them receipt-based\n")
print(f"  {'examples':30} {'all':>6} {'receipt-based only':>19}")
accuracies = {}
for name, demos in SETS.items():
    with ThreadPoolExecutor(max_workers=12) as pool:
        answers = list(pool.map(lambda c: ask(demos, c), TEST))
    right = [a == c["payment_code"] for a, c in zip(answers, TEST)]
    rcp_right = [ok for ok, c in zip(right, TEST) if c["payment_code"].endswith("RCP")]
    accuracies[name] = sum(right) / len(right)
    print(f"  {name:30} {sum(right) / len(right):>6.0%} {sum(rcp_right) / len(rcp_right):>19.0%}")

orders = [v for k, v in accuracies.items() if k.startswith("2 INV")]
print(f"\nspread across the three orders of the same examples: "
      f"{100 * (max(orders) - min(orders)):.0f} points")
print(f"leaving the receipt label out entirely: "
      f"{100 * (accuracies['4 INV, no receipt example'] - max(orders)):+.0f} points against the best order")
