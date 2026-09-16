# When the answer key is wrong too. The three approaches from the bake-off, scored against
# labels with a known share of mistakes added — first at random, then where the model errs.
# No model is called: the answers are the ones 02_rules_vs_model.py recorded.

import json
import random
import re
from pathlib import Path

CATEGORIES = ["Delivery", "Quality", "Billing", "Returns", "Account"]
K = len(CATEGORIES)
bake = json.loads(Path("code/06/_bake_off.json").read_text())
rows = [json.loads(line) for line in
        Path("data/meridian/documents/tickets/tickets.jsonl").read_text().splitlines()][:200]
truth, model = bake["truth"], bake["model_answers"]

RULES = [                                 # the second draft, from 03_hybrid.py
    ("Returns",  r"return|collection|RMA|wrong size|send back"),
    ("Account",  r"second delivery address|remove |add a |pricing|account"),
    ("Billing",  r"invoice|charg|credit note|tax|vat|billed|payment|PO number"),
    ("Quality",  r"defect|damag|broken|falling apart|tearing|not the same|thinner|bad"),
    ("Delivery", r"deliver|arriv|late|missing|shipment|where is|dispatch|turned up"),
]


def by_rule(text: str) -> str | None:
    return next((c for c, p in RULES if re.search(p, text, re.IGNORECASE)), None)


rules = [by_rule(r["body"]) for r in rows]
SYSTEMS = {
    "rules, first draft": bake["rule_answers"],
    "the model": model,
    "rules + model": [r if r is not None else m for r, m in zip(rules, model)],
    "a perfect system": truth,
}


def accuracy(answers: list, labels: list) -> float:
    return sum(a == t for a, t in zip(answers, labels)) / len(labels)


def with_random_noise(rate: float, rng: random.Random) -> list[str]:
    labels = list(truth)
    for i in rng.sample(range(len(labels)), round(rate * len(labels))):
        others = [c for c in CATEGORIES if c != labels[i]]
        labels[i] = rng.choice(others)
    return labels


def expected(true_acc: float, rate: float) -> float:
    # A right answer scores if its label was not flipped; a wrong one
    # scores only if the flip happened to land on it.
    return true_acc * (1 - rate) + (1 - true_acc) * rate / (K - 1)


rng = random.Random(6)
DRAWS = 2_000
print("Accuracy against labels with a share flipped at random (mean of 2,000 draws)\n")
print(f"  {'system':<20} {'clean':>7} {'5% wrong':>9} {'10% wrong':>10} {'20% wrong':>10}")
noisy = {rate: [with_random_noise(rate, rng) for _ in range(DRAWS)] for rate in (0.05, 0.10, 0.20)}
for name, answers in SYSTEMS.items():
    cells = [f"{accuracy(answers, truth):>7.1%}"]
    for rate, width in ((0.05, 9), (0.10, 10), (0.20, 10)):
        mean = sum(accuracy(answers, labels) for labels in noisy[rate]) / DRAWS
        cells.append(f"{mean:>{width}.1%}")
    print(f"  {name:<20} {' '.join(cells)}")

gap_clean = accuracy(SYSTEMS["rules + model"], truth) - accuracy(model, truth)
shrink = 1 - 0.10 - 0.10 / (K - 1)
print(f"\n  formula at 10% wrong: the model {expected(accuracy(model, truth), 0.10):.1%}, "
      f"a perfect system {expected(1.0, 0.10):.1%}")
print(f"  the {gap_clean * 100:.1f}-point gap between hybrid and model shrinks by a")
print(f"  factor of {shrink:.3f}, to {gap_clean * shrink * 100:.1f} points")

# The same number of wrong labels, placed where a labeller who reads like the model would put
# them: on tickets the model got wrong, relabelled with the model's answer.
model_wrong = [i for i, (m, t) in enumerate(zip(model, truth)) if m != t and m in CATEGORIES]
model_right = [i for i, (m, t) in enumerate(zip(model, truth)) if m == t]
n = 20                                                          # 10% of the labels
print(f"\n{n} wrong labels (10%), placed three ways: the model's accuracy\n")
print(f"  {'wrong labels placed':<32} {'measured':>8}  {'corrected for 10%':>17}")
placements = {
    "at random": None,
    "copying the model's mistakes": model_wrong[:n],
    "on tickets the model had right": model_right[:n],
}
for label, where in placements.items():
    if where is None:
        score = sum(accuracy(model, labels) for labels in noisy[0.10]) / DRAWS
    else:
        labels = list(truth)
        for i in where:
            labels[i] = (model[i] if model[i] != truth[i]
                         else next(c for c in CATEGORIES if c != truth[i]))
        score = accuracy(model, labels)
    corrected = (score - 0.10 / (K - 1)) / (1 - 0.10 - 0.10 / (K - 1))
    print(f"  {label:<32} {score:>8.1%}  {corrected:>17.1%}")
print(f"\n  the model's accuracy against the clean labels: {accuracy(model, truth):.1%}")
