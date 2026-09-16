# A judge with known error rates still gives a biased pass rate. If you know how often it
# passes wrong answers and fails right ones, you can correct for both — and see when the
# correction is not worth trusting. Reads code/22/_judge.json, written by 01.

import json
from math import sqrt

d = json.load(open("code/22/_judge.json"))
labels = d["labels"]
CASES = 120                                   # the size of Chapter 21's golden set


def rates(verdicts: list[int]) -> tuple[float, float]:
    """Sensitivity: right answers passed. Specificity: wrong answers failed."""
    pos = [v for v, l in zip(verdicts, labels) if l]
    neg = [v for v, l in zip(verdicts, labels) if not l]
    return sum(pos) / len(pos), 1 - sum(neg) / len(neg)


def corrected(observed: float, sensitivity: float, specificity: float) -> float | None:
    """The Rogan–Gladen estimate of the true pass rate from the judge's pass rate."""
    denominator = sensitivity + specificity - 1
    if denominator <= 0:
        return None
    return min(1.0, max(0.0, (observed + specificity - 1) / denominator))


for name in ("no reference", "with the reference"):
    sens, spec = rates(d[name]["votes"])
    print(f"{name} judge: passes {sens:.0%} of right answers, fails {spec:.0%} of wrong ones")
    print(f"  informedness (sensitivity + specificity - 1): {sens + spec - 1:.2f}")
    for true_rate in (0.70, 0.85):
        observed = true_rate * sens + (1 - true_rate) * (1 - spec)
        estimate = corrected(observed, sens, spec)
        shown = "cannot be recovered" if estimate is None else f"{estimate:.0%}"
        print(f"  a system truly right {true_rate:.0%} of the time would score "
              f"{observed:.0%}; corrected: {shown}")
    # How much one judging error moves the corrected estimate, per point of observed rate.
    if sens + spec - 1 > 0:
        gain = 1 / (sens + spec - 1)
        observed = 0.85 * sens + 0.15 * (1 - spec)
        noise = 1.96 * sqrt(observed * (1 - observed) / CASES)
        print(f"  each point of observed pass rate moves the estimate {gain:.1f} points;")
        print(f"  on {CASES} cases the observed rate is uncertain by about ±{noise:.0%}, "
              f"so the estimate by ±{min(1.0, noise * gain):.0%}")
    print()
