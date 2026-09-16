# timeout: 600
# Depends on the Meridian index (meridian_index.py) and the golden set.
# "Similarity is not a confidence score", measured: the top similarity for questions the documents
# answer against questions nothing in the corpus answers, and two relative signals beside it.

import sys

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED                          # noqa: E402
from clarity.evals.runner import load                           # noqa: E402
from meridian_index import load_index                           # noqa: E402

chunks, vectors = load_index()
cases = [c for c in load() if c["kind"] in ("document", "unanswerable")]
answerable = np.array([c["kind"] == "document" for c in cases])
q = np.array([d.embedding for d in OpenAI().embeddings.create(
    model=MODEL_EMBED, input=[c["question"] for c in cases]).data])
q /= np.linalg.norm(q, axis=1, keepdims=True)

ranked = np.sort(q @ vectors.T, axis=1)[:, ::-1]
SIGNALS = {
    "top similarity": ranked[:, 0],
    "margin over second": ranked[:, 0] - ranked[:, 1],
    "top vs next 50 (z)": (ranked[:, 0] - ranked[:, 1:51].mean(axis=1))
                          / ranked[:, 1:51].std(axis=1),
}


def auc(score: np.ndarray) -> float:
    pos, neg = score[answerable], score[~answerable]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def best_threshold(score, mask):
    """The cut that best separates the two groups, as the mean of the two hit rates."""
    s, a = score[mask], answerable[mask]
    return max(np.unique(s), key=lambda t: ((s[a] >= t).mean() + (s[~a] < t).mean()) / 2)


folds = np.arange(len(cases)) % 2
print(f"{answerable.sum()} questions the documents answer, {(~answerable).sum()} that nothing "
      f"in the corpus answers\n")
print(f"  {'':20}{'answerable':>18}{'unanswerable':>18}{'':>5}{'kept':>6}{'refused':>8}")
print(f"  {'signal':20}{'median, range':>18}{'median, range':>18}{'AUC':>6}{'ans.':>6}{'unans.':>8}")
for name, score in SIGNALS.items():
    kept = refused = 0
    for fold in (0, 1):          # choose the cut on half the questions, apply it to the other half
        cut = best_threshold(score, folds == fold)
        test = folds != fold
        kept += np.sum(score[test & answerable] >= cut)
        refused += np.sum(score[test & ~answerable] < cut)
    cells = "".join(f"{np.median(g):.2f}, {g.min():.2f}-{g.max():.2f}".rjust(18)
                    for g in (score[answerable], score[~answerable]))
    print(f"  {name:20}{cells}{auc(score):>6.2f}{kept / answerable.sum():>6.0%}"
          f"{refused / (~answerable).sum():>8.0%}")

top = SIGNALS["top similarity"]
print(f"\na fixed cut at 0.4 kept {np.mean(top[answerable] >= 0.4):.0%} of the answerable questions "
      f"and refused {np.mean(top[~answerable] < 0.4):.0%} of")
print("the unanswerable ones; 'kept' and 'refused' above use a cut chosen on the")
print("other half of the questions.")
unans = [(top[i], cases[i]["question"]) for i in np.flatnonzero(~answerable)]
ans = [(top[i], cases[i]["question"]) for i in np.flatnonzero(answerable)]
print("\nhighest-scoring unanswerable question, and lowest-scoring answerable one:")
for s, text in (max(unans), min(ans)):
    print(f"  {s:.3f}  {text}")
