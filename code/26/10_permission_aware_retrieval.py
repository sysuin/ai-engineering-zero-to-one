# Permissions inside a tenant, measured on Meridian's index. Contracts are readable by legal
# and finance, tickets by support, and the reviews by everyone. A sales analyst and a lawyer ask
# the same questions: what does each way of enforcing the access list return, and what does a
# cache keyed only on the question hand the analyst?

import sys

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED                          # noqa: E402
from clarity.evals.runner import load                           # noqa: E402
from meridian_index import load_index                           # noqa: E402

K = 5
chunks, vectors = load_index()
AUDIENCE = {"contract": {"legal", "finance"}, "ticket": {"support"}}   # everything else: all
access = [AUDIENCE.get(c["kind"], {"all"}) for c in chunks]
USERS = {"sales analyst": {"all"}, "lawyer": {"all", "legal"}}

cases = load()
GROUPS = {
    "contracts": [c["question"] for c in cases if (c.get("source") or "").startswith("contract")][:6],
    "reviews": [c["question"] for c in cases if (c.get("source") or "").startswith("qbr")][:6],
    # questions that tickets and the reviews' commentary can both speak to
    "support": ["What drove the rise in support contacts in 2024 Q4, and which supplier was involved?",
                "What were customers complaining about in late 2024?",
                "Which products had quality complaints?",
                "Why were orders arriving late?",
                "What problems did customers report with deliveries?",
                "Were there billing disputes, and what caused them?"],
}
questions = [text for group in GROUPS.values() for text in group]
q = np.array([d.embedding for d in OpenAI().embeddings.create(model=MODEL_EMBED,
                                                              input=questions).data])
scores = dict(zip(questions, q @ vectors.T))


def readable(i: int, groups: set[str]) -> bool:
    return bool(access[i] & groups)


def pre_filter(row: np.ndarray, groups: set[str]) -> list[int]:
    """A filter inside the search: rank only what this user may read."""
    allowed = [i for i in np.argsort(-row) if readable(i, groups)]
    return allowed[:K]


def post_hide(row: np.ndarray, groups: set[str]) -> list[int]:
    """Search everything, then hide what this user may not read."""
    return [i for i in np.argsort(-row)[:K] if readable(i, groups)]


analyst, lawyer = USERS["sales analyst"], USERS["lawyer"]
cache = {text: pre_filter(scores[text], lawyer) for text in questions}   # the lawyer asked first
METHODS = (("filter inside the search", lambda text: pre_filter(scores[text], analyst)),
           ("search, then hide", lambda text: post_hide(scores[text], analyst)),
           ("cache keyed on the question", lambda text: cache[text]))

print(f"what a sales analyst receives, top {K} passages per question;")
print("the cache was filled by a lawyer\n")
print(f"  {'questions about':16}{'method':30}{'passages':>9}{'empty':>7}{'not readable':>14}")
for group, texts in GROUPS.items():
    for label, method in METHODS:
        got = [method(text) for text in texts]
        print(f"  {group:16}{label:30}{np.mean([len(g) for g in got]):>9.1f}"
              f"{sum(not g for g in got):>7}{sum(not readable(i, analyst) for g in got for i in g):>14}")
        group = ""
