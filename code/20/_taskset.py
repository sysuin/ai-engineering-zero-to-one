# skip
"""
Eight composite questions, each needing both halves of Meridian.

Every one requires a figure that only the warehouse has and a cause that only the
documents have. That is deliberate: these are the questions a multi-agent design is
supposed to be good at, so if a team cannot beat one agent here it will not beat it
anywhere.

Scoring is two independent points per task — the number, and the cause — because a
system that gets the arithmetic right and invents the reason is not half correct in any
useful sense, and neither is the reverse.
"""
import re

# figure: the value that must appear, and how close counts
# cause:  words that must appear, any one of them
TASKS = [
    {"q": "What was revenue in 2024 Q3, and what does the company give as the "
          "reason the Midwest fell?",
     "figure": 8461841.81, "tolerance": 0.005, "cause": ["halloway"]},

    {"q": "What was gross margin in 2025 Q1, and what explanation is given for the "
          "change?",
     "figure": 31.4, "tolerance": 0.02, "cause": ["voss"]},

    {"q": "How many support tickets relate to the 2024 Q4 quality problem, and which "
          "supplier was involved?",
     "figure": None, "tolerance": 0, "cause": ["pemberton"]},

    {"q": "What was revenue in 2024 Q2, and what new category appears in the data "
          "from that quarter?",
     "figure": 8074847.91, "tolerance": 0.005, "cause": ["sanitation"]},

    {"q": "By how much did revenue change from 2024 Q3 to 2024 Q4, and what does the "
          "commentary say was behind it?",
     "figure": -195773.45, "tolerance": 0.02, "cause": ["halloway", "midwest"]},

    {"q": "What was gross margin in 2024 Q2 versus 2025 Q1, and which supplier's "
          "pricing is named as a cause of the later figure?",
     "figure": 31.4, "tolerance": 0.02, "cause": ["voss"]},

    {"q": "How many orders were there in 2025 Q1, and what does the quarterly review "
          "say about margin that quarter?",
     "figure": 5847, "tolerance": 0.005, "cause": ["voss", "cost"]},

    {"q": "What was revenue in 2025 Q2, and what does the company say about the "
          "Midwest region by then?",
     "figure": 8633630.78, "tolerance": 0.005, "cause": ["halloway", "midwest"]},
]


NUMBER = re.compile(r"(-?)\$?(\d[\d,]*(?:\.\d+)?)\s*(million|m\b|bn|billion|k\b|thousand)?",
                    re.IGNORECASE)
SCALE = {"million": 1e6, "m": 1e6, "bn": 1e9, "billion": 1e9, "k": 1e3, "thousand": 1e3}


def numbers(text: str) -> list[float]:
    """Every number in the text, with 'million', 'm', 'k' and friends applied."""
    found = []
    for sign, digits, unit in NUMBER.findall(text):
        try:
            value = float(digits.replace(",", ""))
        except ValueError:
            continue
        found.append(value * SCALE.get(unit.lower(), 1) if unit else value)
    return found


def score(task: dict, answer: str) -> tuple[int, int]:
    """Two points, judged separately: did the number appear, did the cause appear."""
    text = answer or ""
    cause = 1 if any(word in text.lower() for word in task["cause"]) else 0
    if task["figure"] is None:
        return cause, cause          # no figure to check; the cause carries both

    # A number anywhere in the answer within tolerance of the target, in whatever shape
    # the model wrote it. An earlier version matched substrings of the rounded target —
    # "31" for a margin of 31.4 — which counted any answer containing "2031" as right,
    # and joined every digit in the answer into one string before looking for numbers.
    wanted = abs(task["figure"])
    figure = int(any(abs(value - wanted) <= wanted * task["tolerance"]
                     for value in numbers(text)))
    return figure, cause
