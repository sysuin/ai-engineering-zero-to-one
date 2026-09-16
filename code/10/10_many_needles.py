# timeout: 1800
# One planted fact was found at every depth. How many can the model find at once? K override
# codes, one per depot, planted at random through the 29,000-token appendix; the model is asked
# to list them all.

import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
APPENDIX = Path("data/meridian/documents/awkward/03-long-appendix.md").read_text()
SECTIONS = [s for s in APPENDIX.split("## ") if s.strip()][1:]
CITIES = ["Albany", "Atlanta", "Austin", "Boise", "Boston", "Buffalo", "Charlotte", "Chicago",
          "Cleveland", "Columbus", "Dallas", "Denver", "Detroit", "El Paso", "Fresno", "Houston",
          "Indianapolis", "Jacksonville", "Kansas City", "Las Vegas", "Louisville", "Memphis",
          "Miami", "Milwaukee", "Nashville", "Newark", "Oakland", "Omaha", "Orlando", "Phoenix",
          "Pittsburgh", "Portland", "Raleigh", "Reno", "Richmond", "Sacramento", "San Diego",
          "Seattle", "Tampa", "Tucson"]
QUESTION = ("List every depot emergency override code stated in the document, one per line, "
            "as 'City: code'. List nothing else.")
RUNS = 3


def trial(k: int, seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    truth = {city: f"QX-{rng.randrange(1000, 10000)}" for city in rng.sample(CITIES, k)}
    body = list(SECTIONS)
    for city, code in truth.items():
        at = rng.randrange(len(body) + 1)
        body.insert(at, f"C.{900 + len(body)} Depot override code\n\nThe emergency depot "
                        f"override code for the {city} facility is {code}.\n")
    document = "## " + "## ".join(body)
    answer = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=800,
        messages=[{"role": "user", "content": f"<document>\n{document}\n</document>\n\n{QUESTION}"}],
    ).choices[0].message.content or ""
    listed = {m.group(1).strip(): m.group(2)
              for m in re.finditer(r"([A-Z][A-Za-z .]+?)\s*:\s*(QX-\d{4})", answer)}
    found = sum(listed.get(city) == code for city, code in truth.items())
    invented = sum(truth.get(city) != code for city, code in listed.items())
    return found, invented


print(f"override codes planted at random through {len(SECTIONS)} sections; {RUNS} runs each\n")
print(f"  {'codes planted':>13}{'found':>12}{'recall':>9}{'invented':>10}{'worst run':>11}")
for k in (1, 5, 10, 20, 40):
    with ThreadPoolExecutor(max_workers=RUNS) as pool:
        results = list(pool.map(lambda s: trial(k, 1000 * k + s), range(RUNS)))
    found = sum(f for f, _ in results)
    invented = sum(i for _, i in results)
    worst = min(f for f, _ in results)
    print(f"  {k:>13}{found:>8}/{k * RUNS:<3}{found / (k * RUNS):>9.0%}{invented:>10}"
          f"{worst:>8}/{k}")
