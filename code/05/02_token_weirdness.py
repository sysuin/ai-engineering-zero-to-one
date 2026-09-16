# Character-level questions, asked of something that has never seen a character.
#
# Each question is asked five times, because asking once tells you almost nothing.

import tiktoken
from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
encoder = tiktoken.encoding_for_model(MODEL_FAST)
TRIALS = 5


def ask(question: str) -> str:
    return client.chat.completions.create(
        model=MODEL_FAST,
        messages=[{"role": "user", "content": question}],
        temperature=0,
    ).choices[0].message.content.strip()


TESTS = [
    ("Count a letter",
     "How many times does the letter r appear in 'strawberry'? Number only.", "3"),
    ("Count it in a phrase",
     "How many times does the letter r appear in "
     "'Meridian Supply Corporation refrigerator'? Number only.", "6"),
    ("Compare decimals",
     "Which is larger, 9.11 or 9.9? Answer with the number only.", "9.9"),
    ("Reverse a name",
     "Reverse the string 'Meridian'. Reversed string only.", "naidireM"),
]

print(f"Each question asked {TRIALS} times at temperature 0\n")
tally = {}
for label, question, truth in TESTS:
    answers = [ask(question) for _ in range(TRIALS)]
    hits = sum(truth.lower().replace(" ", "") in a.lower().replace(" ", "")
               for a in answers)
    distinct = len(set(answers))
    tally[label] = (hits, distinct)
    print(f"  {label:22} {hits}/{TRIALS} correct, {distinct} distinct answer(s)")
    for answer in sorted(set(answers)):
        mark = "ok " if truth.lower().replace(" ", "") in answer.lower().replace(" ", "") \
               else "   "
        print(f"      {mark} {answer[:40]!r}")

famous = tally["Count a letter"][0] + tally["Compare decimals"][0]
nearby = tally["Count it in a phrase"][0] + tally["Reverse a name"][0]
unstable = [label for label, (_, distinct) in tally.items() if distinct > 1]
print()
print(f"The two famous questions: {famous}/{2 * TRIALS} right. The two a step away from "
      f"them: {nearby}/{2 * TRIALS}.")
if famous > nearby:
    print("The famous failures have been drilled into training data because they became")
    print("famous; the weakness underneath them has not gone anywhere.")
else:
    print("On this run the nearby questions did as well as the famous ones — which is a")
    print("reason to test your own cases, not a reason to trust character-level answers.")
print("Answers that changed between identical attempts: "
      + (", ".join(unstable) if unstable else "none — at temperature 0, on this run"))
print()
print("Here is why. The model never sees letters:")
for word in ("strawberry", "Meridian", "refrigerator"):
    pieces = [encoder.decode([i]) for i in encoder.encode(word)]
    print(f"   {word:14} -> {pieces}")

print()
print("It is being asked to spell by something for which spelling does not exist.")
print("The fix is not a better prompt. The fix is a tool that can count — Chapter 16.")
