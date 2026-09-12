# The six parts, applied. Two versions of Clarity's summarising prompt, side by side.

from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST
from clarity.prompts import load, versions

client = OpenAI()
review = Path("data/meridian/documents/quarterly-reviews/qbr-2024-Q3.md").read_text()

print("Prompt versions on disk:", versions("summarize"))
print()

for version in versions("summarize"):
    system = load("summarize", version=version)
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user",
                   "content": f"<document>\n{review}\n</document>"}],
    )
    text = (response.choices[0].message.content or "").strip()

    print(f"--- v{version}  ({len(system)} chars of prompt, "
          f"{response.usage.completion_tokens} tokens out)")
    print(text[:520])
    print()

print("v1 is three lines and produces something usable. v2 names the audience, states")
print("the format, and replaces a prohibition with a positive instruction — and the")
print("difference shows up as structure you can rely on rather than accuracy you cannot")
print("measure yet. Chapter 21 is where 'better' stops being a matter of opinion.")
