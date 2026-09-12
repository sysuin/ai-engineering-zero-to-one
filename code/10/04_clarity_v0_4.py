# timeout: 900
# Clarity v0.4 on the 400-item appendix.

import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                        # noqa: E402
from clarity.v0_4.longdoc import answer, chunk, select       # noqa: E402

encoder = tiktoken.encoding_for_model(MODEL_FAST)
document = Path("data/meridian/documents/awkward/03-long-appendix.md").read_text()
whole = len(encoder.encode(document))

chunks = chunk(document)
orphans = sum(1 for c in chunks if not c.text.lstrip().startswith(("#", "[")))
print(f"document      {whole:,} tokens")
print(f"chunks        {len(chunks)}, median {sorted(c.tokens for c in chunks)[len(chunks)//2]} tokens")
print(f"orphaned      {orphans}  (chunks that no longer say what they are about)")

QUESTIONS = [
    "What revision is specification item 205 at?",
    "Is specification item 118 approved for food-contact use?",
    "What is the emergency depot override code?",
]

print()
for question in QUESTIONS:
    chosen = select(chunks, question)
    reply, sent = answer(document, question)
    print(f"Q: {question}")
    print(f"   sent {len(chosen)} chunk(s), {sent:,} prompt tokens "
          f"({sent / whole:.1%} of the document)")
    print(f"   A: {reply[:96]}")
    print()

print("The third question has no answer in this document, and Clarity says so rather")
print("than producing something. That escape hatch is Chapter 9, and it becomes the")
print("abstention gate in Chapter 13.")
