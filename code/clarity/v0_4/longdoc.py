"""
Clarity v0.4 — reading a document that is too large to send economically.

Note the wording. Meridian's 400-item appendix is about 29,000 tokens and fits in the
context window several times over. The reason not to send it on every question is that
doing so costs three hundred times as much as sending the part that matters, for exactly
the same answer.

Three capabilities, in the order you should reach for them:

    chunk()      cut a document without destroying its structure
    select()     send only the chunks that could contain the answer
    digest()     when the question is about the whole document, summarise once and reuse
"""
from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_FAST      # noqa: E402

encoder = tiktoken.encoding_for_model(MODEL_FAST)

# Repeated furniture: page banners, footers, classification marks. Left in, every chunk
# carries it, and every chunk therefore looks a little like every other chunk.
BOILERPLATE = [
    re.compile(r"^MERIDIAN SUPPLY CO\..*$", re.M),
    re.compile(r"^Page \d+ of \d+.*$", re.M),
]


@dataclass
class Chunk:
    text: str
    heading: str
    index: int

    @property
    def tokens(self) -> int:
        return len(encoder.encode(self.text))


def clean(text: str) -> str:
    for pattern in BOILERPLATE:
        text = pattern.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk(document: str, max_tokens: int = 400) -> list[Chunk]:
    """
    Split on headings first, falling back to paragraphs only inside an oversized
    section. Every chunk keeps the heading it belongs to, so a chunk retrieved on its
    own still says what it is about.
    """
    document = clean(document)
    parts = [p for p in re.split(r"(?=^#{1,6} )", document, flags=re.M) if p.strip()]

    chunks: list[Chunk] = []
    for part in parts:
        heading = part.strip().split("\n")[0].lstrip("# ").strip()
        if len(encoder.encode(part)) <= max_tokens:
            chunks.append(Chunk(part.strip(), heading, len(chunks)))
            continue
        # Too big: split by paragraph, and repeat the heading on every piece so the
        # later pieces are not orphans.
        current = ""
        for para in part.split("\n\n"):
            if current and len(encoder.encode(current + para)) > max_tokens:
                chunks.append(Chunk(f"[{heading}]\n{current.strip()}", heading,
                                    len(chunks)))
                current = ""
            current += para + "\n\n"
        if current.strip():
            chunks.append(Chunk(f"[{heading}]\n{current.strip()}", heading, len(chunks)))
    return chunks


def select(chunks: list[Chunk], question: str, budget: int = 4_000) -> list[Chunk]:
    """
    Keyword overlap, and nothing cleverer.

    This is deliberately the crudest possible selection. It is enough to cut the cost
    of a targeted question by two orders of magnitude, and its failures are exactly
    what Part III exists to fix.
    """
    words = {w.lower() for w in re.findall(r"[A-Za-z0-9.]{3,}", question)}
    scored = sorted(
        chunks,
        key=lambda c: -sum(w in c.text.lower() for w in words))

    chosen, used = [], 0
    for candidate in scored:
        if not any(w in candidate.text.lower() for w in words):
            break
        if used + candidate.tokens > budget:
            break
        chosen.append(candidate)
        used += candidate.tokens
    return sorted(chosen, key=lambda c: c.index)


def digest(chunks: list[Chunk], client: OpenAI | None = None,
           group: int = 40) -> str:
    """Summarise every chunk once, so whole-document questions never resend the document."""
    client = client or OpenAI()
    groups = ["\n\n".join(c.text for c in chunks[i:i + group])
              for i in range(0, len(chunks), group)]

    def one(text: str) -> str:
        return (client.chat.completions.create(
            model=MODEL_FAST, temperature=0, max_completion_tokens=1200,
            messages=[{"role": "user", "content":
                       "Summarise the material below as terse one-line facts. Keep every "
                       "identifier and number exactly as written.\n\n" + text}],
        ).choices[0].message.content or "")

    with ThreadPoolExecutor(max_workers=8) as pool:
        return "\n".join(pool.map(one, groups))


def answer(document: str, question: str, client: OpenAI | None = None) -> tuple[str, int]:
    """Answer a targeted question, sending only what could contain the answer."""
    client = client or OpenAI()
    chosen = select(chunk(document), question)
    context = "\n\n".join(c.text for c in chosen)
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200,
        messages=[{"role": "system", "content":
                   "Answer only from the excerpts given. If they do not contain the "
                   "answer, reply NOT IN THE EXCERPTS."},
                  {"role": "user", "content":
                   f"<excerpts>\n{context}\n</excerpts>\n\n{question}"}],
    )
    return ((response.choices[0].message.content or "").strip(),
            response.usage.prompt_tokens)
