# AI Engineering from Zero to One — the code

The companion code for *AI Engineering from Zero to One* by Sunny Singh: every listing in the
book, **Clarity** — the question-answering system it builds across 32 chapters — and the
generator for **Meridian**, the fictional company whose documents and warehouse every
example runs against.

Nothing in the book's output was typed by hand. Each listing here was executed, and its
real output is saved beside it, so you can compare what you get with what the book shows.

## Start

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
cp .env.example .env                                  # then put your API key in it
python3 code/00_check_setup.py                        # checks every step, in order
```

Appendix B of the book walks through each of those on macOS, Windows and Linux, and says
how to set a spending limit before your first call.

Then the dataset, which is generated rather than downloaded:

```bash
python3 code/meridian/generate.py      # about forty seconds, no network
python3 code/meridian/verify.py
python3 code/meridian_index.py --build # the one step that costs money — a few cents
```

## Where things are

```
code/NN/            the listings for chapter NN, each with its captured .out
code/clarity/v1_0/  Clarity as it stands at the end of the book — start reading here
code/clarity/v0_*   Clarity as it stood at the end of each chapter, kept on purpose
code/clarity/       platform/, evals/ and config.py, the only file that names a model
code/meridian/      the dataset generator
tests/              the evaluation suite as tests; they skip without an API key
```

`make help` lists the everyday commands: `make ask Q="..."`, `make eval`, `make demo`,
`make serve`.

## Checking your output against the book

```bash
python3 code/_runner.py 14      # re-run chapter 14's listings and capture what they print
python3 code/_runner.py --check # which captured outputs no longer match their listing
```

Your numbers will not match the book's exactly. The models are not deterministic, and
Chapter 24 is about exactly that — what should match is the shape of each result.

## Licence

MIT — see [LICENSE](LICENSE). The licence covers this code and the dataset generator, not
the book.
