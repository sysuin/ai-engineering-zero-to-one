# The same Meridian text, counted by four tokenisers. A price per million tokens compares
# providers only after both counts are taken on your own text.

from pathlib import Path

import tiktoken

DOCS = Path("data/meridian/documents")
TEXTS = {
    "a quarterly review": (DOCS / "quarterly-reviews/qbr-2024-Q3.md").read_text(),
    "a contract": (DOCS / "contracts/contract-MSC-2022-100.md").read_text(),
    "support tickets": "\n".join(Path(DOCS / "tickets/tickets.jsonl").read_text().splitlines()[:80]),
    "the 400-item appendix": (DOCS / "awkward/03-long-appendix.md").read_text(),
    "a German email": "Guten Morgen. Wir haben unsere Lieferung gestern erhalten, aber zwei der fünf Kartons waren "
                      "beschädigt und die Reinigungstücher darin waren nass. Könnten Sie eine Abholung veranlassen "
                      "und uns bis Freitag Ersatz schicken? Unsere Kundennummer ist 40211. Vielen Dank für Ihre Hilfe.",
    "generated SQL": "SELECT region, ROUND(100.0 * SUM(gross_profit) / NULLIF(SUM(revenue), 0), 1) AS margin_pct\n"
                     "FROM v_sales\nWHERE year = 2024 AND quarter IN (3, 4)\nGROUP BY region\nORDER BY margin_pct DESC;",
}
ENCODINGS = ["o200k_base", "cl100k_base", "p50k_base", "r50k_base"]
encoders = {name: tiktoken.get_encoding(name) for name in ENCODINGS}

print(f"  {'text':24}{'characters':>11}" + "".join(f"{e:>13}" for e in ENCODINGS))
for label, text in TEXTS.items():
    counts = [len(encoders[e].encode(text)) for e in ENCODINGS]
    print(f"  {label:24}{len(text):>11,}" + "".join(f"{c:>13,}" for c in counts))
    print(f"  {'':24}{'':>11}" + "".join(f"{c / counts[0]:>12.2f}x" for c in counts))

print("\nratios are against o200k_base; each row is the same text, so a difference is the tokeniser alone")
