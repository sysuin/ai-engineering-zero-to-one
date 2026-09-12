# skip
"""
Thirty questions with a known right answer, for measuring retrieval rather than guessing.

Each entry names a string that appears in exactly the chunk that answers it, so a
retrieved set can be scored automatically. Building this took about an hour of reading
Meridian's documents, which is the honest cost of being able to measure anything at all.

The questions are deliberately mixed:
  · natural phrasing that shares no words with the document  (embeddings should win)
  · exact identifiers and clause references                  (BM25 should win)
  · questions that name a quarter or a contract              (metadata should win)
  · questions where a dozen documents look equally relevant  (reranking should win)
"""

# (question, a string that appears only in the chunk that answers it, tag)
QUESTIONS = [
    # --- one document obviously answers it
    ("Which account did not renew?", "Halloway Group did not renew", "plain"),
    ("When did the Sanitation category launch?",
     "Sanitation category launched this quarter", "plain"),
    ("Why did gross margin fall in 2025 Q1?",
     "Voss Industrial applied a 12% increase", "plain"),
    ("What caused the spike in support contacts?",
     "batch defect in cloths supplied by Pemberton", "plain"),
    ("What is the racked storage capacity at Dallas?", "Racked storage", "plain"),
    ("What happens at the Columbus depot in December?",
     "close for inventory count", "plain"),

    # --- the answer sits in one of twelve near-identical sections
    ("What was total revenue in 2024 Q3?", "Revenue for 2024 Q3 was", "near-duplicate"),
    ("What was total revenue in 2023 Q1?", "Revenue for 2023 Q1 was", "near-duplicate"),
    ("What was gross margin in 2025 Q4?", "Revenue for 2025 Q4 was", "near-duplicate"),
    ("How many orders were there in 2024 Q1?", "Revenue for 2024 Q1 was",
     "near-duplicate"),
    ("Which region was weakest in 2023 Q3?", "2023 Q3", "near-duplicate"),
    ("What was revenue by category in 2025 Q2?", "Revenue for 2025 Q2 was",
     "near-duplicate"),

    # --- an exact identifier, which embeddings blur together
    ("What is the price adjustment cap in agreement MSC-2022-100?",
     "MSC-2022-100", "identifier"),
    ("What are the payment terms in MSC-2023-109?", "MSC-2023-109", "identifier"),
    ("What is the liability cap in MSC-2024-118?", "MSC-2024-118", "identifier"),
    ("Which supplier is MSC-2025-127 with?", "MSC-2025-127", "identifier"),
    ("What does clause 4.2 entitle the buyer to?",
     "three consecutive months entitles the Buyer", "identifier"),
    ("What is specification item 205's revision?", "C.205", "identifier"),

    # --- phrased the way a person asks, sharing no words with the text
    ("How long do we have to settle an invoice?", "days of the date of a valid invoice",
     "paraphrase"),
    ("Can we walk away from the agreement early?", "terminate this Agreement for "
     "convenience", "paraphrase"),
    ("What if the goods turn up damaged?", "reject non-conforming goods", "paraphrase"),
    ("How much can we claim if they cause us a loss?",
     "aggregate liability shall not exceed", "paraphrase"),
    ("Do they have to hit a delivery target?", "order lines complete and on time",
     "paraphrase"),
    ("Which country's courts decide a dispute?", "governed by the laws", "paraphrase"),

    # --- customer-voice questions answered by a ticket
    ("Are customers complaining about cloths falling apart?",
     "cloths falling apart", "ticket"),
    ("Has anyone been charged twice?", "charged us twice", "ticket"),
    ("Is anyone asking about sales tax exemption?", "sales tax", "ticket"),

    # --- questions needing a document nobody would guess
    ("What is the purchase order approval threshold?",
     "require a second approval", "obscure"),
    ("What insurance does a supplier need to be onboarded?",
     "product liability", "obscure"),
    ("What was received short on the Columbus goods received note?",
     "12 short", "obscure"),
]
