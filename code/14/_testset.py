# skip
"""
Thirty questions with a known right answer, for measuring retrieval rather than guessing.

Each entry names a string that appears in the chunk that answers it — or, where the answer's
wording is shared by every contract, the document and section it must come from — so a
retrieved set can be scored automatically. Where the answer is stated in more than one way,
the entry is a list, and a chunk matching any of them counts. Building this took about an
hour of reading Meridian's documents; pooling later showed where that reading was wrong.

The questions are deliberately mixed:
  · natural phrasing, mostly in words the document does not use   (embeddings should win)
  · exact identifiers and clause references                         (BM25 should win)
  · questions that name a quarter or a contract                     (metadata should win)
  · questions where a dozen documents look equally relevant         (reranking should win)
"""

# (question, a string in the answering chunk OR a (source, heading prefix) pair OR a list of
#  either, tag)
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
    # Pooling (§14.2) found the margin stated twice more: in the review's commentary, and in
    # the December newsletter's "fourth quarter".
    ("What was gross margin in 2025 Q4?",
     ["Revenue for 2025 Q4 was", ("qbr-2025-Q4.md", "Commentary"),
      "gross margin at 30.8%"], "near-duplicate"),
    ("How many orders were there in 2024 Q1?", "Revenue for 2024 Q1 was",
     "near-duplicate"),
    # These two first named a string from the wrong section: "2023 Q3" is in the summary and
    # the title block, neither of which names a region; the summary gives total revenue, not
    # revenue by category.
    ("Which region was weakest in 2023 Q3?", ("qbr-2023-Q3.md", "Commentary"),
     "near-duplicate"),
    ("What was revenue by category in 2025 Q2?", ("qbr-2025-Q2.md", "Revenue by category"),
     "near-duplicate"),

    # --- an exact identifier, which embeddings blur together
    # The clause that answers these is worded identically in every contract, so the truth is
    # "that section of that contract". A first version scored the contract's title block as
    # correct, because it is the only chunk that contains the reference — and it does not
    # contain the answer.
    ("What is the price adjustment cap in agreement MSC-2022-100?",
     ("contract-MSC-2022-100.md", "2."), "identifier"),
    ("What are the payment terms in MSC-2023-109?",
     ("contract-MSC-2023-109.md", "3."), "identifier"),
    ("What is the liability cap in MSC-2024-118?",
     ("contract-MSC-2024-118.md", "6."), "identifier"),
    ("Which supplier is MSC-2025-127 with?", "MSC-2025-127", "identifier"),
    ("What does clause 4.2 entitle the buyer to?",
     "three consecutive months entitles the Buyer", "identifier"),
    ("What is specification item 205's revision?", "C.205", "identifier"),

    # --- phrased the way a person asks, in words the answering clause does not use
    # Four of these clauses exist in two wordings across the forty contracts, and the first
    # version credited only one — so a retriever that found the other was scored as wrong.
    ("How long do we have to settle an invoice?",
     ["days of the date of a valid invoice", "settle all undisputed invoices"], "paraphrase"),
    ("Can we walk away from the agreement early?",
     ["terminate this Agreement for convenience", "may terminate for convenience"],
     "paraphrase"),
    ("What if the goods turn up damaged?", "reject non-conforming goods", "paraphrase"),
    ("How much can we claim if they cause us a loss?",
     ["aggregate liability shall not exceed", "aggregate liability shall exceed"], "paraphrase"),
    ("Do they have to hit a delivery target?",
     ["order lines complete and on time", "order lines delivered in full and on time"],
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
