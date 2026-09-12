# skip
"""
Twenty-four questions about Meridian: twelve answerable, twelve not.

The unanswerable ones are the point. They are plausible, specific, and about the right
company — the sort of thing a user genuinely asks — and nothing in the corpus answers
them. A retrieval system that cannot tell these two groups apart will answer all
twenty-four, and be wrong about half of them with complete fluency.

Reused by Chapters 14 and 21.
"""

ANSWERABLE = [
    ("What was total revenue in 2024 Q3?", "qbr-2024-Q3.md"),
    ("Which region was weakest in 2024 Q3?", "qbr-2024-Q3.md"),
    ("Which account did not renew, and when?", "qbr-2024-Q3.md"),
    ("What explanation does the company give for the Midwest fall?", "qbr-2024-Q3.md"),
    ("When did the Sanitation category launch?", "qbr-2024-Q2.md"),
    ("Why did gross margin fall in 2025 Q1?", "qbr-2025-Q1.md"),
    ("What caused the spike in support contacts in 2024 Q4?", "qbr-2024-Q4.md"),
    ("How many days notice is required before a price rise under MSC-2022-100?",
     "contract-MSC-2022-100.md"),
    ("Which law governs the supplier agreements?", "contract-MSC-2022-100.md"),
    ("What happens if a supplier misses the delivery target three months running?",
     "contract-MSC-2022-100.md"),
    ("How long does the buyer have to reject non-conforming goods?",
     "contract-MSC-2022-100.md"),
    ("What is the racked storage capacity at Dallas?", "02-rotated-table.md"),
]

# Plausible, specific, and absent. Each one is a thing a real user would ask.
UNANSWERABLE = [
    "What is Meridian's employee headcount?",
    "Who is the Chief Financial Officer?",
    "What was revenue in 2026 Q1?",
    "What is the customer churn rate?",
    "Which insurer underwrites the product liability cover?",
    "What is the average order value in the West region?",
    "How many vehicles are in the delivery fleet?",
    "What is the target for on-time delivery in 2026?",
    "Which warehouse management system does Meridian use?",
    "What is the annual contract value of the Voss Industrial agreement?",
    "How many people work at the Columbus depot?",
    "What is Meridian's carbon reduction target?",
]
