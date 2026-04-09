"""
One-time embed-off: compare MiniLM-L6-v2 vs Voyage-3 on 3 document pairs
covering the three similarity regimes we care about.

Run this script ONCE during development:
    VOYAGE_API_KEY=your_key python -m app.embeddings.evaluator

Read the output, then update ACTIVE_EMBEDDER in app/pipeline/similarity.py
to either 'local' or 'voyage' based on the winner. Document the result
and metric values in README.md under the "Design Decisions" section.

This script is NOT called at runtime — it informs a one-time design decision.
"""
import numpy as np
from scipy.stats import spearmanr

EVAL_PAIRS = [
    {
        "label": "near_duplicate_invoice",
        "doc1": (
            "Invoice INV-2024-001. Date: 15 Jan 2024. Amount due: $4,500.00. "
            "Bill to: Acme Corporation. Payment terms: NET-30."
        ),
        "doc2": (
            "Invoice number INV-2024-001 dated January 15, 2024. "
            "Total amount: $4,500.00. Customer: Acme Corporation. Terms: NET-30."
        ),
        "expected": 0.95,
    },
    {
        "label": "paraphrase_ticket",
        "doc1": (
            "Ticket TKT-555: The authentication service is returning HTTP 500 "
            "errors on the login endpoint. Affects all users. Priority: HIGH."
        ),
        "doc2": (
            "Please investigate TKT-555. The login system keeps crashing with "
            "internal server errors. All users are impacted. Urgent fix required."
        ),
        "expected": 0.75,
    },
    {
        "label": "unrelated",
        "doc1": "Invoice INV-2024-001. Date: 15 Jan 2024. Amount due: $4,500.00.",
        "doc2": (
            "The weather forecast for London this weekend shows overcast skies "
            "with a chance of light rain on Saturday morning."
        ),
        "expected": 0.10,
    },
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def evaluate_backend(embed_fn, name: str) -> list:
    print(f"\nRunning {name}...")
    scores = []
    for pair in EVAL_PAIRS:
        e1 = embed_fn(pair["doc1"])
        e2 = embed_fn(pair["doc2"])
        scores.append(cosine(e1, e2))
    return scores


def run():
    from app.embeddings.local import embed_single as local_embed
    from app.embeddings.claude_embed import embed_single as voyage_embed

    local_scores = evaluate_backend(local_embed, "MiniLM-L6-v2")
    voyage_scores = evaluate_backend(voyage_embed, "Voyage-3")
    expected = [p["expected"] for p in EVAL_PAIRS]

    local_corr, _ = spearmanr(local_scores, expected)
    voyage_corr, _ = spearmanr(voyage_scores, expected)

    print("\n=== Embed-Off Results ===\n")
    header = f"{'Pair':<30} {'MiniLM':>10} {'Voyage-3':>10} {'Expected':>10}"
    print(header)
    print("-" * len(header))
    for i, pair in enumerate(EVAL_PAIRS):
        print(
            f"{pair['label']:<30} {local_scores[i]:>10.4f} "
            f"{voyage_scores[i]:>10.4f} {expected[i]:>10.2f}"
        )

    print(f"\nSpearman ranking correlation vs expected:")
    print(f"  MiniLM-L6-v2 : {local_corr:.4f}")
    print(f"  Voyage-3     : {voyage_corr:.4f}")

    winner = "voyage" if voyage_corr > local_corr else "local"
    print(f"\n{'='*40}")
    print(f"WINNER: {winner.upper()}")
    print(f"{'='*40}")
    print(
        f"\nAction: Open app/pipeline/similarity.py and set:\n"
        f"  ACTIVE_EMBEDDER = '{winner}'\n"
        f"Then document these results in README.md under 'Design Decisions'."
    )


if __name__ == "__main__":
    run()
