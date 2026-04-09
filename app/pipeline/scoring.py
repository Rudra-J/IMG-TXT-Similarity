"""
Score aggregation module: combines all similarity signals into a single
interpretable final score using justified, fixed weights.

Weight justification:
- semantic (0.40): The most robust signal. Handles OCR noise, paraphrasing,
  and synonym variation. Dominant weight because it captures meaning intent.
- entity (0.35): Critical for enterprise correctness. A $4,500 vs $4,050
  discrepancy must pull the final score below any "similar" threshold.
  Second-highest weight because factual accuracy cannot be overridden by fluency.
- lexical (0.15): Useful for exact identifier matches and structured data,
  but brittle against any paraphrasing. Lowest weight because it's a subset
  of what semantic already captures, plus it fails on rephrasing.
- layout (+-0.10): Structural bonus/penalty. Capped at 10% so it can nudge
  but not override the content-based signals. Documents with aligned structure
  are more likely to be true duplicates.

Note: weights sum to 0.90 (not 1.0) by design — layout adjustment uses the
remaining +-0.10 budget. Final is always clamped to [0.0, 1.0].
"""


def combine_scores(
    lexical: float,
    semantic: float,
    entity: float,
    layout_adjustment: float,
) -> float:
    """
    Weighted aggregation of similarity signals into a final score.

    Args:
        lexical: TF-IDF cosine similarity in [0.0, 1.0].
        semantic: Embedding cosine similarity in [0.0, 1.0].
        entity: Entity overlap ratio in [0.0, 1.0].
        layout_adjustment: Layout alignment delta in [-0.10, +0.10].

    Returns:
        Final similarity score in [0.0, 1.0].
    """
    base = 0.40 * semantic + 0.35 * entity + 0.15 * lexical
    final = base + layout_adjustment
    return max(0.0, min(1.0, final))
