"""
Score aggregation module: combines similarity signals into a single
interpretable final score using justified, fixed weights.

Weight justification:
- semantic (0.75): The most robust signal. Handles OCR noise, paraphrasing,
  and synonym variation. Dominant weight because it captures meaning intent.
- lexical (0.15): Useful for exact identifier matches and structured data,
  but brittle against any paraphrasing. Lower weight because it's a subset
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
    layout_adjustment: float,
) -> float:
    """
    Weighted aggregation of similarity signals into a final score.

    Args:
        lexical: TF-IDF cosine similarity in [0.0, 1.0].
        semantic: Embedding cosine similarity in [0.0, 1.0].
        layout_adjustment: Layout alignment delta in [-0.10, +0.10].

    Returns:
        Final similarity score in [0.0, 1.0].
    """
    base = 0.75 * semantic + 0.15 * lexical
    final = base + layout_adjustment
    return max(0.0, min(1.0, final))
