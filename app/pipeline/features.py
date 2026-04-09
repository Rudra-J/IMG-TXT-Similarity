"""
Layout features module: infers document structure from line order.

Why layout matters:
Documents have semantic zones. In an invoice, the header contains the invoice
number and date, the body contains line items, and the footer contains totals
and payment terms. Two documents that mention '$4,500' in different zones carry
different meaning — one may be a subtotal, the other the final amount.

This module approximates layout-aware models (e.g., LayoutLM) using only
line position, without requiring a vision model. It extracts three features:
- line_index: absolute position (0 = first line)
- relative_position: float in [0, 1] — 0 is top, 1 is bottom
- zone: 'top' | 'middle' | 'bottom' — coarse 3-zone classification

These heuristics are approximate: OCR flattens multi-column layouts, so
the 'top zone' may contain content that visually appears in multiple columns.
"""
import numpy as np
from typing import List, Tuple, Dict


def compute_layout_features(lines: List[Tuple[str, list]]) -> List[Dict]:
    """
    Extract layout features from a list of (text, bbox) tuples.

    Args:
        lines: Ordered list of (text, bounding_box) tuples from OCR output,
               or (text, []) for plain text lines.

    Returns:
        List of dicts with keys: text, line_index, relative_position, zone.
    """
    n = len(lines)
    if n == 0:
        return []

    features = []
    for i, (text, bbox) in enumerate(lines):
        relative_pos = i / max(n - 1, 1)

        if relative_pos < 0.33:
            zone = "top"
        elif relative_pos < 0.66:
            zone = "middle"
        else:
            zone = "bottom"

        features.append({
            "text": text,
            "line_index": i,
            "relative_position": round(relative_pos, 4),
            "zone": zone,
        })
    return features


def compute_layout_features_from_text(text: str) -> List[Dict]:
    """
    Extract layout features from plain text by splitting on newlines.
    Used for text-mode inputs that skip OCR.

    Args:
        text: Raw or preprocessed text string.

    Returns:
        Same structure as compute_layout_features().
    """
    lines = [(line, []) for line in text.split("\n") if line.strip()]
    return compute_layout_features(lines)


def compute_layout_adjustment(
    features1: List[Dict], features2: List[Dict]
) -> float:
    """
    Compute a layout alignment score in [-0.10, +0.10].

    Positive value: documents share similar zone distributions (structurally aligned).
    Negative value: zone distributions differ (one doc is top-heavy, the other bottom-heavy).

    Method: compare zone distribution histograms (top/middle/bottom fraction)
    using cosine similarity, then map [0, 1] -> [-0.10, +0.10].

    This penalizes, for example, an invoice where the total appears in the top zone
    being compared against one where it appears in the bottom zone.

    Args:
        features1, features2: Layout feature dicts from compute_layout_features().

    Returns:
        Float in [-0.10, +0.10].
    """
    if not features1 or not features2:
        return 0.0

    def zone_vector(features: List[Dict]) -> np.ndarray:
        n = len(features)
        counts = {"top": 0, "middle": 0, "bottom": 0}
        for f in features:
            counts[f["zone"]] += 1
        return np.array([counts[z] / n for z in ("top", "middle", "bottom")],
                        dtype=float)

    v1 = zone_vector(features1)
    v2 = zone_vector(features2)

    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0

    alignment = float(np.dot(v1, v2) / norm)
    # Map cosine similarity [0, 1] -> adjustment [-0.10, +0.10]
    return round((alignment - 0.5) * 0.20, 4)
