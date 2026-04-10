"""
Explainability module: builds structured output from similarity signals.
"""
import os
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")


def build_explanation(
    lexical: float,
    semantic: float,
    final: float,
    mode: str,
) -> Dict[str, Any]:
    """
    Build a structured explainability dict from similarity signals.

    Args:
        lexical, semantic, final: Similarity scores from pipeline.
        mode: Comparison mode — 'text-image' | 'image-image' | 'text-text'.

    Returns:
        Dict with keys: mode, scores, explanation.
    """
    if final >= 0.85:
        verdict = "highly similar"
    elif final >= 0.65:
        verdict = "moderately similar"
    elif final >= 0.40:
        verdict = "partially similar"
    else:
        verdict = "largely dissimilar"

    explanation = (
        f"The documents are {verdict} (final score: {final:.4f}). "
        f"Semantic similarity: {semantic:.4f}, "
        f"lexical similarity: {lexical:.4f}."
    )

    return {
        "mode": mode,
        "scores": {
            "lexical": round(lexical, 4),
            "semantic": round(semantic, 4),
            "final": round(final, 4),
        },
        "explanation": explanation,
    }


def render_html(result: Dict[str, Any]) -> str:
    """
    Render the explainability result as an HTML report using report.html template.

    Args:
        result: Dict returned by build_explanation().

    Returns:
        HTML string suitable for returning as an HTMLResponse.
    """
    env = Environment(loader=FileSystemLoader(os.path.abspath(_TEMPLATES_DIR)))
    template = env.get_template("report.html")
    return template.render(**result)
