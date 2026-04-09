"""
Explainability module: builds structured output from all similarity signals.

Why explainability matters in enterprise workflows:
- A similarity score of 0.72 is meaningless without context.
- Auditors need to know: which entities matched, which differed, and why
  the system flagged or approved the document pair.
- The structured output (matched/mismatched entities, score breakdown,
  natural language explanation) provides a traceable decision log.
"""
import os
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")


def build_explanation(
    lexical: float,
    semantic: float,
    entity: float,
    final: float,
    entities1: Dict[str, List[str]],
    entities2: Dict[str, List[str]],
    mode: str,
) -> Dict[str, Any]:
    """
    Build a structured explainability dict from similarity signals.

    Args:
        lexical, semantic, entity, final: Similarity scores from pipeline.
        entities1, entities2: Combined entity dicts (regex + spaCy) per document.
        mode: Comparison mode — 'text-image' | 'image-image' | 'text-text'.

    Returns:
        Dict with keys: mode, scores, entities, mismatches, explanation.
    """
    all_keys = set(entities1.keys()) | set(entities2.keys())
    matched, only_in_doc1, only_in_doc2 = [], [], []

    for key in sorted(all_keys):
        set1 = {str(e).lower().strip() for e in entities1.get(key, [])}
        set2 = {str(e).lower().strip() for e in entities2.get(key, [])}
        matched.extend(sorted(set1 & set2))
        only_in_doc1.extend(sorted(set1 - set2))
        only_in_doc2.extend(sorted(set2 - set1))

    mismatches = []
    if only_in_doc1:
        mismatches.append(f"Found only in document 1: {', '.join(sorted(only_in_doc1))}")
    if only_in_doc2:
        mismatches.append(f"Found only in document 2: {', '.join(sorted(only_in_doc2))}")

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
        f"Semantic similarity: {semantic:.4f}, entity overlap: {entity:.4f}, "
        f"lexical similarity: {lexical:.4f}."
    )
    if mismatches:
        explanation += f" Key differences detected: {'; '.join(mismatches)}."

    return {
        "mode": mode,
        "scores": {
            "lexical": round(lexical, 4),
            "semantic": round(semantic, 4),
            "entity": round(entity, 4),
            "final": round(final, 4),
        },
        "entities": {
            "matched": sorted(set(matched)),
            "only_in_doc1": sorted(set(only_in_doc1)),
            "only_in_doc2": sorted(set(only_in_doc2)),
        },
        "mismatches": mismatches,
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
