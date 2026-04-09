"""
Similarity computation module: three complementary similarity signals.

Each signal captures a different aspect of document similarity:
- Lexical: exact vocabulary overlap (brittle — fails on paraphrasing)
- Semantic: meaning similarity (robust — handles synonyms and paraphrasing)
- Entity: factual accuracy (critical — catches amount/ID mismatches)

The active embedding backend is selected once based on the embed-off results
in app/embeddings/evaluator.py. Change ACTIVE_EMBEDDER to 'local' or 'voyage'
after running the evaluator.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import numpy as np
from typing import Dict, List

# --- EMBED-OFF RESULT ---
# Set to 'local' (MiniLM) or 'voyage' (Voyage-3) based on evaluator.py output.
# Default: 'local' — safe fallback that requires no API key.
ACTIVE_EMBEDDER = "local"


def _get_embed_single():
    """Return the embed_single function for the active backend."""
    if ACTIVE_EMBEDDER == "voyage":
        from app.embeddings.claude_embed import embed_single
    else:
        from app.embeddings.local import embed_single
    return embed_single


def compute_lexical_similarity(text1: str, text2: str) -> float:
    """
    TF-IDF cosine similarity between two texts.

    Why brittle: vocabulary must overlap for a score > 0. "car" and "automobile"
    score 0.0 despite identical meaning. Useful for exact token matches —
    invoice IDs, amounts, and codes that appear verbatim in both documents.

    Args:
        text1, text2: Preprocessed text strings.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if either text is empty.
    """
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text1, text2])
        return float(sklearn_cosine(tfidf[0], tfidf[1])[0][0])
    except ValueError:
        return 0.0


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Cosine similarity of document embeddings using the winning embed backend.

    Why better than lexical: captures meaning regardless of exact wording.
    "The payment is overdue" and "This invoice has not been settled" score
    high because their embeddings encode the same concept.

    OCR robustness: minor OCR errors ('0' to 'O') rarely affect embeddings
    significantly, unlike TF-IDF which treats them as completely different tokens.

    Args:
        text1, text2: Preprocessed text strings.

    Returns:
        Float in [0.0, 1.0].
    """
    embed_single = _get_embed_single()
    e1 = embed_single(text1)
    e2 = embed_single(text2)
    return float(np.clip(np.dot(e1, e2), 0.0, 1.0))


def compute_entity_similarity(
    entities1: Dict[str, List[str]], entities2: Dict[str, List[str]]
) -> float:
    """
    Jaccard-style overlap ratio across all entity categories.

    Why critical: semantic similarity cannot distinguish $4,500 from $4,050.
    Two invoices with one digit different in the amount are NOT similar for
    validation purposes, but their text embeddings would score > 0.95.

    This metric ensures factual discrepancies dominate the final score
    (entity weight = 0.35, second highest after semantic at 0.40).

    Args:
        entities1, entities2: Entity dicts from regex_extractor or spacy_extractor.

    Returns:
        Float in [0.0, 1.0]. Returns 1.0 if both dicts are empty.
    """
    all_keys = set(entities1.keys()) | set(entities2.keys())
    if not all_keys:
        return 1.0

    total_intersection = 0
    total_union = 0

    for key in all_keys:
        set1 = {str(e).lower().strip() for e in entities1.get(key, [])}
        set2 = {str(e).lower().strip() for e in entities2.get(key, [])}
        if not set1 and not set2:
            continue
        total_intersection += len(set1 & set2)
        total_union += len(set1 | set2)

    return total_intersection / total_union if total_union > 0 else 1.0
