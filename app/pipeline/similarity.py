"""
Similarity computation module: two complementary similarity signals.

Each signal captures a different aspect of document similarity:
- Lexical: exact vocabulary overlap (brittle — fails on paraphrasing)
- Semantic: meaning similarity (robust — handles synonyms and paraphrasing)

The active embedding backend is selected once based on the embed-off results
in app/embeddings/evaluator.py. Change ACTIVE_EMBEDDER to 'local' or 'voyage'
after running the evaluator.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import numpy as np

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


