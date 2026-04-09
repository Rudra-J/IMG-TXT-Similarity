"""
Local embedding backend using sentence-transformers (all-MiniLM-L6-v2).

Advantages:
- Runs fully offline — no API key, no network calls
- Fast inference: ~5ms per sentence on CPU
- ~80MB model size

Limitations vs Voyage-3:
- Smaller context window (256 tokens)
- Lower ranking correlation on paraphrase benchmarks
- Less robust on domain-specific text (legal, financial)

The embed-off in app/embeddings/evaluator.py determines whether this or
Voyage-3 is used in the production pipeline.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts. Returns a (N, 384) float32 array, unit-normalized.

    Args:
        texts: List of strings to embed.

    Returns:
        numpy array of shape (len(texts), 384), each row L2-normalized.
    """
    model = _get_model()
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_single(text: str) -> np.ndarray:
    """Embed a single text string. Returns 1-D array of shape (384,)."""
    return embed([text])[0]
