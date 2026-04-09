"""
Voyage AI embedding backend via the voyageai SDK.

Model: voyage-3 — Anthropic's recommended general-purpose embedding model.
Requires: VOYAGE_API_KEY environment variable (get from https://voyageai.com).

Advantages vs MiniLM:
- Larger context window (32K tokens)
- Higher quality on retrieval/similarity benchmarks
- Better on domain-specific text (financial, legal, technical)

Disadvantages:
- Requires API key and network access
- Latency: ~200ms per batch (network round-trip)
- Cost: billed per token

The embed-off in app/embeddings/evaluator.py determines whether this or
MiniLM is used in the production pipeline.
"""
import os
import numpy as np
import voyageai
from typing import List

MODEL_NAME = "voyage-3"
_client: voyageai.Client = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "VOYAGE_API_KEY environment variable is not set. "
                "Get your key from https://voyageai.com and set it before using "
                "the Voyage AI embedding backend."
            )
        _client = voyageai.Client(api_key=api_key)
    return _client


def embed(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using Voyage-3. Returns unit-normalized float32 array.

    Args:
        texts: List of strings to embed.

    Returns:
        numpy array of shape (len(texts), D) where D is the Voyage-3 dimension.
        Each row is L2-normalized.
    """
    client = _get_client()
    result = client.embed(texts, model=MODEL_NAME, input_type="document")
    vecs = np.array(result.embeddings, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def embed_single(text: str) -> np.ndarray:
    """Embed a single text string. Returns 1-D normalized array."""
    return embed([text])[0]
