# Multimodal Document Similarity System

A lightweight FastAPI service that compares two documents — across text and image
formats — using OCR, semantic embeddings, and layout heuristics.
Returns a scored similarity result with detailed explainability output.

## Problem Framing

Enterprise workflows routinely need to verify that two documents refer to the same
information: Does this scanned invoice match the original text record? Are these two
support tickets describing the same issue? Is this image attachment a duplicate?

Heavy multimodal transformers (CLIP, LayoutLM, GPT-4V) can answer these questions
but require significant compute and are opaque. This system uses a hybrid pipeline
that is fast, interpretable, and deployable on CPU hardware.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph IN["Input — POST /compare"]
        M{{"mode"}}
        TXT["text file (.txt)"]
        IMG["image file (.png/.jpg)"]
    end

    OCR["EasyOCR\nextract_text_from_image()"]
    PRE["preprocess()\nnormalize · denoise · preserve IDs & amounts"]

    subgraph FE["Feature Extraction"]
        A["Semantic Embeddings\nMiniLM-L6-v2 / Voyage-3"]
        B["TF-IDF Vectors"]
        E["Layout Heuristics\ntop / middle / bottom zones"]
    end

    subgraph SC["Similarity Computation"]
        F["semantic_similarity()"]
        G["lexical_similarity()"]
        I["layout_adjustment()"]
    end

    FINAL["combine_scores()\n0.75 × semantic + 0.15 × lexical ± 0.10 × layout"]

    subgraph OUT["Explainability Output"]
        J["JSON\nscores · explanation"]
        K["HTML Report\nscore bars · explanation"]
    end

    M -->|"text-image / text-text"| TXT --> PRE
    M -->|"text-image / image-image"| IMG --> OCR --> PRE
    PRE --> A & B & E
    A --> F
    B --> G
    E --> I
    F & G & I --> FINAL
    FINAL --> J & K
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | HTML frontend |
| `/compare` | POST | Compare two documents |
| `/test` | GET | Run synthetic test suite |
| `/evaluate` | POST | Benchmark against labeled pairs |

### POST /compare

Multipart form. Required field: `mode` (text-image, image-image, text-text).

- **text-image:** `text_file` (.txt) + `image_file` (.png/.jpg)
- **image-image:** `image_file_1` + `image_file_2`
- **text-text:** `text_file_1` + `text_file_2`

Optional: `?format=html` for rendered HTML report.

**Example response:**
```json
{
  "mode": "text-image",
  "scores": {
    "lexical": 0.72,
    "semantic": 0.88,
    "final": 0.78
  },
  "explanation": "The documents are moderately similar (final score: 0.7800)..."
}
```

---

## Running the Service

```bash
pip install -r requirements.txt

# Start the server (local MiniLM, no API key needed)
uvicorn app.main:app --reload

# With Voyage AI embeddings (optional upgrade):
VOYAGE_API_KEY=your_key uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` for the UI.

---

## Running Tests

```bash
# Unit tests (pipeline modules)
pytest tests/test_pipeline.py -v

# Integration tests (API endpoints)
pytest tests/test_api.py -v

# All tests
pytest tests/ -v
```

---

## Design Decisions

### Why OCR (EasyOCR)?
Image documents must be converted to text before NLP can be applied. EasyOCR is pure
Python — no system Tesseract install required — and returns bounding boxes alongside
text, enabling layout heuristics. It handles common document fonts and runs on CPU.

### Why Semantic Embeddings?
Lexical methods (TF-IDF) fail on paraphrasing: "invoice total" vs "amount due" score
0.0 despite identical meaning. Embeddings capture semantic similarity regardless of
exact wording, and are robust to minor OCR character errors.

### Embedding Backend: Embed-Off Results

Two backends were evaluated: MiniLM-L6-v2 (local) and Voyage-3 (Voyage AI API).

The one-time embed-off (`python -m app.embeddings.evaluator`) compares both on:
- Near-duplicate invoice pair (expected high similarity ~0.95)
- Paraphrased ticket pair (expected medium ~0.75)
- Unrelated document pair (expected low ~0.10)

Winner is hardcoded as `ACTIVE_EMBEDDER` in `app/pipeline/similarity.py`.
Default: `'local'` (MiniLM) — runs offline with no API key.
To use Voyage-3: set `VOYAGE_API_KEY` and change `ACTIVE_EMBEDDER = "voyage"`.

Local MiniLM smoke test results:
- Near-duplicate invoice score: 0.8829
- Unrelated document score: 0.0736
- Correctly ranks similar > unrelated ✓

### Why Lightweight Layout Heuristics?
LayoutLM and similar models require GPU and large model weights. Our zone-based
heuristics (top/middle/bottom) capture structural alignment at near-zero compute
cost. Documents with totals in matching zones are more likely to be true duplicates.
The ±0.10 cap ensures layout nudges but never overrides content signals.

---

## Score Weights

```
final = 0.75 × semantic + 0.15 × lexical ± 0.10 × layout
```

| Signal | Weight | Justification |
|---|---|---|
| Semantic | 0.75 | Most robust — handles paraphrasing, OCR noise, synonyms |
| Lexical | 0.15 | Useful for exact matches; brittle against rephrasing |
| Layout | ±0.10 | Structural nudge; capped to prevent dominating score |

---

## Interpreting Scores

| Range | Interpretation |
|---|---|
| 0.85 – 1.00 | Highly similar — likely same document or near-duplicate |
| 0.65 – 0.84 | Moderately similar — same topic, some differences |
| 0.40 – 0.64 | Partially similar — related but significant divergence |
| 0.00 – 0.39 | Largely dissimilar — different documents |

> **Note:** The maximum score from content signals alone is 0.90 (semantic + lexical weights
> sum to 0.90). Scores above 0.90 require a positive layout adjustment. A score of 0.85–0.90 with
> no layout bonus indicates near-perfect content similarity.

---

## Trade-offs

### Accuracy vs Compute
- MiniLM inference: ~5ms/document on CPU
- Voyage-3: ~200ms (network round-trip)
- EasyOCR: 1–3s per image
- Fast enough for real-time API use; not for bulk batch processing

### OCR Limitations
- Character misreads (0↔O, 1↔I) can reduce lexical similarity
- Multi-column layouts are flattened, disrupting layout heuristics
- Noisy, low-contrast, or skewed images degrade recall significantly

### Heuristic Limitations
- Zone-based layout is a coarse approximation (no pixel-level spatial reasoning)
- Works best with conventional top-header/bottom-footer document structure
- Fails on unusual layouts (vertical text, circular diagrams, multi-column)

---

## Limitations

1. OCR errors propagate through the entire pipeline
2. Layout heuristics assume linear, top-to-bottom document structure
3. Voyage-3 requires API connectivity and is billed per token
4. No persistent storage — results are not saved between requests

---

## Future Improvements

- **LayoutLM / LayoutLMv3:** Full layout-aware transformer using bounding boxes
- **Vision-language models (GPT-4V, Claude):** Direct image understanding without OCR
- **Better OCR:** Google Cloud Vision or document-specific preprocessing pipelines
- **Async batch processing:** Queue-based pipeline for bulk document comparison
