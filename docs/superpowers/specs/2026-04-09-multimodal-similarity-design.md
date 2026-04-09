# Multimodal Document Similarity System — Design Spec
**Date:** 2026-04-09

---

## 1. Problem Statement

Compare two documents — one raw text, one image — and determine how similar they are in meaning, structure, and key information. Target enterprise use cases: invoice validation, document deduplication, ticket attachment verification.

The system must be lightweight, interpretable, and scalable. No heavy multimodal transformers.

---

## 2. Delivery Format

FastAPI REST API with a simple HTML frontend (Jinja2 template, served by FastAPI). Accepts multipart form uploads. Three comparison modes supported: text vs image, image vs image, text vs text.

---

## 3. Project Structure

```
multimodal-similarity/
├── app/
│   ├── main.py                    ← FastAPI app entry point
│   ├── api/
│   │   └── routes.py              ← GET /, POST /compare, POST /evaluate, GET /test
│   ├── pipeline/
│   │   ├── ocr.py                 ← extract_text_from_image() via easyocr
│   │   ├── preprocess.py          ← preprocess() normalization + noise removal
│   │   ├── features.py            ← compute_layout_features()
│   │   ├── similarity.py          ← compute_lexical_similarity(), compute_semantic_similarity(), compute_entity_similarity()
│   │   ├── scoring.py             ← combine_scores() weighted aggregation
│   │   └── explainability.py      ← build_explanation() → structured JSON + HTML
│   ├── embeddings/
│   │   ├── local.py               ← MiniLM via sentence-transformers
│   │   ├── claude_embed.py        ← Voyage AI embeddings via Anthropic SDK
│   │   └── evaluator.py           ← one-time embed-off comparison script
│   ├── entities/
│   │   ├── regex_extractor.py     ← dates, amounts, IDs via regex
│   │   └── spacy_extractor.py     ← names, orgs via spaCy NER
│   └── templates/
│       └── index.html             ← single-page HTML frontend (Jinja2)
├── tests/
│   ├── synthetic_pairs.py         ← built-in test document pairs with expected ranges
│   └── test_pipeline.py           ← unit tests per module
├── requirements.txt
└── README.md
```

---

## 4. API Endpoints

### `GET /`
- Serves the HTML frontend (`templates/index.html`)

### `POST /compare`
- **Input:** multipart form with fields depending on mode:
  - `mode`: `text-image` | `image-image` | `text-text`
  - `text-image`: `text_file` (.txt) + `image_file` (.png/.jpg/.pdf)
  - `image-image`: `image_file_1` + `image_file_2`
  - `text-text`: `text_file_1` + `text_file_2`
- **Query params:** `?format=json|html` (default: `json`)
- **Output:** structured JSON or HTML report

**Response shape:**
```json
{
  "mode": "text-image",
  "scores": {
    "lexical": 0.72,
    "semantic": 0.91,
    "entity": 0.65,
    "final": 0.79
  },
  "entities": {
    "matched": ["INV-2024-001", "15 Jan 2024", "$4,500.00"],
    "only_in_doc1": ["NET-30"],
    "only_in_doc2": ["GST 18%"]
  },
  "mismatches": ["Total amount differs: $4,500 vs $4,050"],
  "explanation": "Documents are highly similar in structure and entity overlap..."
}
```

### `POST /evaluate`
- **Input:** JSON array of labeled pairs `[{mode, files, expected_similarity}, ...]`
- **Output:** per-pair actual vs expected scores, overall RMSE

### `GET /test`
- Runs built-in synthetic test suite (one pair per mode)
- Returns pass/fail per pair with actual scores

---

## 5. Pipeline Architecture

The pipeline is mode-aware. Image inputs always pass through OCR first; text inputs skip it. After that, both documents are preprocessed identically and fed into the same feature/similarity/scoring stages.

```
Mode: text-image
  [text_file]  ──────────────────────────── preprocess() → clean_doc1
  [image_file] → OCR → raw_ocr_text ──────  preprocess() → clean_doc2

Mode: image-image
  [image_file_1] → OCR → preprocess() → clean_doc1
  [image_file_2] → OCR → preprocess() → clean_doc2

Mode: text-text
  [text_file_1] → preprocess() → clean_doc1
  [text_file_2] → preprocess() → clean_doc2

                Both texts ────────────────────────────────────┤
                                                               ▼
                                    ┌──────────────────────────────────────┐
                                    │          Feature Extraction          │
                                    │  - Semantic embeddings (winner)      │
                                    │  - TF-IDF vectors                    │
                                    │  - Regex entities (dates/amounts/IDs)│
                                    │  - spaCy NER (names/orgs)            │
                                    │  - Layout heuristics                 │
                                    └──────────────┬───────────────────────┘
                                                   ▼
                                    ┌──────────────────────────────────────┐
                                    │       Similarity Computation         │
                                    │  - lexical_similarity (TF-IDF cos)   │
                                    │  - semantic_similarity (winner embed)│
                                    │  - entity_similarity (overlap ratio) │
                                    └──────────────┬───────────────────────┘
                                                   ▼
                                    combine_scores() → weighted final score
                                                   ▼
                                    build_explanation() → JSON / HTML
```

---

## 6. Required Functions

| Function | Module | Purpose |
|---|---|---|
| `extract_text_from_image()` | `pipeline/ocr.py` | easyocr → raw text + line structure |
| `preprocess()` | `pipeline/preprocess.py` | normalize, remove noise, preserve IDs |
| `compute_layout_features()` | `pipeline/features.py` | line index, relative position, grouping |
| `compute_lexical_similarity()` | `pipeline/similarity.py` | TF-IDF cosine similarity |
| `compute_semantic_similarity()` | `pipeline/similarity.py` | embedding cosine similarity (winning model) |
| `compute_entity_similarity()` | `pipeline/similarity.py` | entity overlap ratio |
| `combine_scores()` | `pipeline/scoring.py` | weighted aggregation → final score |
| `build_explanation()` | `pipeline/explainability.py` | structured JSON + optional HTML |

---

## 7. Score Weighting

```
final = 0.40 × semantic + 0.35 × entity + 0.15 × lexical ± 0.10 × layout_adjustment
```

**Justification:**
- **Semantic (0.40):** Captures meaning, robust to paraphrasing and OCR noise — the most reliable signal
- **Entity (0.35):** Critical for correctness in invoice/ticket workflows — a $4,500 vs $4,050 discrepancy must dominate the score
- **Lexical (0.15):** Fast signal, useful for exact identifier matches, but brittle on paraphrasing
- **Layout (±0.10):** Reward for structurally aligned sections, penalty for mismatched key regions — approximates layout-aware models without the compute cost

---

## 8. OCR Strategy

- **Library:** `easyocr` — pure Python, no system Tesseract install required
- **Output:** raw text string + list of `(line_text, bounding_box)` tuples for layout heuristics
- **Limitations documented in code:**
  - Character misreads (0 vs O, 1 vs I) can corrupt entity extraction
  - Loss of table/column structure flattens layout to line order
  - Low-contrast or noisy images degrade accuracy significantly

---

## 9. Entity Extraction Strategy

Two-layer approach:

**Regex (`entities/regex_extractor.py`):**
- Dates: `\d{1,2}[/-]\d{1,2}[/-]\d{2,4}`, ISO format, written month variants
- Amounts: `\$[\d,]+(\.\d{2})?`, currency patterns
- IDs: alphanumeric identifiers matching patterns like `INV-\d+`, `TKT-\d+`

**spaCy NER (`entities/spacy_extractor.py`):**
- `en_core_web_sm` model
- Extracts PERSON, ORG, GPE entity types
- Complements regex for unstructured named entities

---

## 10. Layout Heuristics

Extracted from OCR line order (no vision model required):
- **Line index:** absolute position in document
- **Relative position:** top (first 33%), middle (33–66%), bottom (last 33%)
- **Section grouping:** lines within 2 positions of each other form a group

Used in scoring:
- Lines in matching relative positions get a similarity bonus
- Key entities appearing in mismatched positions (e.g., total amount at top vs bottom) trigger a penalty

---

## 11. Embedding Strategy

### One-Time Embed-Off (`embeddings/evaluator.py`)
Run once during development. Evaluates both models on 3 synthetic pairs:
- Near-duplicate invoice pair
- Paraphrased ticket pair
- Unrelated document pair

**Metrics:**
- Cosine similarity ranking correlation (does the model rank near-duplicates higher than unrelated docs?)
- Intra-class cohesion (similar docs cluster together)
- Inter-class separation (dissimilar docs score low)

**Output:** printed comparison report. Winner hardcoded into `pipeline/similarity.py`. Decision documented in README with metric values.

### Models
- **Local:** `sentence-transformers/all-MiniLM-L6-v2` — fast, runs offline, ~80MB
- **Claude API (Voyage):** `voyage-3` via Anthropic SDK — Voyage AI embeddings (Anthropic-acquired), richer semantic understanding, requires `ANTHROPIC_API_KEY`

---

## 12. Testing Plan

### Synthetic Suite (`GET /test`)
One pair per mode with expected similarity ranges:
| Pair | Mode | Expected Range |
|---|---|---|
| Near-duplicate invoice | text-image | > 0.85 |
| Paraphrased ticket | image-image | 0.55 – 0.80 |
| Unrelated documents | text-text | < 0.30 |

### Custom Evaluation (`POST /evaluate`)
User-provided labeled pairs → per-pair RMSE report.

### Unit Tests (`tests/test_pipeline.py`)
- OCR output is a non-empty string
- Preprocess preserves numeric IDs and dates
- Regex extractor finds known amounts/dates in test strings
- spaCy extractor finds known org/person names
- combine_scores output is in [0, 1]

---

## 13. Frontend UI (`app/templates/index.html`)

Single-page HTML served at `GET /`. No JS framework — plain HTML + CSS + vanilla JS.

**Layout:**
- Mode dropdown at the top: `Text vs Image` | `Image vs Image` | `Text vs Text`
- File upload fields update dynamically based on selected mode (JS shows/hides fields)
- Submit button triggers `POST /compare` via `fetch()`, results render inline below
- "Run Tests" button calls `GET /test`, shows pass/fail table inline

**Results section (rendered after compare):**
- Score bars (color-coded: green > 0.75, yellow 0.50–0.75, red < 0.50) for lexical, semantic, entity, final
- Entity comparison table: matched | doc1 only | doc2 only
- Mismatch list
- Explanation paragraph

## 14. HTML Report

When `?format=html` is passed to `/compare` directly (API use):
- Rendered summary with color-coded score bars
- Side-by-side entity comparison table
- Highlighted mismatch descriptions
- Human-readable explanation paragraph

---

## 15. Dependencies

```
fastapi
uvicorn
easyocr
sentence-transformers
anthropic
voyageai           # Voyage AI embeddings via Anthropic
scikit-learn
spacy              # + python -m spacy download en_core_web_sm
Pillow
python-multipart
jinja2             # for HTML report rendering
```

---

## 16. Out of Scope

- Authentication / API keys for the FastAPI service itself
- Persistent storage of comparison results
- PDF text extraction (PDFs treated as images via OCR only)
- Batch async processing
