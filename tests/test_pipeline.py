import os
import pytest
from PIL import Image, ImageDraw, ImageFont


def _make_test_image(text: str, path: str) -> str:
    """Create a simple white PNG with black text for OCR testing."""
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), text, fill=(0, 0, 0))
    img.save(path)
    return path


def test_extract_text_from_image_returns_nonempty_string(tmp_path):
    from app.pipeline.ocr import extract_text_from_image
    img_path = str(tmp_path / "test.png")
    _make_test_image("Hello World", img_path)
    raw_text, lines = extract_text_from_image(img_path)
    assert isinstance(raw_text, str)
    assert len(raw_text.strip()) > 0


def test_extract_text_from_image_returns_lines(tmp_path):
    from app.pipeline.ocr import extract_text_from_image
    img_path = str(tmp_path / "test.png")
    _make_test_image("Hello World", img_path)
    raw_text, lines = extract_text_from_image(img_path)
    assert isinstance(lines, list)
    # Each line is (text_str, bbox)
    for text, bbox in lines:
        assert isinstance(text, str)


def test_preprocess_lowercases():
    from app.pipeline.preprocess import preprocess
    assert preprocess("Hello WORLD") == "hello world"


def test_preprocess_normalizes_whitespace():
    from app.pipeline.preprocess import preprocess
    assert preprocess("foo   bar\t\nbaz") == "foo bar baz"


def test_preprocess_preserves_invoice_id():
    from app.pipeline.preprocess import preprocess
    # IDs like INV-001 must survive preprocessing — they are critical entities
    result = preprocess("Invoice INV-001 due")
    assert "inv-001" in result


def test_preprocess_preserves_amount():
    from app.pipeline.preprocess import preprocess
    result = preprocess("Total: $4,500.00")
    assert "4,500.00" in result or "4500" in result


def test_preprocess_preserves_date():
    from app.pipeline.preprocess import preprocess
    result = preprocess("Date: 2024-01-15")
    assert "2024-01-15" in result


def test_compute_layout_features_zones():
    from app.pipeline.features import compute_layout_features
    lines = [(f"line {i}", []) for i in range(9)]
    features = compute_layout_features(lines)
    assert features[0]["zone"] == "top"
    assert features[4]["zone"] == "middle"
    assert features[8]["zone"] == "bottom"


def test_compute_layout_features_line_index():
    from app.pipeline.features import compute_layout_features
    lines = [("a", []), ("b", []), ("c", [])]
    features = compute_layout_features(lines)
    assert features[0]["line_index"] == 0
    assert features[2]["line_index"] == 2


def test_compute_layout_features_empty():
    from app.pipeline.features import compute_layout_features
    assert compute_layout_features([]) == []


def test_compute_layout_features_from_text():
    from app.pipeline.features import compute_layout_features_from_text
    text = "line one\nline two\nline three"
    features = compute_layout_features_from_text(text)
    assert len(features) == 3
    assert features[0]["text"] == "line one"


def test_compute_layout_adjustment_identical_docs():
    from app.pipeline.features import compute_layout_features_from_text, compute_layout_adjustment
    text = "header\nbody line 1\nbody line 2\nfooter total"
    f = compute_layout_features_from_text(text)
    adj = compute_layout_adjustment(f, f)
    # Identical layout should give a positive or zero adjustment
    assert adj >= 0.0


def test_compute_layout_adjustment_bounds():
    from app.pipeline.features import compute_layout_features_from_text, compute_layout_adjustment
    f1 = compute_layout_features_from_text("a\nb\nc\nd\ne\nf")
    f2 = compute_layout_features_from_text("x\ny\nz")
    adj = compute_layout_adjustment(f1, f2)
    assert -0.10 <= adj <= 0.10


def test_regex_extractor_finds_amount():
    from app.entities.regex_extractor import extract_entities
    result = extract_entities("Total due: $4,500.00")
    assert any("4,500.00" in a or "4500" in a for a in result["amounts"])


def test_regex_extractor_finds_iso_date():
    from app.entities.regex_extractor import extract_entities
    result = extract_entities("Invoice date: 2024-01-15")
    assert any("2024-01-15" in d for d in result["dates"])


def test_regex_extractor_finds_slash_date():
    from app.entities.regex_extractor import extract_entities
    result = extract_entities("Due date: 01/15/2024")
    assert len(result["dates"]) > 0


def test_regex_extractor_finds_invoice_id():
    from app.entities.regex_extractor import extract_entities
    result = extract_entities("Reference: INV-2024-001")
    assert any("INV-2024-001" in i for i in result["ids"])


def test_regex_extractor_empty_text():
    from app.entities.regex_extractor import extract_entities
    result = extract_entities("")
    assert result == {"dates": [], "amounts": [], "ids": []}


def test_spacy_extractor_finds_org():
    from app.entities.spacy_extractor import extract_entities
    result = extract_entities("Bill to: Acme Corporation. Contact: John Smith.")
    assert any("Acme" in org for org in result["orgs"]) or len(result["orgs"]) > 0


def test_spacy_extractor_finds_person():
    from app.entities.spacy_extractor import extract_entities
    result = extract_entities("Prepared by: John Smith, Senior Accountant.")
    assert any("John" in p or "Smith" in p for p in result["persons"])


def test_spacy_extractor_returns_correct_keys():
    from app.entities.spacy_extractor import extract_entities
    result = extract_entities("Hello world")
    assert "persons" in result
    assert "orgs" in result
    assert "locations" in result


def test_spacy_extractor_empty_text():
    from app.entities.spacy_extractor import extract_entities
    result = extract_entities("")
    assert result == {"persons": [], "orgs": [], "locations": []}


def test_local_embed_single_returns_vector():
    from app.embeddings.local import embed_single
    import numpy as np
    vec = embed_single("This is a test sentence.")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert len(vec) == 384  # MiniLM-L6-v2 output dimension


def test_local_embed_single_is_normalized():
    from app.embeddings.local import embed_single
    import numpy as np
    vec = embed_single("Normalize me.")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5


def test_local_embed_similar_texts_score_higher():
    from app.embeddings.local import embed_single
    import numpy as np
    v1 = embed_single("The invoice total is five thousand dollars.")
    v2 = embed_single("Total amount due: $5,000.")
    v3 = embed_single("The weather today is sunny and warm.")
    sim_related = float(np.dot(v1, v2))
    sim_unrelated = float(np.dot(v1, v3))
    assert sim_related > sim_unrelated


def test_voyage_embed_single_returns_vector():
    """Requires VOYAGE_API_KEY env var. Skip if not set."""
    import os, pytest
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")
    from app.embeddings.claude_embed import embed_single
    import numpy as np
    vec = embed_single("This is a test sentence.")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert len(vec) > 0


def test_voyage_embed_single_is_normalized():
    import os, pytest
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")
    from app.embeddings.claude_embed import embed_single
    import numpy as np
    vec = embed_single("Normalize me.")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4
