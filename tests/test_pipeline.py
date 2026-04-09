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
