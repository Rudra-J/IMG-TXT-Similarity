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
