import io
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient


def _image_bytes(text: str = "Test text") -> bytes:
    """Create a minimal PNG image with text for testing."""
    img = Image.new("RGB", (400, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_client():
    from app.main import app
    return TestClient(app)


def test_compare_text_text_returns_scores():
    client = get_client()
    response = client.post(
        "/compare",
        data={"mode": "text-text"},
        files=[
            ("text_file_1", ("a.txt", io.BytesIO(b"Invoice INV-001 total $4500"), "text/plain")),
            ("text_file_2", ("b.txt", io.BytesIO(b"Invoice INV-001 amount $4500"), "text/plain")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert "scores" in body
    assert "final" in body["scores"]
    assert 0.0 <= body["scores"]["final"] <= 1.0


def test_compare_invalid_mode_returns_400():
    client = get_client()
    response = client.post(
        "/compare",
        data={"mode": "invalid"},
        files=[
            ("text_file_1", ("a.txt", io.BytesIO(b"hello"), "text/plain")),
            ("text_file_2", ("b.txt", io.BytesIO(b"world"), "text/plain")),
        ],
    )
    assert response.status_code == 400


def test_compare_html_format():
    client = get_client()
    response = client.post(
        "/compare?format=html",
        data={"mode": "text-text"},
        files=[
            ("text_file_1", ("a.txt", io.BytesIO(b"hello world document"), "text/plain")),
            ("text_file_2", ("b.txt", io.BytesIO(b"hello world document"), "text/plain")),
        ],
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_root_returns_html():
    client = get_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_compare_text_image_mode():
    client = get_client()
    img_bytes = _image_bytes("Invoice INV-001 total $4500")
    response = client.post(
        "/compare",
        data={"mode": "text-image"},
        files=[
            ("text_file", ("doc.txt", io.BytesIO(b"Invoice INV-001 total $4500"), "text/plain")),
            ("image_file", ("doc.png", io.BytesIO(img_bytes), "image/png")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert "scores" in body
    assert 0.0 <= body["scores"]["final"] <= 1.0
