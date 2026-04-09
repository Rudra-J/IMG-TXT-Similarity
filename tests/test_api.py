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


def test_get_test_returns_results():
    client = get_client()
    response = client.get("/test")
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert len(body["results"]) == 3


def test_get_test_has_pass_fail():
    client = get_client()
    response = client.get("/test")
    body = response.json()
    for result in body["results"]:
        assert "label" in result
        assert "mode" in result
        assert "final_score" in result
        assert "passed" in result


def test_evaluate_returns_rmse():
    client = get_client()
    payload = [
        {
            "mode": "text-text",
            "doc1_text": "Invoice INV-001 total $4500",
            "doc2_text": "Invoice INV-001 amount $4500",
            "expected": 0.9,
        },
        {
            "mode": "text-text",
            "doc1_text": "Invoice INV-001 total $4500",
            "doc2_text": "The weather is sunny today in London",
            "expected": 0.1,
        },
    ]
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "rmse" in body
    assert "pairs" in body
    assert len(body["pairs"]) == 2
    for pair in body["pairs"]:
        assert "actual" in pair
        assert "expected" in pair
        assert "error" in pair
