"""
OCR module: extracts text and line structure from images using EasyOCR.

OCR Limitations (important for downstream accuracy):
- Character misreads: '0' vs 'O', '1' vs 'I', '5' vs 'S' are common. These
  corrupt entity extraction (e.g., invoice ID "INV-001" becomes "INV-OO1").
- Structure loss: table rows and columns are flattened to a single line order.
  Column alignment is not preserved — multi-column invoices become garbled.
- Low-quality images: blur, skew, low contrast significantly degrade recall.
  Pre-processing (contrast enhancement, deskewing) would help but is out of scope.

These limitations propagate into similarity scoring: OCR errors reduce entity
match rates and introduce false lexical differences even between identical documents.
"""
import easyocr
from typing import List, Tuple

# Lazy-loaded singleton reader — EasyOCR model load is expensive (~2s on first call)
_reader = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        # gpu=False: safe default; set True if CUDA is available for speed
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text_from_image(image_path: str) -> Tuple[str, List[Tuple[str, list]]]:
    """
    Extract text from an image file using EasyOCR.

    Args:
        image_path: Absolute or relative path to a .png / .jpg / .pdf image.

    Returns:
        raw_text: Full extracted text joined by newlines, top-to-bottom order.
        lines: List of (line_text, bounding_box) tuples preserving spatial order.
               bounding_box is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (4 corner points).
               Used by compute_layout_features() to infer document zones.
    """
    reader = _get_reader()
    # results: list of (bbox, text, confidence_score)
    results = reader.readtext(image_path)

    # Sort top-to-bottom by the y-coordinate of the top-left corner
    results_sorted = sorted(results, key=lambda r: r[0][0][1])

    lines = [(text, bbox) for bbox, text, _conf in results_sorted]
    raw_text = "\n".join(text for text, _ in lines)
    return raw_text, lines
