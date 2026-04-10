import math
import os
import tempfile
from typing import List as TypingList, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from tests.synthetic_pairs import SYNTHETIC_PAIRS
from app.pipeline import explainability, features, ocr
from app.pipeline import preprocess as prep
from app.pipeline import scoring, similarity

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=os.path.abspath(_TEMPLATES_DIR))
router = APIRouter()


class EvalPair(BaseModel):
    mode: str
    doc1_text: str
    doc2_text: str
    expected: float


async def _save_upload(file: UploadFile) -> str:
    """Save an uploaded file to a temp path. Caller must unlink when done."""
    suffix = os.path.splitext(file.filename or "")[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        return tmp.name


def _extract(path: str, is_image: bool):
    """
    Extract clean text and layout features from a file.
    Image files go through OCR; text files are read directly.
    Returns (clean_text, layout_features).
    """
    if is_image:
        raw_text, lines = ocr.extract_text_from_image(path)
        layout = features.compute_layout_features(lines)
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        layout = features.compute_layout_features_from_text(raw_text)
    return prep.preprocess(raw_text), layout


def _run_pipeline(text1: str, layout1, text2: str, layout2, mode: str) -> dict:
    """Run the full similarity pipeline on two preprocessed texts."""
    lex = similarity.compute_lexical_similarity(text1, text2)
    sem = similarity.compute_semantic_similarity(text1, text2)
    layout_adj = features.compute_layout_adjustment(layout1, layout2)
    final = scoring.combine_scores(lex, sem, layout_adj)

    return explainability.build_explanation(lex, sem, final, mode)


@router.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.post("/compare")
async def compare(
    mode: str = Form(...),
    format: str = Query(default="json"),
    text_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    text_file_1: Optional[UploadFile] = File(None),
    text_file_2: Optional[UploadFile] = File(None),
    image_file_1: Optional[UploadFile] = File(None),
    image_file_2: Optional[UploadFile] = File(None),
):
    """
    Compare two documents. Mode determines which file fields are used:
    - text-image: text_file + image_file
    - image-image: image_file_1 + image_file_2
    - text-text:  text_file_1 + text_file_2
    """
    paths = []
    try:
        if mode == "text-image":
            if not text_file or not image_file:
                raise HTTPException(400, "text-image mode requires text_file and image_file")
            p1 = await _save_upload(text_file)
            paths.append(p1)
            p2 = await _save_upload(image_file)
            paths.append(p2)
            t1, l1 = _extract(p1, is_image=False)
            t2, l2 = _extract(p2, is_image=True)

        elif mode == "image-image":
            if not image_file_1 or not image_file_2:
                raise HTTPException(400, "image-image mode requires image_file_1 and image_file_2")
            p1 = await _save_upload(image_file_1)
            paths.append(p1)
            p2 = await _save_upload(image_file_2)
            paths.append(p2)
            t1, l1 = _extract(p1, is_image=True)
            t2, l2 = _extract(p2, is_image=True)

        elif mode == "text-text":
            if not text_file_1 or not text_file_2:
                raise HTTPException(400, "text-text mode requires text_file_1 and text_file_2")
            p1 = await _save_upload(text_file_1)
            paths.append(p1)
            p2 = await _save_upload(text_file_2)
            paths.append(p2)
            t1, l1 = _extract(p1, is_image=False)
            t2, l2 = _extract(p2, is_image=False)

        else:
            raise HTTPException(400, f"Invalid mode '{mode}'. Use: text-image, image-image, text-text")

        result = _run_pipeline(t1, l1, t2, l2, mode)

        if format == "html":
            return HTMLResponse(explainability.render_html(result))
        return JSONResponse(result)

    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass


@router.get("/test")
async def run_tests():
    """
    Run the built-in synthetic test suite.
    Executes one pair per mode and checks scores against expected ranges.
    """
    results = []
    for pair in SYNTHETIC_PAIRS:
        t1 = prep.preprocess(pair["doc1_text"])
        t2 = prep.preprocess(pair["doc2_text"])
        l1 = features.compute_layout_features_from_text(pair["doc1_text"])
        l2 = features.compute_layout_features_from_text(pair["doc2_text"])

        pipeline_result = _run_pipeline(t1, l1, t2, l2, pair["mode"])
        final = pipeline_result["scores"]["final"]

        results.append({
            "label": pair["label"],
            "mode": pair["mode"],
            "final_score": final,
            "expected_range": f"{pair['expected_min']} \u2013 {pair['expected_max']}",
            "passed": pair["expected_min"] <= final <= pair["expected_max"],
            "scores": pipeline_result["scores"],
        })

    all_passed = all(r["passed"] for r in results)
    return {"all_passed": all_passed, "results": results}


@router.post("/evaluate")
async def evaluate(pairs: TypingList[EvalPair]):
    """
    Evaluate the pipeline against labeled document pairs.
    Accepts text content directly for programmatic benchmarking.
    Returns per-pair actual vs expected scores and overall RMSE.
    """
    pair_results = []
    squared_errors = []

    for pair in pairs:
        t1 = prep.preprocess(pair.doc1_text)
        t2 = prep.preprocess(pair.doc2_text)
        l1 = features.compute_layout_features_from_text(pair.doc1_text)
        l2 = features.compute_layout_features_from_text(pair.doc2_text)

        pipeline_result = _run_pipeline(t1, l1, t2, l2, pair.mode)
        actual = pipeline_result["scores"]["final"]
        error = actual - pair.expected
        squared_errors.append(error ** 2)

        pair_results.append({
            "mode": pair.mode,
            "actual": round(actual, 4),
            "expected": pair.expected,
            "error": round(error, 4),
            "scores": pipeline_result["scores"],
        })

    rmse = math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else 0.0
    return {"rmse": round(rmse, 4), "pairs": pair_results}
