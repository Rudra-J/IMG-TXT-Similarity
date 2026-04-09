import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.entities import regex_extractor, spacy_extractor
from app.pipeline import explainability, features, ocr
from app.pipeline import preprocess as prep
from app.pipeline import scoring, similarity

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=os.path.abspath(_TEMPLATES_DIR))
router = APIRouter()


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
    # Merge regex (structured tokens) and spaCy (named entities) for each doc
    e1 = {**regex_extractor.extract_entities(text1), **spacy_extractor.extract_entities(text1)}
    e2 = {**regex_extractor.extract_entities(text2), **spacy_extractor.extract_entities(text2)}

    lex = similarity.compute_lexical_similarity(text1, text2)
    sem = similarity.compute_semantic_similarity(text1, text2)
    ent = similarity.compute_entity_similarity(e1, e2)
    layout_adj = features.compute_layout_adjustment(layout1, layout2)
    final = scoring.combine_scores(lex, sem, ent, layout_adj)

    return explainability.build_explanation(lex, sem, ent, final, e1, e2, mode)


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
