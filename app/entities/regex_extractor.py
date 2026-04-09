"""
Regex-based entity extractor: finds structured information critical for
enterprise document comparison.

Why entity matching is critical:
- Lexical and semantic similarity operate on surface form and meaning.
  They cannot detect that $4,500 != $4,050 — both are "an amount."
- In invoice validation, a single digit difference in an amount or a
  transposed ID digit means the documents are substantively different,
  regardless of how similar they look overall.
- Entity similarity therefore carries a 0.35 weight in the final score —
  higher than lexical (0.15) — to ensure factual correctness dominates.
"""
import re
from typing import Dict, List

# Dates: slash/hyphen numeric, ISO, and written-month formats
_DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?"
    r"|dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
]

# Amounts: dollar signs with optional thousands separators and decimal places
_AMOUNT_PATTERNS = [
    r"\$[\d,]+(?:\.\d{2})?",
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:usd|eur|gbp)\b",
]

# IDs: uppercase letter prefix followed by digits, with optional hyphens
_ID_PATTERNS = [
    r"\b[A-Z]{2,10}-\d[\d\-]*\b",   # INV-2024-001, TKT-555
    r"\b[A-Z]{2,6}\d{4,}\b",        # REF20240115
]


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract dates, monetary amounts, and document IDs from text.

    Args:
        text: Preprocessed or raw text string.

    Returns:
        Dict with keys 'dates', 'amounts', 'ids', each a deduplicated list.
    """
    if not text:
        return {"dates": [], "amounts": [], "ids": []}

    entities: Dict[str, List[str]] = {"dates": [], "amounts": [], "ids": []}

    for pattern in _DATE_PATTERNS:
        entities["dates"].extend(re.findall(pattern, text, re.IGNORECASE))

    for pattern in _AMOUNT_PATTERNS:
        entities["amounts"].extend(re.findall(pattern, text, re.IGNORECASE))

    for pattern in _ID_PATTERNS:
        entities["ids"].extend(re.findall(pattern, text))

    # Deduplicate preserving first-occurrence order
    for key in entities:
        seen = set()
        deduped = []
        for item in entities[key]:
            norm = item.lower().strip()
            if norm not in seen:
                seen.add(norm)
                deduped.append(item.strip())
        entities[key] = deduped

    return entities
