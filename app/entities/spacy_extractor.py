"""
spaCy NER entity extractor: complements regex extraction with named entities
that don't follow fixed patterns (company names, person names, locations).

Regex handles structured tokens (INV-001, $4,500). spaCy handles the unstructured
layer: "Acme Corp", "John Smith", "London" — entities critical for verifying
that two documents refer to the same parties and context.
"""
import spacy
from typing import Dict, List

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        # en_core_web_sm: small English model, fast inference, ~12MB
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities (persons, organizations, locations) using spaCy NER.

    Args:
        text: Preprocessed or raw text string.

    Returns:
        Dict with keys 'persons', 'orgs', 'locations', each a deduplicated list.
    """
    if not text or not text.strip():
        return {"persons": [], "orgs": [], "locations": []}

    nlp = _get_nlp()
    # Guard against very long texts exceeding spaCy's default max_length
    doc = nlp(text[:100_000])

    entities: Dict[str, List[str]] = {"persons": [], "orgs": [], "locations": []}
    seen: Dict[str, set] = {"persons": set(), "orgs": set(), "locations": set()}

    for ent in doc.ents:
        text_norm = ent.text.strip().lower()
        if ent.label_ == "PERSON" and text_norm not in seen["persons"]:
            entities["persons"].append(ent.text.strip())
            seen["persons"].add(text_norm)
        elif ent.label_ == "ORG" and text_norm not in seen["orgs"]:
            entities["orgs"].append(ent.text.strip())
            seen["orgs"].add(text_norm)
        elif ent.label_ in ("GPE", "LOC") and text_norm not in seen["locations"]:
            entities["locations"].append(ent.text.strip())
            seen["locations"].add(text_norm)

    return entities
