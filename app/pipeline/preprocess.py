"""
Text preprocessing module: normalizes both raw text and OCR-extracted text
into a consistent form for downstream feature extraction.

Applied to BOTH documents (original text and OCR output) to ensure a level
playing field — both go through identical normalization.

Trade-offs:
- We lowercase everything: safe, improves token matching across case variations.
- We normalize unicode (NFKC): converts curly quotes, half-width chars, etc.
- We collapse whitespace: OCR often introduces extra spaces between characters.
- We preserve numbers, dashes, dots, commas, $, /: these form dates, amounts,
  and IDs. Stripping them would harm lexical matching.
- We do NOT stem or lemmatize: stemming would mangle ID patterns like INV-001.
- We do NOT remove stopwords: they contribute to TF-IDF and layout coherence.
"""
import re
import unicodedata


def preprocess(text: str) -> str:
    """
    Normalize and lightly clean text for similarity comparison.

    Preserves: numbers, dates, amounts ($), identifiers (INV-001), slashes,
    colons, commas, hyphens, @ signs, # signs.
    Removes: unicode noise, control characters, symbols not in structured tokens.

    Args:
        text: Raw text string (from file or OCR output).

    Returns:
        Cleaned, lowercased, whitespace-normalized string.
    """
    # Step 1: Unicode normalization — converts smart quotes, ligatures, etc.
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Lowercase — reduces vocabulary size without losing information
    text = text.lower()

    # Step 3: Collapse all whitespace (spaces, tabs, newlines) to single space
    text = re.sub(r"\s+", " ", text)

    # Step 4: Remove characters that are not word chars, whitespace, or
    # structured token punctuation. Kept: - . , $ / : @ # (for IDs/dates/amounts)
    text = re.sub(r"[^\w\s\-\.\,\$\/\:\@\#]", "", text)

    return text.strip()
