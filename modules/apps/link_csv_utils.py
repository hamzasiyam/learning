"""Helpers for parsing links and normalizing CSV Location/Note content."""

import csv
import re

FORMULA_PREFIXES = ("=", "+", "-", "@")


def extract_page_number(text: str) -> str:
    """Extract a page number from free text or URL-like strings.

    Args:
        text (str): Source text that may contain patterns like `Page-12`.

    Returns:
        str: Page number as digits, or empty string if not found.
    """
    # If text is empty, no page number can be extracted.
    if not text:
        return ""
    # Prefer explicit "page" patterns first (more semantically correct).
    match = re.search(r"page[-_\s]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback to trailing number when "page" label is missing.
    trailing_num = re.search(r"(\d+)(?:\.[a-zA-Z0-9]+)?$", text.strip())
    return trailing_num.group(1) if trailing_num else ""


def sanitize_csv_text(value: str) -> str:
    """Normalize text for safe and clean CSV output.

    Args:
        value (str): Input text that may include control chars or formulas.

    Returns:
        str: Cleaned one-line text safe to store in CSV.
    """
    # Empty input is normalized to an empty output string.
    if not value:
        return ""
    cleaned = str(value)
    # Fix common mojibake bullet symbol and remove control characters.
    cleaned = cleaned.replace("â€¢", "•")
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", cleaned)
    # Collapse whitespace to keep CSV cells readable.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Escape spreadsheet formulas for security.
    if cleaned.startswith(FORMULA_PREFIXES):
        cleaned = "'" + cleaned
    return cleaned


def format_note_bullets(note: str) -> str:
    """Format bullet notes with blank lines for readability.

    Args:
        note (str): Raw note text.

    Returns:
        str: Formatted note text.
    """
    # If note is empty, return empty text and let caller handle placeholders.
    if not note:
        return ""
    # Normalize first so we do bullet formatting on clean text.
    text = sanitize_csv_text(note)
    # Insert a blank line before each bullet for better spacing.
    text = re.sub(r"\s*•\s*", "\n\n• ", text)
    return text.lstrip()


def note_or_placeholder(note: str, use_placeholder: bool = True, placeholder_text: str = ".addnote") -> str:
    """Return a formatted note or default placeholder text.

    Args:
        note (str): Input note text.
        use_placeholder (bool): If True, use placeholder when note is blank.
        placeholder_text (str): Placeholder to use when enabled and blank.

    Returns:
        str: Formatted note, placeholder, or empty string when disabled.
    """
    # Apply canonical formatting first.
    formatted = format_note_bullets(note)
    # If no content remains after formatting, use optional placeholder.
    if formatted:
        return formatted
    return sanitize_csv_text(placeholder_text) if use_placeholder else ""


def parse_link_line(line: str) -> str:
    """Parse one pasted line and extract an image URL.

    Args:
        line (str): Raw line that may be plain URL or markdown image syntax.

    Returns:
        str: Extracted URL or empty string if not parseable.
    """
    raw = line.strip()
    # If the line is empty, return nothing.
    if not raw:
        return ""
    # First try markdown image syntax ![](url).
    md_match = re.search(r"!\[.*?\]\((https?://[^\s)]+)\)", raw)
    if md_match:
        return md_match.group(1).strip()
    # Otherwise treat first token as the URL.
    return raw.split()[0].strip()


def read_location_note_rows(csv_path: str):
    """Read a CSV file and extract normalized Location/Note rows.

    Args:
        csv_path (str): Path to source CSV file.

    Returns:
        list[dict[str, str]]: Rows with normalized `Location` and `Note`.
    """
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        # If the CSV has no header, there are no named fields to map.
        if not reader.fieldnames:
            return []
        # Resolve header names case-insensitively.
        names = {name.strip().lower(): name for name in reader.fieldnames if name}
        location_key = names.get("location")
        note_key = names.get("note")
        rows = []
        for row in reader:
            # Sanitize both fields before downstream matching/merging.
            location = sanitize_csv_text(row.get(location_key, "") if location_key else "")
            note = format_note_bullets(row.get(note_key, "") if note_key else "")
            # Skip fully empty rows to avoid noisy output.
            if not location and not note:
                continue
            rows.append({"Location": location, "Note": note})
        return rows

