"""Tests for link parsing and CSV text normalization in ``link_csv_utils``."""

from modules.apps.link_csv_utils import (
    extract_page_number,
    format_note_bullets,
    note_or_placeholder,
    parse_link_line,
    sanitize_csv_text,
)


def test_extract_page_number_from_various_inputs():
    """Page numbers are parsed from ``Page-N`` patterns or trailing digits.

    Returns:
        None: Asserts only.
    """
    assert extract_page_number("https://x.com/Page-12.png") == "12"
    assert extract_page_number("chapter_44.jpg") == "44"
    assert extract_page_number("no-number") == ""


def test_sanitize_and_note_helpers():
    """Sanitization trims text; placeholders honor optional flags.

    Returns:
        None: Asserts only.
    """
    assert sanitize_csv_text("  test  ") == "test"
    assert note_or_placeholder("") == ".addnote"
    assert note_or_placeholder("", use_placeholder=False) == ""
    assert note_or_placeholder("", use_placeholder=True, placeholder_text="custom") == "custom"
    assert "• item" in format_note_bullets("•item")


def test_parse_link_line_markdown_and_plain_url():
    """Markdown image syntax and plain URLs both yield a single URL string.

    Returns:
        None: Asserts only.
    """
    md = "![img](https://i.ibb.co/abc/Page-3.png)"
    assert parse_link_line(md) == "https://i.ibb.co/abc/Page-3.png"
    assert parse_link_line("https://example.com/x.png trailing") == "https://example.com/x.png"
