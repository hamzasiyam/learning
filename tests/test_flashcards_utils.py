import pandas as pd

from modules.apps.flashcards_utils import (
    ensure_highlight_note_columns,
    escape_formula_cells,
    filename_from_title,
    normalize_note_prefix,
)


def test_filename_from_title_sanitizes_and_falls_back():
    assert filename_from_title("  ") == "merged"
    assert filename_from_title('A:Book*Title?') == "ABookTitle"


def test_escape_formula_cells():
    assert escape_formula_cells("=1+1") == "'=1+1"
    assert escape_formula_cells("hello") == "hello"


def test_normalize_note_prefix():
    assert normalize_note_prefix(".note") == "Answer: .note"
    assert normalize_note_prefix("note") == "note"


def test_ensure_highlight_note_columns_with_numbered_columns():
    df = pd.DataFrame([["h1", "n1"], ["h2", "n2"]])
    out = ensure_highlight_note_columns(df, "sample.csv")
    assert list(out.columns) == ["Highlight", "Note"]
    assert out.iloc[0]["Highlight"] == "h1"

