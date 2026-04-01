"""Utility helpers for preparing and sanitizing flashcard CSV data."""

import re

FORMULA_PREFIXES = ("=", "+", "-", "@")


def filename_from_title(title: str, max_length: int = 200) -> str:
    """Create a safe output filename from a user-provided title.

    Args:
        title (str): Human-readable book/article title entered by the user.
        max_length (int): Maximum allowed filename length.

    Returns:
        str: A sanitized filename safe for most filesystems.
    """
    # If no meaningful title is provided, return a stable default name.
    if not title or not title.strip():
        return "merged"
    # Remove characters that are invalid in Windows file names.
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title.strip())
    # Collapse repeated whitespace and trim trailing dots/spaces.
    s = re.sub(r"\s+", " ", s).strip(" .")
    # If sanitization empties the text, fall back to a safe default.
    if not s:
        return "merged"
    # Enforce a maximum filename length.
    if len(s) > max_length:
        s = s[:max_length].rstrip(" .")
    return s or "merged"


def escape_formula_cells(value):
    """Prevent spreadsheet formula execution by escaping dangerous prefixes.

    Args:
        value: Any scalar cell value from a dataframe column.

    Returns:
        The original value or a prefixed string (with `'`) for safety.
    """
    # If the cell starts with a formula trigger, prefix with apostrophe.
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    # Otherwise, keep the value unchanged.
    return value


def normalize_note_prefix(value):
    """Normalize note values that begin with a dot into answer-prefixed text.

    Args:
        value: A note cell value.

    Returns:
        str | Any: Updated note text when applicable; otherwise original value.
    """
    # If a note starts with '.', convert it to the explicit "Answer:" format.
    if isinstance(value, str) and value.lstrip().startswith("."):
        return "Answer: " + value
    # If the rule does not apply, return the value as-is.
    return value


def ensure_highlight_note_columns(df, file_path: str):
    """Ensure dataframe has `Highlight` and `Note` columns in the expected order.

    Args:
        df: Input pandas dataframe loaded from CSV.
        file_path (str): Source file path (used in validation error message).

    Returns:
        DataFrame: A dataframe restricted/renamed to `Highlight` and `Note`.

    Raises:
        ValueError: If the dataframe has fewer than two usable columns.
    """
    # If named columns are already present, keep only the required pair.
    if {"Highlight", "Note"}.issubset(df.columns):
        return df[["Highlight", "Note"]]
    # If there are fewer than 2 columns, we cannot map to expected schema.
    if df.shape[1] < 2:
        raise ValueError(f"{file_path} must have at least 2 columns to map to 'Highlight' and 'Note'")
    # Otherwise map the first two columns to the expected field names.
    df = df.iloc[:, :2].copy()
    df.columns = ["Highlight", "Note"]
    return df

