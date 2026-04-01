"""Shared filesystem and formatting helpers used across tools."""

import os
import re
import tempfile

_WIN_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


def sanitize_filename(text: str) -> str:
    """Remove filesystem-invalid characters from raw text.

    Args:
        text (str): Raw text to sanitize.

    Returns:
        str: Sanitized text.
    """
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip()


def safe_filename(title: str, max_length: int = 200) -> str:
    """Build a safe filename from title text with platform-aware defaults.

    Args:
        title (str): Candidate filename text from user input.
        max_length (int): Maximum allowed output length.

    Returns:
        str: Safe filename suitable for output file creation.
    """
    # If title is blank, use a stable default.
    if not title or not title.strip():
        return "Combined"
    # Clean invalid characters and normalize spacing.
    s = sanitize_filename(title)
    s = re.sub(r"\s+", " ", s).strip(" .")
    # If sanitization removes all text, fall back.
    if not s:
        return "Combined"
    # Truncate overly long names.
    if len(s) > max_length:
        s = s[:max_length].rstrip(" .")
    if not s:
        return "Combined"
    # Handle Windows reserved device names.
    if s.upper() in _WIN_RESERVED:
        return f"{s}_document"
    return s


def format_seconds(s: float) -> str:
    """Format elapsed seconds into `M:SS` or `H:MM:SS`.

    Args:
        s (float): Duration in seconds.

    Returns:
        str: Human-readable duration string.
    """
    # Guard against negative durations from noisy progress estimates.
    s = max(0, int(s))
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    # If hour component exists, return full format.
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    # Otherwise use short minute format.
    return f"{minutes}:{seconds:02d}"


def format_bytes(num_bytes: float) -> str:
    """Convert byte count into a readable unit string.

    Args:
        num_bytes (float): Number of bytes.

    Returns:
        str: Human-friendly size text (B, KB, MB, or GB).
    """
    # Return early for zero/negative values.
    if num_bytes <= 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        # If value is small enough for this unit, format and return.
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} GB"


def make_unique_output_dir(base_dir: str) -> str:
    """Generate a unique directory path by appending incremental suffixes.

    Args:
        base_dir (str): Desired base output directory.

    Returns:
        str: Non-existing directory path.
    """
    # If the base directory does not exist yet, it is already unique.
    if not os.path.exists(base_dir):
        return base_dir
    n = 2
    while True:
        candidate = f"{base_dir}_{n}"
        # Return the first available suffix.
        if not os.path.exists(candidate):
            return candidate
        n += 1


def make_temp_dir_near_output(output_pdf: str) -> str:
    """Create a temporary folder near the target PDF path.

    Args:
        output_pdf (str): Destination PDF file path.

    Returns:
        str: Path to created temporary directory.
    """
    # Build a stable parent temp folder near output for easier cleanup.
    base_dir = os.path.dirname(os.path.abspath(output_pdf))
    safe_tmp = os.path.join(base_dir, "_video_to_pdf_frames_tmp")
    os.makedirs(safe_tmp, exist_ok=True)
    return tempfile.mkdtemp(dir=safe_tmp)


def cleanup_dir(path: str) -> None:
    """Best-effort recursive directory cleanup without raising errors.

    Args:
        path (str): Directory path to delete.

    Returns:
        None: This function performs side effects only.
    """
    try:
        for root_dir, dirs, files in os.walk(path, topdown=False):
            for filename in files:
                try:
                    # Remove file if possible; ignore per-file failures.
                    os.remove(os.path.join(root_dir, filename))
                except Exception:
                    pass
            for dirname in dirs:
                try:
                    # Remove empty subdirectory; ignore if still in use.
                    os.rmdir(os.path.join(root_dir, dirname))
                except Exception:
                    pass
        os.rmdir(path)
    except Exception:
        # Suppress cleanup errors to avoid hiding the main operation result.
        pass


def sanitize_folder_name(name: str) -> str:
    """Create a folder-safe name by replacing invalid characters.

    Args:
        name (str): Raw folder name candidate.

    Returns:
        str: Safe folder name or `video` fallback.
    """
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "video"

