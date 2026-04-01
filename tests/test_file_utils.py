import os

from modules.core import file_utils


def test_safe_filename_handles_blank_and_reserved_names():
    assert file_utils.safe_filename("") == "Combined"
    assert file_utils.safe_filename("CON") == "CON_document"


def test_safe_filename_removes_invalid_chars_and_truncates():
    raw = ' My:Invalid*Name? '
    assert file_utils.safe_filename(raw) == "MyInvalidName"

    long_name = "a" * 500
    safe = file_utils.safe_filename(long_name, max_length=20)
    assert len(safe) == 20


def test_format_helpers():
    assert file_utils.format_seconds(65) == "1:05"
    assert file_utils.format_seconds(3661) == "1:01:01"
    assert file_utils.format_bytes(0) == "0 B"
    assert file_utils.format_bytes(1024).endswith("KB")


def test_make_unique_output_dir(tmp_path):
    base = tmp_path / "output"
    base.mkdir()
    candidate = file_utils.make_unique_output_dir(str(base))
    assert candidate.endswith("_2")
    assert not os.path.exists(candidate)

