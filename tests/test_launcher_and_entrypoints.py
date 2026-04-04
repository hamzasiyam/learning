"""Smoke tests: launcher registry points at real files; core modules import cleanly."""

import importlib
import py_compile
from pathlib import Path

import pytest

from modules.launcher.program_launcher import PROGRAMS


def test_launcher_program_script_paths_exist():
    """Every ``PROGRAMS[].module_name`` must resolve to an existing ``.py`` file.

    Returns:
        None: Asserts only.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for program in PROGRAMS:
        module_parts = program.module_name.split(".")
        module_path = repo_root.joinpath(*module_parts).with_suffix(".py")
        assert module_path.exists(), f"Missing module file: {module_path}"


def test_entrypoints_and_modules_import():
    """Import main and app wrappers; compile huge modules; import shared libraries.

    Skips PDF merger import if ``tkinterdnd2`` is missing (optional drag-and-drop).

    Returns:
        None: Asserts or ``pytest.skip`` only.
    """
    importlib.import_module("main")
    importlib.import_module("apps.flashcards")
    importlib.import_module("apps.image_links_and_notebooklm")
    importlib.import_module("apps.merge_pdf")
    repo_root = Path(__file__).resolve().parents[1]
    py_compile.compile(str(repo_root / "apps" / "video_to_pdf.py"), doraise=True)
    py_compile.compile(str(repo_root / "apps" / "quiz_pdf.py"), doraise=True)
    importlib.import_module("modules.core.config")
    importlib.import_module("modules.core.file_utils")
    importlib.import_module("modules.core.pdf_utils")
    importlib.import_module("modules.core.similarity")
    importlib.import_module("modules.core.video_processing")
    importlib.import_module("modules.apps.flashcards_app")
    importlib.import_module("modules.apps.flashcards_utils")
    importlib.import_module("modules.apps.link_csv_app")
    importlib.import_module("modules.apps.link_csv_utils")
    try:
        importlib.import_module("modules.apps.pdf_merger_app")
    except ModuleNotFoundError as exc:
        if exc.name == "tkinterdnd2":
            pytest.skip("tkinterdnd2 is not installed in this environment")
        raise
    importlib.import_module("modules.launcher.program_launcher")
