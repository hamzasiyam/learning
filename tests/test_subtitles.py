"""Tests for subtitle parsing and frame alignment."""

import os
import tempfile

from modules.core.subtitles import (
    align_cues_to_frame_paths,
    load_subtitle_file,
    parse_frame_timestamp,
    parse_vtt,
)


def test_parse_frame_timestamp():
    assert parse_frame_timestamp(r"C:\tmp\v000\t000045.jpg") == 45
    assert parse_frame_timestamp("frame.jpg") is None


def test_parse_vtt_basic():
    content = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello <b>world</b>

00:00:05.000 --> 00:00:07.500
Second line
"""
    with tempfile.NamedTemporaryFile("w", suffix=".vtt", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    try:
        cues = parse_vtt(path)
        assert len(cues) == 2
        assert cues[0].text == "Hello world"
        assert cues[0].start == 1.0
        assert cues[1].text == "Second line"
    finally:
        os.unlink(path)


def test_align_cues_to_frames():
    from modules.core.subtitles import SubtitleCue

    cues = [
        SubtitleCue(0, 4, "Intro"),
        SubtitleCue(4, 8, "Middle"),
        SubtitleCue(8, 12, "End"),
    ]
    paths = [r"t000000.jpg", r"t000003.jpg", r"t000006.jpg"]
    aligned = align_cues_to_frame_paths(paths, cues, seconds_step=3)
    assert "Intro" in aligned[paths[0]]
    assert "Middle" in aligned[paths[1]]
    assert "End" in aligned[paths[2]]
