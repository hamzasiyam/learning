"""Parse WebVTT / transcript text files and align cues to extracted video frames."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

FRAME_TIMESTAMP_RE = re.compile(r"t(\d+)", re.IGNORECASE)

# WebVTT / SRT-style: 00:00:01.000 --> 00:00:04.000
CUE_LINE_RE = re.compile(
    r"^(\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?|\d{1,2}:\d{2}(?:[.,]\d{1,3})?)"
    r"\s*-->\s*"
    r"(\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?|\d{1,2}:\d{2}(?:[.,]\d{1,3})?)"
)

# [00:01:23] or (00:01:23) or 00:01:23 - text
INLINE_TIMESTAMP_RE = re.compile(
    r"^[\[\(]?"
    r"(\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?|\d{1,2}:\d{2}(?:[.,]\d{1,3})?|\d{1,4})"
    r"[\]\)]?\s*[-:–]?\s*(.*)$"
)

# Standalone timestamp line (YouTube-style transcript)
STANDALONE_TIME_RE = re.compile(
    r"^(\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?|\d{1,2}:\d{2}(?:[.,]\d{1,3})?|\d{1,4})\s*$"
)

HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class SubtitleCue:
    """One timed subtitle segment."""

    start: float
    end: float
    text: str


def parse_frame_timestamp(path: str) -> Optional[int]:
    """Read capture time in seconds from a frame filename like ``t000045.jpg``."""
    match = FRAME_TIMESTAMP_RE.search(path)
    if not match:
        return None
    return int(match.group(1))


def _time_to_seconds(token: str) -> float:
    """Convert ``H:MM:SS.mmm``, ``M:SS``, or plain seconds to float seconds."""
    token = token.strip().replace(",", ".")
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    parts = token.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    raise ValueError(f"Unrecognized timestamp: {token}")


def _clean_cue_text(text: str) -> str:
    text = HTML_TAG_RE.sub("", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return " ".join(text.split())


def parse_vtt(path: str) -> List[SubtitleCue]:
    """Parse a WebVTT (or SRT-like) subtitle file into timed cues."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw_lines = [line.rstrip("\n\r") for line in f]

    cues: List[SubtitleCue] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        i += 1
        if not line or line.upper() == "WEBVTT" or line.startswith("NOTE") or line.isdigit():
            continue
        match = CUE_LINE_RE.match(line)
        if not match:
            continue
        start = _time_to_seconds(match.group(1))
        end = _time_to_seconds(match.group(2))
        if end <= start:
            end = start + 2.0
        text_lines: List[str] = []
        while i < len(raw_lines) and raw_lines[i].strip():
            text_lines.append(_clean_cue_text(raw_lines[i]))
            i += 1
        text = " ".join(t for t in text_lines if t).strip()
        if text:
            cues.append(SubtitleCue(start=start, end=end, text=text))
    return cues


def _parse_timestamped_txt(lines: List[str]) -> Optional[List[SubtitleCue]]:
    """Try inline or standalone timestamp formats; return None if not detected."""
    cues: List[SubtitleCue] = []
    pending_start: Optional[float] = None
    found_timestamp = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        standalone = STANDALONE_TIME_RE.match(line)
        if standalone:
            found_timestamp = True
            pending_start = _time_to_seconds(standalone.group(1))
            continue

        inline = INLINE_TIMESTAMP_RE.match(line)
        if inline:
            found_timestamp = True
            start = _time_to_seconds(inline.group(1))
            text = _clean_cue_text(inline.group(2))
            end = start + 3.0 if not cues else max(start + 1.0, cues[-1].end)
            if text:
                cues.append(SubtitleCue(start=start, end=end, text=text))
            pending_start = None
            continue

        if pending_start is not None:
            found_timestamp = True
            text = _clean_cue_text(line)
            if text:
                end = pending_start + max(2.0, len(text) / 14.0)
                cues.append(SubtitleCue(start=pending_start, end=end, text=text))
            pending_start = None

    if not found_timestamp:
        return None

    cues.sort(key=lambda c: c.start)
    for idx in range(len(cues) - 1):
        if cues[idx].end > cues[idx + 1].start:
            cues[idx] = SubtitleCue(start=cues[idx].start, end=cues[idx + 1].start, text=cues[idx].text)
    return cues


def _parse_plain_txt(lines: List[str], video_duration_sec: Optional[float]) -> List[SubtitleCue]:
    """Split plain transcript text across the video duration when no timestamps exist."""
    paragraphs: List[str] = []
    buf: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        paragraphs.append(" ".join(buf))

    if not paragraphs:
        return []

    duration = float(video_duration_sec) if video_duration_sec and video_duration_sec > 0 else len(paragraphs) * 5.0
    slot = duration / len(paragraphs)
    cues: List[SubtitleCue] = []
    for idx, text in enumerate(paragraphs):
        start = idx * slot
        end = (idx + 1) * slot if idx < len(paragraphs) - 1 else duration
        cues.append(SubtitleCue(start=start, end=end, text=_clean_cue_text(text)))
    return cues


def parse_transcript_txt(path: str, video_duration_sec: Optional[float] = None) -> List[SubtitleCue]:
    """Parse a ``.txt`` transcript (timestamped or plain) into cues."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()
    timestamped = _parse_timestamped_txt(lines)
    if timestamped:
        return timestamped
    return _parse_plain_txt(lines, video_duration_sec)


def load_subtitle_file(path: str, video_duration_sec: Optional[float] = None) -> List[SubtitleCue]:
    """Load cues from ``.vtt``, ``.webvtt``, or ``.txt``."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in ("vtt", "webvtt"):
        return parse_vtt(path)
    if ext == "txt":
        return parse_transcript_txt(path, video_duration_sec=video_duration_sec)
    raise ValueError(f"Unsupported subtitle file type: .{ext}")


def align_cues_to_frame_paths(
    frame_paths: List[str],
    cues: List[SubtitleCue],
    seconds_step: int,
) -> Dict[str, str]:
    """Map each frame path to subtitle text spoken during its capture window."""
    if not cues:
        return {path: "" for path in frame_paths}

    step = max(1, int(seconds_step))
    result: Dict[str, str] = {}
    for path in frame_paths:
        t_sec = parse_frame_timestamp(path)
        if t_sec is None:
            result[path] = ""
            continue
        window_start = float(t_sec)
        window_end = window_start + step
        parts: List[str] = []
        for cue in cues:
            if cue.start < window_end and cue.end > window_start:
                parts.append(cue.text)
        # De-duplicate while preserving order
        seen = set()
        unique_parts = []
        for part in parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)
        result[path] = "\n".join(unique_parts).strip()
    return result


def align_cues_to_frame_indices(
    frame_paths: List[str],
    cues: List[SubtitleCue],
    seconds_step: int,
) -> Dict[int, str]:
    """Return ``{frame_index: subtitle_text}`` for a single video's frames."""
    by_path = align_cues_to_frame_paths(frame_paths, cues, seconds_step)
    return {idx: by_path.get(path, "") for idx, path in enumerate(frame_paths)}
