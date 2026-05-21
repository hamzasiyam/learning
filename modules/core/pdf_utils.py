"""PDF construction and merging utilities for image/video workflows."""

import os
import tempfile
from io import BytesIO
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .config import OPTIMIZE_MAX_WIDTH, OPTIMIZE_QUALITY
from .file_utils import cleanup_dir

try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject
except Exception:
    PdfReader = None
    PdfWriter = None
    NameObject = None

for _logger_name in ("pypdf", "PyPDF2"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

MERGE_SEPARATOR_PX = 3
FRAME_WATERMARK_ALPHA = 0.88
FRAME_WATERMARK_COLOR = (255, 255, 255)
FRAME_WATERMARK_STROKE_COLOR = (0, 0, 0)
LEARNING_UI_SCALE = 1.0


def _normalize_bgr(images: List[np.ndarray]) -> List[np.ndarray]:
    """Convert a list of images to BGR, handling grayscale and BGRA."""
    out = []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        out.append(img)
    return out


def add_frame_number_watermark(image: np.ndarray, frame_num: int) -> np.ndarray:
    """Overlay a subtle bottom-right frame number watermark on an image."""
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        base = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        base = image.copy()

    h, w = base.shape[:2]
    min_side = max(1, min(h, w))
    margin = max(14, int(min_side * 0.055))
    # Roughly 2x larger than prior sizing for print readability.
    font_scale = max(2.0, min_side / 320.0)
    thickness = max(4, int(min_side * 0.008))
    text = str(frame_num)

    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    if text_w > int(w * 0.35) and text_w > 0:
        font_scale *= (w * 0.35) / text_w
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

    x = max(margin, w - text_w - margin)
    y = max(text_h + margin, h - margin)

    overlay = base.copy()
    stroke_thickness = max(2, thickness + 2)
    cv2.putText(
        overlay,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        FRAME_WATERMARK_STROKE_COLOR,
        stroke_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        FRAME_WATERMARK_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    return cv2.addWeighted(overlay, FRAME_WATERMARK_ALPHA, base, 1.0 - FRAME_WATERMARK_ALPHA, 0)


def _truncate_footer_text(text: str, max_width: int, font_scale: float, thickness: int) -> str:
    """Trim footer text with ellipsis so it fits available width."""
    if not text or max_width <= 0:
        return ""
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] <= max_width:
        return text
    ellipsis = "..."
    candidate = text
    while candidate:
        candidate = candidate[:-1]
        trial = candidate.rstrip() + ellipsis
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] <= max_width:
            return trial
    return ""


def _group_images_for_pages(
    image_paths: List[str],
    columns: int,
    image_source_titles: Optional[List[Optional[str]]] = None,
):
    """Group ordered images into page groups without crossing source transitions."""
    if image_source_titles is None:
        source_titles: List[Optional[str]] = [None] * len(image_paths)
    else:
        if len(image_source_titles) != len(image_paths):
            raise ValueError("image_source_titles must match image_paths length")
        source_titles = [s.strip() if isinstance(s, str) and s.strip() else None for s in image_source_titles]

    groups = []
    current_paths: List[str] = []
    current_titles: List[Optional[str]] = []
    current_indices: List[int] = []
    current_source: Optional[str] = None

    for idx, (path, source_title) in enumerate(zip(image_paths, source_titles)):
        if current_paths and (len(current_paths) >= columns or source_title != current_source):
            groups.append((current_paths, current_titles, current_indices))
            current_paths, current_titles, current_indices = [], [], []

        if not current_paths:
            current_source = source_title

        current_paths.append(path)
        current_titles.append(source_title)
        current_indices.append(idx)

    if current_paths:
        groups.append((current_paths, current_titles, current_indices))
    return groups


def rasterize_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 220,
    progress_callback=None,
    stop_event=None,
) -> List[str]:
    """Render each PDF page to a lossless PNG image."""
    if fitz is None:
        raise RuntimeError(
            "PDF page-to-image conversion requires PyMuPDF.\n"
            "Install it by running: pip install pymupdf"
        )
    os.makedirs(output_dir, exist_ok=True)
    zoom = max(0.5, float(dpi) / 72.0)
    matrix = fitz.Matrix(zoom, zoom)

    output_paths: List[str] = []
    with fitz.open(pdf_path) as doc:
        total = len(doc)
        for i, page in enumerate(doc):
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Stopped")
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = os.path.join(output_dir, f"page_{i + 1:05d}.png")
            pix.save(out_path)
            output_paths.append(out_path)
            if progress_callback:
                progress_callback(i + 1, total, f"Rasterizing PDF page {i + 1} of {total}")
    return output_paths


def stitch_images_row(images: List[np.ndarray]) -> np.ndarray:
    """Place multiple images side by side into a single row.

    All images are normalised to BGR and scaled to a common height (the
    minimum in the group) so the horizontal join is seamless.
    """
    if len(images) == 1:
        return _normalize_bgr(images)[0]

    bgr = _normalize_bgr(images)
    target_h = min(im.shape[0] for im in bgr)
    resized = []
    for img in bgr:
        h, w = img.shape[:2]
        if h != target_h:
            new_w = int(w * target_h / h)
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(img)

    separator = np.zeros((target_h, MERGE_SEPARATOR_PX, 3), dtype=np.uint8)
    stitched_parts: List[np.ndarray] = []
    for idx, img in enumerate(resized):
        if idx:
            stitched_parts.append(separator)
        stitched_parts.append(img)
    return np.hstack(stitched_parts)


def stitch_images_grid(images: List[np.ndarray], per_row: int) -> np.ndarray:
    """Arrange images into a grid with *per_row* columns.

    Rows are stitched horizontally, then stacked vertically.  An incomplete
    last row is padded with black to match the width of full rows.
    """
    if len(images) == 1:
        return _normalize_bgr(images)[0]
    if per_row >= len(images):
        return stitch_images_row(images)

    rows: List[np.ndarray] = []
    for i in range(0, len(images), per_row):
        rows.append(stitch_images_row(images[i:i + per_row]))

    target_w = max(r.shape[1] for r in rows)
    padded = []
    for row in rows:
        h, w = row.shape[:2]
        if w < target_w:
            pad = np.zeros((h, target_w - w, 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        padded.append(row)

    separator = np.zeros((MERGE_SEPARATOR_PX, target_w, 3), dtype=np.uint8)
    stacked_parts: List[np.ndarray] = []
    for idx, row in enumerate(padded):
        if idx:
            stacked_parts.append(separator)
        stacked_parts.append(row)
    return np.vstack(stacked_parts)


def _scale_rect(rect: Tuple[int, int, int, int], sx: float, sy: float) -> Tuple[int, int, int, int]:
    """Scale an ``(x1, y1, x2, y2)`` rectangle by horizontal/vertical factors."""
    x1, y1, x2, y2 = rect
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def stitch_images_row_with_rects(
    images: List[np.ndarray],
    rects: List[Optional[Tuple[int, int, int, int]]],
) -> Tuple[np.ndarray, List[Optional[Tuple[int, int, int, int]]]]:
    """Stitch a row while carrying optional rectangles through resize/join steps."""
    if len(images) != len(rects):
        raise ValueError("images and rects must have the same length")
    if len(images) == 1:
        return _normalize_bgr(images)[0], [rects[0]]

    bgr = _normalize_bgr(images)
    target_h = min(im.shape[0] for im in bgr)
    resized = []
    resized_rects: List[Optional[Tuple[int, int, int, int]]] = []
    for img, rect in zip(bgr, rects):
        h, w = img.shape[:2]
        sx = sy = 1.0
        if h != target_h:
            new_w = int(w * target_h / h)
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
            sx = new_w / float(w)
            sy = target_h / float(h)
        resized.append(img)
        resized_rects.append(_scale_rect(rect, sx, sy) if rect else None)

    separator = np.zeros((target_h, MERGE_SEPARATOR_PX, 3), dtype=np.uint8)
    stitched_parts: List[np.ndarray] = []
    translated_rects: List[Optional[Tuple[int, int, int, int]]] = []
    x_offset = 0
    for idx, (img, rect) in enumerate(zip(resized, resized_rects)):
        if idx:
            stitched_parts.append(separator)
            x_offset += MERGE_SEPARATOR_PX
        stitched_parts.append(img)
        if rect:
            x1, y1, x2, y2 = rect
            translated_rects.append((x1 + x_offset, y1, x2 + x_offset, y2))
        else:
            translated_rects.append(None)
        x_offset += img.shape[1]
    return np.hstack(stitched_parts), translated_rects


def stitch_images_grid_with_rects(
    images: List[np.ndarray],
    per_row: int,
    rects: List[Optional[Tuple[int, int, int, int]]],
) -> Tuple[np.ndarray, List[Optional[Tuple[int, int, int, int]]]]:
    """Arrange images in a grid and translate optional rectangles accordingly."""
    if len(images) != len(rects):
        raise ValueError("images and rects must have the same length")
    if len(images) == 1:
        return _normalize_bgr(images)[0], [rects[0]]
    if per_row >= len(images):
        return stitch_images_row_with_rects(images, rects)

    rows: List[np.ndarray] = []
    row_rects_all: List[List[Optional[Tuple[int, int, int, int]]]] = []
    for i in range(0, len(images), per_row):
        row_img, row_rects = stitch_images_row_with_rects(images[i:i + per_row], rects[i:i + per_row])
        rows.append(row_img)
        row_rects_all.append(row_rects)

    target_w = max(r.shape[1] for r in rows)
    padded = []
    for row in rows:
        h, w = row.shape[:2]
        if w < target_w:
            pad = np.zeros((h, target_w - w, 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        padded.append(row)

    separator = np.zeros((MERGE_SEPARATOR_PX, target_w, 3), dtype=np.uint8)
    stacked_parts: List[np.ndarray] = []
    translated_rects: List[Optional[Tuple[int, int, int, int]]] = []
    y_offset = 0
    for idx, (row, row_rects) in enumerate(zip(padded, row_rects_all)):
        if idx:
            stacked_parts.append(separator)
            y_offset += MERGE_SEPARATOR_PX
        stacked_parts.append(row)
        for rect in row_rects:
            if rect:
                x1, y1, x2, y2 = rect
                translated_rects.append((x1, y1 + y_offset, x2, y2 + y_offset))
            else:
                translated_rects.append(None)
        y_offset += row.shape[0]
    return np.vstack(stacked_parts), translated_rects


def _wrap_text_lines(
    text: str,
    max_width_px: int,
    font_scale: float,
    thickness: int,
) -> List[str]:
    """Wrap plain text by rendered width so drawn text never overflows."""
    text = " ".join((text or "").split())
    if not text:
        return []
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        trial_w = cv2.getTextSize(
            trial, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0][0]
        if trial_w <= max_width_px or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_subtitle_panel(
    subtitle_text: str,
    target_height: int,
    panel_width: Optional[int] = None,
    render_text: bool = True,
) -> np.ndarray:
    """Render subtitle text as a light panel image for side-by-side layout.

    A small empty box is reserved at the bottom-right for manual icon/checkmark
    annotations after printing.
    """
    target_height = max(120, int(target_height))
    panel_width = max(300, int(panel_width or 420))

    padding = 22
    header_h = 42
    icon_box_x1, icon_box_y1, icon_box_x2, icon_box_y2 = _subtitle_icon_box_rect(
        panel_width, target_height, padding=padding
    )
    icon_box_h = icon_box_y2 - icon_box_y1
    icon_box_reserve_h = icon_box_h + padding
    # Intentionally large for print readability.
    font_scale = 1.45
    thickness = 2
    line_gap = 12
    text_left = padding
    text_right = panel_width - padding
    max_text_w = max(80, text_right - text_left)
    lines = _wrap_text_lines(
        subtitle_text.replace("\n", " "),
        max_width_px=max_text_w,
        font_scale=font_scale,
        thickness=thickness,
    )
    if not lines:
        lines = [""]

    usable_h = max(20, target_height - header_h - (2 * padding) - icon_box_reserve_h)
    # Shrink only as needed; keep text readable.
    while True:
        line_sizes = [
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            for line in lines
        ]
        text_block_h = sum(h for _, h in line_sizes) + line_gap * max(0, len(lines) - 1)
        if text_block_h <= usable_h or font_scale <= 1.1:
            break
        font_scale = max(1.1, font_scale * 0.92)
        lines = _wrap_text_lines(
            subtitle_text.replace("\n", " "),
            max_width_px=max_text_w,
            font_scale=font_scale,
            thickness=thickness,
        ) or [""]

    line_sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        for line in lines
    ]
    line_h = max((h for _, h in line_sizes), default=12)
    slot_h = max(1, line_h + line_gap)
    max_lines = max(1, usable_h // slot_h)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1].rstrip()
        while last:
            trial = f"{last}..."
            tw = cv2.getTextSize(
                trial, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )[0][0]
            if tw <= max_text_w:
                lines[-1] = trial
                break
            last = last[:-1]
        if not last:
            lines[-1] = "..."
        line_sizes = [
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            for line in lines
        ]
    text_block_h = sum(h for _, h in line_sizes) + line_gap * max(0, len(lines) - 1)

    panel = np.full((target_height, panel_width, 3), 248, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, target_height - 1), (210, 210, 210), 1)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, header_h), (238, 238, 238), -1)
    if render_text:
        y = header_h + padding + max(0, (usable_h - text_block_h) // 2) + (
            line_sizes[0][1] if line_sizes else 0
        )
        for line, (tw, th) in zip(lines, line_sizes):
            cv2.putText(
                panel,
                line,
                (text_left, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (25, 25, 25),
                thickness,
                cv2.LINE_AA,
            )
            y += th + line_gap

    header = "Subtitles"
    cv2.putText(
        panel,
        header,
        (padding, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (85, 85, 85),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(panel, (icon_box_x1, icon_box_y1), (icon_box_x2, icon_box_y2), (180, 180, 180), -1)
    cv2.rectangle(panel, (icon_box_x1, icon_box_y1), (icon_box_x2, icon_box_y2), (120, 120, 120), 2)
    return panel


def _subtitle_icon_box_rect(
    panel_width: int,
    panel_height: int,
    padding: int = 22,
) -> Tuple[int, int, int, int]:
    """Return the bottom-right annotation box rectangle for the subtitle panel."""
    panel_width = max(1, int(panel_width))
    panel_height = max(1, int(panel_height))
    box_w, box_h = _learning_box_dims_from_panel_height(panel_height)
    x2 = panel_width - padding
    y2 = panel_height - padding
    x1 = x2 - box_w
    y1 = y2 - box_h
    return x1, y1, x2, y2


def _learning_box_dims_from_panel_height(panel_height: int) -> Tuple[int, int]:
    """Compute (width, height) for the learning radio box.

    Sizing flow:
      1. Pick a radio diameter from panel height.
      2. Pick an icon font height roughly 1.3x the radio.
      3. Pack each option column as: top_pad + radio + gap + icon + bottom_pad.
      4. Make the column wide enough to fit either the radio or the icon.
      5. Final box = 3 * column_w by content_h (no square forcing, no dead space).
    """
    panel_height = max(1, int(panel_height))
    radio = max(16, min(26, int(panel_height * 0.05)))
    icon_h = int(radio * 1.4)
    top_pad = max(4, int(radio * 0.30))
    gap = max(3, int(radio * 0.25))
    bottom_pad = max(4, int(radio * 0.30))
    h_pad = max(5, int(radio * 0.35))
    column_w = max(radio + (2 * h_pad), icon_h + (2 * h_pad))
    box_w = column_w * 3
    box_h = top_pad + radio + gap + icon_h + bottom_pad
    return int(box_w * LEARNING_UI_SCALE), int(box_h * LEARNING_UI_SCALE)


def compose_frame_with_subtitle_and_icon_box(
    frame: np.ndarray,
    subtitle_text: str,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    """Compose frame+subtitle panel and return icon-box rect in composed coords."""
    text = (subtitle_text or "").strip()
    if not text:
        return frame, None, None

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    h, w = frame.shape[:2]
    panel_w = max(300, min(700, int(w * 0.50)))
    panel = render_subtitle_panel(text, target_height=h, panel_width=panel_w, render_text=False)
    box_x1, box_y1, box_x2, box_y2 = _subtitle_icon_box_rect(panel_w, h)
    composed = stitch_images_row([frame, panel])
    x_shift = w + MERGE_SEPARATOR_PX
    icon_box = (box_x1 + x_shift, box_y1, box_x2 + x_shift, box_y2)
    panel_rect = (x_shift, 0, x_shift + panel_w, h)
    return composed, icon_box, panel_rect


def compose_frame_with_subtitle(frame: np.ndarray, subtitle_text: str) -> np.ndarray:
    """Place the video frame on the left and wrapped subtitle text on the right."""
    composed, _icon_box, _panel_rect = compose_frame_with_subtitle_and_icon_box(frame, subtitle_text)
    return composed


def _wrap_pdf_text_lines(c, text: str, max_width: float, font_name: str, font_size: float) -> List[str]:
    """Wrap text for ReportLab rendering within a maximum width."""
    content = " ".join((text or "").split())
    if not content:
        return []
    words = content.split()
    lines: List[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if c.stringWidth(trial, font_name, font_size) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_selectable_subtitle_text(
    c,
    subtitle_text: str,
    panel_rect: Tuple[float, float, float, float],
) -> None:
    """Draw selectable subtitle text inside a subtitle panel rectangle."""
    if not subtitle_text:
        return
    x1, y1, x2, y2 = panel_rect
    panel_w = max(1.0, x2 - x1)
    panel_h = max(1.0, y2 - y1)
    pad_x = max(14.0, panel_w * 0.05)
    pad_y = max(12.0, panel_h * 0.05)
    header_h = max(24.0, panel_h * 0.08)
    _box_x1, box_y1, _box_x2, box_y2 = _subtitle_icon_box_rect(int(panel_w), int(panel_h), padding=int(pad_y))
    icon_box_h = max(1.0, float(box_y2 - box_y1))
    reserve_bottom = icon_box_h + pad_y

    text_left = x1 + pad_x
    text_right = x2 - pad_x
    max_text_w = max(30.0, text_right - text_left)
    text_top = y2 - header_h - pad_y
    text_bottom = y1 + reserve_bottom
    available_h = max(20.0, text_top - text_bottom)

    font_name = "Helvetica"
    # Keep subtitle text large for readability/marking in print workflows.
    font_size = max(20.0, min(40.0, panel_h * 0.090))
    line_gap = max(4.0, font_size * 0.20)
    lines = _wrap_pdf_text_lines(c, subtitle_text, max_text_w, font_name, font_size)
    if not lines:
        return

    max_lines = max(1, int(available_h // (font_size + line_gap)))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and c.stringWidth(f"{last}...", font_name, font_size) > max_text_w:
            last = last[:-1]
        lines[-1] = f"{last}..." if last else "..."

    y = text_top - font_size
    c.setFillColorRGB(0.10, 0.10, 0.10)
    c.setFont(font_name, font_size)
    for line in lines:
        if y < text_bottom:
            break
        c.drawString(text_left, y, line)
        y -= (font_size + line_gap)


def _draw_learning_radio_group(c, field_name: str, rect: Tuple[float, float, float, float]) -> None:
    """Draw one 3-option icon panel (check/x/star) in the provided PDF rect."""
    from reportlab.lib.colors import Color

    x1, y1, x2, y2 = rect
    box_w = max(20.0, float(x2 - x1))
    box_h = max(20.0, float(y2 - y1))
    option_w = box_w / 3.0
    option_h = box_h

    options = [
        ("check", "✓"),
        ("x", "✗"),
        ("star", "★"),
    ]
    option_symbol_colors = [
        Color(0.12, 0.62, 0.20),  # green check
        Color(0.85, 0.18, 0.18),  # red x
        Color(0.90, 0.74, 0.06),  # yellow star
    ]

    top_pad = option_h * 0.12
    bottom_pad = option_h * 0.12
    icon_h = max(10.0, option_h - top_pad - bottom_pad)

    # Frame and column dividers.
    c.setStrokeColor(Color(0.55, 0.55, 0.55))
    c.setLineWidth(max(1.0, LEARNING_UI_SCALE))
    c.rect(x1, y1, box_w, box_h, stroke=1, fill=0)
    for idx in range(1, 3):
        sx = x1 + (option_w * idx)
        c.line(sx, y1, sx, y2)

    for idx, (_value, symbol) in enumerate(options):
        opt_x1 = x1 + (idx * option_w)
        cx = opt_x1 + (option_w * 0.5)

        # Icon centered in its column with generous margins.
        icon_font = min(option_w * 0.78, icon_h * 1.02)
        icon_font = max(12.0 * LEARNING_UI_SCALE, icon_font)
        icon_baseline_y = y1 + bottom_pad + ((icon_h - (icon_font * 0.80)) / 2.0)

        c.setFont("Helvetica-Bold", int(icon_font))
        c.setFillColor(option_symbol_colors[idx])
        c.drawCentredString(cx, icon_baseline_y, symbol)
    c.setFillColor(Color(0, 0, 0))


def add_page_number_bar(
    image: np.ndarray,
    page_num: int,
    bar_height: int,
    total_pages: Optional[int] = None,
    footer_label: Optional[str] = None,
    right_label: Optional[str] = None,
) -> np.ndarray:
    """Append a black footer bar and page number text to an image.

    Args:
        image (np.ndarray): Source image array (BGR/BGRA/gray).
        page_num (int): Page number to render.
        bar_height (int): Footer bar height in pixels.
        total_pages (Optional[int]): Total pages in this export batch. If provided,
            footer text becomes `Page X/Y`.
        footer_label (Optional[str]): Explicit label overriding the default format.

    Returns:
        np.ndarray: Updated image array with footer bar.
    """
    if bar_height <= 0:
        return image
    h, w = image.shape[:2]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    page_canvas = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
    page_canvas[:h, :] = image
    if footer_label:
        label = footer_label
    elif total_pages and total_pages > 0:
        label = f"Page {page_num}/{total_pages}"
    else:
        label = f"Page {page_num}"
    left_font_scale = max(0.9, bar_height / 42.0)
    left_thickness = max(2, int(bar_height / 18.0))
    baseline_y = h + int(bar_height * 0.72)
    cv2.putText(
        page_canvas,
        label,
        (12, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        left_font_scale,
        (255, 255, 255),
        left_thickness,
        cv2.LINE_AA,
    )
    if right_label:
        right_font_scale = max(0.8, bar_height / 46.0)
        right_thickness = max(1, int(bar_height / 24.0))
        left_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, left_font_scale, left_thickness)[0][0]
        available = max(10, w - 36 - left_w)
        fitted = _truncate_footer_text(right_label, available, right_font_scale, right_thickness)
        if fitted:
            right_w = cv2.getTextSize(fitted, cv2.FONT_HERSHEY_SIMPLEX, right_font_scale, right_thickness)[0][0]
            cv2.putText(
                page_canvas,
                fitted,
                (w - right_w - 12, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                right_font_scale,
                (210, 210, 210),
                right_thickness,
                cv2.LINE_AA,
            )
    return page_canvas


def export_images_as_png(
    image_paths: List[str],
    output_dir: str,
    progress_callback=None,
    bar_height: int = 36,
    start_page_number: int = 1,
    total_pages: Optional[int] = None,
    columns: int = 1,
    per_row: int = 0,
    lossless_optimize: bool = False,
    image_source_titles: Optional[List[Optional[str]]] = None,
    subtitle_texts: Optional[List[str]] = None,
    include_subtitles: bool = False,
) -> str:
    """Export selected images as numbered PNG files with optional footer bar.

    Args:
        image_paths (List[str]): Input image paths.
        output_dir (str): Destination folder for PNG files.
        progress_callback: Optional callback for progress updates.
        bar_height (int): Footer bar height in pixels.
        start_page_number (int): First page number to assign.
        total_pages (Optional[int]): Total pages in batch for `Page X/Y` footer text.
        columns (int): Total images per output file.
        per_row (int): Images per row within the grid (0 = same as columns).
        lossless_optimize (bool): If True, uses strongest PNG compression
            without changing image pixels.

    Returns:
        str: Output directory path containing exported PNG files.
    """
    columns = max(1, int(columns))
    per_row = max(1, int(per_row)) if per_row else columns
    os.makedirs(output_dir, exist_ok=True)

    total_images = len(image_paths)
    groups = _group_images_for_pages(image_paths, columns, image_source_titles=image_source_titles)
    num_groups = len(groups)
    if total_pages is None:
        total_pages = max(0, start_page_number - 1) + total_images

    for i, (group_paths, group_titles, group_indices) in enumerate(groups):
        if progress_callback:
            progress_callback(i, num_groups, f"Exporting PNG {i + 1} of {num_groups}")

        frames = []
        for img_idx, img_path in enumerate(group_paths):
            frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise RuntimeError(f"Could not read image for PNG export: {os.path.basename(img_path)}")
            frame_num = start_page_number + group_indices[img_idx]
            frame = add_frame_number_watermark(frame, frame_num)
            if include_subtitles and subtitle_texts is not None:
                global_idx = group_indices[img_idx]
                subtitle = (subtitle_texts[global_idx] or "").strip()
                if subtitle:
                    frame = compose_frame_with_subtitle(frame, subtitle)
            frames.append(frame)

        first_img = start_page_number + group_indices[0]
        last_img = start_page_number + group_indices[-1]

        if columns > 1:
            footer = f"Photos {first_img}-{last_img} / {total_pages}"
        elif total_pages and total_pages > 0:
            footer = f"Page {first_img}/{total_pages}"
        else:
            footer = f"Page {first_img}"

        if len(frames) > 1:
            composite = stitch_images_grid(frames, per_row)
        else:
            composite = frames[0]
        composite = add_page_number_bar(
            composite,
            page_num=first_img,
            bar_height=bar_height,
            footer_label=footer,
            right_label=group_titles[0] if group_titles and group_titles[0] else None,
        )
        if columns > 1:
            out_path = os.path.join(output_dir, f"Photos {first_img}-{last_img}.png")
        else:
            out_path = os.path.join(output_dir, f"Page {first_img}.png")
        write_params = [cv2.IMWRITE_PNG_COMPRESSION, 9] if lossless_optimize else []
        if not cv2.imwrite(out_path, composite, write_params):
            raise RuntimeError(f"Could not write PNG file: {os.path.basename(out_path)}")
    if progress_callback:
        progress_callback(num_groups, num_groups, "PNG export complete")
    return output_dir


def ensure_pdf_merge_available():
    """Ensure optional PDF dependency is available.

    Returns:
        None: Raises if dependency is missing.
    """
    if PdfReader is None or PdfWriter is None:
        raise RuntimeError("Missing required library: pypdf.\nInstall it by running:  pip install pypdf")


def _safe_pdf_reader(pdf_path: str):
    """Open a PDF in tolerant mode to handle minor structural issues."""
    try:
        return PdfReader(pdf_path, strict=False)
    except TypeError:
        return PdfReader(pdf_path)


def append_pdf_to_writer(writer, pdf_path):
    """Append all pages from one PDF file into an existing writer.

    Args:
        writer: pypdf PdfWriter instance.
        pdf_path (str): Source PDF path.

    Returns:
        None: Mutates `writer` by adding pages.
    """
    reader = _safe_pdf_reader(pdf_path)
    for page in reader.pages:
        writer.add_page(page)


def _copy_acroform_root(reader, writer) -> None:
    """Best-effort copy of AcroForm root to preserve interactive form behavior."""
    try:
        root = getattr(reader, "trailer", {}).get("/Root", {})
        acro_form = root.get("/AcroForm") if root else None
        if acro_form is not None:
            key = NameObject("/AcroForm") if NameObject is not None else "/AcroForm"
            writer._root_object.update({key: acro_form})
    except Exception:
        pass


def meta_to_pdf_metadata(meta):
    """Convert app metadata dictionary into PDF metadata keys.

    Args:
        meta (dict): User-supplied metadata.

    Returns:
        dict: PDF metadata dictionary using slash-prefixed keys.
    """
    pm = {}
    if meta.get("title"):
        pm["/Title"] = meta["title"]
    if meta.get("author"):
        pm["/Author"] = meta["author"]
    if meta.get("tags"):
        pm["/Keywords"] = meta["tags"]
    if meta.get("type"):
        pm["/Type"] = meta["type"]
    if meta.get("published"):
        s = meta["published"].strip().replace("-", "").replace("/", "")[:8]
        if len(s) == 8 and s.isdigit():
            pm["/CreationDate"] = f"D:{s}000000"
    if meta.get("language"):
        pm["/Language"] = meta["language"]
    if meta.get("cover_image"):
        pm["/CoverImage"] = meta["cover_image"]
    return pm


def add_metadata_to_pdf(pdf_path, meta):
    """Apply metadata to an existing PDF file in-place.

    Args:
        pdf_path (str): PDF file path.
        meta (dict): Metadata source.

    Returns:
        None: Overwrites file with updated metadata.
    """
    pm = meta_to_pdf_metadata(meta)
    if not pm:
        return
    ensure_pdf_merge_available()
    reader = _safe_pdf_reader(pdf_path)
    writer = PdfWriter()
    # Preserve document-level structures (e.g., AcroForm fields) when available.
    # Re-adding pages alone can drop interactive form behavior in some viewers.
    if hasattr(writer, "clone_document_from_reader"):
        writer.clone_document_from_reader(reader)
    else:
        for page in reader.pages:
            writer.add_page(page)
        try:
            root = getattr(reader, "trailer", {}).get("/Root", {})
            acro_form = root.get("/AcroForm") if root else None
            if acro_form is not None:
                key = NameObject("/AcroForm") if NameObject is not None else "/AcroForm"
                writer._root_object.update({key: acro_form})
        except Exception:
            pass
    writer.add_metadata(pm)
    with open(pdf_path, "wb") as f:
        writer.write(f)


def estimate_pdf_size_bytes(image_paths, optimize=False):
    """Estimate output PDF size from input image sizes.

    Args:
        image_paths (list[str]): Image file paths.
        optimize (bool): Whether optimization is enabled.

    Returns:
        int: Estimated PDF size in bytes.
    """
    try:
        total = sum(os.path.getsize(p) for p in image_paths if os.path.exists(p))
    except Exception:
        return 0
    if optimize:
        total = int(total * 0.45)
    return int(total * (1.15 if optimize else 1.12))


def build_pdf_from_images(
    image_paths,
    output_pdf,
    meta,
    bar_height=36,
    progress_callback=None,
    optimize=False,
    start_page_number=1,
    progress_offset=0,
    progress_total=None,
    total_pages: Optional[int] = None,
    columns: int = 1,
    per_row: int = 0,
    image_source_titles: Optional[List[Optional[str]]] = None,
    lossless_pages: bool = False,
    apply_frame_watermark: bool = True,
    optimize_max_width: Optional[int] = None,
    optimize_quality: Optional[int] = None,
    subtitle_texts: Optional[List[str]] = None,
    include_subtitles: bool = False,
    include_subtitles_as_text: bool = False,
):
    """Build a PDF from ordered images and optional page footer numbering.

    Args:
        image_paths (list[str]): Input images in page order.
        output_pdf (str): Destination PDF path.
        meta (dict): Metadata fields for title/author/type.
        bar_height (int): Footer bar height in points.
        progress_callback: Optional callback for progress updates.
        optimize (bool): If True, downscale/compress temporary images.
        start_page_number (int): First page number displayed in footer.
        progress_offset (int): Completed pages from previous batch.
        progress_total (int | None): Global total pages for progress.
        total_pages (int | None): Global total pages for footer text (`Page X/Y`).
        columns (int): Total images per page.
        per_row (int): Images per row within the grid (0 = same as columns).
        subtitle_texts (Optional[List[str]]): Subtitle text aligned with image_paths.
        include_subtitles (bool): When True, render subtitle panel beside each frame.

    Returns:
        None: Writes a PDF to disk.
    """
    from reportlab.lib.colors import black

    columns = max(1, int(columns))
    if subtitle_texts is not None and len(subtitle_texts) != len(image_paths):
        raise ValueError("subtitle_texts must match image_paths length")
    per_row = max(1, int(per_row)) if per_row else columns
    effective_max_width = int(optimize_max_width or OPTIMIZE_MAX_WIDTH)
    effective_quality = int(optimize_quality or OPTIMIZE_QUALITY)
    total_images = len(image_paths)
    groups = _group_images_for_pages(image_paths, columns, image_source_titles=image_source_titles)

    c = canvas.Canvas(output_pdf)
    c.setTitle(meta.get("title", ""))
    c.setAuthor(meta.get("author", ""))
    c.setSubject(meta.get("type", ""))

    num_groups = len(groups)
    if total_pages is None:
        total_pages = total_images

    tmp_dir = tempfile.mkdtemp(prefix="video_to_pdf_opt_")
    page_specs = []
    try:
        full_total = progress_total or num_groups
        for gi, (group_paths, group_titles, group_indices) in enumerate(groups):
            if progress_callback:
                progress_callback(
                    progress_offset + gi + 1, full_total,
                    f"Writing page {gi + 1} of {num_groups}",
                )

            cv_frames = []
            frame_icon_boxes: List[Optional[Tuple[int, int, int, int]]] = []
            frame_panel_rects: List[Optional[Tuple[int, int, int, int]]] = []
            frame_subtitles: List[str] = []
            for img_idx, img_path in enumerate(group_paths):
                global_idx = group_indices[img_idx]
                subtitle = ""
                if include_subtitles and subtitle_texts is not None:
                    subtitle = (subtitle_texts[global_idx] or "").strip()
                frame_subtitles.append(subtitle)

                if optimize and os.path.exists(img_path):
                    frame = cv2.imread(img_path)
                    if frame is not None:
                        fh, fw = frame.shape[:2]
                        if fw > effective_max_width:
                            frame = cv2.resize(
                                frame,
                                (effective_max_width, int(fh * effective_max_width / fw)),
                                interpolation=cv2.INTER_AREA,
                            )
                        frame_num = start_page_number + global_idx
                        if apply_frame_watermark:
                            frame = add_frame_number_watermark(frame, frame_num)
                        if subtitle:
                            frame, icon_box, panel_rect = compose_frame_with_subtitle_and_icon_box(frame, subtitle)
                        else:
                            icon_box = None
                            panel_rect = None
                        cv_frames.append(frame)
                        frame_icon_boxes.append(icon_box)
                        frame_panel_rects.append(panel_rect)
                        continue
                frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if frame is not None:
                    frame_num = start_page_number + global_idx
                    if apply_frame_watermark:
                        frame = add_frame_number_watermark(frame, frame_num)
                    if subtitle:
                        frame, icon_box, panel_rect = compose_frame_with_subtitle_and_icon_box(frame, subtitle)
                    else:
                        icon_box = None
                        panel_rect = None
                else:
                    icon_box = None
                    panel_rect = None
                cv_frames.append(frame)
                frame_icon_boxes.append(icon_box)
                frame_panel_rects.append(panel_rect)

            if any(f is None for f in cv_frames):
                bad = [os.path.basename(p) for p, f in zip(group_paths, cv_frames) if f is None]
                raise RuntimeError(f"Could not read image(s): {', '.join(bad)}")

            if len(cv_frames) > 1:
                if include_subtitles:
                    composite, composite_icon_boxes = stitch_images_grid_with_rects(
                        cv_frames, per_row, frame_icon_boxes
                    )
                    _composite_for_panels, composite_panel_rects = stitch_images_grid_with_rects(
                        cv_frames, per_row, frame_panel_rects
                    )
                else:
                    composite = stitch_images_grid(cv_frames, per_row)
                    composite_icon_boxes = []
                    composite_panel_rects = []
            else:
                composite = cv_frames[0]
                composite_icon_boxes = [frame_icon_boxes[0]] if include_subtitles else []
                composite_panel_rects = [frame_panel_rects[0]] if include_subtitles else []
            if composite.ndim == 2:
                composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
            elif composite.shape[2] == 4:
                composite = cv2.cvtColor(composite, cv2.COLOR_BGRA2BGR)

            ext = "png" if lossless_pages else "jpg"
            tmp_path = os.path.join(tmp_dir, f"page_{gi:06d}.{ext}")
            if lossless_pages:
                cv2.imwrite(tmp_path, composite, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                cv2.imwrite(tmp_path, composite, [cv2.IMWRITE_JPEG_QUALITY, effective_quality if optimize else 95])

            first_img = start_page_number + group_indices[0]
            last_img = start_page_number + group_indices[-1]

            if columns > 1:
                footer_text = f"Photos {first_img}-{last_img} / {total_pages}"
            elif total_pages and total_pages > 0:
                footer_text = f"Page {first_img}/{total_pages}"
            else:
                footer_text = f"Page {first_img}"

            source_title = group_titles[0] if group_titles and group_titles[0] else ""
            img = ImageReader(tmp_path)
            w, h = img.getSize()
            subtitle_overlays = []
            if include_subtitles and include_subtitles_as_text:
                for panel_rect, subtitle_text in zip(composite_panel_rects, frame_subtitles):
                    if panel_rect and subtitle_text:
                        subtitle_overlays.append(
                            {
                                "panel_rect": panel_rect,
                                "text": subtitle_text,
                            }
                        )
            page_specs.append(
                {
                    "tmp_path": tmp_path,
                    "w": w,
                    "h": h,
                    "footer_text": footer_text,
                    "source_title": source_title,
                    "icon_boxes": composite_icon_boxes,
                    "subtitle_overlays": subtitle_overlays,
                }
            )

        if page_specs:
            page_w = max(spec["w"] for spec in page_specs)
            image_area_h = max(spec["h"] for spec in page_specs)
            # Keep a stable portrait layout for all pages in this PDF, including leftovers.
            image_area_h = max(image_area_h, page_w + 1)
            page_h = image_area_h + bar_height

            for page_idx, spec in enumerate(page_specs):
                img = ImageReader(spec["tmp_path"])
                w = spec["w"]
                h = spec["h"]
                footer_text = spec["footer_text"]
                source_title = spec["source_title"]
                icon_boxes = spec.get("icon_boxes", [])
                subtitle_overlays = spec.get("subtitle_overlays", [])
                c.setPageSize((page_w, page_h))
                draw_x = 0
                draw_y = bar_height + (image_area_h - h)
                c.drawImage(img, draw_x, draw_y, width=w, height=h)
                for overlay in subtitle_overlays:
                    panel_rect = overlay.get("panel_rect")
                    text = overlay.get("text", "")
                    if not panel_rect:
                        continue
                    px1, py1, px2, py2 = panel_rect
                    pdf_panel_rect = (
                        draw_x + float(px1),
                        draw_y + float(h - py2),
                        draw_x + float(px2),
                        draw_y + float(h - py1),
                    )
                    _draw_selectable_subtitle_text(c, text, pdf_panel_rect)
                for box_idx, icon_box in enumerate(icon_boxes):
                    if not icon_box:
                        continue
                    x1, y1, x2, y2 = icon_box
                    pdf_rect = (
                        draw_x + float(x1),
                        draw_y + float(h - y2),
                        draw_x + float(x2),
                        draw_y + float(h - y1),
                    )
                    field_name = f"learn_mark_p{page_idx + 1}_i{box_idx + 1}"
                    _draw_learning_radio_group(c, field_name, pdf_rect)
                c.setFillColor(black)
                c.rect(0, 0, page_w, bar_height, fill=1, stroke=0)
                c.setFillColorRGB(1, 1, 1)
                footer_font = max(16, int(bar_height * 0.50))
                c.setFont("Helvetica-Bold", footer_font)
                c.drawString(12, bar_height * 0.22, footer_text)
                if source_title:
                    max_label_width = max(20, page_w - 190)
                    fitted = source_title
                    right_font = max(12, int(bar_height * 0.36))
                    while c.stringWidth(fitted, "Helvetica", right_font) > max_label_width and len(fitted) > 1:
                        fitted = fitted[:-1]
                    if fitted != source_title and len(fitted) > 1:
                        fitted = fitted[:-1] + "..."
                    c.setFillColorRGB(0.82, 0.82, 0.82)
                    c.setFont("Helvetica", right_font)
                    c.drawRightString(page_w - 12, bar_height * 0.22, fitted)
                c.showPage()
    finally:
        cleanup_dir(tmp_dir)
    c.save()
    try:
        add_metadata_to_pdf(output_pdf, meta)
    except Exception:
        pass


def estimate_page_bytes(image_path, optimize=False):
    """Estimate output bytes for a single page image.

    Args:
        image_path (str): Image file path.
        optimize (bool): Whether optimization is enabled.

    Returns:
        int: Estimated per-page byte size.
    """
    try:
        img_bytes = os.path.getsize(image_path)
    except Exception:
        img_bytes = 350 * 1024
    factor = 0.62 if optimize else 1.08
    floor = 120 * 1024
    return max(floor, int(img_bytes * factor))


def split_images_by_size(image_paths, target_size_mb, optimize=False):
    """Split image list into size-limited batches.

    Args:
        image_paths (list[str]): Ordered image paths.
        target_size_mb (float): Approximate target output size per part.
        optimize (bool): Whether optimization is enabled.

    Returns:
        List[List[str]]: Batches of image paths.
    """
    target_bytes = max(1, int(target_size_mb * 1024 * 1024))
    batches: List[List[str]] = []
    current: List[str] = []
    current_bytes = 0
    for path in image_paths:
        page_bytes = estimate_page_bytes(path, optimize=optimize)
        # Start a new batch when current one would exceed target size.
        if current and (current_bytes + page_bytes) > target_bytes:
            batches.append(current)
            current, current_bytes = [], 0
        current.append(path)
        current_bytes += page_bytes
    if current:
        batches.append(current)
    return batches


def build_split_pdfs(image_paths, output_pdf, meta, target_size_mb, bar_height=36,
                     optimize=False, progress_callback=None, columns: int = 1,
                     per_row: int = 0,
                     image_source_titles: Optional[List[Optional[str]]] = None,
                     optimize_max_width: Optional[int] = None,
                     optimize_quality: Optional[int] = None):
    """Export one or more PDF parts constrained by target size.

    Args:
        image_paths (list[str]): Input page images.
        output_pdf (str): Base output PDF path.
        meta (dict): Metadata values.
        target_size_mb (float): Target size per output part in MB.
        bar_height (int): Footer bar height.
        optimize (bool): Whether optimization is enabled.
        progress_callback: Optional callback for progress updates.
        columns (int): Total images per page.
        per_row (int): Images per row within the grid (0 = same as columns).

    Returns:
        list[str]: Output PDF paths (single item if no split needed).
    """
    total_pages = len(image_paths)
    if image_source_titles is not None and len(image_source_titles) != total_pages:
        raise ValueError("image_source_titles must match image_paths length")
    batches = split_images_by_size(image_paths, target_size_mb, optimize=optimize)
    if len(batches) <= 1:
        build_pdf_from_images(
            image_paths,
            output_pdf,
            meta,
            bar_height=bar_height,
            optimize=optimize,
            progress_callback=progress_callback,
            total_pages=total_pages,
            columns=columns,
            per_row=per_row,
            image_source_titles=image_source_titles,
            optimize_max_width=optimize_max_width,
            optimize_quality=optimize_quality,
        )
        return [output_pdf]

    base, ext = os.path.splitext(output_pdf)
    outputs: List[str] = []
    offset = 0
    for idx, batch in enumerate(batches, 1):
        batch_titles = (
            image_source_titles[offset:offset + len(batch)]
            if image_source_titles is not None
            else None
        )
        first_photo = offset + 1
        last_photo = offset + len(batch)
        part_out = f"{base}_photos_{first_photo}-{last_photo}{ext}"
        part_meta = dict(meta)
        title = meta.get("title", "Untitled").strip() or "Untitled"
        part_meta["title"] = f"{title} (Photos {first_photo}-{last_photo})"
        build_pdf_from_images(
            batch,
            part_out,
            part_meta,
            bar_height=bar_height,
            optimize=optimize,
            start_page_number=offset + 1,
            progress_offset=offset,
            progress_total=total_pages,
            total_pages=total_pages,
            progress_callback=progress_callback,
            columns=columns,
            per_row=per_row,
            image_source_titles=batch_titles,
            optimize_max_width=optimize_max_width,
            optimize_quality=optimize_quality,
        )
        offset += len(batch)
        outputs.append(part_out)
    return outputs


def split_existing_pdf_by_size(
    pdf_path: str,
    output_pdf: str,
    target_size_mb: float,
    metadata: Optional[dict] = None,
) -> List[str]:
    """Split an existing PDF into parts that respect a target size.

    Splitting is page-based and preserves page order. Size checks use actual
    serialized PDF bytes in-memory for accurate part boundaries.
    """
    ensure_pdf_merge_available()
    target_bytes = max(1, int(float(target_size_mb) * 1024 * 1024))
    reader = _safe_pdf_reader(pdf_path)
    total_pages = len(reader.pages)
    if total_pages == 0:
        return [output_pdf]

    pages_per_part: List[List[int]] = []
    current: List[int] = []

    def part_size_bytes(page_indexes: List[int]) -> int:
        writer = PdfWriter()
        for pi in page_indexes:
            writer.add_page(reader.pages[pi])
        if metadata:
            writer.add_metadata(metadata)
        buff = BytesIO()
        writer.write(buff)
        return buff.tell()

    for page_index in range(total_pages):
        trial = current + [page_index]
        trial_size = part_size_bytes(trial)
        if current and trial_size > target_bytes:
            pages_per_part.append(current)
            current = [page_index]
        else:
            current = trial
    if current:
        pages_per_part.append(current)

    if len(pages_per_part) <= 1:
        return [output_pdf]

    base, ext = os.path.splitext(output_pdf)
    outputs: List[str] = []
    for idx, page_indexes in enumerate(pages_per_part, 1):
        out_path = f"{base}_part_{idx:03d}{ext}"
        writer = PdfWriter()
        if hasattr(writer, "clone_document_from_reader"):
            # Preserve interactive form widgets by cloning full document first,
            # then removing pages not belonging to this part.
            writer.clone_document_from_reader(reader)
            keep = set(page_indexes)
            for pi in range(total_pages - 1, -1, -1):
                if pi not in keep:
                    writer.remove_page(pi)
        else:
            for pi in page_indexes:
                writer.add_page(reader.pages[pi])
            _copy_acroform_root(reader, writer)
        if metadata:
            writer.add_metadata(metadata)
        with open(out_path, "wb") as f:
            writer.write(f)
        outputs.append(out_path)
    return outputs


def split_existing_pdf_by_page_count(
    pdf_path: str,
    output_pdf: str,
    pages_per_part: int = 5,
    metadata: Optional[dict] = None,
) -> List[str]:
    """Split an existing PDF into fixed page-count parts."""
    ensure_pdf_merge_available()
    per_part = max(1, int(pages_per_part))
    reader = _safe_pdf_reader(pdf_path)
    total_pages = len(reader.pages)
    if total_pages == 0:
        return [output_pdf]
    if total_pages <= per_part:
        return [output_pdf]

    base, ext = os.path.splitext(output_pdf)
    outputs: List[str] = []
    for idx, start in enumerate(range(0, total_pages, per_part), 1):
        page_indexes = list(range(start, min(start + per_part, total_pages)))
        out_path = f"{base}_part_{idx:03d}{ext}"
        writer = PdfWriter()
        if hasattr(writer, "clone_document_from_reader"):
            # Preserve interactive form widgets by cloning full document first,
            # then removing pages not belonging to this part.
            writer.clone_document_from_reader(reader)
            keep = set(page_indexes)
            for pi in range(total_pages - 1, -1, -1):
                if pi not in keep:
                    writer.remove_page(pi)
        else:
            for pi in page_indexes:
                writer.add_page(reader.pages[pi])
            _copy_acroform_root(reader, writer)
        if metadata:
            writer.add_metadata(metadata)
        with open(out_path, "wb") as f:
            writer.write(f)
        outputs.append(out_path)
    return outputs

