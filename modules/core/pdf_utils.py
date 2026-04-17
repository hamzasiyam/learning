"""PDF construction and merging utilities for image/video workflows."""

import os
import tempfile
from io import BytesIO
import logging
from typing import List, Optional

import cv2
import numpy as np
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .config import OPTIMIZE_MAX_WIDTH, OPTIMIZE_QUALITY
from .file_utils import cleanup_dir

try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    PdfReader = None
    PdfWriter = None

for _logger_name in ("pypdf", "PyPDF2"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

MERGE_SEPARATOR_PX = 3
FRAME_WATERMARK_ALPHA = 0.34
FRAME_WATERMARK_COLOR = (140, 140, 140)


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
    margin = max(10, int(min_side * 0.04))
    font_scale = max(1.0, min_side / 650.0)
    thickness = max(2, int(min_side * 0.004))
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
    left_font_scale = 0.6
    left_thickness = 2
    baseline_y = h + int(bar_height * 0.75)
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
        right_font_scale = 0.55
        right_thickness = 1
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
            frames.append(add_frame_number_watermark(frame, frame_num))

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
    for page in reader.pages:
        writer.add_page(page)
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

    Returns:
        None: Writes a PDF to disk.
    """
    from reportlab.lib.colors import black

    columns = max(1, int(columns))
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
    try:
        full_total = progress_total or num_groups
        for gi, (group_paths, group_titles, group_indices) in enumerate(groups):
            if progress_callback:
                progress_callback(
                    progress_offset + gi + 1, full_total,
                    f"Writing page {gi + 1} of {num_groups}",
                )

            cv_frames = []
            for img_idx, img_path in enumerate(group_paths):
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
                        frame_num = start_page_number + group_indices[img_idx]
                        if apply_frame_watermark:
                            frame = add_frame_number_watermark(frame, frame_num)
                        cv_frames.append(frame)
                        continue
                frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if frame is not None:
                    frame_num = start_page_number + group_indices[img_idx]
                    if apply_frame_watermark:
                        frame = add_frame_number_watermark(frame, frame_num)
                cv_frames.append(frame)

            if any(f is None for f in cv_frames):
                bad = [os.path.basename(p) for p, f in zip(group_paths, cv_frames) if f is None]
                raise RuntimeError(f"Could not read image(s): {', '.join(bad)}")

            if len(cv_frames) > 1:
                composite = stitch_images_grid(cv_frames, per_row)
            else:
                composite = cv_frames[0]
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

            img = ImageReader(tmp_path)
            w, h = img.getSize()
            c.setPageSize((w, h + bar_height))
            c.drawImage(img, 0, bar_height, width=w, height=h)
            c.setFillColor(black)
            c.rect(0, 0, w, bar_height, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1)
            c.setFont("Helvetica", 14)
            c.drawString(12, bar_height * 0.25, footer_text)
            source_title = group_titles[0] if group_titles and group_titles[0] else ""
            if source_title:
                max_label_width = max(20, w - 190)
                fitted = source_title
                while c.stringWidth(fitted, "Helvetica", 12) > max_label_width and len(fitted) > 1:
                    fitted = fitted[:-1]
                if fitted != source_title and len(fitted) > 1:
                    fitted = fitted[:-1] + "..."
                c.setFillColorRGB(0.82, 0.82, 0.82)
                c.setFont("Helvetica", 12)
                c.drawRightString(w - 12, bar_height * 0.25, fitted)
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
        for pi in page_indexes:
            writer.add_page(reader.pages[pi])
        if metadata:
            writer.add_metadata(metadata)
        with open(out_path, "wb") as f:
            writer.write(f)
        outputs.append(out_path)
    return outputs

