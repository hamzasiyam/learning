"""PDF construction and merging utilities for image/video workflows."""

import os
import tempfile
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

    return np.hstack(resized)


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

    return np.vstack(padded)


def add_page_number_bar(
    image: np.ndarray,
    page_num: int,
    bar_height: int,
    total_pages: Optional[int] = None,
    footer_label: Optional[str] = None,
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
    cv2.putText(
        page_canvas,
        label,
        (12, h + int(bar_height * 0.75)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
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

    Returns:
        str: Output directory path containing exported PNG files.
    """
    columns = max(1, int(columns))
    per_row = max(1, int(per_row)) if per_row else columns
    os.makedirs(output_dir, exist_ok=True)

    total_images = len(image_paths)
    groups = [image_paths[i:i + columns] for i in range(0, total_images, columns)]
    num_groups = len(groups)
    if total_pages is None:
        total_pages = max(0, start_page_number - 1) + total_images

    for i, group in enumerate(groups):
        if progress_callback:
            progress_callback(i, num_groups, f"Exporting PNG {i + 1} of {num_groups}")

        frames = []
        for img_path in group:
            frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise RuntimeError(f"Could not read image for PNG export: {os.path.basename(img_path)}")
            frames.append(frame)

        first_img = start_page_number + i * columns
        last_img = first_img + len(group) - 1

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
        )
        if columns > 1:
            out_path = os.path.join(output_dir, f"Photos {first_img}-{last_img}.png")
        else:
            out_path = os.path.join(output_dir, f"Page {first_img}.png")
        if not cv2.imwrite(out_path, composite):
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


def append_pdf_to_writer(writer, pdf_path):
    """Append all pages from one PDF file into an existing writer.

    Args:
        writer: pypdf PdfWriter instance.
        pdf_path (str): Source PDF path.

    Returns:
        None: Mutates `writer` by adding pages.
    """
    reader = PdfReader(pdf_path)
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
    reader = PdfReader(pdf_path)
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
    total_images = len(image_paths)
    groups = [image_paths[i:i + columns] for i in range(0, total_images, columns)]

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
        for gi, group in enumerate(groups):
            if progress_callback:
                progress_callback(
                    progress_offset + gi + 1, full_total,
                    f"Writing page {gi + 1} of {num_groups}",
                )

            cv_frames = []
            for img_path in group:
                if optimize and os.path.exists(img_path):
                    frame = cv2.imread(img_path)
                    if frame is not None:
                        fh, fw = frame.shape[:2]
                        if fw > OPTIMIZE_MAX_WIDTH:
                            frame = cv2.resize(
                                frame,
                                (OPTIMIZE_MAX_WIDTH, int(fh * OPTIMIZE_MAX_WIDTH / fw)),
                                interpolation=cv2.INTER_AREA,
                            )
                        cv_frames.append(frame)
                        continue
                cv_frames.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))

            if any(f is None for f in cv_frames):
                bad = [os.path.basename(p) for p, f in zip(group, cv_frames) if f is None]
                raise RuntimeError(f"Could not read image(s): {', '.join(bad)}")

            if len(cv_frames) > 1:
                composite = stitch_images_grid(cv_frames, per_row)
            else:
                composite = cv_frames[0]
            if composite.ndim == 2:
                composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
            elif composite.shape[2] == 4:
                composite = cv2.cvtColor(composite, cv2.COLOR_BGRA2BGR)

            tmp_path = os.path.join(tmp_dir, f"page_{gi:06d}.jpg")
            cv2.imwrite(tmp_path, composite, [cv2.IMWRITE_JPEG_QUALITY, OPTIMIZE_QUALITY if optimize else 95])

            first_img = start_page_number + gi * columns
            last_img = first_img + len(group) - 1

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
                     per_row: int = 0):
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
        )
        return [output_pdf]

    base, ext = os.path.splitext(output_pdf)
    outputs: List[str] = []
    offset = 0
    for idx, batch in enumerate(batches, 1):
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
        )
        offset += len(batch)
        outputs.append(part_out)
    return outputs

