"""PDF construction and merging utilities for image/video workflows."""

import os
import tempfile
from typing import List

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


def add_page_number_bar(image: np.ndarray, page_num: int, bar_height: int) -> np.ndarray:
    """Append a black footer bar and page number text to an image.

    Args:
        image (np.ndarray): Source image array (BGR/BGRA/gray).
        page_num (int): Page number to render.
        bar_height (int): Footer bar height in pixels.

    Returns:
        np.ndarray: Updated image array with footer bar.
    """
    # If bar height is disabled, return image unchanged.
    if bar_height <= 0:
        return image
    h, w = image.shape[:2]
    # Normalize image to BGR so footer drawing works consistently.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    page_canvas = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
    page_canvas[:h, :] = image
    cv2.putText(
        page_canvas,
        f"Page {page_num}",
        (12, h + int(bar_height * 0.75)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return page_canvas


def export_images_as_png(image_paths: List[str], output_dir: str, progress_callback=None, bar_height: int = 36, start_page_number: int = 1) -> str:
    """Export selected images as numbered PNG files with optional footer bar.

    Args:
        image_paths (List[str]): Input image paths.
        output_dir (str): Destination folder for PNG files.
        progress_callback: Optional callback for progress updates.
        bar_height (int): Footer bar height in pixels.
        start_page_number (int): First page number to assign.

    Returns:
        str: Output directory path containing exported PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(image_paths)
    for i, img_path in enumerate(image_paths):
        page_num = start_page_number + i
        # Report progress before processing each image.
        if progress_callback:
            progress_callback(i, total, f"Exporting PNG {i + 1} of {total}")
        frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"Could not read image for PNG export: {os.path.basename(img_path)}")
        frame = add_page_number_bar(frame, page_num=page_num, bar_height=bar_height)
        out_path = os.path.join(output_dir, f"Page {page_num}.png")
        if not cv2.imwrite(out_path, frame):
            raise RuntimeError(f"Could not write PNG file: {os.path.basename(out_path)}")
    if progress_callback:
        progress_callback(total, total, "PNG export complete")
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


def build_pdf_from_images(image_paths, output_pdf, meta, bar_height=36, progress_callback=None, optimize=False, start_page_number=1, progress_offset=0, progress_total=None):
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

    Returns:
        None: Writes a PDF to disk.
    """
    from reportlab.lib.colors import black

    c = canvas.Canvas(output_pdf)
    c.setTitle(meta.get("title", ""))
    c.setAuthor(meta.get("author", ""))
    c.setSubject(meta.get("type", ""))

    tmp_opt = tempfile.mkdtemp(prefix="video_to_pdf_opt_") if optimize else None
    try:
        full_total = progress_total or len(image_paths)
        for i, img_path in enumerate(image_paths, 1):
            page_num = start_page_number + i - 1
            # Emit progress for this page render.
            if progress_callback:
                progress_callback(progress_offset + i, full_total, f"Writing page {page_num} of {start_page_number + len(image_paths) - 1}")
            # If optimization is enabled, create a temporary compressed image.
            if optimize and os.path.exists(img_path):
                frame = cv2.imread(img_path)
                if frame is not None:
                    h, w = frame.shape[:2]
                    if w > OPTIMIZE_MAX_WIDTH:
                        frame = cv2.resize(frame, (OPTIMIZE_MAX_WIDTH, int(h * OPTIMIZE_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
                    opt_path = os.path.join(tmp_opt, f"opt_{i:06d}.jpg")
                    cv2.imwrite(opt_path, frame, [cv2.IMWRITE_JPEG_QUALITY, OPTIMIZE_QUALITY])
                    img_path = opt_path

            img = ImageReader(img_path)
            w, h = img.getSize()
            c.setPageSize((w, h + bar_height))
            c.drawImage(img, 0, bar_height, width=w, height=h)
            c.setFillColor(black)
            c.rect(0, 0, w, bar_height, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1)
            c.setFont("Helvetica", 14)
            c.drawString(12, bar_height * 0.25, f"Page {page_num}")
            c.showPage()
    finally:
        if tmp_opt:
            cleanup_dir(tmp_opt)
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


def build_split_pdfs(image_paths, output_pdf, meta, target_size_mb, bar_height=36, optimize=False, progress_callback=None):
    """Export one or more PDF parts constrained by target size.

    Args:
        image_paths (list[str]): Input page images.
        output_pdf (str): Base output PDF path.
        meta (dict): Metadata values.
        target_size_mb (float): Target size per output part in MB.
        bar_height (int): Footer bar height.
        optimize (bool): Whether optimization is enabled.
        progress_callback: Optional callback for progress updates.

    Returns:
        list[str]: Output PDF paths (single item if no split needed).
    """
    batches = split_images_by_size(image_paths, target_size_mb, optimize=optimize)
    if len(batches) <= 1:
        build_pdf_from_images(image_paths, output_pdf, meta, bar_height=bar_height, optimize=optimize, progress_callback=progress_callback)
        return [output_pdf]

    base, ext = os.path.splitext(output_pdf)
    total_pages = len(image_paths)
    outputs: List[str] = []
    offset = 0
    for idx, batch in enumerate(batches, 1):
        part_out = f"{base}_part_{idx:03d}{ext}"
        part_meta = dict(meta)
        title = meta.get("title", "Untitled").strip() or "Untitled"
        part_meta["title"] = f"{title} (Part {idx} of {len(batches)})"
        build_pdf_from_images(
            batch,
            part_out,
            part_meta,
            bar_height=bar_height,
            optimize=optimize,
            start_page_number=offset + 1,
            progress_offset=offset,
            progress_total=total_pages,
            progress_callback=progress_callback,
        )
        offset += len(batch)
        outputs.append(part_out)
    return outputs

