"""Main desktop workflow for converting videos/images/PDFs into PDF outputs.

This module contains the legacy all-in-one implementation used by the launcher.
Core reusable logic is also available in `modules.core.*` for modular imports.
"""

import os
import re
import sys
import time
import tempfile
import threading
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import cv2
from modules.core import config as cfg
from modules.core import file_utils as futils
from modules.core import pdf_utils as pdfm
from modules.core import similarity as sim
from modules.core import video_processing as vpm

PdfReader = pdfm.PdfReader
PdfWriter = pdfm.PdfWriter


@contextmanager
def suppress_ffmpeg_stderr():
    """Redirect stderr to /dev/null to suppress FFmpeg H.264 warnings."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, "w") as devnull:
        backup_fd = os.dup(stderr_fd)
        try:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            os.dup2(backup_fd, stderr_fd)
            os.close(backup_fd)


import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar, Combobox

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# =========================
# Configuration / Presets
# =========================
ORIGINAL_RES_WIDTH = 99999

PRESETS = {
    "Every 3 seconds — original resolution": {"seconds": 3,  "width": ORIGINAL_RES_WIDTH, "quality": 100},
    "Every 1 second — original resolution":  {"seconds": 1,  "width": ORIGINAL_RES_WIDTH, "quality": 100},
    "Every 8 seconds — small file (~15 MB)": {"seconds": 8,  "width": 480,  "quality": 38},
    "Every 8 seconds — sharp text":          {"seconds": 8,  "width": 1280, "quality": 72},
    "Every 8 seconds — smallest size (slides)": {"seconds": 8,  "width": 1024, "quality": 62},
    "Every 10 seconds — lower resolution":   {"seconds": 10, "width": 1024, "quality": 58},
    "Every 15 seconds — recommended for lectures": {"seconds": 15, "width": 1400, "quality": 68},
    "Every 30 seconds — high compression":   {"seconds": 30, "width": 900,  "quality": 48},
    "Every 5 seconds — detailed":            {"seconds": 5,  "width": 1600, "quality": 75},
}

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v")
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp")

# =========================
# Perceptual Hash Settings
# =========================
PHASH_SIZE = 16

# Similarity threshold: lower = stricter (fewer frames grouped as similar)
# Higher = looser (more frames grouped as similar, fewer unique)
# Default 10 = frames must be 97%+ identical to be grouped
DEFAULT_THRESHOLD = 10

# Colors for similarity badges in the frame grid
SIM_COLOR_REPRESENTATIVE = "#d0e8ff"   # Light blue  — first frame in a group (representative)
SIM_COLOR_VERY_SIMILAR   = "#c8f5c8"   # Light green — 90%+ similar to group representative
SIM_COLOR_SIMILAR        = "#fff3c0"   # Light yellow — 75–89% similar
SIM_COLOR_SOMEWHAT_SIMILAR = "#ffe0c0" # Light orange — 50–74% similar
SIM_COLOR_UNIQUE         = "#f0f0f0"   # Light grey  — unique, no similar frames found


def similarity_badge_color(similarity_percent: float) -> str:
    if similarity_percent >= 90: return SIM_COLOR_VERY_SIMILAR
    if similarity_percent >= 75: return SIM_COLOR_SIMILAR
    if similarity_percent >= 50: return SIM_COLOR_SOMEWHAT_SIMILAR
    return SIM_COLOR_UNIQUE


def compute_phash(img_path: str) -> Optional[np.ndarray]:
    """Compute a perceptual hash (fingerprint) for an image."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (PHASH_SIZE * 4, PHASH_SIZE * 4), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(img))
        dct_low = dct[:PHASH_SIZE, :PHASH_SIZE]
        mean_val = (dct_low.sum() - dct_low[0, 0]) / (PHASH_SIZE * PHASH_SIZE - 1)
        return (dct_low > mean_val).flatten()
    except Exception:
        return None


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.count_nonzero(h1 != h2))


def similarity_percent(hamming: int, bits: int = PHASH_SIZE * PHASH_SIZE) -> float:
    return round(100.0 * (1.0 - hamming / bits), 1)


@dataclass
class SimilarityGroup:
    group_id: int
    frame_paths: List[str]    = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)
    representative_path: str  = ""

    @property
    def is_unique(self) -> bool:
        return len(self.frame_paths) == 1

    @property
    def average_similarity(self) -> float:
        if len(self.similarities) <= 1:
            return 100.0
        return round(sum(self.similarities[1:]) / (len(self.similarities) - 1), 1)

    @property
    def last_frame_path(self) -> str:
        return self.frame_paths[-1] if self.frame_paths else ""


def cluster_frames_by_similarity(
    frame_paths: List[str],
    threshold: int = DEFAULT_THRESHOLD,
    progress_callback=None
) -> List[SimilarityGroup]:
    """
    Group frames that look visually similar using perceptual hashing.
    threshold: maximum Hamming distance to consider two frames 'the same scene'.
    Lower threshold = stricter grouping (more groups, fewer duplicates skipped).
    """
    if not frame_paths:
        return []

    total = len(frame_paths)
    hashes: List[Optional[np.ndarray]] = []

    for i, path in enumerate(frame_paths):
        hashes.append(compute_phash(path))
        if progress_callback and (i % max(1, total // 20) == 0):
            progress_callback(i, total, f"Analysing frame {i + 1} of {total}")

    groups: List[SimilarityGroup] = []
    representative_hashes: List[Optional[np.ndarray]] = []

    for i, path in enumerate(frame_paths):
        h = hashes[i]
        if h is None:
            # Can't hash this image, treat it as unique
            g = SimilarityGroup(group_id=len(groups), frame_paths=[path],
                                similarities=[100.0], representative_path=path)
            groups.append(g)
            representative_hashes.append(None)
            continue

        best_group_index = None
        best_distance = threshold + 1

        for gi, rep_hash in enumerate(representative_hashes):
            if rep_hash is None:
                continue
            distance = hamming_distance(h, rep_hash)
            if distance <= threshold and distance < best_distance:
                best_distance = distance
                best_group_index = gi

        if best_group_index is not None:
            groups[best_group_index].frame_paths.append(path)
            groups[best_group_index].similarities.append(similarity_percent(best_distance))
        else:
            g = SimilarityGroup(group_id=len(groups), frame_paths=[path],
                                similarities=[100.0], representative_path=path)
            groups.append(g)
            representative_hashes.append(h)

    if progress_callback:
        progress_callback(total, total, "Done")

    return groups


# =========================
# Data model
# =========================
@dataclass
class QueueItem:
    kind: str   # "video" | "pdf"
    path: str


# =========================
# Global state
# =========================
stop_event        = threading.Event()
last_output_pdf: Optional[str] = None
conversion_thread: Optional[threading.Thread] = None
extra_images: List[str] = []


# =========================
# Filename / formatting helpers
# =========================
_WIN_RESERVED = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5",
    "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4",
    "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
})


def sanitize_filename(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip()


def safe_filename(title: str, max_length: int = 200) -> str:
    if not title or not title.strip():
        return "Combined"
    s = sanitize_filename(title)
    s = re.sub(r"\s+", " ", s).strip(" .")
    if not s:
        return "Combined"
    if len(s) > max_length:
        s = s[:max_length].rstrip(" .")
    if not s:
        return "Combined"
    if s.upper() in _WIN_RESERVED:
        return f"{s}_document"
    return s


def format_seconds(s: float) -> str:
    s = max(0, int(s))
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def format_bytes(num_bytes: float) -> str:
    if num_bytes <= 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} GB"


def make_unique_output_dir(base_dir: str) -> str:
    """Return a directory path that does not currently exist."""
    if not os.path.exists(base_dir):
        return base_dir
    n = 2
    while True:
        candidate = f"{base_dir}_{n}"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def add_page_number_bar(image: np.ndarray, page_num: int, bar_height: int) -> np.ndarray:
    """Add a black bottom bar with white page number text to match PDF pages."""
    if bar_height <= 0:
        return image

    h, w = image.shape[:2]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    canvas = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
    canvas[:h, :] = image

    text = f"Page {page_num}"
    text_x = 12
    text_y = h + int(bar_height * 0.75)
    cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def export_images_as_png(image_paths: List[str], output_dir: str, progress_callback=None,
                         bar_height: int = 36, start_page_number: int = 1) -> str:
    """Export image paths to PNG files with page bars, named by page number."""
    os.makedirs(output_dir, exist_ok=True)
    total = len(image_paths)
    for i, img_path in enumerate(image_paths):
        page_num = start_page_number + i
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


def estimate_video_pdf_size(video_path, seconds_step, max_width, quality):
    with suppress_ffmpeg_stderr():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        fps     = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total_f <= 0:
            return None
        duration = total_f / fps
        num_frames = int(duration // seconds_step) + 1
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or 720
        if w > max_width:
            scale = max_width / float(w)
            w, h = int(w * scale), int(h * scale)
        bytes_per_pixel = 0.10 * (quality / 60.0)
        return num_frames * max(1.0, w * h * bytes_per_pixel)
    finally:
        cap.release()


def estimate_queue_output_size():
    settings = PRESETS[preset_var.get()]
    step      = int(settings["seconds"])
    max_width = int(settings["width"])
    quality   = int(settings["quality"])
    total     = 0.0
    unknown_count = 0
    for item in queue_items:
        if item.kind == "pdf":
            try:
                total += os.path.getsize(item.path)
            except Exception:
                pass
        else:
            est = estimate_video_pdf_size(item.path, step, max_width, quality)
            if est is None:
                unknown_count += 1
            else:
                total += est
    # Apply overhead factor and typical compression
    return total * 1.05 * 0.85, unknown_count


# =========================
# UI thread-safe helpers
# =========================
def ui_call(fn, *args, **kwargs):
    root.after(0, lambda: fn(*args, **kwargs))

def ui_set_status(text):  ui_call(status_label.config, text=text)
def ui_set_eta(text):     ui_call(eta_label.config, text=text)
def ui_set_estimated_size(text): ui_call(size_label.config, text=text)

def ui_set_progress(value, maximum):
    def _set():
        progress_bar["maximum"] = max(1, int(maximum))
        progress_bar["value"]   = max(0, int(value))
    ui_call(_set)

def ui_enable_open_location(enabled):
    ui_call(open_location_btn.config, state=("normal" if enabled else "disabled"))

def ui_set_buttons_running(running):
    state = "disabled" if running else "normal"
    for btn in [add_videos_btn, add_pdfs_btn, add_images_btn,
                move_up_btn, move_down_btn, remove_btn, clear_list_btn,
                reverse_btn, build_btn, estimate_btn]:
        ui_call(btn.config, state=state)
    try:
        ui_call(extract_select_btn.config, state=state)
    except NameError:
        pass
    ui_call(preset_combo.config, state=("disabled" if running else "readonly"))
    ui_call(queue_lb.config, state=state)

def ui_popup_done(output_pdf):
    ui_call(lambda: messagebox.showinfo(
        "PDF Created",
        f"Your PDF has been created:\n\n{os.path.basename(output_pdf)}"
    ))


# =========================
# Temp directory helpers
# =========================
def make_temp_dir_near_output(output_pdf):
    base_dir = os.path.dirname(os.path.abspath(output_pdf))
    safe_tmp = os.path.join(base_dir, "_video_to_pdf_frames_tmp")
    os.makedirs(safe_tmp, exist_ok=True)
    return tempfile.mkdtemp(dir=safe_tmp)


def cleanup_dir(path):
    try:
        for root_dir, dirs, files in os.walk(path, topdown=False):
            for f in files:
                try:
                    os.remove(os.path.join(root_dir, f))
                except Exception:
                    pass
            for d in dirs:
                try:
                    os.rmdir(os.path.join(root_dir, d))
                except Exception:
                    pass
        os.rmdir(path)
    except Exception:
        pass


def resize_frame(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# =========================
# Video → temp PDF
# =========================
def video_to_temp_pdf(video_path, temp_pdf_path, preset_name, meta, progress_callback=None):
    s = PRESETS[preset_name]
    seconds_step = int(s["seconds"])
    max_width    = int(s["width"])
    quality      = int(s["quality"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {os.path.basename(video_path)}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_f / fps) if (fps > 0 and total_f > 0) else 0.0
    targets  = list(range(0, int(duration) + 1, seconds_step)) if duration > 0 else None

    frames_dir = os.path.join(os.path.dirname(temp_pdf_path),
                              "_frames_" + str(int(time.time())))
    os.makedirs(frames_dir, exist_ok=True)

    try:
        with suppress_ffmpeg_stderr():
            t0 = time.time()
            img_paths: List[str] = []

            if targets is not None:
                if progress_callback:
                    progress_callback(0, len(targets), "Calculating time remaining...",
                                      f"Extracting frames: {os.path.basename(video_path)}")
                for idx, tsec in enumerate(targets, 1):
                    if stop_event.is_set():
                        raise RuntimeError("Stopped")
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = resize_frame(frame, max_width)
                    img_path = os.path.join(frames_dir, f"t{tsec:06d}.jpg")
                    if cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                        img_paths.append(img_path)
                    if progress_callback:
                        elapsed = time.time() - t0
                        if idx >= 2:
                            eta = f"Time remaining: {format_seconds(max(0, (len(targets) - idx) * elapsed / idx))}"
                        else:
                            eta = "Calculating time remaining..."
                        progress_callback(idx, len(targets), eta,
                                          f"Extracting frames: {os.path.basename(video_path)}")
            else:
                idx = tsec = 0
                while True:
                    if stop_event.is_set():
                        raise RuntimeError("Stopped")
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = resize_frame(frame, max_width)
                    img_path = os.path.join(frames_dir, f"t{tsec:06d}.jpg")
                    if cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                        img_paths.append(img_path)
                    idx += 1
                    tsec += seconds_step
                    if progress_callback:
                        progress_callback(idx, max(1, idx), "Calculating time remaining...",
                                          f"Extracting frames: {os.path.basename(video_path)}")
            cap.release()

        img_paths.sort()
        if not img_paths:
            raise RuntimeError(f"No frames could be extracted from: {os.path.basename(video_path)}")

        c = canvas.Canvas(temp_pdf_path)
        c.setTitle(meta.get("title", ""))
        c.setAuthor(meta.get("author", ""))
        c.setSubject(meta.get("type", ""))
        for img_path in img_paths:
            if stop_event.is_set():
                raise RuntimeError("Stopped")
            img = ImageReader(img_path)
            w, h = img.getSize()
            c.setPageSize((w, h))
            c.drawImage(img, 0, 0, width=w, height=h)
            c.showPage()
        c.save()

    finally:
        try:
            cap.release()
        except Exception:
            pass
        cleanup_dir(frames_dir)


# =========================
# Extract all frames
# =========================
def estimate_total_frames(video_paths, preset_name):
    seconds_step = int(PRESETS[preset_name]["seconds"])
    total = 0
    for vp in video_paths:
        with suppress_ffmpeg_stderr():
            cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            tf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps > 0 and tf > 0:
                total += int(tf / fps) // seconds_step + 1
            else:
                total += 1
        finally:
            cap.release()
    return max(1, total)


def _sanitize_folder_name(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "video"


def extract_all_frames_from_videos(
    video_paths, preset_name, output_dir,
    progress_callback=None, estimated_total=None
):
    s = PRESETS[preset_name]
    seconds_step = int(s["seconds"])
    max_width    = int(s["width"])
    quality      = int(s["quality"])

    all_paths: List[str] = []
    video_collections: List[Tuple[str, List[str]]] = []
    step_max = estimated_total or 1

    for v_idx, vp in enumerate(video_paths):
        if stop_event.is_set():
            raise RuntimeError("Stopped")
        with suppress_ffmpeg_stderr():
            cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {os.path.basename(vp)}")

        vname  = os.path.basename(vp)
        folder = f"v{v_idx:03d}_{_sanitize_folder_name(os.path.splitext(vname)[0])}"
        vdir   = os.path.join(output_dir, folder)
        os.makedirs(vdir, exist_ok=True)
        vpaths: List[str] = []

        with suppress_ffmpeg_stderr():
            fps     = cap.get(cv2.CAP_PROP_FPS) or 0.0
            tf      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = (tf / fps) if (fps > 0 and tf > 0) else 0.0

            if duration > 0:
                for tsec in range(0, int(duration) + 1, seconds_step):
                    if stop_event.is_set():
                        raise RuntimeError("Stopped")
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = resize_frame(frame, max_width)
                    img_path = os.path.join(vdir, f"t{tsec:06d}.jpg")
                    if cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                        all_paths.append(img_path)
                        vpaths.append(img_path)
                    if progress_callback:
                        progress_callback(len(all_paths), step_max, "",
                                          f"Extracting: {vname}")
            else:
                next_t = 0
                fc = 0
                fps = fps or 25.0
                while True:
                    if stop_event.is_set():
                        raise RuntimeError("Stopped")
                    ret, frame = cap.read()
                    if not ret:
                        break
                    pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                    pos_sec = (pos / 1000.0) if pos >= 0 else (fc / fps)
                    if pos_sec >= next_t:
                        frame = resize_frame(frame, max_width)
                        img_path = os.path.join(vdir, f"t{int(next_t):06d}.jpg")
                        if cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                            all_paths.append(img_path)
                            vpaths.append(img_path)
                        next_t += seconds_step
                        if progress_callback:
                            progress_callback(len(all_paths), step_max, "",
                                              f"Extracting: {vname}")
                    fc += 1
            cap.release()

        video_collections.append((vname, vpaths))

    return all_paths, video_collections


# =========================
# PDF building
# =========================
OPTIMIZE_MAX_WIDTH = 1600
OPTIMIZE_QUALITY   = 82


def estimate_pdf_size_bytes(image_paths, optimize=False):
    try:
        total = sum(os.path.getsize(p) for p in image_paths if os.path.exists(p))
    except Exception:
        return 0
    if optimize:
        total = int(total * 0.45)
    return int(total * (1.15 if optimize else 1.12))


def build_pdf_from_images(image_paths, output_pdf, meta, bar_height=36,
                          progress_callback=None, optimize=False,
                          start_page_number=1, progress_offset=0, progress_total=None):
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
            if progress_callback:
                progress_callback(
                    progress_offset + i,
                    full_total,
                    f"Writing page {page_num} of {start_page_number + len(image_paths) - 1}"
                )
            if optimize and os.path.exists(img_path):
                frame = cv2.imread(img_path)
                if frame is not None:
                    h, w = frame.shape[:2]
                    if w > OPTIMIZE_MAX_WIDTH:
                        frame = cv2.resize(
                            frame,
                            (OPTIMIZE_MAX_WIDTH, int(h * OPTIMIZE_MAX_WIDTH / w)),
                            interpolation=cv2.INTER_AREA
                        )
                    opt_path = os.path.join(tmp_opt, f"opt_{i:06d}.jpg")
                    cv2.imwrite(opt_path, frame, [cv2.IMWRITE_JPEG_QUALITY, OPTIMIZE_QUALITY])
                    img_path = opt_path

            img = ImageReader(img_path)
            w, h = img.getSize()
            c.setPageSize((w, h + bar_height))
            c.drawImage(img, 0, bar_height, width=w, height=h)
            # Bottom black bar with page number
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
    try:
        img_bytes = os.path.getsize(image_path)
    except Exception:
        img_bytes = 350 * 1024
    factor = 0.62 if optimize else 1.08
    floor  = 120 * 1024
    return max(floor, int(img_bytes * factor))


def split_images_by_size(image_paths, target_size_mb, optimize=False):
    target_bytes = max(1, int(target_size_mb * 1024 * 1024))
    batches: List[List[str]] = []
    current: List[str] = []
    current_bytes = 0
    for p in image_paths:
        page_bytes = estimate_page_bytes(p, optimize=optimize)
        if current and (current_bytes + page_bytes) > target_bytes:
            batches.append(current)
            current, current_bytes = [], 0
        current.append(p)
        current_bytes += page_bytes
    if current:
        batches.append(current)
    return batches


def build_split_pdfs(image_paths, output_pdf, meta, target_size_mb, bar_height=36,
                     optimize=False, progress_callback=None):
    batches = split_images_by_size(image_paths, target_size_mb, optimize=optimize)
    if len(batches) <= 1:
        build_pdf_from_images(image_paths, output_pdf, meta, bar_height=bar_height,
                              optimize=optimize, progress_callback=progress_callback)
        return [output_pdf]

    base, ext = os.path.splitext(output_pdf)
    total_pages = len(image_paths)
    outputs: List[str] = []
    offset = 0
    for idx, batch in enumerate(batches, 1):
        part_out  = f"{base}_part_{idx:03d}{ext}"
        part_meta = dict(meta)
        title     = meta.get("title", "Untitled").strip() or "Untitled"
        part_meta["title"] = f"{title} (Part {idx} of {len(batches)})"
        build_pdf_from_images(
            batch, part_out, part_meta,
            bar_height=bar_height, optimize=optimize,
            start_page_number=offset + 1,
            progress_offset=offset,
            progress_total=total_pages,
            progress_callback=progress_callback,
        )
        offset += len(batch)
        outputs.append(part_out)
    return outputs


# =========================
# PDF metadata helpers
# =========================
def ensure_pdf_merge_available():
    if PdfReader is None or PdfWriter is None:
        raise RuntimeError(
            "Missing required library: pypdf.\n"
            "Install it by running:  pip install pypdf"
        )


def append_pdf_to_writer(writer, pdf_path):
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        writer.add_page(page)


def meta_to_pdf_metadata(meta):
    pm = {}
    if meta.get("title"):    pm["/Title"]    = meta["title"]
    if meta.get("author"):   pm["/Author"]   = meta["author"]
    if meta.get("tags"):     pm["/Keywords"] = meta["tags"]
    if meta.get("type"):     pm["/Type"]     = meta["type"]
    if meta.get("published"):
        s = meta["published"].strip().replace("-", "").replace("/", "")[:8]
        if len(s) == 8 and s.isdigit():
            pm["/CreationDate"] = f"D:{s}000000"
    if meta.get("language"):    pm["/Language"]   = meta["language"]
    if meta.get("cover_image"): pm["/CoverImage"] = meta["cover_image"]
    return pm


def add_metadata_to_pdf(pdf_path, meta):
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


# =========================
# Queue + main GUI helpers
# =========================
PRESETS = cfg.PRESETS
VIDEO_EXTS = cfg.VIDEO_EXTS
IMAGE_EXTS = cfg.IMAGE_EXTS

DEFAULT_THRESHOLD = cfg.DEFAULT_THRESHOLD
SIM_COLOR_REPRESENTATIVE = cfg.SIM_COLOR_REPRESENTATIVE
SIM_COLOR_VERY_SIMILAR = cfg.SIM_COLOR_VERY_SIMILAR
SIM_COLOR_SIMILAR = cfg.SIM_COLOR_SIMILAR
SIM_COLOR_SOMEWHAT_SIMILAR = cfg.SIM_COLOR_SOMEWHAT_SIMILAR
SIM_COLOR_UNIQUE = cfg.SIM_COLOR_UNIQUE

SimilarityGroup = sim.SimilarityGroup
similarity_badge_color = sim.similarity_badge_color
cluster_frames_by_similarity = sim.cluster_frames_by_similarity

sanitize_filename = futils.sanitize_filename
safe_filename = futils.safe_filename
format_seconds = futils.format_seconds
format_bytes = futils.format_bytes
make_unique_output_dir = futils.make_unique_output_dir
make_temp_dir_near_output = futils.make_temp_dir_near_output
cleanup_dir = futils.cleanup_dir

suppress_ffmpeg_stderr = vpm.suppress_ffmpeg_stderr
resize_frame = vpm.resize_frame
estimate_video_pdf_size = vpm.estimate_video_pdf_size
video_to_temp_pdf = vpm.video_to_temp_pdf
estimate_total_frames = vpm.estimate_total_frames
extract_all_frames_from_videos = vpm.extract_all_frames_from_videos

add_page_number_bar = pdfm.add_page_number_bar
export_images_as_png = pdfm.export_images_as_png
estimate_pdf_size_bytes = pdfm.estimate_pdf_size_bytes
build_pdf_from_images = pdfm.build_pdf_from_images
estimate_page_bytes = pdfm.estimate_page_bytes
split_images_by_size = pdfm.split_images_by_size
build_split_pdfs = pdfm.build_split_pdfs
ensure_pdf_merge_available = pdfm.ensure_pdf_merge_available
append_pdf_to_writer = pdfm.append_pdf_to_writer
meta_to_pdf_metadata = pdfm.meta_to_pdf_metadata
add_metadata_to_pdf = pdfm.add_metadata_to_pdf
PdfWriter = pdfm.PdfWriter

queue_items: List[QueueItem] = []


def refresh_queue_listbox(select_index=None, yview_top=None):
    queue_lb.delete(0, tk.END)
    for item in queue_items:
        tag = "[Video]" if item.kind == "video" else "[PDF]"
        queue_lb.insert(tk.END, f"{tag}  {os.path.basename(item.path)}")
    if select_index is not None and 0 <= select_index < len(queue_items):
        queue_lb.selection_set(select_index)
        queue_lb.activate(select_index)
    if yview_top is not None:
        queue_lb.yview_moveto(yview_top)


def update_size_estimate():
    if not queue_items:
        ui_set_estimated_size("Estimated output size: —")
        return
    try:
        total_bytes, unknown = estimate_queue_output_size()
        suffix = " (some videos could not be estimated)" if unknown else ""
        ui_set_estimated_size(f"Estimated output size: ~{format_bytes(total_bytes)}{suffix}")
    except Exception:
        ui_set_estimated_size("Estimated output size: —")


def add_videos():
    paths = filedialog.askopenfilenames(
        title="Select Video Files",
        filetypes=[("Video Files", " ".join(VIDEO_EXTS))]
    )
    for p in paths:
        queue_items.append(QueueItem("video", p))
    if paths:
        refresh_queue_listbox(len(queue_items) - 1)
        update_size_estimate()


def add_pdfs():
    paths = filedialog.askopenfilenames(
        title="Select PDF Files",
        filetypes=[("PDF Files", "*.pdf")]
    )
    for p in paths:
        queue_items.append(QueueItem("pdf", p))
    if paths:
        refresh_queue_listbox(len(queue_items) - 1)
        update_size_estimate()


def add_images_to_main():
    paths = filedialog.askopenfilenames(
        title="Select Images to Include in PDF",
        filetypes=[("Image Files", " ".join(IMAGE_EXTS)), ("All Files", "*.*")]
    )
    added = 0
    for p in paths:
        if p not in extra_images:
            extra_images.append(p)
            added += 1
    if added:
        img_count_lbl.config(text=f"  {len(extra_images)} standalone image(s) will be appended to the PDF")


def remove_selected_item():
    sel = queue_lb.curselection()
    if not sel:
        return
    idx = sel[0]
    queue_items.pop(idx)
    refresh_queue_listbox(min(idx, len(queue_items) - 1))
    update_size_estimate()


def clear_queue():
    queue_items.clear()
    refresh_queue_listbox(None)
    update_size_estimate()


def reverse_queue():
    if queue_items:
        queue_items.reverse()
        refresh_queue_listbox(None)
        update_size_estimate()


def move_item(delta):
    if queue_lb.cget("state") == "disabled":
        return
    sel = queue_lb.curselection()
    if not sel:
        return
    i = sel[0]
    j = i + delta
    if 0 <= j < len(queue_items):
        yv = queue_lb.yview()[0]
        queue_items[i], queue_items[j] = queue_items[j], queue_items[i]
        refresh_queue_listbox(j, yview_top=yv)
        update_size_estimate()


def move_with_keyboard(delta):
    def handler(event):
        if queue_lb.cget("state") == "disabled":
            return "break"
        if queue_lb.curselection():
            move_item(delta)
            return "break"
    return handler


# =========================
# Build combined PDF
# =========================
def build_combined_pdf():
    global conversion_thread, last_output_pdf
    if conversion_thread and conversion_thread.is_alive():
        ui_set_status("A build is already running — please wait or stop it first.")
        return
    if not queue_items:
        ui_set_status("Please add at least one video or PDF to the queue.")
        return

    meta = {
        "title":       title_var.get().strip() or "Untitled",
        "author":      author_var.get().strip(),
        "tags":        tags_var.get().strip(),
        "type":        type_var.get().strip() or "PDF",
        "published":   published_var.get().strip(),
        "language":    language_var.get().strip(),
        "cover_image": cover_var.get().strip(),
    }
    out = filedialog.asksaveasfilename(
        title="Save Combined PDF As",
        initialfile=safe_filename(meta["title"]) + ".pdf",
        defaultextension=".pdf",
        filetypes=[("PDF Files", "*.pdf")]
    )
    if not out:
        return

    preset_name = preset_var.get()
    stop_event.clear()
    last_output_pdf = None
    ui_enable_open_location(False)
    ui_set_buttons_running(True)
    ui_set_status("Starting...")
    ui_set_eta("Time remaining: —")
    ui_set_progress(0, 1)

    def worker():
        global last_output_pdf
        try:
            ensure_pdf_merge_available()
            temp_root = make_temp_dir_near_output(out)
            writer    = PdfWriter()
            try:
                total = len(queue_items)
                done  = 0
                for idx, item in enumerate(queue_items, 1):
                    if stop_event.is_set():
                        raise RuntimeError("Stopped")
                    label = f"Item {idx} of {total}"
                    if item.kind == "pdf":
                        ui_set_status(f"{label} — Adding PDF: {os.path.basename(item.path)}")
                        ui_set_progress(done, total)
                        append_pdf_to_writer(writer, item.path)
                        done += 1
                        ui_set_progress(done, total)
                    else:
                        ui_set_status(f"{label} — Converting video: {os.path.basename(item.path)}")
                        tmp = os.path.join(temp_root, f"__video_{idx:03d}.pdf")
                        def prog(sv, sm, et, st):
                            ui_set_status(f"{label} — {st}")
                            ui_set_eta(et)
                        video_to_temp_pdf(item.path, tmp, preset_name, meta, prog, stop_event=stop_event)
                        append_pdf_to_writer(writer, tmp)
                        done += 1
                        ui_set_progress(done, total)

                if extra_images:
                    ui_set_status("Appending standalone images...")
                    img_tmp = os.path.join(temp_root, "__extra_images.pdf")
                    build_pdf_from_images(extra_images, img_tmp, meta, bar_height=0)
                    append_pdf_to_writer(writer, img_tmp)

                ui_set_status("Writing final PDF file...")
                pm = meta_to_pdf_metadata(meta)
                if pm:
                    writer.add_metadata(pm)
                try:
                    for page in writer.pages:
                        page.compress_content_streams()
                except Exception:
                    pass
                with open(out, "wb") as f:
                    writer.write(f)
            finally:
                cleanup_dir(temp_root)

            last_output_pdf = out
            ui_set_status("Finished! PDF created successfully.")
            ui_set_eta("Time remaining: 0:00")
            ui_set_progress(1, 1)
            ui_enable_open_location(True)
            ui_popup_done(out)

        except Exception as e:
            msg = str(e).lower().strip()
            ui_set_status("Stopped by user." if msg == "stopped" else f"Error: {e}")
        finally:
            ui_set_buttons_running(False)

    conversion_thread = threading.Thread(target=worker, daemon=True)
    conversion_thread.start()


# =========================
# Extract Frames & Select
# =========================
def extract_frames_and_select():
    global conversion_thread
    if conversion_thread and conversion_thread.is_alive():
        ui_set_status("A build is already running — please wait or stop it first.")
        return
    video_paths = [item.path for item in queue_items if item.kind == "video"]
    if not video_paths:
        messagebox.showwarning("No Videos", "Please add at least one video to the queue first.")
        return

    meta = {
        "title":       title_var.get().strip() or "Untitled",
        "author":      author_var.get().strip(),
        "tags":        tags_var.get().strip(),
        "type":        type_var.get().strip() or "PDF",
        "published":   published_var.get().strip(),
        "language":    language_var.get().strip(),
        "cover_image": cover_var.get().strip(),
    }
    preset_name = preset_var.get()
    stop_event.clear()
    ui_set_buttons_running(True)
    ui_enable_open_location(False)

    temp_dir = tempfile.mkdtemp(prefix="video_to_pdf_select_")
    estimated = estimate_total_frames(video_paths, preset_name)
    ui_set_status(f"Extracting frames... (estimated {estimated} total frames)")
    ui_set_eta("Time remaining: —")
    ui_set_progress(0, 1)

    def worker():
        t0 = time.time()
        try:
            def prog(sv, sm, _eta, st):
                elapsed = time.time() - t0
                if sv > 0 and elapsed > 0.5 and sm > sv:
                    eta = format_seconds((sm - sv) / (sv / elapsed))
                else:
                    eta = "—"
                ui_set_status(f"{st}  —  {sv} of approximately {sm}")
                ui_set_eta(f"Time remaining: ~{eta}")
                ui_set_progress(sv, max(1, sm))

            all_paths, video_collections = extract_all_frames_from_videos(
                video_paths, preset_name, temp_dir,
                progress_callback=prog, estimated_total=estimated, stop_event=stop_event
            )

            if not all_paths:
                ui_call(lambda: messagebox.showerror("Error", "No frames could be extracted."))
                cleanup_dir(temp_dir)
                return

            ui_set_status("Opening frame selector with AI similarity analysis...")
            ui_set_eta("")
            ui_set_progress(1, 1)
            ui_set_buttons_running(False)
            ui_call(lambda: FrameSelectionWindow(
                root, video_collections, temp_dir, meta,
                extra_image_paths=list(extra_images)
            ))

        except RuntimeError as e:
            msg = str(e).lower().strip()
            ui_set_status("Stopped by user." if msg == "stopped" else f"Error: {e}")
            cleanup_dir(temp_dir)
            ui_set_buttons_running(False)
        except Exception as e:
            ui_set_status(f"Error: {e}")
            ui_call(lambda: messagebox.showerror("Error", str(e)))
            cleanup_dir(temp_dir)
            ui_set_buttons_running(False)

    conversion_thread = threading.Thread(target=worker, daemon=True)
    conversion_thread.start()


def stop_conversion():
    stop_event.set()
    ui_set_status("Stopping... please wait.")


def shutdown_app():
    stop_event.set()
    root.after(200, root.destroy)


def open_output_folder():
    if last_output_pdf and os.path.exists(last_output_pdf):
        subprocess.run(["explorer", "/select,", os.path.normpath(last_output_pdf)])


# =========================
# Frame Selection Window
# =========================

# Grid display constants
MAX_SLOTS_IN_GRID    = 20
THUMBNAIL_MAX_WIDTH  = 95    # Smaller thumbnails help fit denser pages (up to 20 thumbs/page)
ZOOM_MAX_WIDTH       = 900
PAGE_NUMBER_BAR_PTS  = 36
FRAMES_PER_PAGE_OPTIONS = (6, 8, 12, 16, 20)


class FrameSelectionWindow:
    """
    Five-column window for reviewing and selecting extracted video frames.

    Column 1: List of source videos
    Column 2: AI Similarity Groups list + controls
    Column 3: Visual previews of each group's last frame
    Column 4: Paginated thumbnail grid with similarity badges
    Column 5: Action buttons (smart select, keep/skip, PDF export, etc.)
    """

    def __init__(self, parent, video_collections, temp_dir, meta, extra_image_paths=None):
        self.video_collections  = video_collections    # List of (video_name, [frame_paths])
        self.temp_dir           = temp_dir
        self.meta               = meta
        self.extra_image_paths: List[str] = list(extra_image_paths or [])

        # keep[(collection_index, frame_index)] = True/False
        self.keep: Dict[Tuple[int, int], bool] = {}
        for ci, (_, paths) in enumerate(video_collections):
            for fi in range(len(paths)):
                self.keep[(ci, fi)] = False

        self.similarity_groups: Dict[int, List[SimilarityGroup]] = {}
        self.frame_to_group:    Dict[int, Tuple[int, float]]     = {}

        self.selected_collection_idx = 0
        self.page_index = 0
        self.frames_per_page = 20
        self._current_filter: Optional[int] = None
        self._filtered_indices: List[int] = []

        # Thumbnail cache: (collection_index, frame_index) -> path to resized PNG
        self._thumb_cache: Dict[Tuple[int, int], str] = {}

        self.win = tk.Toplevel(parent)
        self.win.title("Select Frames to Include in PDF — AI Similarity Analysis")
        self.win.state("zoomed")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._thumb_dir  = os.path.join(temp_dir, "_thumbs")
        os.makedirs(self._thumb_dir, exist_ok=True)
        self._zoom_win  = None
        self._zoom_ref  = None
        self._preview_thumb_refs: List = []

        # ── Root paned window ───────────────────────────────────────
        paned = tk.PanedWindow(self.win, orient=tk.HORIZONTAL, sashrelief="raised", sashwidth=5)
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        # ═══════════ COLUMN 1 — Source Videos ══════════════════════
        c1 = tk.Frame(paned, bg="#f0f4f8")

        tk.Label(c1, text="Source Videos", font=("", 11, "bold"),
                 bg="#2c3e50", fg="white", pady=6, padx=8,
                 anchor="w").pack(fill="x")

        self.video_count_label = tk.Label(c1, text="", font=("", 9),
                                          bg="#f0f4f8", fg="#555")
        self.video_count_label.pack(anchor="w", padx=6, pady=(4, 0))

        vf = tk.Frame(c1, bg="#f0f4f8")
        vf.pack(fill="both", expand=True, padx=4, pady=4)
        vsb = tk.Scrollbar(vf)
        vsb.pack(side="right", fill="y")
        self.video_listbox = tk.Listbox(
            vf, font=("", 10), yscrollcommand=vsb.set,
            exportselection=False, selectbackground="#3498db",
            selectforeground="white", activestyle="dotbox",
            relief="flat", bd=0
        )
        vsb.config(command=self.video_listbox.yview)
        self.video_listbox.pack(fill="both", expand=True)
        self.video_listbox.bind("<<ListboxSelect>>", self._on_video_selected)

        for i, (vname, _) in enumerate(video_collections):
            self.video_listbox.insert(tk.END, f"  {vname}")
            self.video_listbox.itemconfig(i, bg="#ffffff" if i % 2 == 0 else "#eaf2ff")

        self.video_count_label.config(text=f"{len(video_collections)} video(s) in queue")
        paned.add(c1, minsize=220)

        # ═══════════ COLUMN 2 — AI Similarity Groups ═══════════════
        c2 = tk.Frame(paned, bg="#f8f8f0")

        tk.Label(c2, text="AI Similarity Groups", font=("", 11, "bold"),
                 bg="#27ae60", fg="white", pady=6, padx=8,
                 anchor="w").pack(fill="x")

        tk.Label(
            c2,
            text=(
                "Frames that look nearly identical are\n"
                "grouped together. You can keep just the\n"
                "last frame of each group instead of all."
            ),
            font=("", 8), bg="#f8f8f0", fg="#555",
            justify="left"
        ).pack(anchor="w", padx=6, pady=(4, 0))

        # Sensitivity slider
        sens_frame = tk.Frame(c2, bg="#f8f8f0")
        sens_frame.pack(fill="x", padx=6, pady=(6, 0))
        tk.Label(sens_frame, text="Grouping strictness:", font=("", 9, "bold"),
                 bg="#f8f8f0").pack(anchor="w")
        tk.Label(
            sens_frame,
            text="Lower = stricter (more groups, catches subtle differences)\n"
                 "Higher = looser (fewer groups, more frames grouped together)",
            font=("", 7), bg="#f8f8f0", fg="#666", justify="left"
        ).pack(anchor="w")
        slider_row = tk.Frame(c2, bg="#f8f8f0")
        slider_row.pack(fill="x", padx=6, pady=(2, 0))
        tk.Label(slider_row, text="Strict", font=("", 8), bg="#f8f8f0", fg="#888").pack(side="left")
        self.threshold_var = tk.IntVar(value=DEFAULT_THRESHOLD)
        tk.Scale(
            slider_row, from_=2, to=50, orient=tk.HORIZONTAL,
            variable=self.threshold_var, length=120, showvalue=True,
            bg="#f8f8f0", troughcolor="#c8e6c9"
        ).pack(side="left", padx=2)
        tk.Label(slider_row, text="Loose", font=("", 8), bg="#f8f8f0", fg="#888").pack(side="left")
        tk.Button(
            c2, text="Re-analyse with new strictness setting",
            font=("", 8), command=self._rerun_clustering,
            bg="#e8f5e9", relief="groove"
        ).pack(fill="x", padx=6, pady=(4, 2))

        # Color legend
        legend = tk.Frame(c2, bg="#f8f8f0")
        legend.pack(anchor="w", padx=6, pady=4)
        tk.Label(legend, text="Color key:", font=("", 8, "bold"), bg="#f8f8f0").pack(anchor="w")
        for color, label in [
            (SIM_COLOR_REPRESENTATIVE, "Representative (first in group)"),
            (SIM_COLOR_VERY_SIMILAR,   "Very similar (90%+)"),
            (SIM_COLOR_SIMILAR,        "Similar (75–89%)"),
            (SIM_COLOR_SOMEWHAT_SIMILAR, "Somewhat similar (50–74%)"),
            (SIM_COLOR_UNIQUE,         "Unique frame"),
        ]:
            row = tk.Frame(legend, bg="#f8f8f0")
            row.pack(anchor="w")
            tk.Label(row, text="  ", bg=color, width=2, relief="flat").pack(side="left")
            tk.Label(row, text=f"  {label}", font=("", 7), bg="#f8f8f0").pack(side="left")

        # Groups listbox
        glf = tk.Frame(c2, bg="#f8f8f0")
        glf.pack(fill="both", expand=True, padx=4, pady=2)
        gsb = tk.Scrollbar(glf)
        gsb.pack(side="right", fill="y")
        self.groups_listbox = tk.Listbox(
            glf, font=("", 9), yscrollcommand=gsb.set,
            exportselection=False, selectbackground="#27ae60",
            selectforeground="white", relief="flat", bd=0
        )
        gsb.config(command=self.groups_listbox.yview)
        self.groups_listbox.pack(fill="both", expand=True)
        self.groups_listbox.bind("<<ListboxSelect>>", self._on_group_selected)

        # Group action buttons
        ga = tk.Frame(c2, bg="#f8f8f0")
        ga.pack(fill="x", padx=4, pady=(2, 4))
        tk.Button(ga, text="Show All Frames",    font=("", 8),
                  command=self._show_all_frames).pack(fill="x", pady=1)
        tk.Button(ga, text="Keep All in Group",  font=("", 8), bg="#e8f5e9",
                  command=self._keep_group).pack(fill="x", pady=1)
        tk.Button(ga, text="Skip All in Group",  font=("", 8), bg="#ffebee",
                  command=self._skip_group).pack(fill="x", pady=1)

        self.group_info_label = tk.Label(c2, text="", font=("", 8), fg="#555",
                                         bg="#f8f8f0", wraplength=220, justify="left")
        self.group_info_label.pack(anchor="w", padx=6, pady=2)

        paned.add(c2, minsize=220)

        # ═══════════ COLUMN 3 — Group Preview Thumbnails ═══════════
        c3 = tk.Frame(paned, bg="#f0f8f0")

        tk.Label(c3, text="Group Previews", font=("", 11, "bold"),
                 bg="#1e8449", fg="white", pady=6, padx=8,
                 anchor="w").pack(fill="x")
        tk.Label(
            c3,
            text="Shows the last frame of each group.\nClick a preview to filter the grid.",
            font=("", 8), bg="#f0f8f0", fg="#555", justify="left"
        ).pack(anchor="w", padx=6, pady=(4, 2))

        self._preview_canvas = tk.Canvas(c3, bg="#f0f8f0", highlightthickness=0)
        preview_vsb = tk.Scrollbar(c3, orient="vertical",
                                   command=self._preview_canvas.yview)
        self._preview_canvas.configure(yscrollcommand=preview_vsb.set)
        preview_vsb.pack(side="right", fill="y")
        self._preview_canvas.pack(fill="both", expand=True, padx=2, pady=(0, 4))
        self._preview_inner = tk.Frame(self._preview_canvas, bg="#f0f8f0")
        self._preview_window_id = self._preview_canvas.create_window(
            (0, 0), window=self._preview_inner, anchor="nw"
        )
        self._preview_inner.bind("<Configure>", self._on_preview_resized)
        self._preview_canvas.bind("<Configure>", self._on_preview_canvas_resized)
        self._preview_canvas.bind("<MouseWheel>", self._on_preview_scrolled)
        self._preview_inner.bind("<MouseWheel>", self._on_preview_scrolled)

        paned.add(c3, minsize=180)

        # ═══════════ COLUMN 4 — Frame Thumbnail Grid ═══════════════
        c4 = tk.Frame(paned)

        header4 = tk.Frame(c4, bg="#2980b9")
        header4.pack(fill="x")
        tk.Label(header4, text="Frame Preview Grid", font=("", 11, "bold"),
                 bg="#2980b9", fg="white", pady=6, padx=8).pack(side="left", fill="y")
        self.page_range_label = tk.Label(header4, text="", font=("", 10),
                                         bg="#2980b9", fg="white")
        self.page_range_label.pack(side="left", padx=12)
        self.filter_label = tk.Label(header4, text="", font=("", 9, "italic"),
                                     fg="#d6eaf8", bg="#2980b9")
        self.filter_label.pack(side="left", padx=6)

        info4 = tk.Frame(c4, bg="#f9f9f9")
        info4.pack(fill="x", padx=4, pady=2)
        self.selection_count_label = tk.Label(info4, text="", font=("", 9), bg="#f9f9f9")
        self.selection_count_label.pack(side="left", padx=4)
        self.estimated_size_label = tk.Label(info4, text="", font=("", 9),
                                             bg="#f9f9f9", fg="#555")
        self.estimated_size_label.pack(side="left", padx=8)

        # Frames per page control
        pprow = tk.Frame(c4, bg="#f9f9f9")
        pprow.pack(fill="x", padx=4)
        tk.Label(pprow, text="Thumbnails per page:", font=("", 9),
                 bg="#f9f9f9").pack(side="left")
        self.per_page_var = tk.StringVar(value="20")
        per_page_combo = Combobox(
            pprow, textvariable=self.per_page_var,
            values=[str(n) for n in FRAMES_PER_PAGE_OPTIONS],
            state="readonly", width=4
        )
        per_page_combo.pack(side="left", padx=3)
        per_page_combo.bind("<<ComboboxSelected>>", self._on_per_page_changed)
        tk.Label(pprow, text="   Use ←/→ to move pages, S to keep page, D to delete page.",
                 font=("", 8), fg="#777", bg="#f9f9f9").pack(side="left")

        # Thumbnail grid
        self.grid_frame = tk.Frame(c4)
        self.grid_frame.pack(fill="both", expand=True, padx=4, pady=3)

        self.grid_cells:  List[Tuple] = []
        self.check_vars:  List[tk.BooleanVar] = []
        self.photo_refs:  List[Optional[tk.PhotoImage]] = []

        for slot in range(MAX_SLOTS_IN_GRID):
            cell      = tk.Frame(self.grid_frame, relief="ridge", bd=2, bg="light gray")
            badge_lbl = tk.Label(cell, text="", font=("Helvetica", 7, "bold"),
                                 bg="light gray", pady=1, padx=2)
            badge_lbl.pack(fill="x")
            img_lbl = tk.Label(cell, text="", image=None, compound="top", cursor="hand2")
            img_lbl.pack(fill="both", expand=True, padx=2)
            img_lbl.bind("<Button-1>", lambda e, s=slot: self._on_thumbnail_click(s))
            img_lbl.bind("<Button-3>", lambda e, s=slot: self._on_thumbnail_right_click(s))
            v  = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(cell, text="Include in PDF", variable=v,
                                font=("", 8), command=self._on_checkbox_toggled)
            cb.pack(pady=1)
            self.grid_cells.append((cell, badge_lbl, img_lbl, cb))
            self.check_vars.append(v)
            self.photo_refs.append(None)

        # Pagination
        nav = tk.Frame(c4, bg="#ecf0f1")
        nav.pack(fill="x", padx=4, pady=4)
        self.prev_page_btn = tk.Button(nav, text="← Previous Page", width=16,
                                       font=("", 9), command=self._prev_page)
        self.prev_page_btn.pack(side="left", padx=4)
        self.next_page_btn = tk.Button(nav, text="Next Page →", width=16,
                                       font=("", 9), command=self._next_page)
        self.next_page_btn.pack(side="left", padx=4)
        self.keep_page_btn = tk.Button(
            nav, text="Keep This Page", width=14, font=("", 9),
            bg="#2e7d32", fg="white", command=self._keep_current_page
        )
        self.keep_page_btn.pack(side="left", padx=4)
        self.skip_page_btn = tk.Button(
            nav, text="Skip This Page", width=14, font=("", 9),
            bg="#c62828", fg="white", command=self._skip_current_page
        )
        self.skip_page_btn.pack(side="left", padx=4)

        self.page_action_hint = tk.Label(
            nav,
            text="Tip: Press S to keep this page, D to delete this page while browsing thumbnails.",
            font=("", 8), fg="#455a64", bg="#ecf0f1", anchor="w"
        )
        self.page_action_hint.pack(side="left", padx=(10, 0))

        paned.add(c4, minsize=500)

        # ═══════════ COLUMN 5 — Action Buttons ═════════════════════
        c5 = tk.Frame(paned, bg="#2c3e50", width=180)
        c5.pack_propagate(False)

        def _section_header(text, color="#34495e"):
            tk.Label(c5, text=text, font=("", 9, "bold"), bg=color, fg="white",
                     pady=5, padx=8, anchor="w").pack(fill="x", pady=(10, 2))

        def _action_button(text, cmd, bg="#455a64", fg="white"):
            tk.Button(
                c5, text=text, command=cmd, bg=bg, fg=fg, font=("", 9),
                relief="flat", cursor="hand2", pady=5,
                activebackground="#607d8b", activeforeground="white",
                wraplength=150, justify="center"
            ).pack(fill="x", padx=8, pady=2)

        _section_header("Smart Auto-Select", "#1a5276")
        tk.Label(
            c5,
            text="Automatically selects the last (most complete)\nframe from each similar group,\nplus every unique frame.",
            font=("", 8), bg="#2c3e50", fg="#aaaaaa", wraplength=155, justify="left"
        ).pack(anchor="w", padx=8, pady=(2, 0))
        _action_button("Auto-Select This Video\n(last frame per group + unique)", self._smart_select_current, bg="#1a6b2a")
        _action_button("Auto-Select All Videos\n(applies to every video in queue)", self._smart_select_all_videos, bg="#145a32")

        tk.Frame(c5, height=1, bg="#4a6572").pack(fill="x", padx=8, pady=6)

        _section_header("Manual Frame Selection", "#34495e")
        _action_button("Show All Frames",         self._show_all_frames,  bg="#546e7a")
        _action_button("Mark All Shown as Keep",  self._keep_group,       bg="#2e7d32")
        _action_button("Mark All Shown as Skip",  self._skip_group,       bg="#c62828")

        tk.Frame(c5, height=1, bg="#4a6572").pack(fill="x", padx=8, pady=6)

        _section_header("Extra Images to Append", "#5d4037")
        self.extra_images_label = tk.Label(
            c5, text="", font=("", 8), bg="#2c3e50",
            fg="#aaaaaa", wraplength=155, justify="left"
        )
        self.extra_images_label.pack(anchor="w", padx=8, pady=2)
        self._update_extra_images_label()
        _action_button("Add Extra Images",    self._import_extra_images, bg="#6d4c41")
        _action_button("Clear Extra Images",  self._clear_extra_images,  bg="#795548")

        tk.Frame(c5, height=1, bg="#4a6572").pack(fill="x", padx=8, pady=6)

        _section_header("Export to PDF", "#1565c0")

        self.optimize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            c5, text="Compress output (smaller file size)", variable=self.optimize_var,
            bg="#2c3e50", fg="white", selectcolor="#1e2e3e",
            activebackground="#2c3e50", activeforeground="white",
            font=("", 8), command=self._update_selection_size_label
        ).pack(anchor="w", padx=10, pady=2)

        self.split_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            c5, text="Split into multiple files", variable=self.split_var,
            bg="#2c3e50", fg="white", selectcolor="#1e2e3e",
            activebackground="#2c3e50", activeforeground="white", font=("", 8)
        ).pack(anchor="w", padx=10, pady=(2, 0))

        self.export_pngs_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            c5, text="Also export selected frames as PNG files", variable=self.export_pngs_var,
            bg="#2c3e50", fg="white", selectcolor="#1e2e3e",
            activebackground="#2c3e50", activeforeground="white", font=("", 8)
        ).pack(anchor="w", padx=10, pady=(2, 0))

        split_row = tk.Frame(c5, bg="#2c3e50")
        split_row.pack(fill="x", padx=10, pady=(0, 2))
        tk.Label(split_row, text="Max size per file (MB):", bg="#2c3e50",
                 fg="#d7dde2", font=("", 8)).pack(side="left")
        self.split_size_var = tk.StringVar(value="10")
        tk.Entry(split_row, textvariable=self.split_size_var, width=5,
                 bg="#1f2d3a", fg="white", insertbackground="white",
                 relief="flat").pack(side="left", padx=4)

        _action_button("Create PDF from Selected Frames", self._create_pdf, bg="#1565c0")
        _action_button("Show File Size Estimate",         self._show_size_info, bg="#37474f")

        tk.Frame(c5, height=1, bg="#4a6572").pack(fill="x", padx=8, pady=6)
        _action_button("Close Window", self._on_close, bg="#b71c1c")

        paned.add(c5, minsize=170)

        # Keyboard navigation
        self.win.bind("<Left>",  lambda e: (self._prev_page(), "break"))
        self.win.bind("<Right>", lambda e: (self._next_page(), "break"))
        self.win.bind("<KeyPress-s>", lambda e: (self._keep_current_page(), "break"))
        self.win.bind("<KeyPress-S>", lambda e: (self._keep_current_page(), "break"))
        self.win.bind("<KeyPress-d>", lambda e: (self._skip_current_page(), "break"))
        self.win.bind("<KeyPress-D>", lambda e: (self._skip_current_page(), "break"))

        self._apply_grid_layout()
        self.video_listbox.selection_set(0)
        self._on_video_selected(None)

    # ─── Preview column scroll helpers ────────────────────────────
    def _on_preview_resized(self, event=None):
        self._preview_canvas.configure(
            scrollregion=self._preview_canvas.bbox("all")
        )

    def _on_preview_canvas_resized(self, event):
        self._preview_canvas.itemconfig(self._preview_window_id, width=event.width)

    def _on_preview_scrolled(self, event):
        self._preview_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ─── Rebuild group preview thumbnails (column 3) ──────────────
    def _rebuild_group_previews(self, collection_index: int):
        for widget in self._preview_inner.winfo_children():
            widget.destroy()
        self._preview_thumb_refs.clear()

        groups = self.similarity_groups.get(collection_index, [])
        if not groups:
            tk.Label(self._preview_inner, text="Analysing...",
                     bg="#f0f8f0", fg="#999", font=("", 8)).pack(pady=10)
            return

        duplicate_groups = [g for g in groups if not g.is_unique]
        unique_groups    = [g for g in groups if g.is_unique]
        thumb_width = 140

        if duplicate_groups:
            tk.Label(
                self._preview_inner,
                text=f"Similar Groups  ({len(duplicate_groups)} groups)",
                bg="#d4efdf", fg="#1a5e2a", font=("", 8, "bold"),
                anchor="w", padx=4
            ).pack(fill="x", pady=(4, 1))

        for g in duplicate_groups:
            self._add_group_preview_tile(
                g.last_frame_path,
                f"Group {g.group_id + 1}  —  {len(g.frame_paths)} frames  (~{g.average_similarity:.0f}% similar)",
                thumb_width, bg="#e8f8f0", hover_bg="#c8eeda",
                on_click=lambda gid=g.group_id: self._filter_to_group(gid)
            )

        if unique_groups:
            tk.Frame(self._preview_inner, height=1, bg="#aaaaaa").pack(fill="x", padx=4, pady=3)
            tk.Label(
                self._preview_inner,
                text=f"Unique Frames  ({len(unique_groups)} frames)",
                bg="#d6eaf8", fg="#1a4e6d", font=("", 8, "bold"),
                anchor="w", padx=4
            ).pack(fill="x", pady=(0, 1))
            first_unique_path = unique_groups[0].frame_paths[0] if unique_groups else ""
            self._add_group_preview_tile(
                first_unique_path,
                f"All Unique Frames  ({len(unique_groups)} total)",
                thumb_width, bg="#eaf4fb", hover_bg="#cce5f5",
                on_click=self._filter_to_unique
            )

        self._on_preview_resized()

    def _add_group_preview_tile(self, img_path, label_text, thumb_width,
                                bg, hover_bg, on_click):
        tile = tk.Frame(self._preview_inner, bg=bg, cursor="hand2", relief="flat")
        tile.pack(fill="x", padx=4, pady=2)

        photo = self._load_preview_thumbnail(img_path, thumb_width)
        if photo:
            img_label = tk.Label(tile, image=photo, bg=bg, cursor="hand2")
            img_label.pack(pady=(3, 0))
            img_label.bind("<Button-1>", lambda e: on_click())
            img_label.bind("<MouseWheel>", self._on_preview_scrolled)
            self._preview_thumb_refs.append(photo)
        else:
            tk.Label(tile, text="[preview unavailable]",
                     bg=bg, fg="#999", font=("", 7), height=3).pack()

        text_label = tk.Label(tile, text=label_text, bg=bg, fg="#333",
                              font=("", 8), wraplength=thumb_width)
        text_label.pack(pady=(1, 3))

        def on_enter(e, t=tile):
            t.config(bg=hover_bg)
            for child in t.winfo_children():
                try:
                    child.config(bg=hover_bg)
                except Exception:
                    pass

        def on_leave(e, t=tile, b=bg):
            t.config(bg=b)
            for child in t.winfo_children():
                try:
                    child.config(bg=b)
                except Exception:
                    pass

        for widget in [tile, text_label]:
            widget.bind("<Button-1>", lambda e: on_click())
            widget.bind("<MouseWheel>", self._on_preview_scrolled)
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

    def _load_preview_thumbnail(self, img_path: str, max_width: int) -> Optional[tk.PhotoImage]:
        if not img_path or not os.path.exists(img_path):
            return None
        try:
            frame = cv2.imread(img_path)
            if frame is None:
                return None
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(h * max_width / w)),
                                   interpolation=cv2.INTER_AREA)
            tmp_path = os.path.join(self._thumb_dir, f"_preview_{abs(hash(img_path))}.png")
            cv2.imwrite(tmp_path, frame)
            return tk.PhotoImage(file=tmp_path)
        except Exception:
            return None

    def _filter_to_group(self, group_id: int):
        ci     = self.selected_collection_idx
        groups = self.similarity_groups.get(ci, [])
        dup    = [g for g in groups if not g.is_unique]
        for lb_idx, g in enumerate(dup, 1):
            if g.group_id == group_id:
                self.groups_listbox.selection_clear(0, tk.END)
                self.groups_listbox.selection_set(lb_idx)
                self.groups_listbox.see(lb_idx)
                self._on_group_selected(None)
                return

    def _filter_to_unique(self):
        ci     = self.selected_collection_idx
        groups = self.similarity_groups.get(ci, [])
        dup    = [g for g in groups if not g.is_unique]
        lb_idx = len(dup) + 2
        self.groups_listbox.selection_clear(0, tk.END)
        self.groups_listbox.selection_set(lb_idx)
        self.groups_listbox.see(lb_idx)
        self._on_group_selected(None)

    # ─── Grid layout ──────────────────────────────────────────────
    def _apply_grid_layout(self):
        n     = self.frames_per_page
        if n >= 20:
            ncols = 5
        elif n >= 12:
            ncols = 4
        else:
            ncols = 3
        nrows = (n + ncols - 1) // ncols
        for c in range(5):
            self.grid_frame.columnconfigure(c, weight=1 if c < ncols else 0)
        for r in range(5):
            self.grid_frame.rowconfigure(r, weight=1 if r < nrows else 0)
        for slot in range(MAX_SLOTS_IN_GRID):
            cell = self.grid_cells[slot][0]
            if slot < n:
                row, col = slot // ncols, slot % ncols
                cell.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            else:
                cell.grid_remove()

    def _on_per_page_changed(self, event=None):
        try:
            n = int(self.per_page_var.get())
            if n in FRAMES_PER_PAGE_OPTIONS:
                self.frames_per_page = n
                self._apply_grid_layout()
                self._refresh_grid()
        except Exception:
            pass

    # ─── Video selection ──────────────────────────────────────────
    def _on_video_selected(self, event):
        sel = self.video_listbox.curselection()
        if sel:
            self.selected_collection_idx = sel[0]
        ci = self.selected_collection_idx
        for i in range(len(self.video_collections)):
            is_selected = (i == ci)
            self.video_listbox.itemconfig(
                i,
                bg="#3498db" if is_selected else ("#ffffff" if i % 2 == 0 else "#eaf2ff"),
                fg="white" if is_selected else "black"
            )
        _, paths = self.video_collections[ci]
        self._filtered_indices = list(range(len(paths)))
        self._current_filter   = None
        self.filter_label.config(text="")
        self.group_info_label.config(text="")
        self.page_index = 0
        self._start_clustering(ci)
        self._refresh_grid()

    # ─── Clustering ───────────────────────────────────────────────
    def _start_clustering(self, ci: int):
        if ci in self.similarity_groups:
            self._rebuild_frame_lookup(ci)
            self._populate_groups_listbox(ci)
            self._rebuild_group_previews(ci)
            return
        _, paths = self.video_collections[ci]
        if not paths:
            self.similarity_groups[ci] = []
            self.frame_to_group = {}
            self._populate_groups_listbox(ci)
            self._rebuild_group_previews(ci)
            return
        self.groups_listbox.delete(0, tk.END)
        self.groups_listbox.insert(tk.END, "Analysing frames, please wait...")
        self.win.update_idletasks()
        threshold = self.threshold_var.get()

        def _run():
            groups = cluster_frames_by_similarity(paths, threshold=threshold)
            self.similarity_groups[ci] = groups
            self.win.after(0, lambda: self._clustering_finished(ci))

        threading.Thread(target=_run, daemon=True).start()

    def _clustering_finished(self, ci: int):
        self._rebuild_frame_lookup(ci)
        self._populate_groups_listbox(ci)
        self._rebuild_group_previews(ci)
        self._refresh_grid()

    def _rebuild_frame_lookup(self, ci: int):
        """Build a mapping from frame_index → (group_id, similarity_percent)."""
        self.frame_to_group = {}
        _, paths = self.video_collections[ci]
        path_to_index = {p: fi for fi, p in enumerate(paths)}
        for group in self.similarity_groups.get(ci, []):
            for path, sim in zip(group.frame_paths, group.similarities):
                fi = path_to_index.get(path)
                if fi is not None:
                    self.frame_to_group[fi] = (group.group_id, sim)

    def _populate_groups_listbox(self, ci: int):
        self.groups_listbox.delete(0, tk.END)
        groups = self.similarity_groups.get(ci, [])
        if not groups:
            self.groups_listbox.insert(tk.END, "(no frames found)")
            return

        dup_groups  = [g for g in groups if not g.is_unique]
        uniq_groups = [g for g in groups if g.is_unique]

        self.groups_listbox.insert(
            tk.END, f"─── {len(dup_groups)} similar group(s) ───"
        )
        for g in dup_groups:
            self.groups_listbox.insert(
                tk.END,
                f"  Group {g.group_id + 1}:  {len(g.frame_paths)} frames  "
                f"(~{g.average_similarity:.0f}% similar)"
            )
        self.groups_listbox.insert(
            tk.END, f"─── {len(uniq_groups)} unique frame(s) ───"
        )
        self.groups_listbox.insert(tk.END, "  [All unique frames]")

    def _rerun_clustering(self):
        ci = self.selected_collection_idx
        self.similarity_groups.pop(ci, None)
        self._current_filter = None
        self._start_clustering(ci)

    # ─── Group filtering ──────────────────────────────────────────
    def _on_group_selected(self, event):
        sel = self.groups_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        ci  = self.selected_collection_idx
        groups     = self.similarity_groups.get(ci, [])
        dup_groups  = [g for g in groups if not g.is_unique]
        uniq_groups = [g for g in groups if g.is_unique]
        _, paths   = self.video_collections[ci]

        # idx 0 is the "similar groups" header, idx len(dup)+1 is the "unique" header
        if idx == 0 or idx == len(dup_groups) + 1:
            return

        if 1 <= idx <= len(dup_groups):
            g        = dup_groups[idx - 1]
            path_set = set(g.frame_paths)
            self._filtered_indices = [fi for fi, p in enumerate(paths) if p in path_set]
            self._current_filter   = g.group_id
            self.filter_label.config(
                text=f"Showing Group {g.group_id + 1} — {len(g.frame_paths)} frames (~{g.average_similarity:.0f}% similar)"
            )
            self.group_info_label.config(
                text=(
                    f"Group {g.group_id + 1}: {len(g.frame_paths)} frames that look nearly identical.\n"
                    f"Average similarity: {g.average_similarity:.0f}%.\n"
                    f"Tip: You only need to keep the last frame."
                )
            )
        elif idx == len(dup_groups) + 2:
            path_set = set(p for g in uniq_groups for p in g.frame_paths)
            self._filtered_indices = [fi for fi, p in enumerate(paths) if p in path_set]
            self._current_filter   = -1
            self.filter_label.config(
                text=f"Showing {len(self._filtered_indices)} unique frames"
            )
            self.group_info_label.config(
                text="These frames have no similar counterparts.\nEach shows a distinct scene."
            )
        else:
            return

        self.page_index = 0
        self._refresh_grid()

    def _show_all_frames(self):
        ci = self.selected_collection_idx
        _, paths = self.video_collections[ci]
        self._filtered_indices = list(range(len(paths)))
        self._current_filter   = None
        self.filter_label.config(text="")
        self.group_info_label.config(text="")
        self.page_index = 0
        self._refresh_grid()

    def _keep_group(self):
        if not self._filtered_indices:
            return
        ci = self.selected_collection_idx
        for fi in self._filtered_indices:
            self.keep[(ci, fi)] = True
        self._refresh_grid()
        self._update_selection_count()

    def _skip_group(self):
        if not self._filtered_indices:
            return
        ci = self.selected_collection_idx
        for fi in self._filtered_indices:
            self.keep[(ci, fi)] = False
        self._refresh_grid()
        self._update_selection_count()

    def _iter_current_page_indices(self) -> List[int]:
        start = self.page_index * self.frames_per_page
        end = min(start + self.frames_per_page, len(self._filtered_indices))
        return self._filtered_indices[start:end]

    def _keep_current_page(self):
        page_indices = self._iter_current_page_indices()
        if not page_indices:
            return
        ci = self.selected_collection_idx
        for fi in page_indices:
            self.keep[(ci, fi)] = True
        self.page_action_hint.config(text="Kept all thumbnails on this page (shortcut: S keep, D delete).")
        self._refresh_grid()
        self._update_selection_count()

    def _skip_current_page(self):
        page_indices = self._iter_current_page_indices()
        if not page_indices:
            return
        ci = self.selected_collection_idx
        for fi in page_indices:
            self.keep[(ci, fi)] = False
        self.page_action_hint.config(text="Deleted all thumbnails on this page from selection (shortcut: S keep, D delete).")
        self._refresh_grid()
        self._update_selection_count()

    # ─── Smart selection ──────────────────────────────────────────
    def _smart_select_current(self):
        """For the current video, keep the last frame of each similar group + all unique frames."""
        ci = self.selected_collection_idx
        _, all_paths = self.video_collections[ci]
        groups = self.similarity_groups.get(ci, [])
        if not groups:
            messagebox.showinfo(
                "Not Ready",
                "The similarity analysis for this video has not finished yet.\n"
                "Please wait a moment and try again."
            )
            return
        path_to_index = {p: fi for fi, p in enumerate(all_paths)}
        for fi in range(len(all_paths)):
            self.keep[(ci, fi)] = False
        selected = 0
        for g in groups:
            pick = g.frame_paths[0] if g.is_unique else g.last_frame_path
            fi   = path_to_index.get(pick)
            if fi is not None:
                self.keep[(ci, fi)] = True
                selected += 1
        self._show_all_frames()
        self._refresh_grid()
        self._update_selection_count()
        dup_count  = sum(1 for g in groups if not g.is_unique)
        uniq_count = sum(1 for g in groups if g.is_unique)
        messagebox.showinfo(
            "Auto-Selection Complete",
            f"Selected {selected} frames:\n"
            f"  • Last frame from {dup_count} similar group(s)\n"
            f"  • All {uniq_count} unique frame(s)\n\n"
            f"You can fine-tune the selection manually before creating the PDF."
        )

    def _smart_select_all_videos(self):
        """Apply auto-selection to every video in the queue, clustering any not yet done."""
        pending = [ci for ci in range(len(self.video_collections))
                   if ci not in self.similarity_groups]

        if not pending:
            self._apply_auto_select_all()
            return

        # Show progress dialog while clustering
        prog_win = tk.Toplevel(self.win)
        prog_win.title("Analysing Videos...")
        prog_win.geometry("420x140")
        prog_win.transient(self.win)
        prog_win.grab_set()
        prog_win.resizable(False, False)

        tk.Label(prog_win, text="Analysing similarity for all videos...",
                 font=("", 10, "bold")).pack(pady=(16, 4))
        status_lbl = tk.Label(prog_win, text="", font=("", 9), fg="#555")
        status_lbl.pack()
        bar = Progressbar(prog_win, length=360, mode="determinate")
        bar.pack(pady=8)

        completed = [0]
        lock = threading.Lock()

        def cluster_one(ci):
            _, paths  = self.video_collections[ci]
            threshold = self.threshold_var.get()
            groups    = cluster_frames_by_similarity(paths, threshold=threshold)
            self.similarity_groups[ci] = groups
            with lock:
                completed[0] += 1
                done = completed[0]

            def _update():
                bar["value"]   = done
                bar["maximum"] = len(pending)
                vname = self.video_collections[ci][0]
                status_lbl.config(text=f"Done: {vname}  ({done} of {len(pending)})")
                if done == len(pending):
                    prog_win.destroy()
                    self._apply_auto_select_all()
            self.win.after(0, _update)

        for ci in pending:
            threading.Thread(target=cluster_one, args=(ci,), daemon=True).start()

    def _apply_auto_select_all(self):
        """Apply auto-selection across every video collection."""
        total_selected = 0
        total_dup      = 0
        total_uniq     = 0
        for ci, (_, all_paths) in enumerate(self.video_collections):
            groups = self.similarity_groups.get(ci, [])
            if not groups:
                continue
            self._rebuild_frame_lookup_for(ci)
            path_to_index = {p: fi for fi, p in enumerate(all_paths)}
            for fi in range(len(all_paths)):
                self.keep[(ci, fi)] = False
            for g in groups:
                pick = g.frame_paths[0] if g.is_unique else g.last_frame_path
                fi   = path_to_index.get(pick)
                if fi is not None:
                    self.keep[(ci, fi)] = True
                    total_selected += 1
            total_dup  += sum(1 for g in groups if not g.is_unique)
            total_uniq += sum(1 for g in groups if g.is_unique)

        ci = self.selected_collection_idx
        self._rebuild_frame_lookup(ci)
        self._populate_groups_listbox(ci)
        self._rebuild_group_previews(ci)
        self._show_all_frames()
        self._refresh_grid()
        self._update_selection_count()
        messagebox.showinfo(
            "Auto-Selection Complete — All Videos",
            f"Selected {total_selected} frames across {len(self.video_collections)} video(s):\n"
            f"  • Last frame from {total_dup} similar group(s)\n"
            f"  • All {total_uniq} unique frame(s)\n\n"
            f"You can fine-tune the selection manually before creating the PDF."
        )

    def _rebuild_frame_lookup_for(self, ci: int):
        """Rebuild frame_to_group for a specific collection (used during all-video select)."""
        _, paths      = self.video_collections[ci]
        path_to_index = {p: fi for fi, p in enumerate(paths)}
        for group in self.similarity_groups.get(ci, []):
            for path, sim in zip(group.frame_paths, group.similarities):
                fi = path_to_index.get(path)
                if fi is not None and ci == self.selected_collection_idx:
                    self.frame_to_group[fi] = (group.group_id, sim)

    # ─── Thumbnail generation (cached) ────────────────────────────
    def _get_thumbnail(self, img_path: str, ci: int, fi: int) -> str:
        """Return path to a cached thumbnail PNG, creating it if needed."""
        cache_key = (ci, fi)
        if cache_key in self._thumb_cache:
            cached = self._thumb_cache[cache_key]
            if os.path.exists(cached):
                return cached

        out_path = os.path.join(self._thumb_dir, f"v{ci}_f{fi}.png")
        if os.path.exists(out_path):
            self._thumb_cache[cache_key] = out_path
            return out_path

        try:
            frame = cv2.imread(img_path)
            if frame is None:
                return ""
            h, w = frame.shape[:2]
            if w > THUMBNAIL_MAX_WIDTH:
                frame = cv2.resize(
                    frame,
                    (THUMBNAIL_MAX_WIDTH, int(h * THUMBNAIL_MAX_WIDTH / w)),
                    interpolation=cv2.INTER_AREA
                )
            cv2.imwrite(out_path, frame)
            self._thumb_cache[cache_key] = out_path
            return out_path
        except Exception:
            return ""

    # ─── Grid refresh ─────────────────────────────────────────────
    def _refresh_grid(self):
        ci           = self.selected_collection_idx
        _, all_paths = self.video_collections[ci]
        display_list = self._filtered_indices
        n            = self.frames_per_page
        total_frames = len(display_list)
        start        = self.page_index * n
        end          = min(start + n, total_frames)
        total_pages  = max(1, (total_frames + n - 1) // n)
        video_name   = self.video_collections[ci][0]
        filter_note  = "  [filtered view]" if self._current_filter is not None else ""

        self.page_range_label.config(
            text=(
                f"{video_name}  |  Frames {start + 1}–{end} of {total_frames}{filter_note}"
                f"  (Page {self.page_index + 1} of {total_pages})"
            )
        )
        self._update_selection_count()

        for slot in range(MAX_SLOTS_IN_GRID):
            display_pos = start + slot
            cell, badge_lbl, img_lbl, cb = self.grid_cells[slot]

            if slot < n and display_pos < total_frames:
                fi      = display_list[display_pos]
                path    = all_paths[fi]
                thumb   = self._get_thumbnail(path, ci, fi)

                if thumb and os.path.exists(thumb):
                    try:
                        photo = tk.PhotoImage(file=thumb)
                        self.photo_refs[slot] = photo
                        img_lbl.config(image=photo, text="")
                    except Exception:
                        img_lbl.config(image="", text=f"Frame {fi + 1}")
                else:
                    img_lbl.config(image="", text=f"Frame {fi + 1}")

                group_info = self.frame_to_group.get(fi)
                if group_info:
                    gid, sim = group_info
                    if sim >= 99.9:
                        badge_text = f"Frame {fi + 1}  [Group {gid + 1} representative]"
                        badge_bg   = SIM_COLOR_REPRESENTATIVE
                    else:
                        badge_text = f"Frame {fi + 1}  —  {sim:.0f}% similar to representative"
                        badge_bg   = similarity_badge_color(sim)
                else:
                    badge_text = f"Frame {fi + 1}"
                    badge_bg   = "light gray"

                badge_lbl.config(text=badge_text, bg=badge_bg)
                self.check_vars[slot].set(self.keep.get((ci, fi), False))
                cb.config(state="normal")
                self._update_cell_color(slot, fi, ci, badge_bg)
            else:
                self.photo_refs[slot] = None
                for w in (cell, badge_lbl):
                    w.config(bg="#f5f5f5")
                img_lbl.config(image="", text="", bg="#f5f5f5")
                self.check_vars[slot].set(False)
                cb.config(state="disabled")

        self.prev_page_btn.config(
            state="normal" if self.page_index > 0 else "disabled"
        )
        self.next_page_btn.config(
            state="normal" if end < total_frames else "disabled"
        )

    def _update_cell_color(self, slot, fi, ci, default_bg):
        cell, badge_lbl, img_lbl, _ = self.grid_cells[slot]
        bg = "#aaffaa" if self.keep.get((ci, fi), False) else default_bg
        cell.config(bg=bg)
        badge_lbl.config(bg=bg)

    # ─── Click and checkbox handlers ──────────────────────────────
    def _on_thumbnail_click(self, slot):
        ci   = self.selected_collection_idx
        disp = self._filtered_indices
        pos  = self.page_index * self.frames_per_page + slot
        if pos >= len(disp):
            return
        fi = disp[pos]
        self.keep[(ci, fi)] = not self.keep.get((ci, fi), False)
        self.check_vars[slot].set(self.keep[(ci, fi)])
        ginfo = self.frame_to_group.get(fi)
        if ginfo and ginfo[1] < 99.9:
            bg = similarity_badge_color(ginfo[1])
        elif ginfo:
            bg = SIM_COLOR_REPRESENTATIVE
        else:
            bg = "light gray"
        self._update_cell_color(slot, fi, ci, bg)
        self._update_selection_count()

    def _on_checkbox_toggled(self):
        ci    = self.selected_collection_idx
        disp  = self._filtered_indices
        start = self.page_index * self.frames_per_page
        for slot in range(self.frames_per_page):
            pos = start + slot
            if pos < len(disp):
                self.keep[(ci, disp[pos])] = self.check_vars[slot].get()
        self._update_selection_count()

    def _update_selection_count(self):
        total_selected = sum(1 for v in self.keep.values() if v)
        total_frames   = sum(len(p) for _, p in self.video_collections)
        extra          = len(self.extra_image_paths)
        extra_str      = f"  +{extra} extra image(s)" if extra else ""
        self.selection_count_label.config(
            text=f"Selected: {total_selected} of {total_frames} frames{extra_str}"
        )
        self._update_selection_size_label()

    def _update_selection_size_label(self):
        all_paths = self._get_selected_paths() + self.extra_image_paths
        if not all_paths:
            self.estimated_size_label.config(text="")
            return
        optimize = self.optimize_var.get()
        estimate = estimate_pdf_size_bytes(all_paths, optimize=optimize)
        label    = f"Estimated PDF size: ~{format_bytes(estimate)}"
        if optimize:
            label += " (compressed)"
        self.estimated_size_label.config(text=label)

    def _get_selected_paths(self) -> List[str]:
        out = []
        for ci, (_, paths) in enumerate(self.video_collections):
            for fi in range(len(paths)):
                if self.keep.get((ci, fi), False):
                    out.append(paths[fi])
        return out

    # ─── Pagination ───────────────────────────────────────────────
    def _sync_checkboxes_to_keep(self):
        """Save current checkbox states before changing page."""
        ci    = self.selected_collection_idx
        disp  = self._filtered_indices
        start = self.page_index * self.frames_per_page
        for slot in range(self.frames_per_page):
            pos = start + slot
            if pos < len(disp):
                self.keep[(ci, disp[pos])] = self.check_vars[slot].get()

    def _prev_page(self):
        self._sync_checkboxes_to_keep()
        self.page_index = max(0, self.page_index - 1)
        self._refresh_grid()

    def _next_page(self):
        self._sync_checkboxes_to_keep()
        total = max(1, (len(self._filtered_indices) + self.frames_per_page - 1)
                    // self.frames_per_page)
        self.page_index = min(self.page_index + 1, total - 1)
        self._refresh_grid()

    # ─── Zoom on right-click ──────────────────────────────────────
    def _on_thumbnail_right_click(self, slot):
        ci   = self.selected_collection_idx
        _, paths = self.video_collections[ci]
        disp = self._filtered_indices
        pos  = self.page_index * self.frames_per_page + slot
        if pos >= len(disp):
            return
        if self._zoom_win and self._zoom_win.winfo_exists():
            self._zoom_win.destroy()
            self._zoom_win = None
            return
        fi   = disp[pos]
        path = paths[fi]
        try:
            frame = cv2.imread(path)
            if frame is None:
                return
            h, w = frame.shape[:2]
            if w > ZOOM_MAX_WIDTH:
                frame = cv2.resize(frame, (ZOOM_MAX_WIDTH, int(h * ZOOM_MAX_WIDTH / w)),
                                   interpolation=cv2.INTER_AREA)
            zoom_path = os.path.join(self._thumb_dir, "_zoom.png")
            cv2.imwrite(zoom_path, frame)
            ginfo = self.frame_to_group.get(fi)
            if ginfo and ginfo[1] < 99.9:
                extra = f"Group {ginfo[0] + 1}, {ginfo[1]:.0f}% similar"
            elif ginfo:
                extra = f"Group {ginfo[0] + 1} representative"
            else:
                extra = "Unique frame"
            zw = tk.Toplevel(self.win)
            zw.title(f"Frame {fi + 1}  —  {extra}  —  right-click to close")
            zw.protocol("WM_DELETE_WINDOW", lambda: self._close_zoom(zw))
            photo = tk.PhotoImage(file=zoom_path)
            self._zoom_ref = photo
            lbl = tk.Label(zw, image=photo, cursor="hand2")
            lbl.pack(padx=2, pady=2)
            lbl.bind("<Button-3>", lambda e: self._close_zoom(zw))
            zw.bind("<Button-3>",  lambda e: self._close_zoom(zw))
            self._zoom_win = zw
        except Exception:
            pass

    def _close_zoom(self, zw):
        if zw.winfo_exists():
            zw.destroy()
        self._zoom_win = None
        self._zoom_ref = None

    # ─── Extra images ─────────────────────────────────────────────
    def _update_extra_images_label(self):
        n = len(self.extra_image_paths)
        if n == 0:
            self.extra_images_label.config(
                text="No extra images added.\nYou can add images here to include\nthem at the end of the PDF."
            )
        else:
            self.extra_images_label.config(
                text=f"{n} image{'s' if n != 1 else ''} will be appended\nat the end of the PDF."
            )

    def _import_extra_images(self):
        paths = filedialog.askopenfilenames(
            title="Select Images to Include in PDF",
            filetypes=[("Image Files", " ".join(IMAGE_EXTS)), ("All Files", "*.*")]
        )
        if paths:
            added = 0
            for p in paths:
                if p not in self.extra_image_paths:
                    self.extra_image_paths.append(p)
                    added += 1
            self._update_extra_images_label()
            self._update_selection_size_label()
            messagebox.showinfo(
                "Images Added",
                f"Added {added} image(s).  Total extra images: {len(self.extra_image_paths)}"
            )

    def _clear_extra_images(self):
        if not self.extra_image_paths:
            messagebox.showinfo("Nothing to Clear", "No extra images have been added.")
            return
        if messagebox.askyesno(
            "Clear Extra Images",
            f"Remove all {len(self.extra_image_paths)} extra images from the list?"
        ):
            self.extra_image_paths.clear()
            self._update_extra_images_label()
            self._update_selection_size_label()

    # ─── Size info popup ──────────────────────────────────────────
    def _show_size_info(self):
        selected = self._get_selected_paths()
        all_paths = selected + self.extra_image_paths
        if not all_paths:
            messagebox.showinfo("Nothing Selected",
                                "Please select at least one frame first.")
            return
        normal_size    = estimate_pdf_size_bytes(all_paths, optimize=False)
        optimized_size = estimate_pdf_size_bytes(all_paths, optimize=True)
        messagebox.showinfo(
            "PDF Size Estimate",
            f"Video frames selected:  {len(selected)}\n"
            f"Extra images added:     {len(self.extra_image_paths)}\n"
            f"Total pages in PDF:     {len(all_paths)}\n\n"
            f"Normal quality:    approximately {format_bytes(normal_size)}\n"
            f"Compressed quality: approximately {format_bytes(optimized_size)}\n\n"
            f"Note: These are estimates. Actual file size may vary."
        )

    # ─── Create PDF ───────────────────────────────────────────────
    def _create_pdf(self):
        self._sync_checkboxes_to_keep()
        selected     = self._get_selected_paths()
        all_images   = selected + self.extra_image_paths
        if not all_images:
            messagebox.showerror(
                "Nothing Selected",
                "Please select at least one frame or add extra images before creating the PDF."
            )
            return

        doc_title = self.meta.get("title", "").strip() or "Untitled"
        output_path = filedialog.asksaveasfilename(
            title="Save PDF As",
            initialfile=safe_filename(doc_title) + ".pdf",
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if not output_path:
            return

        if self.extra_image_paths and selected:
            confirmed = messagebox.askyesno(
                "Confirm PDF Contents",
                f"The PDF will contain:\n"
                f"  {len(selected)} video frame(s)\n"
                f"  {len(self.extra_image_paths)} extra image(s)\n"
                f"  {len(all_images)} total pages\n\n"
                f"Proceed?"
            )
            if not confirmed:
                return

        split_enabled = self.split_var.get()
        export_pngs = self.export_pngs_var.get()
        target_size_mb = 10.0
        if split_enabled:
            try:
                target_size_mb = float(self.split_size_var.get().strip())
                if target_size_mb <= 0:
                    raise ValueError
            except Exception:
                messagebox.showerror(
                    "Invalid Split Size",
                    "Please enter a valid file size in megabytes (e.g. 10)."
                )
                return

        # Progress dialog
        prog_win = tk.Toplevel(self.win)
        prog_win.title("Creating PDF...")
        prog_win.transient(self.win)
        prog_win.resizable(False, False)
        prog_win.grab_set()
        tk.Label(prog_win, text="Creating PDF, please wait...",
                 font=("", 10, "bold")).pack(padx=12, pady=(10, 6))
        prog_status = tk.Label(prog_win, text="", font=("", 9), fg="#555")
        prog_status.pack(padx=12, pady=(0, 4))
        prog_bar = Progressbar(prog_win, length=380, mode="determinate")
        prog_bar.pack(padx=12, pady=(0, 12))

        def on_progress(step, total, msg):
            def _update():
                prog_bar["maximum"] = max(1, int(total))
                prog_bar["value"]   = max(0, int(step))
                prog_status.config(text=msg)
            self.win.after(0, _update)

        def run():
            global last_output_pdf
            try:
                if split_enabled:
                    outputs = build_split_pdfs(
                        all_images, output_path, self.meta,
                        target_size_mb=target_size_mb,
                        bar_height=PAGE_NUMBER_BAR_PTS,
                        optimize=self.optimize_var.get(),
                        progress_callback=on_progress,
                    )
                    last_output_pdf = outputs[0] if outputs else output_path
                    done_msg = (
                        f"Created {len(outputs)} PDF file(s) in the same folder as:\n"
                        f"{os.path.basename(output_path)}\n\n"
                        f"Total: {len(selected)} frames + {len(self.extra_image_paths)} extra images"
                    )
                else:
                    build_pdf_from_images(
                        all_images, output_path, self.meta,
                        bar_height=PAGE_NUMBER_BAR_PTS,
                        optimize=self.optimize_var.get(),
                        progress_callback=on_progress,
                    )
                    last_output_pdf = output_path
                    done_msg = (
                        f"PDF saved as:\n{os.path.basename(output_path)}\n\n"
                        f"Contains {len(selected)} frames + {len(self.extra_image_paths)} extra images"
                    )

                if export_pngs and selected:
                    pdf_base = os.path.splitext(os.path.basename(output_path))[0]
                    png_dir_base = os.path.join(
                        os.path.dirname(output_path),
                        f"{pdf_base}_selected_frames_png"
                    )
                    png_dir = make_unique_output_dir(png_dir_base)
                    export_images_as_png(
                        selected,
                        png_dir,
                        progress_callback=on_progress,
                        bar_height=PAGE_NUMBER_BAR_PTS,
                        start_page_number=1,
                    )
                    done_msg += (
                        f"\n\nSelected frames were also exported as PNG files in:\n"
                        f"{os.path.basename(png_dir)}"
                    )

                self.win.after(0, lambda: messagebox.showinfo("PDF Created Successfully", done_msg))
                try:
                    ui_call(ui_enable_open_location, True)
                except Exception:
                    pass
            except Exception as e:
                self.win.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.win.after(0, prog_win.destroy)

        threading.Thread(target=run, daemon=True).start()

    def _on_close(self):
        try:
            cleanup_dir(self.temp_dir)
        except Exception:
            pass
        self.win.destroy()
        try:
            ui_call(ui_enable_open_location, True)
        except Exception:
            pass


# =========================
# ═══ MAIN WINDOW ═══
# =========================
root = tk.Tk()
root.title("Video and PDF to Combined PDF Converter")
root.geometry("1060x820")
root.minsize(900, 700)

# ─── Document metadata ────────────────────────────────────────────
mf = tk.LabelFrame(
    root,
    text="Document Information (optional — will be embedded in the PDF)",
    font=("", 9, "bold"), padx=8, pady=6
)
mf.pack(fill="x", padx=12, pady=(10, 4))

field_definitions = [
    ("Title",              "title_var",     "",    None),
    ("Author",             "author_var",    "",    None),
    ("Keywords / Tags",    "tags_var",      "",    None),
    ("Document Type",      "type_var",      "PDF", None),
    ("Publication Date",   "published_var", "",    "e.g. 2026-02-08"),
    ("Language",           "language_var",  "",    None),
    ("Cover Image Path",   "cover_var",     "",    None),
]

vars_dict = {}
for row_idx, (label, var_name, default, hint) in enumerate(field_definitions):
    tk.Label(mf, text=label + ":").grid(row=row_idx, column=0, sticky="w", pady=2)
    v = tk.StringVar(value=default)
    vars_dict[var_name] = v
    tk.Entry(mf, textvariable=v, width=56).grid(row=row_idx, column=1, padx=8, pady=2, sticky="w")
    if hint:
        tk.Label(mf, text=hint, fg="gray", font=("", 8)).grid(row=row_idx, column=2, sticky="w")

title_var     = vars_dict["title_var"]
author_var    = vars_dict["author_var"]
tags_var      = vars_dict["tags_var"]
type_var      = vars_dict["type_var"]
published_var = vars_dict["published_var"]
language_var  = vars_dict["language_var"]
cover_var     = vars_dict["cover_var"]

row_idx += 1
tk.Label(mf, text="Frame Capture Interval:").grid(row=row_idx, column=0, sticky="w", pady=2)
preset_var = tk.StringVar(value="Every 3 seconds — original resolution")
preset_combo = Combobox(
    mf, textvariable=preset_var,
    values=list(PRESETS.keys()),
    state="readonly", width=44
)
preset_combo.grid(row=row_idx, column=1, padx=8, pady=4, sticky="w")
preset_combo.bind("<<ComboboxSelected>>", lambda e: update_size_estimate())
tk.Label(mf, text="How often to capture a frame from each video",
         fg="gray", font=("", 8)).grid(row=row_idx, column=2, sticky="w")

# ─── Queue ────────────────────────────────────────────────────────
content = tk.Frame(root)
content.pack(fill="both", expand=True, padx=12, pady=4)

qf = tk.LabelFrame(
    content,
    text="Files to Convert  (videos will be converted to PDF pages; PDFs will be merged as-is)",
    font=("", 9, "bold")
)
qf.pack(fill="both", expand=True)

queue_lb = tk.Listbox(qf, height=12, font=("", 9), selectbackground="#3498db",
                       selectforeground="white")
queue_lb.pack(fill="both", expand=True, padx=4, pady=4)
queue_lb.bind("<Up>",   move_with_keyboard(-1))
queue_lb.bind("<Down>", move_with_keyboard(1))

qbf = tk.Frame(qf)
qbf.pack(fill="x", padx=4, pady=(0, 6))

add_videos_btn = tk.Button(qbf, text="Add Videos",   width=14, command=add_videos)
add_videos_btn.pack(side="left", padx=2)

add_pdfs_btn   = tk.Button(qbf, text="Add PDFs",     width=12, command=add_pdfs)
add_pdfs_btn.pack(side="left", padx=2)

add_images_btn = tk.Button(qbf, text="Add Images",   width=12, command=add_images_to_main)
add_images_btn.pack(side="left", padx=2)

move_up_btn    = tk.Button(qbf, text="Move Up",      width=9,  command=lambda: move_item(-1))
move_up_btn.pack(side="left", padx=2)

move_down_btn  = tk.Button(qbf, text="Move Down",    width=9,  command=lambda: move_item(1))
move_down_btn.pack(side="left", padx=2)

remove_btn     = tk.Button(qbf, text="Remove",       width=8,  command=remove_selected_item)
remove_btn.pack(side="left", padx=2)

reverse_btn    = tk.Button(qbf, text="Reverse Order", width=12, command=reverse_queue)
reverse_btn.pack(side="left", padx=2)

clear_list_btn = tk.Button(qbf, text="Clear All",    width=8,  command=clear_queue)
clear_list_btn.pack(side="left", padx=2)

img_count_lbl = tk.Label(qf, text="", font=("", 8), fg="#555", anchor="w")
img_count_lbl.pack(anchor="w", padx=6, pady=(0, 2))

# ─── Status bar ───────────────────────────────────────────────────
sf = tk.Frame(root, relief="sunken", bd=1)
sf.pack(fill="x", padx=12, pady=4)

status_label = tk.Label(sf, text="Ready — add files to the queue to get started.",
                         anchor="w", font=("", 9))
status_label.pack(anchor="w", padx=6, pady=(4, 0))

eta_label    = tk.Label(sf, text="", anchor="w", font=("", 9), fg="#555")
eta_label.pack(anchor="w", padx=6)

size_label   = tk.Label(sf, text="Estimated output size: —", anchor="w",
                         font=("", 9), fg="#555")
size_label.pack(anchor="w", padx=6)

estimate_btn = tk.Button(sf, text="Refresh Size Estimate", command=update_size_estimate, width=20)
estimate_btn.pack(anchor="w", padx=6, pady=(2, 4))

progress_bar = Progressbar(sf, length=980, mode="determinate")
progress_bar.pack(padx=6, pady=(0, 6))

# ─── Action buttons ───────────────────────────────────────────────
bf = tk.Frame(root)
bf.pack(fill="x", padx=12, pady=(4, 8))

build_btn = tk.Button(
    bf, text="Generate Combined PDF",
    width=26, font=("", 10, "bold"),
    bg="#1a5276", fg="white",
    command=build_combined_pdf
)
build_btn.pack(side="left", padx=6)

extract_select_btn = tk.Button(
    bf, text="Extract Frames and Select",
    width=26, font=("", 10, "bold"),
    bg="#1a6b2a", fg="white",
    command=extract_frames_and_select
)
extract_select_btn.pack(side="left", padx=6)

tk.Button(bf, text="Stop", width=8, bg="#c0392b", fg="white",
          command=stop_conversion).pack(side="left", padx=6)

open_location_btn = tk.Button(
    bf, text="Open Output Folder", width=18,
    state="disabled", command=open_output_folder
)
open_location_btn.pack(side="left", padx=6)

tk.Button(bf, text="Exit", fg="red", width=8,
          command=shutdown_app).pack(side="right", padx=6)

# ─── Help text ────────────────────────────────────────────────────
help_text = (
    "How to use:  "
    "1. Add videos or PDFs to the queue.  "
    "2. Choose a frame capture interval.  "
    "3. Click 'Generate Combined PDF' to convert automatically, "
    "or 'Extract Frames and Select' to review and hand-pick which frames to include.  "
    "In the frame selector, use 'Auto-Select' to automatically keep one frame per group and all unique frames.  "
    "To get fewer unique frames, increase the 'Grouping Strictness' slider in the frame selector.  "
    "If the 'Generate Combined PDF' button fails, install the required library by running:  pip install pypdf"
)
tk.Label(
    root, text=help_text,
    font=("", 8), wraplength=1020, justify="left",
    fg="#555"
).pack(padx=12, pady=(0, 8))

root.mainloop()
