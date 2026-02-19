import os
import re
import time
import tempfile
import threading
import subprocess
from io import BytesIO
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar, Combobox

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# PDF merge (install if needed: pip install pypdf)
try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    PdfReader = None
    PdfWriter = None


# =========================
# Configuration / Presets
# =========================
# Use a very large width for "original resolution" (no downscale)
ORIGINAL_RES_WIDTH = 99999

PRESETS = {
    "3 sec (Original resolution)": {"seconds": 3, "width": ORIGINAL_RES_WIDTH, "quality": 100},
    "1 sec (Original resolution)": {"seconds": 1, "width": ORIGINAL_RES_WIDTH, "quality": 100},
    "8 sec (Small file ~15MB)": {"seconds": 8, "width": 480, "quality": 38},
    "8 sec (Sharp text)": {"seconds": 8, "width": 1280, "quality": 72},
    "8 sec (Smallest / Slides Only)": {"seconds": 8, "width": 1024, "quality": 62},
    "10 sec (Lower Res)": {"seconds": 10, "width": 1024, "quality": 58},
    "Lecture (Recommended)": {"seconds": 15, "width": 1400, "quality": 68},
    "High Compression": {"seconds": 30, "width": 900, "quality": 48},
    "Detailed": {"seconds": 5, "width": 1600, "quality": 75},
}

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v")


# =========================
# Data model
# =========================
@dataclass
class QueueItem:
    kind: str  # "video" | "pdf"
    path: str


# =========================
# Global state
# =========================
stop_event = threading.Event()
last_output_pdf: Optional[str] = None
conversion_thread: Optional[threading.Thread] = None


# =========================
# Helpers
# =========================
# Windows reserved names (no extension) that cannot be used as filenames
_WIN_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5",
     "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
     "LPT6", "LPT7", "LPT8", "LPT9"}
)


def sanitize_filename(text: str) -> str:
    """Remove characters invalid in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip()


def filename_for_windows(title: str, max_length: int = 200) -> str:
    """
    Build a Windows-safe filename from a document title.
    Strips invalid chars, trims length, and avoids reserved names.
    """
    if not title or not title.strip():
        return "Combined"
    s = sanitize_filename(title)
    s = re.sub(r"\s+", " ", s).strip(" .")  # collapse spaces, trim dots/spaces
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
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:d}:{sec:02d}"


def format_bytes(num_bytes: float) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def estimate_video_pdf_bytes(video_path: str, seconds_step: int, max_width: int, quality: int) -> Optional[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total_frames <= 0:
            return None
        duration_sec = total_frames / fps
        frames = int(duration_sec // seconds_step) + 1

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w <= 0 or h <= 0:
            w, h = 1280, 720
        if w > max_width:
            scale = max_width / float(w)
            w = int(w * scale)
            h = int(h * scale)

        # Rough JPEG size estimate per frame (bytes per pixel scaled by quality).
        base_bpp = 0.10  # ~0.10 bytes/pixel at quality ~60
        bpp = base_bpp * (quality / 60.0)
        per_frame_bytes = max(1.0, w * h * bpp)
        return frames * per_frame_bytes
    finally:
        cap.release()


def estimate_output_size_bytes() -> (float, int):
    settings = PRESETS[preset_var.get()]
    seconds_step = int(settings["seconds"])
    max_width = int(settings["width"])
    quality = int(settings["quality"])

    total = 0.0
    unknown_videos = 0
    for item in queue_items:
        if item.kind == "pdf":
            try:
                total += os.path.getsize(item.path)
            except Exception:
                pass
        else:
            est = estimate_video_pdf_bytes(item.path, seconds_step, max_width, quality)
            if est is None:
                unknown_videos += 1
            else:
                total += est
    # Add a small overhead for PDF structure, then estimate optimization savings.
    total *= 1.05
    optimized_total = total * 0.85
    return optimized_total, unknown_videos


def ui_call(fn, *args, **kwargs):
    root.after(0, lambda: fn(*args, **kwargs))


def ui_set_status(text: str):
    ui_call(status_label.config, text=text)


def ui_set_eta(text: str):
    ui_call(eta_label.config, text=text)


def ui_set_estimated_size(text: str):
    ui_call(size_label.config, text=text)


def ui_set_progress(value: int, maximum: int):
    def _set():
        progress_bar["maximum"] = max(1, int(maximum))
        progress_bar["value"] = max(0, int(value))

    ui_call(_set)


def ui_enable_open_location(enabled: bool):
    ui_call(open_location_btn.config, state=("normal" if enabled else "disabled"))


def ui_set_buttons_running(running: bool):
    state = "disabled" if running else "normal"
    ui_call(add_videos_btn.config, state=state)
    ui_call(add_pdfs_btn.config, state=state)
    ui_call(move_up_btn.config, state=state)
    ui_call(move_down_btn.config, state=state)
    ui_call(remove_btn.config, state=state)
    ui_call(clear_list_btn.config, state=state)
    ui_call(reverse_btn.config, state=state)
    ui_call(build_btn.config, state=state)
    try:
        ui_call(extract_select_btn.config, state=state)
    except NameError:
        pass
    ui_call(preset_combo.config, state=("disabled" if running else "readonly"))
    ui_call(listbox.config, state=state)
    ui_call(estimate_btn.config, state=state)


def ui_popup_done(output_pdf: str):
    def _pop():
        messagebox.showinfo(
            "PDF Created",
            f"Your PDF has been created successfully:\n\n{os.path.basename(output_pdf)}",
        )

    ui_call(_pop)


def make_temp_dir_near_output(output_pdf: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(output_pdf))
    safe_tmp = os.path.join(base_dir, "_video_to_pdf_frames_tmp")
    os.makedirs(safe_tmp, exist_ok=True)
    return tempfile.mkdtemp(dir=safe_tmp)


def cleanup_dir(path: str):
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
        try:
            os.rmdir(path)
        except Exception:
            pass
    except Exception:
        pass


def resize_frame(frame, max_width: int):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# =========================
# Video -> temp PDF (no UI side effects)
# =========================
def video_to_temp_pdf(
    video_path: str,
    temp_pdf_path: str,
    preset_name: str,
    meta: dict,
    progress_cb=None,   # (step_value:int, step_max:int, eta_text:str, status_text:str)
):
    settings = PRESETS[preset_name]
    seconds_step = int(settings["seconds"])
    max_width = int(settings["width"])
    quality = int(settings["quality"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {os.path.basename(video_path)}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if fps > 0 and total_frames > 0:
        duration_sec = total_frames / fps
    else:
        duration_sec = 0.0

    if duration_sec > 0:
        targets = list(range(0, int(duration_sec) + 1, seconds_step))
    else:
        targets = None

    # frames temp dir
    frames_dir = os.path.join(os.path.dirname(temp_pdf_path), "_frames_" + str(int(time.time())))
    os.makedirs(frames_dir, exist_ok=True)

    try:
        start_time = time.time()
        written = 0
        img_paths: List[str] = []

        if targets is not None:
            if progress_cb:
                progress_cb(0, len(targets), "ETA: --", f"Extracting frames: {os.path.basename(video_path)}")

            for idx, tsec in enumerate(targets, start=1):
                if stop_event.is_set():
                    raise RuntimeError("Stopped")

                cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    break

                frame = resize_frame(frame, max_width=max_width)

                img_path = os.path.join(frames_dir, f"t{tsec:06d}.jpg")
                # JPEG encode/write
                ok = cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                if ok:
                    img_paths.append(img_path)
                    written += 1

                if progress_cb:
                    elapsed = time.time() - start_time
                    if idx >= 2:
                        per_item = elapsed / idx
                        eta = max(0.0, (len(targets) - idx) * per_item)
                        eta_text = f"ETA: {format_seconds(eta)}"
                    else:
                        eta_text = "ETA: --"
                    progress_cb(idx, len(targets), eta_text, f"Extracting frames: {os.path.basename(video_path)}")
        else:
            idx = 0
            tsec = 0
            if progress_cb:
                progress_cb(0, 1, "ETA: --", f"Extracting frames: {os.path.basename(video_path)}")

            while True:
                if stop_event.is_set():
                    raise RuntimeError("Stopped")

                cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    break

                frame = resize_frame(frame, max_width=max_width)

                img_path = os.path.join(frames_dir, f"t{tsec:06d}.jpg")
                ok = cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                if ok:
                    img_paths.append(img_path)
                    written += 1

                idx += 1
                if progress_cb:
                    progress_cb(idx, max(1, idx), "ETA: --", f"Extracting frames: {os.path.basename(video_path)}")

                tsec += seconds_step

        cap.release()

        img_paths.sort()
        if not img_paths:
            raise RuntimeError(f"No frames extracted from: {os.path.basename(video_path)}")

        # Build temp PDF
        if progress_cb:
            progress_cb(0, max(1, len(img_paths)), "ETA: --", f"Building PDF pages: {os.path.basename(video_path)}")

        c = canvas.Canvas(temp_pdf_path)
        c.setTitle(meta.get("title", ""))
        c.setAuthor(meta.get("author", ""))
        c.setSubject(meta.get("type", ""))

        for i, img_path in enumerate(img_paths, start=1):
            if stop_event.is_set():
                raise RuntimeError("Stopped")
            img = ImageReader(img_path)
            w, h = img.getSize()
            c.setPageSize((w, h))
            c.drawImage(img, 0, 0, width=w, height=h)
            c.showPage()
            if progress_cb:
                progress_cb(i, len(img_paths), "ETA: --", f"Building PDF pages: {os.path.basename(video_path)}")

        c.save()

    finally:
        try:
            cap.release()
        except Exception:
            pass
        cleanup_dir(frames_dir)


# =========================
# Extract frames only (for selection workflow)
# =========================
def estimate_total_frames(video_paths: List[str], preset_name: str) -> int:
    """Quick estimate of how many frames will be extracted (for ETA). Uses video metadata."""
    settings = PRESETS[preset_name]
    seconds_step = int(settings["seconds"])
    total = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps > 0 and total_frames > 0:
                duration_sec = total_frames / fps
                total += int(duration_sec) // seconds_step + 1
            else:
                total += 1
        finally:
            cap.release()
    return max(1, total)


def _sanitize_folder_name(name: str) -> str:
    """Sanitize a string for use as a folder name (remove invalid filesystem chars)."""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "video"


def extract_all_frames_from_videos(
    video_paths: List[str],
    preset_name: str,
    output_dir: str,
    progress_cb=None,
    estimated_total: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """Extract frames from all videos into output_dir. Each video's frames go in a separate
    subfolder named after the video file. Returns (all_paths, video_collections) where
    video_collections is [(video_basename, [paths]), ...] for GUI grouping.
    Uses seek-based extraction when duration is known (fast: only decodes needed frames);
    falls back to frame-by-frame when metadata is missing so no frames are missed.
    Quality is unchanged: same resolution and JPEG quality either way.
    """
    settings = PRESETS[preset_name]
    seconds_step = int(settings["seconds"])
    max_width = int(settings["width"])
    quality = int(settings["quality"])

    all_paths: List[str] = []
    video_collections: List[Tuple[str, List[str]]] = []
    step_max = estimated_total if estimated_total is not None and estimated_total > 0 else 1

    for v_idx, video_path in enumerate(video_paths):
        if stop_event.is_set():
            raise RuntimeError("Stopped")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {os.path.basename(video_path)}")

        # Create a separate folder for this video's frames
        video_basename = os.path.basename(video_path)
        base_name = os.path.splitext(video_basename)[0]
        folder_name = f"v{v_idx:03d}_{_sanitize_folder_name(base_name)}"
        video_frames_dir = os.path.join(output_dir, folder_name)
        os.makedirs(video_frames_dir, exist_ok=True)
        paths_for_video: List[str] = []

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0

        if duration_sec > 0:
            # Fast path: seek to each target time and read one frame (decode only ~1/step frames)
            targets = list(range(0, int(duration_sec) + 1, seconds_step))
            for idx, tsec in enumerate(targets):
                if stop_event.is_set():
                    raise RuntimeError("Stopped")
                cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = resize_frame(frame, max_width=max_width)
                img_path = os.path.join(video_frames_dir, f"t{tsec:06d}.jpg")
                ok = cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                if ok:
                    all_paths.append(img_path)
                    paths_for_video.append(img_path)
                if progress_cb:
                    progress_cb(len(all_paths), step_max, "", f"Extracting: {os.path.basename(video_path)}")
        else:
            # Fallback: no reliable duration — read every frame and sample by time (no frames missed)
            next_target_sec = 0
            frame_count = 0
            fps = fps or 25.0
            while True:
                if stop_event.is_set():
                    raise RuntimeError("Stopped")
                ret, frame = cap.read()
                if not ret:
                    break
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                pos_sec = (pos_msec / 1000.0) if pos_msec >= 0 else (frame_count / fps)
                if pos_sec >= next_target_sec:
                    frame = resize_frame(frame, max_width=max_width)
                    tsec = int(next_target_sec)
                    img_path = os.path.join(video_frames_dir, f"t{tsec:06d}.jpg")
                    ok = cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                    if ok:
                        all_paths.append(img_path)
                        paths_for_video.append(img_path)
                    next_target_sec += seconds_step
                    if progress_cb:
                        progress_cb(
                            len(all_paths),
                            step_max,
                            "",
                            f"Extracting: {os.path.basename(video_path)}",
                        )
                frame_count += 1

        cap.release()
        video_collections.append((video_basename, paths_for_video))

    return all_paths, video_collections


# When optimizing PDF, resize images above this width and re-save at this JPEG quality
OPTIMIZE_MAX_WIDTH = 1600
OPTIMIZE_QUALITY = 82


def estimate_pdf_size_bytes(image_paths: List[str], optimize: bool = False) -> int:
    """Estimate final PDF size from selected image file sizes. optimize=True uses a smaller factor."""
    try:
        total = sum(os.path.getsize(p) for p in image_paths if os.path.exists(p))
    except Exception:
        return 0
    overhead = 1.12 if not optimize else 1.15  # PDF structure + bar
    if optimize:
        total = int(total * 0.45)  # rough reduction from resize+lower quality
    return int(total * overhead)


def build_pdf_from_selected_images(
    image_paths: List[str],
    output_pdf: str,
    meta: dict,
    bar_height: int = 36,
    progress_cb=None,
    optimize: bool = False,
    split_max_bytes: Optional[int] = None,
):
    """Build a PDF with one page per image; each page has image then black bar with photo number (#1, #2, ...).
    If optimize=True, images are resized (max width 1600) and re-encoded at lower JPEG quality for smaller file size."""
    from reportlab.lib.colors import black

    split_temp_dir = None
    build_target = output_pdf
    if split_max_bytes:
        split_temp_dir = tempfile.mkdtemp(prefix="video_to_pdf_split_")
        build_target = os.path.join(split_temp_dir, "combined_for_split.pdf")

    c = canvas.Canvas(build_target)
    c.setTitle(meta.get("title", ""))
    c.setAuthor(meta.get("author", ""))
    c.setSubject(meta.get("type", ""))

    temp_opt_dir = None
    if optimize:
        temp_opt_dir = tempfile.mkdtemp(prefix="video_to_pdf_opt_")
    try:
        for i, img_path in enumerate(image_paths, start=1):
            if progress_cb:
                progress_cb(i, len(image_paths), f"Page {i}/{len(image_paths)}")
            if optimize and os.path.exists(img_path):
                frame = cv2.imread(img_path)
                if frame is not None:
                    h, w = frame.shape[:2]
                    if w > OPTIMIZE_MAX_WIDTH:
                        scale = OPTIMIZE_MAX_WIDTH / float(w)
                        new_w = OPTIMIZE_MAX_WIDTH
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    opt_path = os.path.join(temp_opt_dir, f"opt_{i:06d}.jpg")
                    cv2.imwrite(opt_path, frame, [cv2.IMWRITE_JPEG_QUALITY, OPTIMIZE_QUALITY])
                    img_path = opt_path
            img = ImageReader(img_path)
            w, h = img.getSize()
            page_w = w
            page_h = h + bar_height
            c.setPageSize((page_w, page_h))
            c.drawImage(img, 0, bar_height, width=w, height=h)
            c.setFillColor(black)
            c.rect(0, 0, page_w, bar_height, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1)
            c.setFont("Helvetica", 14)
            c.drawString(12, bar_height * 0.25, f"#{i}")
            c.showPage()
    finally:
        if temp_opt_dir:
            cleanup_dir(temp_opt_dir)
    c.save()
    try:
        add_metadata_to_pdf_file(build_target, meta)
    except Exception:
        pass

    if split_max_bytes:
        try:
            split_files = split_pdf_file_by_size(
                build_target,
                split_max_bytes,
                output_dir=os.path.dirname(output_pdf),
                output_base_name=os.path.splitext(os.path.basename(output_pdf))[0],
            )
            if not split_files:
                raise RuntimeError("Split failed: no output files were created.")
            return split_files
        finally:
            if split_temp_dir:
                cleanup_dir(split_temp_dir)

    return [output_pdf]


# =========================
# Frame selection window (configurable images per page, keep/discard, Create PDF)
# =========================
MAX_IMAGES_PER_PAGE = 15  # 3x5 grid
THUMB_MAX_WIDTH = 220
ZOOM_MAX_WIDTH = 900  # max width for right-click zoom view
BAR_HEIGHT_PTS = 36
IMAGES_PER_PAGE_CHOICES = (8, 12, 15)
SPLIT_PART_MAX_BYTES = 10 * 1024 * 1024


# =========================
# Video-based frame selection (frames grouped by source video file)
# =========================
class VideoCollectionFrameSelectionWindow:
    """Frame selection with frames grouped by source video. Click a video to view its frames."""

    def __init__(self, parent: tk.Tk, video_collections: List[Tuple[str, List[str]]], temp_dir: str, meta: dict):
        # video_collections: [(video_basename, [paths]), ...]
        self.video_collections = video_collections
        self.temp_dir = temp_dir
        self.meta = meta
        self.selected_collection_idx = 0
        self.keep = {}  # (coll_idx, frame_idx) -> bool
        for ci, (_, paths) in enumerate(video_collections):
            for fi in range(len(paths)):
                self.keep[(ci, fi)] = False
        self.page_index = 0
        self.images_per_page = 8

        self.win = tk.Toplevel(parent)
        self.win.title("Select Frames for PDF — by Video")
        self.win.geometry("1150x750")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._thumb_dir = os.path.join(temp_dir, "_thumbs")
        os.makedirs(self._thumb_dir, exist_ok=True)
        self._zoom_window = None
        self._zoom_photo_ref = None

        paned = tk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=8, pady=8)

        left_frame = tk.Frame(paned, width=240)
        tk.Label(left_frame, text="Source videos", font=("", 11, "bold")).pack(anchor="w")
        tk.Label(left_frame, text="Click a video to view its frames", fg="gray").pack(anchor="w")
        scroll = tk.Scrollbar(left_frame)
        self.collection_listbox = tk.Listbox(left_frame, height=20, width=32, font=("", 10), yscrollcommand=scroll.set)
        scroll.config(command=self.collection_listbox.yview)
        self.collection_listbox.pack(fill="both", expand=True, pady=6)
        scroll.pack(side="right", fill="y")
        self.collection_listbox.bind("<<ListboxSelect>>", self._on_collection_select)
        for name, paths in video_collections:
            n = len(paths)
            self.collection_listbox.insert(tk.END, f"{name} — {n} frame{'s' if n != 1 else ''}")
        left_frame.pack(fill="both", expand=True)
        paned.add(left_frame, minsize=220)

        right_frame = tk.Frame(paned)
        top = tk.Frame(right_frame)
        top.pack(fill="x", pady=(0, 6))
        self.range_label = tk.Label(top, text="", font=("", 11))
        self.range_label.pack(anchor="w")
        self.count_label = tk.Label(top, text="", font=("", 10))
        self.count_label.pack(anchor="w")
        self.size_estimate_label = tk.Label(top, text="Estimated PDF size: —", font=("", 10))
        self.size_estimate_label.pack(anchor="w")

        row0 = tk.Frame(right_frame)
        row0.pack(fill="x", pady=2)
        tk.Label(row0, text="Images per page:").pack(side="left", padx=(0, 6))
        self.per_page_var = tk.StringVar(value="8")
        per_page_combo = Combobox(
            row0, textvariable=self.per_page_var,
            values=[str(n) for n in IMAGES_PER_PAGE_CHOICES],
            state="readonly", width=4,
        )
        per_page_combo.pack(side="left", padx=2)
        per_page_combo.bind("<<ComboboxSelected>>", self._on_per_page_change)
        tk.Label(row0, text="(use ← → arrow keys to change page)").pack(side="left", padx=12)

        self.grid_frame = tk.Frame(right_frame)
        self.grid_frame.pack(fill="both", expand=True, padx=4, pady=6)
        self.cell_frames = []
        self.check_vars = []
        self.photo_refs = []
        for slot in range(MAX_IMAGES_PER_PAGE):
            cell = tk.Frame(self.grid_frame, relief="ridge", bd=2, bg="light gray")
            lbl = tk.Label(cell, text="", image=None, compound="top", cursor="hand2")
            lbl.pack(fill="both", expand=True, padx=2, pady=2)
            lbl.bind("<Button-1>", lambda e, s=slot: self._on_photo_click(s))
            lbl.bind("<Button-3>", lambda e, s=slot: self._on_photo_right_click(s))
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(cell, text="Keep", variable=var, command=self._on_toggle)
            cb.pack(pady=2)
            self.cell_frames.append((cell, lbl, cb))
            self.check_vars.append(var)
            self.photo_refs.append(None)

        bot = tk.Frame(right_frame)
        bot.pack(fill="x", pady=8)
        self.prev_btn = tk.Button(bot, text="Previous", width=12, command=self._prev_page)
        self.prev_btn.pack(side="left", padx=6)
        self.next_btn = tk.Button(bot, text="Next", width=12, command=self._next_page)
        self.next_btn.pack(side="left", padx=6)
        self.optimize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bot, text="Optimize PDF (smaller file)", variable=self.optimize_var, command=self._update_size_estimate).pack(side="left", padx=12)
        self.split_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bot, text="Split into 10 MB parts", variable=self.split_var).pack(side="left", padx=6)
        tk.Button(bot, text="Create PDF", width=14, command=self._create_pdf).pack(side="left", padx=20)
        tk.Button(bot, text=" i ", width=3, command=self._show_size_info).pack(side="left", padx=4)

        right_frame.pack(fill="both", expand=True)
        paned.add(right_frame, minsize=700)

        def on_left(e):
            if self.page_index > 0:
                self._prev_page()
            return "break"

        def on_right(e):
            self._next_page()
            return "break"

        self.win.bind("<Left>", on_left)
        self.win.bind("<Right>", on_right)

        self._apply_grid_layout()
        self.collection_listbox.selection_set(0)
        self._on_collection_select(None)

    def _current_frames(self) -> List[str]:
        return self.video_collections[self.selected_collection_idx][1]

    def _on_collection_select(self, event):
        sel = self.collection_listbox.curselection()
        if sel:
            self.selected_collection_idx = sel[0]
            self.page_index = 0
            self._refresh_page()

    def _apply_grid_layout(self):
        n = self.images_per_page
        ncols, nrows = (4, 2) if n == 8 else ((4, 3) if n == 12 else (5, 3))
        for c in range(5):
            self.grid_frame.columnconfigure(c, weight=1 if c < ncols else 0)
        for r in range(3):
            self.grid_frame.rowconfigure(r, weight=1 if r < nrows else 0)
        for slot in range(MAX_IMAGES_PER_PAGE):
            cell, lbl, cb = self.cell_frames[slot]
            if slot < n:
                row, col = (slot // 5, slot % 5) if n == 15 else (slot // ncols, slot % ncols)
                cell.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            else:
                cell.grid_remove()

    def _on_per_page_change(self, event=None):
        try:
            n = int(self.per_page_var.get())
            if n in IMAGES_PER_PAGE_CHOICES:
                self.images_per_page = n
                self._apply_grid_layout()
                self._refresh_page()
        except (ValueError, TypeError):
            pass

    def _make_thumbnail(self, img_path: str, coll_idx: int, frame_idx: int) -> str:
        out = os.path.join(self._thumb_dir, f"v{coll_idx}_f{frame_idx}.png")
        if os.path.exists(out):
            return out
        try:
            frame = cv2.imread(img_path)
            if frame is None:
                return ""
            h, w = frame.shape[:2]
            if w > THUMB_MAX_WIDTH:
                scale = THUMB_MAX_WIDTH / float(w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out, frame)
            return out
        except Exception:
            return ""

    def _on_photo_click(self, slot: int):
        frames = self._current_frames()
        start = self.page_index * self.images_per_page
        idx = start + slot
        if idx >= len(frames):
            return
        ci, fi = self.selected_collection_idx, idx
        self.keep[(ci, fi)] = not self.keep.get((ci, fi), False)
        self.check_vars[slot].set(self.keep[(ci, fi)])
        self._update_cell_appearance(slot)
        self._update_count_label()

    def _update_cell_appearance(self, slot: int):
        frames = self._current_frames()
        start = self.page_index * self.images_per_page
        idx = start + slot
        if idx < len(frames):
            ci, fi = self.selected_collection_idx, idx
            cell, _, _ = self.cell_frames[slot]
            cell.config(bg="light green" if self.keep.get((ci, fi), False) else "light gray")

    def _on_toggle(self):
        frames = self._current_frames()
        start = self.page_index * self.images_per_page
        for i in range(self.images_per_page):
            idx = start + i
            if idx < len(frames):
                ci, fi = self.selected_collection_idx, idx
                self.keep[(ci, fi)] = self.check_vars[i].get()
        self._update_count_label()

    def _update_count_label(self):
        total_selected = sum(1 for k in self.keep.values() if k)
        total_frames = sum(len(p) for _, p in self.video_collections)
        self.count_label.config(text=f"Selected: {total_selected} of {total_frames} frames to keep")
        self._update_size_estimate()

    def _update_size_estimate(self):
        selected = self._get_selected_paths_ordered()
        if not selected:
            self.size_estimate_label.config(text="Estimated PDF size: —")
            return
        optimize = self.optimize_var.get()
        est = estimate_pdf_size_bytes(selected, optimize=optimize)
        self.size_estimate_label.config(text=f"Estimated PDF size: ~{format_bytes(est)}" + (" (optimized)" if optimize else ""))

    def _get_selected_paths_ordered(self) -> List[str]:
        out = []
        for ci, (_, paths) in enumerate(self.video_collections):
            for fi in range(len(paths)):
                if self.keep.get((ci, fi), False):
                    out.append(paths[fi])
        return out

    def _refresh_page(self):
        frames = self._current_frames()
        n = self.images_per_page
        start = self.page_index * n
        end = min(start + n, len(frames))
        total = len(frames)
        total_pages = max(1, (total + n - 1) // n)
        video_name = self.video_collections[self.selected_collection_idx][0]
        self.range_label.config(
            text=f"{video_name} — Frames {start + 1}–{end} of {total}  (page {self.page_index + 1} of {total_pages})"
        )
        self._update_count_label()

        for i in range(MAX_IMAGES_PER_PAGE):
            idx = start + i
            cell, lbl, cb = self.cell_frames[i]
            if i < n and idx < len(frames):
                path = frames[idx]
                thumb_path = self._make_thumbnail(path, self.selected_collection_idx, idx)
                if thumb_path and os.path.exists(thumb_path):
                    try:
                        photo = tk.PhotoImage(file=thumb_path)
                        self.photo_refs[i] = photo
                        lbl.config(image=photo, text="")
                    except Exception:
                        lbl.config(image="", text="")
                else:
                    lbl.config(image="", text="")
                ci, fi = self.selected_collection_idx, idx
                self.check_vars[i].set(self.keep.get((ci, fi), False))
                cb.config(state="normal")
                self._update_cell_appearance(i)
            else:
                self.photo_refs[i] = None
                cell.config(bg="SystemButtonFace")
                lbl.config(image="", text="")
                self.check_vars[i].set(False)
                cb.config(state="disabled")

        self.prev_btn.config(state="normal" if self.page_index > 0 else "disabled")
        self.next_btn.config(state="normal" if end < total else "disabled")

    def _prev_page(self):
        frames = self._current_frames()
        n = self.images_per_page
        start = self.page_index * n
        for i in range(n):
            idx = start + i
            if idx < len(frames):
                ci, fi = self.selected_collection_idx, idx
                self.keep[(ci, fi)] = self.check_vars[i].get()
        self.page_index = max(0, self.page_index - 1)
        self._refresh_page()

    def _next_page(self):
        frames = self._current_frames()
        n = self.images_per_page
        start = self.page_index * n
        for i in range(n):
            idx = start + i
            if idx < len(frames):
                ci, fi = self.selected_collection_idx, idx
                self.keep[(ci, fi)] = self.check_vars[i].get()
        total_pages = max(1, (len(frames) + n - 1) // n)
        self.page_index = min(self.page_index + 1, total_pages - 1)
        self._refresh_page()

    def _on_photo_right_click(self, slot: int):
        frames = self._current_frames()
        idx = self.page_index * self.images_per_page + slot
        if idx >= len(frames):
            return
        if self._zoom_window and self._zoom_window.winfo_exists():
            self._zoom_window.destroy()
            self._zoom_window = None
            self._zoom_photo_ref = None
            return
        path = frames[idx]
        try:
            frame = cv2.imread(path)
            if frame is None:
                return
            h, w = frame.shape[:2]
            if w > ZOOM_MAX_WIDTH:
                scale = ZOOM_MAX_WIDTH / float(w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            zoom_path = os.path.join(self._thumb_dir, "_zoom_preview.png")
            cv2.imwrite(zoom_path, frame)
            zoom_win = tk.Toplevel(self.win)
            zoom_win.title(f"Frame #{idx + 1} — right-click to close")
            zoom_win.protocol("WM_DELETE_WINDOW", lambda: self._close_zoom(zoom_win))
            photo = tk.PhotoImage(file=zoom_path)
            self._zoom_photo_ref = photo
            lbl = tk.Label(zoom_win, image=photo, cursor="hand2")
            lbl.pack(padx=2, pady=2)
            lbl.bind("<Button-3>", lambda e: self._close_zoom(zoom_win))
            zoom_win.bind("<Button-3>", lambda e: self._close_zoom(zoom_win))
            self._zoom_window = zoom_win
        except Exception:
            pass

    def _close_zoom(self, zoom_win):
        if zoom_win.winfo_exists():
            zoom_win.destroy()
        self._zoom_window = None
        self._zoom_photo_ref = None

    def _show_size_info(self):
        selected = self._get_selected_paths_ordered()
        if not selected:
            messagebox.showinfo("Info", "Select at least one frame to see size estimates.")
            return
        normal = estimate_pdf_size_bytes(selected, optimize=False)
        optimized = estimate_pdf_size_bytes(selected, optimize=True)
        msg = f"Current selection: {len(selected)} frames\n\n"
        msg += f"PDF size (no optimization): ~{format_bytes(normal)}\n"
        msg += f"PDF size (with optimization): ~{format_bytes(optimized)}"
        messagebox.showinfo("PDF Size Info", msg)

    def _create_pdf(self):
        frames = self._current_frames()
        n = self.images_per_page
        start = self.page_index * n
        for i in range(n):
            idx = start + i
            if idx < len(frames):
                ci, fi = self.selected_collection_idx, idx
                self.keep[(ci, fi)] = self.check_vars[i].get()

        selected = self._get_selected_paths_ordered()
        if not selected:
            messagebox.showerror("Error", "Select at least one photo to keep.")
            return

        doc_title = self.meta.get("title", "").strip() or "Untitled"
        safe_name = filename_for_windows(doc_title) + ".pdf"
        output_pdf = filedialog.asksaveasfilename(
            title="Save PDF",
            initialfile=safe_name,
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
        )
        if not output_pdf:
            return

        def run():
            global last_output_pdf
            try:
                created_files = build_pdf_from_selected_images(
                    selected, output_pdf, self.meta,
                    bar_height=BAR_HEIGHT_PTS,
                    optimize=self.optimize_var.get(),
                    split_max_bytes=(SPLIT_PART_MAX_BYTES if self.split_var.get() else None),
                )
                last_output_pdf = created_files[0] if created_files else output_pdf
                if self.split_var.get():
                    msg = f"Saved split PDFs starting with: {os.path.basename(output_pdf)}"
                else:
                    msg = f"Saved: {os.path.basename(output_pdf)}"
                self.win.after(0, lambda: messagebox.showinfo("PDF Created", msg))
            except Exception as e:
                self.win.after(0, lambda: messagebox.showerror("Error", str(e)))
            try:
                ui_call(ui_enable_open_location, True)
            except NameError:
                pass

        threading.Thread(target=run, daemon=True).start()

    def _on_close(self):
        try:
            cleanup_dir(self.temp_dir)
        except Exception:
            pass
        self.win.destroy()
        try:
            ui_call(ui_enable_open_location, True)
        except NameError:
            pass


class FrameSelectionWindow:
    def __init__(self, parent: tk.Tk, all_paths: List[str], temp_dir: str, meta: dict):
        self.all_paths = all_paths
        self.temp_dir = temp_dir
        self.meta = meta
        self.keep = [False] * len(all_paths)  # start deselected; user selects which to keep
        self.page_index = 0
        self.images_per_page = 8
        self.win = tk.Toplevel(parent)
        self.win.title("Select Frames for PDF")
        self.win.geometry("1000x720")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._thumb_dir = os.path.join(temp_dir, "_thumbs")
        os.makedirs(self._thumb_dir, exist_ok=True)
        self._zoom_window = None
        self._zoom_photo_ref = None

        self.win.bind("<Left>", self._on_left_key)
        self.win.bind("<Right>", self._on_right_key)

        # Status: "Photos 1–8 of 120" and "Selected: 95 of 120 to keep"
        top = tk.Frame(self.win)
        top.pack(fill="x", padx=10, pady=8)
        self.range_label = tk.Label(top, text="", font=("", 11))
        self.range_label.pack(anchor="w")
        self.count_label = tk.Label(top, text="", font=("", 10))
        self.count_label.pack(anchor="w")
        self.size_estimate_label = tk.Label(top, text="Estimated PDF size: —", font=("", 10))
        self.size_estimate_label.pack(anchor="w")

        # Images per page dropdown
        row0 = tk.Frame(self.win)
        row0.pack(fill="x", padx=10, pady=2)
        tk.Label(row0, text="Images per page:").pack(side="left", padx=(0, 6))
        self.per_page_var = tk.StringVar(value="8")
        per_page_combo = Combobox(
            row0,
            textvariable=self.per_page_var,
            values=[str(n) for n in IMAGES_PER_PAGE_CHOICES],
            state="readonly",
            width=4,
        )
        per_page_combo.pack(side="left", padx=2)
        per_page_combo.bind("<<ComboboxSelected>>", self._on_per_page_change)
        tk.Label(row0, text="(use ← → arrow keys to change page)").pack(side="left", padx=12)

        # Grid: up to 15 cells; layout depends on images_per_page (8→2x4, 12→3x4, 15→3x5)
        self.grid_frame = tk.Frame(self.win)
        self.grid_frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.cell_frames = []  # (cell_frame, lbl, cb) per slot 0..14
        self.check_vars = []
        self.photo_refs = []
        for slot in range(MAX_IMAGES_PER_PAGE):
            cell = tk.Frame(self.grid_frame, relief="ridge", bd=2, bg="light gray")
            lbl = tk.Label(cell, text="", image=None, compound="top", cursor="hand2")
            lbl.pack(fill="both", expand=True, padx=2, pady=2)
            lbl.bind("<Button-1>", lambda e, s=slot: self._on_photo_click(s))
            lbl.bind("<Button-3>", lambda e, s=slot: self._on_photo_right_click(s))
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(cell, text="Keep", variable=var, command=self._on_toggle)
            cb.pack(pady=2)
            self.cell_frames.append((cell, lbl, cb))
            self.check_vars.append(var)
            self.photo_refs.append(None)

        # Navigation + Create PDF
        bot = tk.Frame(self.win)
        bot.pack(fill="x", padx=10, pady=10)
        self.prev_btn = tk.Button(bot, text="Previous", width=12, command=self._prev_page)
        self.prev_btn.pack(side="left", padx=6)
        self.next_btn = tk.Button(bot, text="Next", width=12, command=self._next_page)
        self.next_btn.pack(side="left", padx=6)
        self.optimize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bot, text="Optimize PDF (smaller file)", variable=self.optimize_var, command=self._update_size_estimate).pack(side="left", padx=12)
        self.split_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bot, text="Split into 10 MB parts", variable=self.split_var).pack(side="left", padx=6)
        tk.Button(bot, text="Create PDF", width=14, command=self._create_pdf).pack(side="left", padx=20)
        tk.Button(bot, text=" i ", width=3, command=self._show_size_info).pack(side="left", padx=4)

        note = (
            "Previews: full-size frames and small preview images are stored in a temporary folder. "
            "They are deleted when you close this window or create the PDF. Nothing is saved to your own folders."
        )
        tk.Label(self.win, text=note, wraplength=980, justify="left", fg="gray").pack(padx=10, pady=4)

        self._apply_grid_layout()
        self._refresh_page()

    def _apply_grid_layout(self):
        """Show only the cells for current images_per_page; layout 2x4, 3x4, or 3x5."""
        n = self.images_per_page
        if n == 8:
            ncols, nrows = 4, 2
        elif n == 12:
            ncols, nrows = 4, 3
        else:
            ncols, nrows = 5, 3
        for c in range(5):
            self.grid_frame.columnconfigure(c, weight=1 if c < ncols else 0)
        for r in range(3):
            self.grid_frame.rowconfigure(r, weight=1 if r < nrows else 0)
        for slot in range(MAX_IMAGES_PER_PAGE):
            cell, lbl, cb = self.cell_frames[slot]
            if slot < n:
                if n == 15:
                    row, col = slot // 5, slot % 5
                else:
                    row, col = slot // ncols, slot % ncols
                cell.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            else:
                cell.grid_remove()

    def _on_left_key(self, event):
        if self.page_index > 0:
            self._prev_page()
        return "break"

    def _on_right_key(self, event):
        self._next_page()
        return "break"

    def _on_per_page_change(self, event=None):
        try:
            n = int(self.per_page_var.get())
            if n in IMAGES_PER_PAGE_CHOICES:
                self.images_per_page = n
                total_pages = (len(self.all_paths) + n - 1) // n if n else 1
                self.page_index = min(self.page_index, max(0, total_pages - 1))
                self._apply_grid_layout()
                self._refresh_page()
        except (ValueError, TypeError):
            pass

    def _thumb_path(self, img_path: str, index: int) -> str:
        return os.path.join(self._thumb_dir, f"thumb_{index:06d}.png")

    def _make_thumbnail(self, img_path: str, index: int) -> str:
        """Save thumbnail as PNG so Tk PhotoImage can display it on all platforms."""
        out = self._thumb_path(img_path, index)
        if os.path.exists(out):
            return out
        try:
            frame = cv2.imread(img_path)
            if frame is None:
                return ""
            h, w = frame.shape[:2]
            if w > THUMB_MAX_WIDTH:
                scale = THUMB_MAX_WIDTH / float(w)
                new_w = THUMB_MAX_WIDTH
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out, frame)  # PNG by default when path ends in .png
            return out
        except Exception:
            return ""

    def _on_photo_right_click(self, slot: int):
        """Right-click: zoom in (show enlarged image) or zoom back (close zoom)."""
        if slot >= self.images_per_page:
            return
        start = self.page_index * self.images_per_page
        idx = start + slot
        if idx >= len(self.all_paths):
            return
        if self._zoom_window is not None and self._zoom_window.winfo_exists():
            self._zoom_window.destroy()
            self._zoom_window = None
            self._zoom_photo_ref = None
            return
        img_path = self.all_paths[idx]
        try:
            frame = cv2.imread(img_path)
            if frame is None:
                return
            h, w = frame.shape[:2]
            if w > ZOOM_MAX_WIDTH:
                scale = ZOOM_MAX_WIDTH / float(w)
                new_w = ZOOM_MAX_WIDTH
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            zoom_path = os.path.join(self._thumb_dir, "_zoom_preview.png")
            cv2.imwrite(zoom_path, frame)
            zoom_win = tk.Toplevel(self.win)
            zoom_win.title(f"Photo #{idx + 1} — right-click to close")
            zoom_win.protocol("WM_DELETE_WINDOW", lambda: self._close_zoom(zoom_win))
            photo = tk.PhotoImage(file=zoom_path)
            self._zoom_photo_ref = photo
            lbl = tk.Label(zoom_win, image=photo, cursor="hand2")
            lbl.pack(padx=2, pady=2)
            lbl.bind("<Button-3>", lambda e: self._close_zoom(zoom_win))
            zoom_win.bind("<Button-3>", lambda e: self._close_zoom(zoom_win))
            self._zoom_window = zoom_win
        except Exception:
            pass

    def _close_zoom(self, zoom_win):
        if zoom_win.winfo_exists():
            zoom_win.destroy()
        self._zoom_window = None
        self._zoom_photo_ref = None

    def _on_photo_click(self, slot: int):
        """Toggle keep/discard when user clicks the photo (slot on current page)."""
        if slot >= self.images_per_page:
            return
        start = self.page_index * self.images_per_page
        idx = start + slot
        if idx >= len(self.all_paths):
            return
        self.keep[idx] = not self.keep[idx]
        self.check_vars[slot].set(self.keep[idx])
        self._update_cell_appearance(slot)
        self._update_count_label()

    def _update_cell_appearance(self, slot: int):
        """Set cell border color: green = keep, gray = discard."""
        if slot >= len(self.cell_frames):
            return
        cell, lbl, cb = self.cell_frames[slot]
        start = self.page_index * self.images_per_page
        idx = start + slot
        if idx < len(self.all_paths):
            cell.config(bg="light green" if self.keep[idx] else "light gray")

    def _on_toggle(self):
        start = self.page_index * self.images_per_page
        for i in range(self.images_per_page):
            idx = start + i
            if idx < len(self.all_paths):
                self.keep[idx] = self.check_vars[i].get()
        self._update_count_label()

    def _update_count_label(self):
        n = sum(1 for k in self.keep if k)
        self.count_label.config(text=f"Selected: {n} of {len(self.all_paths)} to keep")
        self._update_size_estimate()

    def _update_size_estimate(self):
        selected = [p for p, k in zip(self.all_paths, self.keep) if k]
        if not selected:
            self.size_estimate_label.config(text="Estimated PDF size: —")
            return
        optimize = self.optimize_var.get()
        est_bytes = estimate_pdf_size_bytes(selected, optimize=optimize)
        self.size_estimate_label.config(text=f"Estimated PDF size: ~{format_bytes(est_bytes)}" + (" (optimized)" if optimize else ""))

    def _show_size_info(self):
        """Show popup with would-be PDF size and optimized PDF size for current selection."""
        selected = [p for p, k in zip(self.all_paths, self.keep) if k]
        if not selected:
            messagebox.showinfo("PDF size", "Select at least one photo to see size estimates.")
            return
        size_normal = estimate_pdf_size_bytes(selected, optimize=False)
        size_optimized = estimate_pdf_size_bytes(selected, optimize=True)
        msg = (
            f"Estimated PDF size (as-is):  ~{format_bytes(size_normal)}\n\n"
            f"Estimated PDF size (optimized):  ~{format_bytes(size_optimized)}"
        )
        messagebox.showinfo("PDF size", msg)

    def _refresh_page(self):
        n = self.images_per_page
        start = self.page_index * n
        end = min(start + n, len(self.all_paths))
        total = len(self.all_paths)
        total_pages = (total + n - 1) // n if n else 1
        left_to_review = max(0, total - end)

        self.range_label.config(
            text=f"Photos {start + 1}–{end} of {total}  (page {self.page_index + 1} of {total_pages})  —  {left_to_review} left to review"
        )
        self._update_count_label()

        for i in range(MAX_IMAGES_PER_PAGE):
            idx = start + i
            cell, lbl, cb = self.cell_frames[i]
            if i < n and idx < len(self.all_paths):
                path = self.all_paths[idx]
                thumb_path = self._make_thumbnail(path, idx)
                if thumb_path and os.path.exists(thumb_path):
                    try:
                        photo = tk.PhotoImage(file=thumb_path)
                        self.photo_refs[i] = photo
                        lbl.config(image=photo, text=f"#{idx + 1}")
                    except Exception:
                        lbl.config(image="", text=f"#{idx + 1}\n(no preview)")
                else:
                    lbl.config(image="", text=f"#{idx + 1}")
                self.check_vars[i].set(self.keep[idx])
                cb.config(state="normal")
                self._update_cell_appearance(i)
            else:
                self.photo_refs[i] = None
                cell.config(bg="SystemButtonFace")
                lbl.config(image="", text="")
                self.check_vars[i].set(False)
                cb.config(state="disabled")
        self.prev_btn.config(state="normal" if self.page_index > 0 else "disabled")
        self.next_btn.config(state="normal" if end < total else "disabled")

    def _prev_page(self):
        n = self.images_per_page
        start = self.page_index * n
        for i in range(n):
            idx = start + i
            if idx < len(self.all_paths):
                self.keep[idx] = self.check_vars[i].get()
        self.page_index = max(0, self.page_index - 1)
        self._refresh_page()

    def _next_page(self):
        n = self.images_per_page
        start = self.page_index * n
        for i in range(n):
            idx = start + i
            if idx < len(self.all_paths):
                self.keep[idx] = self.check_vars[i].get()
        total_pages = (len(self.all_paths) + n - 1) // n if n else 1
        self.page_index = min(self.page_index + 1, total_pages - 1)
        self._refresh_page()

    def _create_pdf(self):
        # Sync current page checkboxes
        n = self.images_per_page
        start = self.page_index * n
        for i in range(n):
            idx = start + i
            if idx < len(self.all_paths):
                self.keep[idx] = self.check_vars[i].get()

        selected = [p for p, k in zip(self.all_paths, self.keep) if k]
        if not selected:
            messagebox.showerror("Error", "Select at least one photo to keep.")
            return

        # Title from main GUI is the default filename for the final PDF
        doc_title = self.meta.get("title", "").strip() or "Untitled"
        safe_name = filename_for_windows(doc_title) + ".pdf"
        output_pdf = filedialog.asksaveasfilename(
            title="Save PDF",
            initialfile=safe_name,
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
        )
        if not output_pdf:
            return

        try:
            created_files = build_pdf_from_selected_images(
                selected,
                output_pdf,
                self.meta,
                bar_height=BAR_HEIGHT_PTS,
                optimize=self.optimize_var.get(),
                split_max_bytes=(SPLIT_PART_MAX_BYTES if self.split_var.get() else None),
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        cleanup_dir(self.temp_dir)
        if self.split_var.get():
            messagebox.showinfo("Success", f"PDF split into 10 MB parts and saved using base name:\n{os.path.basename(output_pdf)}")
        else:
            messagebox.showinfo("Success", f"PDF created with {len(selected)} pages:\n{os.path.basename(output_pdf)}")
        self.win.destroy()
        ui_call(ui_enable_open_location, True)
        global last_output_pdf
        last_output_pdf = created_files[0] if created_files else output_pdf

    def _on_close(self):
        cleanup_dir(self.temp_dir)
        self.win.destroy()
        ui_call(ui_enable_open_location, True)


# =========================
# PDF merge helpers
# =========================
def ensure_pdf_merge_available():
    if PdfReader is None or PdfWriter is None:
        raise RuntimeError("Missing dependency: pypdf. Install with: pip install pypdf")


def append_pdf_to_writer(writer: "PdfWriter", pdf_path: str):
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        writer.add_page(page)


def meta_to_pdf_metadata(meta: dict) -> dict:
    """Build pypdf metadata dict from Readwise-style meta. All keys optional."""
    pdf_meta = {}
    if meta.get("title"):
        pdf_meta["/Title"] = meta["title"]
    if meta.get("author"):
        pdf_meta["/Author"] = meta["author"]
    if meta.get("tags"):
        pdf_meta["/Keywords"] = meta["tags"]
    if meta.get("type"):
        pdf_meta["/Type"] = meta["type"]
    if meta.get("published"):
        # PDF CreationDate format: D:YYYYMMDDHHmmSS
        s = meta["published"].strip().replace("-", "").replace("/", "")[:8]
        if len(s) == 8 and s.isdigit():
            pdf_meta["/CreationDate"] = f"D:{s}000000"
    if meta.get("language"):
        pdf_meta["/Language"] = meta["language"]
    if meta.get("cover_image"):
        pdf_meta["/CoverImage"] = meta["cover_image"]
    return pdf_meta


def add_metadata_to_pdf_file(pdf_path: str, meta: dict):
    """Read a PDF, add metadata, write back. Used for PDFs created by reportlab (e.g. from selected frames)."""
    pdf_meta = meta_to_pdf_metadata(meta)
    if not pdf_meta:
        return
    ensure_pdf_merge_available()
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata(pdf_meta)
    with open(pdf_path, "wb") as f:
        writer.write(f)


def split_pdf_file_by_size(
    pdf_path: str,
    max_size_bytes: int,
    output_dir: Optional[str] = None,
    output_base_name: Optional[str] = None,
) -> List[str]:
    """Split a PDF into sequential parts that are each <= max_size_bytes when possible."""
    ensure_pdf_merge_available()
    if max_size_bytes <= 0:
        raise ValueError("max_size_bytes must be greater than 0")

    reader = PdfReader(pdf_path)
    if not reader.pages:
        return []

    base_dir = output_dir or os.path.dirname(pdf_path)
    base_name = output_base_name or os.path.splitext(os.path.basename(pdf_path))[0]
    part_paths: List[str] = []
    part_index = 1
    writer = PdfWriter()

    def writer_size_bytes(pdf_writer: "PdfWriter") -> int:
        buffer = BytesIO()
        pdf_writer.write(buffer)
        return len(buffer.getvalue())

    def save_part(current_writer: "PdfWriter", index: int) -> str:
        part_path = os.path.join(base_dir, f"{base_name}_part{index:02d}.pdf")
        with open(part_path, "wb") as f:
            current_writer.write(f)
        return part_path

    for page in reader.pages:
        writer.add_page(page)
        if writer_size_bytes(writer) <= max_size_bytes:
            continue

        if len(writer.pages) == 1:
            part_paths.append(save_part(writer, part_index))
            part_index += 1
            writer = PdfWriter()
            continue

        overflow_page = writer.pages.pop()
        part_paths.append(save_part(writer, part_index))
        part_index += 1

        writer = PdfWriter()
        writer.add_page(overflow_page)

    if len(writer.pages) > 0:
        part_paths.append(save_part(writer, part_index))

    return part_paths


# =========================
# Queue + UI list
# =========================
queue_items: List[QueueItem] = []


def refresh_listbox(select_index: Optional[int] = None, yview_top: Optional[float] = None):
    listbox.delete(0, tk.END)
    for it in queue_items:
        tag = "[VIDEO]" if it.kind == "video" else "[PDF]"
        listbox.insert(tk.END, f"{tag} {it.path}")
    if select_index is not None and 0 <= select_index < len(queue_items):
        listbox.selection_set(select_index)
        listbox.activate(select_index)
    if yview_top is not None:
        listbox.yview_moveto(yview_top)


def update_estimate_label():
    if not queue_items:
        ui_set_estimated_size("Estimated size: --")
        return
    try:
        total_bytes, unknown_videos = estimate_output_size_bytes()
        if unknown_videos:
            ui_set_estimated_size(f"Estimated size (optimized): ~{format_bytes(total_bytes)} (some videos unknown)")
        else:
            ui_set_estimated_size(f"Estimated size (optimized): ~{format_bytes(total_bytes)}")
    except Exception:
        ui_set_estimated_size("Estimated size: --")


def add_videos():
    paths = filedialog.askopenfilenames(
        title="Select Video Files",
        filetypes=[("Video Files", " ".join(VIDEO_EXTS))],
    )
    if not paths:
        return
    for p in paths:
        queue_items.append(QueueItem("video", p))
    refresh_listbox(len(queue_items) - 1)
    update_estimate_label()


def add_pdfs():
    paths = filedialog.askopenfilenames(
        title="Select PDF Files",
        filetypes=[("PDF Files", "*.pdf")],
    )
    if not paths:
        return
    for p in paths:
        queue_items.append(QueueItem("pdf", p))
    refresh_listbox(len(queue_items) - 1)
    update_estimate_label()


def remove_selected():
    sel = listbox.curselection()
    if not sel:
        return
    idx = sel[0]
    queue_items.pop(idx)
    refresh_listbox(min(idx, len(queue_items) - 1))
    update_estimate_label()


def clear_list():
    queue_items.clear()
    refresh_listbox(None)
    update_estimate_label()


def reverse_queue():
    """Reverse queue order: top-to-bottom becomes bottom-to-top and vice versa."""
    if not queue_items:
        return
    queue_items.reverse()
    refresh_listbox(None)
    update_estimate_label()


def move_selected(delta: int):
    if listbox.cget("state") == "disabled":
        return
    sel = listbox.curselection()
    if not sel:
        return
    i = sel[0]
    j = i + delta
    if j < 0 or j >= len(queue_items):
        return
    yview_top = listbox.yview()[0]
    queue_items[i], queue_items[j] = queue_items[j], queue_items[i]
    refresh_listbox(j, yview_top=yview_top)
    update_estimate_label()


def move_selected_with_keys(delta: int):
    def _handler(event):
        if listbox.cget("state") == "disabled":
            return "break"
        if listbox.curselection():
            move_selected(delta)
            return "break"
        return None

    return _handler


# =========================
# Build combined PDF
# =========================
def build_combined_pdf():
    global conversion_thread, last_output_pdf

    if conversion_thread and conversion_thread.is_alive():
        ui_set_status("Build already running")
        return

    if not queue_items:
        ui_set_status("Add at least one video or PDF")
        return

    title = title_var.get().strip()
    author = author_var.get().strip()
    tags = tags_var.get().strip()
    type_ = type_var.get().strip()
    published = published_var.get().strip()
    language = language_var.get().strip()
    cover_image = cover_var.get().strip()

    doc_title = title or "Untitled"
    safe_name = filename_for_windows(doc_title)
    output_pdf = filedialog.asksaveasfilename(
        title="Save Combined PDF",
        initialfile=f"{safe_name}.pdf",
        defaultextension=".pdf",
        filetypes=[("PDF Files", "*.pdf")],
    )
    if not output_pdf:
        return

    preset_name = preset_var.get()
    meta = {
        "title": doc_title,
        "author": author,
        "tags": tags,
        "type": type_ or "PDF",
        "published": published,
        "language": language,
        "cover_image": cover_image,
    }

    stop_event.clear()
    last_output_pdf = None
    ui_enable_open_location(False)
    ui_set_buttons_running(True)
    ui_set_status("Starting...")
    ui_set_eta("ETA: --")
    ui_set_progress(0, 1)

    def worker():
        global last_output_pdf
        try:
            ensure_pdf_merge_available()

            temp_root = make_temp_dir_near_output(output_pdf)
            temp_pdfs: List[str] = []
            writer = PdfWriter()

            try:
                # Rough overall progress: per-item milestones + per-video internal progress.
                total_items = len(queue_items)
                overall_done = 0

                for idx, item in enumerate(queue_items, start=1):
                    if stop_event.is_set():
                        raise RuntimeError("Stopped")

                    base = f"Item {idx}/{total_items}"

                    if item.kind == "pdf":
                        ui_set_status(f"{base} - Adding PDF: {os.path.basename(item.path)}")
                        ui_set_eta("ETA: --")
                        # Progress tick per item
                        ui_set_progress(overall_done, total_items)

                        append_pdf_to_writer(writer, item.path)
                        overall_done += 1
                        ui_set_progress(overall_done, total_items)

                    else:
                        # VIDEO -> temp pdf then append
                        ui_set_status(f"{base} - Converting video: {os.path.basename(item.path)}")
                        ui_set_eta("ETA: --")

                        temp_pdf_path = os.path.join(temp_root, f"__video_{idx:03d}.pdf")
                        temp_pdfs.append(temp_pdf_path)

                        def prog_cb(step_val, step_max, eta_text, status_text):
                            # Map per-video progress into UI without breaking overall bar:
                            # show per-video as "secondary" in status/eta, overall bar tracks items.
                            ui_set_status(f"{base} - {status_text}")
                            ui_set_eta(eta_text)

                        video_to_temp_pdf(
                            video_path=item.path,
                            temp_pdf_path=temp_pdf_path,
                            preset_name=preset_name,
                            meta=meta,
                            progress_cb=prog_cb,
                        )

                        ui_set_status(f"{base} - Merging video pages: {os.path.basename(item.path)}")
                        ui_set_eta("ETA: --")
                        append_pdf_to_writer(writer, temp_pdf_path)

                        overall_done += 1
                        ui_set_progress(overall_done, total_items)

                ui_set_status("Writing final PDF...")
                ui_set_eta("ETA: --")
                pdf_meta = meta_to_pdf_metadata(meta)
                if pdf_meta:
                    writer.add_metadata(pdf_meta)
                try:
                    for page in writer.pages:
                        page.compress_content_streams()
                except Exception:
                    pass
                with open(output_pdf, "wb") as f:
                    writer.write(f)

            finally:
                cleanup_dir(temp_root)

            last_output_pdf = output_pdf
            ui_set_status("Done")
            ui_set_eta("ETA: 0:00")
            ui_set_progress(1, 1)
            ui_enable_open_location(True)
            ui_popup_done(output_pdf)

        except Exception as e:
            if str(e).lower().strip() == "stopped":
                ui_set_status("Stopped")
            else:
                ui_set_status(f"Error: {e}")
        finally:
            ui_set_buttons_running(False)

    conversion_thread = threading.Thread(target=worker, daemon=True)
    conversion_thread.start()


# =========================
# Extract Frames & Select workflow
# =========================
def extract_frames_and_select():
    global conversion_thread

    if conversion_thread and conversion_thread.is_alive():
        ui_set_status("Build already running")
        return

    video_paths = [item.path for item in queue_items if item.kind == "video"]
    if not video_paths:
        ui_set_status("Add at least one video for Extract Frames & Select")
        messagebox.showwarning("No videos", "Add at least one video to the queue for this option.")
        return

    title = title_var.get().strip()
    author = author_var.get().strip()
    tags = tags_var.get().strip()
    type_ = type_var.get().strip()
    published = published_var.get().strip()
    language = language_var.get().strip()
    cover_image = cover_var.get().strip()
    doc_title = title or "Untitled"
    meta = {
        "title": doc_title,
        "author": author,
        "tags": tags,
        "type": type_ or "PDF",
        "published": published,
        "language": language,
        "cover_image": cover_image,
    }

    preset_name = preset_var.get()
    stop_event.clear()
    ui_set_buttons_running(True)
    ui_enable_open_location(False)
    ui_set_status("Extracting frames...")
    ui_set_eta("ETA: --")
    ui_set_progress(0, 1)

    temp_dir = tempfile.mkdtemp(prefix="video_to_pdf_select_")
    estimated_total = estimate_total_frames(video_paths, preset_name)
    ui_set_status(f"Extracting frames... (est. ~{estimated_total} frames)")
    ui_set_eta("ETA: --")

    def worker():
        start_time = time.time()
        try:
            def prog_cb(step_val, step_max, _eta, status_text):
                elapsed = time.time() - start_time
                if step_val > 0 and step_max > step_val and elapsed > 0.5:
                    rate = step_val / elapsed
                    remaining_sec = (step_max - step_val) / rate
                    eta_str = format_seconds(remaining_sec)
                else:
                    eta_str = "–"
                ui_set_status(f"{status_text}  —  {step_val} of ~{step_max} frames")
                ui_set_eta(f"Preview available in ~{eta_str}")
                ui_set_progress(step_val, max(1, step_max))

            all_paths, video_collections = extract_all_frames_from_videos(
                video_paths,
                preset_name,
                temp_dir,
                progress_cb=prog_cb,
                estimated_total=estimated_total,
            )
            if not all_paths:
                ui_call(lambda: messagebox.showerror("Error", "No frames could be extracted from the videos."))
                cleanup_dir(temp_dir)
                return

            ui_set_status("Select frames to include in PDF")
            ui_set_eta("")
            ui_set_progress(1, 1)
            ui_set_buttons_running(False)
            ui_call(lambda: VideoCollectionFrameSelectionWindow(root, video_collections, temp_dir, meta))
        except RuntimeError as e:
            if str(e).lower().strip() == "stopped":
                ui_set_status("Stopped")
            else:
                ui_set_status(f"Error: {e}")
                ui_call(lambda: messagebox.showerror("Error", str(e)))
            cleanup_dir(temp_dir)
            ui_set_buttons_running(False)
        except Exception as e:
            ui_set_status(f"Error: {e}")
            ui_call(lambda: messagebox.showerror("Error", str(e)))
            cleanup_dir(temp_dir)
            ui_set_buttons_running(False)

    conversion_thread = threading.Thread(target=worker, daemon=True)
    conversion_thread.start()


def stop_build():
    stop_event.set()
    ui_set_status("Stopping...")
    ui_set_eta("ETA: --")


def shutdown():
    stop_event.set()
    root.after(200, root.destroy)


def open_file_location():
    if last_output_pdf and os.path.exists(last_output_pdf):
        subprocess.run(["explorer", "/select,", os.path.normpath(last_output_pdf)])


# =========================
# GUI
# =========================
root = tk.Tk()
root.title("Video/PDF Queue → Combined PDF")
root.geometry("820x720")

# Meta (all optional — Readwise-style fields)
meta_frame = tk.Frame(root)
meta_frame.pack(fill="x", padx=10, pady=8)

tk.Label(meta_frame, text="Title (optional)").grid(row=0, column=0, sticky="w")
title_var = tk.StringVar()
tk.Entry(meta_frame, textvariable=title_var, width=60).grid(row=0, column=1, padx=6, pady=2, sticky="w")

tk.Label(meta_frame, text="Author (optional)").grid(row=1, column=0, sticky="w")
author_var = tk.StringVar()
tk.Entry(meta_frame, textvariable=author_var, width=60).grid(row=1, column=1, padx=6, pady=2, sticky="w")

tk.Label(meta_frame, text="Document tags (optional)").grid(row=2, column=0, sticky="w")
tags_var = tk.StringVar()
tk.Entry(meta_frame, textvariable=tags_var, width=60).grid(row=2, column=1, padx=6, pady=2, sticky="w")

tk.Label(meta_frame, text="Type (optional)").grid(row=3, column=0, sticky="w")
type_var = tk.StringVar(value="PDF")
tk.Entry(meta_frame, textvariable=type_var, width=60).grid(row=3, column=1, padx=6, pady=2, sticky="w")

tk.Label(meta_frame, text="Published (optional)").grid(row=4, column=0, sticky="w")
published_var = tk.StringVar()
tk.Entry(meta_frame, textvariable=published_var, width=60).grid(row=4, column=1, padx=6, pady=2, sticky="w")
tk.Label(meta_frame, text="e.g. 2026-02-08", fg="gray").grid(row=4, column=2, sticky="w", padx=4)

tk.Label(meta_frame, text="Language (optional)").grid(row=5, column=0, sticky="w")
language_var = tk.StringVar()
tk.Entry(meta_frame, textvariable=language_var, width=60).grid(row=5, column=1, padx=6, pady=2, sticky="w")

tk.Label(meta_frame, text="Cover image (optional)").grid(row=6, column=0, sticky="w")
cover_var = tk.StringVar()
tk.Entry(meta_frame, textvariable=cover_var, width=60).grid(row=6, column=1, padx=6, pady=2, sticky="w")
tk.Label(meta_frame, text="path", fg="gray").grid(row=6, column=2, sticky="w", padx=4)

tk.Label(meta_frame, text="Compression Preset").grid(row=7, column=0, sticky="w")
preset_var = tk.StringVar(value="3 sec (Original resolution)")
preset_combo = Combobox(
    meta_frame,
    textvariable=preset_var,
    values=list(PRESETS.keys()),
    state="readonly",
    width=40,
)
preset_combo.grid(row=7, column=1, padx=6, pady=4, sticky="w")
preset_combo.bind("<<ComboboxSelected>>", lambda e: update_estimate_label())

# Queue controls
queue_frame = tk.Frame(root)
queue_frame.pack(fill="both", expand=True, padx=10, pady=8)

left = tk.Frame(queue_frame)
left.pack(side="left", fill="both", expand=True)

tk.Label(left, text="Queue (order is preserved in the final PDF)").pack(anchor="w")
listbox = tk.Listbox(left, height=16, width=95)
listbox.pack(fill="both", expand=True, pady=6)
listbox.bind("<Up>", move_selected_with_keys(-1))
listbox.bind("<Down>", move_selected_with_keys(1))

right = tk.Frame(queue_frame)
right.pack(side="left", fill="y", padx=10)

add_videos_btn = tk.Button(right, text="Add Videos", width=18, command=add_videos)
add_videos_btn.pack(pady=4)

add_pdfs_btn = tk.Button(right, text="Add PDFs", width=18, command=add_pdfs)
add_pdfs_btn.pack(pady=4)

move_up_btn = tk.Button(right, text="Move Up", width=18, command=lambda: move_selected(-1))
move_up_btn.pack(pady=10)

move_down_btn = tk.Button(right, text="Move Down", width=18, command=lambda: move_selected(1))
move_down_btn.pack(pady=4)

remove_btn = tk.Button(right, text="Remove Selected", width=18, command=remove_selected)
remove_btn.pack(pady=10)

reverse_btn = tk.Button(right, text="Reverse order", width=18, command=reverse_queue)
reverse_btn.pack(pady=4)

clear_list_btn = tk.Button(right, text="Clear List", width=18, command=clear_list)
clear_list_btn.pack(pady=4)

# Status + progress
status_frame = tk.Frame(root)
status_frame.pack(fill="x", padx=10, pady=8)

status_label = tk.Label(status_frame, text="Idle")
status_label.pack(anchor="w")

eta_label = tk.Label(status_frame, text="ETA: --")
eta_label.pack(anchor="w")

size_label = tk.Label(status_frame, text="Estimated size: --")
size_label.pack(anchor="w")

estimate_btn = tk.Button(status_frame, text="Update Estimate", command=update_estimate_label, width=16)
estimate_btn.pack(anchor="w", pady=4)

progress_bar = Progressbar(status_frame, length=780, mode="determinate")
progress_bar.pack(pady=8)

# Bottom buttons
bottom = tk.Frame(root)
bottom.pack(fill="x", padx=10, pady=10)

build_btn = tk.Button(bottom, text="Generate Combined PDF", command=build_combined_pdf, width=22)
build_btn.pack(side="left", padx=6)

extract_select_btn = tk.Button(
    bottom, text="Extract Frames & Select", command=extract_frames_and_select, width=22
)
extract_select_btn.pack(side="left", padx=6)

stop_btn = tk.Button(bottom, text="Stop", command=stop_build, width=10)
stop_btn.pack(side="left", padx=6)

open_location_btn = tk.Button(bottom, text="Open File Location", command=open_file_location, state="disabled", width=16)
open_location_btn.pack(side="left", padx=6)

exit_btn = tk.Button(bottom, text="Exit", command=shutdown, fg="red", width=10)
exit_btn.pack(side="right", padx=6)

note = (
    "Note: Videos are converted to temporary PDFs and then merged with PDFs in your queue order.\n"
    "If you see 'Missing dependency: pypdf', run:  pip install pypdf"
)
tk.Label(root, text=note, wraplength=790, justify="left").pack(padx=10, pady=6)

root.mainloop()
