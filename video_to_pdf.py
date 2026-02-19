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


@contextmanager
def suppress_ffmpeg_stderr():
    """Redirect stderr to devnull to suppress FFmpeg H.264 warnings (e.g. mmco: unref short failure)."""
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

try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    PdfReader = None
    PdfWriter = None


# =========================
# Configuration / Presets
# =========================
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
# AI Perceptual Hash Similarity
# =========================
PHASH_SIZE = 16          # 16x16 = 256-bit hash
SIMILAR_THRESHOLD = 10   # Hamming distance <= this = "similar" (out of 256 bits)

# Color coding for similarity badges
SIM_COLOR_REPR  = "#d0e8ff"  # blue:   representative frame of a group
SIM_COLOR_HIGH  = "#c8f5c8"  # green:  >=90% similar
SIM_COLOR_MED   = "#fff3c0"  # yellow: 75-89%
SIM_COLOR_LOW   = "#ffe0c0"  # orange: 50-74%
SIM_COLOR_UNIQUE= "#f0f0f0"  # gray:   unique (no similar partner)


def sim_badge_color(sim: float) -> str:
    if sim >= 90:
        return SIM_COLOR_HIGH
    elif sim >= 75:
        return SIM_COLOR_MED
    elif sim >= 50:
        return SIM_COLOR_LOW
    return SIM_COLOR_UNIQUE


def compute_phash(img_path: str) -> Optional[np.ndarray]:
    """Perceptual hash via DCT. Returns 256-bit bool array, or None on failure."""
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
    frame_paths: List[str] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)  # [100.0 for rep, sim% for rest]
    representative_path: str = ""

    @property
    def is_unique(self) -> bool:
        return len(self.frame_paths) == 1

    @property
    def avg_similarity(self) -> float:
        if len(self.similarities) <= 1:
            return 100.0
        return round(sum(self.similarities[1:]) / (len(self.similarities) - 1), 1)

    @property
    def label(self) -> str:
        n = len(self.frame_paths)
        return f"Group {self.group_id + 1}: {n} frame{'s' if n != 1 else ''}" + (
            " (unique)" if self.is_unique else f"  ~{self.avg_similarity:.0f}% sim"
        )


def cluster_frames_by_similarity(
    frame_paths: List[str],
    threshold: int = SIMILAR_THRESHOLD,
    progress_cb=None,
) -> List[SimilarityGroup]:
    """
    Greedy single-pass clustering by perceptual hash distance.
    Each frame joins the first group whose representative is within `threshold` Hamming distance,
    otherwise starts a new group.
    """
    if not frame_paths:
        return []

    total = len(frame_paths)
    hashes: List[Optional[np.ndarray]] = []

    for i, p in enumerate(frame_paths):
        hashes.append(compute_phash(p))
        if progress_cb and (i % max(1, total // 20) == 0):
            progress_cb(i, total, f"Hashing {i + 1}/{total}")

    groups: List[SimilarityGroup] = []
    rep_hashes: List[Optional[np.ndarray]] = []

    for i, p in enumerate(frame_paths):
        h = hashes[i]
        if h is None:
            g = SimilarityGroup(group_id=len(groups), frame_paths=[p],
                                similarities=[100.0], representative_path=p)
            groups.append(g)
            rep_hashes.append(None)
            continue

        best_gi = None
        best_dist = threshold + 1
        for gi, rh in enumerate(rep_hashes):
            if rh is None:
                continue
            d = hamming_distance(h, rh)
            if d <= threshold and d < best_dist:
                best_dist = d
                best_gi = gi

        if best_gi is not None:
            sim = similarity_percent(best_dist)
            groups[best_gi].frame_paths.append(p)
            groups[best_gi].similarities.append(sim)
        else:
            g = SimilarityGroup(group_id=len(groups), frame_paths=[p],
                                similarities=[100.0], representative_path=p)
            groups.append(g)
            rep_hashes.append(h)

    if progress_cb:
        progress_cb(total, total, "Done")

    return groups


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
_WIN_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5",
     "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
     "LPT6", "LPT7", "LPT8", "LPT9"}
)


def sanitize_filename(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip()


def filename_for_windows(title: str, max_length: int = 200) -> str:
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
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


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
    with suppress_ffmpeg_stderr():
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
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or 720
        if w > max_width:
            scale = max_width / float(w)
            w, h = int(w * scale), int(h * scale)
        bpp = 0.10 * (quality / 60.0)
        return frames * max(1.0, w * h * bpp)
    finally:
        cap.release()


def estimate_output_size_bytes():
    settings = PRESETS[preset_var.get()]
    ss, mw, q = int(settings["seconds"]), int(settings["width"]), int(settings["quality"])
    total, unknown = 0.0, 0
    for item in queue_items:
        if item.kind == "pdf":
            try:
                total += os.path.getsize(item.path)
            except Exception:
                pass
        else:
            est = estimate_video_pdf_bytes(item.path, ss, mw, q)
            if est is None:
                unknown += 1
            else:
                total += est
    return total * 1.05 * 0.85, unknown


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
    for btn in [add_videos_btn, add_pdfs_btn, move_up_btn, move_down_btn,
                remove_btn, clear_list_btn, reverse_btn, build_btn, estimate_btn]:
        ui_call(btn.config, state=state)
    try:
        ui_call(extract_select_btn.config, state=state)
    except NameError:
        pass
    ui_call(preset_combo.config, state=("disabled" if running else "readonly"))
    ui_call(listbox.config, state=state)


def ui_popup_done(output_pdf: str):
    ui_call(lambda: messagebox.showinfo(
        "PDF Created", f"Your PDF has been created successfully:\n\n{os.path.basename(output_pdf)}"
    ))


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
        os.rmdir(path)
    except Exception:
        pass


def resize_frame(frame, max_width: int):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# =========================
# Video -> temp PDF
# =========================
def video_to_temp_pdf(video_path, temp_pdf_path, preset_name, meta, progress_cb=None):
    s = PRESETS[preset_name]
    seconds_step, max_width, quality = int(s["seconds"]), int(s["width"]), int(s["quality"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {os.path.basename(video_path)}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (total_f / fps) if (fps > 0 and total_f > 0) else 0.0
    targets = list(range(0, int(duration_sec) + 1, seconds_step)) if duration_sec > 0 else None

    frames_dir = os.path.join(os.path.dirname(temp_pdf_path), "_frames_" + str(int(time.time())))
    os.makedirs(frames_dir, exist_ok=True)
    try:
        with suppress_ffmpeg_stderr():
            start_time = time.time()
            img_paths: List[str] = []

            if targets is not None:
                if progress_cb:
                    progress_cb(0, len(targets), "ETA: --", f"Extracting: {os.path.basename(video_path)}")
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
                    if progress_cb:
                        elapsed = time.time() - start_time
                        eta = f"ETA: {format_seconds(max(0, (len(targets) - idx) * elapsed / idx))}" if idx >= 2 else "ETA: --"
                        progress_cb(idx, len(targets), eta, f"Extracting: {os.path.basename(video_path)}")
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
                    if progress_cb:
                        progress_cb(idx, max(1, idx), "ETA: --", f"Extracting: {os.path.basename(video_path)}")

            cap.release()
        img_paths.sort()
        if not img_paths:
            raise RuntimeError(f"No frames extracted: {os.path.basename(video_path)}")

        c = canvas.Canvas(temp_pdf_path)
        c.setTitle(meta.get("title", ""))
        c.setAuthor(meta.get("author", ""))
        c.setSubject(meta.get("type", ""))
        for i, img_path in enumerate(img_paths, 1):
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
# Extract frames
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
            tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            total += int(tf / fps) // seconds_step + 1 if fps > 0 and tf > 0 else 1
        finally:
            cap.release()
    return max(1, total)


def _sanitize_folder_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "video"


def extract_all_frames_from_videos(video_paths, preset_name, output_dir, progress_cb=None, estimated_total=None):
    s = PRESETS[preset_name]
    seconds_step, max_width, quality = int(s["seconds"]), int(s["width"]), int(s["quality"])

    all_paths: List[str] = []
    video_collections: List[Tuple[str, List[str]]] = []
    step_max = estimated_total or 1

    for v_idx, vp in enumerate(video_paths):
        if stop_event.is_set():
            raise RuntimeError("Stopped")
        with suppress_ffmpeg_stderr():
            cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {os.path.basename(vp)}")

        vname = os.path.basename(vp)
        folder = f"v{v_idx:03d}_{_sanitize_folder_name(os.path.splitext(vname)[0])}"
        vdir = os.path.join(output_dir, folder)
        os.makedirs(vdir, exist_ok=True)
        vpaths: List[str] = []

        with suppress_ffmpeg_stderr():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_sec = (tf / fps) if (fps > 0 and tf > 0) else 0.0

            if duration_sec > 0:
                for tsec in range(0, int(duration_sec) + 1, seconds_step):
                    if stop_event.is_set():
                        raise RuntimeError("Stopped")
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = resize_frame(frame, max_width)
                    ip = os.path.join(vdir, f"t{tsec:06d}.jpg")
                    if cv2.imwrite(ip, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                        all_paths.append(ip)
                        vpaths.append(ip)
                    if progress_cb:
                        progress_cb(len(all_paths), step_max, "", f"Extracting: {vname}")
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
                        ip = os.path.join(vdir, f"t{int(next_t):06d}.jpg")
                        if cv2.imwrite(ip, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                            all_paths.append(ip)
                            vpaths.append(ip)
                        next_t += seconds_step
                        if progress_cb:
                            progress_cb(len(all_paths), step_max, "", f"Extracting: {vname}")
                    fc += 1

            cap.release()
        video_collections.append((vname, vpaths))

    return all_paths, video_collections


OPTIMIZE_MAX_WIDTH = 1600
OPTIMIZE_QUALITY = 82


def estimate_pdf_size_bytes(image_paths, optimize=False):
    try:
        total = sum(os.path.getsize(p) for p in image_paths if os.path.exists(p))
    except Exception:
        return 0
    if optimize:
        total = int(total * 0.45)
    return int(total * (1.15 if optimize else 1.12))


def build_pdf_from_selected_images(image_paths, output_pdf, meta, bar_height=36, progress_cb=None, optimize=False):
    from reportlab.lib.colors import black
    c = canvas.Canvas(output_pdf)
    c.setTitle(meta.get("title", ""))
    c.setAuthor(meta.get("author", ""))
    c.setSubject(meta.get("type", ""))
    tmp_opt = tempfile.mkdtemp(prefix="video_to_pdf_opt_") if optimize else None
    try:
        for i, ip in enumerate(image_paths, 1):
            if progress_cb:
                progress_cb(i, len(image_paths), f"Page {i}/{len(image_paths)}")
            if optimize and os.path.exists(ip):
                fr = cv2.imread(ip)
                if fr is not None:
                    h, w = fr.shape[:2]
                    if w > OPTIMIZE_MAX_WIDTH:
                        fr = cv2.resize(fr, (OPTIMIZE_MAX_WIDTH, int(h * OPTIMIZE_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
                    opt_p = os.path.join(tmp_opt, f"opt_{i:06d}.jpg")
                    cv2.imwrite(opt_p, fr, [cv2.IMWRITE_JPEG_QUALITY, OPTIMIZE_QUALITY])
                    ip = opt_p
            img = ImageReader(ip)
            w, h = img.getSize()
            c.setPageSize((w, h + bar_height))
            c.drawImage(img, 0, bar_height, width=w, height=h)
            c.setFillColor(black)
            c.rect(0, 0, w, bar_height, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1)
            c.setFont("Helvetica", 14)
            c.drawString(12, bar_height * 0.25, f"#{i}")
            c.showPage()
    finally:
        if tmp_opt:
            cleanup_dir(tmp_opt)
    c.save()
    try:
        add_metadata_to_pdf_file(output_pdf, meta)
    except Exception:
        pass


# =========================
# UI Constants
# =========================
MAX_IMAGES_PER_PAGE = 15
THUMB_MAX_WIDTH = 220
ZOOM_MAX_WIDTH = 900
BAR_HEIGHT_PTS = 36
IMAGES_PER_PAGE_CHOICES = (8, 12, 15)


# =========================
# VideoCollectionFrameSelectionWindow  (with AI similarity groups)
# =========================
class VideoCollectionFrameSelectionWindow:
    def __init__(self, parent, video_collections, temp_dir, meta):
        self.video_collections = video_collections
        self.temp_dir = temp_dir
        self.meta = meta

        self.keep: Dict[Tuple[int, int], bool] = {}
        for ci, (_, paths) in enumerate(video_collections):
            for fi in range(len(paths)):
                self.keep[(ci, fi)] = False

        # similarity_groups[ci] -> List[SimilarityGroup]  (filled lazily per video)
        self.similarity_groups: Dict[int, List[SimilarityGroup]] = {}
        # frame_to_group[fi] -> (group_id, similarity%)  for currently selected video
        self.frame_to_group: Dict[int, Tuple[int, float]] = {}

        self.selected_collection_idx = 0
        self.page_index = 0
        self.images_per_page = 8
        self._current_filter: Optional[int] = None   # None=all, -1=unique, >=0=group_id
        self._filtered_indices: List[int] = []

        self.win = tk.Toplevel(parent)
        self.win.title("Extract Frames & Select  —  AI Similarity Groups")
        self.win.geometry("1300x820")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._thumb_dir = os.path.join(temp_dir, "_thumbs")
        os.makedirs(self._thumb_dir, exist_ok=True)
        self._zoom_win = None
        self._zoom_ref = None

        # ── Outer paned layout ──────────────────────────────────────
        paned = tk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        # ── LEFT PANEL ──────────────────────────────────────────────
        left = tk.Frame(paned, width=280)

        tk.Label(left, text="Source Videos", font=("", 10, "bold")).pack(anchor="w", pady=(0, 2))
        vsb = tk.Scrollbar(left)
        self.vid_lb = tk.Listbox(left, height=5, width=36, font=("", 9),
                                  yscrollcommand=vsb.set, exportselection=False)
        vsb.config(command=self.vid_lb.yview)
        self.vid_lb.pack(fill="x")
        vsb.pack(side="right", fill="y")
        self.vid_lb.bind("<<ListboxSelect>>", self._on_video_select)
        for vname, vpaths in video_collections:
            self.vid_lb.insert(tk.END, f"{vname} ({len(vpaths)} frames)")

        # Divider
        tk.Frame(left, height=2, bg="#bbbbbb").pack(fill="x", pady=8)

        # AI groups header
        hdr = tk.Frame(left)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🤖 AI Similarity Groups", font=("", 10, "bold")).pack(side="left")

        tk.Label(left,
                 text="Frames clustered by visual similarity (pHash).\nClick a group → filter the frame grid.",
                 fg="gray", font=("", 8), justify="left").pack(anchor="w")

        # Sensitivity
        thr_row = tk.Frame(left)
        thr_row.pack(fill="x", pady=3)
        tk.Label(thr_row, text="Sensitivity:", font=("", 9)).pack(side="left")
        self.thr_var = tk.IntVar(value=SIMILAR_THRESHOLD)
        tk.Scale(thr_row, from_=2, to=50, orient=tk.HORIZONTAL,
                 variable=self.thr_var, length=120, showvalue=True).pack(side="left", padx=3)
        tk.Button(thr_row, text="Re-cluster", font=("", 8), command=self._rerun_clustering).pack(side="left", padx=2)

        # Legend
        leg = tk.Frame(left)
        leg.pack(anchor="w", pady=2)
        for color, label in [(SIM_COLOR_REPR,  "Rep"),
                              (SIM_COLOR_HIGH,  "≥90%"),
                              (SIM_COLOR_MED,   "75-89%"),
                              (SIM_COLOR_LOW,   "50-74%"),
                              (SIM_COLOR_UNIQUE,"Unique")]:
            tk.Label(leg, text="■", fg=color, font=("", 11)).pack(side="left")
            tk.Label(leg, text=label, font=("", 7)).pack(side="left", padx=(0, 4))

        # Groups listbox
        gsb = tk.Scrollbar(left)
        self.grp_lb = tk.Listbox(left, height=16, width=36, font=("", 9),
                                  yscrollcommand=gsb.set, exportselection=False)
        gsb.config(command=self.grp_lb.yview)
        self.grp_lb.pack(fill="both", expand=True, pady=(2, 0))
        gsb.pack(side="right", fill="y")
        self.grp_lb.bind("<<ListboxSelect>>", self._on_group_select)

        # Group action buttons
        gbr = tk.Frame(left)
        gbr.pack(fill="x", pady=4)
        tk.Button(gbr, text="Show All", font=("", 9), command=self._show_all).pack(side="left", padx=2)
        tk.Button(gbr, text="✓ Keep Group", font=("", 9), command=self._keep_group).pack(side="left", padx=2)
        tk.Button(gbr, text="✗ Skip Group", font=("", 9), command=self._skip_group).pack(side="left", padx=2)

        self.grp_info = tk.Label(left, text="", font=("", 8), fg="#555", wraplength=260, justify="left")
        self.grp_info.pack(anchor="w")

        left.pack(fill="both", expand=True)
        paned.add(left, minsize=270)

        # ── RIGHT PANEL ─────────────────────────────────────────────
        right = tk.Frame(paned)

        top = tk.Frame(right)
        top.pack(fill="x", pady=(0, 3))
        self.range_lbl = tk.Label(top, text="", font=("", 11))
        self.range_lbl.pack(anchor="w")
        self.count_lbl = tk.Label(top, text="", font=("", 10))
        self.count_lbl.pack(anchor="w")
        self.size_lbl = tk.Label(top, text="Estimated PDF size: —", font=("", 10))
        self.size_lbl.pack(anchor="w")
        self.filter_lbl = tk.Label(top, text="", font=("", 9, "italic"), fg="#005bbb")
        self.filter_lbl.pack(anchor="w")

        row0 = tk.Frame(right)
        row0.pack(fill="x", pady=2)
        tk.Label(row0, text="Per page:").pack(side="left")
        self.pp_var = tk.StringVar(value="8")
        Combobox(row0, textvariable=self.pp_var,
                 values=[str(n) for n in IMAGES_PER_PAGE_CHOICES],
                 state="readonly", width=4).pack(side="left", padx=3)
        row0.winfo_children()[-1].bind("<<ComboboxSelected>>", self._on_per_page_change)
        tk.Label(row0, text="  ← → to page").pack(side="left", padx=6)

        self.grid_frame = tk.Frame(right)
        self.grid_frame.pack(fill="both", expand=True, padx=3, pady=3)
        self.cells: List[Tuple] = []   # (cell_frame, sim_label, img_label, checkbutton)
        self.chk_vars: List[tk.BooleanVar] = []
        self.photo_refs: List[Optional[tk.PhotoImage]] = []

        for slot in range(MAX_IMAGES_PER_PAGE):
            cell = tk.Frame(self.grid_frame, relief="ridge", bd=2, bg="light gray")
            # Similarity badge row (top of cell)
            sim_lbl = tk.Label(cell, text="", font=("Helvetica", 7, "bold"),
                                bg="light gray", pady=1, padx=2)
            sim_lbl.pack(fill="x")
            img_lbl = tk.Label(cell, text="", image=None, compound="top", cursor="hand2")
            img_lbl.pack(fill="both", expand=True, padx=2)
            img_lbl.bind("<Button-1>", lambda e, s=slot: self._on_click(s))
            img_lbl.bind("<Button-3>", lambda e, s=slot: self._on_rclick(s))
            v = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(cell, text="Keep", variable=v, command=self._on_toggle)
            cb.pack(pady=1)
            self.cells.append((cell, sim_lbl, img_lbl, cb))
            self.chk_vars.append(v)
            self.photo_refs.append(None)

        bot = tk.Frame(right)
        bot.pack(fill="x", pady=6)
        self.prev_btn = tk.Button(bot, text="◀ Prev", width=10, command=self._prev_page)
        self.prev_btn.pack(side="left", padx=4)
        self.next_btn = tk.Button(bot, text="Next ▶", width=10, command=self._next_page)
        self.next_btn.pack(side="left", padx=4)
        self.opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bot, text="Optimize PDF", variable=self.opt_var,
                        command=self._update_size).pack(side="left", padx=8)
        tk.Button(bot, text="📄 Create PDF", width=14, bg="#206a34", fg="white",
                   command=self._create_pdf).pack(side="left", padx=14)
        tk.Button(bot, text="ℹ Size", width=8, command=self._size_info).pack(side="left", padx=2)

        right.pack(fill="both", expand=True)
        paned.add(right, minsize=720)

        self.win.bind("<Left>",  lambda e: (self._prev_page(), "break"))
        self.win.bind("<Right>", lambda e: (self._next_page(), "break"))

        self._apply_grid()

        self.vid_lb.selection_set(0)
        self._on_video_select(None)

    # ── Grid layout ────────────────────────────────────────────────

    def _apply_grid(self):
        n = self.images_per_page
        ncols = 4 if n <= 12 else 5
        nrows = 2 if n == 8 else 3
        for c in range(5):
            self.grid_frame.columnconfigure(c, weight=1 if c < ncols else 0)
        for r in range(3):
            self.grid_frame.rowconfigure(r, weight=1 if r < nrows else 0)
        for slot in range(MAX_IMAGES_PER_PAGE):
            cell = self.cells[slot][0]
            if slot < n:
                row = slot // (5 if n == 15 else ncols)
                col = slot % (5 if n == 15 else ncols)
                cell.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            else:
                cell.grid_remove()

    def _on_per_page_change(self, event=None):
        try:
            n = int(self.pp_var.get())
            if n in IMAGES_PER_PAGE_CHOICES:
                self.images_per_page = n
                self._apply_grid()
                self._refresh()
        except Exception:
            pass

    # ── Video selection ────────────────────────────────────────────

    def _on_video_select(self, event):
        sel = self.vid_lb.curselection()
        if sel:
            self.selected_collection_idx = sel[0]
        ci = self.selected_collection_idx
        _, paths = self.video_collections[ci]
        self._filtered_indices = list(range(len(paths)))
        self._current_filter = None
        self.filter_lbl.config(text="")
        self.grp_info.config(text="")
        self.page_index = 0
        self._start_clustering(ci)
        self._refresh()

    # ── Clustering ─────────────────────────────────────────────────

    def _start_clustering(self, ci: int):
        if ci in self.similarity_groups:
            self._rebuild_lookup(ci)
            self._populate_group_list(ci)
            return

        _, paths = self.video_collections[ci]
        if not paths:
            self.similarity_groups[ci] = []
            self.frame_to_group = {}
            self._populate_group_list(ci)
            return

        self.grp_lb.delete(0, tk.END)
        self.grp_lb.insert(tk.END, "⏳ Analysing similarity…")
        self.win.update_idletasks()

        threshold = self.thr_var.get()

        def _run():
            groups = cluster_frames_by_similarity(paths, threshold=threshold)
            self.similarity_groups[ci] = groups
            self.win.after(0, lambda: self._clustering_done(ci))

        threading.Thread(target=_run, daemon=True).start()

    def _clustering_done(self, ci: int):
        self._rebuild_lookup(ci)
        self._populate_group_list(ci)
        self._refresh()

    def _rebuild_lookup(self, ci: int):
        self.frame_to_group = {}
        _, paths = self.video_collections[ci]
        path_to_fi = {p: fi for fi, p in enumerate(paths)}
        for g in self.similarity_groups.get(ci, []):
            for p, sim in zip(g.frame_paths, g.similarities):
                fi = path_to_fi.get(p)
                if fi is not None:
                    self.frame_to_group[fi] = (g.group_id, sim)

    def _populate_group_list(self, ci: int):
        self.grp_lb.delete(0, tk.END)
        groups = self.similarity_groups.get(ci, [])
        if not groups:
            self.grp_lb.insert(tk.END, "(no frames)")
            return

        dup = [g for g in groups if not g.is_unique]
        uniq = [g for g in groups if g.is_unique]

        self.grp_lb.insert(tk.END, f"━━ {len(dup)} similar groups ━━")
        for g in dup:
            self.grp_lb.insert(tk.END, f"  Grp {g.group_id + 1}:  {len(g.frame_paths)} frames  ~{g.avg_similarity:.0f}% sim")

        self.grp_lb.insert(tk.END, f"━━ {len(uniq)} unique frames ━━")
        self.grp_lb.insert(tk.END, "  [All unique frames]")

    def _rerun_clustering(self):
        ci = self.selected_collection_idx
        self.similarity_groups.pop(ci, None)
        self._current_filter = None
        self._start_clustering(ci)

    # ── Group selection → filter ───────────────────────────────────

    def _on_group_select(self, event):
        sel = self.grp_lb.curselection()
        if not sel:
            return
        idx = sel[0]
        ci = self.selected_collection_idx
        groups = self.similarity_groups.get(ci, [])
        dup = [g for g in groups if not g.is_unique]
        uniq = [g for g in groups if g.is_unique]
        _, paths = self.video_collections[ci]

        # index layout: 0=dup-header, 1..len(dup)=dup groups,
        #               len(dup)+1=uniq-header, len(dup)+2=all-unique
        if idx == 0 or idx == len(dup) + 1:
            return  # header row

        if 1 <= idx <= len(dup):
            g = dup[idx - 1]
            path_set = set(g.frame_paths)
            self._filtered_indices = [fi for fi, p in enumerate(paths) if p in path_set]
            self._current_filter = g.group_id
            self.filter_lbl.config(text=f"🔍 Group {g.group_id + 1}: {len(g.frame_paths)} frames  (~{g.avg_similarity:.0f}% similarity)")
            self.grp_info.config(text=f"Group {g.group_id + 1}: {len(g.frame_paths)} similar frames.\nFrame #1 in group = representative (blue badge).")
        elif idx == len(dup) + 2:
            path_set = set(p for g in uniq for p in g.frame_paths)
            self._filtered_indices = [fi for fi, p in enumerate(paths) if p in path_set]
            self._current_filter = -1
            self.filter_lbl.config(text=f"🔍 Showing {len(self._filtered_indices)} unique (non-duplicated) frames")
            self.grp_info.config(text="These frames have no similar partners.")
        else:
            return

        self.page_index = 0
        self._refresh()

    def _show_all(self):
        ci = self.selected_collection_idx
        _, paths = self.video_collections[ci]
        self._filtered_indices = list(range(len(paths)))
        self._current_filter = None
        self.filter_lbl.config(text="")
        self.grp_info.config(text="")
        self.page_index = 0
        self._refresh()

    def _keep_group(self):
        if not self._filtered_indices:
            return
        ci = self.selected_collection_idx
        for fi in self._filtered_indices:
            self.keep[(ci, fi)] = True
        self._refresh()
        self._update_count()

    def _skip_group(self):
        if not self._filtered_indices:
            return
        ci = self.selected_collection_idx
        for fi in self._filtered_indices:
            self.keep[(ci, fi)] = False
        self._refresh()
        self._update_count()

    # ── Thumbnail helpers ──────────────────────────────────────────

    def _make_thumb(self, img_path: str, ci: int, fi: int) -> str:
        out = os.path.join(self._thumb_dir, f"v{ci}_f{fi}.png")
        if os.path.exists(out):
            return out
        try:
            fr = cv2.imread(img_path)
            if fr is None:
                return ""
            h, w = fr.shape[:2]
            if w > THUMB_MAX_WIDTH:
                fr = cv2.resize(fr, (THUMB_MAX_WIDTH, int(h * THUMB_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out, fr)
            return out
        except Exception:
            return ""

    # ── Page refresh ───────────────────────────────────────────────

    def _refresh(self):
        ci = self.selected_collection_idx
        _, all_paths = self.video_collections[ci]
        disp = self._filtered_indices
        n = self.images_per_page
        total_d = len(disp)
        start = self.page_index * n
        end = min(start + n, total_d)
        total_p = max(1, (total_d + n - 1) // n)
        vname = self.video_collections[ci][0]
        filt = "" if self._current_filter is None else " [filtered]"
        self.range_lbl.config(text=f"{vname}  —  {start + 1}–{end} of {total_d}{filt}  (p.{self.page_index + 1}/{total_p})")
        self._update_count()

        for slot in range(MAX_IMAGES_PER_PAGE):
            dp = start + slot
            cell, sim_lbl, img_lbl, cb = self.cells[slot]

            if slot < n and dp < total_d:
                fi = disp[dp]
                path = all_paths[fi]
                thumb = self._make_thumb(path, ci, fi)
                if thumb and os.path.exists(thumb):
                    try:
                        ph = tk.PhotoImage(file=thumb)
                        self.photo_refs[slot] = ph
                        img_lbl.config(image=ph, text="")
                    except Exception:
                        img_lbl.config(image="", text=f"#{fi + 1}")
                else:
                    img_lbl.config(image="", text=f"#{fi + 1}")

                # Similarity badge
                ginfo = self.frame_to_group.get(fi)
                if ginfo is not None:
                    gid, sim = ginfo
                    if sim >= 99.9:  # representative
                        badge = f"#{fi + 1}  [Rep G{gid + 1}]"
                        bg = SIM_COLOR_REPR
                    else:
                        badge = f"#{fi + 1}  {sim:.0f}% sim"
                        bg = sim_badge_color(sim)
                else:
                    badge = f"#{fi + 1}"
                    bg = "light gray"

                sim_lbl.config(text=badge, bg=bg)
                self.chk_vars[slot].set(self.keep.get((ci, fi), False))
                cb.config(state="normal")
                self._cell_color(slot, fi, ci, bg)
            else:
                self.photo_refs[slot] = None
                for w in (cell, sim_lbl):
                    w.config(bg="#f5f5f5")
                img_lbl.config(image="", text="", bg="#f5f5f5")
                self.chk_vars[slot].set(False)
                cb.config(state="disabled")

        self.prev_btn.config(state="normal" if self.page_index > 0 else "disabled")
        self.next_btn.config(state="normal" if end < total_d else "disabled")

    def _cell_color(self, slot: int, fi: int, ci: int, default_bg: str):
        cell, sim_lbl, img_lbl, _ = self.cells[slot]
        if self.keep.get((ci, fi), False):
            bg = "light green"
        else:
            bg = default_bg
        cell.config(bg=bg)
        sim_lbl.config(bg=bg)

    # ── Click / toggle ─────────────────────────────────────────────

    def _on_click(self, slot: int):
        ci = self.selected_collection_idx
        disp = self._filtered_indices
        dp = self.page_index * self.images_per_page + slot
        if dp >= len(disp):
            return
        fi = disp[dp]
        self.keep[(ci, fi)] = not self.keep.get((ci, fi), False)
        self.chk_vars[slot].set(self.keep[(ci, fi)])
        ginfo = self.frame_to_group.get(fi)
        bg = (sim_badge_color(ginfo[1]) if ginfo and ginfo[1] < 99.9 else
              (SIM_COLOR_REPR if ginfo else "light gray"))
        self._cell_color(slot, fi, ci, bg)
        self._update_count()

    def _on_toggle(self):
        ci = self.selected_collection_idx
        disp = self._filtered_indices
        start = self.page_index * self.images_per_page
        for slot in range(self.images_per_page):
            dp = start + slot
            if dp < len(disp):
                fi = disp[dp]
                self.keep[(ci, fi)] = self.chk_vars[slot].get()
        self._update_count()

    def _update_count(self):
        total_sel = sum(1 for v in self.keep.values() if v)
        total_all = sum(len(p) for _, p in self.video_collections)
        self.count_lbl.config(text=f"Selected: {total_sel} of {total_all} frames to keep")
        self._update_size()

    def _update_size(self):
        sel = self._selected_paths()
        if not sel:
            self.size_lbl.config(text="Estimated PDF size: —")
            return
        opt = self.opt_var.get()
        est = estimate_pdf_size_bytes(sel, optimize=opt)
        self.size_lbl.config(text=f"Estimated PDF size: ~{format_bytes(est)}" + (" (opt)" if opt else ""))

    def _selected_paths(self) -> List[str]:
        out = []
        for ci, (_, paths) in enumerate(self.video_collections):
            for fi in range(len(paths)):
                if self.keep.get((ci, fi), False):
                    out.append(paths[fi])
        return out

    # ── Pagination ─────────────────────────────────────────────────

    def _sync_page(self):
        ci = self.selected_collection_idx
        disp = self._filtered_indices
        start = self.page_index * self.images_per_page
        for slot in range(self.images_per_page):
            dp = start + slot
            if dp < len(disp):
                self.keep[(ci, disp[dp])] = self.chk_vars[slot].get()

    def _prev_page(self):
        self._sync_page()
        self.page_index = max(0, self.page_index - 1)
        self._refresh()

    def _next_page(self):
        self._sync_page()
        n = self.images_per_page
        total_p = max(1, (len(self._filtered_indices) + n - 1) // n)
        self.page_index = min(self.page_index + 1, total_p - 1)
        self._refresh()

    # ── Zoom ───────────────────────────────────────────────────────

    def _on_rclick(self, slot: int):
        ci = self.selected_collection_idx
        _, paths = self.video_collections[ci]
        disp = self._filtered_indices
        dp = self.page_index * self.images_per_page + slot
        if dp >= len(disp):
            return
        if self._zoom_win and self._zoom_win.winfo_exists():
            self._zoom_win.destroy()
            self._zoom_win = None
            return
        fi = disp[dp]
        path = paths[fi]
        try:
            fr = cv2.imread(path)
            if fr is None:
                return
            h, w = fr.shape[:2]
            if w > ZOOM_MAX_WIDTH:
                fr = cv2.resize(fr, (ZOOM_MAX_WIDTH, int(h * ZOOM_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
            zp = os.path.join(self._thumb_dir, "_zoom.png")
            cv2.imwrite(zp, fr)
            ginfo = self.frame_to_group.get(fi)
            extra = f"Group {ginfo[0] + 1}, {ginfo[1]:.0f}% sim" if ginfo and ginfo[1] < 99.9 else (
                f"Group {ginfo[0] + 1} Representative" if ginfo else "no data yet")
            zw = tk.Toplevel(self.win)
            zw.title(f"Frame #{fi + 1}  —  {extra}  —  right-click to close")
            zw.protocol("WM_DELETE_WINDOW", lambda: self._close_zoom(zw))
            ph = tk.PhotoImage(file=zp)
            self._zoom_ref = ph
            lbl = tk.Label(zw, image=ph, cursor="hand2")
            lbl.pack(padx=2, pady=2)
            lbl.bind("<Button-3>", lambda e: self._close_zoom(zw))
            zw.bind("<Button-3>", lambda e: self._close_zoom(zw))
            self._zoom_win = zw
        except Exception:
            pass

    def _close_zoom(self, zw):
        if zw.winfo_exists():
            zw.destroy()
        self._zoom_win = None
        self._zoom_ref = None

    # ── Info / Create PDF ──────────────────────────────────────────

    def _size_info(self):
        sel = self._selected_paths()
        if not sel:
            messagebox.showinfo("Info", "Select at least one frame first.")
            return
        norm = estimate_pdf_size_bytes(sel, optimize=False)
        opt  = estimate_pdf_size_bytes(sel, optimize=True)
        messagebox.showinfo("PDF Size Info",
                            f"Selection: {len(sel)} frames\n\n"
                            f"Normal:    ~{format_bytes(norm)}\n"
                            f"Optimized: ~{format_bytes(opt)}")

    def _create_pdf(self):
        self._sync_page()
        sel = self._selected_paths()
        if not sel:
            messagebox.showerror("Error", "Select at least one frame to keep.")
            return
        doc_title = self.meta.get("title", "").strip() or "Untitled"
        out = filedialog.asksaveasfilename(
            title="Save PDF",
            initialfile=filename_for_windows(doc_title) + ".pdf",
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
        )
        if not out:
            return

        def run():
            global last_output_pdf
            try:
                build_pdf_from_selected_images(sel, out, self.meta,
                                                bar_height=BAR_HEIGHT_PTS,
                                                optimize=self.opt_var.get())
                last_output_pdf = out
                self.win.after(0, lambda: messagebox.showinfo("Done", f"Saved: {os.path.basename(out)}"))
                try:
                    ui_call(ui_enable_open_location, True)
                except Exception:
                    pass
            except Exception as e:
                self.win.after(0, lambda: messagebox.showerror("Error", str(e)))

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
# Simple flat FrameSelectionWindow (kept for compatibility)
# =========================
class FrameSelectionWindow:
    def __init__(self, parent, all_paths, temp_dir, meta):
        self.all_paths = all_paths
        self.temp_dir = temp_dir
        self.meta = meta
        self.keep = [False] * len(all_paths)
        self.page_index = 0
        self.images_per_page = 8
        self.win = tk.Toplevel(parent)
        self.win.title("Select Frames for PDF")
        self.win.geometry("1000x720")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self._thumb_dir = os.path.join(temp_dir, "_thumbs")
        os.makedirs(self._thumb_dir, exist_ok=True)
        self._zoom_win = None
        self._zoom_ref = None
        self.win.bind("<Left>", lambda e: self._prev_page() or "break")
        self.win.bind("<Right>", lambda e: self._next_page() or "break")

        top = tk.Frame(self.win)
        top.pack(fill="x", padx=10, pady=8)
        self.range_lbl = tk.Label(top, text="", font=("", 11))
        self.range_lbl.pack(anchor="w")
        self.count_lbl = tk.Label(top, text="", font=("", 10))
        self.count_lbl.pack(anchor="w")
        self.size_lbl = tk.Label(top, text="—", font=("", 10))
        self.size_lbl.pack(anchor="w")

        row0 = tk.Frame(self.win)
        row0.pack(fill="x", padx=10)
        tk.Label(row0, text="Per page:").pack(side="left")
        self.pp_var = tk.StringVar(value="8")
        pp = Combobox(row0, textvariable=self.pp_var,
                      values=[str(n) for n in IMAGES_PER_PAGE_CHOICES],
                      state="readonly", width=4)
        pp.pack(side="left", padx=3)
        pp.bind("<<ComboboxSelected>>", self._on_pp_change)

        self.grid_frame = tk.Frame(self.win)
        self.grid_frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.cells = []
        self.chk_vars = []
        self.photo_refs = []
        for slot in range(MAX_IMAGES_PER_PAGE):
            cell = tk.Frame(self.grid_frame, relief="ridge", bd=2, bg="light gray")
            lbl = tk.Label(cell, text="", cursor="hand2")
            lbl.pack(fill="both", expand=True, padx=2, pady=2)
            lbl.bind("<Button-1>", lambda e, s=slot: self._click(s))
            lbl.bind("<Button-3>", lambda e, s=slot: self._rclick(s))
            v = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(cell, text="Keep", variable=v, command=self._toggle)
            cb.pack(pady=2)
            self.cells.append((cell, lbl, cb))
            self.chk_vars.append(v)
            self.photo_refs.append(None)

        bot = tk.Frame(self.win)
        bot.pack(fill="x", padx=10, pady=10)
        self.prev_btn = tk.Button(bot, text="Previous", width=12, command=self._prev_page)
        self.prev_btn.pack(side="left", padx=6)
        self.next_btn = tk.Button(bot, text="Next", width=12, command=self._next_page)
        self.next_btn.pack(side="left", padx=6)
        self.opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bot, text="Optimize PDF", variable=self.opt_var,
                        command=self._update_size).pack(side="left", padx=8)
        tk.Button(bot, text="Create PDF", width=14, command=self._create_pdf).pack(side="left", padx=16)

        self._apply_grid()
        self._refresh()

    def _apply_grid(self):
        n = self.images_per_page
        ncols = 4 if n <= 12 else 5
        nrows = 2 if n == 8 else 3
        for c in range(5):
            self.grid_frame.columnconfigure(c, weight=1 if c < ncols else 0)
        for r in range(3):
            self.grid_frame.rowconfigure(r, weight=1 if r < nrows else 0)
        for slot in range(MAX_IMAGES_PER_PAGE):
            cell, _, _ = self.cells[slot]
            if slot < n:
                row = slot // (5 if n == 15 else ncols)
                col = slot % (5 if n == 15 else ncols)
                cell.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            else:
                cell.grid_remove()

    def _on_pp_change(self, event=None):
        try:
            n = int(self.pp_var.get())
            if n in IMAGES_PER_PAGE_CHOICES:
                self.images_per_page = n
                self._apply_grid()
                self._refresh()
        except Exception:
            pass

    def _thumb(self, path, idx):
        out = os.path.join(self._thumb_dir, f"thumb_{idx:06d}.png")
        if os.path.exists(out):
            return out
        try:
            fr = cv2.imread(path)
            if fr is None:
                return ""
            h, w = fr.shape[:2]
            if w > THUMB_MAX_WIDTH:
                fr = cv2.resize(fr, (THUMB_MAX_WIDTH, int(h * THUMB_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out, fr)
            return out
        except Exception:
            return ""

    def _click(self, slot):
        start = self.page_index * self.images_per_page
        idx = start + slot
        if idx >= len(self.all_paths):
            return
        self.keep[idx] = not self.keep[idx]
        self.chk_vars[slot].set(self.keep[idx])
        cell, _, _ = self.cells[slot]
        cell.config(bg="light green" if self.keep[idx] else "light gray")
        self._update_count()

    def _rclick(self, slot):
        start = self.page_index * self.images_per_page
        idx = start + slot
        if idx >= len(self.all_paths):
            return
        if self._zoom_win and self._zoom_win.winfo_exists():
            self._zoom_win.destroy()
            self._zoom_win = None
            return
        try:
            fr = cv2.imread(self.all_paths[idx])
            if fr is None:
                return
            h, w = fr.shape[:2]
            if w > ZOOM_MAX_WIDTH:
                fr = cv2.resize(fr, (ZOOM_MAX_WIDTH, int(h * ZOOM_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
            zp = os.path.join(self._thumb_dir, "_zoom.png")
            cv2.imwrite(zp, fr)
            zw = tk.Toplevel(self.win)
            zw.title(f"#{idx + 1} — right-click to close")
            ph = tk.PhotoImage(file=zp)
            self._zoom_ref = ph
            lbl = tk.Label(zw, image=ph, cursor="hand2")
            lbl.pack()
            lbl.bind("<Button-3>", lambda e: zw.destroy())
            self._zoom_win = zw
        except Exception:
            pass

    def _toggle(self):
        start = self.page_index * self.images_per_page
        for i in range(self.images_per_page):
            idx = start + i
            if idx < len(self.all_paths):
                self.keep[idx] = self.chk_vars[i].get()
        self._update_count()

    def _update_count(self):
        n = sum(self.keep)
        self.count_lbl.config(text=f"Selected: {n} of {len(self.all_paths)}")
        self._update_size()

    def _update_size(self):
        sel = [p for p, k in zip(self.all_paths, self.keep) if k]
        if not sel:
            self.size_lbl.config(text="—")
            return
        est = estimate_pdf_size_bytes(sel, optimize=self.opt_var.get())
        self.size_lbl.config(text=f"~{format_bytes(est)}")

    def _refresh(self):
        n = self.images_per_page
        start = self.page_index * n
        end = min(start + n, len(self.all_paths))
        total = len(self.all_paths)
        total_p = max(1, (total + n - 1) // n)
        self.range_lbl.config(text=f"{start + 1}–{end} of {total}  (p.{self.page_index + 1}/{total_p})")
        self._update_count()
        for i in range(MAX_IMAGES_PER_PAGE):
            idx = start + i
            cell, lbl, cb = self.cells[i]
            if i < n and idx < len(self.all_paths):
                t = self._thumb(self.all_paths[idx], idx)
                if t and os.path.exists(t):
                    try:
                        ph = tk.PhotoImage(file=t)
                        self.photo_refs[i] = ph
                        lbl.config(image=ph, text=f"#{idx + 1}")
                    except Exception:
                        lbl.config(image="", text=f"#{idx + 1}")
                else:
                    lbl.config(image="", text=f"#{idx + 1}")
                self.chk_vars[i].set(self.keep[idx])
                cb.config(state="normal")
                cell.config(bg="light green" if self.keep[idx] else "light gray")
            else:
                self.photo_refs[i] = None
                cell.config(bg="#eeeeee")
                lbl.config(image="", text="")
                self.chk_vars[i].set(False)
                cb.config(state="disabled")
        self.prev_btn.config(state="normal" if self.page_index > 0 else "disabled")
        self.next_btn.config(state="normal" if end < len(self.all_paths) else "disabled")

    def _sync(self):
        start = self.page_index * self.images_per_page
        for i in range(self.images_per_page):
            idx = start + i
            if idx < len(self.all_paths):
                self.keep[idx] = self.chk_vars[i].get()

    def _prev_page(self):
        self._sync()
        self.page_index = max(0, self.page_index - 1)
        self._refresh()

    def _next_page(self):
        self._sync()
        total_p = max(1, (len(self.all_paths) + self.images_per_page - 1) // self.images_per_page)
        self.page_index = min(self.page_index + 1, total_p - 1)
        self._refresh()

    def _create_pdf(self):
        self._sync()
        sel = [p for p, k in zip(self.all_paths, self.keep) if k]
        if not sel:
            messagebox.showerror("Error", "Select at least one frame.")
            return
        doc_title = self.meta.get("title", "").strip() or "Untitled"
        out = filedialog.asksaveasfilename(
            title="Save PDF",
            initialfile=filename_for_windows(doc_title) + ".pdf",
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
        )
        if not out:
            return
        try:
            build_pdf_from_selected_images(sel, out, self.meta, bar_height=BAR_HEIGHT_PTS,
                                            optimize=self.opt_var.get())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        cleanup_dir(self.temp_dir)
        messagebox.showinfo("Done", f"{len(sel)} pages saved to {os.path.basename(out)}")
        self.win.destroy()
        ui_call(ui_enable_open_location, True)
        global last_output_pdf
        last_output_pdf = out

    def _on_close(self):
        cleanup_dir(self.temp_dir)
        self.win.destroy()
        ui_call(ui_enable_open_location, True)


# =========================
# PDF merge helpers
# =========================
def ensure_pdf_merge_available():
    if PdfReader is None or PdfWriter is None:
        raise RuntimeError("Missing dependency: pypdf. Install: pip install pypdf")


def append_pdf_to_writer(writer, pdf_path):
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        writer.add_page(page)


def meta_to_pdf_metadata(meta):
    pm = {}
    if meta.get("title"):   pm["/Title"] = meta["title"]
    if meta.get("author"):  pm["/Author"] = meta["author"]
    if meta.get("tags"):    pm["/Keywords"] = meta["tags"]
    if meta.get("type"):    pm["/Type"] = meta["type"]
    if meta.get("published"):
        s = meta["published"].strip().replace("-", "").replace("/", "")[:8]
        if len(s) == 8 and s.isdigit():
            pm["/CreationDate"] = f"D:{s}000000"
    if meta.get("language"):    pm["/Language"] = meta["language"]
    if meta.get("cover_image"): pm["/CoverImage"] = meta["cover_image"]
    return pm


def add_metadata_to_pdf_file(pdf_path, meta):
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
# Queue + UI
# =========================
queue_items: List[QueueItem] = []


def refresh_listbox(select_index=None, yview_top=None):
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
        total_bytes, unknown = estimate_output_size_bytes()
        suffix = " (some videos unknown)" if unknown else ""
        ui_set_estimated_size(f"Estimated size (optimized): ~{format_bytes(total_bytes)}{suffix}")
    except Exception:
        ui_set_estimated_size("Estimated size: --")


def add_videos():
    paths = filedialog.askopenfilenames(title="Select Video Files",
                                         filetypes=[("Video Files", " ".join(VIDEO_EXTS))])
    for p in paths:
        queue_items.append(QueueItem("video", p))
    if paths:
        refresh_listbox(len(queue_items) - 1)
        update_estimate_label()


def add_pdfs():
    paths = filedialog.askopenfilenames(title="Select PDF Files",
                                         filetypes=[("PDF Files", "*.pdf")])
    for p in paths:
        queue_items.append(QueueItem("pdf", p))
    if paths:
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
    if queue_items:
        queue_items.reverse()
        refresh_listbox(None)
        update_estimate_label()


def move_selected(delta):
    if listbox.cget("state") == "disabled":
        return
    sel = listbox.curselection()
    if not sel:
        return
    i = sel[0]
    j = i + delta
    if 0 <= j < len(queue_items):
        yv = listbox.yview()[0]
        queue_items[i], queue_items[j] = queue_items[j], queue_items[i]
        refresh_listbox(j, yview_top=yv)
        update_estimate_label()


def move_selected_with_keys(delta):
    def handler(event):
        if listbox.cget("state") == "disabled":
            return "break"
        if listbox.curselection():
            move_selected(delta)
            return "break"
    return handler


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

    meta = {
        "title": title_var.get().strip() or "Untitled",
        "author": author_var.get().strip(),
        "tags": tags_var.get().strip(),
        "type": type_var.get().strip() or "PDF",
        "published": published_var.get().strip(),
        "language": language_var.get().strip(),
        "cover_image": cover_var.get().strip(),
    }
    out = filedialog.asksaveasfilename(
        title="Save Combined PDF",
        initialfile=filename_for_windows(meta["title"]) + ".pdf",
        defaultextension=".pdf",
        filetypes=[("PDF Files", "*.pdf")],
    )
    if not out:
        return

    preset_name = preset_var.get()
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
            temp_root = make_temp_dir_near_output(out)
            writer = PdfWriter()
            try:
                total = len(queue_items)
                done = 0
                for idx, item in enumerate(queue_items, 1):
                    if stop_event.is_set():
                        raise RuntimeError("Stopped")
                    base = f"Item {idx}/{total}"
                    if item.kind == "pdf":
                        ui_set_status(f"{base} - Adding: {os.path.basename(item.path)}")
                        ui_set_progress(done, total)
                        append_pdf_to_writer(writer, item.path)
                        done += 1
                        ui_set_progress(done, total)
                    else:
                        ui_set_status(f"{base} - Converting: {os.path.basename(item.path)}")
                        tmp = os.path.join(temp_root, f"__video_{idx:03d}.pdf")

                        def prog(sv, sm, et, st):
                            ui_set_status(f"{base} - {st}")
                            ui_set_eta(et)

                        video_to_temp_pdf(item.path, tmp, preset_name, meta, prog)
                        append_pdf_to_writer(writer, tmp)
                        done += 1
                        ui_set_progress(done, total)

                ui_set_status("Writing final PDF…")
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
            ui_set_status("Done")
            ui_set_eta("ETA: 0:00")
            ui_set_progress(1, 1)
            ui_enable_open_location(True)
            ui_popup_done(out)
        except Exception as e:
            ui_set_status("Stopped" if str(e).lower().strip() == "stopped" else f"Error: {e}")
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
        ui_set_status("Build already running")
        return
    video_paths = [item.path for item in queue_items if item.kind == "video"]
    if not video_paths:
        messagebox.showwarning("No videos", "Add at least one video to the queue.")
        return

    meta = {
        "title": title_var.get().strip() or "Untitled",
        "author": author_var.get().strip(),
        "tags": tags_var.get().strip(),
        "type": type_var.get().strip() or "PDF",
        "published": published_var.get().strip(),
        "language": language_var.get().strip(),
        "cover_image": cover_var.get().strip(),
    }
    preset_name = preset_var.get()
    stop_event.clear()
    ui_set_buttons_running(True)
    ui_enable_open_location(False)

    temp_dir = tempfile.mkdtemp(prefix="video_to_pdf_select_")
    est = estimate_total_frames(video_paths, preset_name)
    ui_set_status(f"Extracting frames… (est. ~{est})")
    ui_set_eta("ETA: --")
    ui_set_progress(0, 1)

    def worker():
        t0 = time.time()
        try:
            def prog(sv, sm, _eta, st):
                el = time.time() - t0
                eta = format_seconds((sm - sv) / (sv / el)) if sv > 0 and el > 0.5 and sm > sv else "–"
                ui_set_status(f"{st}  —  {sv} of ~{sm}")
                ui_set_eta(f"ETA ~{eta}")
                ui_set_progress(sv, max(1, sm))

            all_paths, vc = extract_all_frames_from_videos(
                video_paths, preset_name, temp_dir, progress_cb=prog, estimated_total=est)
            if not all_paths:
                ui_call(lambda: messagebox.showerror("Error", "No frames extracted."))
                cleanup_dir(temp_dir)
                return
            ui_set_status("Opening selector with AI similarity analysis…")
            ui_set_eta("")
            ui_set_progress(1, 1)
            ui_set_buttons_running(False)
            ui_call(lambda: VideoCollectionFrameSelectionWindow(root, vc, temp_dir, meta))
        except RuntimeError as e:
            ui_set_status("Stopped" if str(e).lower().strip() == "stopped" else f"Error: {e}")
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
    ui_set_status("Stopping…")


def shutdown():
    stop_event.set()
    root.after(200, root.destroy)


def open_file_location():
    if last_output_pdf and os.path.exists(last_output_pdf):
        subprocess.run(["explorer", "/select,", os.path.normpath(last_output_pdf)])


# =========================
# Main GUI
# =========================
root = tk.Tk()
root.title("Video/PDF → Combined PDF  [AI Similarity Grouping]")
root.geometry("820x730")

mf = tk.Frame(root)
mf.pack(fill="x", padx=10, pady=8)

fields = [
    ("Title (optional)", "title_var", ""),
    ("Author (optional)", "author_var", ""),
    ("Document tags (optional)", "tags_var", ""),
    ("Type (optional)", "type_var", "PDF"),
    ("Published (optional)", "published_var", ""),
    ("Language (optional)", "language_var", ""),
    ("Cover image (optional)", "cover_var", ""),
]
vars_dict = {}
for row_idx, (label, var_name, default) in enumerate(fields):
    tk.Label(mf, text=label).grid(row=row_idx, column=0, sticky="w")
    v = tk.StringVar(value=default)
    vars_dict[var_name] = v
    tk.Entry(mf, textvariable=v, width=60).grid(row=row_idx, column=1, padx=6, pady=2, sticky="w")
    if var_name == "published_var":
        tk.Label(mf, text="e.g. 2026-02-08", fg="gray").grid(row=row_idx, column=2, sticky="w", padx=4)

title_var    = vars_dict["title_var"]
author_var   = vars_dict["author_var"]
tags_var     = vars_dict["tags_var"]
type_var     = vars_dict["type_var"]
published_var= vars_dict["published_var"]
language_var = vars_dict["language_var"]
cover_var    = vars_dict["cover_var"]

row_idx += 1
tk.Label(mf, text="Compression Preset").grid(row=row_idx, column=0, sticky="w")
preset_var = tk.StringVar(value="3 sec (Original resolution)")
preset_combo = Combobox(mf, textvariable=preset_var, values=list(PRESETS.keys()),
                         state="readonly", width=40)
preset_combo.grid(row=row_idx, column=1, padx=6, pady=4, sticky="w")
preset_combo.bind("<<ComboboxSelected>>", lambda e: update_estimate_label())

# Queue
qf = tk.Frame(root)
qf.pack(fill="both", expand=True, padx=10, pady=8)

lf = tk.Frame(qf)
lf.pack(side="left", fill="both", expand=True)
tk.Label(lf, text="Queue (order preserved in final PDF)").pack(anchor="w")
listbox = tk.Listbox(lf, height=14, width=95)
listbox.pack(fill="both", expand=True, pady=6)
listbox.bind("<Up>", move_selected_with_keys(-1))
listbox.bind("<Down>", move_selected_with_keys(1))

rf = tk.Frame(qf)
rf.pack(side="left", fill="y", padx=10)
add_videos_btn = tk.Button(rf, text="Add Videos",      width=18, command=add_videos)
add_videos_btn.pack(pady=4)
add_pdfs_btn   = tk.Button(rf, text="Add PDFs",        width=18, command=add_pdfs)
add_pdfs_btn.pack(pady=4)
move_up_btn    = tk.Button(rf, text="Move Up",         width=18, command=lambda: move_selected(-1))
move_up_btn.pack(pady=8)
move_down_btn  = tk.Button(rf, text="Move Down",       width=18, command=lambda: move_selected(1))
move_down_btn.pack(pady=4)
remove_btn     = tk.Button(rf, text="Remove Selected", width=18, command=remove_selected)
remove_btn.pack(pady=8)
reverse_btn    = tk.Button(rf, text="Reverse Order",   width=18, command=reverse_queue)
reverse_btn.pack(pady=4)
clear_list_btn = tk.Button(rf, text="Clear List",      width=18, command=clear_list)
clear_list_btn.pack(pady=4)

# Status
sf = tk.Frame(root)
sf.pack(fill="x", padx=10, pady=6)
status_label = tk.Label(sf, text="Idle")
status_label.pack(anchor="w")
eta_label = tk.Label(sf, text="ETA: --")
eta_label.pack(anchor="w")
size_label = tk.Label(sf, text="Estimated size: --")
size_label.pack(anchor="w")
estimate_btn = tk.Button(sf, text="Update Estimate", command=update_estimate_label, width=16)
estimate_btn.pack(anchor="w", pady=3)
progress_bar = Progressbar(sf, length=780, mode="determinate")
progress_bar.pack(pady=6)

# Bottom
bf = tk.Frame(root)
bf.pack(fill="x", padx=10, pady=8)
build_btn = tk.Button(bf, text="Generate Combined PDF", width=22, command=build_combined_pdf)
build_btn.pack(side="left", padx=6)
extract_select_btn = tk.Button(bf, text="Extract Frames & Select", width=22, command=extract_frames_and_select)
extract_select_btn.pack(side="left", padx=6)
tk.Button(bf, text="Stop", width=10, command=stop_build).pack(side="left", padx=6)
open_location_btn = tk.Button(bf, text="Open File Location", width=16,
                                state="disabled", command=open_file_location)
open_location_btn.pack(side="left", padx=6)
tk.Button(bf, text="Exit", fg="red", width=10, command=shutdown).pack(side="right", padx=6)

tk.Label(root,
         text="✨ 'Extract Frames & Select' now auto-groups similar frames using AI perceptual hashing (pHash).\n"
              "Left panel shows similarity groups — click any group to filter the frame grid. "
              "Each frame shows its similarity % badge.\n"
              "Adjust the Sensitivity slider and click Re-cluster to tune grouping. "
              "pip install pypdf  if merge is missing.",
         wraplength=790, justify="left", fg="#333333").pack(padx=10, pady=4)

root.mainloop()