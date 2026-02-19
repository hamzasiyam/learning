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
    """Redirect stderr to devnull to suppress FFmpeg H.264 warnings."""
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
    "3 sec (Original resolution)": {"seconds": 3,  "width": ORIGINAL_RES_WIDTH, "quality": 100},
    "1 sec (Original resolution)": {"seconds": 1,  "width": ORIGINAL_RES_WIDTH, "quality": 100},
    "8 sec (Small file ~15MB)":    {"seconds": 8,  "width": 480,  "quality": 38},
    "8 sec (Sharp text)":          {"seconds": 8,  "width": 1280, "quality": 72},
    "8 sec (Smallest / Slides)":   {"seconds": 8,  "width": 1024, "quality": 62},
    "10 sec (Lower Res)":          {"seconds": 10, "width": 1024, "quality": 58},
    "Lecture (Recommended)":       {"seconds": 15, "width": 1400, "quality": 68},
    "High Compression":            {"seconds": 30, "width": 900,  "quality": 48},
    "Detailed":                    {"seconds": 5,  "width": 1600, "quality": 75},
}

VIDEO_EXTS  = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v")
IMAGE_EXTS  = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp")


# =========================
# AI Perceptual Hash
# =========================
PHASH_SIZE        = 16
SIMILAR_THRESHOLD = 10

SIM_COLOR_REPR   = "#d0e8ff"
SIM_COLOR_HIGH   = "#c8f5c8"
SIM_COLOR_MED    = "#fff3c0"
SIM_COLOR_LOW    = "#ffe0c0"
SIM_COLOR_UNIQUE = "#f0f0f0"


def sim_badge_color(sim: float) -> str:
    if sim >= 90: return SIM_COLOR_HIGH
    if sim >= 75: return SIM_COLOR_MED
    if sim >= 50: return SIM_COLOR_LOW
    return SIM_COLOR_UNIQUE


def compute_phash(img_path: str) -> Optional[np.ndarray]:
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
    frame_paths: List[str]  = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)
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
    def last_frame_path(self) -> str:
        return self.frame_paths[-1] if self.frame_paths else ""


def cluster_frames_by_similarity(frame_paths: List[str], threshold: int = SIMILAR_THRESHOLD,
                                  progress_cb=None) -> List[SimilarityGroup]:
    if not frame_paths:
        return []
    total = len(frame_paths)
    hashes: List[Optional[np.ndarray]] = []
    for i, p in enumerate(frame_paths):
        hashes.append(compute_phash(p))
        if progress_cb and (i % max(1, total // 20) == 0):
            progress_cb(i, total, f"Hashing {i+1}/{total}")

    groups: List[SimilarityGroup] = []
    rep_hashes: List[Optional[np.ndarray]] = []

    for i, p in enumerate(frame_paths):
        h = hashes[i]
        if h is None:
            g = SimilarityGroup(group_id=len(groups), frame_paths=[p],
                                similarities=[100.0], representative_path=p)
            groups.append(g); rep_hashes.append(None)
            continue
        best_gi, best_dist = None, threshold + 1
        for gi, rh in enumerate(rep_hashes):
            if rh is None: continue
            d = hamming_distance(h, rh)
            if d <= threshold and d < best_dist:
                best_dist = d; best_gi = gi
        if best_gi is not None:
            groups[best_gi].frame_paths.append(p)
            groups[best_gi].similarities.append(similarity_percent(best_dist))
        else:
            g = SimilarityGroup(group_id=len(groups), frame_paths=[p],
                                similarities=[100.0], representative_path=p)
            groups.append(g); rep_hashes.append(h)

    if progress_cb:
        progress_cb(total, total, "Done")
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
extra_images: List[str] = []   # images added in the main GUI


# =========================
# Helpers
# =========================
_WIN_RESERVED = frozenset({"CON","PRN","AUX","NUL","COM1","COM2","COM3","COM4","COM5",
                            "COM6","COM7","COM8","COM9","LPT1","LPT2","LPT3","LPT4",
                            "LPT5","LPT6","LPT7","LPT8","LPT9"})


def sanitize_filename(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip()


def filename_for_windows(title: str, max_length: int = 200) -> str:
    if not title or not title.strip():
        return "Combined"
    s = sanitize_filename(title)
    s = re.sub(r"\s+", " ", s).strip(" .")
    if not s: return "Combined"
    if len(s) > max_length: s = s[:max_length].rstrip(" .")
    if not s: return "Combined"
    if s.upper() in _WIN_RESERVED: return f"{s}_document"
    return s


def format_seconds(s: float) -> str:
    s = max(0, int(s))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


def format_bytes(num_bytes: float) -> str:
    if num_bytes <= 0: return "0 B"
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def estimate_video_pdf_bytes(video_path, seconds_step, max_width, quality):
    with suppress_ffmpeg_stderr():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total_f <= 0: return None
        dur = total_f / fps
        frames = int(dur // seconds_step) + 1
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
            try: total += os.path.getsize(item.path)
            except Exception: pass
        else:
            est = estimate_video_pdf_bytes(item.path, ss, mw, q)
            if est is None: unknown += 1
            else: total += est
    return total * 1.05 * 0.85, unknown


def ui_call(fn, *args, **kwargs):
    root.after(0, lambda: fn(*args, **kwargs))

def ui_set_status(text): ui_call(status_label.config, text=text)
def ui_set_eta(text):    ui_call(eta_label.config, text=text)
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
    try: ui_call(extract_select_btn.config, state=state)
    except NameError: pass
    ui_call(preset_combo.config, state=("disabled" if running else "readonly"))
    ui_call(queue_lb.config, state=state)

def ui_popup_done(output_pdf):
    ui_call(lambda: messagebox.showinfo(
        "PDF Created", f"Your PDF has been created:\n\n{os.path.basename(output_pdf)}"))


def make_temp_dir_near_output(output_pdf):
    base_dir = os.path.dirname(os.path.abspath(output_pdf))
    safe_tmp = os.path.join(base_dir, "_video_to_pdf_frames_tmp")
    os.makedirs(safe_tmp, exist_ok=True)
    return tempfile.mkdtemp(dir=safe_tmp)


def cleanup_dir(path):
    try:
        for root_dir, dirs, files in os.walk(path, topdown=False):
            for f in files:
                try: os.remove(os.path.join(root_dir, f))
                except Exception: pass
            for d in dirs:
                try: os.rmdir(os.path.join(root_dir, d))
                except Exception: pass
        os.rmdir(path)
    except Exception: pass


def resize_frame(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width: return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# =========================
# Video → temp PDF
# =========================
def video_to_temp_pdf(video_path, temp_pdf_path, preset_name, meta, progress_cb=None):
    s = PRESETS[preset_name]
    seconds_step, max_width, quality = int(s["seconds"]), int(s["width"]), int(s["quality"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {os.path.basename(video_path)}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur   = (total_f / fps) if (fps > 0 and total_f > 0) else 0.0
    targets = list(range(0, int(dur) + 1, seconds_step)) if dur > 0 else None
    frames_dir = os.path.join(os.path.dirname(temp_pdf_path), "_frames_" + str(int(time.time())))
    os.makedirs(frames_dir, exist_ok=True)
    try:
        with suppress_ffmpeg_stderr():
            t0 = time.time()
            img_paths: List[str] = []
            if targets is not None:
                if progress_cb: progress_cb(0, len(targets), "ETA: --", f"Extracting: {os.path.basename(video_path)}")
                for idx, tsec in enumerate(targets, 1):
                    if stop_event.is_set(): raise RuntimeError("Stopped")
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                    ret, frame = cap.read()
                    if not ret: break
                    frame = resize_frame(frame, max_width)
                    ip = os.path.join(frames_dir, f"t{tsec:06d}.jpg")
                    if cv2.imwrite(ip, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                        img_paths.append(ip)
                    if progress_cb:
                        el = time.time() - t0
                        eta = f"ETA: {format_seconds(max(0,(len(targets)-idx)*el/idx))}" if idx >= 2 else "ETA: --"
                        progress_cb(idx, len(targets), eta, f"Extracting: {os.path.basename(video_path)}")
            else:
                idx = tsec = 0
                while True:
                    if stop_event.is_set(): raise RuntimeError("Stopped")
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                    ret, frame = cap.read()
                    if not ret: break
                    frame = resize_frame(frame, max_width)
                    ip = os.path.join(frames_dir, f"t{tsec:06d}.jpg")
                    if cv2.imwrite(ip, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                        img_paths.append(ip)
                    idx += 1; tsec += seconds_step
                    if progress_cb: progress_cb(idx, max(1, idx), "ETA: --", f"Extracting: {os.path.basename(video_path)}")
            cap.release()
        img_paths.sort()
        if not img_paths:
            raise RuntimeError(f"No frames extracted: {os.path.basename(video_path)}")
        c = canvas.Canvas(temp_pdf_path)
        c.setTitle(meta.get("title", "")); c.setAuthor(meta.get("author", "")); c.setSubject(meta.get("type", ""))
        for ip in img_paths:
            if stop_event.is_set(): raise RuntimeError("Stopped")
            img = ImageReader(ip)
            w, h = img.getSize()
            c.setPageSize((w, h)); c.drawImage(img, 0, 0, width=w, height=h); c.showPage()
        c.save()
    finally:
        try: cap.release()
        except Exception: pass
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
        if not cap.isOpened(): continue
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            tf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            total += int(tf / fps) // seconds_step + 1 if fps > 0 and tf > 0 else 1
        finally: cap.release()
    return max(1, total)


def _sanitize_folder_name(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "video"


def extract_all_frames_from_videos(video_paths, preset_name, output_dir, progress_cb=None, estimated_total=None):
    s = PRESETS[preset_name]
    seconds_step, max_width, quality = int(s["seconds"]), int(s["width"]), int(s["quality"])
    all_paths: List[str] = []
    video_collections: List[Tuple[str, List[str]]] = []
    step_max = estimated_total or 1

    for v_idx, vp in enumerate(video_paths):
        if stop_event.is_set(): raise RuntimeError("Stopped")
        with suppress_ffmpeg_stderr():
            cap = cv2.VideoCapture(vp)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open: {os.path.basename(vp)}")
        vname  = os.path.basename(vp)
        folder = f"v{v_idx:03d}_{_sanitize_folder_name(os.path.splitext(vname)[0])}"
        vdir   = os.path.join(output_dir, folder)
        os.makedirs(vdir, exist_ok=True)
        vpaths: List[str] = []

        with suppress_ffmpeg_stderr():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            tf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            dur = (tf / fps) if (fps > 0 and tf > 0) else 0.0
            if dur > 0:
                for tsec in range(0, int(dur) + 1, seconds_step):
                    if stop_event.is_set(): raise RuntimeError("Stopped")
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                    ret, frame = cap.read()
                    if not ret: break
                    frame = resize_frame(frame, max_width)
                    ip = os.path.join(vdir, f"t{tsec:06d}.jpg")
                    if cv2.imwrite(ip, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                        all_paths.append(ip); vpaths.append(ip)
                    if progress_cb: progress_cb(len(all_paths), step_max, "", f"Extracting: {vname}")
            else:
                next_t, fc, fps = 0, 0, fps or 25.0
                while True:
                    if stop_event.is_set(): raise RuntimeError("Stopped")
                    ret, frame = cap.read()
                    if not ret: break
                    pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                    pos_sec = (pos / 1000.0) if pos >= 0 else (fc / fps)
                    if pos_sec >= next_t:
                        frame = resize_frame(frame, max_width)
                        ip = os.path.join(vdir, f"t{int(next_t):06d}.jpg")
                        if cv2.imwrite(ip, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                            all_paths.append(ip); vpaths.append(ip)
                        next_t += seconds_step
                        if progress_cb: progress_cb(len(all_paths), step_max, "", f"Extracting: {vname}")
                    fc += 1
            cap.release()
        video_collections.append((vname, vpaths))
    return all_paths, video_collections


OPTIMIZE_MAX_WIDTH = 1600
OPTIMIZE_QUALITY   = 82


def estimate_pdf_size_bytes(image_paths, optimize=False):
    try: total = sum(os.path.getsize(p) for p in image_paths if os.path.exists(p))
    except Exception: return 0
    if optimize: total = int(total * 0.45)
    return int(total * (1.15 if optimize else 1.12))


def build_pdf_from_selected_images(image_paths, output_pdf, meta, bar_height=36, progress_cb=None, optimize=False):
    from reportlab.lib.colors import black
    c = canvas.Canvas(output_pdf)
    c.setTitle(meta.get("title", "")); c.setAuthor(meta.get("author", "")); c.setSubject(meta.get("type", ""))
    tmp_opt = tempfile.mkdtemp(prefix="video_to_pdf_opt_") if optimize else None
    try:
        for i, ip in enumerate(image_paths, 1):
            if progress_cb: progress_cb(i, len(image_paths), f"Page {i}/{len(image_paths)}")
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
            c.setFillColor(black); c.rect(0, 0, w, bar_height, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica", 14)
            c.drawString(12, bar_height * 0.25, f"#{i}")
            c.showPage()
    finally:
        if tmp_opt: cleanup_dir(tmp_opt)
    c.save()
    try: add_metadata_to_pdf_file(output_pdf, meta)
    except Exception: pass


# =========================
# UI Constants
# =========================
MAX_IMAGES_PER_PAGE   = 12
THUMB_MAX_WIDTH       = 200
ZOOM_MAX_WIDTH        = 900
BAR_HEIGHT_PTS        = 36
IMAGES_PER_PAGE_CHOICES = (6, 8, 12)


# =========================
# VideoCollectionFrameSelectionWindow  — 5-column layout
# =========================
#
#  ┌─────────────┬──────────────────┬──────────────────┬──────────────────────────┬───────────────┐
#  │ Col 1       │ Col 2            │ Col 3            │ Col 4                    │ Col 5         │
#  │ Source      │ AI Similarity    │ Group Previews   │ Frame Preview Grid       │ Actions       │
#  │ Videos      │ Groups           │ (last frame of   │ (paginated thumbnails    │ (all buttons, │
#  │             │ (group list)     │  each group)     │  with similarity badges) │  stacked)     │
#  └─────────────┴──────────────────┴──────────────────┴──────────────────────────┴───────────────┘

class VideoCollectionFrameSelectionWindow:
    def __init__(self, parent, video_collections, temp_dir, meta, extra_image_paths=None):
        self.video_collections = video_collections
        self.temp_dir = temp_dir
        self.meta = meta
        self.extra_image_paths: List[str] = list(extra_image_paths or [])

        self.keep: Dict[Tuple[int, int], bool] = {}
        for ci, (_, paths) in enumerate(video_collections):
            for fi in range(len(paths)):
                self.keep[(ci, fi)] = False

        self.similarity_groups: Dict[int, List[SimilarityGroup]] = {}
        self.frame_to_group: Dict[int, Tuple[int, float]] = {}

        self.selected_collection_idx = 0
        self.page_index = 0
        self.images_per_page = 8
        self._current_filter: Optional[int] = None
        self._filtered_indices: List[int] = []

        self.win = tk.Toplevel(parent)
        self.win.title("Extract Frames & Select  —  AI Similarity Groups")
        self.win.state("zoomed")   # start maximised for the 4-column layout
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._thumb_dir = os.path.join(temp_dir, "_thumbs")
        os.makedirs(self._thumb_dir, exist_ok=True)
        self._zoom_win  = None
        self._zoom_ref  = None
        self._thumb_refs: List = []   # keep preview PhotoImages alive

        # ── Root PanedWindow ────────────────────────────────────────
        paned = tk.PanedWindow(self.win, orient=tk.HORIZONTAL, sashrelief="raised", sashwidth=5)
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        # ═══════════════════════════════════════════════════════════
        # COL 1 — Source Videos
        # ═══════════════════════════════════════════════════════════
        c1 = tk.Frame(paned, bg="#f0f4f8")
        c1.pack(fill="both", expand=True)

        tk.Label(c1, text="Source Videos", font=("", 11, "bold"),
                 bg="#2c3e50", fg="white", pady=5).pack(fill="x")

        # Video count badge
        self.vid_count_lbl = tk.Label(c1, text="", font=("", 9), bg="#f0f4f8", fg="#555")
        self.vid_count_lbl.pack(anchor="w", padx=6, pady=(4, 0))

        vf = tk.Frame(c1, bg="#f0f4f8")
        vf.pack(fill="both", expand=True, padx=4, pady=4)
        vsb = tk.Scrollbar(vf)
        vsb.pack(side="right", fill="y")
        self.vid_lb = tk.Listbox(vf, font=("", 10), yscrollcommand=vsb.set,
                                  exportselection=False, selectbackground="#3498db",
                                  selectforeground="white", activestyle="dotbox",
                                  relief="flat", bd=0)
        vsb.config(command=self.vid_lb.yview)
        self.vid_lb.pack(fill="both", expand=True)
        self.vid_lb.bind("<<ListboxSelect>>", self._on_video_select)

        for vname, vpaths in video_collections:
            self.vid_lb.insert(tk.END, f"  {vname}")
            # colour-code: every other item for readability
        for i in range(len(video_collections)):
            self.vid_lb.itemconfig(i, bg="#ffffff" if i % 2 == 0 else "#eaf2ff")

        self.vid_count_lbl.config(text=f"{len(video_collections)} video(s) in queue")

        paned.add(c1, minsize=220)

        # ═══════════════════════════════════════════════════════════
        # COL 2 — AI Similarity Groups
        # ═══════════════════════════════════════════════════════════
        c2 = tk.Frame(paned, bg="#f8f8f0")
        c2.pack(fill="both", expand=True)

        tk.Label(c2, text="🤖 AI Similarity Groups", font=("", 11, "bold"),
                 bg="#27ae60", fg="white", pady=5).pack(fill="x")

        # Sensitivity row
        thr_row = tk.Frame(c2, bg="#f8f8f0")
        thr_row.pack(fill="x", padx=6, pady=(6, 0))
        tk.Label(thr_row, text="Sensitivity:", font=("", 9), bg="#f8f8f0").pack(side="left")
        self.thr_var = tk.IntVar(value=SIMILAR_THRESHOLD)
        tk.Scale(thr_row, from_=2, to=50, orient=tk.HORIZONTAL, variable=self.thr_var,
                 length=100, showvalue=True, bg="#f8f8f0").pack(side="left", padx=3)
        tk.Button(thr_row, text="Re-cluster", font=("", 8),
                  command=self._rerun_clustering).pack(side="left", padx=2)

        # Legend
        leg = tk.Frame(c2, bg="#f8f8f0")
        leg.pack(anchor="w", padx=6, pady=2)
        for color, lbl in [(SIM_COLOR_REPR,"Rep"),(SIM_COLOR_HIGH,"≥90%"),
                            (SIM_COLOR_MED,"75-89%"),(SIM_COLOR_LOW,"50-74%"),(SIM_COLOR_UNIQUE,"Unique")]:
            tk.Label(leg, text="■", fg=color, font=("", 11), bg="#f8f8f0").pack(side="left")
            tk.Label(leg, text=lbl, font=("", 7), bg="#f8f8f0").pack(side="left", padx=(0,3))

        # Groups listbox
        glf = tk.Frame(c2, bg="#f8f8f0")
        glf.pack(fill="x", padx=4, pady=2)
        gsb = tk.Scrollbar(glf)
        gsb.pack(side="right", fill="y")
        self.grp_lb = tk.Listbox(glf, height=14, font=("", 9), yscrollcommand=gsb.set,
                                  exportselection=False, selectbackground="#27ae60",
                                  selectforeground="white", relief="flat", bd=0)
        gsb.config(command=self.grp_lb.yview)
        self.grp_lb.pack(fill="x")
        self.grp_lb.bind("<<ListboxSelect>>", self._on_group_select)

        # Group actions
        ga = tk.Frame(c2, bg="#f8f8f0")
        ga.pack(fill="x", padx=4, pady=(2, 0))
        tk.Button(ga, text="Show All",    font=("", 8), command=self._show_all   ).pack(side="left", padx=1)
        tk.Button(ga, text="✓ Keep",      font=("", 8), command=self._keep_group ).pack(side="left", padx=1)
        tk.Button(ga, text="✗ Skip",      font=("", 8), command=self._skip_group ).pack(side="left", padx=1)

        self.grp_info = tk.Label(c2, text="", font=("", 8), fg="#555", bg="#f8f8f0",
                                  wraplength=220, justify="left")
        self.grp_info.pack(anchor="w", padx=6, pady=2)

        paned.add(c2, minsize=210)

        # ═══════════════════════════════════════════════════════════
        # COL 3 — Group Previews (last frame of each group)
        # ═══════════════════════════════════════════════════════════
        c3 = tk.Frame(paned, bg="#f0f8f0")
        c3.pack(fill="both", expand=True)

        tk.Label(c3, text="Group Previews", font=("", 11, "bold"),
                 bg="#1e8449", fg="white", pady=5).pack(fill="x")
        tk.Label(c3, text="Last frame of each group\nClick to filter grid \u2192",
                 font=("", 8), bg="#f0f8f0", fg="#555").pack(anchor="w", padx=6, pady=(4, 2))

        self._preview_canvas = tk.Canvas(c3, bg="#f0f8f0", highlightthickness=0)
        prev_vsb = tk.Scrollbar(c3, orient="vertical", command=self._preview_canvas.yview)
        self._preview_canvas.configure(yscrollcommand=prev_vsb.set)
        prev_vsb.pack(side="right", fill="y")
        self._preview_canvas.pack(fill="both", expand=True, padx=2, pady=(0, 4))
        self._preview_inner = tk.Frame(self._preview_canvas, bg="#f0f8f0")
        self._preview_cwin  = self._preview_canvas.create_window((0, 0), window=self._preview_inner, anchor="nw")
        self._preview_inner.bind("<Configure>", self._on_preview_configure)
        self._preview_canvas.bind("<Configure>", self._on_preview_canvas_configure)
        self._preview_canvas.bind("<MouseWheel>", self._on_preview_scroll)
        self._preview_inner.bind("<MouseWheel>", self._on_preview_scroll)

        paned.add(c3, minsize=180)

        # ═══════════════════════════════════════════════════════════
        # COL 4 — Frame Preview Grid
        # ═══════════════════════════════════════════════════════════
        c4 = tk.Frame(paned)
        c4.pack(fill="both", expand=True)

        # Status bar across top of col 4
        top4 = tk.Frame(c4, bg="#ecf0f1", relief="flat", bd=0)
        top4.pack(fill="x", padx=0, pady=0)
        tk.Label(top4, text="Frame Preview Grid", font=("", 11, "bold"),
                 bg="#2980b9", fg="white", pady=5, padx=8).pack(side="left", fill="y")
        self.range_lbl  = tk.Label(top4, text="", font=("", 10), bg="#ecf0f1")
        self.range_lbl.pack(side="left", padx=12)
        self.filter_lbl = tk.Label(top4, text="", font=("", 9, "italic"), fg="#1a5276", bg="#ecf0f1")
        self.filter_lbl.pack(side="left", padx=6)

        info4 = tk.Frame(c4, bg="#f9f9f9")
        info4.pack(fill="x", padx=4, pady=2)
        self.count_lbl = tk.Label(info4, text="", font=("", 9), bg="#f9f9f9")
        self.count_lbl.pack(side="left", padx=4)
        self.size_lbl  = tk.Label(info4, text="", font=("", 9), bg="#f9f9f9", fg="#555")
        self.size_lbl.pack(side="left", padx=8)

        # Per page
        pprow = tk.Frame(c4, bg="#f9f9f9")
        pprow.pack(fill="x", padx=4)
        tk.Label(pprow, text="Per page:", font=("", 9), bg="#f9f9f9").pack(side="left")
        self.pp_var = tk.StringVar(value="8")
        pp_cb = Combobox(pprow, textvariable=self.pp_var,
                         values=[str(n) for n in IMAGES_PER_PAGE_CHOICES],
                         state="readonly", width=4)
        pp_cb.pack(side="left", padx=3)
        pp_cb.bind("<<ComboboxSelected>>", self._on_per_page_change)
        tk.Label(pprow, text="  \u2190 \u2192 to navigate pages", font=("", 8), fg="#777", bg="#f9f9f9").pack(side="left")

        # Grid
        self.grid_frame = tk.Frame(c4)
        self.grid_frame.pack(fill="both", expand=True, padx=4, pady=3)
        self.cells:      List[Tuple] = []
        self.chk_vars:   List[tk.BooleanVar] = []
        self.photo_refs: List[Optional[tk.PhotoImage]] = []

        for slot in range(MAX_IMAGES_PER_PAGE):
            cell = tk.Frame(self.grid_frame, relief="ridge", bd=2, bg="light gray")
            sim_lbl = tk.Label(cell, text="", font=("Helvetica", 7, "bold"),
                                bg="light gray", pady=1, padx=2)
            sim_lbl.pack(fill="x")
            img_lbl = tk.Label(cell, text="", image=None, compound="top", cursor="hand2")
            img_lbl.pack(fill="both", expand=True, padx=2)
            img_lbl.bind("<Button-1>", lambda e, s=slot: self._on_click(s))
            img_lbl.bind("<Button-3>", lambda e, s=slot: self._on_rclick(s))
            v  = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(cell, text="Keep", variable=v, command=self._on_toggle)
            cb.pack(pady=1)
            self.cells.append((cell, sim_lbl, img_lbl, cb))
            self.chk_vars.append(v)
            self.photo_refs.append(None)

        # Pagination row
        nav = tk.Frame(c4, bg="#ecf0f1", relief="flat")
        nav.pack(fill="x", padx=4, pady=4)
        self.prev_btn = tk.Button(nav, text="\u25c4  Prev", width=10, font=("", 9),
                                   command=self._prev_page)
        self.prev_btn.pack(side="left", padx=4)
        self.next_btn = tk.Button(nav, text="Next  \u25ba", width=10, font=("", 9),
                                   command=self._next_page)
        self.next_btn.pack(side="left", padx=4)

        paned.add(c4, minsize=500)

        # ═══════════════════════════════════════════════════════════
        # COL 5 — Action Buttons (right panel)
        # ═══════════════════════════════════════════════════════════
        c5 = tk.Frame(paned, bg="#2c3e50", width=190)
        c5.pack_propagate(False)
        c5.pack(fill="y")

        def _section_hdr(text, color="#34495e"):
            tk.Label(c5, text=text, font=("", 9, "bold"), bg=color, fg="white",
                     pady=4, padx=6, anchor="w").pack(fill="x", pady=(8, 2), padx=0)

        def _btn(text, cmd, bg="#455a64", fg="white", w=22):
            tk.Button(c5, text=text, command=cmd, bg=bg, fg=fg, font=("", 9),
                      relief="flat", cursor="hand2", width=w, pady=4,
                      activebackground="#607d8b", activeforeground="white"
                      ).pack(fill="x", padx=8, pady=2)

        # ── Smart Selection ─────────────────────────────────────────
        _section_hdr("✨  Smart Selection", "#1a5276")
        _btn("Select Last Frame\nof Each Group\n+ All Unique", self._smart_select,
             bg="#1a6b2a", w=22)
        _btn("Select All Videos\n(same as above)", self._smart_select_all,
             bg="#145a32", w=22)
        tk.Frame(c5, height=1, bg="#4a6572").pack(fill="x", padx=8, pady=4)

        # ── Frame Selection ─────────────────────────────────────────
        _section_hdr("Frame Selection", "#34495e")
        _btn("Show All Frames",  self._show_all,     bg="#546e7a")
        _btn("✓ Keep Group",     self._keep_group,   bg="#2e7d32")
        _btn("✗ Skip Group",     self._skip_group,   bg="#c62828")
        tk.Frame(c5, height=1, bg="#4a6572").pack(fill="x", padx=8, pady=4)

        # ── Extra Images ────────────────────────────────────────────
        _section_hdr("🖼  Extra Images", "#5d4037")
        self.extra_count_lbl = tk.Label(c5, text="", font=("", 8), bg="#2c3e50",
                                         fg="#aaaaaa", wraplength=170, justify="left")
        self.extra_count_lbl.pack(anchor="w", padx=8, pady=2)
        self._update_extra_label()
        _btn("Add Images",   self._import_images, bg="#6d4c41")
        _btn("Clear Images", self._clear_images,  bg="#795548")
        tk.Frame(c5, height=1, bg="#4a6572").pack(fill="x", padx=8, pady=4)

        # ── PDF Output ──────────────────────────────────────────────
        _section_hdr("📄  PDF Output", "#1565c0")
        self.opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(c5, text="Optimize (smaller file)", variable=self.opt_var,
                        bg="#2c3e50", fg="white", selectcolor="#1e2e3e",
                        activebackground="#2c3e50", activeforeground="white",
                        font=("", 8), command=self._update_size
                        ).pack(anchor="w", padx=10, pady=2)
        _btn("📄  Create PDF",  self._create_pdf, bg="#1565c0")
        _btn("ℹ  Size Info",   self._size_info,  bg="#37474f")

        tk.Frame(c5, height=1, bg="#4a6572").pack(fill="x", padx=8, pady=4)
        _btn("✖  Close",       self._on_close,   bg="#b71c1c")

        paned.add(c5, minsize=185)

        # Key bindings
        self.win.bind("<Left>",  lambda e: (self._prev_page(), "break"))
        self.win.bind("<Right>", lambda e: (self._next_page(), "break"))

        self._apply_grid()
        self.vid_lb.selection_set(0)
        self._on_video_select(None)

    # ── Preview scroll helpers ──────────────────────────────────────
    def _on_preview_configure(self, event=None):
        self._preview_canvas.configure(scrollregion=self._preview_canvas.bbox("all"))

    def _on_preview_canvas_configure(self, event):
        self._preview_canvas.itemconfig(self._preview_cwin, width=event.width)

    def _on_preview_scroll(self, event):
        self._preview_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── Rebuild group preview tiles ─────────────────────────────────
    def _rebuild_preview_tiles(self, ci: int):
        for w in self._preview_inner.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        groups = self.similarity_groups.get(ci, [])
        if not groups:
            tk.Label(self._preview_inner, text="Clustering\u2026", bg="#f0f8f0",
                     fg="#999", font=("", 8)).pack(pady=10)
            return

        dup   = [g for g in groups if not g.is_unique]
        uniq  = [g for g in groups if g.is_unique]
        _, all_paths = self.video_collections[ci]
        thumb_w = 150

        # similar groups
        if dup:
            tk.Label(self._preview_inner,
                     text=f"Similar Groups ({len(dup)})",
                     bg="#d4efdf", fg="#1a5e2a", font=("", 8, "bold"),
                     anchor="w", padx=4).pack(fill="x", pady=(4, 1))
        for g in dup:
            self._add_preview_tile(g.last_frame_path,
                                   f"G{g.group_id+1}  {len(g.frame_paths)}f  ~{g.avg_similarity:.0f}%",
                                   thumb_w, bg="#e8f8f0", hover="#c8eeda",
                                   cb=lambda gid=g.group_id: self._filter_to_group(gid))

        # unique frames tile
        if uniq:
            tk.Frame(self._preview_inner, height=1, bg="#aaaaaa").pack(fill="x", padx=4, pady=3)
            tk.Label(self._preview_inner,
                     text=f"Unique Frames ({len(uniq)})",
                     bg="#d6eaf8", fg="#1a4e6d", font=("", 8, "bold"),
                     anchor="w", padx=4).pack(fill="x", pady=(0, 1))
            first_unique = uniq[0].frame_paths[0] if uniq else ""
            self._add_preview_tile(first_unique,
                                   f"All Unique  {len(uniq)} frames",
                                   thumb_w, bg="#eaf4fb", hover="#cce5f5",
                                   cb=self._filter_to_unique)

        self._on_preview_configure()

    def _add_preview_tile(self, img_path, label_text, thumb_w, bg, hover, cb):
        tile = tk.Frame(self._preview_inner, bg=bg, cursor="hand2", relief="flat")
        tile.pack(fill="x", padx=4, pady=2)

        # thumbnail
        ph = self._load_preview_thumb(img_path, thumb_w)
        if ph:
            il = tk.Label(tile, image=ph, bg=bg, cursor="hand2")
            il.pack(pady=(3,0))
            il.bind("<Button-1>", lambda e: cb())
            il.bind("<MouseWheel>", self._on_preview_scroll)
            self._thumb_refs.append(ph)
        else:
            tk.Label(tile, text="[no img]", bg=bg, fg="#999", font=("",7), height=3).pack()

        lb = tk.Label(tile, text=label_text, bg=bg, fg="#333", font=("",8), wraplength=thumb_w)
        lb.pack(pady=(1,3))
        lb.bind("<Button-1>", lambda e: cb())

        for w in [tile, lb]:
            w.bind("<Button-1>", lambda e: cb())
            w.bind("<MouseWheel>", self._on_preview_scroll)
            w.bind("<Enter>", lambda e, t=tile: t.config(bg=hover) or [c.config(bg=hover) for c in t.winfo_children() if hasattr(c,'config')])
            w.bind("<Leave>", lambda e, t=tile, b=bg: t.config(bg=b) or [c.config(bg=b) for c in t.winfo_children() if hasattr(c,'config')])

    def _load_preview_thumb(self, img_path: str, max_w: int) -> Optional[tk.PhotoImage]:
        if not img_path or not os.path.exists(img_path):
            return None
        try:
            fr = cv2.imread(img_path)
            if fr is None: return None
            h, w = fr.shape[:2]
            if w > max_w:
                fr = cv2.resize(fr, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
            tmp = os.path.join(self._thumb_dir, f"_prev_{abs(hash(img_path))}.png")
            cv2.imwrite(tmp, fr)
            return tk.PhotoImage(file=tmp)
        except Exception:
            return None

    def _filter_to_group(self, group_id: int):
        ci = self.selected_collection_idx
        groups = self.similarity_groups.get(ci, [])
        dup = [g for g in groups if not g.is_unique]
        for lb_idx, g in enumerate(dup, 1):
            if g.group_id == group_id:
                self.grp_lb.selection_clear(0, tk.END)
                self.grp_lb.selection_set(lb_idx)
                self.grp_lb.see(lb_idx)
                self._on_group_select(None)
                return

    def _filter_to_unique(self):
        ci = self.selected_collection_idx
        groups = self.similarity_groups.get(ci, [])
        dup = [g for g in groups if not g.is_unique]
        lb_idx = len(dup) + 2
        self.grp_lb.selection_clear(0, tk.END)
        self.grp_lb.selection_set(lb_idx)
        self.grp_lb.see(lb_idx)
        self._on_group_select(None)

    # ── Grid layout ─────────────────────────────────────────────────
    def _apply_grid(self):
        n     = self.images_per_page
        ncols = 4 if n >= 8 else 3
        nrows = (n + ncols - 1) // ncols
        for c in range(4):
            self.grid_frame.columnconfigure(c, weight=1 if c < ncols else 0)
        for r in range(4):
            self.grid_frame.rowconfigure(r, weight=1 if r < nrows else 0)
        for slot in range(MAX_IMAGES_PER_PAGE):
            cell = self.cells[slot][0]
            if slot < n:
                row, col = slot // ncols, slot % ncols
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
        except Exception: pass

    # ── Video selection ─────────────────────────────────────────────
    def _on_video_select(self, event):
        sel = self.vid_lb.curselection()
        if sel:
            self.selected_collection_idx = sel[0]
        ci = self.selected_collection_idx
        # Highlight active video
        for i in range(len(self.video_collections)):
            self.vid_lb.itemconfig(i, bg="#3498db" if i == ci else ("#ffffff" if i%2==0 else "#eaf2ff"),
                                    fg="white" if i == ci else "black")
        _, paths = self.video_collections[ci]
        self._filtered_indices = list(range(len(paths)))
        self._current_filter = None
        self.filter_lbl.config(text="")
        self.grp_info.config(text="")
        self.page_index = 0
        self._start_clustering(ci)
        self._refresh()

    # ── Clustering ──────────────────────────────────────────────────
    def _start_clustering(self, ci: int):
        if ci in self.similarity_groups:
            self._rebuild_lookup(ci)
            self._populate_group_list(ci)
            self._rebuild_preview_tiles(ci)
            return
        _, paths = self.video_collections[ci]
        if not paths:
            self.similarity_groups[ci] = []
            self.frame_to_group = {}
            self._populate_group_list(ci)
            self._rebuild_preview_tiles(ci)
            return
        self.grp_lb.delete(0, tk.END)
        self.grp_lb.insert(tk.END, "⏳ Analysing…")
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
        self._rebuild_preview_tiles(ci)
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
            self.grp_lb.insert(tk.END, "(no frames)"); return
        dup  = [g for g in groups if not g.is_unique]
        uniq = [g for g in groups if g.is_unique]
        self.grp_lb.insert(tk.END, f"━━ {len(dup)} similar groups ━━")
        for g in dup:
            self.grp_lb.insert(tk.END, f"  G{g.group_id+1}: {len(g.frame_paths)}f  ~{g.avg_similarity:.0f}%")
        self.grp_lb.insert(tk.END, f"━━ {len(uniq)} unique ━━")
        self.grp_lb.insert(tk.END, "  [All unique frames]")

    def _rerun_clustering(self):
        ci = self.selected_collection_idx
        self.similarity_groups.pop(ci, None)
        self._current_filter = None
        self._start_clustering(ci)

    # ── Group filter ────────────────────────────────────────────────
    def _on_group_select(self, event):
        sel = self.grp_lb.curselection()
        if not sel: return
        idx = sel[0]
        ci  = self.selected_collection_idx
        groups = self.similarity_groups.get(ci, [])
        dup  = [g for g in groups if not g.is_unique]
        uniq = [g for g in groups if g.is_unique]
        _, paths = self.video_collections[ci]
        if idx == 0 or idx == len(dup) + 1: return  # headers
        if 1 <= idx <= len(dup):
            g = dup[idx - 1]
            path_set = set(g.frame_paths)
            self._filtered_indices = [fi for fi, p in enumerate(paths) if p in path_set]
            self._current_filter   = g.group_id
            self.filter_lbl.config(text=f"🔍 G{g.group_id+1}: {len(g.frame_paths)} frames  (~{g.avg_similarity:.0f}% sim)")
            self.grp_info.config(text=f"Group {g.group_id+1}: {len(g.frame_paths)} similar frames.")
        elif idx == len(dup) + 2:
            path_set = set(p for g in uniq for p in g.frame_paths)
            self._filtered_indices = [fi for fi, p in enumerate(paths) if p in path_set]
            self._current_filter   = -1
            self.filter_lbl.config(text=f"🔍 {len(self._filtered_indices)} unique frames")
            self.grp_info.config(text="Frames with no similar partners.")
        else: return
        self.page_index = 0
        self._refresh()

    def _show_all(self):
        ci = self.selected_collection_idx
        _, paths = self.video_collections[ci]
        self._filtered_indices = list(range(len(paths)))
        self._current_filter   = None
        self.filter_lbl.config(text="")
        self.grp_info.config(text="")
        self.page_index = 0
        self._refresh()

    def _keep_group(self):
        if not self._filtered_indices: return
        ci = self.selected_collection_idx
        for fi in self._filtered_indices:
            self.keep[(ci, fi)] = True
        self._refresh(); self._update_count()

    def _skip_group(self):
        if not self._filtered_indices: return
        ci = self.selected_collection_idx
        for fi in self._filtered_indices:
            self.keep[(ci, fi)] = False
        self._refresh(); self._update_count()

    # ── Smart select ────────────────────────────────────────────────
    def _smart_select(self):
        ci = self.selected_collection_idx
        _, all_paths = self.video_collections[ci]
        groups = self.similarity_groups.get(ci, [])
        if not groups:
            messagebox.showinfo("Not Ready", "Similarity analysis not complete yet.")
            return
        path_to_fi = {p: fi for fi, p in enumerate(all_paths)}
        for fi in range(len(all_paths)):
            self.keep[(ci, fi)] = False
        sel_count = 0
        for g in groups:
            pick = g.frame_paths[0] if g.is_unique else g.last_frame_path
            fi   = path_to_fi.get(pick)
            if fi is not None:
                self.keep[(ci, fi)] = True
                sel_count += 1
        self._show_all()
        self._refresh()
        self._update_count()
        dup_n  = sum(1 for g in groups if not g.is_unique)
        uniq_n = sum(1 for g in groups if g.is_unique)
        messagebox.showinfo("Smart Selection Done",
            f"Selected {sel_count} frames:\n"
            f"• Last frame from {dup_n} similar group(s)\n"
            f"• All {uniq_n} unique frame(s)")



    def _smart_select_all(self):
        """Cluster ALL videos (if needed), then apply smart selection across all of them."""
        # Collect which videos still need clustering
        pending = [ci for ci in range(len(self.video_collections))
                   if ci not in self.similarity_groups]

        if not pending:
            # All already clustered — apply immediately
            self._apply_smart_select_all()
            return

        # Show a progress dialog and run clustering for all pending videos in parallel
        prog_win = tk.Toplevel(self.win)
        prog_win.title("Clustering…")
        prog_win.geometry("400x120")
        prog_win.transient(self.win)
        prog_win.grab_set()
        prog_win.resizable(False, False)

        tk.Label(prog_win, text="Analysing similarity for all videos…",
                 font=("", 10, "bold")).pack(pady=(16, 4))
        status_lbl = tk.Label(prog_win, text="", font=("", 9), fg="#555")
        status_lbl.pack()
        bar = Progressbar(prog_win, length=340, mode="determinate")
        bar.pack(pady=8)

        total_pending = len(pending)
        completed = [0]          # mutable counter shared across threads
        lock = threading.Lock()

        def cluster_one(ci):
            _, paths = self.video_collections[ci]
            threshold = self.thr_var.get()
            groups = cluster_frames_by_similarity(paths, threshold=threshold)
            self.similarity_groups[ci] = groups
            with lock:
                completed[0] += 1
                done = completed[0]
            # Update progress bar on the UI thread
            def _update():
                bar["value"] = done
                bar["maximum"] = total_pending
                vname = self.video_collections[ci][0]
                status_lbl.config(text=f"Done: {vname}  ({done}/{total_pending})")
                if done == total_pending:
                    prog_win.destroy()
                    self._apply_smart_select_all()
            self.win.after(0, _update)

        for ci in pending:
            threading.Thread(target=cluster_one, args=(ci,), daemon=True).start()

    def _apply_smart_select_all(self):
        """Apply smart selection (last frame per group + all unique) across all videos."""
        total_sel = 0
        total_dup = 0
        total_uniq = 0
        for ci, (_, all_paths) in enumerate(self.video_collections):
            groups = self.similarity_groups.get(ci, [])
            if not groups:
                continue
            # Also update the lookup table for each video so badges are correct
            self._rebuild_lookup_for(ci)
            path_to_fi = {p: fi for fi, p in enumerate(all_paths)}
            for fi in range(len(all_paths)):
                self.keep[(ci, fi)] = False
            for g in groups:
                pick = g.frame_paths[0] if g.is_unique else g.last_frame_path
                fi   = path_to_fi.get(pick)
                if fi is not None:
                    self.keep[(ci, fi)] = True
                    total_sel += 1
            total_dup  += sum(1 for g in groups if not g.is_unique)
            total_uniq += sum(1 for g in groups if g.is_unique)

        # Refresh group list and preview tiles for currently visible video
        ci = self.selected_collection_idx
        self._rebuild_lookup(ci)
        self._populate_group_list(ci)
        self._rebuild_preview_tiles(ci)
        self._show_all()
        self._refresh()
        self._update_count()
        messagebox.showinfo("Smart Selection \u2014 All Videos",
            f"Selected {total_sel} frames across {len(self.video_collections)} video(s):\n"
            f"\u2022 Last frame from {total_dup} similar group(s)\n"
            f"\u2022 All {total_uniq} unique frame(s)")

    def _rebuild_lookup_for(self, ci: int):
        """Rebuild frame_to_group for a specific collection index (used during all-video select)."""
        _, paths = self.video_collections[ci]
        path_to_fi = {p: fi for fi, p in enumerate(paths)}
        for g in self.similarity_groups.get(ci, []):
            for p, sim in zip(g.frame_paths, g.similarities):
                fi = path_to_fi.get(p)
                if fi is not None and ci == self.selected_collection_idx:
                    self.frame_to_group[fi] = (g.group_id, sim)

    # ── Thumbnails ──────────────────────────────────────────────────
    def _make_thumb(self, img_path: str, ci: int, fi: int) -> str:
        out = os.path.join(self._thumb_dir, f"v{ci}_f{fi}.png")
        if os.path.exists(out): return out
        try:
            fr = cv2.imread(img_path)
            if fr is None: return ""
            h, w = fr.shape[:2]
            if w > THUMB_MAX_WIDTH:
                fr = cv2.resize(fr, (THUMB_MAX_WIDTH, int(h * THUMB_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out, fr); return out
        except Exception: return ""

    # ── Page refresh ────────────────────────────────────────────────
    def _refresh(self):
        ci     = self.selected_collection_idx
        _, all_paths = self.video_collections[ci]
        disp   = self._filtered_indices
        n      = self.images_per_page
        total_d = len(disp)
        start  = self.page_index * n
        end    = min(start + n, total_d)
        total_p = max(1, (total_d + n - 1) // n)
        vname  = self.video_collections[ci][0]
        filt   = " [filtered]" if self._current_filter is not None else ""
        self.range_lbl.config(
            text=f"{vname}  |  {start+1}–{end} of {total_d}{filt}  (p.{self.page_index+1}/{total_p})")
        self._update_count()

        for slot in range(MAX_IMAGES_PER_PAGE):
            dp = start + slot
            cell, sim_lbl, img_lbl, cb = self.cells[slot]
            if slot < n and dp < total_d:
                fi   = disp[dp]
                path = all_paths[fi]
                thumb = self._make_thumb(path, ci, fi)
                if thumb and os.path.exists(thumb):
                    try:
                        ph = tk.PhotoImage(file=thumb)
                        self.photo_refs[slot] = ph
                        img_lbl.config(image=ph, text="")
                    except Exception:
                        img_lbl.config(image="", text=f"#{fi+1}")
                else:
                    img_lbl.config(image="", text=f"#{fi+1}")
                ginfo = self.frame_to_group.get(fi)
                if ginfo:
                    gid, sim = ginfo
                    if sim >= 99.9:
                        badge = f"#{fi+1}  [Rep G{gid+1}]"; bg = SIM_COLOR_REPR
                    else:
                        badge = f"#{fi+1}  {sim:.0f}%";     bg = sim_badge_color(sim)
                else:
                    badge = f"#{fi+1}"; bg = "light gray"
                sim_lbl.config(text=badge, bg=bg)
                self.chk_vars[slot].set(self.keep.get((ci, fi), False))
                cb.config(state="normal")
                self._cell_color(slot, fi, ci, bg)
            else:
                self.photo_refs[slot] = None
                for w in (cell, sim_lbl): w.config(bg="#f5f5f5")
                img_lbl.config(image="", text="", bg="#f5f5f5")
                self.chk_vars[slot].set(False)
                cb.config(state="disabled")

        self.prev_btn.config(state="normal" if self.page_index > 0 else "disabled")
        self.next_btn.config(state="normal" if end < total_d else "disabled")

    def _cell_color(self, slot, fi, ci, default_bg):
        cell, sim_lbl, img_lbl, _ = self.cells[slot]
        bg = "light green" if self.keep.get((ci, fi), False) else default_bg
        cell.config(bg=bg); sim_lbl.config(bg=bg)

    # ── Click / toggle ───────────────────────────────────────────────
    def _on_click(self, slot):
        ci   = self.selected_collection_idx
        disp = self._filtered_indices
        dp   = self.page_index * self.images_per_page + slot
        if dp >= len(disp): return
        fi   = disp[dp]
        self.keep[(ci, fi)] = not self.keep.get((ci, fi), False)
        self.chk_vars[slot].set(self.keep[(ci, fi)])
        ginfo = self.frame_to_group.get(fi)
        bg = (sim_badge_color(ginfo[1]) if ginfo and ginfo[1] < 99.9 else
              (SIM_COLOR_REPR if ginfo else "light gray"))
        self._cell_color(slot, fi, ci, bg)
        self._update_count()

    def _on_toggle(self):
        ci   = self.selected_collection_idx
        disp = self._filtered_indices
        start = self.page_index * self.images_per_page
        for slot in range(self.images_per_page):
            dp = start + slot
            if dp < len(disp):
                self.keep[(ci, disp[dp])] = self.chk_vars[slot].get()
        self._update_count()

    def _update_count(self):
        total_sel = sum(1 for v in self.keep.values() if v)
        total_all = sum(len(p) for _, p in self.video_collections)
        extra     = len(self.extra_image_paths)
        estr      = f"  +{extra} img" if extra else ""
        self.count_lbl.config(text=f"Selected: {total_sel}/{total_all} frames{estr}")
        self._update_size()

    def _update_size(self):
        sel = self._selected_paths() + self.extra_image_paths
        if not sel:
            self.size_lbl.config(text=""); return
        opt = self.opt_var.get()
        est = estimate_pdf_size_bytes(sel, optimize=opt)
        self.size_lbl.config(text=f"~{format_bytes(est)}" + (" opt" if opt else ""))

    def _selected_paths(self) -> List[str]:
        out = []
        for ci, (_, paths) in enumerate(self.video_collections):
            for fi in range(len(paths)):
                if self.keep.get((ci, fi), False):
                    out.append(paths[fi])
        return out

    # ── Pagination ───────────────────────────────────────────────────
    def _sync_page(self):
        ci    = self.selected_collection_idx
        disp  = self._filtered_indices
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
        total_p = max(1, (len(self._filtered_indices) + self.images_per_page - 1) // self.images_per_page)
        self.page_index = min(self.page_index + 1, total_p - 1)
        self._refresh()

    # ── Zoom ─────────────────────────────────────────────────────────
    def _on_rclick(self, slot):
        ci   = self.selected_collection_idx
        _, paths = self.video_collections[ci]
        disp = self._filtered_indices
        dp   = self.page_index * self.images_per_page + slot
        if dp >= len(disp): return
        if self._zoom_win and self._zoom_win.winfo_exists():
            self._zoom_win.destroy(); self._zoom_win = None; return
        fi   = disp[dp]
        path = paths[fi]
        try:
            fr = cv2.imread(path)
            if fr is None: return
            h, w = fr.shape[:2]
            if w > ZOOM_MAX_WIDTH:
                fr = cv2.resize(fr, (ZOOM_MAX_WIDTH, int(h * ZOOM_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
            zp = os.path.join(self._thumb_dir, "_zoom.png")
            cv2.imwrite(zp, fr)
            ginfo = self.frame_to_group.get(fi)
            extra = (f"G{ginfo[0]+1}, {ginfo[1]:.0f}% sim" if ginfo and ginfo[1] < 99.9 else
                     (f"G{ginfo[0]+1} Rep" if ginfo else "no data"))
            zw = tk.Toplevel(self.win)
            zw.title(f"Frame #{fi+1}  —  {extra}  —  right-click to close")
            zw.protocol("WM_DELETE_WINDOW", lambda: self._close_zoom(zw))
            ph = tk.PhotoImage(file=zp)
            self._zoom_ref = ph
            lbl = tk.Label(zw, image=ph, cursor="hand2"); lbl.pack(padx=2, pady=2)
            lbl.bind("<Button-3>", lambda e: self._close_zoom(zw))
            zw.bind("<Button-3>",  lambda e: self._close_zoom(zw))
            self._zoom_win = zw
        except Exception: pass

    def _close_zoom(self, zw):
        if zw.winfo_exists(): zw.destroy()
        self._zoom_win = None; self._zoom_ref = None

    # ── Extra images ─────────────────────────────────────────────────
    def _update_extra_label(self):
        n = len(self.extra_image_paths)
        if n == 0:
            self.extra_count_lbl.config(text="No extra images\n(added from main GUI\nor here)")
        else:
            self.extra_count_lbl.config(text=f"{n} image{'s' if n!=1 else ''} will\nbe appended to PDF")

    def _import_images(self):
        paths = filedialog.askopenfilenames(
            title="Select Images to Include",
            filetypes=[("Images", " ".join(IMAGE_EXTS)), ("All Files", "*.*")])
        if paths:
            added = 0
            for p in paths:
                if p not in self.extra_image_paths:
                    self.extra_image_paths.append(p); added += 1
            self._update_extra_label(); self._update_size()
            messagebox.showinfo("Images Added", f"Added {added} image(s).  Total: {len(self.extra_image_paths)}")

    def _clear_images(self):
        if not self.extra_image_paths:
            messagebox.showinfo("Info", "No extra images to clear."); return
        if messagebox.askyesno("Clear Images", f"Remove all {len(self.extra_image_paths)} imported images?"):
            self.extra_image_paths.clear()
            self._update_extra_label(); self._update_size()

    # ── Size info ─────────────────────────────────────────────────────
    def _size_info(self):
        sel = self._selected_paths() + self.extra_image_paths
        if not sel:
            messagebox.showinfo("Info", "Select at least one frame first."); return
        norm = estimate_pdf_size_bytes(sel, optimize=False)
        opt  = estimate_pdf_size_bytes(sel, optimize=True)
        messagebox.showinfo("PDF Size Info",
            f"Video frames: {len(self._selected_paths())}\n"
            f"Extra images: {len(self.extra_image_paths)}\n"
            f"Total pages:  {len(sel)}\n\n"
            f"Normal:    ~{format_bytes(norm)}\n"
            f"Optimized: ~{format_bytes(opt)}")

    # ── Create PDF ────────────────────────────────────────────────────
    def _create_pdf(self):
        self._sync_page()
        sel = self._selected_paths()
        all_images = sel + self.extra_image_paths
        if not all_images:
            messagebox.showerror("Error", "Select at least one frame or add images."); return
        doc_title = self.meta.get("title", "").strip() or "Untitled"
        out = filedialog.asksaveasfilename(
            title="Save PDF",
            initialfile=filename_for_windows(doc_title) + ".pdf",
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")])
        if not out: return
        if self.extra_image_paths and sel:
            msg = (f"PDF will contain:\n• {len(sel)} video frame(s)\n"
                   f"• {len(self.extra_image_paths)} extra image(s)\nTotal: {len(all_images)} pages\nProceed?")
            if not messagebox.askyesno("Confirm", msg): return
        def run():
            global last_output_pdf
            try:
                build_pdf_from_selected_images(all_images, out, self.meta,
                                                bar_height=BAR_HEIGHT_PTS, optimize=self.opt_var.get())
                last_output_pdf = out
                self.win.after(0, lambda: messagebox.showinfo("Done",
                    f"Saved: {os.path.basename(out)}\n({len(sel)} frames + {len(self.extra_image_paths)} images)"))
                try: ui_call(ui_enable_open_location, True)
                except Exception: pass
            except Exception as e:
                self.win.after(0, lambda: messagebox.showerror("Error", str(e)))
        threading.Thread(target=run, daemon=True).start()

    def _on_close(self):
        try: cleanup_dir(self.temp_dir)
        except Exception: pass
        self.win.destroy()
        try: ui_call(ui_enable_open_location, True)
        except Exception: pass


# =========================
# PDF merge helpers
# =========================
def ensure_pdf_merge_available():
    if PdfReader is None or PdfWriter is None:
        raise RuntimeError("Missing dependency: pypdf.  Install:  pip install pypdf")

def append_pdf_to_writer(writer, pdf_path):
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        writer.add_page(page)

def meta_to_pdf_metadata(meta):
    pm = {}
    if meta.get("title"):   pm["/Title"]   = meta["title"]
    if meta.get("author"):  pm["/Author"]  = meta["author"]
    if meta.get("tags"):    pm["/Keywords"]= meta["tags"]
    if meta.get("type"):    pm["/Type"]    = meta["type"]
    if meta.get("published"):
        s = meta["published"].strip().replace("-","").replace("/","")[:8]
        if len(s) == 8 and s.isdigit():
            pm["/CreationDate"] = f"D:{s}000000"
    if meta.get("language"):    pm["/Language"]   = meta["language"]
    if meta.get("cover_image"): pm["/CoverImage"] = meta["cover_image"]
    return pm

def add_metadata_to_pdf_file(pdf_path, meta):
    pm = meta_to_pdf_metadata(meta)
    if not pm: return
    ensure_pdf_merge_available()
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for page in reader.pages: writer.add_page(page)
    writer.add_metadata(pm)
    with open(pdf_path, "wb") as f: writer.write(f)


# =========================
# Queue + Main GUI helpers
# =========================
queue_items: List[QueueItem] = []


def refresh_queue_lb(select_index=None, yview_top=None):
    queue_lb.delete(0, tk.END)
    for it in queue_items:
        tag = "[VIDEO]" if it.kind == "video" else "[PDF]"
        queue_lb.insert(tk.END, f"{tag} {it.path}")
    if select_index is not None and 0 <= select_index < len(queue_items):
        queue_lb.selection_set(select_index)
        queue_lb.activate(select_index)
    if yview_top is not None:
        queue_lb.yview_moveto(yview_top)


def update_estimate_label():
    if not queue_items:
        ui_set_estimated_size("Estimated size: --"); return
    try:
        total_bytes, unknown = estimate_output_size_bytes()
        suffix = " (some videos unknown)" if unknown else ""
        ui_set_estimated_size(f"Estimated size: ~{format_bytes(total_bytes)}{suffix}")
    except Exception:
        ui_set_estimated_size("Estimated size: --")


def add_videos():
    paths = filedialog.askopenfilenames(title="Select Video Files",
        filetypes=[("Video Files", " ".join(VIDEO_EXTS))])
    for p in paths: queue_items.append(QueueItem("video", p))
    if paths:
        refresh_queue_lb(len(queue_items) - 1)
        update_estimate_label()

def add_pdfs():
    paths = filedialog.askopenfilenames(title="Select PDF Files",
        filetypes=[("PDF Files", "*.pdf")])
    for p in paths: queue_items.append(QueueItem("pdf", p))
    if paths:
        refresh_queue_lb(len(queue_items) - 1)
        update_estimate_label()

def add_images_main():
    paths = filedialog.askopenfilenames(title="Select Images to Include in PDF",
        filetypes=[("Image Files", " ".join(IMAGE_EXTS)), ("All Files", "*.*")])
    added = 0
    for p in paths:
        if p not in extra_images:
            extra_images.append(p); added += 1
    if added:
        img_count_lbl.config(text=f"📎 {len(extra_images)} image(s) queued")

def remove_selected():
    sel = queue_lb.curselection()
    if not sel: return
    idx = sel[0]
    queue_items.pop(idx)
    refresh_queue_lb(min(idx, len(queue_items) - 1))
    update_estimate_label()

def clear_list():
    queue_items.clear()
    refresh_queue_lb(None)
    update_estimate_label()

def reverse_queue():
    if queue_items:
        queue_items.reverse()
        refresh_queue_lb(None)
        update_estimate_label()

def move_selected(delta):
    if queue_lb.cget("state") == "disabled": return
    sel = queue_lb.curselection()
    if not sel: return
    i = sel[0]; j = i + delta
    if 0 <= j < len(queue_items):
        yv = queue_lb.yview()[0]
        queue_items[i], queue_items[j] = queue_items[j], queue_items[i]
        refresh_queue_lb(j, yview_top=yv)
        update_estimate_label()

def move_selected_with_keys(delta):
    def handler(event):
        if queue_lb.cget("state") == "disabled": return "break"
        if queue_lb.curselection(): move_selected(delta); return "break"
    return handler


# =========================
# Build combined PDF
# =========================
def build_combined_pdf():
    global conversion_thread, last_output_pdf
    if conversion_thread and conversion_thread.is_alive():
        ui_set_status("Build already running"); return
    if not queue_items:
        ui_set_status("Add at least one video or PDF"); return

    meta = {
        "title":      title_var.get().strip() or "Untitled",
        "author":     author_var.get().strip(),
        "tags":       tags_var.get().strip(),
        "type":       type_var.get().strip() or "PDF",
        "published":  published_var.get().strip(),
        "language":   language_var.get().strip(),
        "cover_image":cover_var.get().strip(),
    }
    out = filedialog.asksaveasfilename(
        title="Save Combined PDF",
        initialfile=filename_for_windows(meta["title"]) + ".pdf",
        defaultextension=".pdf",
        filetypes=[("PDF Files", "*.pdf")])
    if not out: return

    preset_name = preset_var.get()
    stop_event.clear()
    last_output_pdf = None
    ui_enable_open_location(False)
    ui_set_buttons_running(True)
    ui_set_status("Starting…"); ui_set_eta("ETA: --"); ui_set_progress(0, 1)

    def worker():
        global last_output_pdf
        try:
            ensure_pdf_merge_available()
            temp_root = make_temp_dir_near_output(out)
            writer    = PdfWriter()
            try:
                total = len(queue_items); done = 0
                for idx, item in enumerate(queue_items, 1):
                    if stop_event.is_set(): raise RuntimeError("Stopped")
                    base = f"Item {idx}/{total}"
                    if item.kind == "pdf":
                        ui_set_status(f"{base} - Adding: {os.path.basename(item.path)}")
                        ui_set_progress(done, total)
                        append_pdf_to_writer(writer, item.path)
                        done += 1; ui_set_progress(done, total)
                    else:
                        ui_set_status(f"{base} - Converting: {os.path.basename(item.path)}")
                        tmp = os.path.join(temp_root, f"__video_{idx:03d}.pdf")
                        def prog(sv, sm, et, st):
                            ui_set_status(f"{base} - {st}"); ui_set_eta(et)
                        video_to_temp_pdf(item.path, tmp, preset_name, meta, prog)
                        append_pdf_to_writer(writer, tmp)
                        done += 1; ui_set_progress(done, total)
                # Append extra images as pages
                if extra_images:
                    ui_set_status("Appending extra images…")
                    img_tmp = os.path.join(temp_root, "__extra_images.pdf")
                    build_pdf_from_selected_images(extra_images, img_tmp, meta, bar_height=0)
                    append_pdf_to_writer(writer, img_tmp)

                ui_set_status("Writing final PDF…")
                pm = meta_to_pdf_metadata(meta)
                if pm: writer.add_metadata(pm)
                try:
                    for page in writer.pages: page.compress_content_streams()
                except Exception: pass
                with open(out, "wb") as f: writer.write(f)
            finally:
                cleanup_dir(temp_root)
            last_output_pdf = out
            ui_set_status("Done"); ui_set_eta("ETA: 0:00"); ui_set_progress(1, 1)
            ui_enable_open_location(True); ui_popup_done(out)
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
        ui_set_status("Build already running"); return
    video_paths = [item.path for item in queue_items if item.kind == "video"]
    if not video_paths:
        messagebox.showwarning("No videos", "Add at least one video to the queue."); return

    meta = {
        "title":      title_var.get().strip() or "Untitled",
        "author":     author_var.get().strip(),
        "tags":       tags_var.get().strip(),
        "type":       type_var.get().strip() or "PDF",
        "published":  published_var.get().strip(),
        "language":   language_var.get().strip(),
        "cover_image":cover_var.get().strip(),
    }
    preset_name = preset_var.get()
    stop_event.clear()
    ui_set_buttons_running(True)
    ui_enable_open_location(False)

    temp_dir = tempfile.mkdtemp(prefix="video_to_pdf_select_")
    est = estimate_total_frames(video_paths, preset_name)
    ui_set_status(f"Extracting frames… (est. ~{est})")
    ui_set_eta("ETA: --"); ui_set_progress(0, 1)

    def worker():
        t0 = time.time()
        try:
            def prog(sv, sm, _eta, st):
                el  = time.time() - t0
                eta = format_seconds((sm - sv) / (sv / el)) if sv > 0 and el > 0.5 and sm > sv else "–"
                ui_set_status(f"{st}  —  {sv} of ~{sm}")
                ui_set_eta(f"ETA ~{eta}")
                ui_set_progress(sv, max(1, sm))

            all_paths, vc = extract_all_frames_from_videos(
                video_paths, preset_name, temp_dir, progress_cb=prog, estimated_total=est)
            if not all_paths:
                ui_call(lambda: messagebox.showerror("Error", "No frames extracted."))
                cleanup_dir(temp_dir); return

            ui_set_status("Opening selector with AI similarity analysis…")
            ui_set_eta(""); ui_set_progress(1, 1); ui_set_buttons_running(False)
            # Pass extra_images from main GUI into the selector
            ui_call(lambda: VideoCollectionFrameSelectionWindow(
                root, vc, temp_dir, meta, extra_image_paths=list(extra_images)))
        except RuntimeError as e:
            ui_set_status("Stopped" if str(e).lower().strip() == "stopped" else f"Error: {e}")
            cleanup_dir(temp_dir); ui_set_buttons_running(False)
        except Exception as e:
            ui_set_status(f"Error: {e}")
            ui_call(lambda: messagebox.showerror("Error", str(e)))
            cleanup_dir(temp_dir); ui_set_buttons_running(False)

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
# ═══ MAIN GUI ═══
# =========================
root = tk.Tk()
root.title("Video/PDF → Combined PDF  [AI Similarity Grouping]")
root.geometry("1000x800")

# ── Metadata fields ─────────────────────────────────────────────
mf = tk.LabelFrame(root, text="Document Metadata (optional)", font=("", 9, "bold"),
                    padx=6, pady=4)
mf.pack(fill="x", padx=10, pady=(8, 4))

fields = [
    ("Title",         "title_var",    ""),
    ("Author",        "author_var",   ""),
    ("Tags",          "tags_var",     ""),
    ("Type",          "type_var",     "PDF"),
    ("Published",     "published_var",""),
    ("Language",      "language_var", ""),
    ("Cover image",   "cover_var",    ""),
]
vars_dict = {}
for row_idx, (label, var_name, default) in enumerate(fields):
    tk.Label(mf, text=label + ":").grid(row=row_idx, column=0, sticky="w", pady=1)
    v = tk.StringVar(value=default)
    vars_dict[var_name] = v
    tk.Entry(mf, textvariable=v, width=55).grid(row=row_idx, column=1, padx=6, pady=1, sticky="w")
    if var_name == "published_var":
        tk.Label(mf, text="e.g. 2026-02-08", fg="gray", font=("",8)).grid(row=row_idx, column=2, sticky="w")

title_var     = vars_dict["title_var"]
author_var    = vars_dict["author_var"]
tags_var      = vars_dict["tags_var"]
type_var      = vars_dict["type_var"]
published_var = vars_dict["published_var"]
language_var  = vars_dict["language_var"]
cover_var     = vars_dict["cover_var"]

row_idx += 1
tk.Label(mf, text="Preset:").grid(row=row_idx, column=0, sticky="w", pady=1)
preset_var = tk.StringVar(value="3 sec (Original resolution)")
preset_combo = Combobox(mf, textvariable=preset_var, values=list(PRESETS.keys()),
                         state="readonly", width=38)
preset_combo.grid(row=row_idx, column=1, padx=6, pady=3, sticky="w")
preset_combo.bind("<<ComboboxSelected>>", lambda e: update_estimate_label())

# ── Main content area: Queue ────────────────────────────────────
content = tk.Frame(root)
content.pack(fill="both", expand=True, padx=10, pady=4)

qf = tk.LabelFrame(content, text="Video / PDF / Image Queue", font=("", 9, "bold"))
qf.pack(fill="both", expand=True)

queue_lb = tk.Listbox(qf, height=12, font=("", 9))
queue_lb.pack(fill="both", expand=True, padx=4, pady=4)
queue_lb.bind("<Up>",   move_selected_with_keys(-1))
queue_lb.bind("<Down>", move_selected_with_keys(1))

qbf = tk.Frame(qf)
qbf.pack(fill="x", padx=4, pady=(0, 4))
add_videos_btn = tk.Button(qbf, text="Add Videos",  width=14, command=add_videos)
add_videos_btn.pack(side="left", padx=2)
add_pdfs_btn   = tk.Button(qbf, text="Add PDFs",    width=12, command=add_pdfs)
add_pdfs_btn.pack(side="left", padx=2)
add_images_btn = tk.Button(qbf, text="Add Images",  width=12, command=add_images_main)
add_images_btn.pack(side="left", padx=2)
move_up_btn    = tk.Button(qbf, text="↑ Up",        width=6,  command=lambda: move_selected(-1))
move_up_btn.pack(side="left", padx=2)
move_down_btn  = tk.Button(qbf, text="↓ Down",      width=6,  command=lambda: move_selected(1))
move_down_btn.pack(side="left", padx=2)
remove_btn     = tk.Button(qbf, text="Remove",      width=8,  command=remove_selected)
remove_btn.pack(side="left", padx=2)
reverse_btn    = tk.Button(qbf, text="Reverse",     width=8,  command=reverse_queue)
reverse_btn.pack(side="left", padx=2)
clear_list_btn = tk.Button(qbf, text="Clear",       width=6,  command=clear_list)
clear_list_btn.pack(side="left", padx=2)

# Small label showing count of queued extra images
img_count_lbl = tk.Label(qf, text="", font=("", 8), fg="#555", anchor="w")
img_count_lbl.pack(anchor="w", padx=6, pady=(0, 2))

# ── Status bar ──────────────────────────────────────────────────
sf = tk.Frame(root, relief="sunken", bd=1)
sf.pack(fill="x", padx=10, pady=4)
status_label = tk.Label(sf, text="Idle", anchor="w")
status_label.pack(anchor="w", padx=4)
eta_label    = tk.Label(sf, text="ETA: --", anchor="w")
eta_label.pack(anchor="w", padx=4)
size_label   = tk.Label(sf, text="Estimated size: --", anchor="w")
size_label.pack(anchor="w", padx=4)
estimate_btn = tk.Button(sf, text="Update Estimate", command=update_estimate_label, width=16)
estimate_btn.pack(anchor="w", padx=4, pady=2)
progress_bar = Progressbar(sf, length=960, mode="determinate")
progress_bar.pack(padx=4, pady=4)

# ── Bottom action buttons ────────────────────────────────────────
bf = tk.Frame(root)
bf.pack(fill="x", padx=10, pady=6)
build_btn = tk.Button(bf, text="⚙  Generate Combined PDF", width=24, font=("",10,"bold"),
                       bg="#1a5276", fg="white", command=build_combined_pdf)
build_btn.pack(side="left", padx=6)
extract_select_btn = tk.Button(bf, text="🎞  Extract Frames & Select", width=24, font=("",10,"bold"),
                                bg="#1a6b2a", fg="white", command=extract_frames_and_select)
extract_select_btn.pack(side="left", padx=6)
tk.Button(bf, text="⏹ Stop", width=10, command=stop_build).pack(side="left", padx=6)
open_location_btn = tk.Button(bf, text="📂 Open Location", width=16,
                                state="disabled", command=open_file_location)
open_location_btn.pack(side="left", padx=6)
tk.Button(bf, text="Exit", fg="red", width=8, command=shutdown).pack(side="right", padx=6)

tk.Label(root,
         text="Tip: Use 'Add Images' to append standalone images to any PDF output.  "
              "In the frame selector, use ✨ Smart Select to auto-pick the last frame per group + all unique frames.  "
              "pip install pypdf  if merge fails.",
         font=("",8), wraplength=980, justify="left", fg="#555").pack(padx=10, pady=(0,6))

root.mainloop()