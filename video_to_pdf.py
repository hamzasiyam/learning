import cv2
import os
import re
import time
import tempfile
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar, Combobox

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# =========================
# Configuration / Presets
# =========================
PRESETS = {
    "8 sec (Smallest / Slides Only)": {"seconds": 8, "width": 900, "quality": 50},
    "10 sec (Lower Res)": {"seconds": 10, "width": 1024, "quality": 55},
    "Lecture (Recommended)": {"seconds": 15, "width": 1280, "quality": 60},
    "High Compression": {"seconds": 30, "width": 900, "quality": 45},  # reduced resolution
    "Detailed": {"seconds": 5, "width": 1600, "quality": 70},
}

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v")


# =========================
# Global state
# =========================
stop_event = threading.Event()
last_output_pdf = None
conversion_thread = None


# =========================
# Helpers
# =========================
def sanitize_filename(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "", text).strip()


def format_seconds(s: float) -> str:
    s = max(0, int(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:d}:{sec:02d}"


def ui_call(fn, *args, **kwargs):
    # Tkinter is not thread-safe; always schedule UI updates on the main thread.
    root.after(0, lambda: fn(*args, **kwargs))


def ui_set_status(text: str):
    ui_call(status_label.config, text=text)


def ui_set_eta(text: str):
    ui_call(eta_label.config, text=text)


def ui_set_progress(value: int, maximum: int):
    def _set():
        progress_bar["maximum"] = maximum
        progress_bar["value"] = value

    ui_call(_set)


def ui_enable_open_location(enabled: bool):
    ui_call(open_location_btn.config, state=("normal" if enabled else "disabled"))


def ui_popup_done(output_pdf: str):
    def _pop():
        messagebox.showinfo(
            "PDF Created",
            f"Your PDF has been created successfully:\n\n{os.path.basename(output_pdf)}",
        )

    ui_call(_pop)


def make_temp_dir_near_output(output_pdf: str) -> str:
    """
    Antivirus can slow down temp JPG writes heavily. To make exclusions easier,
    store temp frames in a folder next to the output PDF (user can exclude that folder).
    """
    base_dir = os.path.dirname(os.path.abspath(output_pdf))
    safe_tmp = os.path.join(base_dir, "_video_to_pdf_frames_tmp")
    os.makedirs(safe_tmp, exist_ok=True)
    # Use a unique subfolder per run
    return tempfile.mkdtemp(dir=safe_tmp)


def cleanup_dir(path: str):
    # Best-effort recursive cleanup
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


def write_jpeg(path: str, frame, quality: int) -> str:
    """
    Runs in a worker thread. OpenCV JPEG encoding is native code and typically
    releases the GIL, so a thread pool can help on multi-core systems.
    """
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return path


# =========================
# Core conversion
# =========================
def convert_video_to_pdf(video_path: str, output_pdf: str, preset_name: str, meta: dict):
    global last_output_pdf

    settings = PRESETS[preset_name]
    seconds_step = int(settings["seconds"])
    max_width = int(settings["width"])
    quality = int(settings["quality"])

    stop_event.clear()
    ui_set_status("Opening video...")
    ui_set_eta("ETA: --")
    ui_enable_open_location(False)
    ui_set_progress(0, 1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        ui_set_status("Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Compute duration (fallback if frame count missing)
    if fps > 0 and total_frames > 0:
        duration_sec = total_frames / fps
    else:
        duration_sec = cap.get(cv2.CAP_PROP_POS_MSEC)
        duration_sec = 0 if duration_sec is None else duration_sec / 1000.0

    if duration_sec <= 0 and total_frames > 0 and fps > 0:
        duration_sec = total_frames / fps

    # Target timestamps: 0, step, 2*step, ...
    # (frame jumping / timestamp seeking: faster than decoding every frame)
    if duration_sec > 0:
        targets = list(range(0, int(duration_sec) + 1, seconds_step))
    else:
        # Fallback: approximate 30 minutes if duration not known (still works, may stop early)
        targets = list(range(0, 60 * 60, seconds_step))

    ui_set_status("Extracting frames (jump-seek)...")
    ui_set_progress(0, len(targets))

    temp_dir = make_temp_dir_near_output(output_pdf)
    images = []
    start_time = time.time()

    # Thread pool for JPEG encoding/writes
    # (Not “decode in parallel” — decode stays single-stream; compression/writes can parallelize.)
    max_workers = max(2, min(8, (os.cpu_count() or 4)))
    futures = []

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            processed = 0

            for tsec in targets:
                if stop_event.is_set():
                    ui_set_status("Stopped")
                    return

                # Seek by time for faster skipping
                cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    # Likely hit end of video
                    break

                frame = resize_frame(frame, max_width=max_width)

                img_path = os.path.join(temp_dir, f"t{tsec:06d}.jpg")
                futures.append(pool.submit(write_jpeg, img_path, frame, quality))

                processed += 1
                ui_set_progress(processed, len(targets))

                # ETA estimation
                elapsed = time.time() - start_time
                if processed >= 2:
                    # Remaining based on original target count; more stable estimate
                    total_est = len(targets)
                    remaining = max(0, total_est - processed)
                    per_item = elapsed / processed
                    eta = remaining * per_item
                    ui_set_eta(f"ETA: {format_seconds(eta)}")
                else:
                    ui_set_eta("ETA: --")

            # Collect results (and stop early if requested)
            for f in as_completed(futures):
                if stop_event.is_set():
                    ui_set_status("Stopped")
                    return
                try:
                    images.append(f.result())
                except Exception:
                    # If one image fails to write, skip it
                    pass

    finally:
        cap.release()

    if stop_event.is_set():
        ui_set_status("Stopped")
        return

    # Keep images in chronological order
    images.sort()

    if not images:
        ui_set_status("No frames extracted")
        return

    # Build PDF (page size = image size, per page)
    ui_set_status("Building PDF...")
    ui_set_eta("ETA: --")

    try:
        c = canvas.Canvas(output_pdf)
        c.setTitle(meta["title"])
        c.setAuthor(meta["author"])
        c.setSubject(meta["subject"])

        for img_path in images:
            if stop_event.is_set():
                ui_set_status("Stopped")
                return
            img = ImageReader(img_path)
            w, h = img.getSize()
            c.setPageSize((w, h))
            c.drawImage(img, 0, 0, width=w, height=h)
            c.showPage()

        c.save()

    except Exception as e:
        ui_set_status(f"PDF build error: {e}")
        return
    finally:
        # Cleanup temp frames
        cleanup_dir(temp_dir)

    # Done
    last_output_pdf = output_pdf
    ui_set_status("Done")
    ui_set_eta("ETA: 0:00")
    ui_enable_open_location(True)
    ui_popup_done(output_pdf)


# =========================
# UI actions
# =========================
def select_video_and_run():
    global conversion_thread

    if conversion_thread and conversion_thread.is_alive():
        ui_set_status("Conversion already running")
        return

    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", " ".join(VIDEO_EXTS))],
    )
    if not video_path:
        return

    course = course_var.get().strip()
    lecture = lecture_var.get().strip()
    author = author_var.get().strip()

    if not course or not lecture:
        ui_set_status("Course and Lecture required")
        return

    safe_name = sanitize_filename(f"{course} - {lecture}") or "Lecture"
    output_pdf = filedialog.asksaveasfilename(
        title="Save PDF",
        initialfile=f"{safe_name}.pdf",
        defaultextension=".pdf",
        filetypes=[("PDF Files", "*.pdf")],
    )
    if not output_pdf:
        return

    meta = {
        "title": f"{course} – {lecture}",
        "author": author if author else "Instructor",
        "subject": course,
    }

    preset_name = preset_var.get()

    # Start conversion thread
    stop_event.clear()
    conversion_thread = threading.Thread(
        target=convert_video_to_pdf,
        args=(video_path, output_pdf, preset_name, meta),
        daemon=True,
    )
    conversion_thread.start()


def clear_for_new_video():
    global last_output_pdf
    stop_event.set()
    last_output_pdf = None
    ui_set_status("Idle")
    ui_set_eta("ETA: --")
    ui_set_progress(0, 1)
    ui_enable_open_location(False)


def shutdown():
    stop_event.set()
    root.after(200, root.destroy)


def open_file_location():
    if last_output_pdf and os.path.exists(last_output_pdf):
        subprocess.run(
            ["explorer", "/select,", os.path.normpath(last_output_pdf)],
            shell=True,
        )


# =========================
# GUI
# =========================
root = tk.Tk()
root.title("Lecture Video → PDF (Fast, Optimized)")
root.geometry("620x500")

tk.Label(root, text="Course Title").pack()
course_var = tk.StringVar()
tk.Entry(root, textvariable=course_var, width=70).pack(pady=2)

tk.Label(root, text="Lecture Title").pack()
lecture_var = tk.StringVar()
tk.Entry(root, textvariable=lecture_var, width=70).pack(pady=2)

tk.Label(root, text="Author / Instructor").pack()
author_var = tk.StringVar()
tk.Entry(root, textvariable=author_var, width=70).pack(pady=2)

tk.Label(root, text="Compression Preset").pack()
preset_var = tk.StringVar(value="Lecture (Recommended)")
Combobox(
    root,
    textvariable=preset_var,
    values=list(PRESETS.keys()),
    state="readonly",
    width=40,
).pack(pady=4)

tk.Button(root, text="Select Video File", command=select_video_and_run).pack(pady=10)

status_label = tk.Label(root, text="Idle")
status_label.pack(pady=6)

eta_label = tk.Label(root, text="ETA: --")
eta_label.pack(pady=2)

progress_bar = Progressbar(root, length=560, mode="determinate")
progress_bar.pack(pady=10)

# Buttons row
btns = tk.Frame(root)
btns.pack(pady=12)

tk.Button(btns, text="Clear / New Video", command=clear_for_new_video).pack(side="left", padx=8)

open_location_btn = tk.Button(
    btns,
    text="Open File Location",
    command=open_file_location,
    state="disabled",
)
open_location_btn.pack(side="left", padx=8)

tk.Button(btns, text="Exit / Stop All", command=shutdown, fg="red").pack(side="left", padx=8)

# Note for performance (antivirus)
note = (
    "Performance note: temp frames are written next to your output PDF in "
    "'_video_to_pdf_frames_tmp'. Excluding that folder from antivirus can speed up processing."
)
tk.Label(root, text=note, wraplength=590, justify="left").pack(pady=10)

root.mainloop()
