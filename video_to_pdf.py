import os
import re
import time
import tempfile
import threading
import subprocess
from dataclasses import dataclass
from typing import List, Optional

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
PRESETS = {
    "8 sec (Smallest / Slides Only)": {"seconds": 8, "width": 900, "quality": 50},
    "10 sec (Lower Res)": {"seconds": 10, "width": 1024, "quality": 55},
    "Lecture (Recommended)": {"seconds": 15, "width": 1280, "quality": 60},
    "High Compression": {"seconds": 30, "width": 900, "quality": 45},
    "Detailed": {"seconds": 5, "width": 1600, "quality": 70},
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
    root.after(0, lambda: fn(*args, **kwargs))


def ui_set_status(text: str):
    ui_call(status_label.config, text=text)


def ui_set_eta(text: str):
    ui_call(eta_label.config, text=text)


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
    ui_call(build_btn.config, state=state)
    ui_call(preset_combo.config, state=("disabled" if running else "readonly"))
    ui_call(listbox.config, state=state)


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
        c.setSubject(meta.get("subject", ""))

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
# PDF merge helpers
# =========================
def ensure_pdf_merge_available():
    if PdfReader is None or PdfWriter is None:
        raise RuntimeError("Missing dependency: pypdf. Install with: pip install pypdf")


def append_pdf_to_writer(writer: "PdfWriter", pdf_path: str):
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        writer.add_page(page)


# =========================
# Queue + UI list
# =========================
queue_items: List[QueueItem] = []


def refresh_listbox(select_index: Optional[int] = None):
    listbox.delete(0, tk.END)
    for it in queue_items:
        tag = "[VIDEO]" if it.kind == "video" else "[PDF]"
        listbox.insert(tk.END, f"{tag} {it.path}")
    if select_index is not None and 0 <= select_index < len(queue_items):
        listbox.selection_set(select_index)
        listbox.activate(select_index)


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


def remove_selected():
    sel = listbox.curselection()
    if not sel:
        return
    idx = sel[0]
    queue_items.pop(idx)
    refresh_listbox(min(idx, len(queue_items) - 1))


def clear_list():
    queue_items.clear()
    refresh_listbox(None)


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
    queue_items[i], queue_items[j] = queue_items[j], queue_items[i]
    refresh_listbox(j)


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

    course = course_var.get().strip()
    lecture = lecture_var.get().strip()
    author = author_var.get().strip()
    if not course or not lecture:
        ui_set_status("Course and Lecture required")
        return

    safe_name = sanitize_filename(f"{course} - {lecture}") or "Combined"
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
        "title": f"{course} – {lecture}",
        "author": author if author else "Instructor",
        "subject": course,
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
root.geometry("820x620")

# Meta
meta_frame = tk.Frame(root)
meta_frame.pack(fill="x", padx=10, pady=8)

tk.Label(meta_frame, text="Course Title").grid(row=0, column=0, sticky="w")
course_var = tk.StringVar()
tk.Entry(meta_frame, textvariable=course_var, width=60).grid(row=0, column=1, padx=6, pady=2, sticky="w")

tk.Label(meta_frame, text="Lecture Title").grid(row=1, column=0, sticky="w")
lecture_var = tk.StringVar()
tk.Entry(meta_frame, textvariable=lecture_var, width=60).grid(row=1, column=1, padx=6, pady=2, sticky="w")

tk.Label(meta_frame, text="Author / Instructor").grid(row=2, column=0, sticky="w")
author_var = tk.StringVar()
tk.Entry(meta_frame, textvariable=author_var, width=60).grid(row=2, column=1, padx=6, pady=2, sticky="w")

tk.Label(meta_frame, text="Compression Preset").grid(row=3, column=0, sticky="w")
preset_var = tk.StringVar(value="Lecture (Recommended)")
preset_combo = Combobox(
    meta_frame,
    textvariable=preset_var,
    values=list(PRESETS.keys()),
    state="readonly",
    width=40,
)
preset_combo.grid(row=3, column=1, padx=6, pady=4, sticky="w")

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

clear_list_btn = tk.Button(right, text="Clear List", width=18, command=clear_list)
clear_list_btn.pack(pady=4)

# Status + progress
status_frame = tk.Frame(root)
status_frame.pack(fill="x", padx=10, pady=8)

status_label = tk.Label(status_frame, text="Idle")
status_label.pack(anchor="w")

eta_label = tk.Label(status_frame, text="ETA: --")
eta_label.pack(anchor="w")

progress_bar = Progressbar(status_frame, length=780, mode="determinate")
progress_bar.pack(pady=8)

# Bottom buttons
bottom = tk.Frame(root)
bottom.pack(fill="x", padx=10, pady=10)

build_btn = tk.Button(bottom, text="Generate Combined PDF", command=build_combined_pdf, width=22)
build_btn.pack(side="left", padx=6)

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