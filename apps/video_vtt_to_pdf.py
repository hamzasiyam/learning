"""Standalone GUI: extract VTT-timestamp frames and build a subtitle PDF."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from typing import List, Tuple

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Combobox, Progressbar

from modules.core import config as cfg
from modules.core import file_utils as futils
from modules.core import pdf_utils as pdfm
from modules.core import subtitles as subm

PAGE_NUMBER_BAR_PTS = 64
DEFAULT_MAX_WIDTH = int(cfg.PRESETS["Every 8 seconds — sharp text"]["width"])
DEFAULT_JPEG_QUALITY = 95
PAGES_PER_SPLIT = 5
BALANCED_PDF_MAX_WIDTH = 1400
BALANCED_PDF_QUALITY = 72


def _resize_frame(frame, max_width: int):
    """Resize frame to a max width while preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS for display in subtitle icon area."""
    total = max(0, int(round(seconds)))
    hh, rem = divmod(total, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


class VttTimestampPdfApp(tk.Tk):
    """Small GUI for creating a PDF from video frames at VTT cue starts."""

    def __init__(self):
        super().__init__()
        self.title("Video + VTT Timestamp Frames to PDF")
        self.geometry("780x360")
        self.minsize(760, 330)

        self.video_var = tk.StringVar()
        self.vtt_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.frames_per_row_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="Choose a video and a VTT file.")

        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.last_output_pdf: str | None = None

        self._build_ui()

    def _build_ui(self):
        outer = tk.Frame(self, padx=12, pady=12)
        outer.pack(fill="both", expand=True)

        tk.Label(
            outer,
            text="Create PDF from transcript timestamps",
            font=("", 13, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(
            outer,
            text=(
                "Extract one frame at each VTT cue start timestamp and build a PDF using:\n"
                "12 images per page, with selectable 1 or 2 images per row.\n"
                "Output is saved in a dedicated folder, split every 5 pages."
            ),
            justify="left",
            fg="#555",
        ).pack(anchor="w", pady=(0, 10))

        self._path_row(outer, "Video file:", self.video_var, self._pick_video)
        self._path_row(outer, "Transcript file:", self.vtt_var, self._pick_vtt)
        self._path_row(outer, "Output PDF:", self.output_var, self._pick_output_pdf)
        layout_row = tk.Frame(outer)
        layout_row.pack(anchor="w", pady=(4, 6))
        tk.Label(layout_row, text="Frames per row:", width=14, anchor="w").pack(side="left")
        per_row_combo = Combobox(
            layout_row,
            textvariable=self.frames_per_row_var,
            values=["1", "2"],
            state="readonly",
            width=4,
        )
        per_row_combo.pack(side="left")

        controls = tk.Frame(outer)
        controls.pack(fill="x", pady=(10, 4))

        self.create_btn = tk.Button(
            controls,
            text="Create PDF",
            width=20,
            command=self._start_create,
            bg="#1565c0",
            fg="white",
        )
        self.create_btn.pack(side="left")

        self.stop_btn = tk.Button(
            controls,
            text="Stop",
            width=12,
            command=self._stop_current,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=(8, 0))

        self.open_btn = tk.Button(
            controls,
            text="Open Output Location",
            width=20,
            command=self._open_output_location,
            state="disabled",
        )
        self.open_btn.pack(side="left", padx=(8, 0))

        self.clear_btn = tk.Button(
            controls,
            text="Clear All",
            width=12,
            command=self._clear_all,
        )
        self.clear_btn.pack(side="left", padx=(8, 0))

        tk.Button(
            controls,
            text="Exit",
            width=12,
            command=self.destroy,
        ).pack(side="right")

        self.progress = Progressbar(outer, mode="determinate", length=740)
        self.progress.pack(fill="x", pady=(10, 4))

        tk.Label(
            outer,
            textvariable=self.status_var,
            justify="left",
            anchor="w",
            fg="#444",
        ).pack(fill="x")

    def _path_row(self, parent, label: str, var: tk.StringVar, browse_cmd):
        row = tk.Frame(parent)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, width=14, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(row, text="Browse...", width=12, command=browse_cmd).pack(side="left")

    def _pick_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All Files", "*.*")],
        )
        if not path:
            return
        self.video_var.set(path)
        if not self.output_var.get().strip():
            base = os.path.splitext(os.path.basename(path))[0]
            default_name = futils.safe_filename(base) + "_vtt_frames.pdf"
            self.output_var.set(os.path.join(os.path.dirname(path), default_name))

    def _pick_vtt(self):
        path = filedialog.askopenfilename(
            title="Select Transcript File",
            filetypes=[
                ("Transcript Files", "*.vtt *.webvtt *.srt *.txt"),
                ("WebVTT", "*.vtt *.webvtt"),
                ("SubRip", "*.srt"),
                ("Text", "*.txt"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.vtt_var.set(path)

    def _pick_output_pdf(self):
        current = self.output_var.get().strip()
        initial_dir = os.path.dirname(current) if current else os.getcwd()
        initial_name = os.path.basename(current) if current else "timestamp_frames.pdf"
        path = filedialog.asksaveasfilename(
            title="Save PDF As",
            initialdir=initial_dir,
            initialfile=initial_name,
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
        )
        if path:
            self.output_var.set(path)

    def _set_running(self, running: bool):
        self.create_btn.config(state="disabled" if running else "normal")
        self.clear_btn.config(state="disabled" if running else "normal")
        self.stop_btn.config(state="normal" if running else "disabled")

    def _reset_progress(self):
        self.progress["maximum"] = 1
        self.progress["value"] = 0

    def _update_progress(self, value: int, maximum: int, status: str):
        def _apply():
            self.progress["maximum"] = max(1, int(maximum))
            self.progress["value"] = max(0, int(value))
            self.status_var.set(status)

        self.after(0, _apply)

    def _start_create(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        video_path = self.video_var.get().strip()
        vtt_path = self.vtt_var.get().strip()
        output_pdf = self.output_var.get().strip()

        if not video_path or not os.path.isfile(video_path):
            messagebox.showerror("Missing Video", "Please select a valid video file.")
            return
        if not vtt_path or not os.path.isfile(vtt_path):
            messagebox.showerror("Missing Transcript", "Please select a valid transcript file (.vtt, .webvtt, .srt, .txt).")
            return
        if not output_pdf:
            messagebox.showerror("Missing Output", "Please choose where to save the output PDF.")
            return

        self._set_running(True)
        self.stop_event.clear()
        self.open_btn.config(state="disabled")
        self.last_output_pdf = None
        self._update_progress(0, 1, "Loading transcript cues...")

        def worker():
            try:
                cues = subm.load_subtitle_file(vtt_path)
                if not cues:
                    raise RuntimeError("No subtitle cues were found in the transcript file.")

                base_parent = os.path.dirname(os.path.abspath(output_pdf)) or os.getcwd()
                os.makedirs(base_parent, exist_ok=True)
                video_base_name = futils.safe_filename(os.path.splitext(os.path.basename(video_path))[0]) or "video"
                run_dir_base = os.path.join(base_parent, video_base_name)
                run_dir = futils.make_unique_output_dir(run_dir_base)
                os.makedirs(run_dir, exist_ok=True)

                temp_frames_root = os.path.join(run_dir, "_temp_frames")
                os.makedirs(temp_frames_root, exist_ok=True)
                frames_dir = tempfile.mkdtemp(prefix="vtt_frames_", dir=temp_frames_root)

                frame_paths: List[str] = []
                subtitle_texts: List[str] = []
                subtitle_timestamps: List[str] = []
                skipped: List[Tuple[int, float]] = []
                combined_pdf_path = os.path.join(run_dir, "__combined_unsplit.pdf")
                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        raise RuntimeError(f"Could not open video: {os.path.basename(video_path)}")

                    try:
                        total = len(cues)
                        for idx, cue in enumerate(cues, start=1):
                            if self.stop_event.is_set():
                                raise RuntimeError("Stopped")
                            self._update_progress(
                                idx - 1,
                                total + 1,
                                f"Extracting frame {idx} of {total} at {cue.start:.3f}s",
                            )
                            cap.set(cv2.CAP_PROP_POS_MSEC, float(cue.start) * 1000.0)
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                skipped.append((idx, cue.start))
                                continue
                            frame = _resize_frame(frame, DEFAULT_MAX_WIDTH)
                            out_name = f"t{int(round(cue.start * 1000)):09d}_{idx:04d}.jpg"
                            out_path = os.path.join(frames_dir, out_name)
                            ok = cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_JPEG_QUALITY])
                            if not ok:
                                skipped.append((idx, cue.start))
                                continue
                            frame_paths.append(out_path)
                            subtitle_texts.append(cue.text.strip())
                            subtitle_timestamps.append(_format_timestamp(cue.start))
                    finally:
                        cap.release()

                    if not frame_paths:
                        raise RuntimeError("No frames could be extracted from the provided cue timestamps.")
                    if self.stop_event.is_set():
                        raise RuntimeError("Stopped")

                    title = os.path.splitext(os.path.basename(video_path))[0]
                    meta = {"title": title}
                    source_titles = [title] * len(frame_paths)
                    self._update_progress(
                        len(cues),
                        len(cues) + 1,
                        f"Building and splitting PDF from {len(frame_paths)} extracted frames...",
                    )

                    def on_pdf_progress(step, total, msg):
                        if self.stop_event.is_set():
                            raise RuntimeError("Stopped")
                        self._update_progress(
                            len(cues) + min(step, max(1, total)),
                            len(cues) + max(1, total) + 1,
                            msg,
                        )

                    pdfm.build_pdf_from_images(
                        frame_paths,
                        combined_pdf_path,
                        meta=meta,
                        bar_height=PAGE_NUMBER_BAR_PTS,
                        optimize=True,
                        optimize_max_width=BALANCED_PDF_MAX_WIDTH,
                        optimize_quality=BALANCED_PDF_QUALITY,
                        total_pages=len(frame_paths),
                        columns=12,
                        per_row=2 if self.frames_per_row_var.get().strip() == "2" else 1,
                        image_source_titles=source_titles,
                        subtitle_texts=subtitle_texts,
                        subtitle_timestamps=subtitle_timestamps,
                        include_subtitles=True,
                        include_subtitles_as_text=True,
                        progress_callback=on_pdf_progress,
                    )
                    if self.stop_event.is_set():
                        raise RuntimeError("Stopped")

                    split_target = os.path.join(run_dir, f"{video_base_name}.pdf")
                    outputs = pdfm.split_existing_pdf_by_page_count(
                        combined_pdf_path,
                        split_target,
                        pages_per_part=PAGES_PER_SPLIT,
                        metadata=pdfm.meta_to_pdf_metadata(meta),
                    )
                    existing_outputs = [path for path in outputs if os.path.exists(path)]
                    if not existing_outputs:
                        shutil.copy2(combined_pdf_path, split_target)
                        existing_outputs = [split_target]
                    renamed_outputs: List[str] = []
                    for idx, old_path in enumerate(sorted(existing_outputs), 1):
                        new_name = f"{video_base_name} Part {idx}.pdf"
                        new_path = os.path.join(run_dir, new_name)
                        if os.path.abspath(old_path) != os.path.abspath(new_path):
                            if os.path.exists(new_path):
                                os.remove(new_path)
                            os.replace(old_path, new_path)
                        renamed_outputs.append(new_path)
                    existing_outputs = renamed_outputs
                finally:
                    futils.cleanup_dir(temp_frames_root)
                    try:
                        os.remove(combined_pdf_path)
                    except Exception:
                        pass

                self.last_output_pdf = existing_outputs[0]
                success_msg = (
                    f"PDF(s) created successfully.\n\n"
                    f"Output folder:\n{run_dir}\n\n"
                    f"PDF part(s): {len(existing_outputs)} ({PAGES_PER_SPLIT} pages each, except final part)\n"
                    f"Used {len(frame_paths)} frame(s) from transcript timestamps."
                )
                if skipped:
                    success_msg += f"\n\nSkipped {len(skipped)} cue timestamp(s) that could not be decoded."

                self.after(0, lambda: messagebox.showinfo("Done", success_msg))
                self.after(0, lambda: self.open_btn.config(state="normal"))
                self._update_progress(1, 1, "Finished.")
            except Exception as exc:
                msg = str(exc).strip()
                if msg.lower() == "stopped":
                    self._update_progress(0, 1, "Stopped.")
                    self.after(0, lambda: messagebox.showinfo("Stopped", "Current operation was stopped."))
                else:
                    err_text = str(exc)
                    self._update_progress(0, 1, f"Error: {err_text}")
                    self.after(0, lambda text=err_text: messagebox.showerror("Error", text))
            finally:
                self.after(0, lambda: self._set_running(False))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _open_output_location(self):
        if self.last_output_pdf and os.path.exists(self.last_output_pdf):
            subprocess.run(["explorer", "/select,", os.path.normpath(self.last_output_pdf)])

    def _stop_current(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.status_var.set("Stopping... please wait.")

    def _clear_all(self):
        """Reset all inputs and progress so a new run can start quickly."""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.video_var.set("")
        self.vtt_var.set("")
        self.output_var.set("")
        self.last_output_pdf = None
        self.open_btn.config(state="disabled")
        self._reset_progress()
        self.status_var.set("Cleared. Choose a video and a VTT file.")


def run():
    """Launch the standalone VTT timestamp frame app."""
    app = VttTimestampPdfApp()
    app.mainloop()


if __name__ == "__main__":
    run()

