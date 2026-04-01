"""
Folder Watch -> Images to PDF (GUI) with size optimization + size estimate

Changes in this version
- Much smaller PDFs without large quality loss (defaults: 200 DPI cap + JPEG quality 85 + 4:2:0 subsampling)
- Shows an "Estimated PDF size" BEFORE compiling (preflights by processing images to temp and summing output bytes)
- Estimate auto-updates when images/options change (runs in a background thread so UI stays responsive)

Dependencies:
  pip install pillow reportlab
"""

import os
import sys
import time
import math
import tempfile
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def safe_abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def open_file_location_select(file_path: str) -> None:
    file_path = safe_abspath(file_path)

    if sys.platform.startswith("win"):
        subprocess.run(["explorer", "/select,", file_path], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", "-R", file_path], check=False)
    else:
        folder = os.path.dirname(file_path)
        subprocess.run(["xdg-open", folder], check=False)


def wait_until_stable(path: str, timeout_s: float = 10.0, poll_s: float = 0.25) -> bool:
    end = time.time() + timeout_s
    last_size = -1
    stable_count = 0

    while time.time() < end:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1

        if size == last_size and size > 0:
            stable_count += 1
            if stable_count >= 3:
                return True
        else:
            stable_count = 0
            last_size = size

        time.sleep(poll_s)

    return False


def human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KB", "MB", "GB", "TB"]
    n = float(num_bytes)
    for u in units:
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.2f} {u}"
    return f"{n:.2f} PB"


def preset_to_params(preset: str):
    """
    Returns (max_dpi_cap, jpeg_quality, subsampling)
    subsampling: Pillow JPEG subsampling:
      0 = 4:4:4 (best, biggest)
      1 = 4:2:2
      2 = 4:2:0 (smallest, typically fine for photos)
    """
    preset = (preset or "").lower()
    if preset == "high quality":
        return (300, 88, 1)
    if preset == "balanced":
        return (200, 85, 2)
    if preset == "small":
        return (170, 82, 2)
    # default
    return (200, 85, 2)


def preprocess_image_for_pdf(
    src_path: str,
    temp_dir: str,
    max_dpi_cap: int,
    jpeg_quality: int,
    subsampling: int,
) -> str:
    """
    Prepare image for embedding:
    - EXIF rotate
    - Flatten transparency onto white
    - Convert to RGB
    - Downscale to letter size at max_dpi_cap (e.g., 200 DPI => 1700x2200)
    - Save optimized JPEG
    """
    # letter inches
    letter_w_in, letter_h_in = 8.5, 11.0
    max_w = int(round(letter_w_in * max_dpi_cap))
    max_h = int(round(letter_h_in * max_dpi_cap))

    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im)

        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            if im.mode != "RGBA":
                im = im.convert("RGBA")
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        else:
            if im.mode != "RGB":
                im = im.convert("RGB")

        w, h = im.size
        scale = min(max_w / w, max_h / h, 1.0)  # never upscale
        if scale < 1.0:
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            im = im.resize(new_size, Image.Resampling.LANCZOS)

        out_name = os.path.splitext(os.path.basename(src_path))[0] + ".jpg"
        out_path = os.path.join(temp_dir, out_name)

        im.save(
            out_path,
            "JPEG",
            quality=int(jpeg_quality),
            optimize=True,
            progressive=True,
            subsampling=int(subsampling),
        )

    return out_path


class DoneDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, pdf_path: str, pdf_bytes: int):
        super().__init__(parent)
        self.title("PDF Created")
        self.pdf_path = pdf_path
        self.resizable(False, False)

        frm = ttk.Frame(self, padding=14)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="PDF successfully created:").grid(row=0, column=0, sticky="w")
        ttk.Label(frm, text=pdf_path, wraplength=560).grid(row=1, column=0, sticky="w", pady=(6, 2))
        ttk.Label(frm, text=f"Final size: {human_size(pdf_bytes)}").grid(row=2, column=0, sticky="w", pady=(0, 12))

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, sticky="e")

        ttk.Button(btns, text="Open file location", command=self.open_location).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="Close", command=self.destroy).grid(row=0, column=1)

        self.transient(parent)
        self.grab_set()
        self.update_idletasks()

        # Center over parent
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        w, h = self.winfo_width(), self.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"+{x}+{y}")

    def open_location(self):
        open_file_location_select(self.pdf_path)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Folder Watch → Images to PDF")
        self.minsize(900, 560)

        # State
        self.watch_folder = ""
        self.baseline_files = set()
        self.running = False
        self.known_new_files = []
        self.known_new_set = set()
        self.scan_job = None

        # Estimate state
        self._estimate_job = None          # debounce handle
        self._estimate_thread = None
        self._estimate_cancel = threading.Event()
        self._estimate_lock = threading.Lock()
        self._estimate_cache = {}          # key -> processed_bytes (JPEG output)
        self._last_estimate_bytes = 0

        # UI vars
        self.folder_var = tk.StringVar(value="No folder selected")
        self.status_var = tk.StringVar(value="Select a folder to watch.")
        self.estimate_var = tk.StringVar(value="Estimated PDF size: —")
        self.preset_var = tk.StringVar(value="Balanced")  # High Quality / Balanced / Small
        self.fit_to_page_var = tk.BooleanVar(value=True)
        self.avoid_upscale_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._style_defaults()
        self._bind_option_traces()

    def _style_defaults(self):
        try:
            self.tk.call("tk", "scaling", 1.15)
        except Exception:
            pass

    def _bind_option_traces(self):
        # Update estimate when options change
        self.preset_var.trace_add("write", lambda *_: self.schedule_estimate())
        self.fit_to_page_var.trace_add("write", lambda *_: self.schedule_estimate())
        self.avoid_upscale_var.trace_add("write", lambda *_: self.schedule_estimate())

    def _build_ui(self):
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        top = ttk.LabelFrame(outer, text="Watch Folder", padding=10)
        top.pack(fill="x", padx=2, pady=(0, 10))

        row0 = ttk.Frame(top)
        row0.pack(fill="x")

        ttk.Button(row0, text="Choose folder…", command=self.choose_folder).pack(side="left")
        ttk.Label(row0, textvariable=self.folder_var).pack(side="left", padx=10, fill="x", expand=True)

        row1 = ttk.Frame(top)
        row1.pack(fill="x", pady=(10, 0))

        self.start_btn = ttk.Button(row1, text="Start", command=self.start_watching, state="disabled")
        self.start_btn.pack(side="left")

        self.stop_btn = ttk.Button(row1, text="Stop", command=self.stop_watching, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))

        ttk.Label(row1, textvariable=self.status_var).pack(side="left", padx=12, fill="x", expand=True)

        mid = ttk.LabelFrame(outer, text="Images detected after Start (remove any before compiling)", padding=10)
        mid.pack(fill="both", expand=True, padx=2, pady=(0, 10))

        mid_grid = ttk.Frame(mid)
        mid_grid.pack(fill="both", expand=True)

        mid_grid.columnconfigure(0, weight=1)
        mid_grid.rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(mid_grid, height=12, activestyle="none")
        self.listbox.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(mid_grid, orient="vertical", command=self.listbox.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=sb.set)

        actions = ttk.Frame(mid)
        actions.pack(fill="x", pady=(10, 0))

        ttk.Button(actions, text="Remove selected", command=self.remove_selected).pack(side="left")
        ttk.Button(actions, text="Clear list", command=self.clear_list).pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.estimate_var).pack(side="right")

        bottom = ttk.LabelFrame(outer, text="PDF Options", padding=10)
        bottom.pack(fill="x", padx=2)

        opts = ttk.Frame(bottom)
        opts.pack(fill="x")

        ttk.Label(opts, text="Compression preset:").pack(side="left")
        preset = ttk.Combobox(
            opts,
            textvariable=self.preset_var,
            values=["High Quality", "Balanced", "Small"],
            state="readonly",
            width=14,
        )
        preset.pack(side="left", padx=(8, 14))

        ttk.Checkbutton(opts, text="Fit image within page margins", variable=self.fit_to_page_var).pack(side="left")
        ttk.Checkbutton(opts, text="Avoid upscaling small images (reduces blur)", variable=self.avoid_upscale_var).pack(side="left", padx=14)

        compile_row = ttk.Frame(bottom)
        compile_row.pack(fill="x", pady=(10, 0))

        self.compile_btn = ttk.Button(compile_row, text="Compile PDF…", command=self.compile_pdf, state="disabled")
        self.compile_btn.pack(side="left")

        ttk.Button(compile_row, text="Refresh estimate", command=self.run_estimate_now).pack(side="left", padx=(8, 0))

        ttk.Label(compile_row, text="One image per page • Letter size").pack(side="left", padx=12)

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select folder to watch")
        if not folder:
            return
        self.watch_folder = safe_abspath(folder)
        self.folder_var.set(self.watch_folder)
        self.status_var.set("Ready. Click Start to begin watching (only new images after Start are included).")
        self.start_btn.configure(state="normal")

    def start_watching(self):
        if not self.watch_folder or not os.path.isdir(self.watch_folder):
            messagebox.showerror("Folder not found", "Please choose a valid folder to watch.")
            return

        try:
            existing = {
                safe_abspath(os.path.join(self.watch_folder, f))
                for f in os.listdir(self.watch_folder)
                if is_image_name(f)
            }
        except Exception as e:
            messagebox.showerror("Error", f"Could not read folder:\n{e}")
            return

        self.baseline_files = existing
        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.compile_btn.configure(state="normal")
        self.status_var.set("Watching… Add images to the folder now.")
        self._schedule_scan()

    def stop_watching(self):
        self.running = False
        if self.scan_job is not None:
            try:
                self.after_cancel(self.scan_job)
            except Exception:
                pass
            self.scan_job = None

        self.start_btn.configure(state="normal" if self.watch_folder else "disabled")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Stopped. Click Start to watch again (baseline resets).")

    def _schedule_scan(self):
        if not self.running:
            return
        self.scan_folder_once()
        self.scan_job = self.after(900, self._schedule_scan)

    def scan_folder_once(self):
        if not self.watch_folder:
            return

        try:
            names = os.listdir(self.watch_folder)
        except Exception:
            return

        added_any = False

        for name in names:
            if not is_image_name(name):
                continue
            full = safe_abspath(os.path.join(self.watch_folder, name))

            if full in self.baseline_files or full in self.known_new_set:
                continue

            if not wait_until_stable(full, timeout_s=6.0, poll_s=0.25):
                continue

            try:
                with Image.open(full) as im:
                    im.verify()
            except Exception:
                continue

            self.known_new_files.append(full)
            self.known_new_set.add(full)
            self.listbox.insert("end", os.path.basename(full))
            added_any = True

        self.status_var.set(f"Watching… {len(self.known_new_files)} image(s) queued.")

        if added_any:
            self.schedule_estimate()

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        sel.sort(reverse=True)
        for idx in sel:
            if 0 <= idx < len(self.known_new_files):
                path = self.known_new_files[idx]
                self.known_new_set.discard(path)
                del self.known_new_files[idx]
                self.listbox.delete(idx)

        self.status_var.set(f"Watching… {len(self.known_new_files)} image(s) queued.")
        self.schedule_estimate()

    def clear_list(self):
        self.known_new_files.clear()
        self.known_new_set.clear()
        self.listbox.delete(0, "end")
        self.status_var.set("Watching… 0 image(s) queued.")
        self.estimate_var.set("Estimated PDF size: —")
        self._last_estimate_bytes = 0

    # --------- Size estimate ---------
    def schedule_estimate(self):
        # debounce
        if self._estimate_job is not None:
            try:
                self.after_cancel(self._estimate_job)
            except Exception:
                pass
        self._estimate_job = self.after(350, self.run_estimate_now)

    def run_estimate_now(self):
        if not self.known_new_files:
            self.estimate_var.set("Estimated PDF size: —")
            self._last_estimate_bytes = 0
            return

        # cancel any previous estimate
        if self._estimate_thread and self._estimate_thread.is_alive():
            self._estimate_cancel.set()

        self._estimate_cancel = threading.Event()
        self.estimate_var.set("Estimated PDF size: calculating…")

        t = threading.Thread(target=self._estimate_worker, args=(self._estimate_cancel,), daemon=True)
        self._estimate_thread = t
        t.start()

    def _estimate_worker(self, cancel_evt: threading.Event):
        preset = self.preset_var.get()
        max_dpi_cap, jpeg_quality, subsampling = preset_to_params(preset)

        # Processing bytes sum is a close proxy for final PDF size because ReportLab embeds JPEG streams.
        total = 0
        overhead = 3500 * max(1, len(self.known_new_files))  # rough PDF structure overhead per page

        # Use a temp dir for actual JPEG outputs to get a reliable estimate.
        try:
            with tempfile.TemporaryDirectory(prefix="imgpdf_est_") as tmp:
                for i, src in enumerate(self.known_new_files, start=1):
                    if cancel_evt.is_set():
                        return

                    try:
                        st = os.stat(src)
                        key = (src, st.st_mtime_ns, st.st_size, max_dpi_cap, jpeg_quality, subsampling)
                    except OSError:
                        continue

                    with self._estimate_lock:
                        cached = self._estimate_cache.get(key)

                    if cached is None:
                        try:
                            out = preprocess_image_for_pdf(
                                src_path=src,
                                temp_dir=tmp,
                                max_dpi_cap=max_dpi_cap,
                                jpeg_quality=jpeg_quality,
                                subsampling=subsampling,
                            )
                            b = os.path.getsize(out)
                        except Exception:
                            b = 0

                        with self._estimate_lock:
                            self._estimate_cache[key] = b
                    else:
                        b = cached

                    total += b

                    # progressive UI update
                    est_now = total + overhead
                    self.after(0, lambda e=est_now, n=i: self.estimate_var.set(
                        f"Estimated PDF size: {human_size(int(e))}  (processed {n}/{len(self.known_new_files)})"
                    ))

                est_final = int(total + overhead)
                self._last_estimate_bytes = est_final
                self.after(0, lambda: self.estimate_var.set(f"Estimated PDF size: {human_size(est_final)}"))
        except Exception:
            self.after(0, lambda: self.estimate_var.set("Estimated PDF size: —"))

    # --------- PDF compile ---------
    def compile_pdf(self):
        if not self.known_new_files:
            messagebox.showwarning("No images", "No new images have been detected (after Start).")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save PDF as…",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not save_path:
            return
        save_path = safe_abspath(save_path)

        preset = self.preset_var.get()
        max_dpi_cap, jpeg_quality, subsampling = preset_to_params(preset)

        fit_within_margins = bool(self.fit_to_page_var.get())
        avoid_upscale = bool(self.avoid_upscale_var.get())

        page_w, page_h = letter
        margin = 36  # 0.5 inch
        avail_w = page_w - 2 * margin
        avail_h = page_h - 2 * margin

        try:
            with tempfile.TemporaryDirectory(prefix="imgpdf_") as tmp:
                c = canvas.Canvas(save_path, pagesize=letter)

                for src in self.known_new_files:
                    prepped = preprocess_image_for_pdf(
                        src_path=src,
                        temp_dir=tmp,
                        max_dpi_cap=max_dpi_cap,
                        jpeg_quality=jpeg_quality,
                        subsampling=subsampling,
                    )

                    with Image.open(prepped) as im:
                        iw_px, ih_px = im.size

                    # Treat pixels as points at 72 dpi; we're fitting to page anyway.
                    nat_w = float(iw_px)
                    nat_h = float(ih_px)

                    if fit_within_margins:
                        scale_fit = min(avail_w / nat_w, avail_h / nat_h)
                        if avoid_upscale:
                            scale_fit = min(scale_fit, 1.0)
                        draw_w = nat_w * scale_fit
                        draw_h = nat_h * scale_fit
                    else:
                        draw_w, draw_h = nat_w, nat_h

                    x = (page_w - draw_w) / 2
                    y = (page_h - draw_h) / 2

                    c.drawImage(
                        prepped,
                        x,
                        y,
                        width=draw_w,
                        height=draw_h,
                        preserveAspectRatio=True,
                        anchor="c",
                        mask=None,
                    )
                    c.showPage()

                c.save()

            pdf_bytes = os.path.getsize(save_path)
            DoneDialog(self, save_path, pdf_bytes)

        except Exception as e:
            messagebox.showerror("Failed", f"Could not create PDF:\n{e}")

    def on_close(self):
        self.stop_watching()
        try:
            if self._estimate_thread and self._estimate_thread.is_alive():
                self._estimate_cancel.set()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
