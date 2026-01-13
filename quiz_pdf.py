import os
import time
import threading
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, PageBreak

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FolderToPDFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Folder to PDF Compiler")
        self.root.geometry("600x450")

        self.watch_folder = None
        self.running = False

        self.baseline_files = set()   # Files present when monitoring starts
        self.tracked_files = set()    # Files added after start
        self.image_list = []

        self.build_ui()

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(pady=10)

        tk.Button(top, text="Set Folder to Watch", command=self.select_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Run Monitoring", command=self.start_monitoring).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Stop", command=self.stop_monitoring).pack(side=tk.LEFT, padx=5)

        mid = tk.Frame(self.root)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.listbox = tk.Listbox(mid, selectmode=tk.MULTIPLE)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(mid, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        bottom = tk.Frame(self.root)
        bottom.pack(pady=10)

        tk.Button(bottom, text="Remove Selected", command=self.remove_selected).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom, text="Compile PDF", command=self.compile_pdf).pack(side=tk.LEFT, padx=5)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.watch_folder = folder
            self.reset_state()

    def reset_state(self):
        self.baseline_files.clear()
        self.tracked_files.clear()
        self.image_list.clear()
        self.listbox.delete(0, tk.END)

    def start_monitoring(self):
        if not self.watch_folder:
            messagebox.showerror("Error", "Please select a folder first.")
            return
        if self.running:
            return

        self.baseline_files = {
            f for f in os.listdir(self.watch_folder)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        }

        self.running = True
        threading.Thread(target=self.monitor_folder, daemon=True).start()

    def stop_monitoring(self):
        self.running = False

    def monitor_folder(self):
        while self.running:
            try:
                current_files = {
                    f for f in os.listdir(self.watch_folder)
                    if f.lower().endswith(SUPPORTED_EXTENSIONS)
                }

                new_files = current_files - self.baseline_files - self.tracked_files

                for file in sorted(new_files):
                    full_path = os.path.join(self.watch_folder, file)
                    self.tracked_files.add(file)
                    self.image_list.append(full_path)
                    self.listbox.insert(tk.END, file)

            except Exception:
                pass

            time.sleep(1)

    def remove_selected(self):
        selected = list(self.listbox.curselection())
        for index in reversed(selected):
            del self.image_list[index]
            self.listbox.delete(index)

    def compile_pdf(self):
        if not self.image_list:
            messagebox.showerror("Error", "No images to compile.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if not save_path:
            return

        try:
            doc = SimpleDocTemplate(
                save_path,
                pagesize=LETTER,
                leftMargin=36,
                rightMargin=36,
                topMargin=36,
                bottomMargin=36
            )

            elements = []
            temp_files = []

            max_width = LETTER[0] - 72
            max_height = LETTER[1] - 72

            for img_path in self.image_list:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((max_width, max_height), Image.LANCZOS)

                temp_path = img_path + "_temp.jpg"
                img.save(temp_path, "JPEG", quality=75, optimize=True)
                temp_files.append(temp_path)

                elements.append(RLImage(temp_path, width=img.width, height=img.height))
                elements.append(PageBreak())

            doc.build(elements)

            for temp in temp_files:
                if os.path.exists(temp):
                    os.remove(temp)

            self.show_success_popup(save_path)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_success_popup(self, pdf_path):
        popup = tk.Toplevel(self.root)
        popup.title("PDF Created")
        popup.geometry("350x150")
        popup.resizable(False, False)

        tk.Label(
            popup,
            text="PDF created successfully.",
            font=("Segoe UI", 11)
        ).pack(pady=20)

        btn_frame = tk.Frame(popup)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="Open File Location",
            width=18,
            command=lambda: self.open_file_location(pdf_path)
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Close",
            width=10,
            command=popup.destroy
        ).pack(side=tk.LEFT, padx=5)

    def open_file_location(self, path):
        try:
            if sys.platform.startswith("win"):
                os.startfile(os.path.normpath(path), "explore")
            elif sys.platform == "darwin":
                subprocess.run(["open", "-R", path])
            else:
                subprocess.run(["xdg-open", os.path.dirname(path)])
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = FolderToPDFApp(root)
    root.mainloop()
