import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from pypdf import PdfReader, PdfWriter
import os
import subprocess
import sys


class PDFMergerApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Merger (Drag to Reorder)")
        self.geometry("650x450")

        self.pdf_files = []
        self.output_path = None

        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.listbox = tk.Listbox(frame, selectmode=tk.SINGLE)
        self.listbox.pack(fill=tk.BOTH, expand=True)

        self.listbox.drop_target_register(DND_FILES)
        self.listbox.dnd_bind("<<Drop>>", self.drop_reorder)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="Add PDFs", width=15, command=self.add_pdfs).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Remove Selected", width=15, command=self.remove_selected).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Merge PDFs", width=15, command=self.merge_pdfs).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Open File Location", width=18, command=self.open_file_location).pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self.progress.pack(fill=tk.X, padx=10, pady=10)

    def add_pdfs(self):
        files = filedialog.askopenfilenames(
            title="Select PDFs",
            filetypes=[("PDF Files", "*.pdf")]
        )
        for f in files:
            if f not in self.pdf_files:
                self.pdf_files.append(f)
                self.listbox.insert(tk.END, os.path.basename(f))

    def remove_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        index = sel[0]
        self.listbox.delete(index)
        del self.pdf_files[index]

    def drop_reorder(self, event):
        items = self.tk.splitlist(event.data)
        target_index = self.listbox.nearest(event.y)

        for item in items:
            path = item.strip("{}")
            if path in self.pdf_files:
                old_index = self.pdf_files.index(path)
                self.pdf_files.insert(target_index, self.pdf_files.pop(old_index))

        self.refresh_listbox()

    def refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for f in self.pdf_files:
            self.listbox.insert(tk.END, os.path.basename(f))

    def merge_pdfs(self):
        if not self.pdf_files:
            messagebox.showerror("Error", "No PDFs selected")
            return

        self.output_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
            title="Save Merged PDF"
        )

        if not self.output_path:
            return

        try:
            writer = PdfWriter()

            total_pages = sum(len(PdfReader(p).pages) for p in self.pdf_files)
            self.progress["maximum"] = total_pages
            self.progress["value"] = 0
            self.update_idletasks()

            for pdf in self.pdf_files:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    writer.add_page(page)
                    self.progress["value"] += 1
                    self.update_idletasks()

            # Safe optimization
            writer.add_metadata({})

            with open(self.output_path, "wb") as f:
                writer.write(f)

            messagebox.showinfo(
                "Completed",
                "PDFs merged successfully.\n\nYou can now open the file location."
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def open_file_location(self):
        if not self.output_path or not os.path.exists(self.output_path):
            messagebox.showerror("Error", "Merged file not found")
            return

        if sys.platform.startswith("win"):
            subprocess.run(["explorer", "/select,", os.path.normpath(self.output_path)])
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", "-R", self.output_path])
        else:
            subprocess.run(["xdg-open", os.path.dirname(self.output_path)])


if __name__ == "__main__":
    app = PDFMergerApp()
    app.mainloop()
