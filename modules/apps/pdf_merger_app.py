"""Tkinter drag-and-drop PDF merger application."""

import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pypdf import PdfReader, PdfWriter
from tkinterdnd2 import DND_FILES, TkinterDnD


class PDFMergerApp(TkinterDnD.Tk):
    """Desktop UI for selecting, ordering, and merging PDF files."""

    def __init__(self):
        """Initialize merger window and application state.

        Returns:
            None: Initializes GUI app.
        """
        super().__init__()
        self.title("PDF Merger (Drag to Reorder)")
        self.geometry("650x450")

        self.pdf_files = []
        self.output_path = None
        self.create_widgets()

    def create_widgets(self):
        """Create and layout all UI controls.

        Returns:
            None: Builds widget tree.
        """
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
        """Open file picker and append selected PDFs to queue.

        Returns:
            None: Updates queue and listbox.
        """
        files = filedialog.askopenfilenames(title="Select PDFs", filetypes=[("PDF Files", "*.pdf")])
        for file_path in files:
            # Ignore duplicates so each input appears only once.
            if file_path not in self.pdf_files:
                self.pdf_files.append(file_path)
                self.listbox.insert(tk.END, os.path.basename(file_path))

    def remove_selected(self):
        """Remove selected PDF from queue.

        Returns:
            None: Mutates queue/list UI.
        """
        sel = self.listbox.curselection()
        # If no row is selected, do nothing.
        if not sel:
            return
        index = sel[0]
        self.listbox.delete(index)
        del self.pdf_files[index]

    def drop_reorder(self, event):
        """Handle drag/drop reorder events in listbox.

        Args:
            event: TkinterDnD event with dropped items.

        Returns:
            None: Reorders queue and refreshes list.
        """
        items = self.tk.splitlist(event.data)
        target_index = self.listbox.nearest(event.y)
        for item in items:
            path = item.strip("{}")
            # If dropped item exists in queue, move it near target index.
            if path in self.pdf_files:
                old_index = self.pdf_files.index(path)
                self.pdf_files.insert(target_index, self.pdf_files.pop(old_index))
        self.refresh_listbox()

    def refresh_listbox(self):
        """Redraw listbox from current queue order.

        Returns:
            None: Updates UI state.
        """
        self.listbox.delete(0, tk.END)
        for file_path in self.pdf_files:
            self.listbox.insert(tk.END, os.path.basename(file_path))

    def merge_pdfs(self):
        """Merge queued PDFs into one output file with progress updates.

        Returns:
            None: Writes merged file and shows dialogs.
        """
        # If no PDFs were selected, show validation error.
        if not self.pdf_files:
            messagebox.showerror("Error", "No PDFs selected")
            return

        self.output_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")], title="Save Merged PDF")
        # If user cancels save dialog, abort merge.
        if not self.output_path:
            return

        try:
            writer = PdfWriter()
            # Count total pages first for accurate progress bar range.
            total_pages = sum(len(PdfReader(path).pages) for path in self.pdf_files)
            self.progress["maximum"] = total_pages
            self.progress["value"] = 0
            self.update_idletasks()

            for pdf in self.pdf_files:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    # Append each page and advance progress indicator.
                    writer.add_page(page)
                    self.progress["value"] += 1
                    self.update_idletasks()

            writer.add_metadata({})
            with open(self.output_path, "wb") as handle:
                writer.write(handle)

            messagebox.showinfo("Completed", "PDFs merged successfully.\n\nYou can now open the file location.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def open_file_location(self):
        """Open OS file manager and highlight merged output file.

        Returns:
            None: Executes system file manager command.
        """
        # If output path is missing or file no longer exists, show error.
        if not self.output_path or not os.path.exists(self.output_path):
            messagebox.showerror("Error", "Merged file not found")
            return

        # Run platform-specific "open folder/select file" command.
        if sys.platform.startswith("win"):
            subprocess.run(["explorer", "/select,", os.path.normpath(self.output_path)])
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", "-R", self.output_path])
        else:
            subprocess.run(["xdg-open", os.path.dirname(self.output_path)])


def run():
    """Launch the PDF merger GUI application.

    Returns:
        None: Starts Tkinter event loop.
    """
    app = PDFMergerApp()
    app.mainloop()

