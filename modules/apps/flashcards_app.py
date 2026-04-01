"""Tkinter app for merging multiple flashcard CSVs into one export."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd

from .flashcards_utils import (
    ensure_highlight_note_columns,
    escape_formula_cells,
    filename_from_title,
    normalize_note_prefix,
)


class MergeCsvApp:
    """Desktop UI for combining CSV files with title/author metadata."""

    def __init__(self, root: tk.Tk):
        """Initialize UI state and create widgets.

        Args:
            root (tk.Tk): Root Tkinter window.

        Returns:
            None: Initializes app instance.
        """
        self.root = root
        self.root.title("Merge CSV")
        self.root.geometry("720x520")

        self.file_paths = []
        self.title_var = tk.StringVar()
        self.author_var = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        """Construct and lay out all widgets for the merge tool.

        Returns:
            None: Builds interface elements.
        """
        meta_frame = tk.Frame(self.root)
        meta_frame.pack(fill="x", padx=10, pady=8)

        tk.Label(meta_frame, text="Title").grid(row=0, column=0, sticky="w")
        tk.Entry(meta_frame, textvariable=self.title_var, width=60).grid(row=0, column=1, padx=6, pady=2, sticky="w")

        tk.Label(meta_frame, text="Author").grid(row=1, column=0, sticky="w")
        tk.Entry(meta_frame, textvariable=self.author_var, width=60).grid(row=1, column=1, padx=6, pady=2, sticky="w")

        queue_frame = tk.Frame(self.root)
        queue_frame.pack(fill="both", expand=True, padx=10, pady=8)

        left = tk.Frame(queue_frame)
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="Selected CSV Files").pack(anchor="w")
        self.listbox = tk.Listbox(left, height=14, width=75)
        self.listbox.pack(fill="both", expand=True, pady=6)

        right = tk.Frame(queue_frame)
        right.pack(side="left", fill="y", padx=10)
        tk.Button(right, text="Add CSVs", width=16, command=self.add_files).pack(pady=4)
        tk.Button(right, text="Remove Selected", width=16, command=self.remove_selected).pack(pady=4)
        tk.Button(right, text="Clear List", width=16, command=self.clear_files).pack(pady=4)

        bottom = tk.Frame(self.root)
        bottom.pack(fill="x", padx=10, pady=10)
        self.status_label = tk.Label(bottom, text="Idle")
        self.status_label.pack(anchor="w")
        tk.Button(bottom, text="Merge CSVs", width=18, command=self.merge_csvs).pack(pady=6)

    def add_files(self):
        """Open file picker and add selected CSV files to queue.

        Returns:
            None: Updates internal queue and listbox.
        """
        paths = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])
        # If user cancels dialog, do nothing.
        if not paths:
            return
        for p in paths:
            # Avoid duplicate queue entries.
            if p not in self.file_paths:
                self.file_paths.append(p)
        self.refresh_list()

    def remove_selected(self):
        """Remove currently selected files from queue.

        Returns:
            None: Mutates queued file list and refreshes UI.
        """
        sel = list(self.listbox.curselection())
        # If no row is selected, nothing to remove.
        if not sel:
            return
        for idx in reversed(sel):
            self.file_paths.pop(idx)
        self.refresh_list()

    def clear_files(self):
        """Clear all selected input files.

        Returns:
            None: Resets queue and listbox.
        """
        self.file_paths = []
        self.refresh_list()

    def refresh_list(self):
        """Refresh queue listbox and status label.

        Returns:
            None: Updates visual state only.
        """
        self.listbox.delete(0, tk.END)
        for p in self.file_paths:
            self.listbox.insert(tk.END, p)
        self.status_label.config(text=f"{len(self.file_paths)} file(s) selected")

    def merge_csvs(self):
        """Validate inputs, merge queued CSVs, and write combined output file.

        Returns:
            None: Saves file and displays success/error dialogs.
        """
        title_value = self.title_var.get().strip()
        author_value = self.author_var.get().strip()
        # If title is missing, stop and show validation error.
        if not title_value:
            messagebox.showerror("Error", "Title cannot be empty.")
            return
        # If author is missing, stop and show validation error.
        if not author_value:
            messagebox.showerror("Error", "Author cannot be empty.")
            return
        # If no input files exist, stop and show validation error.
        if not self.file_paths:
            messagebox.showerror("Error", "Please select at least one CSV file.")
            return

        base_name = filename_from_title(title_value) + ".csv"
        parent_dir = os.path.dirname(os.path.abspath(self.file_paths[0]))
        output_path = os.path.join(parent_dir, base_name)

        dataframes = []
        try:
            for file in self.file_paths:
                # Load raw CSV rows without assuming header row.
                df = pd.read_csv(file, header=None)
                # Normalize into expected Highlight/Note schema.
                df = ensure_highlight_note_columns(df, file)
                df["Title"] = title_value
                df["Author"] = author_value
                df["Note"] = df["Note"].map(normalize_note_prefix)
                # Escape formula-like cell values for spreadsheet safety.
                df = df.apply(lambda col: col.map(escape_formula_cells))
                dataframes.append(df)

            # Combine all rows and keep canonical column order.
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df = combined_df[["Title", "Author", "Highlight", "Note"]]
            combined_df.to_csv(output_path, index=False)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        messagebox.showinfo("Success", f"Combined CSV saved to:\n{output_path}")
        self.status_label.config(text=f"Saved: {os.path.basename(output_path)}")


def run():
    """Launch the flashcards merger GUI application.

    Returns:
        None: Starts Tkinter main loop.
    """
    root = tk.Tk()
    MergeCsvApp(root)
    root.mainloop()

