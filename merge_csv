import os
import re
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

FORMULA_PREFIXES = ("=", "+", "-", "@")


def filename_from_title(title: str, max_length: int = 200) -> str:
    """Build a Windows-safe filename from the title."""
    if not title or not title.strip():
        return "merged"
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title.strip())
    s = re.sub(r"\s+", " ", s).strip(" .")
    if not s:
        return "merged"
    if len(s) > max_length:
        s = s[:max_length].rstrip(" .")
    return s or "merged"


def escape_formula_cells(value):
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def normalize_note_prefix(value):
    if isinstance(value, str) and value.lstrip().startswith("."):
        return "Answer: " + value
    return value


def ensure_highlight_note_columns(df, file_path: str):
    if {"Highlight", "Note"}.issubset(df.columns):
        return df[["Highlight", "Note"]]
    if df.shape[1] < 2:
        raise ValueError(
            f"{file_path} must have at least 2 columns to map to 'Highlight' and 'Note'"
        )
    df = df.iloc[:, :2].copy()
    df.columns = ["Highlight", "Note"]
    return df


class MergeCsvApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Merge CSV")
        self.root.geometry("720x520")

        self.file_paths = []

        self.title_var = tk.StringVar()
        self.author_var = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        meta_frame = tk.Frame(self.root)
        meta_frame.pack(fill="x", padx=10, pady=8)

        tk.Label(meta_frame, text="Title").grid(row=0, column=0, sticky="w")
        tk.Entry(meta_frame, textvariable=self.title_var, width=60).grid(
            row=0, column=1, padx=6, pady=2, sticky="w"
        )

        tk.Label(meta_frame, text="Author").grid(row=1, column=0, sticky="w")
        tk.Entry(meta_frame, textvariable=self.author_var, width=60).grid(
            row=1, column=1, padx=6, pady=2, sticky="w"
        )

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

        tk.Button(
            bottom,
            text="Merge CSVs",
            width=18,
            command=self.merge_csvs,
        ).pack(pady=6)

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=[("CSV files", "*.csv")],
        )
        if not paths:
            return
        for p in paths:
            if p not in self.file_paths:
                self.file_paths.append(p)
        self.refresh_list()

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            self.file_paths.pop(idx)
        self.refresh_list()

    def clear_files(self):
        self.file_paths = []
        self.refresh_list()

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for p in self.file_paths:
            self.listbox.insert(tk.END, p)
        self.status_label.config(text=f"{len(self.file_paths)} file(s) selected")

    def merge_csvs(self):
        title_value = self.title_var.get().strip()
        author_value = self.author_var.get().strip()

        if not title_value:
            messagebox.showerror("Error", "Title cannot be empty.")
            return
        if not author_value:
            messagebox.showerror("Error", "Author cannot be empty.")
            return
        if not self.file_paths:
            messagebox.showerror("Error", "Please select at least one CSV file.")
            return

        base_name = filename_from_title(title_value) + ".csv"
        parent_dir = os.path.dirname(os.path.abspath(self.file_paths[0]))
        output_path = os.path.join(parent_dir, base_name)

        dataframes = []
        try:
            for file in self.file_paths:
                df = pd.read_csv(file, header=None)
                df = ensure_highlight_note_columns(df, file)
                df["Title"] = title_value
                df["Author"] = author_value
                df["Note"] = df["Note"].map(normalize_note_prefix)
                df = df.apply(lambda col: col.map(escape_formula_cells))
                dataframes.append(df)

            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df = combined_df[["Title", "Author", "Highlight", "Note"]]
            combined_df.to_csv(output_path, index=False)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        messagebox.showinfo("Success", f"Combined CSV saved to:\n{output_path}")
        self.status_label.config(text=f"Saved: {os.path.basename(output_path)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MergeCsvApp(root)
    root.mainloop()
