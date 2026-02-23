import csv
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

FORMULA_PREFIXES = ("=", "+", "-", "@")


def extract_page_number(url: str) -> str:
    """Extract a page number from a URL like .../Page-5.png."""
    if not url:
        return ""
    match = re.search(r"page[-_\s]?(\d+)", url, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    trailing_num = re.search(r"(\d+)(?:\.[a-zA-Z0-9]+)?$", url)
    return trailing_num.group(1) if trailing_num else ""


def split_link_and_note(line: str):
    """Parse one pasted line into URL and note."""
    raw = line.strip()
    if not raw:
        return "", ""

    md_match = re.search(r"!\[.*?\]\((https?://[^\s)]+)\)", raw)
    if md_match:
        url = md_match.group(1)
        rest = raw[md_match.end() :].strip(" \t|,-")
        return url, rest

    for delimiter in ("\t", " | ", " , ", "|", ","):
        if delimiter in raw:
            left, right = raw.split(delimiter, 1)
            return left.strip(), right.strip()

    return raw, ""


def sanitize_csv_text(value: str) -> str:
    """Normalize free text so it is safe to write into CSV cells."""
    if not value:
        return ""
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.startswith(FORMULA_PREFIXES):
        cleaned = "'" + cleaned
    return cleaned


def read_location_note_rows(csv_path: str):
    """Read a Location/Note CSV and return normalized rows."""
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []

        names = {name.strip().lower(): name for name in reader.fieldnames if name}
        location_key = names.get("location")
        note_key = names.get("note")

        if not location_key or not note_key:
            raise ValueError("CSV must include 'Location' and 'Note' columns.")

        rows = []
        for row in reader:
            rows.append(
                {
                    "Location": sanitize_csv_text(row.get(location_key, "")),
                    "Note": sanitize_csv_text(row.get(note_key, "")),
                }
            )
        return rows


class LinkCsvBuilderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Links to Location/Note CSV")
        self.root.geometry("760x560")

        self._build_ui()

    def _build_ui(self):
        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=12, pady=(12, 6))

        tk.Label(body, text="Paste link + note rows below").pack(anchor="w")

        self.links_text = tk.Text(body, wrap="word", height=18)
        self.links_text.pack(fill="both", expand=True, pady=(4, 8))

        hint = (
            "Accepted formats per line:\n"
            "• https://.../Page-5.png\n"
            "• https://.../Page-5.png<TAB>Your note\n"
            "• ![](https://.../Page-5.png) | Your note\n"
            "Output CSV columns are: Location, Note (Location is page number parsed from link)."
        )
        tk.Label(body, text=hint, fg="#555").pack(anchor="w")

        actions = tk.Frame(self.root)
        actions.pack(fill="x", padx=12, pady=12)

        tk.Button(actions, text="Build CSV", width=14, command=self.build_csv).pack(
            side="left"
        )
        tk.Button(
            actions,
            text="Merge Location/Note CSVs",
            width=24,
            command=self.merge_location_note_csvs,
        ).pack(side="left", padx=8)
        tk.Button(actions, text="Clear", width=12, command=self.clear_all).pack(
            side="left"
        )

    def clear_all(self):
        self.links_text.delete("1.0", tk.END)

    def _prompt_output_path(self, initialfile: str = "location_note.csv"):
        return filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            initialfile=initialfile,
            filetypes=[("CSV files", "*.csv")],
        )

    def _write_rows(self, output_path: str, rows):
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Location", "Note"])
            writer.writeheader()
            writer.writerows(rows)

    def build_csv(self):
        raw_lines = self.links_text.get("1.0", tk.END).splitlines()
        lines = [line.strip() for line in raw_lines if line.strip()]

        if not lines:
            messagebox.showerror("No links", "Please paste at least one link.")
            return

        rows = []
        for line in lines:
            url, note = split_link_and_note(line)
            rows.append(
                {
                    "Location": sanitize_csv_text(extract_page_number(url)),
                    "Note": sanitize_csv_text(note),
                }
            )

        output_path = self._prompt_output_path()
        if not output_path:
            return

        try:
            self._write_rows(output_path, rows)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not write CSV:\n{exc}")
            return

        messagebox.showinfo(
            "Done",
            f"CSV created successfully with {len(rows)} rows:\n{output_path}",
        )

    def merge_location_note_csvs(self):
        csv_paths = filedialog.askopenfilenames(
            title="Select Location/Note CSV files",
            filetypes=[("CSV files", "*.csv")],
        )
        if not csv_paths:
            return

        merged_rows = []
        for path in csv_paths:
            try:
                merged_rows.extend(read_location_note_rows(path))
            except Exception as exc:
                messagebox.showerror(
                    "Invalid CSV",
                    f"Could not read {Path(path).name}:\n{exc}",
                )
                return

        if not merged_rows:
            messagebox.showerror("No rows", "Selected CSV files contain no data rows.")
            return

        output_path = self._prompt_output_path("merged_location_note.csv")
        if not output_path:
            return

        try:
            self._write_rows(output_path, merged_rows)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not write merged CSV:\n{exc}")
            return

        messagebox.showinfo(
            "Done",
            f"Merged {len(csv_paths)} files into {len(merged_rows)} rows:\n{output_path}",
        )


if __name__ == "__main__":
    root = tk.Tk()
    LinkCsvBuilderApp(root)
    root.mainloop()
