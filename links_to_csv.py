import csv
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

FORMULA_PREFIXES = ("=", "+", "-", "@")


def extract_page_number(text: str) -> str:
    """Extract a page number from text like .../Page-5.png or 'page 5'."""
    if not text:
        return ""
    match = re.search(r"page[-_\s]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    trailing_num = re.search(r"(\d+)(?:\.[a-zA-Z0-9]+)?$", text.strip())
    return trailing_num.group(1) if trailing_num else ""


def sanitize_csv_text(value: str) -> str:
    """Normalize free text so it is safe to write into CSV cells."""
    if not value:
        return ""
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", str(value))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.startswith(FORMULA_PREFIXES):
        cleaned = "'" + cleaned
    return cleaned


def parse_link_line(line: str) -> str:
    """Return URL from a pasted line (plain URL or markdown image)."""
    raw = line.strip()
    if not raw:
        return ""
    md_match = re.search(r"!\[.*?\]\((https?://[^\s)]+)\)", raw)
    if md_match:
        return md_match.group(1).strip()
    return raw.split()[0].strip()


def read_location_note_rows(csv_path: str):
    """Read rows from a CSV and keep whatever Location/Note data is present."""
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []

        names = {name.strip().lower(): name for name in reader.fieldnames if name}
        location_key = names.get("location")
        note_key = names.get("note")

        rows = []
        for row in reader:
            location = sanitize_csv_text(row.get(location_key, "") if location_key else "")
            note = sanitize_csv_text(row.get(note_key, "") if note_key else "")
            if not location and not note:
                continue
            rows.append({"Location": location, "Note": note})
        return rows


class LinkCsvBuilderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Links + Location/Note CSV Merger")
        self.root.geometry("820x600")
        self._build_ui()

    def _build_ui(self):
        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=12, pady=(12, 6))

        tk.Label(body, text="Paste image links below (one per line)").pack(anchor="w")

        self.links_text = tk.Text(body, wrap="word", height=18)
        self.links_text.pack(fill="both", expand=True, pady=(4, 8))

        hint = (
            "Example link: https://i.ibb.co/HfL7smbV/Page-1.png\n"
            "Then click 'Merge with Location/Note CSVs' to select multiple CSV files.\n"
            "Output columns: Highlight Location, Note."
        )
        tk.Label(body, text=hint, fg="#555").pack(anchor="w")

        actions = tk.Frame(self.root)
        actions.pack(fill="x", padx=12, pady=12)

        tk.Button(
            actions,
            text="Merge with Location/Note CSVs",
            width=30,
            command=self.merge_with_location_note_csvs,
        ).pack(side="left")
        tk.Button(actions, text="Clear", width=12, command=self.clear_all).pack(
            side="left", padx=8
        )

    def clear_all(self):
        self.links_text.delete("1.0", tk.END)

    def _prompt_output_path(self, initialfile: str = "highlight_location_note.csv"):
        return filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            initialfile=initialfile,
            filetypes=[("CSV files", "*.csv")],
        )

    def _build_page_link_map(self):
        lines = [line.strip() for line in self.links_text.get("1.0", tk.END).splitlines() if line.strip()]
        if not lines:
            return None

        page_to_link = {}
        for line in lines:
            url = parse_link_line(line)
            page = extract_page_number(url)
            if not page:
                continue
            page_to_link.setdefault(page, sanitize_csv_text(url))
        return page_to_link

    def merge_with_location_note_csvs(self):
        page_to_link = self._build_page_link_map()
        if page_to_link is None:
            messagebox.showerror("No links", "Please paste at least one image link.")
            return

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
                messagebox.showerror("Invalid CSV", f"Could not read {Path(path).name}:\n{exc}")
                return

        if not merged_rows:
            messagebox.showerror("No rows", "Selected CSV files contain no usable rows.")
            return

        output_rows = []
        for row in merged_rows:
            location = row.get("Location", "")
            note = row.get("Note", "")

            mapped_link = page_to_link.get(location, "")
            if not mapped_link and location:
                mapped_link = page_to_link.get(extract_page_number(location), "")

            if mapped_link and location:
                highlight_location = f"{mapped_link} {location}"
            elif mapped_link:
                highlight_location = mapped_link
            else:
                highlight_location = location

            output_rows.append(
                {
                    "Highlight Location": sanitize_csv_text(highlight_location),
                    "Note": sanitize_csv_text(note),
                }
            )

        output_path = self._prompt_output_path()
        if not output_path:
            return

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Highlight Location", "Note"])
                writer.writeheader()
                writer.writerows(output_rows)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not write merged CSV:\n{exc}")
            return

        messagebox.showinfo(
            "Done",
            f"Merged {len(csv_paths)} files into {len(output_rows)} rows:\n{output_path}",
        )


if __name__ == "__main__":
    root = tk.Tk()
    LinkCsvBuilderApp(root)
    root.mainloop()
