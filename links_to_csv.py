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
    cleaned = str(value)
    cleaned = cleaned.replace("â€¢", "•")
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.startswith(FORMULA_PREFIXES):
        cleaned = "'" + cleaned
    return cleaned


def format_note_bullets(note: str) -> str:
    """Ensure bullet notes are separated by a blank line."""
    if not note:
        return ""
    text = sanitize_csv_text(note)
    text = re.sub(r"\s*•\s*", "\n\n• ", text)
    return text.lstrip()




def note_or_placeholder(note: str) -> str:
    """Return note text or the required placeholder when note is missing."""
    formatted = format_note_bullets(note)
    return formatted if formatted else ".add note"

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
            note = format_note_bullets(row.get(note_key, "") if note_key else "")
            if not location and not note:
                continue
            rows.append({"Location": location, "Note": note})
        return rows


class LinkCsvBuilderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Links + Location/Note CSV Merger")
        self.root.geometry("820x600")
        self.title_var = tk.StringVar()
        self.author_var = tk.StringVar()
        self._build_ui()

    def _build_ui(self):
        meta = tk.Frame(self.root)
        meta.pack(fill="x", padx=12, pady=(12, 4))

        tk.Label(meta, text="Title").grid(row=0, column=0, sticky="w")
        tk.Entry(meta, textvariable=self.title_var, width=72).grid(
            row=0, column=1, padx=8, pady=3, sticky="w"
        )

        tk.Label(meta, text="Author").grid(row=1, column=0, sticky="w")
        tk.Entry(meta, textvariable=self.author_var, width=72).grid(
            row=1, column=1, padx=8, pady=3, sticky="w"
        )

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=12, pady=(4, 6))

        tk.Label(body, text="Paste image links below (one per line)").pack(anchor="w")

        self.links_text = tk.Text(body, wrap="word", height=18)
        self.links_text.pack(fill="both", expand=True, pady=(4, 8))

        hint = (
            "Example link: https://i.ibb.co/HfL7smbV/Page-1.png\n"
            "Then click 'Merge with Location/Note CSVs' to select multiple CSV files.\n"
            "Output columns: Highlight, Title, Author, Location, Note."
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
        self.title_var.set("")
        self.author_var.set("")
        self.links_text.delete("1.0", tk.END)

    def _prompt_output_path(self, initialfile: str = "highlight_location_note.csv"):
        return filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            initialfile=initialfile,
            filetypes=[("CSV files", "*.csv")],
        )

    def _get_pasted_links(self):
        lines = [line.strip() for line in self.links_text.get("1.0", tk.END).splitlines() if line.strip()]
        if not lines:
            return None

        links = []
        for line in lines:
            url = sanitize_csv_text(parse_link_line(line))
            if not url:
                continue
            links.append({"url": url, "location": extract_page_number(url)})
        return links

    def merge_with_location_note_csvs(self):
        links = self._get_pasted_links()
        if links is None:
            messagebox.showerror("No links", "Please paste at least one image link.")
            return

        title = sanitize_csv_text(self.title_var.get())
        author = sanitize_csv_text(self.author_var.get())

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

        notes_by_location = {}
        unmatched_rows = []
        for row in merged_rows:
            location = row.get("Location", "")
            note = row.get("Note", "")
            key = extract_page_number(location) if location else ""
            if key:
                notes_by_location.setdefault(key, []).append(note)
            else:
                unmatched_rows.append({"Location": location, "Note": note})

        output_rows = []
        matched_keys = set()
        for link in links:
            location = link.get("location", "")
            matched_notes = notes_by_location.get(location, []) if location else []
            combined_note = "\n\n".join([n for n in matched_notes if n])
            if location:
                matched_keys.add(location)

            output_rows.append(
                {
                    "Highlight": sanitize_csv_text(f"![]({link.get('url', '')})"),
                    "Title": title,
                    "Author": author,
                    "Location": sanitize_csv_text(location),
                    "Note": note_or_placeholder(combined_note),
                }
            )

        for key, notes in notes_by_location.items():
            if key in matched_keys:
                continue
            output_rows.append(
                {
                    "Highlight": "",
                    "Title": title,
                    "Author": author,
                    "Location": sanitize_csv_text(key),
                    "Note": note_or_placeholder("\n\n".join([n for n in notes if n])),
                }
            )

        for row in unmatched_rows:
            output_rows.append(
                {
                    "Highlight": "",
                    "Title": title,
                    "Author": author,
                    "Location": sanitize_csv_text(row.get("Location", "")),
                    "Note": note_or_placeholder(row.get("Note", "")),
                }
            )

        output_path = self._prompt_output_path()
        if not output_path:
            return

        try:
            with open(output_path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["Highlight", "Title", "Author", "Location", "Note"]
                )
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
