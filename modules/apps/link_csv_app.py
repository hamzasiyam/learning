"""Tkinter app for merging pasted image links with Location/Note CSVs."""

import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from .link_csv_utils import (
    extract_page_number,
    note_or_placeholder,
    parse_link_line,
    read_location_note_rows,
    sanitize_csv_text,
)


class LinkCsvBuilderApp:
    """Desktop UI for producing Highlight/Location/Note export CSV files."""

    def __init__(self, root: tk.Tk):
        """Initialize app state and build interface widgets.

        Args:
            root (tk.Tk): Root Tkinter window.

        Returns:
            None: Initializes GUI state.
        """
        self.root = root
        self.root.title("Links + Location/Note CSV Merger")
        self.root.geometry("820x600")
        self.title_var = tk.StringVar()
        self.author_var = tk.StringVar()
        self._build_ui()

    def _build_ui(self):
        """Construct all labels, text fields, and action buttons.

        Returns:
            None: Builds UI controls.
        """
        meta = tk.Frame(self.root)
        meta.pack(fill="x", padx=12, pady=(12, 4))
        tk.Label(meta, text="Title").grid(row=0, column=0, sticky="w")
        tk.Entry(meta, textvariable=self.title_var, width=72).grid(row=0, column=1, padx=8, pady=3, sticky="w")
        tk.Label(meta, text="Author").grid(row=1, column=0, sticky="w")
        tk.Entry(meta, textvariable=self.author_var, width=72).grid(row=1, column=1, padx=8, pady=3, sticky="w")

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=12, pady=(4, 6))
        tk.Label(body, text="Paste image links below (one per line)").pack(anchor="w")
        self.links_text = tk.Text(body, wrap="word", height=18)
        self.links_text.pack(fill="both", expand=True, pady=(4, 8))
        hint = (
            "Example link: https://i.ibb.co/HfL7smbV/Page-1.png\n"
            "Click 'Merge with Location/Note CSVs' to select CSVs (optional—cancel to export links only with .addnote).\n"
            "Output columns: Highlight, Title, Author, Location, Note."
        )
        tk.Label(body, text=hint, fg="#555").pack(anchor="w")

        actions = tk.Frame(self.root)
        actions.pack(fill="x", padx=12, pady=12)
        tk.Button(actions, text="Merge with Location/Note CSVs", width=30, command=self.merge_with_location_note_csvs).pack(side="left")
        tk.Button(actions, text="Clear", width=12, command=self.clear_all).pack(side="left", padx=8)

    def clear_all(self):
        """Clear pasted link text area.

        Returns:
            None: Resets text input.
        """
        self.links_text.delete("1.0", tk.END)

    def _prompt_output_path(self, initialfile: str = "highlight_location_note.csv"):
        """Show save dialog and return output CSV path.

        Args:
            initialfile (str): Suggested default file name.

        Returns:
            str: Selected file path, or empty string on cancel.
        """
        return filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            initialfile=initialfile,
            filetypes=[("CSV files", "*.csv")],
        )

    def _get_pasted_links(self):
        """Parse pasted links from text area into structured records.

        Returns:
            list[dict[str, str]] | None: Link records or None when no links.
        """
        lines = [line.strip() for line in self.links_text.get("1.0", tk.END).splitlines() if line.strip()]
        # If no non-empty lines are provided, return None for validation path.
        if not lines:
            return None
        links = []
        for line in lines:
            # Normalize each line into a single URL string.
            url = sanitize_csv_text(parse_link_line(line))
            if not url:
                continue
            links.append({"url": url, "location": extract_page_number(url)})
        return links

    def merge_with_location_note_csvs(self):
        """Merge pasted links with optional CSV Location/Note rows and export.

        Returns:
            None: Writes output CSV and displays status dialogs.
        """
        links = self._get_pasted_links()
        # If no links were pasted, show validation error and stop.
        if links is None:
            messagebox.showerror("No links", "Please paste at least one image link.")
            return

        title = sanitize_csv_text(self.title_var.get())
        author = sanitize_csv_text(self.author_var.get())

        csv_paths = filedialog.askopenfilenames(
            title="Select Location/Note CSV files (optional - cancel to use links only)",
            filetypes=[("CSV files", "*.csv")],
        )

        merged_rows = []
        for path in csv_paths:
            try:
                # Read and normalize each selected CSV source file.
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
            # If location has a page key, bucket note by page number.
            if key:
                notes_by_location.setdefault(key, []).append(note)
            else:
                # Keep rows without location for later append.
                unmatched_rows.append({"Location": location, "Note": note})

        output_rows = []
        matched_keys = set()
        for link in links:
            location = link.get("location", "")
            matched_notes = notes_by_location.get(location, []) if location else []
            combined_note = "\n\n".join([n for n in matched_notes if n])
            if location:
                matched_keys.add(location)
            # Build output row in the expected export schema.
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
            # Add notes that did not match any pasted link page.
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
            # Preserve unmatched rows with blank highlight.
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
        # If user cancels save dialog, abort export.
        if not output_path:
            return

        try:
            with open(output_path, "w", newline="", encoding="utf-8-sig") as handle:
                # Write CSV header and all merged rows.
                writer = csv.DictWriter(handle, fieldnames=["Highlight", "Title", "Author", "Location", "Note"])
                writer.writeheader()
                writer.writerows(output_rows)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not write merged CSV:\n{exc}")
            return

        if csv_paths:
            messagebox.showinfo("Done", f"Merged {len(csv_paths)} files into {len(output_rows)} rows:\n{output_path}")
        else:
            messagebox.showinfo("Done", f"Exported {len(output_rows)} rows (links only, .addnote):\n{output_path}")


def run():
    """Launch the links-to-CSV GUI application.

    Returns:
        None: Starts Tkinter main loop.
    """
    root = tk.Tk()
    LinkCsvBuilderApp(root)
    root.mainloop()

