import csv
import re
import tkinter as tk
from tkinter import filedialog, messagebox

FORMULA_PREFIXES = ("=", "+", "-", "@")


def sanitize_filename(name: str, default: str = "highlights") -> str:
    if not name or not name.strip():
        return default
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or default


def link_to_markdown_image(url: str) -> str:
    return f"![]({url})"


def sanitize_csv_text(value: str) -> str:
    """Normalize free text so it is safe to write into CSV cells."""
    if not value:
        return ""
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.startswith(FORMULA_PREFIXES):
        cleaned = "'" + cleaned
    return cleaned


class LinkCsvBuilderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Link List to CSV")
        self.root.geometry("760x560")

        self.title_var = tk.StringVar()
        self.author_var = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        meta = tk.Frame(self.root)
        meta.pack(fill="x", padx=12, pady=(12, 6))

        tk.Label(meta, text="Title").grid(row=0, column=0, sticky="w")
        tk.Entry(meta, textvariable=self.title_var, width=72).grid(
            row=0, column=1, padx=8, pady=3, sticky="w"
        )

        tk.Label(meta, text="Author").grid(row=1, column=0, sticky="w")
        tk.Entry(meta, textvariable=self.author_var, width=72).grid(
            row=1, column=1, padx=8, pady=3, sticky="w"
        )

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=12, pady=6)

        tk.Label(body, text="Paste links below (one per line)").pack(anchor="w")

        self.links_text = tk.Text(body, wrap="word", height=18)
        self.links_text.pack(fill="both", expand=True, pady=(4, 8))

        hint = (
            "Each link becomes a Highlight entry in this format: ![](https://...png).\n"
            "CSV columns are: Highlight, Title."
        )
        tk.Label(body, text=hint, fg="#555").pack(anchor="w")

        actions = tk.Frame(self.root)
        actions.pack(fill="x", padx=12, pady=12)

        tk.Button(actions, text="Build CSV", width=16, command=self.build_csv).pack(
            side="left"
        )
        tk.Button(actions, text="Clear", width=12, command=self.clear_all).pack(
            side="left", padx=8
        )

    def clear_all(self):
        self.title_var.set("")
        self.author_var.set("")
        self.links_text.delete("1.0", tk.END)

    def build_csv(self):
        raw_title = self.title_var.get()
        title = sanitize_csv_text(raw_title)
        author = sanitize_csv_text(self.author_var.get())

        if not title:
            messagebox.showerror("Missing title", "Please enter a title.")
            return

        raw_lines = self.links_text.get("1.0", tk.END).splitlines()
        links = [line.strip() for line in raw_lines if line.strip()]

        if not links:
            messagebox.showerror("No links", "Please paste at least one link.")
            return

        title_value = f"{title} — {author}" if author else title

        default_name = sanitize_filename(raw_title) + ".csv"
        output_path = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")],
        )
        if not output_path:
            return

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Highlight", "Title"])
                writer.writeheader()
                for link in links:
                    writer.writerow(
                        {
                            "Highlight": sanitize_csv_text(link_to_markdown_image(link)),
                            "Title": sanitize_csv_text(title_value),
                        }
                    )
        except Exception as exc:
            messagebox.showerror("Error", f"Could not write CSV:\n{exc}")
            return

        messagebox.showinfo(
            "Done",
            f"CSV created successfully with {len(links)} rows:\n{output_path}",
        )


if __name__ == "__main__":
    root = tk.Tk()
    LinkCsvBuilderApp(root)
    root.mainloop()
