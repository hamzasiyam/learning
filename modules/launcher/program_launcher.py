"""GUI launcher for starting each tool module from one main window."""

import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox
from typing import List


@dataclass(frozen=True)
class Program:
    """Descriptor for one launchable program entry in the UI list."""

    key: str
    label: str
    module_name: str
    description: str


PROGRAMS: List[Program] = [
    Program("video_to_pdf", "Video + PDF Combiner", "apps.video_to_pdf", "Convert videos, merge PDFs, and select frames."),
    Program("video_vtt_to_pdf", "Video + VTT Timestamp PDF", "apps.video_vtt_to_pdf", "Extract only VTT timestamp frames and build subtitle PDF pages."),
    Program("pdf_merge", "PDF Merger", "apps.merge_pdf", "Merge and reorder PDF files."),
    Program("flashcards", "Flashcards CSV Merger", "apps.flashcards", "Combine CSV highlight/note files with title and author."),
    Program("links_csv", "Links + Notebook CSV Builder", "apps.image_links_and_notebooklm", "Merge image links with Location/Note CSV data."),
    Program("quiz_pdf", "Folder Watch -> Quiz PDF", "apps.quiz_pdf", "Watch a folder and compile new images into an optimized PDF."),
]


class ProgramLauncher(tk.Tk):
    """Main Tkinter launcher window used by `main.py`."""

    def __init__(self):
        """Initialize launcher window, sizing, and base repository path.

        Returns:
            None: Initializes GUI state.
        """
        super().__init__()
        self.title("Learning Tools Launcher")
        self.geometry("760x430")
        self.minsize(720, 380)

        self.repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._build_ui()

    def _build_ui(self):
        """Create all launcher widgets and bind events.

        Returns:
            None: Builds interface components.
        """
        header = tk.Frame(self, padx=12, pady=10)
        header.pack(fill="x")
        tk.Label(header, text="Select a program to launch", font=("", 14, "bold")).pack(anchor="w")
        tk.Label(header, text="Each tool opens in its own window/process.", fg="#666").pack(anchor="w", pady=(2, 0))

        body = tk.Frame(self, padx=12, pady=8)
        body.pack(fill="both", expand=True)

        self.program_list = tk.Listbox(body, activestyle="dotbox", exportselection=False)
        self.program_list.pack(side="left", fill="both", expand=True)
        self.program_list.bind("<<ListboxSelect>>", self._on_selection_changed)
        self.program_list.bind("<Double-Button-1>", lambda _e: self.launch_selected())

        scrollbar = tk.Scrollbar(body, orient="vertical", command=self.program_list.yview)
        scrollbar.pack(side="left", fill="y")
        self.program_list.configure(yscrollcommand=scrollbar.set)

        for idx, program in enumerate(PROGRAMS):
            self.program_list.insert(tk.END, f"{program.label}")
            self.program_list.itemconfig(idx, bg="#ffffff" if idx % 2 == 0 else "#f6f8fa")

        right = tk.Frame(body, padx=12)
        right.pack(side="left", fill="both")
        tk.Label(right, text="Description", font=("", 10, "bold")).pack(anchor="w")
        self.description_label = tk.Label(right, text="Choose a tool from the list.", justify="left", wraplength=300)
        self.description_label.pack(anchor="w", pady=(6, 18))
        self.launch_btn = tk.Button(right, text="Launch Selected", width=20, state="disabled", command=self.launch_selected)
        self.launch_btn.pack(anchor="w")
        tk.Button(right, text="Exit", width=20, command=self.destroy).pack(anchor="w", pady=(8, 0))

    def _selected_program(self):
        """Return the currently selected program descriptor, if any.

        Returns:
            Program | None: Selected program or None when no selection exists.
        """
        sel = self.program_list.curselection()
        # If nothing is selected yet, return None.
        if not sel:
            return None
        return PROGRAMS[sel[0]]

    def _on_selection_changed(self, _event=None):
        """Update description panel and launch button based on selection.

        Args:
            _event: Tkinter event object (unused).

        Returns:
            None: Updates widget state.
        """
        program = self._selected_program()
        # If no program is selected, show default guidance and disable launch.
        if not program:
            self.description_label.config(text="Choose a tool from the list.")
            self.launch_btn.config(state="disabled")
            return
        # Otherwise show selected program details and enable launch.
        self.description_label.config(text=program.description)
        self.launch_btn.config(state="normal")

    def launch_selected(self):
        """Launch the currently selected program in a separate Python process.

        Returns:
            None: Starts subprocess; shows error dialog on failure.
        """
        program = self._selected_program()
        # If no item is selected, there is nothing to launch.
        if not program:
            return
        try:
            # Run as a module so package imports resolve from repository root.
            subprocess.Popen([sys.executable, "-m", program.module_name], cwd=self.repo_root)
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))


def run():
    """Start the launcher application event loop.

    Returns:
        None: Opens GUI and blocks until the window is closed.
    """
    app = ProgramLauncher()
    app.mainloop()

