# Learning tools

A set of desktop tools for turning videos and documents into PDFs, merging PDFs/CSVs, and building PDFs from images. All programs use a graphical interface (Tkinter).

---

## Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt** includes: `opencv-python`, `reportlab`, `pypdf`, `pandas`.

- **quiz_pdf.py** also needs **Pillow** (`pip install pillow`).
- **merge_pdf.py** also needs **tkinterdnd2** for drag-and-drop (`pip install tkinterdnd2`).

---

## Programs

### 1. Video to PDF (`video_to_pdf.py`)

Turns videos (and optional PDFs) into one combined PDF, with optional frame selection and metadata for Readwise Reader.

**Run:** `python video_to_pdf.py`

**What it does:**

- **Queue:** Add video files (mp4, avi, mov, mkv, m4v) and/or existing PDFs. Order is kept in the final PDF.
- **Presets:** Choose how often to sample frames and at what resolution/quality, e.g.:
  - **3 sec (Original resolution)** ? one frame every 3 seconds, full resolution, quality 100 (default).
  - **1 sec (Original resolution)** ? one frame per second, full resolution.
  - Other presets (8 sec, 10 sec, Lecture, etc.) trade off file size vs quality.
- **Metadata (optional):** Title, Author, Subject, Keywords, Creator, Producer. Used in the PDF and as the suggested filename (Windows-safe). Good for Readwise Reader.
- **Two ways to build a PDF:**
  1. **Generate Combined PDF** ? Converts each video to a temporary PDF (using the preset), merges everything in queue order, writes one PDF. No frame selection.
  2. **Extract Frames & Select** ? Extracts frames from all videos (using the preset), then opens a **selection window** where you:
     - See frames in a grid (8, 12, or 15 per page; choose in the dropdown).
     - Use **? / ?** to change pages; **left-click** or checkbox to mark which frames to **keep** (default: all unchecked; you opt in).
     - Use **Keep This Page** / **Skip This Page** to quickly include or remove every frame shown on the current page.
     - **Right-click** a thumbnail to zoom in; right-click again (on zoom or any thumbnail) to close zoom.
     - Click **Create PDF** to build a PDF from the selected frames only. Each page has the image plus a black bar with **#1**, **#2**, … in order.
     - Optional: enable **Split output into parts** and set a target size (default **10 MB**). The tool exports one part at a time and keeps frame numbering continuous across parts (e.g. part 2 can start at #241).
     - While exporting, a progress window shows page-by-page progress so you can see it is still running.
- **Output:** One PDF. Filename is derived from the title you enter. Frames are at the resolution/quality of the chosen preset (e.g. original resolution and quality 100 for the ?Original resolution? presets). Extraction uses seek-based reading when possible so it?s faster without losing quality or skipping intervals.

---

### 2. Merge CSV (`merge_csv`)

Merges several CSV files into one CSV with columns: **Title**, **Author**, **Highlight**, **Note**. Useful for combining Readwise-style highlight exports.

**Run:** `python merge_csv` (no `.py` extension; it?s a Python script)

**What it does:**

- You enter **Title** and **Author** (required).
- You add one or more CSV files. Each file is read with **no header row** (first row is data). The first two columns are treated as Highlight and Note.
- Notes that start with a dot get an **?Answer: ?** prefix. Cells that look like formulas (`=`, `+`, `-`, `@`) are escaped so spreadsheets don?t evaluate them.
- **Output:** A single CSV is saved in the **same folder as the first file** you selected. The filename is **the title you entered** (made safe for Windows). You don?t pick the path; it?s automatic from the title.

---

### 3. Merge PDF (`merge_pdf.py`)

Merges multiple PDFs into one, in the order you choose. Supports **drag-and-drop** to reorder.

**Run:** `python merge_pdf.py`

**What it does:**

- **Add PDFs** to a list. You can **drag and drop** files in the list to reorder them.
- **Remove Selected** to drop a PDF from the list.
- **Merge PDFs** ? You choose where to save. The app concatenates all listed PDFs in order into a single file and shows a progress bar.
- **Open File Location** ? Opens the folder containing the merged PDF (e.g. in Explorer on Windows).

**Requires:** `tkinterdnd2` for drag-and-drop (`pip install tkinterdnd2`).

---

### 4. Quiz PDF / Folder watch ? Images to PDF (`quiz_pdf.py`)

Watches a folder for image files and can compile them into one PDF with size optimization and a size estimate.

**Run:** `python quiz_pdf.py`

**What it does:**

- **Watch folder:** You select a folder. The app can watch for **new** image files (e.g. screenshots, photos) appearing in that folder.
- **Presets:** ?High Quality?, ?Balanced?, or ?Small? control max DPI, JPEG quality, and subsampling so you can balance quality vs file size.
- **Options:** Fit to page, avoid upscaling, etc.
- **Estimated PDF size:** Before building, it estimates the final PDF size (using a background pass over the images) so you know what to expect.
- **Compile:** Builds one PDF from the current (or newly added) images in the folder, with EXIF rotation and transparency flattened to white. Output is optimized JPEGs in the PDF.

**Requires:** `pillow` and `reportlab` (`pip install pillow reportlab`).

---

### 5. Links to CSV (`links_to_csv.py`)

Merges your pasted image links with one or more CSV sheets containing **Location** and **Note**.

**Run:** `python links_to_csv.py`

**What it does:**

- Paste image links (one per line), e.g. `https://i.ibb.co/HfL7smbV/Page-1.png`.
- Click **Merge with Location/Note CSVs** and select multiple CSV files.
- Each selected CSV row is read from `Location` and `Note` columns (rows with partial data are still included).
- Notes with multiple `•` bullets are auto-formatted with a blank line between bullets in output.
- Rows with an empty note are automatically filled with `.add note`.
- The app matches `Location` to page number from link filename (`Page-1` -> `1`).
- Output CSV columns are: **Highlight**, **Title**, **Author**, **Location**, **Note** (where Highlight is `![](link)`).
- The export keeps pasted-link order, always outputs one row per pasted link, includes unmatched CSV rows afterward with blank Highlight, and writes Title/Author on every row.

---

## Summary

| Program        | Purpose |
|----------------|--------|
| **video_to_pdf.py** | Videos (+ optional PDFs) ? one PDF; optional frame selection and numbered pages; metadata for Readwise. |
| **merge_csv**       | Multiple CSVs ? one CSV (Title, Author, Highlight, Note). |
| **merge_pdf.py**    | Multiple PDFs ? one PDF; drag-and-drop reorder. |
| **quiz_pdf.py**     | Folder of images ? one PDF with size estimate and quality presets. |
| **links_to_csv.py** | Pasted image links + multiple Location/Note CSVs ? merged CSV with Highlight, Title, Author, Location, and Note columns. |
