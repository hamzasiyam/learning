"""Video extraction utilities used by the video-to-PDF workflows."""

import os
import time
from contextlib import contextmanager
from typing import List, Tuple

import cv2
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .config import PRESETS
from .file_utils import cleanup_dir, format_seconds, sanitize_folder_name


@contextmanager
def suppress_ffmpeg_stderr():
    """Temporarily redirect stderr to null during noisy FFmpeg/OpenCV calls.

    Returns:
        contextmanager: Use with `with` to silence stderr in scope.
    """
    import os as _os
    import sys as _sys

    stderr_fd = _sys.stderr.fileno()
    with open(_os.devnull, "w") as devnull:
        backup_fd = _os.dup(stderr_fd)
        try:
            _os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            _os.dup2(backup_fd, stderr_fd)
            _os.close(backup_fd)


def resize_frame(frame, max_width):
    """Resize an image frame to fit a maximum width while preserving ratio.

    Args:
        frame: OpenCV image frame.
        max_width (int): Maximum width in pixels.

    Returns:
        np.ndarray: Original or resized frame.
    """
    h, w = frame.shape[:2]
    # If width is already acceptable, keep the original frame.
    if w <= max_width:
        return frame
    # Otherwise, scale proportionally by width ratio.
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def estimate_video_pdf_size(video_path, seconds_step, max_width, quality):
    """Estimate PDF byte size for one video given extraction settings.

    Args:
        video_path (str): Source video path.
        seconds_step (int): Sampling interval in seconds.
        max_width (int): Max output frame width.
        quality (int): JPEG quality used during extraction.

    Returns:
        float | None: Estimated bytes, or None if metadata cannot be read.
    """
    with suppress_ffmpeg_stderr():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total_f <= 0:
            return None
        duration = total_f / fps
        num_frames = int(duration // seconds_step) + 1
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or 720
        if w > max_width:
            scale = max_width / float(w)
            w, h = int(w * scale), int(h * scale)
        bytes_per_pixel = 0.10 * (quality / 60.0)
        return num_frames * max(1.0, w * h * bytes_per_pixel)
    finally:
        cap.release()


def video_to_temp_pdf(video_path, temp_pdf_path, preset_name, meta, progress_callback=None, stop_event=None):
    """Extract frames from video and write them into a temporary PDF file.

    Args:
        video_path (str): Input video path.
        temp_pdf_path (str): Output temporary PDF path.
        preset_name (str): Name of extraction preset from config.
        meta (dict): Metadata used for PDF title/author/subject fields.
        progress_callback: Optional callback for progress updates.
        stop_event: Optional threading event for cancellation.

    Returns:
        None: Writes a temporary PDF file to disk.
    """
    s = PRESETS[preset_name]
    seconds_step = int(s["seconds"])
    max_width = int(s["width"])
    quality = int(s["quality"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {os.path.basename(video_path)}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_f / fps) if (fps > 0 and total_f > 0) else 0.0
    targets = list(range(0, int(duration) + 1, seconds_step)) if duration > 0 else None

    frames_dir = os.path.join(os.path.dirname(temp_pdf_path), "_frames_" + str(int(time.time())))
    os.makedirs(frames_dir, exist_ok=True)

    try:
        with suppress_ffmpeg_stderr():
            t0 = time.time()
            img_paths: List[str] = []

            if targets is not None:
                # Preferred path: seek by timestamp and extract frame per step.
                if progress_callback:
                    progress_callback(0, len(targets), "Calculating time remaining...", f"Extracting frames: {os.path.basename(video_path)}")
                for idx, tsec in enumerate(targets, 1):
                    # If stop is requested, abort conversion.
                    if stop_event is not None and stop_event.is_set():
                        raise RuntimeError("Stopped")
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = resize_frame(frame, max_width)
                    img_path = os.path.join(frames_dir, f"t{tsec:06d}.jpg")
                    if cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                        img_paths.append(img_path)
                    if progress_callback:
                        elapsed = time.time() - t0
                        eta = f"Time remaining: {format_seconds(max(0, (len(targets) - idx) * elapsed / idx))}" if idx >= 2 else "Calculating time remaining..."
                        progress_callback(idx, len(targets), eta, f"Extracting frames: {os.path.basename(video_path)}")
            else:
                # Fallback path for videos without reliable duration metadata.
                idx = tsec = 0
                while True:
                    if stop_event is not None and stop_event.is_set():
                        raise RuntimeError("Stopped")
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = resize_frame(frame, max_width)
                    img_path = os.path.join(frames_dir, f"t{tsec:06d}.jpg")
                    if cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                        img_paths.append(img_path)
                    idx += 1
                    tsec += seconds_step
                    if progress_callback:
                        progress_callback(idx, max(1, idx), "Calculating time remaining...", f"Extracting frames: {os.path.basename(video_path)}")
            cap.release()

        img_paths.sort()
        # If extraction produced no images, fail fast with actionable message.
        if not img_paths:
            raise RuntimeError(f"No frames could be extracted from: {os.path.basename(video_path)}")

        c = canvas.Canvas(temp_pdf_path)
        c.setTitle(meta.get("title", ""))
        c.setAuthor(meta.get("author", ""))
        c.setSubject(meta.get("type", ""))
        for img_path in img_paths:
            # Check cancellation between page writes.
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Stopped")
            img = ImageReader(img_path)
            w, h = img.getSize()
            c.setPageSize((w, h))
            c.drawImage(img, 0, 0, width=w, height=h)
            c.showPage()
        c.save()
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cleanup_dir(frames_dir)


def estimate_total_frames(video_paths, preset_name):
    """Estimate total extracted frame count across multiple videos.

    Args:
        video_paths (list[str]): Input video files.
        preset_name (str): Preset key defining extraction interval.

    Returns:
        int: Estimated frame count used for progress scaling.
    """
    seconds_step = int(PRESETS[preset_name]["seconds"])
    total = 0
    for vp in video_paths:
        with suppress_ffmpeg_stderr():
            cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps > 0 and tf > 0:
                total += int(tf / fps) // seconds_step + 1
            else:
                total += 1
        finally:
            cap.release()
    return max(1, total)


def extract_all_frames_from_videos(video_paths, preset_name, output_dir, progress_callback=None, estimated_total=None, stop_event=None) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """Extract frames from many videos into organized folders.

    Args:
        video_paths (list[str]): Input video files.
        preset_name (str): Preset key from config.
        output_dir (str): Root output folder for extracted frames.
        progress_callback: Optional callback `(done, total, eta, status)`.
        estimated_total: Optional estimated total frame count.
        stop_event: Optional cancellation event.

    Returns:
        Tuple[List[str], List[Tuple[str, List[str]]]]:
            - Flat list of all frame paths.
            - Per-video mapping of video name to extracted frame paths.
    """
    s = PRESETS[preset_name]
    seconds_step = int(s["seconds"])
    max_width = int(s["width"])
    quality = int(s["quality"])

    all_paths: List[str] = []
    video_collections: List[Tuple[str, List[str]]] = []
    step_max = estimated_total or 1

    for v_idx, vp in enumerate(video_paths):
        # Stop immediately if cancellation is requested.
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped")
        with suppress_ffmpeg_stderr():
            cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {os.path.basename(vp)}")

        vname = os.path.basename(vp)
        folder = f"v{v_idx:03d}_{sanitize_folder_name(os.path.splitext(vname)[0])}"
        vdir = os.path.join(output_dir, folder)
        os.makedirs(vdir, exist_ok=True)
        vpaths: List[str] = []

        with suppress_ffmpeg_stderr():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = (tf / fps) if (fps > 0 and tf > 0) else 0.0

            if duration > 0:
                # Timestamp-based extraction for videos with valid metadata.
                for tsec in range(0, int(duration) + 1, seconds_step):
                    if stop_event is not None and stop_event.is_set():
                        raise RuntimeError("Stopped")
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = resize_frame(frame, max_width)
                    img_path = os.path.join(vdir, f"t{tsec:06d}.jpg")
                    if cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                        all_paths.append(img_path)
                        vpaths.append(img_path)
                    if progress_callback:
                        progress_callback(len(all_paths), step_max, "", f"Extracting: {vname}")
            else:
                # Frame-by-frame fallback if duration metadata is unavailable.
                next_t = 0
                fc = 0
                fps = fps or 25.0
                while True:
                    if stop_event is not None and stop_event.is_set():
                        raise RuntimeError("Stopped")
                    ret, frame = cap.read()
                    if not ret:
                        break
                    pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                    pos_sec = (pos / 1000.0) if pos >= 0 else (fc / fps)
                    if pos_sec >= next_t:
                        frame = resize_frame(frame, max_width)
                        img_path = os.path.join(vdir, f"t{int(next_t):06d}.jpg")
                        if cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                            all_paths.append(img_path)
                            vpaths.append(img_path)
                        next_t += seconds_step
                        if progress_callback:
                            progress_callback(len(all_paths), step_max, "", f"Extracting: {vname}")
                    fc += 1
            cap.release()

        video_collections.append((vname, vpaths))

    return all_paths, video_collections

