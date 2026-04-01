"""Perceptual-hash based similarity grouping utilities for extracted frames."""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .config import (
    DEFAULT_THRESHOLD,
    PHASH_SIZE,
    SIM_COLOR_SIMILAR,
    SIM_COLOR_SOMEWHAT_SIMILAR,
    SIM_COLOR_UNIQUE,
    SIM_COLOR_VERY_SIMILAR,
)


def similarity_badge_color(similarity_percent: float) -> str:
    """Map similarity percentage to UI badge color.

    Args:
        similarity_percent (float): Similarity score in percent.

    Returns:
        str: Hex color string used in frame selection UI.
    """
    # Very high similarity gets green.
    if similarity_percent >= 90:
        return SIM_COLOR_VERY_SIMILAR
    # Medium-high similarity gets yellow.
    if similarity_percent >= 75:
        return SIM_COLOR_SIMILAR
    # Lower-but-related similarity gets orange.
    if similarity_percent >= 50:
        return SIM_COLOR_SOMEWHAT_SIMILAR
    # Everything else is treated as unique/neutral.
    return SIM_COLOR_UNIQUE


def compute_phash(img_path: str) -> Optional[np.ndarray]:
    """Compute a perceptual hash for an image using DCT.

    Args:
        img_path (str): Path to source image.

    Returns:
        Optional[np.ndarray]: Flattened boolean hash vector, or None on failure.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # If image fails to load, return None for caller-side handling.
        if img is None:
            return None
        # Resize to stable dimensions before computing frequency-domain hash.
        img = cv2.resize(img, (PHASH_SIZE * 4, PHASH_SIZE * 4), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(img))
        dct_low = dct[:PHASH_SIZE, :PHASH_SIZE]
        # Build binary hash by thresholding low-frequency coefficients.
        mean_val = (dct_low.sum() - dct_low[0, 0]) / (PHASH_SIZE * PHASH_SIZE - 1)
        return (dct_low > mean_val).flatten()
    except Exception:
        return None


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """Count bit differences between two perceptual hashes.

    Args:
        h1 (np.ndarray): First hash.
        h2 (np.ndarray): Second hash.

    Returns:
        int: Hamming distance.
    """
    return int(np.count_nonzero(h1 != h2))


def similarity_percent(hamming: int, bits: int = PHASH_SIZE * PHASH_SIZE) -> float:
    """Convert Hamming distance into similarity percentage.

    Args:
        hamming (int): Number of differing bits.
        bits (int): Total bit count in hash.

    Returns:
        float: Similarity percentage rounded to one decimal.
    """
    return round(100.0 * (1.0 - hamming / bits), 1)


@dataclass
class SimilarityGroup:
    """Container describing one cluster of visually similar frames."""

    group_id: int
    frame_paths: List[str] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)
    representative_path: str = ""

    @property
    def is_unique(self) -> bool:
        return len(self.frame_paths) == 1

    @property
    def average_similarity(self) -> float:
        if len(self.similarities) <= 1:
            return 100.0
        return round(sum(self.similarities[1:]) / (len(self.similarities) - 1), 1)

    @property
    def last_frame_path(self) -> str:
        return self.frame_paths[-1] if self.frame_paths else ""


def cluster_frames_by_similarity(frame_paths: List[str], threshold: int = DEFAULT_THRESHOLD, progress_callback=None) -> List[SimilarityGroup]:
    """Group frame paths by visual similarity using representative hashes.

    Args:
        frame_paths (List[str]): Ordered image paths from extraction step.
        threshold (int): Maximum Hamming distance to consider frames similar.
        progress_callback: Optional callback `(done, total, message)`.

    Returns:
        List[SimilarityGroup]: Grouped frame clusters in traversal order.
    """
    # If no frames were provided, nothing to cluster.
    if not frame_paths:
        return []

    total = len(frame_paths)
    hashes: List[Optional[np.ndarray]] = []

    for i, path in enumerate(frame_paths):
        hashes.append(compute_phash(path))
        # Emit lightweight progress updates during hash computation.
        if progress_callback and (i % max(1, total // 20) == 0):
            progress_callback(i, total, f"Analysing frame {i + 1} of {total}")

    groups: List[SimilarityGroup] = []
    representative_hashes: List[Optional[np.ndarray]] = []

    for i, path in enumerate(frame_paths):
        h = hashes[i]
        # If hashing failed, treat frame as its own standalone group.
        if h is None:
            g = SimilarityGroup(group_id=len(groups), frame_paths=[path], similarities=[100.0], representative_path=path)
            groups.append(g)
            representative_hashes.append(None)
            continue

        best_group_index = None
        best_distance = threshold + 1

        for gi, rep_hash in enumerate(representative_hashes):
            if rep_hash is None:
                continue
            distance = hamming_distance(h, rep_hash)
            # Pick the closest group still under threshold.
            if distance <= threshold and distance < best_distance:
                best_distance = distance
                best_group_index = gi

        if best_group_index is not None:
            # Add current frame to the closest matching group.
            groups[best_group_index].frame_paths.append(path)
            groups[best_group_index].similarities.append(similarity_percent(best_distance))
        else:
            # No existing match; create a new representative group.
            g = SimilarityGroup(group_id=len(groups), frame_paths=[path], similarities=[100.0], representative_path=path)
            groups.append(g)
            representative_hashes.append(h)

    if progress_callback:
        progress_callback(total, total, "Done")

    return groups

