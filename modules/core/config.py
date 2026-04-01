"""Central configuration constants and presets for video/PDF processing."""

ORIGINAL_RES_WIDTH = 99999

PRESETS = {
    "Every 3 seconds — original resolution": {"seconds": 3, "width": ORIGINAL_RES_WIDTH, "quality": 100},
    "Every 1 second — original resolution": {"seconds": 1, "width": ORIGINAL_RES_WIDTH, "quality": 100},
    "Every 8 seconds — small file (~15 MB)": {"seconds": 8, "width": 480, "quality": 38},
    "Every 8 seconds — sharp text": {"seconds": 8, "width": 1280, "quality": 72},
    "Every 8 seconds — smallest size (slides)": {"seconds": 8, "width": 1024, "quality": 62},
    "Every 10 seconds — lower resolution": {"seconds": 10, "width": 1024, "quality": 58},
    "Every 15 seconds — recommended for lectures": {"seconds": 15, "width": 1400, "quality": 68},
    "Every 30 seconds — high compression": {"seconds": 30, "width": 900, "quality": 48},
    "Every 5 seconds — detailed": {"seconds": 5, "width": 1600, "quality": 75},
}

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v")
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp")

PHASH_SIZE = 16
DEFAULT_THRESHOLD = 10

SIM_COLOR_REPRESENTATIVE = "#d0e8ff"
SIM_COLOR_VERY_SIMILAR = "#c8f5c8"
SIM_COLOR_SIMILAR = "#fff3c0"
SIM_COLOR_SOMEWHAT_SIMILAR = "#ffe0c0"
SIM_COLOR_UNIQUE = "#f0f0f0"

OPTIMIZE_MAX_WIDTH = 1600
OPTIMIZE_QUALITY = 82

