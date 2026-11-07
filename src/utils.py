import cv2, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def load_image(path: Path) -> np.ndarray:
    """Read an image (GeoTIFF/JP2/PNG) and convert to RGB float32 [0,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

def register_images(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Align *mov* to *ref* using ECC (enhanced correlation coefficient)."""
    ref_gray = cv2.cvtColor((ref * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mov_gray = cv2.cvtColor((mov * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
    _, warp_matrix = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix,
                                          cv2.MOTION_HOMOGRAPHY, criteria)
    aligned = cv2.warpPerspective(mov, warp_matrix,
                                  (ref.shape[1], ref.shape[0]),
                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned

def compute_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    diff = np.abs(img2 - img1)
    return np.dot(diff, [0.2989, 0.5870, 0.1140])

def apply_threshold(diff: np.ndarray, method: str = "otsu", value: float = None) -> np.ndarray:
    diff_uint8 = (diff * 255).astype(np.uint8)
    if method == "otsu":
        _, mask = cv2.threshold(diff_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "fixed" and value is not None:
        _, mask = cv2.threshold(diff_uint8, int(value * 255), 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("Unsupported threshold method")
    return mask // 255

def clean_mask(mask: np.ndarray, min_area: int = 30) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1
    return cleaned
