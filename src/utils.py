import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import base64
from io import BytesIO

COLORMAP_DICT = {
    'inferno': cv2.COLORMAP_INFERNO,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'jet': cv2.COLORMAP_JET,
    'turbo': cv2.COLORMAP_TURBO,
    'hot': cv2.COLORMAP_HOT,
    'magma': cv2.COLORMAP_MAGMA,
}

OVERLAY_COLORS = {
    'red': [255, 51, 68],        # #ff3344
    'green': [0, 230, 118],      # #00e676
    'cyan': [0, 229, 255],       # #00e5ff
    'yellow': [255, 234, 0],     # #ffea00
    'magenta': [255, 0, 128],    # #ff0080
    'orange': [255, 145, 0],     # #ff9100
    'sage': [93, 123, 111],      # #5D7B6F
    'cream': [234, 231, 214],    # #EAE7D6
}

def load_image(path: Path) -> np.ndarray:
    """Read an image (GeoTIFF/JP2/PNG/JPG) and convert to RGB float32 [0,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = img[..., :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def load_image_from_bytes(data_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into RGB float32 [0,1]."""
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode uploaded image data.")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = img[..., :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def register_images(ref: np.ndarray, mov: np.ndarray, motion_type: str = "homography") -> tuple[np.ndarray, bool, str]:
    """
    Align *mov* to *ref* using ECC (enhanced correlation coefficient).
    Returns (aligned_image, success_boolean, status_message).
    """
    # Ensure identical shape before registration
    if ref.shape[:2] != mov.shape[:2]:
        mov = cv2.resize(mov, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)

    ref_gray = cv2.cvtColor((np.clip(ref, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mov_gray = cv2.cvtColor((np.clip(mov, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    if motion_type == "affine":
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif motion_type == "euclidean":
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3000, 1e-5)

    try:
        _, warp_matrix = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix, warp_mode, criteria)
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(
                mov, warp_matrix, (ref.shape[1], ref.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
        else:
            aligned = cv2.warpAffine(
                mov, warp_matrix, (ref.shape[1], ref.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
        return aligned, True, f"ECC {motion_type.capitalize()} converged successfully"
    except Exception as e:
        return mov, False, f"Alignment fallback (ECC did not converge: {str(e)[:40]})"

def compute_difference(img1: np.ndarray, img2: np.ndarray, method: str = "luminance") -> np.ndarray:
    """
    Compute pixel difference between two images.
    Methods:
    - 'luminance': Perceptual weighted grayscale difference
    - 'euclidean': L2 norm across RGB channels normalized to [0, 1]
    """
    # Resize img2 to img1 if dimensions differ
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)

    diff = np.abs(img2 - img1)
    if method == "euclidean":
        diff_val = np.sqrt(np.sum(diff ** 2, axis=2)) / np.sqrt(3.0)
    else:
        diff_val = np.dot(diff, [0.2989, 0.5870, 0.1140])
    return np.clip(diff_val, 0.0, 1.0)

def apply_threshold(diff: np.ndarray, method: str = "otsu", value: float = None) -> np.ndarray:
    """
    Threshold difference image to produce binary mask [0, 1].
    Maintains compatibility with legacy signature.
    """
    mask, _ = apply_threshold_with_val(diff, method=method, value=value)
    return mask

def apply_threshold_with_val(diff: np.ndarray, method: str = "otsu", value: float = None) -> tuple[np.ndarray, float]:
    """
    Threshold difference image and return both the binary mask and the normalized threshold cutoff (0.0 to 1.0).
    """
    diff_uint8 = np.clip(diff * 255.0, 0, 255).astype(np.uint8)

    if method == "otsu":
        otsu_val, mask = cv2.threshold(diff_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cutoff = float(otsu_val) / 255.0
    elif method == "triangle":
        tri_val, mask = cv2.threshold(diff_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        cutoff = float(tri_val) / 255.0
    elif method == "fixed" and value is not None:
        cutoff = float(np.clip(value, 0.0, 1.0))
        _, mask = cv2.threshold(diff_uint8, int(cutoff * 255), 255, cv2.THRESH_BINARY)
    else:
        otsu_val, mask = cv2.threshold(diff_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cutoff = float(otsu_val) / 255.0

    return (mask // 255).astype(np.uint8), cutoff

def clean_mask(mask: np.ndarray, min_area: int = 30, kernel_size: int = 3, morph_op: str = "open_close") -> np.ndarray:
    """
    Clean mask using morphological operations and connected-component area filtering.
    """
    cleaned, _ = clean_mask_with_stats(mask, min_area=min_area, kernel_size=kernel_size, morph_op=morph_op)
    return cleaned

def clean_mask_with_stats(mask: np.ndarray, min_area: int = 30, kernel_size: int = 3, morph_op: str = "open_close") -> tuple[np.ndarray, dict]:
    """
    Clean mask and return blob morphology statistics.
    """
    k = max(1, kernel_size)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    working_mask = mask.astype(np.uint8)
    if morph_op == "open_close":
        opened = cv2.morphologyEx(working_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        morphed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    elif morph_op == "open":
        morphed = cv2.morphologyEx(working_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    elif morph_op == "close":
        morphed = cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        morphed = working_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)
    cleaned = np.zeros_like(morphed, dtype=np.uint8)
    
    blob_areas = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == i] = 1
            blob_areas.append(area)

    stats_dict = {
        "blob_count": len(blob_areas),
        "blob_areas": blob_areas,
        "max_blob_area": int(max(blob_areas)) if blob_areas else 0,
        "mean_blob_area": round(float(np.mean(blob_areas)), 1) if blob_areas else 0.0
    }
    return cleaned, stats_dict

def create_heatmap(diff: np.ndarray, colormap: str = "inferno") -> np.ndarray:
    """Convert a difference float array [0,1] to an RGB heatmap image [0,255]."""
    diff_uint8 = np.clip(diff * 255.0, 0, 255).astype(np.uint8)
    cv_cmap = COLORMAP_DICT.get(colormap.lower(), cv2.COLORMAP_INFERNO)
    heatmap_bgr = cv2.applyColorMap(diff_uint8, cv_cmap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

def create_overlay(image_rgb: np.ndarray, mask: np.ndarray, color_name: str = "red", alpha: float = 0.55, draw_contours: bool = True) -> np.ndarray:
    """
    Create an alpha-blended overlay of the change mask on top of image_rgb.
    image_rgb: float32 [0,1] or uint8 [0,255] RGB.
    Returns uint8 RGB [0,255].
    """
    if image_rgb.dtype == np.float32 or image_rgb.dtype == np.float64:
        base = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8).copy()
    else:
        base = image_rgb.astype(np.uint8).copy()

    if base.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)

    color_rgb = OVERLAY_COLORS.get(color_name.lower(), OVERLAY_COLORS['red'])
    alpha = float(np.clip(alpha, 0.0, 1.0))

    overlay = base.copy()
    mask_bool = (mask > 0)
    for c in range(3):
        overlay[mask_bool, c] = np.clip(
            (1.0 - alpha) * base[mask_bool, c] + alpha * color_rgb[c], 0, 255
        ).astype(np.uint8)

    if draw_contours and np.any(mask_bool):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_rgb, 2)

    return overlay

def compute_histogram(diff: np.ndarray, bins: int = 64) -> dict:
    """Compute difference intensity histogram for Chart.js rendering."""
    diff_uint8 = np.clip(diff * 255.0, 0, 255).astype(np.uint8)
    hist, bin_edges = np.histogram(diff_uint8, bins=bins, range=(0, 256))
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).tolist()
    return {
        "bin_centers": [round(b, 1) for b in bin_centers],
        "counts": hist.tolist()
    }

def to_base64_png(img: np.ndarray) -> str:
    """Encode an RGB or Grayscale image (uint8 or float) to a PNG base64 Data URI."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)

    if len(img_uint8.shape) == 3:
        bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    else:
        bgr = img_uint8

    success, buffer = cv2.imencode('.png', bgr)
    if not success:
        raise ValueError("Failed to encode image to PNG format.")
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

def generate_summary_figure_bytes(img1: np.ndarray, img2: np.ndarray, diff: np.ndarray, mask: np.ndarray, overlay: np.ndarray, metrics: dict = None) -> bytes:
    """Generate a publication-ready 5-panel composite figure and return raw PNG bytes."""
    fig, axs = plt.subplots(1, 5, figsize=(22, 4.5), dpi=160)
    
    axs[0].imshow(np.clip(img1, 0, 1))
    axs[0].set_title('Date 1 (Before)', fontsize=11, fontweight='bold', pad=8)
    
    axs[1].imshow(np.clip(img2, 0, 1))
    axs[1].set_title('Date 2 (After / Aligned)', fontsize=11, fontweight='bold', pad=8)
    
    im = axs[2].imshow(diff, cmap='inferno')
    axs[2].set_title('Difference Intensity', fontsize=11, fontweight='bold', pad=8)
    plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    
    axs[3].imshow(mask, cmap='gray')
    change_pct = metrics.get('changed_percentage', 0.0) if metrics else 0.0
    axs[4].imshow(overlay)
    axs[3].set_title(f'Change Mask ({change_pct:.1f}%)', fontsize=11, fontweight='bold', pad=8)
    axs[4].set_title('Change Overlay', fontsize=11, fontweight='bold', pad=8)

    for ax in axs:
        ax.axis('off')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
