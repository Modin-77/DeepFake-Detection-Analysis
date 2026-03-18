"""
Data preprocessing utilities for Deepfake Detection.

Provides:
- Histogram equalization (CLAHE and standard) to improve image contrast.
- Sharpening kernels (standard, strong, and edge-enhance) to highlight
  facial details that may distinguish real from synthetic faces.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Histogram Equalization
# ---------------------------------------------------------------------------

def apply_histogram_equalization(image: np.ndarray, method: str = "clahe",
                                  clip_limit: float = 2.0,
                                  tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply histogram equalization to improve image contrast.

    Parameters
    ----------
    image : np.ndarray
        Input image as a NumPy array (BGR or grayscale, uint8).
    method : str, optional
        Equalization method to use. One of:
        - ``"clahe"``  : Contrast Limited Adaptive Histogram Equalization
                         (default, works per channel on color images).
        - ``"global"`` : Standard global histogram equalization.
    clip_limit : float, optional
        Threshold for contrast limiting used in CLAHE (default: 2.0).
    tile_grid_size : tuple, optional
        Size of the grid for CLAHE tiles (default: ``(8, 8)``).

    Returns
    -------
    np.ndarray
        Contrast-enhanced image with the same shape and dtype as *image*.

    Raises
    ------
    ValueError
        If *method* is not one of ``"clahe"`` or ``"global"``.
    """
    if method not in ("clahe", "global"):
        raise ValueError(f"Unknown method '{method}'. Choose 'clahe' or 'global'.")

    if image.ndim == 2:
        # Grayscale image
        return _equalize_single_channel(image, method, clip_limit, tile_grid_size)

    # Color image – operate in YCrCb color space to equalize only luminance
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _equalize_single_channel(
        ycrcb[:, :, 0], method, clip_limit, tile_grid_size
    )
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _equalize_single_channel(channel: np.ndarray, method: str,
                              clip_limit: float,
                              tile_grid_size: tuple) -> np.ndarray:
    """Internal helper: equalize a single-channel (grayscale) image."""
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                 tileGridSize=tile_grid_size)
        return clahe.apply(channel)
    # method == "global"
    return cv2.equalizeHist(channel)


# ---------------------------------------------------------------------------
# Sharpening Kernels
# ---------------------------------------------------------------------------

# Pre-defined sharpening kernels
SHARPENING_KERNELS = {
    "standard": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=np.float32),

    "strong": np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1],
    ], dtype=np.float32),

    "edge_enhance": np.array([
        [ 0,  0,  0],
        [-1,  2, -1],
        [ 0,  0,  0],
    ], dtype=np.float32),
}


def apply_sharpening(image: np.ndarray,
                     kernel_name: str = "standard") -> np.ndarray:
    """Apply a sharpening kernel to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image as a NumPy array (BGR or grayscale, uint8).
    kernel_name : str, optional
        Name of the sharpening kernel. One of:
        - ``"standard"``     : Mild Laplacian-based sharpening (default).
        - ``"strong"``       : Aggressive sharpening that emphasises edges.
        - ``"edge_enhance"`` : Horizontal-edge enhancement.

    Returns
    -------
    np.ndarray
        Sharpened image clipped to [0, 255] with dtype uint8.

    Raises
    ------
    ValueError
        If *kernel_name* is not a recognised key.
    """
    if kernel_name not in SHARPENING_KERNELS:
        valid = ", ".join(f"'{k}'" for k in SHARPENING_KERNELS)
        raise ValueError(
            f"Unknown kernel '{kernel_name}'. Valid options are: {valid}."
        )

    kernel = SHARPENING_KERNELS[kernel_name]
    sharpened = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
