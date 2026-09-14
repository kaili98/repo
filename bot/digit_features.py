import cv2
import numpy as np

# HOG-like feature extraction (gradient-orientation histograms per cell),
# implemented directly on top of cv2 (already a hard dependency) rather than
# adding scikit-image, since these glyphs are tiny (24x24 canonicalized) and
# don't need a full HOG implementation's windowing/block-normalization.
#
# The point of switching from raw pixel correlation (cv2.matchTemplate) to
# this is invariance: two real captures of the same digit rendered at
# different pixel widths (confirmed repeatedly - e.g. a "3" at 7px vs 10px
# wide) look almost nothing alike under pixel correlation, but produce
# similar gradient-orientation histograms once canonicalized to a common
# size, since the *shape* of the strokes is what's being measured instead of
# exact pixel positions.
CELL_SIZE = 4
N_BINS = 9


def hog_features(canon_gray: np.ndarray) -> np.ndarray:
    """canon_gray must already be canonicalized (see arithmetic_puzzle._canonicalize)
    to a fixed square size divisible by CELL_SIZE."""
    img = canon_gray.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0  # unsigned gradient direction

    h, w = img.shape[:2]
    cells_y, cells_x = h // CELL_SIZE, w // CELL_SIZE
    bin_width = 180.0 / N_BINS
    bin_idx = np.minimum((ang / bin_width).astype(np.int32), N_BINS - 1)

    cells = []
    for cy in range(cells_y):
        y0, y1 = cy * CELL_SIZE, (cy + 1) * CELL_SIZE
        for cx in range(cells_x):
            x0, x1 = cx * CELL_SIZE, (cx + 1) * CELL_SIZE
            cell_mag = mag[y0:y1, x0:x1]
            cell_bin = bin_idx[y0:y1, x0:x1]
            hist = np.zeros(N_BINS, dtype=np.float32)
            np.add.at(hist, cell_bin.ravel(), cell_mag.ravel())
            cells.append(hist)

    feat = np.concatenate(cells)
    norm = np.linalg.norm(feat)
    if norm > 1e-6:
        feat = feat / norm
    return feat
