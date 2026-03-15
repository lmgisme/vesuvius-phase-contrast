"""Contrast metrics for evaluating phase retrieval without ink labels.

Quantifies edge enhancement, material separability, and fringe properties
using only the CT image structure (papyrus layers, air gaps, edges).
"""
import numpy as np
from scipy import ndimage
from sklearn.mixture import GaussianMixture


def edge_strength(image: np.ndarray, mask: np.ndarray = None,
                  percentile: float = 95) -> float:
    """Gradient magnitude at given percentile (higher = more edge fringes).

    Args:
        image: 2D float array.
        mask: Boolean mask of valid pixels. If None, uses image > 0.
        percentile: Which percentile of gradient magnitude to return.
    """
    if mask is None:
        mask = image > 0
    gy = ndimage.sobel(image, axis=0)
    gx = ndimage.sobel(image, axis=1)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.percentile(grad_mag[mask], percentile))


def bimodal_snr(image: np.ndarray, mask: np.ndarray = None) -> float:
    """SNR between two dominant intensity populations (papyrus vs air/void).

    Fits a 2-component Gaussian mixture to the intensity histogram.
    Returns |mu1 - mu2| / sqrt((sigma1^2 + sigma2^2) / 2).
    Higher = better material separation.
    """
    if mask is None:
        mask = image > 0
    values = image[mask].ravel().astype(np.float64)
    if len(values) < 100:
        return 0.0

    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(values.reshape(-1, 1))
    mu = gmm.means_.ravel()
    sigma = np.sqrt(gmm.covariances_.ravel())

    return float(abs(mu[0] - mu[1]) / np.sqrt((sigma[0] ** 2 + sigma[1] ** 2) / 2))


def local_contrast(image: np.ndarray, mask: np.ndarray = None,
                   window: int = 32) -> float:
    """Mean local contrast (std/mean) in non-masked windows."""
    if mask is None:
        mask = image > 0
    ny, nx = image.shape
    contrasts = []
    for y in range(0, ny - window, window):
        for x in range(0, nx - window, window):
            patch = image[y:y + window, x:x + window]
            patch_mask = mask[y:y + window, x:x + window]
            if patch_mask.sum() < window * window * 0.9:
                continue  # skip patches that are mostly masked
            vals = patch[patch_mask]
            m = vals.mean()
            if m > 1e-6:
                contrasts.append(vals.std() / m)
    if not contrasts:
        return 0.0
    return float(np.mean(contrasts))


def line_profile(image: np.ndarray, y_start: int, y_end: int,
                 x_pos: int) -> np.ndarray:
    """Extract a vertical line profile at a given x position."""
    return image[y_start:y_end, x_pos].astype(np.float64)


def fringe_amplitude(profile: np.ndarray) -> float:
    """Measure fringe amplitude from a line profile crossing an edge.

    Computes the difference between the max and min of the gradient,
    which captures the bright-dark fringe pair at an interface.
    """
    grad = np.gradient(profile)
    return float(np.max(grad) - np.min(grad))


def edge_sharpness(profile: np.ndarray) -> float:
    """Maximum absolute gradient along a profile."""
    return float(np.max(np.abs(np.gradient(profile))))


def compute_all_metrics(raw: np.ndarray, filtered: np.ndarray,
                        mask: np.ndarray = None) -> dict:
    """Compute all contrast metrics for raw and filtered images.

    Returns dict with keys like 'raw_edge_strength', 'filtered_bimodal_snr', etc.
    """
    if mask is None:
        mask = raw > 0

    results = {}
    for prefix, img in [('raw', raw), ('filtered', filtered)]:
        results[f'{prefix}_edge_strength'] = edge_strength(img, mask)
        results[f'{prefix}_bimodal_snr'] = bimodal_snr(img, mask)
        results[f'{prefix}_local_contrast'] = local_contrast(img, mask)

    # Derived
    if results['raw_edge_strength'] > 0:
        results['edge_reduction_pct'] = 100 * (
            1 - results['filtered_edge_strength'] / results['raw_edge_strength']
        )
    else:
        results['edge_reduction_pct'] = 0.0

    return results
