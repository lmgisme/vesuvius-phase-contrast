"""Paganin-form low-pass filter for reconstructed CT slices.

Applied post-reconstruction, the Paganin filter kernel acts as a
regularized low-pass filter (smoothing), not formal phase retrieval
(which requires application to projection images before reconstruction).
It suppresses high-frequency content including edge-enhancement fringes
from propagation-based phase contrast. The filter in Fourier space is:

    filtered = F^{-1}{ F{I} / (1 + alpha * |k|^2) }

where alpha = lambda * z * (delta/beta) / (4*pi).

The delta/beta ratio is the free parameter controlling filter strength.
Typical values for carbonized papyrus at ~110 keV: 100-2000.
"""
import numpy as np
from scipy import fft as scipy_fft

# Planck constant * speed of light in keV*m
HC_KEV_M = 1.23984198e-9


def energy_to_wavelength(energy_keV: float) -> float:
    """Convert photon energy in keV to wavelength in meters."""
    return HC_KEV_M / energy_keV


def compute_alpha(wavelength_m: float, prop_distance_m: float,
                  delta_beta: float) -> float:
    """Compute the Paganin filter parameter alpha.

    alpha = lambda * z * (delta/beta) / (4*pi)

    Args:
        wavelength_m: X-ray wavelength in meters.
        prop_distance_m: Sample-to-detector propagation distance in meters.
        delta_beta: Ratio of refractive index decrement to absorption index.

    Returns:
        alpha in m^2 (to multiply with |k|^2 in 1/m^2).
    """
    return wavelength_m * prop_distance_m * delta_beta / (4.0 * np.pi)


def _build_k_squared(ny: int, nx: int, pixel_size_m: float) -> np.ndarray:
    """Build |k|^2 grid for 2D FFT."""
    freq_y = scipy_fft.fftfreq(ny, d=pixel_size_m)
    freq_x = scipy_fft.fftfreq(nx, d=pixel_size_m)
    ky, kx = np.meshgrid(freq_y, freq_x, indexing='ij')
    # k in spatial frequency (cycles/m), convert to angular frequency
    k_sq = (2 * np.pi) ** 2 * (kx ** 2 + ky ** 2)
    return k_sq


def paganin_filter_2d(image: np.ndarray, pixel_size_m: float,
                      alpha: float) -> np.ndarray:
    """Apply Paganin low-pass filter to a 2D image.

    Args:
        image: 2D array (any dtype, converted to float64 internally).
        pixel_size_m: Pixel size in meters.
        alpha: Filter parameter from compute_alpha().

    Returns:
        Filtered image as float32.
    """
    img = image.astype(np.float64)
    ny, nx = img.shape

    k_sq = _build_k_squared(ny, nx, pixel_size_m)
    denominator = 1.0 + alpha * k_sq

    img_fft = scipy_fft.fft2(img, workers=-1)
    filtered_fft = img_fft / denominator
    filtered = scipy_fft.ifft2(filtered_fft, workers=-1).real

    return filtered.astype(np.float32)


def paganin_filter_masked(image: np.ndarray, pixel_size_m: float,
                          alpha: float) -> np.ndarray:
    """Apply Paganin filter with mask handling for masked zarr volumes.

    Pixels with value 0 are treated as outside the fragment. They are
    filled with the local mean before filtering to reduce ringing,
    then re-zeroed after.

    Args:
        image: 2D uint8 array from masked zarr volume.
        pixel_size_m: Pixel size in meters.
        alpha: Filter parameter from compute_alpha().

    Returns:
        Filtered image as float32, with mask re-applied.
    """
    mask = image > 0
    if not mask.any():
        return image.astype(np.float32)

    img = image.astype(np.float64)
    # Fill masked region with mean of unmasked region
    fill_val = img[mask].mean()
    img[~mask] = fill_val

    filtered = paganin_filter_2d(img, pixel_size_m, alpha)
    filtered[~mask] = 0.0
    return filtered


def paganin_filter_from_params(image: np.ndarray, pixel_size_um: float,
                               prop_distance_m: float, energy_keV: float,
                               delta_beta: float,
                               handle_mask: bool = True) -> np.ndarray:
    """Convenience: apply Paganin filter using physical scan parameters.

    Args:
        image: 2D array.
        pixel_size_um: Pixel size in micrometers.
        prop_distance_m: Propagation distance in meters.
        energy_keV: X-ray energy in keV.
        delta_beta: Phase-to-absorption ratio (free parameter).
        handle_mask: If True, treat zero pixels as mask.

    Returns:
        Filtered image as float32.
    """
    wavelength = energy_to_wavelength(energy_keV)
    alpha = compute_alpha(wavelength, prop_distance_m, delta_beta)
    pixel_m = pixel_size_um * 1e-6

    if handle_mask:
        return paganin_filter_masked(image, pixel_m, alpha)
    else:
        return paganin_filter_2d(image, pixel_m, alpha)


def sweep_delta_beta(image: np.ndarray, pixel_size_um: float,
                     prop_distance_m: float, energy_keV: float,
                     delta_beta_values: list,
                     handle_mask: bool = True) -> list:
    """Apply Paganin filter at multiple delta/beta values.

    Returns list of (delta_beta, filtered_image) tuples.
    Precomputes the FFT and k^2 grid for efficiency.
    """
    wavelength = energy_to_wavelength(energy_keV)
    pixel_m = pixel_size_um * 1e-6
    img = image.astype(np.float64)

    mask = None
    if handle_mask:
        mask = image > 0
        if mask.any():
            fill_val = img[mask].mean()
            img[~mask] = fill_val

    ny, nx = img.shape
    k_sq = _build_k_squared(ny, nx, pixel_m)
    img_fft = scipy_fft.fft2(img, workers=-1)

    results = []
    for db in delta_beta_values:
        alpha = compute_alpha(wavelength, prop_distance_m, db)
        denominator = 1.0 + alpha * k_sq
        filtered = scipy_fft.ifft2(img_fft / denominator, workers=-1).real
        filtered = filtered.astype(np.float32)
        if mask is not None:
            filtered[~mask] = 0.0
        results.append((db, filtered))

    return results
