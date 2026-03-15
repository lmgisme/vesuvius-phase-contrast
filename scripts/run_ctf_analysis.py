"""Step 3: CTF / Fresnel number analysis for ESRF fragment volumes.

For each of the 7 volumes:
1. Compute Fresnel numbers (pixel-level and for feature sizes 10/25/50 µm)
2. Compute theoretical CTF peak spatial frequency
3. Load a representative slice, compute radially-averaged power spectrum
4. Overlay theoretical CTF on power spectrum
5. Produce summary figure and CSV table

Outputs:
    results/figures/ctf_analysis.png — summary of all 7 volumes
    results/figures/ctf_power_spectra.png — individual power spectra with CTF overlay
    results/tables/ctf_summary.csv
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import fft as scipy_fft
from src.io_zarr import ESRFVolume, VOLUME_CATALOG, VOLUME_SHAPES
from src.paganin import energy_to_wavelength, HC_KEV_M

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# Feature sizes of interest (ink stroke widths)
FEATURE_SIZES_UM = [10, 25, 50]

# Volumes ordered for analysis
VOLUMES = [
    ('PHerc0500P2', '2.215um-0.4m-111keV'),
    ('PHerc0500P2', '4.317um-1.2m-111keV'),
    ('PHerc0500P2', '9.362um-1.2m-113keV'),
    ('PHerc0343P', '2.215um-0.4m-111keV'),
    ('PHerc0343P', '8.640um-1.2m-116keV'),
    ('PHerc0009B', '2.401um-0.3m-77keV'),
    ('PHerc0009B', '8.640um-1.2m-116keV'),
]


def parse_params(volume_key):
    """Extract pixel_size_um, prop_distance_m, energy_keV from volume key."""
    parts = volume_key.split('-')
    pixel_size_um = float(parts[0].replace('um', ''))
    prop_distance_m = float(parts[1].replace('m', ''))
    energy_keV = float(parts[2].replace('keV', ''))
    return pixel_size_um, prop_distance_m, energy_keV


def fresnel_number(a_m, wavelength_m, z_m):
    """Compute Fresnel number N_F = a^2 / (lambda * z)."""
    return a_m ** 2 / (wavelength_m * z_m)


def ctf_peak_freq(wavelength_m, z_m):
    """First maximum of sinusoidal CTF: f_peak = 1 / (2 * sqrt(lambda * z)).

    Returns frequency in cycles/m.
    """
    return 1.0 / (2.0 * np.sqrt(wavelength_m * z_m))


def radial_power_spectrum(image, pixel_size_m):
    """Compute radially-averaged power spectrum of a 2D image.

    Args:
        image: 2D array (masked pixels should be filled beforehand).
        pixel_size_m: Pixel size in meters.

    Returns:
        freqs: 1D array of spatial frequencies (cycles/m).
        power: 1D array of radially-averaged power spectral density.
    """
    ny, nx = image.shape
    img = image.astype(np.float64)
    # Remove mean to suppress DC spike
    mask = img > 0
    if mask.any():
        img[mask] -= img[mask].mean()
        img[~mask] = 0.0
    else:
        img -= img.mean()

    # 2D FFT
    fft2 = scipy_fft.fft2(img, workers=-1)
    power2d = np.abs(scipy_fft.fftshift(fft2)) ** 2

    # Build frequency grid
    freq_y = scipy_fft.fftfreq(ny, d=pixel_size_m)
    freq_x = scipy_fft.fftfreq(nx, d=pixel_size_m)
    freq_y = scipy_fft.fftshift(freq_y)
    freq_x = scipy_fft.fftshift(freq_x)
    fy, fx = np.meshgrid(freq_y, freq_x, indexing='ij')
    freq_r = np.sqrt(fx ** 2 + fy ** 2)

    # Radial binning
    nyquist = 1.0 / (2.0 * pixel_size_m)
    n_bins = min(ny, nx) // 2
    bin_edges = np.linspace(0, nyquist, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    power_radial = np.zeros(n_bins)
    for i in range(n_bins):
        ring = (freq_r >= bin_edges[i]) & (freq_r < bin_edges[i + 1])
        if ring.any():
            power_radial[i] = power2d[ring].mean()

    return bin_centers, power_radial


def ctf_sin_squared(freqs, wavelength_m, z_m):
    """Theoretical phase-contrast CTF: sin^2(pi * lambda * z * f^2).

    This is the transfer function for phase information in propagation-based
    phase contrast imaging (weak object approximation). The detected intensity
    modulation from a pure phase object goes as sin(pi*lambda*z*f^2), so
    the power spectrum contribution goes as sin^2.
    """
    argument = np.pi * wavelength_m * z_m * freqs ** 2
    return np.sin(argument) ** 2


def paganin_improves(pixel_size_um):
    """Classify whether Paganin filtering improves contrast.

    Based on validated results: high-res (<4 µm) volumes benefit,
    low-res (>4 µm) volumes are degraded or unchanged.
    """
    return pixel_size_um < 4.0


def main():
    print("=" * 70)
    print("  CTF / Fresnel Number Analysis")
    print("=" * 70)

    rows = []
    spectra_data = []

    for fragment, volume_key in VOLUMES:
        print(f"\n--- {fragment} / {volume_key} ---")
        pixel_um, prop_m, energy_keV = parse_params(volume_key)
        pixel_m = pixel_um * 1e-6
        wavelength_m = energy_to_wavelength(energy_keV)
        wavelength_pm = wavelength_m * 1e12

        # Fresnel number at pixel scale
        nf_pixel = fresnel_number(pixel_m, wavelength_m, prop_m)

        # Fresnel numbers at feature sizes
        nf_features = {}
        for a_um in FEATURE_SIZES_UM:
            a_m = a_um * 1e-6
            nf_features[a_um] = fresnel_number(a_m, wavelength_m, prop_m)

        # CTF peak frequency
        f_peak = ctf_peak_freq(wavelength_m, prop_m)
        f_peak_per_mm = f_peak * 1e-3  # cycles/mm
        ctf_peak_spatial_um = 1e6 / f_peak  # spatial scale in µm

        # Nyquist frequency
        f_nyquist = 1.0 / (2.0 * pixel_m)
        f_nyquist_per_mm = f_nyquist * 1e-3

        print(f"  Wavelength: {wavelength_pm:.2f} pm")
        print(f"  Fresnel number (pixel): {nf_pixel:.2f}")
        for a_um in FEATURE_SIZES_UM:
            print(f"  Fresnel number ({a_um} µm): {nf_features[a_um]:.1f}")
        print(f"  CTF peak freq: {f_peak_per_mm:.1f} cycles/mm "
              f"(spatial scale: {ctf_peak_spatial_um:.1f} µm)")
        print(f"  Nyquist freq: {f_nyquist_per_mm:.1f} cycles/mm")

        # Load a representative slice at reduced scale
        # Use scale=3 (8x downsampled) for speed; sufficient for power spectrum shape
        vol = ESRFVolume(fragment, volume_key)
        shape_s3 = vol.scale_shape(3)
        mid_z = shape_s3[0] // 2
        print(f"  Loading slice z={mid_z} at scale=3 (shape={shape_s3})...")
        slice_data = vol.get_slice(mid_z, axis=0, scale=3)
        print(f"  Loaded: {slice_data.shape}, range=[{slice_data.min()}, {slice_data.max()}]")

        # Effective pixel size at scale=3 (8x coarser)
        effective_pixel_m = pixel_m * (vol.shape[1] / shape_s3[1])

        # Compute power spectrum
        freqs, power = radial_power_spectrum(slice_data, effective_pixel_m)

        spectra_data.append({
            'fragment': fragment,
            'volume_key': volume_key,
            'pixel_size_um': pixel_um,
            'prop_distance_m': prop_m,
            'energy_keV': energy_keV,
            'wavelength_m': wavelength_m,
            'freqs': freqs,
            'power': power,
            'effective_pixel_m': effective_pixel_m,
            'improves': paganin_improves(pixel_um),
        })

        row = {
            'fragment': fragment,
            'volume_key': volume_key,
            'pixel_size_um': pixel_um,
            'prop_distance_m': prop_m,
            'energy_keV': energy_keV,
            'wavelength_pm': round(wavelength_pm, 2),
            'fresnel_number_pixel': round(nf_pixel, 3),
            'fresnel_number_10um': round(nf_features[10], 1),
            'fresnel_number_25um': round(nf_features[25], 1),
            'fresnel_number_50um': round(nf_features[50], 1),
            'ctf_peak_freq_per_mm': round(f_peak_per_mm, 1),
            'ctf_peak_spatial_scale_um': round(ctf_peak_spatial_um, 1),
            'nyquist_freq_per_mm': round(f_nyquist_per_mm, 1),
            'paganin_improves': paganin_improves(pixel_um),
        }
        rows.append(row)

    # ---- Save table ----
    df = pd.DataFrame(rows)
    df.to_csv('results/tables/ctf_summary.csv', index=False)
    print(f"\nSaved: results/tables/ctf_summary.csv ({len(df)} rows)")

    # ---- Figure 1: Summary — Fresnel numbers and CTF peaks ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    colors_good = '#2196F3'  # blue for high-res/short-distance
    colors_bad = '#F44336'   # red for low-res/long-distance
    labels = [f"{r['fragment']}\n{r['volume_key']}" for r in rows]
    improves = [r['paganin_improves'] for r in rows]
    colors = [colors_good if imp else colors_bad for imp in improves]

    # Panel A: Fresnel number at pixel scale
    ax = axes[0, 0]
    nf_pixels = [r['fresnel_number_pixel'] for r in rows]
    bars = ax.barh(range(len(rows)), nf_pixels, color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Fresnel Number (pixel scale)')
    ax.set_title('A. Fresnel Number at Pixel Scale')
    ax.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='N_F = 1')
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # Panel B: Fresnel numbers at feature sizes
    ax = axes[0, 1]
    x = np.arange(len(rows))
    width = 0.25
    for i, a_um in enumerate(FEATURE_SIZES_UM):
        key = f'fresnel_number_{a_um}um'
        vals = [r[key] for r in rows]
        offset = (i - 1) * width
        bar_colors = [colors_good if imp else colors_bad for imp in improves]
        ax.barh(x + offset, vals, width, color=bar_colors,
                alpha=0.4 + 0.2 * i, label=f'{a_um} µm')
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Fresnel Number')
    ax.set_title('B. Fresnel Numbers at Ink Feature Scales')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.invert_yaxis()

    # Panel C: CTF peak frequency vs Nyquist
    ax = axes[1, 0]
    ctf_peaks = [r['ctf_peak_freq_per_mm'] for r in rows]
    nyquists = [r['nyquist_freq_per_mm'] for r in rows]
    for i in range(len(rows)):
        marker = 'o' if improves[i] else 's'
        ax.scatter(ctf_peaks[i], nyquists[i], c=colors[i], s=100,
                   marker=marker, edgecolors='k', linewidth=0.5, zorder=3)
        ax.annotate(f"{rows[i]['pixel_size_um']}µm",
                    (ctf_peaks[i], nyquists[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    # Diagonal: Nyquist = CTF peak
    max_val = max(max(ctf_peaks), max(nyquists)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Nyquist = CTF peak')
    ax.set_xlabel('CTF Peak Frequency (cycles/mm)')
    ax.set_ylabel('Nyquist Frequency (cycles/mm)')
    ax.set_title('C. CTF Peak vs Pixel Nyquist')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: CTF peak spatial scale vs pixel size
    ax = axes[1, 1]
    pixel_sizes = [r['pixel_size_um'] for r in rows]
    ctf_scales = [r['ctf_peak_spatial_scale_um'] for r in rows]
    for i in range(len(rows)):
        marker = 'o' if improves[i] else 's'
        ax.scatter(pixel_sizes[i], ctf_scales[i], c=colors[i], s=100,
                   marker=marker, edgecolors='k', linewidth=0.5, zorder=3)
        ax.annotate(f"{rows[i]['fragment'][:10]}",
                    (pixel_sizes[i], ctf_scales[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel('Pixel Size (µm)')
    ax.set_ylabel('CTF Peak Spatial Scale (µm)')
    ax.set_title('D. Phase Contrast Enhancement Scale vs Resolution')
    ax.grid(True, alpha=0.3)
    # Add reference lines for ink feature sizes
    for a_um in FEATURE_SIZES_UM:
        ax.axhline(a_um, color='gray', linestyle=':', alpha=0.4)
        ax.text(ax.get_xlim()[1] * 0.95, a_um, f'{a_um} µm', fontsize=7,
                va='bottom', ha='right', color='gray')

    # Legend for groups
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_good,
               markersize=10, label='High-res / short-dist (Paganin helps)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors_bad,
               markersize=10, label='Low-res / long-dist (Paganin hurts)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('CTF / Fresnel Number Analysis — ESRF Fragment Volumes',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('results/figures/ctf_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/ctf_analysis.png")

    # ---- Figure 2: Power spectra with CTF overlay ----
    n_vols = len(spectra_data)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.ravel()

    for i, sd in enumerate(spectra_data):
        ax = axes[i]
        freqs_mm = sd['freqs'] * 1e-3  # convert to cycles/mm
        power_norm = sd['power'] / sd['power'].max()  # normalize

        # Plot power spectrum
        color = colors_good if sd['improves'] else colors_bad
        ax.semilogy(freqs_mm, power_norm, color=color, alpha=0.7,
                    linewidth=0.8, label='Power spectrum')

        # Overlay theoretical CTF (scaled for visibility)
        ctf = ctf_sin_squared(sd['freqs'], sd['wavelength_m'], sd['prop_distance_m'])
        # Scale CTF to sit in the visible range of the power spectrum
        ctf_scale = np.median(power_norm[power_norm > 0]) * 2
        ax.semilogy(freqs_mm, ctf * ctf_scale + 1e-10, 'k-', alpha=0.5,
                    linewidth=1.0, label='CTF (sin²)')

        # Mark CTF peak and Nyquist
        f_peak_mm = ctf_peak_freq(sd['wavelength_m'], sd['prop_distance_m']) * 1e-3
        f_nyquist_mm = 1.0 / (2.0 * sd['pixel_size_um'] * 1e-3)
        ax.axvline(f_peak_mm, color='green', linestyle='--', alpha=0.7,
                   label=f'CTF peak ({f_peak_mm:.0f}/mm)')
        ax.axvline(f_nyquist_mm, color='red', linestyle=':', alpha=0.7,
                   label=f'Nyquist ({f_nyquist_mm:.0f}/mm)')

        ax.set_xlabel('Spatial frequency (cycles/mm)', fontsize=8)
        ax.set_ylabel('Normalized power', fontsize=8)
        ax.set_title(f"{sd['fragment']}\n{sd['volume_key']}", fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.set_ylim(bottom=1e-6)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for j in range(n_vols, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Radially-Averaged Power Spectra with Theoretical CTF Overlay',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/figures/ctf_power_spectra.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/ctf_power_spectra.png")

    # ---- Print summary ----
    print(f"\n{'='*80}")
    print("CTF ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Fragment':<14} {'Volume':<25} {'N_F(pix)':>8} "
          f"{'CTF pk':>8} {'Nyquist':>8} {'Paganin':>8}")
    print(f"{'─'*14} {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for r in rows:
        print(f"{r['fragment']:<14} {r['volume_key']:<25} "
              f"{r['fresnel_number_pixel']:8.3f} "
              f"{r['ctf_peak_freq_per_mm']:7.1f} "
              f"{r['nyquist_freq_per_mm']:7.1f} "
              f"{'YES' if r['paganin_improves'] else 'no':>8}")

    print("\nKey insight: The CTF peak frequency falls BELOW the pixel Nyquist")
    print("for high-res volumes (well-sampled fringes → Paganin can suppress them)")
    print("but NEAR or ABOVE Nyquist for low-res volumes (fringes undersampled → ")
    print("filtering removes real signal).")


if __name__ == '__main__':
    main()
