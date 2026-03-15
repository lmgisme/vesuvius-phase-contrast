"""Verify what the 2-component GMM is actually fitting in ESRF ROIs.

The bimodal_snr metric (used to claim 30% SNR improvement from Paganin
filtering) fits a 2-component Gaussian mixture to pixel intensities in
a 1024x1024 ROI. We need to verify that the two modes correspond to
a physically meaningful distinction (ink vs papyrus) rather than e.g.:
  - Fragment interior vs air/void at the edge
  - Dense vs porous papyrus layers
  - Scan artifacts

For each volume, we:
1. Load a fully-interior ROI (no edge pixels)
2. Fit the GMM and show component assignments spatially
3. Plot the intensity histogram with GMM overlay
4. Check S3 for IR/label data that could validate the modes
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
import s3fs

from src.io_zarr import ESRFVolume, VOLUME_CATALOG
from src.paganin import paganin_filter_from_params

os.makedirs('results/figures', exist_ok=True)

BUCKET = 'vesuvius-challenge-open-data'


def find_interior_roi(vol, roi_size=1024, scale_for_search=3, min_fill=0.99):
    """Find a ROI that is fully inside the fragment (no mask/edge pixels).

    Searches at a downsampled scale, picks the ROI with highest fill fraction.
    Returns (z_full, y_full, x_full) in full-resolution coordinates.
    """
    search_shape = vol.scale_shape(scale_for_search)
    factor = vol.shape[1] / search_shape[1]
    roi_ds = max(int(roi_size / factor), 4)

    best_y, best_x, best_z, best_fill = 0, 0, 0, 0

    # Search multiple z-slices
    z_candidates = [int(search_shape[0] * f) for f in [0.3, 0.4, 0.5, 0.6, 0.7]]

    for z_search in z_candidates:
        if z_search >= search_shape[0]:
            continue
        ds_slice = vol.get_slice(z_search, axis=0, scale=scale_for_search)
        mask = ds_slice > 0

        # Erode the mask to avoid edge regions
        from scipy import ndimage
        eroded = ndimage.binary_erosion(mask, iterations=roi_ds // 2)

        step = max(roi_ds // 4, 1)
        for y in range(0, eroded.shape[0] - roi_ds, step):
            for x in range(0, eroded.shape[1] - roi_ds, step):
                fill = eroded[y:y + roi_ds, x:x + roi_ds].mean()
                if fill > best_fill:
                    best_fill = fill
                    best_y, best_x, best_z = y, x, z_search
                    if fill >= min_fill:
                        break
            if best_fill >= min_fill:
                break
        if best_fill >= min_fill:
            break

    full_y = int(best_y * factor)
    full_x = int(best_x * factor)
    full_z = int(best_z * factor)

    print(f"  Best ROI: z={full_z}, y={full_y}:{full_y + roi_size}, "
          f"x={full_x}:{full_x + roi_size} (fill={best_fill:.3f})")
    return full_z, full_y, full_x


def fit_and_visualize_gmm(roi, mask, title, save_prefix):
    """Fit 2-component GMM and produce diagnostic plots."""
    values = roi[mask].ravel().astype(np.float64)

    # Fit GMM
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(values.reshape(-1, 1))

    mu = gmm.means_.ravel()
    sigma = np.sqrt(gmm.covariances_.ravel())
    weights = gmm.weights_.ravel()

    # Sort by mean
    order = np.argsort(mu)
    mu = mu[order]
    sigma = sigma[order]
    weights = weights[order]

    snr = abs(mu[1] - mu[0]) / np.sqrt((sigma[0]**2 + sigma[1]**2) / 2)

    print(f"  GMM mode 0: µ={mu[0]:.1f}, σ={sigma[0]:.1f}, weight={weights[0]:.3f}")
    print(f"  GMM mode 1: µ={mu[1]:.1f}, σ={sigma[1]:.1f}, weight={weights[1]:.3f}")
    print(f"  Bimodal SNR: {snr:.3f}")
    print(f"  Mode separation: {abs(mu[1] - mu[0]):.1f} intensity units")
    print(f"  Overlap: σ0+σ1={sigma[0]+sigma[1]:.1f} vs gap={abs(mu[1]-mu[0]):.1f} "
          f"({'well-separated' if abs(mu[1]-mu[0]) > sigma[0]+sigma[1] else 'OVERLAPPING'})")

    # Predict component assignments for all pixels
    all_values = roi.ravel().astype(np.float64)
    labels_flat = gmm.predict(all_values.reshape(-1, 1))
    # Reorder labels to match sorted means
    if order[0] != 0:
        labels_flat = 1 - labels_flat
    label_map = labels_flat.reshape(roi.shape)

    # ---- Figure: 4-panel diagnostic ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Panel A: Raw ROI
    vmin, vmax = np.percentile(values, [1, 99])
    ax = axes[0, 0]
    ax.imshow(roi, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title('A. Raw ROI')
    ax.axis('off')

    # Panel B: GMM component assignments
    ax = axes[0, 1]
    cmap = plt.cm.RdYlBu_r
    ax.imshow(label_map, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f'B. GMM assignments\n'
                 f'Blue=mode 0 (µ={mu[0]:.0f}), Red=mode 1 (µ={mu[1]:.0f})')
    ax.axis('off')

    # Panel C: Intensity histogram with GMM overlay
    ax = axes[1, 0]
    bins = np.linspace(vmin, vmax, 200)
    ax.hist(values, bins=bins, density=True, alpha=0.5, color='gray', label='Data')

    x = np.linspace(vmin, vmax, 1000)
    for i in range(2):
        pdf = weights[i] * stats.norm.pdf(x, mu[i], sigma[i])
        color = 'blue' if i == 0 else 'red'
        ax.plot(x, pdf, color=color, linewidth=2,
                label=f'Mode {i}: µ={mu[i]:.0f}, σ={sigma[i]:.0f}, w={weights[i]:.2f}')

    # Combined pdf
    combined = sum(weights[i] * stats.norm.pdf(x, mu[i], sigma[i]) for i in range(2))
    ax.plot(x, combined, 'k--', linewidth=1.5, alpha=0.7, label='GMM combined')

    ax.set_xlabel('Pixel intensity')
    ax.set_ylabel('Density')
    ax.set_title(f'C. Intensity histogram\nBimodal SNR = {snr:.3f}')
    ax.legend(fontsize=8)

    # Panel D: Side-by-side intensity in mode-0 vs mode-1 regions
    ax = axes[1, 1]
    mode0_vals = roi[label_map == 0].ravel()
    mode1_vals = roi[label_map == 1].ravel()
    mode0_frac = (label_map == 0).sum() / label_map.size
    mode1_frac = (label_map == 1).sum() / label_map.size

    ax.hist(mode0_vals, bins=100, density=True, alpha=0.5, color='blue',
            label=f'Mode 0 ({mode0_frac*100:.0f}% of pixels)')
    ax.hist(mode1_vals, bins=100, density=True, alpha=0.5, color='red',
            label=f'Mode 1 ({mode1_frac*100:.0f}% of pixels)')
    ax.set_xlabel('Pixel intensity')
    ax.set_ylabel('Density')
    ax.set_title('D. Intensity distributions per mode')
    ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'results/figures/{save_prefix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: results/figures/{save_prefix}.png")

    return {
        'mu': mu, 'sigma': sigma, 'weights': weights, 'snr': snr,
        'label_map': label_map, 'mode0_frac': mode0_frac, 'mode1_frac': mode1_frac,
    }


def explore_s3_for_labels(fragment_name):
    """Check S3 bucket for IR images, photos, labels, or any validation data."""
    fs = s3fs.S3FileSystem(anon=True)

    print(f"\n  Exploring S3 for {fragment_name}...")
    base = f"{BUCKET}/{fragment_name}"

    try:
        top_level = fs.ls(base)
        print(f"  Top-level contents of {fragment_name}/:")
        for item in top_level:
            short = item.replace(f"{BUCKET}/", "")
            is_dir = fs.isdir(item)
            print(f"    {'[DIR]' if is_dir else '[FILE]'} {short}")
    except Exception as e:
        # Try with 'fragments/' prefix
        base = f"{BUCKET}/fragments/{fragment_name}"
        try:
            top_level = fs.ls(base)
            print(f"  Top-level contents of fragments/{fragment_name}/:")
            for item in top_level:
                short = item.replace(f"{BUCKET}/", "")
                is_dir = fs.isdir(item)
                print(f"    {'[DIR]' if is_dir else '[FILE]'} {short}")
        except Exception as e2:
            print(f"  Could not list: {e2}")
            return None

    # Look for specific directories of interest
    interesting_dirs = ['photos', 'multispectral', 'ir', 'labels', 'inklabels',
                        'paths', 'surface_volume']
    found = {}
    for d in interesting_dirs:
        check_path = f"{base}/{d}"
        try:
            if fs.exists(check_path):
                contents = fs.ls(check_path)
                found[d] = contents[:20]  # first 20 items
                print(f"\n  Found {d}/ ({len(contents)} items):")
                for item in contents[:10]:
                    short = item.replace(f"{BUCKET}/", "")
                    size = ''
                    try:
                        info = fs.info(item)
                        if info.get('size', 0) > 0:
                            size = f" ({info['size'] / 1e6:.1f} MB)"
                    except:
                        pass
                    print(f"    {short}{size}")
                if len(contents) > 10:
                    print(f"    ... and {len(contents) - 10} more")
        except:
            pass

    # Also check for any .png or .jpg at top level
    try:
        all_files = fs.ls(base, detail=True)
        image_files = [f for f in all_files
                       if any(f['name'].endswith(ext) for ext in ['.png', '.jpg', '.tif'])]
        if image_files:
            print(f"\n  Image files at top level:")
            for f in image_files[:10]:
                print(f"    {f['name'].replace(BUCKET + '/', '')} ({f.get('size', 0) / 1e6:.1f} MB)")
    except:
        pass

    return found


def analyze_volume(fragment, volume_key, roi_size=1024):
    """Full GMM verification for one volume."""
    print(f"\n{'='*70}")
    print(f"  {fragment} / {volume_key}")
    print(f"{'='*70}")

    vol = ESRFVolume(fragment, volume_key)

    # Find interior ROI
    z, y, x = find_interior_roi(vol, roi_size)

    # Load at full resolution
    print(f"  Loading {roi_size}x{roi_size} ROI at scale=0...")
    import time
    t0 = time.time()
    roi = vol.get_roi(z=z, y=slice(y, y + roi_size),
                      x=slice(x, x + roi_size), scale=0)
    print(f"  Loaded in {time.time()-t0:.1f}s, shape={roi.shape}, "
          f"dtype={roi.dtype}, range=[{roi.min()}, {roi.max()}]")

    mask = roi > 0
    fill = mask.mean()
    print(f"  Fill fraction: {fill:.4f}")
    if fill < 0.95:
        print("  WARNING: ROI has significant masked area — may not be fully interior")

    # Fit GMM and visualize
    title = f'{fragment} / {volume_key}\nGMM Mode Verification (interior {roi_size}x{roi_size} ROI)'
    prefix = f'gmm_verification_{fragment}_{volume_key.replace(".", "p")}'
    result = fit_and_visualize_gmm(roi, mask, title, prefix)

    # Also fit GMM on Paganin-filtered version for comparison
    print("\n  Applying Paganin filter (δ/β=2000) for comparison...")
    from src.paganin import paganin_filter_from_params
    filtered = paganin_filter_from_params(
        roi, vol.pixel_size_um, vol.prop_distance_m, vol.energy_keV,
        delta_beta=2000, handle_mask=True
    )
    filt_mask = filtered > 0
    title_filt = f'{fragment} / {volume_key}\nGMM after Paganin (δ/β=2000)'
    prefix_filt = f'gmm_verification_{fragment}_{volume_key.replace(".", "p")}_filtered'
    result_filt = fit_and_visualize_gmm(filtered, filt_mask, title_filt, prefix_filt)

    # Interpretation
    print(f"\n  === INTERPRETATION ===")
    mu_raw = result['mu']
    mu_filt = result_filt['mu']
    print(f"  Raw modes: µ=[{mu_raw[0]:.0f}, {mu_raw[1]:.0f}], "
          f"SNR={result['snr']:.3f}")
    print(f"  Filtered modes: µ=[{mu_filt[0]:.0f}, {mu_filt[1]:.0f}], "
          f"SNR={result_filt['snr']:.3f}")
    print(f"  SNR change: {result['snr']:.3f} → {result_filt['snr']:.3f} "
          f"({(result_filt['snr']/result['snr'] - 1)*100:+.1f}%)")

    # Check if modes look like:
    # A) Two distinct materials (sharp bimodal) → plausible ink/papyrus
    # B) Broad single mode with tail (unimodal + noise) → not meaningful
    # C) Interior structure (pores, layers) → not ink
    if result['weights'].min() < 0.05:
        print("  NOTE: One mode has <5% weight — likely fitting noise/outliers, not bimodal")
    if abs(mu_raw[1] - mu_raw[0]) < 2 * max(result['sigma']):
        print("  NOTE: Modes overlap heavily — separation is not clean")

    return result, result_filt


def main():
    # ---- Analyze both volumes ----
    volumes = [
        ('PHerc0343P', '2.215um-0.4m-111keV'),  # strongest Paganin response
        ('PHerc0009B', '2.401um-0.3m-77keV'),    # second strongest
    ]

    results = {}
    for fragment, volume_key in volumes:
        raw_result, filt_result = analyze_volume(fragment, volume_key)
        results[(fragment, volume_key)] = (raw_result, filt_result)

    # ---- Check S3 for IR/label data ----
    for fragment in ['PHerc0343P', 'PHerc0009B']:
        explore_s3_for_labels(fragment)

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("  GMM VERIFICATION SUMMARY")
    print(f"{'='*70}")
    for (frag, vk), (raw_r, filt_r) in results.items():
        print(f"\n  {frag} / {vk}:")
        print(f"    Raw:  modes at {raw_r['mu'][0]:.0f} ({raw_r['weights'][0]*100:.0f}%) "
              f"and {raw_r['mu'][1]:.0f} ({raw_r['weights'][1]*100:.0f}%), "
              f"SNR={raw_r['snr']:.3f}")
        print(f"    Filt: modes at {filt_r['mu'][0]:.0f} ({filt_r['weights'][0]*100:.0f}%) "
              f"and {filt_r['mu'][1]:.0f} ({filt_r['weights'][1]*100:.0f}%), "
              f"SNR={filt_r['snr']:.3f}")

        # Spatial interpretation
        if raw_r['weights'].min() < 0.1:
            print(f"    CONCERN: Minor mode is only {raw_r['weights'].min()*100:.0f}% of pixels")
            print(f"    The GMM may be fitting outliers/artifacts, not a meaningful bimodal structure")
        if raw_r['mode0_frac'] > 0.85 or raw_r['mode1_frac'] > 0.85:
            dominant = 0 if raw_r['mode0_frac'] > 0.85 else 1
            minor = 1 - dominant
            print(f"    Mode {dominant} dominates ({max(raw_r['mode0_frac'], raw_r['mode1_frac'])*100:.0f}%) — "
                  f"mode {minor} likely represents voids/artifacts/edges, NOT ink")

    print("\n  KEY QUESTION: Do the GMM component assignments show spatial patterns")
    print("  consistent with ink (text-like features) or structural features")
    print("  (layering, porosity, scan artifacts)?")
    print("  → Inspect the saved figures to answer this visually.")


if __name__ == '__main__':
    main()
