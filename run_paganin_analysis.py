"""Step 2: Paganin phase retrieval analysis for ESRF fragments.

For each volume:
1. Load representative ROI at full resolution
2. Sweep delta/beta ratio, compute contrast metrics at each value
3. Find optimal delta/beta
4. Compare raw vs filtered
5. Compare across propagation distances (PHerc0500P2 has 0.4m and 1.2m)

Produces figures and summary tables in results/.
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.io_zarr import ESRFVolume
from src.paganin import (energy_to_wavelength, compute_alpha,
                         paganin_filter_from_params, sweep_delta_beta)
from src.contrast_metrics import (edge_strength, bimodal_snr, local_contrast,
                                  line_profile, fringe_amplitude,
                                  compute_all_metrics)

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

DELTA_BETA_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]


def find_good_roi(vol, roi_size=1024, scale_for_search=3):
    """Find a ROI that is fully inside the fragment (no mask pixels).

    Searches at a downsampled scale across multiple z-slices,
    then maps back to full resolution.
    """
    search_shape = vol.scale_shape(scale_for_search)
    factor = vol.shape[1] / search_shape[1]  # scale factor
    roi_ds = max(int(roi_size / factor), 4)  # ROI size in downsampled coords

    best_y, best_x, best_z, best_fill = 0, 0, 0, 0

    # Search multiple z-slices (every 10% of the volume)
    z_candidates = [int(search_shape[0] * f) for f in [0.3, 0.4, 0.5, 0.6, 0.7]]

    for z_search in z_candidates:
        if z_search >= search_shape[0]:
            continue
        ds_slice = vol.get_slice(z_search, axis=0, scale=scale_for_search)
        mask = ds_slice > 0

        # Sliding window search with small step
        step = max(roi_ds // 8, 1)
        for y in range(0, mask.shape[0] - roi_ds, step):
            for x in range(0, mask.shape[1] - roi_ds, step):
                fill = mask[y:y + roi_ds, x:x + roi_ds].mean()
                if fill > best_fill:
                    best_fill = fill
                    best_y, best_x, best_z = y, x, z_search
                    if fill == 1.0:
                        break
            if best_fill == 1.0:
                break
        if best_fill == 1.0:
            break

    # Map back to full resolution
    full_y = int(best_y * factor)
    full_x = int(best_x * factor)
    full_z = int(best_z * factor)

    print(f"  Best ROI: z={full_z}, y={full_y}:{full_y + roi_size}, "
          f"x={full_x}:{full_x + roi_size} (fill={best_fill:.2f})")

    return full_z, full_y, full_x


def analyze_volume(fragment, volume_key, roi_size=1024):
    """Run full Paganin analysis on one volume."""
    print(f"\n{'='*70}")
    print(f"  Analyzing {fragment} / {volume_key}")
    print(f"{'='*70}")

    vol = ESRFVolume(fragment, volume_key)

    # Find a good ROI
    z, y, x = find_good_roi(vol, roi_size)

    # Load ROI at full resolution
    print(f"  Loading {roi_size}x{roi_size} ROI...")
    t0 = time.time()
    roi = vol.get_roi(z=z, y=slice(y, y + roi_size),
                      x=slice(x, x + roi_size), scale=0)
    print(f"  Loaded in {time.time()-t0:.1f}s, shape={roi.shape}, "
          f"range=[{roi.min()}, {roi.max()}]")

    # Check if ROI has enough non-zero content
    mask = roi > 0
    fill = mask.mean()
    print(f"  Fill fraction: {fill:.3f}")
    if fill < 0.5:
        print("  WARNING: ROI is mostly masked, results may be unreliable")

    # Sweep delta/beta
    print(f"  Running delta/beta sweep: {DELTA_BETA_VALUES}")
    t0 = time.time()
    sweep_results = sweep_delta_beta(
        roi, vol.pixel_size_um, vol.prop_distance_m, vol.energy_keV,
        DELTA_BETA_VALUES, handle_mask=True
    )
    print(f"  Sweep done in {time.time()-t0:.1f}s")

    # Compute metrics for each delta/beta
    rows = []
    for db, filtered in sweep_results:
        metrics = compute_all_metrics(roi.astype(np.float32), filtered, mask)
        row = {
            'fragment': fragment,
            'volume_key': volume_key,
            'pixel_size_um': vol.pixel_size_um,
            'prop_distance_m': vol.prop_distance_m,
            'energy_keV': vol.energy_keV,
            'delta_beta': db,
            **metrics,
        }
        rows.append(row)
        print(f"    db={db:5d}: edge_str={metrics['filtered_edge_strength']:.1f}, "
              f"snr={metrics['filtered_bimodal_snr']:.3f}, "
              f"edge_red={metrics['edge_reduction_pct']:.1f}%")

    # Find optimal delta/beta using SNR-gain-per-edge-loss metric.
    # Pure bimodal SNR just rewards smoothing. Instead, find the delta/beta
    # that gives the best trade-off: maximize SNR improvement while
    # preserving edge detail. Use the "efficiency" = SNR_gain / edge_loss.
    raw_snr = rows[0]['raw_bimodal_snr']
    raw_edge = rows[0]['raw_edge_strength']
    efficiencies = []
    for r in rows:
        snr_gain = r['filtered_bimodal_snr'] - raw_snr
        edge_loss = raw_edge - r['filtered_edge_strength']
        # Efficiency: SNR improvement per unit of edge loss
        eff = snr_gain / max(edge_loss, 1e-6)
        efficiencies.append(eff)
        r['efficiency'] = eff

    best_idx = max(range(len(rows)), key=lambda i: efficiencies[i])
    best_db = rows[best_idx]['delta_beta']
    print(f"\n  Optimal delta/beta: {best_db}")
    print(f"    Bimodal SNR: {rows[best_idx]['raw_bimodal_snr']:.3f} (raw) -> "
          f"{rows[best_idx]['filtered_bimodal_snr']:.3f} (filtered)")
    print(f"    Edge reduction: {rows[best_idx]['edge_reduction_pct']:.1f}%")

    # Get the best filtered image
    best_filtered = sweep_results[best_idx][1]

    # ---- FIGURES ----

    # Figure 1: Delta/beta sweep grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    # Raw image
    vmin, vmax = np.percentile(roi[mask], [2, 98])
    axes[0].imshow(roi, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Raw\nSNR={rows[0]["raw_bimodal_snr"]:.3f}')
    axes[0].axis('off')

    # Filtered at each delta/beta
    for i, (db, filtered) in enumerate(sweep_results):
        if i >= 7:
            break
        ax = axes[i + 1]
        ax.imshow(filtered, cmap='gray', vmin=vmin, vmax=vmax)
        snr = rows[i]['filtered_bimodal_snr']
        ax.set_title(f'δ/β={db}\nSNR={snr:.3f}')
        ax.axis('off')

    fig.suptitle(f'{fragment} / {volume_key}\n'
                 f'Paganin filter: δ/β sweep', fontsize=14)
    plt.tight_layout()
    fname = f'paganin_sweep_{fragment}_{volume_key.replace(".", "p")}.png'
    plt.savefig(f'results/figures/{fname}', dpi=150)
    plt.close()
    print(f"  Saved: results/figures/{fname}")

    # Figure 2: SNR vs delta/beta curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    dbs = [r['delta_beta'] for r in rows]
    snrs = [r['filtered_bimodal_snr'] for r in rows]
    edges = [r['filtered_edge_strength'] for r in rows]

    ax1.semilogx(dbs, snrs, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(rows[0]['raw_bimodal_snr'], color='r', linestyle='--',
                label=f'Raw SNR={rows[0]["raw_bimodal_snr"]:.3f}')
    ax1.axvline(best_db, color='g', linestyle=':', alpha=0.5,
                label=f'Optimal δ/β={best_db}')
    ax1.set_xlabel('δ/β ratio')
    ax1.set_ylabel('Bimodal SNR')
    ax1.set_title('Material Separability vs Filter Strength')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(dbs, edges, 'ro-', linewidth=2, markersize=8)
    ax2.axhline(rows[0]['raw_edge_strength'], color='r', linestyle='--',
                label=f'Raw edge={rows[0]["raw_edge_strength"]:.1f}')
    ax2.set_xlabel('δ/β ratio')
    ax2.set_ylabel('Edge Strength (95th percentile gradient)')
    ax2.set_title('Edge Enhancement vs Filter Strength')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{fragment} / {volume_key}', fontsize=13)
    plt.tight_layout()
    fname2 = f'paganin_curves_{fragment}_{volume_key.replace(".", "p")}.png'
    plt.savefig(f'results/figures/{fname2}', dpi=150)
    plt.close()
    print(f"  Saved: results/figures/{fname2}")

    # Figure 3: Before/after comparison with zoom
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Full ROI
    axes[0, 0].imshow(roi, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Raw')
    axes[0, 1].imshow(best_filtered, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Paganin filtered (δ/β={best_db})')

    # Zoomed 256x256 center crop
    cy, cx = roi_size // 2, roi_size // 2
    zs = 128  # zoom half-size
    axes[1, 0].imshow(roi[cy-zs:cy+zs, cx-zs:cx+zs],
                      cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Raw (center zoom)')
    axes[1, 1].imshow(best_filtered[cy-zs:cy+zs, cx-zs:cx+zs],
                      cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'Filtered (center zoom)')

    for ax in axes.ravel():
        ax.axis('off')

    fig.suptitle(f'{fragment} / {volume_key}\n'
                 f'Paganin before/after (δ/β={best_db}, '
                 f'SNR: {rows[0]["raw_bimodal_snr"]:.3f}→'
                 f'{rows[best_idx]["filtered_bimodal_snr"]:.3f})',
                 fontsize=13)
    plt.tight_layout()
    fname3 = f'paganin_comparison_{fragment}_{volume_key.replace(".", "p")}.png'
    plt.savefig(f'results/figures/{fname3}', dpi=150)
    plt.close()
    print(f"  Saved: results/figures/{fname3}")

    # Figure 4: Line profile showing fringe removal
    # Find a horizontal line through the center of the ROI
    mid_x = roi_size // 2
    prof_raw = line_profile(roi.astype(np.float32), 0, roi_size, mid_x)
    prof_filt = line_profile(best_filtered, 0, roi_size, mid_x)

    fig, ax = plt.subplots(figsize=(14, 5))
    y_coords = np.arange(len(prof_raw))
    ax.plot(y_coords, prof_raw, 'b-', alpha=0.7, linewidth=0.8, label='Raw')
    ax.plot(y_coords, prof_filt, 'r-', alpha=0.9, linewidth=1.2,
            label=f'Paganin (δ/β={best_db})')
    ax.set_xlabel('y position (pixels)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'{fragment} / {volume_key}\n'
                 f'Vertical line profile at x={mid_x} — fringe suppression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname4 = f'paganin_profile_{fragment}_{volume_key.replace(".", "p")}.png'
    plt.savefig(f'results/figures/{fname4}', dpi=150)
    plt.close()
    print(f"  Saved: results/figures/{fname4}")

    return rows, roi, best_filtered, best_db


def main():
    all_rows = []

    # Start with the smallest volume for fast iteration
    volumes_to_analyze = [
        ('PHerc0500P2', '9.362um-1.2m-113keV', 1024),
        ('PHerc0500P2', '4.317um-1.2m-111keV', 1024),
        ('PHerc0500P2', '2.215um-0.4m-111keV', 1024),
        ('PHerc0343P', '8.640um-1.2m-116keV', 1024),
        ('PHerc0343P', '2.215um-0.4m-111keV', 1024),
        ('PHerc0009B', '8.640um-1.2m-116keV', 1024),
        ('PHerc0009B', '2.401um-0.3m-77keV', 1024),
    ]

    for fragment, volume_key, roi_size in volumes_to_analyze:
        try:
            rows, _, _, _ = analyze_volume(fragment, volume_key, roi_size)
            all_rows.extend(rows)
        except Exception as e:
            print(f"\n  ERROR analyzing {fragment}/{volume_key}: {e}")
            import traceback
            traceback.print_exc()

    # Save full results
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv('results/tables/paganin_sweep_all.csv', index=False)
        print(f"\nSaved: results/tables/paganin_sweep_all.csv ({len(df)} rows)")

        # Summary table: one row per volume at optimal delta/beta
        summary_rows = []
        for (frag, vk), group in df.groupby(['fragment', 'volume_key']):
            best = group.loc[group['filtered_bimodal_snr'].idxmax()]
            summary_rows.append(best.to_dict())
        summary = pd.DataFrame(summary_rows)
        summary.to_csv('results/tables/paganin_summary.csv', index=False)
        print(f"Saved: results/tables/paganin_summary.csv ({len(summary)} rows)")

        # Print summary
        print(f"\n{'='*80}")
        print("PAGANIN ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Fragment':<14} {'Volume':<25} {'δ/β':>6} "
              f"{'Raw SNR':>8} {'Filt SNR':>9} {'Edge Red':>9}")
        print(f"{'─'*14} {'─'*25} {'─'*6} {'─'*8} {'─'*9} {'─'*9}")
        for _, r in summary.iterrows():
            print(f"{r['fragment']:<14} {r['volume_key']:<25} "
                  f"{r['delta_beta']:6.0f} "
                  f"{r['raw_bimodal_snr']:8.3f} "
                  f"{r['filtered_bimodal_snr']:9.3f} "
                  f"{r['edge_reduction_pct']:8.1f}%")


if __name__ == '__main__':
    main()
