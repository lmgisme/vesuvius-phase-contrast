"""Ground truth validation: apply Paganin filter to DLS fragments and compute AUC.

The ESRF fragments have NO ink labels. To validate that the Paganin filter's
proxy SNR metric correlates with real ink detection, we apply it to the DLS
fragments (frag1-3) which have inklabels.png ground truth.

DLS scan parameters (Diamond Light Source I12 beamline):
- Energy: 54 keV
- Pixel size: ~8 µm (DLS I12 typical for Vesuvius fragments)
- Propagation distance: estimated ~0.1-0.5 m (typical for I12 setup)
  We don't know the exact propagation distance, so we treat delta/beta
  as the sole tuning parameter (alpha = lambda * z * delta_beta / 4pi,
  and we sweep alpha effectively by sweeping delta_beta at fixed z).

For each fragment, we use the peak-SNR layer identified in Phase 1:
- Frag1: layer 59 (SNR=1.37)
- Frag2: layer 64 (SNR=1.02)
- Frag3: layer 49 (SNR=1.16)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from PIL import Image
import tifffile
import gc

from src.paganin import paganin_filter_from_params

# DLS scan parameters
DLS_ENERGY_KEV = 54.0
DLS_PIXEL_SIZE_UM = 8.0  # approximate
DLS_PROP_DISTANCE_M = 0.2  # estimated; the exact value is absorbed into delta/beta

# Peak layers from Phase 1
PEAK_LAYERS = {'frag1': 59, 'frag2': 64, 'frag3': 49}
DELTA_BETA_VALUES = [50, 200, 1000, 5000]

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fragments')

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


def load_dls_layer(frag_name, layer_idx):
    """Load a single layer from DLS surface volume."""
    frag_dir = os.path.join(DATA_DIR, frag_name, '54keV_exposed_surface',
                            'surface_volume')
    tif_path = os.path.join(frag_dir, f'{layer_idx:02d}.tif')
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Layer file not found: {tif_path}")
    img = tifffile.imread(tif_path).astype(np.float32)
    print(f"  Loaded layer {layer_idx}: shape={img.shape}, "
          f"range=[{img.min():.0f}, {img.max():.0f}]")
    return img


def load_ink_labels(frag_name):
    """Load binary ink labels."""
    path = os.path.join(DATA_DIR, frag_name, '54keV_exposed_surface',
                        'inklabels.png')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ink labels not found: {path}")
    labels = np.array(Image.open(path))
    # Convert to binary (may be RGB, grayscale, or palette mode with 0/1)
    if labels.ndim == 3:
        labels = labels[:, :, 0]
    ink_mask = labels > 0
    print(f"  Ink labels: shape={ink_mask.shape}, "
          f"ink_fraction={ink_mask.mean():.4f}")
    return ink_mask


def compute_auc(image, ink_mask, valid_mask=None):
    """Compute pixel-level AUC for ink detection.

    Uses the image intensity as the prediction score.

    Args:
        image: 2D array of intensity values.
        ink_mask: 2D boolean array of ink labels.
        valid_mask: Optional 2D boolean mask of pixels to evaluate on.
            If None, all pixels are used. IMPORTANT: use a fixed valid_mask
            (e.g. from the raw image) for both raw and filtered evaluations
            to ensure apples-to-apples comparison.

    Returns:
        Tuple of (auc_all, auc_valid) where auc_all uses all pixels
        and auc_valid uses only valid_mask pixels. If valid_mask is None,
        both values are the same.
    """
    scores = image.ravel()
    labels = ink_mask.ravel()

    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5, 0.5

    auc_all = roc_auc_score(labels, scores)

    auc_valid = auc_all
    if valid_mask is not None:
        vm = valid_mask.ravel()
        if vm.sum() > 0 and labels[vm].sum() > 0 and labels[vm].sum() < vm.sum():
            auc_valid = roc_auc_score(labels[vm], scores[vm])

    return auc_all, auc_valid


def analyze_fragment(frag_name):
    """Run Paganin analysis on one DLS fragment."""
    print(f"\n{'='*70}")
    print(f"  {frag_name}")
    print(f"{'='*70}")

    layer_idx = PEAK_LAYERS[frag_name]
    raw = load_dls_layer(frag_name, layer_idx)
    ink_mask = load_ink_labels(frag_name)

    # Ensure shapes match (ink labels may be different size)
    if raw.shape != ink_mask.shape:
        print(f"  Shape mismatch: raw={raw.shape}, labels={ink_mask.shape}")
        # Crop to minimum
        h = min(raw.shape[0], ink_mask.shape[0])
        w = min(raw.shape[1], ink_mask.shape[1])
        raw = raw[:h, :w]
        ink_mask = ink_mask[:h, :w]
        print(f"  Cropped to {h}x{w}")

    # Define valid mask from raw image (consistent for all evaluations)
    valid_mask = raw > 0

    # Raw AUC
    raw_auc_all, raw_auc_valid = compute_auc(raw, ink_mask, valid_mask)
    print(f"\n  Raw layer {layer_idx} AUC: all={raw_auc_all:.4f}, valid={raw_auc_valid:.4f}")

    rows = []
    filtered_images = {}

    for db in DELTA_BETA_VALUES:
        filtered = paganin_filter_from_params(
            raw, DLS_PIXEL_SIZE_UM, DLS_PROP_DISTANCE_M, DLS_ENERGY_KEV,
            db, handle_mask=False  # DLS data is not masked
        )
        filt_auc_all, filt_auc_valid = compute_auc(filtered, ink_mask, valid_mask)
        print(f"  δ/β={db:5d}: AUC all={filt_auc_all:.4f}(Δ={filt_auc_all - raw_auc_all:+.4f})  "
              f"valid={filt_auc_valid:.4f}(Δ={filt_auc_valid - raw_auc_valid:+.4f})")

        rows.append({
            'fragment': frag_name,
            'layer': layer_idx,
            'delta_beta': db,
            'raw_auc_all': raw_auc_all,
            'filtered_auc_all': filt_auc_all,
            'auc_change_all': filt_auc_all - raw_auc_all,
            'raw_auc_valid': raw_auc_valid,
            'filtered_auc_valid': filt_auc_valid,
            'auc_change_valid': filt_auc_valid - raw_auc_valid,
        })
        filtered_images[db] = filtered

    # Find best delta/beta
    best_row = max(rows, key=lambda r: r['filtered_auc_all'])
    best_db = best_row['delta_beta']
    print(f"\n  Best δ/β={best_db}: AUC all {raw_auc_all:.4f} → "
          f"{best_row['filtered_auc_all']:.4f}, "
          f"valid {raw_auc_valid:.4f} → {best_row['filtered_auc_valid']:.4f}")

    # ---- Figures ----

    best_filtered = filtered_images[best_db]

    # Figure 1: Before/after with ink overlay
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmin, vmax = np.percentile(raw[raw > 0], [2, 98])

    axes[0, 0].imshow(raw, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Raw (layer {layer_idx})\nAUC={raw_auc_all:.4f}')

    axes[0, 1].imshow(best_filtered, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Paganin δ/β={best_db}\n'
                         f'AUC={best_row["filtered_auc_all"]:.4f}')

    # Ink overlay
    overlay = np.stack([raw / max(vmax, 1)] * 3, axis=-1)
    overlay = np.clip(overlay, 0, 1)
    overlay[ink_mask, 0] = 1.0  # Red overlay on ink
    overlay[ink_mask, 1] *= 0.5
    overlay[ink_mask, 2] *= 0.5
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Ink labels (red overlay)')

    # Zoomed views (center 512x512)
    cy, cx = raw.shape[0] // 2, raw.shape[1] // 2
    zs = 256
    axes[1, 0].imshow(raw[cy-zs:cy+zs, cx-zs:cx+zs],
                      cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Raw (zoom)')
    axes[1, 1].imshow(best_filtered[cy-zs:cy+zs, cx-zs:cx+zs],
                      cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'Filtered (zoom)')
    axes[1, 2].imshow(overlay[cy-zs:cy+zs, cx-zs:cx+zs])
    axes[1, 2].set_title('Ink overlay (zoom)')

    for ax in axes.ravel():
        ax.axis('off')

    fig.suptitle(f'{frag_name}: Paganin Ground Truth Validation\n'
                 f'Layer {layer_idx}, Energy=54keV', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'results/figures/validate_{frag_name}.png', dpi=150)
    plt.close()
    print(f"  Saved: results/figures/validate_{frag_name}.png")

    # Figure 2: ROC curves
    fig, ax = plt.subplots(figsize=(8, 8))

    # Raw ROC (all pixels)
    fpr, tpr, _ = roc_curve(ink_mask.ravel(), raw.ravel())
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Raw (AUC={raw_auc_all:.4f})')

    # Filtered ROC at each delta/beta
    colors = ['green', 'orange', 'red', 'purple']
    for (db, fimg), color in zip(filtered_images.items(), colors):
        fpr, tpr, _ = roc_curve(ink_mask.ravel(), fimg.ravel())
        fauc_all, _ = compute_auc(fimg, ink_mask, valid_mask)
        ax.plot(fpr, tpr, color=color, linewidth=1.5,
                label=f'δ/β={db} (AUC={fauc_all:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{frag_name}: ROC Curves — Raw vs Paganin Filtered')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/figures/validate_roc_{frag_name}.png', dpi=150)
    plt.close()
    print(f"  Saved: results/figures/validate_roc_{frag_name}.png")

    # Cleanup
    del raw, filtered_images, best_filtered, ink_mask
    gc.collect()

    return rows


def main():
    all_rows = []

    for frag in ['frag1', 'frag2', 'frag3']:
        try:
            rows = analyze_fragment(frag)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv('results/tables/ground_truth_auc.csv', index=False)
        print(f"\nSaved: results/tables/ground_truth_auc.csv")

        # Summary
        print(f"\n{'='*80}")
        print("GROUND TRUTH VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"{'Fragment':<10} {'Layer':>5} {'δ/β':>6} "
              f"{'Raw(all)':>9} {'Filt(all)':>10} {'Δ(all)':>8} "
              f"{'Raw(val)':>9} {'Filt(val)':>10} {'Δ(val)':>8}")
        print(f"{'─'*10} {'─'*5} {'─'*6} {'─'*9} {'─'*10} {'─'*8} "
              f"{'─'*9} {'─'*10} {'─'*8}")

        for frag in ['frag1', 'frag2', 'frag3']:
            frag_rows = [r for r in all_rows if r['fragment'] == frag]
            best = max(frag_rows, key=lambda r: r['filtered_auc_all'])
            print(f"{frag:<10} {best['layer']:5d} {best['delta_beta']:6d} "
                  f"{best['raw_auc_all']:9.4f} {best['filtered_auc_all']:10.4f} "
                  f"{best['auc_change_all']:+8.4f} "
                  f"{best['raw_auc_valid']:9.4f} {best['filtered_auc_valid']:10.4f} "
                  f"{best['auc_change_valid']:+8.4f}")


if __name__ == '__main__':
    main()
