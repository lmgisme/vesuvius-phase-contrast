# Paganin-Form Smoothing for Vesuvius ESRF Fragment CT Data

A physics analysis tool for applying Paganin-form smoothing (low-pass
filtering) to already-reconstructed Vesuvius Challenge ESRF fragment CT
data (PHerc0500P2, PHerc0343P, PHerc0009B). The Paganin filter kernel
applied post-reconstruction acts as a regularized low-pass filter, not
phase retrieval in the formal sense (which requires application to
projection images before reconstruction). Provides a Fresnel-number-based
framework for predicting which volumes have significant phase-contrast
content baked into their reconstruction, optimal d/b parameters per
volume, and an OME-Zarr streaming loader for the S3-hosted data.

No full downloads required — all volumes stream from S3.

## Background

The ESRF BM18 beamline scans in propagation-based phase contrast mode. 
Phase gradients in the sample produce Fresnel edge-enhancement fringes 
at intensity boundaries. For carbonized papyrus and carbon-based ink, 
absorption contrast is weak — both materials are primarily carbon. Phase 
contrast may carry more discriminative signal than absorption alone, 
but whether fringes help or hurt ink detection depends on whether they 
are resolvable at the scan's pixel size.

Nobody in the community has published a systematic analysis of the phase 
contrast contribution in these scans.

## Data

Seven OME-Zarr volumes across three ESRF fragments (scanned at BM18, 
May 2025), all uint8 with 128³ chunks and 6 pyramid scales:

| Fragment | Pixel Size | Prop. Distance | Energy | Shape (z,y,x) |
|----------|:-:|:-:|:-:|---|
| PHerc0500P2 | 2.215 µm | 0.4 m | 111 keV | 28096×18209×18209 |
| PHerc0500P2 | 4.317 µm | 1.2 m | 111 keV | 15838×9423×9423 |
| PHerc0500P2 | 9.362 µm | 1.2 m | 113 keV | 7057×4196×4196 |
| PHerc0343P | 2.215 µm | 0.4 m | 111 keV | 20714×13155×13155 |
| PHerc0343P | 8.640 µm | 1.2 m | 116 keV | 5398×5057×5057 |
| PHerc0009B | 2.401 µm | 0.3 m | 77 keV | 29112×28259×28259 |
| PHerc0009B | 8.640 µm | 1.2 m | 116 keV | 9598×7837×7837 |

No ink labels exist for ESRF fragments. Ground truth validation was 
performed on DLS fragments (Frag1–3 with inklabels.png).

## The Fresnel number decision rule

The Fresnel number at pixel scale determines whether phase contrast 
fringes are resolvable:
```
N_F = pixel_size² / (lambda × propagation_distance)
```

- **N_F < 1.2**: Fringes span several pixels, well-resolved. Paganin 
  filtering can selectively suppress them. Use δ/β = 2000–5000.
- **N_F 1.2–3.0**: Marginal. Test with δ/β = 50–200, skip if efficiency 
  metric is negative.
- **N_F > 3.0**: Fringes undersampled or aliased. Filtering cannot 
  distinguish them from real structure. Do not filter.
```python
from src.paganin import energy_to_wavelength

pixel_m = pixel_size_um * 1e-6
wavelength_m = energy_to_wavelength(energy_keV)
N_F = pixel_m**2 / (wavelength_m * prop_distance_m)
```

## Results

### Two behavioral groups

**Group 1 — high-res / short-distance (Paganin helps):**

| Fragment | Pixel (µm) | Prop (m) | N_F | SNR: Raw → Filtered | Optimal δ/β |
|----------|:-:|:-:|:-:|:-:|:-:|
| PHerc0343P 2.215µm | 2.215 | 0.4 | 1.06 | 3.67 → 4.79 (+30%) | 2000 |
| PHerc0009B 2.401µm | 2.401 | 0.3 | 1.10 | 3.23 → 4.20 (+30%) | 5000 |

**Group 2 — low-res / long-distance (Paganin hurts or neutral):**

| Fragment | Pixel (µm) | Prop (m) | N_F | Outcome |
|----------|:-:|:-:|:-:|:-:|
| PHerc0500P2 4.317µm | 4.317 | 1.2 | 1.38 | −1% |
| PHerc0343P 8.640µm | 8.640 | 1.2 | 1.62 | neutral at best |
| PHerc0009B 8.640µm | 8.640 | 1.2 | 1.62 | −5% |

The split is explained by CTF analysis: volumes where the CTF peak 
frequency falls near or below the pixel Nyquist (N_F ≈ 1.1) benefit 
from filtering. Volumes where the CTF peak exceeds Nyquist do not.

### Ground truth validation (DLS fragments)

DLS scan parameters (8 µm pixel, ~0.2 m propagation, 54 keV) give 
Fresnel number ~14 — far from the phase contrast regime. The Paganin 
filter here acts as a low-pass filter, not phase retrieval. AUC 
improvement is correspondingly small:

| Fragment | Raw AUC | Filtered AUC (δ/β=5000) | Δ AUC |
|----------|:---:|:---:|:---:|
| frag1 | 0.745 | 0.753 | +0.008 |
| frag2 | 0.672 | 0.675 | +0.003 |
| frag3 | 0.709 | 0.721 | +0.013 |

The improvement is real but small and reflects noise suppression that
preserves depth structure, not fringe removal. DLS reconstruction does
not include Paganin filtering, so this is first-application smoothing
on raw absorption-reconstructed data.

**Note on ESRF volumes:** The ESRF volumes already have Paganin applied
during reconstruction (metadata.json shows delta_beta=1000 plus unsharp
masking). Applying the Paganin filter again post-reconstruction is
therefore double-filtering on ESRF data. The ESRF before/after
comparisons in this analysis should be interpreted accordingly. The DLS
fragment results above are unaffected by this issue.

### Paganin + depth features (cross-fragment generalization)

Applying Paganin (d/b=200) to each of the 65 DLS surface volume layers 
before computing 3D depth features raises cross-fragment AUC by 
+0.147–0.153 while preserving zero generalization gap. See the companion 
repo [vesuvius-depth-features] for full analysis.

## Usage
```python
from src.paganin import paganin_filter_from_params
from src.io_zarr import ESRFVolume

# Open a volume — streams from S3, no download needed
vol = ESRFVolume('PHerc0343P', '2.215um-0.4m-111keV')
slice_data = vol.get_slice(10000, axis=0, scale=0)

# Apply Paganin filter
filtered = paganin_filter_from_params(
    image=slice_data,
    pixel_size_um=vol.pixel_size_um,
    prop_distance_m=vol.prop_distance_m,
    energy_keV=vol.energy_keV,
    delta_beta=2000,
    handle_mask=True
)
```

Sweep δ/β values efficiently (pre-computes FFT once):
```python
from src.paganin import sweep_delta_beta

results = sweep_delta_beta(
    image=slice_data,
    pixel_size_um=2.215,
    prop_distance_m=0.4,
    energy_keV=111.0,
    delta_beta_values=[50, 200, 1000, 5000],
    handle_mask=True
)
```

## Installation
```bash
pip install numpy scipy s3fs zarr
```

No GPU required. All computation is FFT-based CPU work.

## Limitations

The proxy metrics (bimodal SNR, edge strength) measure papyrus fiber vs 
void contrast, not ink vs papyrus contrast. GMM mode verification 
confirms the two modes correspond to papyrus stratigraphy, not text. The 
"+30% SNR improvement" should be interpreted accordingly.

Direct ink detection validation requires ESRF surface volumes 
(generated by following the papyrus sheet through 3D space via 
segmentation), which do not yet exist for these fragments. The physics 
predicts a real benefit at N_F ≈ 1.1 relative to DLS; this awaits 
experimental confirmation.

## Files

| File | Description |
|------|-------------|
| `src/paganin.py` | Paganin filter (FFT-based, masked support, δ/β sweep) |
| `src/io_zarr.py` | OME-Zarr streaming loader — ESRFVolume class |
| `src/contrast_metrics.py` | Label-free contrast metrics |
| `scripts/discover_data.py` | S3 data catalog |
| `scripts/run_paganin_analysis.py` | Full sweep across all 7 volumes |
| `scripts/run_ctf_analysis.py` | Fresnel number and CTF analysis |
| `scripts/validate_on_dls.py` | Ground truth AUC on DLS fragments |
| `results/tables/paganin_sweep_all.csv` | 56-row sweep results |
| `results/tables/paganin_summary.csv` | Optimal parameters per volume |
| `results/tables/ground_truth_auc.csv` | DLS validation AUCs |
