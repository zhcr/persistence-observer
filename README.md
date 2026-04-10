# Persistence Observer

Reproducibility package for the observation-manifold / persistence-axis
measurement described in Hayes (2026), *Physical Law from
Bandwidth-Limited Observation*.

## What this measures

How much does temporal memory help a bandwidth-limited observer predict
the structure of a physical domain?

We fix everything about the observer — architecture, loss, training,
evaluation — and change only the data. Then we measure one number per
domain: the **persistence advantage P**, the gap between a persistent
observer (GRU) and a memoryless baseline (last-episode-only).

P > 0 means temporal memory helps. P ≈ 0 means each episode is sufficient.

## Quick start

```bash
pip install -r requirements.txt

# Download data for all 13 real-data domains plus the weather control
python measure.py --download

# Measure all domains (1 seed, ~20 min on GPU)
python measure.py --domain all --seeds 1

# Visualize results: comparison table + axis plot
python visualize.py
```

Each measurement row in `results/persistence_measurement.json` includes:

- `source_kind`: `real`, `synthetic_fallback`, or `derived`
- `source_label`: the source used for this run
- `source_note`: a short provenance note

## The empirical law

Across 13 real-data domains from independent instruments, persistence
advantage is described by:

```
P = σ · C · τ · κ(x) · (1 − xδ)
```

where:
- **x** = position on the temporal axis (0 = EM pole, 1 = GW pole)
- **κ(x)** = 9.17x² − 3.96x + 0.79 (channel-dependence, R² = 0.85)
- **C** = channel-match quality (max temporal autocorrelation, measurable before training)
- **σ** = 0.0503 (scale factor, calibrated from GW150914)
- **δ** = carrier flag (1 when EM observes a black hole, 0 otherwise)
- **τ** = depth modifier (provisional, = 1 for all non-GW domains)

See `constants.py` for the frozen calibration values and their derivation.

## Reference results

| Domain | P (paper) | C | Regime | Source |
|--------|-----------|---|--------|--------|
| EHT Sgr A* | ≈ 0 | 0.000 | III: sufficient | EHTC 2022 |
| EHT M87 | ≈ 0 | 0.000 | III: sufficient | EHTC 2019 |
| Type Ia SNe | ≈ 0 | 0.005 | III: expansion | Pantheon+ |
| GW170817 | ≈ 0 | — | III: mixed δ | GWOSC O2 |
| IceCube ν | ≈ 0 | 0.063 | III: decoupled | HEASARC |
| Solar wind | +0.016 | 0.261 | I: carrier | NASA OMNI |
| CMB | +0.095 | 0.980 | I-II: acoustic | Planck |
| S2 orbit | +0.174 | 0.918 | I-II: force obs | Gillessen+ 2009 |
| Quasars | +0.227 | — | I-II: fold | SDSS Stripe 82 |
| Sunspots | +0.238 | 1.000 | I-II: long τ | SILSO |
| FRB | +0.239 | 0.130 | catalog | CHIME |
| GW150914 | +0.298 | 0.987 | II: reference | GWOSC O1 |
| GW190521 | +1.165 | — | II: deep | GWOSC O3 |

**What you should see:** The sign and regime of each domain should match.
Exact P values vary across seeds; magnitudes should be within a factor
of ~2-3x on 1 seed, and converge closer with 10 seeds.

## Data sources and reliability

All data downloads from public sources. No data is redistributed in
this repo. Some public URLs are intermittently unavailable; the pipeline
uses multiple fallback URLs and generates realistic synthetic data as a
last resort. Synthetic fallbacks are now recorded explicitly in the
runtime output and in `results/persistence_measurement.json`.

| Domain | Primary source | Status |
|--------|---------------|--------|
| LIGO strain | [GWOSC](https://gwosc.org) | Stable (API + fallback URLs) |
| EHT visibilities | [EHT GitHub](https://github.com/eventhorizontelescope) | Stable |
| CMB spectrum | [Planck/ESA](http://pla.esac.esa.int) | Stable |
| Type Ia SNe | [Pantheon+](https://github.com/PantheonPlusSH0ES/DataRelease) | Stable |
| Solar wind | [NASA SPDF](https://spdf.gsfc.nasa.gov) | Stable |
| Sunspots | [SILSO](https://www.sidc.be/SILSO/) | Stable |
| S2 orbit | Generated from published orbital parameters | Always works |
| IceCube | [HEASARC](https://heasarc.gsfc.nasa.gov) | Stable (batch query) |
| Weather | [NOAA NCEI](https://www.ncei.noaa.gov) | Stable |
| Quasars | [UW/SDSS](https://faculty.washington.edu/ivezic/) | Intermittent — may use synthetic fallback |
| FRB | [CHIME](https://chime-frb-open-data.github.io) | Intermittent — may use synthetic fallback |

## Bring your own domain

```python
from domains import _build_timeseries_dataset, DOMAINS
from measure import measure_domain

DOMAINS["my_domain"] = {
    "download": my_download_fn,
    "build": my_build_fn,
    "name": "My Domain",
    "regime": "unknown",
}

result = measure_domain("my_domain", seed=0)
print(f"P = {result['P']:+.4f}, C = {result['C']:.3f}")
```

See `template.py` for a complete working example. If your domain is a
simple time series, `_build_timeseries_dataset` handles tensor
construction — you just provide the numpy array.

## Files

| File | What |
|------|------|
| `observer.py` | PersistentObserver model (GRU + memoryless baseline) |
| `domains.py` | Data downloaders and dataset builders for 13 real-data domains plus controls |
| `measure.py` | Main pipeline: download, train, evaluate P and C |
| `constants.py` | Frozen law constants with full derivation context |
| `visualize.py` | Results table, axis plot, and regime summary |
| `template.py` | Bring-your-own-domain example |

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- h5py ≥ 3.8 (for LIGO HDF5 strain files)
- matplotlib ≥ 3.7 (for visualization, optional for measurement)
- GPU recommended (~20 min for all domains on A100; CPU works but slower)

## License

MIT
