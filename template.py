"""Template for adding a new domain to the persistence measurement.

To add your own domain:

1. Implement a download function that fetches public data to data/<name>/
2. Implement a build function that returns the standard tensor dict:
     obs:       (n_worlds, episodes, n_vars, 5)
     query:     (n_worlds, episodes, n_vars, 4)
     target:    (n_worlds, episodes, n_vars, 2)
     edges:     (n_worlds, n_vars, n_vars)
     operators: (n_worlds, n_vars, n_vars)
3. Register it in DOMAINS in domains.py
4. Run: python measure.py --domain your_domain_key

The observation vector format (dim 5):
  [0] visibility flag (1.0 if observed, 0.0 if masked)
  [1] feature 1 (e.g. time fraction, baseline length, position)
  [2] feature 2 (primary signal channel, normalized)
  [3] feature 3 (secondary signal channel or 0.0)
  [4] presence flag (1.0 if the variable exists in this episode)

Example: measuring persistence of a custom time series.
"""

from pathlib import Path

import numpy as np
import torch

from measure import measure_domain
from domains import _build_timeseries_dataset, DOMAINS, DATA_DIR


def download_my_domain():
    """Download your data to data/my_domain/."""
    my_dir = DATA_DIR / "my_domain"
    my_dir.mkdir(parents=True, exist_ok=True)
    out = my_dir / "my_data.npy"
    if out.exists():
        print("  My domain: exists")
        return

    # Replace this with your actual data download.
    # The data should be a 1D or 2D numpy array where rows are
    # sequential observations and columns are signal channels.
    print("  My domain: generating example data ...")
    rng = np.random.RandomState(42)
    n = 10000
    t = np.arange(n, dtype=np.float64)
    signal = np.sin(2 * np.pi * t / 365) + 0.3 * rng.randn(n)
    np.save(out, signal)
    print(f"  My domain: saved ({n} observations)")


def build_my_domain_dataset(*, n_worlds, episodes, n_vars, seed, keep_prob):
    """Build observation tensors from your data."""
    my_dir = DATA_DIR / "my_domain"
    data = np.load(my_dir / "my_data.npy")
    return _build_timeseries_dataset(
        data, n_worlds=n_worlds, episodes=episodes,
        n_vars=n_vars, seed=seed, keep_prob=keep_prob)


if __name__ == "__main__":
    # Register the domain
    DOMAINS["my_domain"] = {
        "download": download_my_domain,
        "build": build_my_domain_dataset,
        "name": "My Domain",
        "regime": "unknown",
    }

    # Download and measure
    download_my_domain()
    result = measure_domain("my_domain", seed=0, epochs=30)

    print(f"\nResult: P = {result['P']:+.4f}, C = {result['C']:.3f}")
    if result["implied_x"] is not None:
        print(f"Implied axis position: x = {result['implied_x']:.3f}")
    else:
        print("Implied axis position: not computable (P ≈ 0 or C ≈ 0)")
