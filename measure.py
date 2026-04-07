"""Measure persistence advantage P and channel-match quality C for a domain.

Usage:
    python measure.py --download                    # download all data
    python measure.py --domain ligo_gw150914        # measure one domain
    python measure.py --domain all --seeds 10       # measure all domains

This is the core pipeline from the paper:
  1. Build dataset from real instrument data
  2. Train two observers: GRU (persistent) and LAST (memoryless)
  3. Evaluate both on held-out test worlds
  4. Compute P = R_last - R_accum (persistence advantage)
  5. Compute C (channel-match quality) from raw observations
  6. Compute implied axis position x from the calibrated law
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from observer import PersistentObserver, train_model, evaluate, DEVICE
from domains import DOMAINS, get_domain_source_info
from constants import implied_x, REFERENCE_RESULTS

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def compute_C(obs_tensor, episodes, n_vars):
    """Compute channel-match quality C from observation tensor.

    C = max over lags of the squared Pearson correlation of observation
    features across consecutive episodes. Measures how much temporal
    structure the signal carries — computable before training.
    """
    obs = obs_tensor.numpy()
    n_worlds = obs.shape[0]
    best_r2 = 0.0

    for feat_idx in [2, 3]:
        for lag in range(1, max(1, episodes - 2)):
            curr_vals, next_vals = [], []
            for wi in range(n_worlds):
                for vi in range(n_vars):
                    pairs = []
                    for ei in range(episodes - lag):
                        if (obs[wi, ei, vi, 0] > 0.5 and
                                obs[wi, ei + lag, vi, 0] > 0.5):
                            pairs.append((
                                obs[wi, ei, vi, feat_idx],
                                obs[wi, ei + lag, vi, feat_idx]))
                    for c, n in pairs:
                        curr_vals.append(c)
                        next_vals.append(n)
            if len(curr_vals) < 10:
                continue
            x = np.array(curr_vals, dtype=np.float64)
            y = np.array(next_vals, dtype=np.float64)
            if x.std() < 1e-10 or y.std() < 1e-10:
                continue
            r = np.corrcoef(x, y)[0, 1]
            if np.isfinite(r):
                best_r2 = max(best_r2, r**2)

    return round(float(best_r2), 4)


def measure_domain(domain_key, *, seed, n_worlds=256, test_worlds=64,
                   episodes=16, n_vars=20, epochs=30, batch_size=32,
                   lr=2e-3, hidden_dim=128, latent_dim=64, keep_prob=0.5,
                   verbose=True):
    """Measure P and C for a single domain at a single seed."""
    info = DOMAINS[domain_key]
    build_fn = info["build"]

    train_data = build_fn(
        n_worlds=n_worlds, episodes=episodes, n_vars=n_vars,
        seed=seed, keep_prob=keep_prob)
    test_data = build_fn(
        n_worlds=test_worlds, episodes=episodes, n_vars=n_vars,
        seed=seed + 999, keep_prob=keep_prob)

    results = {}
    for accum in ["gru", "last"]:
        model = PersistentObserver(
            obs_dim=5, query_dim=4, n_vars=n_vars, n_ops=3,
            hidden_dim=hidden_dim, latent_dim=latent_dim,
            target_dim=2, accum_mode=accum,
        ).to(DEVICE)
        model = train_model(
            model, train_data, epochs=epochs, batch_size=batch_size,
            lr=lr, max_observe=episodes - 1, verbose=False)
        mse = evaluate(
            model, test_data, n_observe=episodes - 1,
            batch_size=batch_size)
        results[accum] = mse

    P = results["last"] - results["gru"]
    C = compute_C(train_data["obs"], episodes, n_vars)
    x = implied_x(P, C) if C > 0.01 and P > 0.001 else None
    source = get_domain_source_info(domain_key)

    row = {
        "domain": domain_key,
        "name": info["name"],
        "seed": seed,
        "gru_mse": results["gru"],
        "last_mse": results["last"],
        "P": round(float(P), 6),
        "C": C,
        "implied_x": round(x, 3) if x is not None else None,
        "regime": info["regime"],
        "source_kind": source["source_kind"],
        "source_label": source["source_label"],
        "source_url": source["source_url"],
        "source_note": source["source_note"],
    }

    if verbose:
        ref = REFERENCE_RESULTS.get(info["name"], {})
        ref_P = ref.get("P", "?")
        x_str = f"{x:.3f}" if x is not None else "--"
        source_str = f"{source['source_kind']}:{source['source_label']}"
        print(f"  {info['name']:20s}  P={P:+.4f}  C={C:.3f}"
              f"  x={x_str}  (paper: P={ref_P})"
              f"  [{source_str}]")

    return row


def main():
    p = argparse.ArgumentParser(
        description="Measure persistence advantage across physical domains")
    p.add_argument("--download", action="store_true",
                   help="Download all domain data")
    p.add_argument("--domain", default="all",
                   choices=list(DOMAINS) + ["all"],
                   help="Domain to measure (default: all)")
    p.add_argument("--seeds", type=int, default=3,
                   help="Number of seeds (default: 3)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-vars", type=int, default=20)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--train-worlds", type=int, default=256)
    p.add_argument("--test-worlds", type=int, default=64)
    args = p.parse_args()

    if args.download:
        print("=" * 60)
        print("DOWNLOADING ALL DOMAIN DATA")
        print("=" * 60)
        for key, info in DOMAINS.items():
            try:
                info["download"]()
            except Exception as e:
                print(f"  {key}: FAILED: {e}")
        print("=" * 60)
        return

    domains = (DOMAINS if args.domain == "all"
               else {args.domain: DOMAINS[args.domain]})

    print("=" * 60)
    print("PERSISTENCE MEASUREMENT")
    print(f"  domains: {list(domains)}")
    print(f"  seeds: 0..{args.seeds}")
    print(f"  device: {DEVICE}")
    print("=" * 60)

    t0 = time.time()
    all_rows = []

    for domain_key in domains:
        for seed in range(args.seeds):
            try:
                row = measure_domain(
                    domain_key, seed=seed,
                    n_worlds=args.train_worlds,
                    test_worlds=args.test_worlds,
                    episodes=args.episodes,
                    n_vars=args.n_vars,
                    epochs=args.epochs)
                all_rows.append(row)
            except Exception as e:
                print(f"  FAILED {domain_key} s={seed}: {e}")

    elapsed = round(time.time() - t0, 1)

    summary = {}
    for domain_key in domains:
        sub = [r for r in all_rows if r["domain"] == domain_key]
        if sub:
            gaps = np.array([r["P"] for r in sub])
            summary[domain_key] = {
                "name": sub[0]["name"],
                "mean_P": round(float(gaps.mean()), 6),
                "std_P": round(float(gaps.std()), 6),
                "C": sub[0]["C"],
                "n_seeds": len(sub),
                "source_kind": sub[0]["source_kind"],
                "source_label": sub[0]["source_label"],
                "source_note": sub[0]["source_note"],
            }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "persistence_measurement.json"
    payload = {
        "rows": all_rows,
        "summary": summary,
        "elapsed_s": elapsed,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"{'Domain':20s}  {'P (mean±std)':18s}  {'C':6s}  {'Source':14s}  Paper P")
    print("-" * 86)
    for key, s in summary.items():
        ref = REFERENCE_RESULTS.get(s["name"], {})
        ref_P = ref.get("P", "?")
        source_tag = (
            "synthetic"
            if s["source_kind"] == "synthetic_fallback"
            else ("derived" if s["source_kind"] == "derived" else "real")
        )
        print(f"{s['name']:20s}  {s['mean_P']:+.4f}±{s['std_P']:.4f}"
              f"      {s['C']:.3f}  {source_tag:14s}  {ref_P}")

    print(f"\nSaved: {out_path}")
    print(f"Elapsed: {elapsed}s ({elapsed / 3600:.2f}h)")


if __name__ == "__main__":
    main()
