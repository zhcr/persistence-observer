"""Visualize persistence results on the temporal axis.

Produces:
  1. A comparison table of measured vs paper values
  2. The temporal axis plot (P vs implied x, with kappa curves)
  3. A summary of which domains matched the paper

Usage:
    python visualize.py                          # use latest results
    python visualize.py results/my_results.json  # use specific file
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from constants import (kappa, predicted_P, implied_x, SIGMA,
                       REFERENCE_RESULTS, KAPPA_A, KAPPA_B, KAPPA_C)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results(path=None):
    if path is None:
        candidates = sorted(RESULTS_DIR.glob("*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("No results found in results/. Run measure.py first.")
            sys.exit(1)
        path = candidates[0]
    else:
        path = Path(path)
    with open(path) as f:
        return json.load(f), path


def print_comparison_table(data):
    """Print measured vs paper values side by side."""
    rows = data.get("rows", [])
    summary = data.get("summary", {})

    by_domain = {}
    for r in rows:
        key = r["domain"]
        by_domain.setdefault(key, []).append(r)

    print()
    print("=" * 78)
    print("PERSISTENCE MEASUREMENT vs PAPER")
    print("=" * 78)
    print(f"{'Domain':<22s} {'Measured P':>11s} {'Paper P':>9s} "
          f"{'C meas':>7s} {'C paper':>8s} {'Source':>10s} {'Match':>6s}")
    print("-" * 92)

    matches = 0
    total = 0

    for key, runs in by_domain.items():
        name = runs[0].get("name", key)
        mean_P = np.mean([r["P"] for r in runs])
        std_P = np.std([r["P"] for r in runs]) if len(runs) > 1 else 0
        C = runs[0].get("C", None)
        source_kind = runs[0].get("source_kind", "unknown")

        ref = REFERENCE_RESULTS.get(name, {})
        paper_P = ref.get("P")
        paper_C = ref.get("C")

        total += 1
        if paper_P is not None:
            sign_match = (mean_P > 0.005) == (paper_P > 0.005) or \
                         (abs(mean_P) < 0.01 and abs(paper_P) < 0.01)
            match_str = "YES" if sign_match else "NO"
            if sign_match:
                matches += 1
        else:
            match_str = "?"

        p_str = f"{mean_P:+.4f}" + (f"±{std_P:.3f}" if std_P > 0 else "")
        pp_str = f"{paper_P:+.3f}" if paper_P is not None else "  --"
        c_str = f"{C:.3f}" if C is not None else "  --"
        cp_str = f"{paper_C:.3f}" if paper_C is not None else "  --"
        source_str = (
            "synthetic"
            if source_kind == "synthetic_fallback"
            else ("derived" if source_kind == "derived" else source_kind)
        )

        print(f"{name:<22s} {p_str:>11s} {pp_str:>9s} "
              f"{c_str:>7s} {cp_str:>8s} {source_str:>10s} {match_str:>6s}")

    print("-" * 92)
    print(f"Sign matches: {matches}/{total}")
    print()


def generate_axis_plot(data, out_path=None):
    """Generate the temporal axis plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation.")
        print("Install with: pip install matplotlib")
        return

    rows = data.get("rows", [])
    by_domain = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: P vs implied x with kappa curves
    x_curve = np.linspace(0, 1.2, 200)
    for C_val, ls, label in [(1.0, "-", "C=1.0"),
                              (0.5, "--", "C=0.5"),
                              (0.25, ":", "C=0.25")]:
        P_curve = [predicted_P(xi, C_val) for xi in x_curve]
        ax1.plot(x_curve, P_curve, ls, color="gray", alpha=0.5, label=label)

    colors_regime = {
        "III": "#d62728", "I": "#2ca02c", "I-II": "#1f77b4",
        "II": "#ff7f0e", "catalog": "#9467bd", "non-physical": "#7f7f7f",
    }

    for key, runs in by_domain.items():
        name = runs[0].get("name", key)
        regime = runs[0].get("regime", "")
        mean_P = np.mean([r["P"] for r in runs])
        x_vals = [r.get("implied_x") for r in runs if r.get("implied_x")]
        x_pos = np.mean(x_vals) if x_vals else None

        base_regime = regime.split(" ")[0].rstrip("(")
        color = colors_regime.get(base_regime, "#333333")

        if x_pos is not None and mean_P > 0.005:
            ax1.scatter(x_pos, mean_P, s=60, c=color, zorder=5, edgecolors="k",
                        linewidths=0.5)
            ax1.annotate(name, (x_pos, mean_P), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
        elif abs(mean_P) < 0.01:
            ax1.scatter(0.0, mean_P, s=40, c=color, marker="v", zorder=5,
                        edgecolors="k", linewidths=0.5)
            ax1.annotate(name, (0.0, mean_P), fontsize=6,
                         xytext=(4, -8), textcoords="offset points")

    # Add paper reference points
    for name, ref in REFERENCE_RESULTS.items():
        paper_P = ref["P"]
        paper_C = ref.get("C")
        if paper_P > 0.01 and paper_C and paper_C > 0.01:
            x_ref = implied_x(paper_P, paper_C)
            if x_ref is not None:
                ax1.scatter(x_ref, paper_P, s=30, facecolors="none",
                            edgecolors="gray", linewidths=0.8, alpha=0.5,
                            zorder=3)

    ax1.set_xlabel("Implied axis position x")
    ax1.set_ylabel("Persistence advantage P")
    ax1.set_title("Temporal Axis: P vs x")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_xlim(-0.1, 1.3)
    ax1.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)

    # Panel B: Measured P vs Paper P
    measured, paper, names_list = [], [], []
    for key, runs in by_domain.items():
        name = runs[0].get("name", key)
        ref = REFERENCE_RESULTS.get(name, {})
        if ref.get("P") is not None:
            measured.append(np.mean([r["P"] for r in runs]))
            paper.append(ref["P"])
            names_list.append(name)

    if measured:
        ax2.scatter(paper, measured, s=60, c="#1f77b4", edgecolors="k",
                    linewidths=0.5, zorder=5)
        for i, name in enumerate(names_list):
            ax2.annotate(name, (paper[i], measured[i]), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
        lim = max(max(abs(v) for v in measured + paper) * 1.1, 0.5)
        ax2.plot([-0.1, lim], [-0.1, lim], "k--", alpha=0.3, linewidth=0.8)
        ax2.set_xlabel("Paper P")
        ax2.set_ylabel("Measured P")
        ax2.set_title("Replication: Measured vs Paper")
        corr = np.corrcoef(paper, measured)[0, 1] if len(paper) > 2 else 0
        ax2.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax2.transAxes,
                 fontsize=10, va="top")

    plt.tight_layout()
    if out_path is None:
        out_path = RESULTS_DIR / "temporal_axis.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.close()


def print_regime_summary(data):
    """Print a plain-language summary of what the results mean."""
    rows = data.get("rows", [])
    by_domain = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)

    zero_P = []
    positive_P = []
    synthetic = []
    for key, runs in by_domain.items():
        name = runs[0].get("name", key)
        mean_P = np.mean([r["P"] for r in runs])
        if runs[0].get("source_kind") == "synthetic_fallback":
            synthetic.append(name)
        if abs(mean_P) < 0.01:
            zero_P.append(name)
        elif mean_P > 0:
            positive_P.append((name, mean_P))

    positive_P.sort(key=lambda x: x[1])

    print("INTERPRETATION")
    print("=" * 60)
    if zero_P:
        print(f"\nRegime III (P ≈ 0, each episode sufficient):")
        for name in zero_P:
            print(f"  {name}")
        print("  -> Temporal memory does not help. Each observation")
        print("     already contains the full signal.")

    if positive_P:
        print(f"\nRegimes I-II (P > 0, accumulation helps):")
        for name, P in positive_P:
            print(f"  {name:<22s}  P = {P:+.4f}")
        print("  -> The observer benefits from remembering past episodes.")
        print("     Higher P = more temporal structure to accumulate.")
    if synthetic:
        print(f"\nSynthetic fallback domains in this run:")
        for name in synthetic:
            print(f"  {name}")
        print("  -> These domains were measured on generated fallback data,")
        print("     not on live public downloads for this run.")
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data, used_path = load_results(path)
    print(f"Using results from: {used_path}")

    print_comparison_table(data)
    print_regime_summary(data)
    generate_axis_plot(data)


if __name__ == "__main__":
    main()
