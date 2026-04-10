"""Frozen law constants from the replication snapshot (2026-03-29).

These constants define the calibrated empirical model for the
observation-manifold / persistence-axis results used in the current
Medusa paper draft:

    P(x, C, tau, delta) = sigma * C * tau * kappa(x) * (1 - x * delta)

where:
    P = persistence advantage (how much temporal memory helps prediction)
    x = position on the temporal axis (0 = EM pole, 1 = GW pole)
    C = channel-match quality (max temporal autocorrelation of the signal)
    tau = depth modifier (provisional, = 1 for all non-GW domains)
    delta = carrier flag (0 = normal, 1 = EM observing a black hole)
    kappa(x) = channel-dependence (quadratic, fit on 24 domains)
    sigma = scale factor (fixed by the GW150914 calibration anchor)

How these constants were determined:

1. kappa(x) was fit by quadratic regression of channel-dependence
   (outlier score change when temporal representations are removed)
   against axis position across 24 domains on the observation manifold.
   R^2 = 0.85, p < 10^-8.

2. sigma is derived, not free. It is fixed by the GW150914 reference
   point: sigma = P_ref / (C_ref * kappa(1)), where P_ref = 0.2979
   and C_ref = 0.9872 are the measured persistence advantage and
   channel-match quality of GW150914 strain.

3. The boundary term (1 - x * delta) enforces the observed zero at
   delta = 1: when electromagnetic radiation observes a black hole
   (EHT), persistence = 0. This is confirmed on Sgr A* and M87.

These constants are FROZEN. They must not be refit when testing new
domains — the point is that new domains are evaluated against the
existing calibration to see where they land on the axis.
"""

import numpy as np

# --- kappa(x) quadratic coefficients ---
# Fit: kappa(x) = a*x^2 + b*x + c on 24-domain observation manifold
# R^2 = 0.85, p < 1e-8
KAPPA_A = 9.1686
KAPPA_B = -3.961
KAPPA_C = 0.7924

# --- Calibration anchor: GW150914 ---
P_REF = 0.297932          # Measured persistence advantage (10 seeds)
C_REF = 0.9872            # Measured channel-match quality

# --- Derived constants ---
KAPPA_1 = KAPPA_A + KAPPA_B + KAPPA_C   # kappa(1) = 6.0
SIGMA = P_REF / (C_REF * KAPPA_1)       # 0.0503


def kappa(x):
    """Channel-dependence as a function of axis position.

    kappa(0) ~ 0.79 (carrier/EM pole: low channel-dependence)
    kappa(1) ~ 6.00 (interface/GW pole: high channel-dependence)
    """
    return KAPPA_A * x**2 + KAPPA_B * x + KAPPA_C


def predicted_P(x, C, tau=1.0, delta=0.0):
    """Predicted persistence advantage from the empirical model.

    Examples:
        predicted_P(1.0, 0.987)  -> 0.298  (GW150914, calibration anchor)
        predicted_P(0.0, 0.5)    -> 0.020  (carrier-pole domain, C=0.5)
        predicted_P(1.0, 1.0, delta=1.0)  -> 0.0  (EM observing BH)
    """
    return SIGMA * C * tau * kappa(x) * (1 - x * delta)


def implied_x(P, C, tau=1.0, delta=0.0):
    """Solve for axis position x given measured P and C.

    Returns the smallest non-negative root of:
        kappa(x) = P / (sigma * C * tau)

    This tells you WHERE on the temporal axis a new domain lands,
    given the frozen calibration. The axis runs from 0 (electromagnetic
    pole, each episode sufficient) to 1 (gravitational wave pole,
    maximal accumulation needed).

    Returns None if no valid root exists (P ~ 0, C ~ 0, or off-curve).
    """
    if C < 1e-10 or abs(P) < 1e-10:
        return None
    if delta > 0:
        return None
    target = P / (SIGMA * C * tau)
    a = KAPPA_A
    b = KAPPA_B
    c = KAPPA_C - target
    disc = b**2 - 4 * a * c
    if disc < 0:
        return None
    x1 = (-b - np.sqrt(disc)) / (2 * a)
    x2 = (-b + np.sqrt(disc)) / (2 * a)
    roots = sorted([r for r in [x1, x2] if r >= 0])
    return float(roots[0]) if roots else None


# --- Paper reference values ---
# These are the published results from the paper (10 seeds each).
# When you run the pipeline, your results should match these in sign
# and approximate magnitude. Exact values will differ across seeds.
REFERENCE_RESULTS = {
    "EHT Sgr A*":   {"P":  0.000, "std": 0.000, "C": 0.000, "regime": "III (sufficient)", "delta": 1},
    "EHT M87":       {"P":  0.000, "std": 0.000, "C": 0.000, "regime": "III (holdout)",    "delta": 1},
    "Type Ia SNe":   {"P":  0.000, "std": 0.001, "C": 0.005, "regime": "III (expansion)",  "delta": 1},
    "GW170817":      {"P": -0.001, "std": 0.003, "C": None,  "regime": "III (mixed δ)",    "delta": 1},
    "IceCube":       {"P": -0.003, "std": 0.006, "C": 0.063, "regime": "III (decoupled)",  "delta": 0},
    "Solar wind":    {"P":  0.016, "std": 0.035, "C": 0.261, "regime": "I (carrier)",      "delta": 0},
    "CMB":           {"P":  0.095, "std": 0.006, "C": 0.980, "regime": "I-II (acoustic)",  "delta": 0},
    "S2 orbit":      {"P":  0.174, "std": 0.015, "C": 0.918, "regime": "I-II (force obs)", "delta": 0},
    "Quasars":       {"P":  0.227, "std": 0.024, "C": None,  "regime": "I-II (fold)",      "delta": 0},
    "Sunspots":      {"P":  0.238, "std": 0.030, "C": 1.000, "regime": "I-II (long τ)",    "delta": 0},
    "FRB":           {"P":  0.239, "std": 0.026, "C": 0.130, "regime": "catalog",          "delta": 0},
    "GW150914":      {"P":  0.298, "std": 0.079, "C": 0.987, "regime": "II (reference)",   "delta": 0},
    "GW190521":      {"P":  1.165, "std": 0.379, "C": None,  "regime": "II (deep)",        "delta": 0},
    "Weather (control)": {"P": 0.244, "std": None, "C": 0.859, "regime": "non-physical",   "delta": 0},
}
