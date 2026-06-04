"""diebold_mariano.py
--------------------
Diebold-Mariano test: DRL agent (run_043) vs Fixed-Rule and Monte Carlo ALM.

Three loss functions (applied monthly over the 85-step test period):
  L1  Squared FR deviation from target  (FR_t - 1.05)^2       [stability]
  L2  Negative portfolio return         -r_p_t                  [performance]
  L3  Buffer shortfall                  max(0, 0.001 - B_t)     [solvency]

Test statistic: Harvey, Leybourne & Newbold (1997) small-sample correction
  DM* = sqrt((T + 1 - 2h + h(h-1)/T) / T) * DM
where h=1 (one-step-ahead), DM = d_bar / sqrt(LRV/T),
LRV estimated via Newey-West with automatic bandwidth selection.

H0: E[L_t(DRL)] = E[L_t(baseline)]  (equal predictive accuracy)
H1: E[L_t(DRL)] < E[L_t(baseline)]  (DRL is better — lower loss)
One-sided p-value reported (lower tail).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from scipy import stats

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Newey-West long-run variance estimator
# ---------------------------------------------------------------------------

def newey_west_lrv(d: np.ndarray, max_lag: int | None = None) -> float:
    """Newey-West HAC long-run variance of the loss differential series d.

    Uses Bartlett kernel with automatic bandwidth = floor(4*(T/100)^(2/9)).

    Args:
        d:        Loss differential series (T,).
        max_lag:  Override automatic bandwidth.

    Returns:
        Long-run variance estimate (scalar).
    """
    T = len(d)
    if max_lag is None:
        max_lag = int(np.floor(4 * (T / 100) ** (2 / 9)))

    d_dm   = d - d.mean()
    gamma0 = np.dot(d_dm, d_dm) / T
    lrv    = gamma0
    for lag in range(1, max_lag + 1):
        w       = 1.0 - lag / (max_lag + 1)          # Bartlett weight
        gamma_l = np.dot(d_dm[lag:], d_dm[:-lag]) / T
        lrv    += 2 * w * gamma_l
    return max(lrv, 1e-12)   # floor to avoid division by zero


# ---------------------------------------------------------------------------
# Diebold-Mariano test (HLN small-sample correction)
# ---------------------------------------------------------------------------

def dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
) -> dict:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold correction.

    Tests H0: E[L_a] = E[L_b]  vs  H1: E[L_a] < E[L_b]  (A is better).

    Args:
        loss_a:  Loss series for model A (DRL).
        loss_b:  Loss series for model B (baseline).
        h:       Forecast horizon (default 1 for monthly steps).

    Returns:
        Dict with keys: d_bar, dm_stat, dm_hln, p_value_one, p_value_two.
    """
    T   = len(loss_a)
    d   = loss_a - loss_b            # negative = A is better
    d_bar = d.mean()

    lrv     = newey_west_lrv(d)
    dm_stat = d_bar / np.sqrt(lrv / T)

    # HLN correction factor
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln     = hln_factor * dm_stat

    # t-distribution with T-1 degrees of freedom (HLN recommendation)
    p_one = float(stats.t.cdf(dm_hln, df=T - 1))   # lower tail: A < B
    p_two = float(2 * min(p_one, 1 - p_one))

    return {
        "d_bar":         float(d_bar),
        "dm_stat":       float(dm_stat),
        "dm_hln":        float(dm_hln),
        "p_value_one":   p_one,
        "p_value_two":   p_two,
        "T":             T,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    model_dir = Path("src/models/run_043")

    # ---- Load trajectories ------------------------------------------------ #
    drl = np.load(model_dir / "trajectory_drl_ppo.npz")
    fxd = np.load(model_dir / "trajectory_fixed-rule.npz")
    mc  = np.load(model_dir / "trajectory_monte_carlo.npz")

    FR_target = 1.05

    # ---- Loss functions --------------------------------------------------- #
    losses = {
        "L1  Squared FR deviation": {
            "DRL":   (drl["FR"] - FR_target) ** 2,
            "Fixed": (fxd["FR"] - FR_target) ** 2,
            "MC":    (mc["FR"]  - FR_target) ** 2,
            "desc":  "$(\\mathrm{FR}_t - 1.05)^2$",
        },
        "L2  Negative portfolio return": {
            "DRL":   -drl["r_p"],
            "Fixed": -fxd["r_p"],
            "MC":    -mc["r_p"],
            "desc":  "$-r_{p,t}$",
        },
        "L3  Buffer shortfall": {
            "DRL":   np.maximum(0.0, 0.001 - drl["B"]),
            "Fixed": np.maximum(0.0, 0.001 - fxd["B"]),
            "MC":    np.maximum(0.0, 0.001 - mc["B"]),
            "desc":  "$\\max(0,\\, 0.001 - B_t)$",
        },
    }

    comparisons = [
        ("DRL vs Fixed-Rule", "DRL", "Fixed"),
        ("DRL vs Monte Carlo", "DRL", "MC"),
    ]

    print("=" * 72)
    print("Diebold-Mariano Test  —  run_042 vs Baselines")
    print(f"  Test period: 85 monthly steps  |  HAC bandwidth: automatic")
    print(f"  H1 (one-sided): DRL has lower loss than baseline")
    print("=" * 72)

    results = {}

    for loss_name, loss_dict in losses.items():
        print(f"\n{loss_name}  [{loss_dict['desc']}]")
        print(f"  {'Comparison':<28}  {'d_bar':>8}  {'DM':>7}  {'DM*':>7}  "
              f"{'p (1-sided)':>11}  {'p (2-sided)':>11}  {'Sig':>4}")
        print("  " + "-" * 70)

        results[loss_name] = {}
        for comp_label, a_key, b_key in comparisons:
            r = dm_test(loss_dict[a_key], loss_dict[b_key])
            sig = ("***" if r["p_value_one"] < 0.01
                   else "**"  if r["p_value_one"] < 0.05
                   else "*"   if r["p_value_one"] < 0.10
                   else "")
            print(f"  {comp_label:<28}  {r['d_bar']:>8.5f}  "
                  f"{r['dm_stat']:>7.3f}  {r['dm_hln']:>7.3f}  "
                  f"{r['p_value_one']:>11.4f}  {r['p_value_two']:>11.4f}  {sig:>4}")
            results[loss_name][comp_label] = r

    # ---- Mean loss summary ----------------------------------------------- #
    print("\n" + "=" * 72)
    print("Mean Loss Summary")
    print("=" * 72)
    print(f"  {'Loss function':<34}  {'DRL':>8}  {'Fixed':>8}  {'MC':>8}")
    print("  " + "-" * 62)
    for loss_name, loss_dict in losses.items():
        print(f"  {loss_name:<34}  "
              f"{loss_dict['DRL'].mean():>8.5f}  "
              f"{loss_dict['Fixed'].mean():>8.5f}  "
              f"{loss_dict['MC'].mean():>8.5f}")

    print("\n  Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print("=" * 72)


if __name__ == "__main__":
    main()
