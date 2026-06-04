"""analyze_run_032_temporal.py — Temporal distribution analysis for run_032.

Loads per-step trajectory arrays saved by evaluate.py and produces:
  (a) DRL agent temporal distribution diagnostics
  (b) Fixed-Rule temporal distribution diagnostics
  (c) Regime-depletion overlap analysis

Usage
-----
    py src/analyze_run_032_temporal.py --log-dir src/models/run_032_rerun

Output is printed to stdout and saved to <log_dir>/temporal_analysis.txt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEPLETION_THRESH = 0.001          # B <= this => depleted (matches metrics.py)
VSTOXX_LO        = 20.0           # matches metrics.py regime thresholds
VSTOXX_HI        = 30.0


def _load_traj(log_dir: Path, safe_name: str) -> dict:
    path = log_dir / f"trajectory_{safe_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _b_histogram(B_dist: np.ndarray, lines: list[str]) -> None:
    """Append buffer-level histogram lines for distribution months."""
    bins   = [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, np.inf]
    labels = ["[0.00, 0.01)", "[0.01, 0.02)", "[0.02, 0.05)",
              "[0.05, 0.10)", "[0.10, 0.15]"]
    total  = max(len(B_dist), 1)
    for i, label in enumerate(labels):
        lo, hi = bins[i], bins[i + 1]
        if i == len(labels) - 1:
            mask = (B_dist >= lo)
        else:
            mask = (B_dist >= lo) & (B_dist < hi)
        n   = int(mask.sum())
        pct = 100.0 * n / total
        lines.append(f"    B in {label}:  {n:3d} months ({pct:5.1f}%)")


def _temporal_stats(traj: dict, label: str, lines: list[str]) -> None:
    """Compute and append temporal distribution stats for one agent."""
    B       = traj["B"].astype(float)
    d       = traj["d_tilde"].astype(float)
    T       = len(d)

    # "B at start of distribution month" = B from the previous step
    # (B[t] in trajectory is post-step state; agent acted on B[t-1])
    B_prev  = np.empty(T)
    B_prev[0]  = B[0]           # no prior step — use current
    B_prev[1:] = B[:-1]

    dist_mask  = d > 1e-8       # months with actual distribution
    n_dist     = int(dist_mask.sum())
    B_at_dist  = B_prev[dist_mask]

    lines.append(f"\n  {label}")
    lines.append(f"  {'=' * 56}")
    lines.append(f"  Total months with d_tilde > 0 : {n_dist} / {T} "
                 f"({100.0 * n_dist / T:.1f}%)")

    if n_dist == 0:
        lines.append("  (no distribution months — histogram not available)")
        return

    lines.append(f"\n  Buffer level at start of distribution months:")
    _b_histogram(B_at_dist, lines)
    lines.append(f"  Median B at distribution months : {np.median(B_at_dist):.4f}")
    lines.append(f"  Mean   B at distribution months : {np.mean(B_at_dist):.4f}")
    lines.append(f"  Mean d_tilde in distribution months : {d[dist_mask].mean():.5f}")

    # Temporal correlation: P(depletion in t+1..t+3 | d>0 at t)
    # vs P(depletion in t+1..t+3 | d==0 at t)
    depl    = B <= DEPLETION_THRESH      # depletion indicator per step

    def _p_depl_next3(mask: np.ndarray) -> float:
        """Fraction of mask months where any of t+1..t+3 is a depletion month."""
        idx  = np.where(mask)[0]
        hits = 0
        for t in idx:
            horizon = depl[t + 1 : min(t + 4, T)]
            if len(horizon) > 0 and horizon.any():
                hits += 1
        return hits / max(len(idx), 1)

    p_depl_given_dist   = _p_depl_next3(dist_mask)
    p_depl_given_nodist = _p_depl_next3(~dist_mask)

    lines.append(f"\n  Forward depletion probability (horizon 3 months):")
    lines.append(f"    P(depletion in t+1..t+3 | d > 0 at t)  : "
                 f"{p_depl_given_dist:.3f}  (n={n_dist})")
    lines.append(f"    P(depletion in t+1..t+3 | d == 0 at t) : "
                 f"{p_depl_given_nodist:.3f}  (n={T - n_dist})")


def _regime_overlap(traj: dict, vstoxx: np.ndarray, label: str,
                    lines: list[str]) -> None:
    """Append regime-depletion overlap table for one agent."""
    B    = traj["B"].astype(float)
    depl = B <= DEPLETION_THRESH

    regimes = {
        "Low    (VSTOXX < 20)":  vstoxx < VSTOXX_LO,
        "Medium (20 <= V < 30)": (vstoxx >= VSTOXX_LO) & (vstoxx < VSTOXX_HI),
        "High   (VSTOXX >= 30)": vstoxx >= VSTOXX_HI,
    }

    lines.append(f"\n  {label}")
    lines.append(f"  {'=' * 56}")
    lines.append(f"  {'Regime':<26}  {'n_regime':>8}  {'n_depl':>7}  "
                 f"{'frac_depl':>10}  {'n_dist_0':>8}  {'frac_dist_0':>12}")
    lines.append(f"  {'-' * 76}")

    d = traj["d_tilde"].astype(float)

    for regime_label, mask in regimes.items():
        # align vstoxx to trajectory length
        m       = mask[:len(B)]
        n_reg   = int(m.sum())
        n_depl  = int((depl & m).sum())
        frac_d  = n_depl / max(n_reg, 1)
        n_zero  = int(((d < 1e-8) & m).sum())
        frac_z  = n_zero / max(n_reg, 1)
        lines.append(f"  {regime_label:<26}  {n_reg:>8}  {n_depl:>7}  "
                     f"{frac_d:>10.3f}  {n_zero:>8}  {frac_z:>12.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Temporal distribution analysis for run_032",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log-dir", type=str,
                   default="src/models/run_032_rerun",
                   help="Directory containing trajectory_*.npz files from evaluate.py")
    p.add_argument("--vstoxx-src", type=str,
                   default=None,
                   help="Optional path to override VSTOXX source (default: auto from pipeline)")
    args = p.parse_args(argv)

    log_dir = Path(args.log_dir)

    # ---- Load vstoxx for regime analysis --------------------------------- #
    from src.data_pipeline import run_pipeline
    results  = run_pipeline()
    vstoxx   = results["z_test_raw"]["vstoxx_level"].values   # length = test months

    # ---- Load trajectories ----------------------------------------------- #
    drl_traj   = _load_traj(log_dir, "drl_ppo")
    fixed_traj = _load_traj(log_dir, "fixed-rule")

    # ---- Build output lines ----------------------------------------------- #
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("Temporal Distribution Analysis -- run_032 (DRL vs Fixed-Rule)")
    lines.append("=" * 64)

    # (a)/(b) Temporal distribution stats
    lines.append("\n[A] Distribution behaviour by agent")
    lines.append("-" * 64)
    _temporal_stats(drl_traj,   "DRL (PPO)", lines)
    _temporal_stats(fixed_traj, "Fixed-Rule", lines)

    # (c) Regime-depletion overlap
    lines.append("\n" + "=" * 64)
    lines.append("[B] Regime-depletion overlap")
    lines.append("    Columns: n_regime = months in regime; n_depl = depletion months")
    lines.append("    (B <= 0.001) within regime; frac_dist_0 = fraction with d=0")
    lines.append("-" * 64)
    _regime_overlap(drl_traj,   vstoxx, "DRL (PPO)", lines)
    _regime_overlap(fixed_traj, vstoxx, "Fixed-Rule", lines)

    lines.append("\n" + "=" * 64)

    output = "\n".join(lines)
    print(output)

    out_path = log_dir / "temporal_analysis.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
