"""
table_regime_equity.py
----------------------
Table 4.5.1 — Equity tilt by VSTOXX regime (run_042 out-of-sample).

For each VSTOXX regime (Low <20, Medium 20-30, High ≥30) computes:
  n          — number of monthly observations
  DRL w_eq   — mean aggregate equity weight reconstructed from portfolio returns
  Fixed w_eq — constant 0.55 (lifecycle aggregate: 0.20*0.85 + 0.35*0.70 + 0.45*0.30)
  Target β̄   — GMM regime risk budget (Low 0.65, Med 0.55, High 0.35)

Equity weight back-computation:
  r_p = w_eq_agg * r_eq + (1 - w_eq_agg) * r_bond
  => e_t = (r_p - 0.55*r_eq - 0.45*r_bond) / (r_eq - r_bond)
  => w_eq_agg = clip(0.55 + e_t, 0.30, 0.80)
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.data_pipeline import run_pipeline
from src.environment   import EnvConfig

# ── Load trajectory ──────────────────────────────────────────────────────────
drl  = np.load("src/models/run_042/trajectory_drl_ppo.npz")
dates = pd.to_datetime(drl["dates"])
T     = len(dates)

# ── Load market data ─────────────────────────────────────────────────────────
results = run_pipeline()
cfg     = EnvConfig()
z_raw   = results["z_test_raw"]

r_eq_all   = (np.exp(z_raw["mom_msci_1m"].values) - 1.0).clip(-0.30, 0.30)
r_bond_all = (-cfg.duration * z_raw["d_swap_10y"].values / 100.0).clip(-0.05, 0.05)
vstoxx_all = z_raw["vstoxx_level"].values

idx_start = cfg.lookback - 1      # 11
r_eq_t    = r_eq_all  [idx_start : idx_start + T]
r_bond_t  = r_bond_all[idx_start : idx_start + T]
vstoxx_t  = vstoxx_all[idx_start : idx_start + T]

# ── Reconstruct w_eq_agg for DRL ─────────────────────────────────────────────
spread = r_eq_t - r_bond_t
with np.errstate(invalid="ignore", divide="ignore"):
    e_t_drl = np.where(
        np.abs(spread) > 0.003,
        (drl["r_p"] - 0.55 * r_eq_t - 0.45 * r_bond_t) / spread,
        np.nan,
    )
w_eq_drl = np.clip(0.55 + e_t_drl, 0.30, 0.80)
w_eq_drl = pd.Series(w_eq_drl, index=dates).ffill().bfill().values

# Fixed-Rule: constant 0.55 aggregate
w_eq_fxd = np.full(T, 0.55)

# ── Regime definitions ────────────────────────────────────────────────────────
regimes = [
    ("Low",    r"$<$20",     vstoxx_t <  20,            0.65),
    ("Medium", "20--30",     (vstoxx_t >= 20) & (vstoxx_t < 30), 0.55),
    ("High",   r"$\geq$30",  vstoxx_t >= 30,            0.35),
]

# ── Print plain text table ────────────────────────────────────────────────────
print("=" * 65)
print("Table 4.5.1 — Equity Tilt by VSTOXX Regime  (run_042, T=85 months)")
print("=" * 65)
print(f"{'Regime':<18} {'VSTOXX':>8} {'n':>5} {'DRL w_eq':>10} {'Fixed w_eq':>12} {'Target b_bar':>12}")
print("-" * 65)
rows = []
for label, band, mask, beta_bar in regimes:
    n        = int(mask.sum())
    drl_mean = float(w_eq_drl[mask].mean())
    fxd_mean = float(w_eq_fxd[mask].mean())
    rows.append((label, band, n, drl_mean, fxd_mean, beta_bar))
    print(f"{label:<18} {band:>8} {n:>5} {drl_mean:>10.4f} {fxd_mean:>12.4f} {beta_bar:>10.2f}")

print("-" * 65)
n_total = sum(r[2] for r in rows)
drl_overall = float(w_eq_drl.mean())
print(f"{'Overall':<18} {'all':>8} {n_total:>5} {drl_overall:>10.4f} {'0.5500':>12} {'--':>10}")
print("=" * 65)

# Additional: distribution of VSTOXX observations across regimes
print("\nVSTOXX regime breakdown:")
for label, band, mask, _ in regimes:
    print(f"  {label:<8}: {mask.sum():>3} months  ({mask.mean()*100:.1f}%)")

# ── LaTeX table ───────────────────────────────────────────────────────────────
print()
print("% -- LaTeX --")
print(r"\begin{table}[htbp]")
print(r"  \centering")
print(r"  \caption{Equity Tilt by VSTOXX Volatility Regime (out-of-sample, 2018--2025).}")
print(r"  \label{tab:regime_equity}")
print(r"  \begin{tabular}{lrrrr}")
print(r"    \toprule")
print(r"    Regime & $n$ & DRL $\bar{w}^{eq}$ & Fixed-Rule $\bar{w}^{eq}$ & Target $\bar{\beta}$ \\")
print(r"    \midrule")
for label, band, n, drl_mean, fxd_mean, beta_bar in rows:
    latex_band = band
    print(f"    {label} (VSTOXX {latex_band}) & {n} & {drl_mean:.4f} & {fxd_mean:.4f} & {beta_bar:.2f} \\\\")
print(r"    \midrule")
print(f"    Overall & {n_total} & {drl_overall:.4f} & 0.5500 & -- \\\\")
print(r"    \bottomrule")
print(r"  \end{tabular}")
print(r"  \begin{tablenotes}\footnotesize")
print(r"  \item \textit{Notes:} DRL aggregate equity weight $\bar{w}^{eq}$ reconstructed from")
print(r"  portfolio returns: $e_t = (r_{p,t} - 0.55\,r^{eq}_t - 0.45\,r^{bond}_t) / (r^{eq}_t - r^{bond}_t)$,")
print(r"  $w^{eq}_{agg,t} = \mathrm{clip}(0.55 + e_t,\,0.30,\,0.80)$.")
print(r"  Fixed-Rule is constant 0.55 (lifecycle aggregate).")
print(r"  Target $\bar{\beta}$ is the GMM risk budget per regime.")
print(r"  \end{tablenotes}")
print(r"\end{table}")
