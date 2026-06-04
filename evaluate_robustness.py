"""evaluate_robustness.py — Robustness evaluation with improved baselines + multi-path.

Policies evaluated
------------------
  A: DRL PPO (run_042)        — historical path only (Option C)
  B: Fixed-Rule ALM           — historical path + 1,000 VAR(1) paths
  C: Constrained MC ALM       — historical path + 1,000 VAR(1) paths
  D: Regime Heuristic         — historical path + 1,000 VAR(1) paths

Design notes
------------
- DRL agent runs on the actual 2018-2025 test path only.  Running it on
  synthetic VAR paths would require reconstructing 31 features from 4 VAR
  variables — a lossy approximation that would bias the comparison against the
  DRL agent.  The historical path gives the cleanest like-for-like comparison.
- The extended VAR adds vstoxx_level as a 4th variable so the Regime Heuristic
  can observe VSTOXX on synthetic paths.
- Constrained MC objective: argmax E[total dist] subject to P(ever deplete) <= 10%.
  The full feasible set is saved to the calibration JSON.
- VecNormalize is reward-only (norm_obs=False) — no observation normalization
  needed at inference time.

Outputs
-------
  results/robustness_multipath.csv      — mean/med/p5/p95 over 1,000 VAR paths
  results/robustness_historical.csv     — all 4 policies on actual 2018-2025 path
  results/baseline_C_calibration.json  — f*, d*, feasible set, calibration diagnostics
  results/robustness_summary.md        — comparison table + flagged findings

Usage
-----
    py -3 evaluate_robustness.py
    py -3 evaluate_robustness.py --model-path src/models/run_043/best_model.zip
    py -3 evaluate_robustness.py --n-paths 200   # smoke test
    py -3 evaluate_robustness.py --no-drl        # skip DRL agent (baselines only)
"""
from __future__ import annotations

import argparse
import io as _io
import json
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.data_pipeline import run_pipeline
from src.environment   import make_env_from_pipeline, EnvConfig, WtpPensionEnv
from src.agent         import AgentConfig, WtpActorCriticPolicy
from src.baselines     import (
    FixedRuleALM, FixedRuleConfig, run_episode,
    _fit_var, _bootstrap_scenarios, _vectorised_simulation,
    MonteCarloConfig,
)
from src.metrics import compute_metrics

try:
    import torch
    from stable_baselines3 import PPO
except ImportError as exc:
    raise ImportError("torch and stable-baselines3 required.\n"
                      "pip install torch stable-baselines3") from exc


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

RESULTS_DIR           = _ROOT / "results"
MODEL_PATH_DEFAULT    = "src/models/run_042/best_model.zip"
N_SCENARIOS           = 1_000
SIM_HORIZON           = 85      # dec 2018 – Dec 2025 (85 months)
SEED                  = 42
DEPLETION_BUDGET      = 0.10    # Constrained MC: max 10 % depletion probability

# Regime Heuristic action parameters (aggregate equity weights map to tilts)
_RH_TILT_LOW  =  0.10   # w_eq_agg = 0.65
_RH_TILT_MED  =  0.00   # w_eq_agg = 0.55
_RH_TILT_HIGH = -0.20   # w_eq_agg = 0.35
_RH_FILL_LOW  = 0.05
_RH_FILL_MED  = 0.02
_RH_FILL_HIGH = 0.00
_RH_DIST_MED  = 0.01    # medium regime only, conditional on FR / B health
_RH_FR_GATE   = 1.05
_RH_B_GATE    = 0.02


# ---------------------------------------------------------------------------
# Regime Heuristic policy (for historical-path run_episode interface)
# ---------------------------------------------------------------------------

class RegimeHeuristicPolicy:
    """Hand-coded policy conditioned on current VSTOXX regime.

    VSTOXX thresholds are the 33rd and 67th percentiles of the TRAINING-PERIOD
    (Jan 1999 - Dec 2015) vstoxx_level distribution.  Thresholds are supplied in
    SCALED units (matching the StandardScaler-transformed observation), because the
    observation vector contains scaled features.

    Feature layout in the 377-dim observation:
      [FR(0), B(1), o_plus(2), fill_used(3), month_norm(4), z_flat(5:377)]
      z_flat = 12 months × 31 features, most-recent month last.
      vstoxx_level is feature index 14 within the 31-dim feature vector.
      Most-recent vstoxx_level: obs[5 + 11*31 + 14] = obs[360].

    Args:
        p33_scaled: Training-period 33rd percentile of scaled vstoxx_level.
        p67_scaled: Training-period 67th percentile of scaled vstoxx_level.
    """

    # Feature index of vstoxx_level within the 31-dim feature vector
    _VSTOXX_FEAT_IDX = 14
    _N_FEATURES      = 31
    _LOOKBACK        = 12
    _N_SOLVENCY      = 5   # FR, B, o_plus, fill_used, month_norm

    def __init__(self, p33_scaled: float, p67_scaled: float) -> None:
        self.p33 = p33_scaled
        self.p67 = p67_scaled

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Map 377-dim observation to action [equity_tilt, fill_rate, dist_rate]."""
        FR = float(obs[0])
        B  = float(obs[1])

        # Most-recent month's vstoxx_level (scaled)
        last_month_start = (self._N_SOLVENCY
                            + (self._LOOKBACK - 1) * self._N_FEATURES)
        vstoxx_s = float(obs[last_month_start + self._VSTOXX_FEAT_IDX])

        if vstoxx_s < self.p33:
            return np.array([_RH_TILT_LOW, _RH_FILL_LOW, 0.0], dtype=np.float32)
        elif vstoxx_s < self.p67:
            dist = _RH_DIST_MED if (FR >= _RH_FR_GATE and B >= _RH_B_GATE) else 0.0
            return np.array([_RH_TILT_MED, _RH_FILL_MED, dist], dtype=np.float32)
        else:
            return np.array([_RH_TILT_HIGH, _RH_FILL_HIGH, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# SB3 adapter
# ---------------------------------------------------------------------------

class _SB3Adapter:
    def __init__(self, model) -> None:
        self._model = model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action


# ---------------------------------------------------------------------------
# 1.  Extended VAR fitting (4 variables: r_eq, d_swap_10y, pi, vstoxx_level)
# ---------------------------------------------------------------------------

def fit_extended_var(
    raw_train:   pd.DataFrame,
    cpi:         pd.DataFrame,
    z_train_raw: pd.DataFrame,
    env_cfg:     EnvConfig,
    max_lag:     int = 6,
) -> dict:
    """Fit a 4-variable VAR on training data and an OLS for d_swap_20y.

    The 4th variable (vstoxx_level) is taken from the unscaled training
    features so that the Regime Heuristic can observe VSTOXX on simulated paths.

    Args:
        raw_train:   Raw monthly training DataFrame from the pipeline.
        cpi:         CPI DataFrame from the pipeline.
        z_train_raw: Unscaled training feature DataFrame (Jan 2000 - Dec 2015).
        env_cfg:     EnvConfig (for duration, clip bounds).
        max_lag:     Maximum VAR lag for BIC selection.

    Returns:
        dict with keys: var_result, lag_order, swap20_beta, swap20_intercept,
        r_eq_clip, r_bond_clip, duration.
    """
    cfg_mc = MonteCarloConfig()

    r_eq_s      = raw_train["Equity_World_MSCI"].pct_change().values
    r_eq_s      = np.clip(r_eq_s, cfg_mc.r_eq_clip[0], cfg_mc.r_eq_clip[1])
    d_swap_10y_s = raw_train["Swap_10Y"].diff().values
    col_20y     = "RTS_20Y" if "RTS_20Y" in raw_train.columns else "Swap_20Y"
    d_swap_20y_s = raw_train[col_20y].diff().values
    pi_s        = (cpi["pi_monthly"].reindex(raw_train.index).fillna(0.0).values)
    vstoxx_s    = z_train_raw["vstoxx_level"].values

    var_df = pd.DataFrame({
        "r_eq":       r_eq_s,
        "d_swap_10y": d_swap_10y_s,
        "pi_monthly": pi_s,
        "vstoxx":     vstoxx_s,
    }, index=raw_train.index).dropna()

    print(f"  Fitting 4-variable VAR (max_lag={max_lag}) "
          f"on {len(var_df)} training months...")
    var_result, lag_order = _fit_var(var_df, max_lag)
    print(f"  VAR lag selected by BIC: P={lag_order}")

    # OLS: d_swap_20y ~ d_swap_10y (for liability return on simulated paths)
    mask = np.isfinite(d_swap_10y_s) & np.isfinite(d_swap_20y_s)
    beta, intercept = np.polyfit(d_swap_10y_s[mask], d_swap_20y_s[mask], 1)
    print(f"  OLS d_swap_20y ~ d_swap_10y: beta={beta:.4f}, "
          f"intercept={intercept:.6f}")

    return {
        "var_result":       var_result,
        "lag_order":        lag_order,
        "swap20_beta":      float(beta),
        "swap20_intercept": float(intercept),
        "r_eq_clip":        cfg_mc.r_eq_clip,
        "r_bond_clip":      cfg_mc.r_bond_clip,
        "duration":         env_cfg.duration,
    }


# ---------------------------------------------------------------------------
# 2.  Scenario generation
# ---------------------------------------------------------------------------

def generate_var_scenarios(
    var_fit:     dict,
    n_scenarios: int = N_SCENARIOS,
    sim_horizon: int = SIM_HORIZON,
    seed:        int = SEED,
) -> dict:
    """Bootstrap n_scenarios paths from the fitted 4-variable VAR.

    Returns:
        dict with keys r_eq, r_bond, r_L_MtM, vstoxx — each (T, N).
    """
    scenarios = _bootstrap_scenarios(
        var_fit["var_result"],
        n_scenarios=n_scenarios,
        sim_horizon=sim_horizon,
        seed=seed,
    )
    # scenarios: (T, N, 4) — cols: [r_eq, d_swap_10y, pi, vstoxx]

    cfg_mc = MonteCarloConfig()
    dur    = var_fit["duration"]

    r_eq_scen    = np.clip(scenarios[:, :, 0],
                           cfg_mc.r_eq_clip[0], cfg_mc.r_eq_clip[1])
    d_10y_scen   = scenarios[:, :, 1]
    vstoxx_scen  = np.maximum(scenarios[:, :, 3], 0.0)   # VSTOXX >= 0

    r_bond_scen  = np.clip(-dur * d_10y_scen / 100.0,
                           cfg_mc.r_bond_clip[0], cfg_mc.r_bond_clip[1])

    d_20y_scen   = (var_fit["swap20_intercept"]
                    + var_fit["swap20_beta"] * d_10y_scen)
    r_L_MtM_scen = -dur * d_20y_scen / 100.0

    return {
        "r_eq":    r_eq_scen,     # (T, N)
        "r_bond":  r_bond_scen,   # (T, N)
        "r_L_MtM": r_L_MtM_scen, # (T, N)
        "vstoxx":  vstoxx_scen,   # (T, N)
    }


# ---------------------------------------------------------------------------
# 3.  Constrained MC calibration
# ---------------------------------------------------------------------------

def calibrate_constrained_mc(
    scenarios:        dict,
    env_cfg:          EnvConfig,
    depletion_budget: float      = DEPLETION_BUDGET,
    f_grid:           list       = None,
    d_grid:           list       = None,
    fr_init:          float      = 1.05,
    b_init:           float      = 0.05,
) -> dict:
    """Grid-search (f, d) maximising E[total dist] s.t. depletion_prob <= budget.

    Reports the full feasible set (all candidates with depletion <= budget),
    sorted by mean total distributions descending.

    Args:
        scenarios:        Output of :func:`generate_var_scenarios`.
        env_cfg:          EnvConfig (for Art. 10d parameters).
        depletion_budget: Maximum allowed empirical depletion probability.
        f_grid, d_grid:   Optional override for the search grid.
        fr_init, b_init:  Initial fund state for calibration paths.

    Returns:
        dict with f_star, d_star, feasible_set, all_results, n_feasible,
        depletion_budget, constraint_binding.
    """
    if f_grid is None:
        f_grid = [round(v, 2) for v in np.arange(0.00, 0.11, 0.01).tolist()]
    if d_grid is None:
        d_grid = [round(v, 2) for v in np.arange(0.00, 0.06, 0.01).tolist()]

    # Build MonteCarloConfig mirroring env_cfg
    cfg_mc = MonteCarloConfig()
    cfg_mc.fr_init              = fr_init
    cfg_mc.b_init               = b_init
    cfg_mc.duration             = env_cfg.duration
    cfg_mc.liability_mtm_weight = env_cfg.liability_mtm_weight
    cfg_mc.r_ufr                = env_cfg.r_ufr
    cfg_mc.b_max                = env_cfg.b_max
    cfg_mc.fr_dist_threshold    = env_cfg.fr_dist_threshold
    cfg_mc.depletion_threshold  = 0.001

    r_eq    = scenarios["r_eq"]
    r_bond  = scenarios["r_bond"]
    r_L_MtM = scenarios["r_L_MtM"]

    print(f"  Constrained MC grid search: "
          f"{len(f_grid)} x {len(d_grid)} = {len(f_grid)*len(d_grid)} combos "
          f"(depletion budget <= {depletion_budget:.0%})...")

    all_results = []
    for f in f_grid:
        for d in d_grid:
            res = _vectorised_simulation(r_eq, r_bond, r_L_MtM, f, d, cfg_mc)
            all_results.append({
                "f": f,
                "d": d,
                "depletion_prob":   res["depletion_prob"],
                "mean_total_dist":  res["mean_total_dist"],
                "fr_terminal_mean": float(res["FR_terminal"].mean()),
                "fr_terminal_p5":   float(np.percentile(res["FR_terminal"], 5)),
            })

    feasible = [r for r in all_results if r["depletion_prob"] <= depletion_budget]
    feasible_sorted = sorted(feasible, key=lambda x: x["mean_total_dist"], reverse=True)

    if feasible_sorted:
        best = feasible_sorted[0]
        f_star, d_star = best["f"], best["d"]
        constraint_binding = best["depletion_prob"] > depletion_budget * 0.5
        print(f"  Constrained MC: f*={f_star:.2f}, d*={d_star:.2f}  "
              f"(depletion={best['depletion_prob']:.3f}, "
              f"mean_dist={best['mean_total_dist']:.4f})")
        print(f"  Feasible set: {len(feasible_sorted)} of {len(all_results)} "
              f"candidates pass depletion <= {depletion_budget:.0%}")
    else:
        # No feasible point: fall back to minimum-depletion policy
        best = min(all_results, key=lambda x: x["depletion_prob"])
        f_star, d_star = best["f"], best["d"]
        constraint_binding = True
        print(f"  WARNING: no feasible point (all depletion > {depletion_budget:.0%}). "
              f"Fallback to min-depletion: f*={f_star:.2f}, d*={d_star:.2f}")

    return {
        "f_star":            f_star,
        "d_star":            d_star,
        "best_result":       best,
        "feasible_set":      feasible_sorted,
        "all_results":       all_results,
        "n_feasible":        len(feasible_sorted),
        "n_total":           len(all_results),
        "depletion_budget":  depletion_budget,
        "constraint_binding": constraint_binding,
    }


# ---------------------------------------------------------------------------
# 4.  Regime Heuristic thresholds (training-period only)
# ---------------------------------------------------------------------------

def derive_heuristic_thresholds(
    z_train_raw: pd.DataFrame,
    z_train:     pd.DataFrame,
) -> dict:
    """Compute VSTOXX 33rd and 67th percentiles from training-period data.

    Returns raw-unit thresholds (for multi-path vectorised simulation) and
    scaled-unit thresholds (for historical-path observation comparison).
    BOTH are derived from Jan 2000 - Dec 2015 training data only.

    Args:
        z_train_raw: Unscaled training features.
        z_train:     Scaled training features (same rows, same scaler).
    """
    raw_vals    = z_train_raw["vstoxx_level"].dropna().values
    scaled_vals = z_train["vstoxx_level"].dropna().values

    p33_raw    = float(np.percentile(raw_vals, 33))
    p67_raw    = float(np.percentile(raw_vals, 67))
    p33_scaled = float(np.percentile(scaled_vals, 33))
    p67_scaled = float(np.percentile(scaled_vals, 67))

    print(f"  Regime Heuristic thresholds (training 2000-2015):")
    print(f"    Raw   VSTOXX: p33={p33_raw:.2f}  p67={p67_raw:.2f}  "
          f"mean={raw_vals.mean():.2f}  "
          f"[{raw_vals.min():.1f}, {raw_vals.max():.1f}]")
    print(f"    Scaled VSTOXX: p33={p33_scaled:.3f}  p67={p67_scaled:.3f}")

    return {
        "p33_raw": p33_raw,
        "p67_raw": p67_raw,
        "p33_scaled": p33_scaled,
        "p67_scaled": p67_scaled,
        "n_train_obs": len(raw_vals),
        "raw_mean": float(raw_vals.mean()),
        "raw_min":  float(raw_vals.min()),
        "raw_max":  float(raw_vals.max()),
    }


# ---------------------------------------------------------------------------
# 5.  Vectorised multi-path simulation (Fixed-Rule, Constrained MC, Heuristic)
# ---------------------------------------------------------------------------

def _run_vectorised_policy(
    r_eq:        np.ndarray,   # (T, N)
    r_bond:      np.ndarray,   # (T, N)
    r_L_MtM:     np.ndarray,   # (T, N)
    vstoxx:      np.ndarray,   # (T, N)
    policy_name: str,          # "fixed_rule" | "constrained_mc" | "regime_heuristic"
    f_fixed:     float,        # used for fixed_rule / constrained_mc
    d_fixed:     float,
    e_fixed:     float,        # equity tilt for fixed_rule / constrained_mc
    p33_raw:     float,        # regime heuristic VSTOXX threshold (raw units)
    p67_raw:     float,
    env_cfg:     EnvConfig,
    fr_init:     float = 1.05,
    b_init:      float = 0.05,
) -> dict:
    """Simulate one policy over all N paths simultaneously.

    Returns:
        dict with (T, N) arrays: FR, B, d_tilde, f_tilde, r_p,
        r_p_young, r_p_mid, r_p_ret.
    """
    T, N   = r_eq.shape
    cfg    = env_cfg
    is_rh  = (policy_name == "regime_heuristic")
    use_lc = cfg.use_lifecycle

    FR               = np.full(N, fr_init, dtype=np.float64)
    B                = np.full(N, b_init,  dtype=np.float64)
    annual_o_plus    = np.zeros(N, dtype=np.float64)
    annual_fill_used = np.zeros(N, dtype=np.float64)

    FR_arr   = np.empty((T, N), dtype=np.float64)
    B_arr    = np.empty((T, N), dtype=np.float64)
    d_arr    = np.empty((T, N), dtype=np.float64)
    f_arr    = np.empty((T, N), dtype=np.float64)
    rp_arr   = np.empty((T, N), dtype=np.float64)
    rpy_arr  = np.empty((T, N), dtype=np.float64)
    rpm_arr  = np.empty((T, N), dtype=np.float64)
    rpr_arr  = np.empty((T, N), dtype=np.float64)

    for t in range(T):
        month = (t % 12) + 1    # 1 = Jan, 12 = Dec (path starts in January)

        # --- Policy action ---------------------------------------------------
        if is_rh:
            vs  = vstoxx[t]     # (N,)
            e_t = np.where(vs < p33_raw, _RH_TILT_LOW,
                           np.where(vs < p67_raw, _RH_TILT_MED, _RH_TILT_HIGH))
            f_t = np.where(vs < p33_raw, _RH_FILL_LOW,
                           np.where(vs < p67_raw, _RH_FILL_MED, _RH_FILL_HIGH))
            # Distribution: medium regime, conditional on current FR and B
            d_t = np.where(
                (vs >= p33_raw) & (vs < p67_raw)
                & (FR >= _RH_FR_GATE) & (B >= _RH_B_GATE),
                _RH_DIST_MED, 0.0,
            )
        else:
            e_t = np.full(N, e_fixed)
            f_t = np.full(N, f_fixed)
            d_t = np.full(N, d_fixed)

        # --- Portfolio return ------------------------------------------------
        req  = r_eq[t]    # (N,)
        rbnd = r_bond[t]

        if use_lc:
            wy = np.clip(cfg.w_eq_young_base + e_t,
                         cfg.w_eq_young_min, cfg.w_eq_young_max)
            wm = np.clip(cfg.w_eq_mid_base   + e_t,
                         cfg.w_eq_mid_min,   cfg.w_eq_mid_max)
            wr = np.clip(cfg.w_eq_ret_base   + e_t,
                         cfg.w_eq_ret_min,   cfg.w_eq_ret_max)
            rpy = wy * req + (1.0 - wy) * rbnd
            rpm = wm * req + (1.0 - wm) * rbnd
            rpr = wr * req + (1.0 - wr) * rbnd
            r_p = cfg.w_young * rpy + cfg.w_mid * rpm + cfg.w_ret * rpr
        else:
            w_eq = np.clip(cfg.w_eq_base + e_t, cfg.w_eq_min, cfg.w_eq_max)
            r_p  = w_eq * req + (1.0 - w_eq) * rbnd
            rpy  = rpm = rpr = r_p

        # --- Liability return ------------------------------------------------
        r_L = (cfg.liability_mtm_weight * r_L_MtM[t]
               + (1.0 - cfg.liability_mtm_weight) * cfg.r_ufr)

        # --- Art. 10d lid 2: annual fill cap ---------------------------------
        over          = r_p - r_L
        annual_o_plus = annual_o_plus + np.maximum(0.0, over)
        budget        = cfg.annual_fill_cap_frac * annual_o_plus
        remaining     = np.maximum(0.0, budget - annual_fill_used)
        f_tilde       = np.minimum(f_t, remaining)
        annual_fill_used = annual_fill_used + f_tilde

        # --- FR and B update -------------------------------------------------
        growth = (1.0 + r_p) / np.maximum(1.0 + r_L, 1e-8)
        FR_new = FR * growth - f_tilde
        B_new  = B  * growth + f_tilde

        # --- Art. 10d lid 4: distribution gate --------------------------------
        can_dist = (FR_new >= cfg.fr_dist_threshold) & (B_new > 0.0)
        d_tilde  = np.where(can_dist, np.minimum(d_t, B_new), 0.0)
        B_new   -= d_tilde

        # --- Art. 10d lid 1: negative buffer absorbed by FR ------------------
        shortfall = B_new < 0.0
        FR_new    = np.where(shortfall, FR_new + B_new, FR_new)
        B_new     = np.maximum(B_new, 0.0)

        # --- Art. 10d lid 1: December cap + annual reset ---------------------
        if month == 12:
            B_new            = np.minimum(B_new, cfg.b_max)
            annual_o_plus    = np.zeros(N, dtype=np.float64)
            annual_fill_used = np.zeros(N, dtype=np.float64)

        FR = FR_new
        B  = B_new

        FR_arr[t]  = FR
        B_arr[t]   = B
        d_arr[t]   = d_tilde
        f_arr[t]   = f_tilde
        rp_arr[t]  = r_p
        rpy_arr[t] = rpy
        rpm_arr[t] = rpm
        rpr_arr[t] = rpr

    return {
        "FR":        FR_arr,
        "B":         B_arr,
        "d_tilde":   d_arr,
        "f_tilde":   f_arr,
        "r_p":       rp_arr,
        "r_p_young": rpy_arr,
        "r_p_mid":   rpm_arr,
        "r_p_ret":   rpr_arr,
    }


# ---------------------------------------------------------------------------
# 6.  Per-path metric computation (batch, on (T, N) arrays)
# ---------------------------------------------------------------------------

def compute_batch_metrics(
    traj:     dict,
    pi:       Optional[np.ndarray] = None,   # (T,) — same CPI broadcast over paths
    lookback: int = 12,
    mvev_floor: float = 1.043,
) -> dict:
    """Compute evaluation metrics for each of N paths.

    Args:
        traj:     Dict of (T, N) arrays from :func:`_run_vectorised_policy`.
        pi:       Monthly CPI inflation ``(T,)``.  Zeros if None.
        lookback: Rolling window for replacement-rate calculation (months).
        mvev_floor: FR threshold for MVEV breach detection.

    Returns:
        Dict mapping metric name → (N,) array.  Also includes scalar
        ``mvev_breach_count`` and ``mvev_breach_rate``.
    """
    FR   = traj["FR"]    # (T, N)
    B    = traj["B"]
    d    = traj["d_tilde"]
    f    = traj["f_tilde"]
    r_p  = traj["r_p"]
    rpy  = traj.get("r_p_young", r_p)
    rpm  = traj.get("r_p_mid",   r_p)
    rpr  = traj.get("r_p_ret",   r_p)
    T, N = FR.shape

    pi_arr = (np.asarray(pi, dtype=np.float64) if pi is not None
              else np.zeros(T, dtype=np.float64))[:T]

    # Terminal FR
    terminal_FR = FR[-1].copy()

    # FR MDD
    running_max = np.maximum.accumulate(FR, axis=0)
    drawdowns   = (running_max - FR) / np.maximum(running_max, 1e-8)
    fr_mdd      = drawdowns.max(axis=0)

    # FR annualised volatility
    fr_changes = np.diff(FR, axis=0)                        # (T-1, N)
    fr_vol_ann = fr_changes.std(axis=0) * np.sqrt(12)

    # Buffer depletion frequency (fraction of months with B <= 0.001)
    buf_dep_freq = (B <= 0.001).mean(axis=0)

    # Total distributions
    total_dist = d.sum(axis=0)

    # Calmar ratio
    fr_monthly_growth = fr_changes / np.maximum(FR[:-1], 1e-8)
    fr_ann_growth     = fr_monthly_growth.mean(axis=0) * 12
    calmar            = fr_ann_growth / (fr_mdd + 1e-8)

    # Cohort real replacement-rate variance (rolling 12M)
    # PPV formula: R_i = r_p_i − f_tilde + d_tilde (no dec_excess in vectorised sim)
    r_young = rpy - f + d
    r_mid   = rpm - f + d
    r_ret   = rpr - f + d

    rr_var_paths = np.zeros(N, dtype=np.float64)
    if T > lookback:
        rr_var_accum = np.zeros(N, dtype=np.float64)
        n_windows    = 0
        for t in range(lookback, T):
            pi_w = pi_arr[t - lookback:t]          # (lookback,)
            rrs  = []
            for c_ret in (r_young, r_mid, r_ret):
                w      = c_ret[t - lookback:t]      # (lookback, N)
                log_rr = np.log1p(w - pi_w[:, None]).sum(axis=0)   # (N,)
                rrs.append(log_rr)
            rr_stack      = np.stack(rrs, axis=0)  # (3, N)
            rr_var_accum += rr_stack.var(axis=0)
            n_windows    += 1
        if n_windows > 0:
            rr_var_paths = rr_var_accum / n_windows

    # MVEV breach: any step where FR < mvev_floor
    mvev_breach = (FR < mvev_floor).any(axis=0)     # (N,) bool

    return {
        "terminal_FR":           terminal_FR,
        "fr_mdd":                fr_mdd,
        "fr_vol_ann":            fr_vol_ann,
        "buffer_depletion_freq": buf_dep_freq,
        "total_dist":            total_dist,
        "calmar":                calmar,
        "cohort_rr_var":         rr_var_paths,
        "mvev_breach":           mvev_breach,
    }


def summarise_batch_metrics(metrics: dict) -> dict:
    """Reduce (N,) metric arrays to mean / median / p5 / p95."""
    out = {}
    for key, arr in metrics.items():
        if key == "mvev_breach":
            out["mvev_breach_count"] = int(np.asarray(arr).sum())
            out["mvev_breach_rate"]  = float(np.asarray(arr).mean())
            continue
        a = np.asarray(arr, dtype=np.float64)
        out[f"{key}_mean"]   = float(a.mean())
        out[f"{key}_median"] = float(np.median(a))
        out[f"{key}_p5"]     = float(np.percentile(a,  5))
        out[f"{key}_p95"]    = float(np.percentile(a, 95))
    return out


# ---------------------------------------------------------------------------
# 7.  Load DRL agent
# ---------------------------------------------------------------------------

def load_drl_agent(model_path: Path, env_cfg: EnvConfig, results: dict):
    """Load the run_043 PPO agent.  Mirrors evaluate.py exactly."""
    _n_regimes = 3
    try:
        with zipfile.ZipFile(str(model_path)) as zf:
            with zf.open("policy.pth") as fz:
                state = torch.load(_io.BytesIO(fz.read()),
                                   map_location="cpu", weights_only=False)
        bias = state.get("wtp_net.gating.linear.bias")
        if bias is not None:
            _n_regimes = int(bias.shape[0])
    except Exception:
        pass

    eval_cfg = AgentConfig(
        n_regimes=_n_regimes, gmm_n_regimes=_n_regimes,
        beta_bar=([0.70, 0.55, 0.40, 0.25] if _n_regimes == 4
                  else [0.65, 0.55, 0.35]),
    )
    env = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)
    model = PPO.load(
        str(model_path), env=env,
        custom_objects={
            "policy_class":  WtpActorCriticPolicy,
            "policy_kwargs": {"wtp_cfg": eval_cfg},
        },
    )
    print(f"  DRL agent loaded (n_regimes={_n_regimes}): {model_path}")
    return _SB3Adapter(model), env


# ---------------------------------------------------------------------------
# 8.  Historical-path evaluation (all 4 policies)
# ---------------------------------------------------------------------------

def run_all_historical(
    results:    dict,
    env_cfg:    EnvConfig,
    model_path: Path,
    f_c:        float,
    d_c:        float,
    heuristic_thresholds: dict,
    run_drl:    bool = True,
    seed:       int  = 0,
) -> tuple[dict, dict]:
    """Run all four policies on the actual 2018-2025 test path.

    Returns:
        (metrics_dict, trajectories_dict) — keyed by policy name.
    """
    pi_test = (results["cpi"]["pi_monthly"]
               .reindex(results["z_test"].index)
               .fillna(0.0).values)

    def make_test():
        return make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=seed)

    agents: dict = {}

    if run_drl:
        drl_adapter, drl_env = load_drl_agent(model_path, env_cfg, results)
        agents["DRL (PPO)"] = (drl_adapter, drl_env)

    agents["Fixed-Rule"] = (
        FixedRuleALM(FixedRuleConfig(equity_tilt=0.0, fill_rate=0.03, dist_rate=0.02)),
        make_test(),
    )
    agents["Constrained MC"] = (
        FixedRuleALM(FixedRuleConfig(equity_tilt=0.0, fill_rate=f_c, dist_rate=d_c)),
        make_test(),
    )
    rh_policy = RegimeHeuristicPolicy(
        p33_scaled=heuristic_thresholds["p33_scaled"],
        p67_scaled=heuristic_thresholds["p67_scaled"],
    )
    agents["Regime Heuristic"] = (rh_policy, make_test())

    trajs: dict   = {}
    metrics: dict = {}

    for name, (agent, env) in agents.items():
        print(f"  [{name}] running test episode...")
        traj = run_episode(agent, env)
        trajs[name] = traj

        m = compute_metrics(traj, pi_monthly=pi_test)
        FR = np.array(traj["FR"])
        m["mvev_breach_count"] = int((FR < 1.043).sum())
        m["mvev_breach_any"]   = bool((FR < 1.043).any())
        metrics[name] = m

    return metrics, trajs


# ---------------------------------------------------------------------------
# 9.  Output helpers
# ---------------------------------------------------------------------------

def _multipath_to_df(summary_by_policy: dict) -> pd.DataFrame:
    """Convert {policy: summary_dict} to a tidy DataFrame."""
    rows = []
    metrics_order = [
        "terminal_FR", "fr_mdd", "fr_vol_ann",
        "buffer_depletion_freq", "total_dist", "calmar", "cohort_rr_var",
    ]
    stats_order = ["mean", "median", "p5", "p95"]

    for policy, summary in summary_by_policy.items():
        for metric in metrics_order:
            for stat in stats_order:
                key = f"{metric}_{stat}"
                rows.append({
                    "policy":  policy,
                    "metric":  metric,
                    "stat":    stat,
                    "value":   summary.get(key, float("nan")),
                })
        # MVEV breach
        rows.append({
            "policy": policy,
            "metric": "mvev_breach",
            "stat":   "count",
            "value":  summary.get("mvev_breach_count", float("nan")),
        })
        rows.append({
            "policy": policy,
            "metric": "mvev_breach",
            "stat":   "rate",
            "value":  summary.get("mvev_breach_rate", float("nan")),
        })
    return pd.DataFrame(rows)


def _historical_to_df(metrics_by_policy: dict) -> pd.DataFrame:
    """Convert historical metrics dict to a tidy DataFrame."""
    rows = []
    for policy, m in metrics_by_policy.items():
        for key, val in m.items():
            rows.append({"policy": policy, "metric": key,
                         "value": float(val) if isinstance(val, (int, float, np.floating)) else val})
    return pd.DataFrame(rows)


def _build_summary_md(
    summary_by_policy: dict,
    hist_metrics:      dict,
    calib:             dict,
    heuristic_thresholds: dict,
) -> str:
    """Generate robustness_summary.md."""

    lines = [
        "# Wtp DRL Pension Fund — Robustness Evaluation Summary",
        "",
        "## Setup",
        "",
        f"- **Multi-path**: {N_SCENARIOS:,} VAR(1)-simulated paths, "
        f"{SIM_HORIZON} months each, FR₀ = 1.05, B₀ = 0.05",
        "- **DRL agent**: historical path only (Jan 2018 – Dec 2025); "
        "synthetic-path evaluation omitted to avoid feature-reconstruction bias",
        f"- **Constrained MC objective**: argmax E[Σ d̃_t] "
        f"s.t. P(ever deplete) ≤ {DEPLETION_BUDGET:.0%}",
        f"- **Regime Heuristic thresholds** (training 2000–2015 only): "
        f"VSTOXX p33 = {heuristic_thresholds['p33_raw']:.2f}, "
        f"p67 = {heuristic_thresholds['p67_raw']:.2f}",
        "",
    ]

    # --- Constrained MC calibration summary ---------------------------------
    lines += [
        "## Baseline C: Constrained MC Calibration",
        "",
        f"- f* = {calib['f_star']:.2f}, d* = {calib['d_star']:.2f}",
        f"- Feasible set: {calib['n_feasible']} of {calib['n_total']} "
        f"(f, d) combinations satisfy depletion ≤ {DEPLETION_BUDGET:.0%}",
    ]
    if calib["feasible_set"]:
        lines += [
            "",
            "Top 10 feasible candidates (by mean total distributions):",
            "",
            "| f | d | depletion_prob | mean_total_dist | FR_terminal_mean |",
            "|---|---|---|---|---|",
        ]
        for r in calib["feasible_set"][:10]:
            lines.append(
                f"| {r['f']:.2f} | {r['d']:.2f} | "
                f"{r['depletion_prob']:.3f} | "
                f"{r['mean_total_dist']:.4f} | "
                f"{r['fr_terminal_mean']:.4f} |"
            )
    lines.append("")

    # --- Historical path table ----------------------------------------------
    lines += [
        "## Historical Path Results (Jan 2018 – Dec 2025)",
        "",
        "| Metric | DRL (PPO) | Fixed-Rule | Constrained MC | Regime Heuristic |",
        "|---|---|---|---|---|",
    ]
    metric_labels = [
        ("fr_terminal",           "FR Terminal"),
        ("fr_mdd",                "FR Max Drawdown"),
        ("fr_vol_ann",            "FR Vol (ann)"),
        ("buffer_depletion_freq", "Buf Depl Freq"),
        ("total_distributions",   "Total Dist"),
        ("calmar_ratio",          "Calmar Ratio"),
        ("cohort_rr_var",         "Cohort RR Var"),
        ("mvev_breach_count",     "MVEV Breach Count"),
    ]
    policy_order = ["DRL (PPO)", "Fixed-Rule", "Constrained MC", "Regime Heuristic"]
    for key, label in metric_labels:
        vals = []
        for p in policy_order:
            v = hist_metrics.get(p, {}).get(key, float("nan"))
            if isinstance(v, bool):
                vals.append("Yes" if v else "No")
            elif isinstance(v, (int,)):
                vals.append(str(v))
            else:
                vals.append(f"{float(v):.4f}" if not np.isnan(float(v)) else "—")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")

    # --- Multi-path table (mean and p5) -------------------------------------
    lines += [
        "## Multi-Path Results (1,000 VAR Paths) — Baselines Only",
        "",
        "### Mean across 1,000 paths",
        "",
        "| Metric | Fixed-Rule | Constrained MC | Regime Heuristic |",
        "|---|---|---|---|",
    ]
    mp_policies = ["Fixed-Rule", "Constrained MC", "Regime Heuristic"]
    mp_metric_labels = [
        ("terminal_FR",           "FR Terminal"),
        ("fr_mdd",                "FR Max Drawdown"),
        ("fr_vol_ann",            "FR Vol (ann)"),
        ("buffer_depletion_freq", "Buf Depl Freq"),
        ("total_dist",            "Total Dist"),
        ("calmar",                "Calmar Ratio"),
        ("cohort_rr_var",         "Cohort RR Var"),
    ]
    for key, label in mp_metric_labels:
        vals = []
        for p in mp_policies:
            v = summary_by_policy.get(p, {}).get(f"{key}_mean", float("nan"))
            vals.append(f"{v:.4f}" if not np.isnan(v) else "—")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")

    lines += [
        "### 5th percentile (tail risk) across 1,000 paths",
        "",
        "| Metric | Fixed-Rule | Constrained MC | Regime Heuristic |",
        "|---|---|---|---|",
    ]
    for key, label in mp_metric_labels:
        vals = []
        for p in mp_policies:
            v = summary_by_policy.get(p, {}).get(f"{key}_p5", float("nan"))
            vals.append(f"{v:.4f}" if not np.isnan(v) else "—")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")

    # --- Diagnostic findings ------------------------------------------------
    lines += ["## Diagnostic Findings", ""]

    drl_h  = hist_metrics.get("DRL (PPO)", {})
    fr_h   = hist_metrics.get("Fixed-Rule", {})
    cmc_h  = hist_metrics.get("Constrained MC", {})
    rh_h   = hist_metrics.get("Regime Heuristic", {})

    # Finding 1: Constrained MC gap vs. unconstrained
    cmc_dep  = cmc_h.get("buffer_depletion_freq", float("nan"))
    fr_dep   = fr_h.get("buffer_depletion_freq", float("nan"))
    cmc_dist = cmc_h.get("total_distributions", float("nan"))
    fr_dist  = fr_h.get("total_distributions", float("nan"))
    drl_dist = drl_h.get("total_distributions", float("nan"))
    lines.append(
        f"**1. Constrained MC vs. Fixed-Rule (unconstrained proxy):** "
        f"Constrained MC achieves depletion = {cmc_dep:.3f} vs. "
        f"Fixed-Rule {fr_dep:.3f}; total dist = {cmc_dist:.4f} vs. "
        f"{fr_dist:.4f}.  "
    )
    if not np.isnan(drl_dist) and not np.isnan(cmc_dist):
        dist_gap = drl_dist - cmc_dist
        lines[-1] += (
            f"DRL agent distributes {dist_gap:+.4f} more than Constrained MC on the historical path.  "
        )
        if abs(dist_gap) < 0.02:
            lines[-1] += "**The gap is small — the better-specified baseline substantially erodes DRL outperformance.**"
        else:
            lines[-1] += "The gap remains material even against the better-specified baseline."
    lines.append("")

    # Finding 2: Regime Heuristic competitiveness
    drl_calmar = drl_h.get("calmar_ratio", float("nan"))
    rh_calmar  = rh_h.get("calmar_ratio", float("nan"))
    drl_fr_t   = drl_h.get("fr_terminal", float("nan"))
    rh_fr_t    = rh_h.get("fr_terminal", float("nan"))
    lines.append(
        f"**2. Regime Heuristic vs. DRL agent:** "
        f"Heuristic Calmar = {rh_calmar:.4f} vs. DRL {drl_calmar:.4f}; "
        f"terminal FR = {rh_fr_t:.4f} vs. {drl_fr_t:.4f}.  "
    )
    if not np.isnan(rh_calmar) and not np.isnan(drl_calmar):
        if abs(rh_calmar - drl_calmar) / (abs(drl_calmar) + 1e-8) < 0.15:
            lines[-1] += (
                "**The Regime Heuristic is within 15% of the DRL agent on Calmar ratio. "
                "This weakens the justification for the LSTM+GMM architecture complexity.**"
            )
        else:
            lines[-1] += (
                "The DRL agent materially outperforms the Regime Heuristic, "
                "supporting the value of learned state-dependent policy."
            )
    lines.append("")

    # Finding 3: Mean vs. tail advantage
    # For multipath we only have baselines; compare them
    rh_mean_dep = summary_by_policy.get("Regime Heuristic", {}).get(
        "buffer_depletion_freq_mean", float("nan"))
    cmc_mean_dep = summary_by_policy.get("Constrained MC", {}).get(
        "buffer_depletion_freq_mean", float("nan"))
    rh_p5_fr  = summary_by_policy.get("Regime Heuristic", {}).get(
        "terminal_FR_p5", float("nan"))
    cmc_p5_fr = summary_by_policy.get("Constrained MC", {}).get(
        "terminal_FR_p5", float("nan"))
    lines.append(
        f"**3. Baseline mean vs. tail comparison (multi-path):** "
        f"Constrained MC mean depletion = {cmc_mean_dep:.3f}, "
        f"Regime Heuristic = {rh_mean_dep:.3f}; "
        f"p5 terminal FR: Constrained MC = {cmc_p5_fr:.4f}, "
        f"Regime Heuristic = {rh_p5_fr:.4f}.  "
        "Note: DRL multi-path tail risk cannot be assessed here; "
        "historical path MVEV breach count is reported above."
    )
    lines.append("")

    # Finding 4: DRL MVEV breaches
    mvev_count = drl_h.get("mvev_breach_count", float("nan"))
    lines.append(
        f"**4. DRL agent MVEV floor breaches (historical path):** "
        f"FR < 1.043 in {mvev_count} of 84 test months.  "
    )
    if not np.isnan(mvev_count):
        if mvev_count == 0:
            lines[-1] += (
                "The near-hard constraint holds on the historical path. "
                "Multi-path evidence is unavailable under Option C."
            )
        else:
            lines[-1] += (
                f"**The MVEV floor is breached on the historical path — "
                f"the thesis claim of near-hard compliance requires qualification.**"
            )
    lines.append("")

    lines += [
        "---",
        "_Generated by evaluate_robustness.py.  "
        "All threshold and calibration values are derived from "
        "Jan 2000 – Dec 2015 training data only._",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Robustness evaluation: improved baselines + multi-path",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", default=MODEL_PATH_DEFAULT,
                   help="Path to run_042 best_model.zip")
    p.add_argument("--n-paths", type=int, default=N_SCENARIOS,
                   help="Number of VAR simulation paths")
    p.add_argument("--no-drl", action="store_true",
                   help="Skip DRL agent (baselines only, faster)")
    p.add_argument("--seed",  type=int, default=SEED)
    p.add_argument("--fr-init", type=float, default=1.05,
                   help="Initial FR for multi-path simulations")
    p.add_argument("--b-init",  type=float, default=0.05,
                   help="Initial buffer for multi-path simulations")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args       = parse_args(argv)
    model_path = Path(args.model_path)
    run_drl    = not args.no_drl
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("Wtp DRL Pension Fund -- Robustness Evaluation")
    print("=" * 68)
    print(f"  Model     : {model_path}")
    print(f"  VAR paths : {args.n_paths}")
    print(f"  Seed      : {args.seed}")
    print(f"  DRL agent : {'enabled (historical path)' if run_drl else 'disabled'}")

    # ---- 1. Data pipeline --------------------------------------------------
    print("\n[1/7] Running data pipeline...")
    results  = run_pipeline()

    # Auto-detect lifecycle from train_config.json
    tc_path  = model_path.parent / "train_config.json"
    lifecycle = True
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        lifecycle = bool(tc.get("lifecycle", True))
        print(f"  use_lifecycle: {lifecycle}  (from train_config.json)")
    env_cfg = EnvConfig(use_lifecycle=lifecycle)

    # ---- 2. Fit extended VAR -----------------------------------------------
    print("\n[2/7] Fitting 4-variable VAR (r_eq, d_swap_10y, pi, vstoxx)...")
    var_fit   = fit_extended_var(
        results["raw_train"], results["cpi"], results["z_train_raw"], env_cfg
    )

    print("\n  Generating VAR scenarios...")
    scenarios = generate_var_scenarios(
        var_fit,
        n_scenarios=args.n_paths,
        sim_horizon=SIM_HORIZON,
        seed=args.seed,
    )

    # ---- 3. Constrained MC calibration -------------------------------------
    print("\n[3/7] Calibrating Constrained MC ALM...")
    calib = calibrate_constrained_mc(
        scenarios, env_cfg,
        depletion_budget=DEPLETION_BUDGET,
        fr_init=args.fr_init,
        b_init=args.b_init,
    )

    # ---- 4. Regime Heuristic thresholds ------------------------------------
    print("\n[4/7] Deriving Regime Heuristic thresholds (training data only)...")
    h_thresh = derive_heuristic_thresholds(
        results["z_train_raw"], results["z_train"]
    )

    # ---- 5. Multi-path simulation ------------------------------------------
    print("\n[5/7] Running multi-path simulation (1,000 VAR paths × 3 policies)...")

    policy_specs = {
        "Fixed-Rule": dict(
            f_fixed=0.03, d_fixed=0.02, e_fixed=0.00,
        ),
        "Constrained MC": dict(
            f_fixed=calib["f_star"], d_fixed=calib["d_star"], e_fixed=0.00,
        ),
        "Regime Heuristic": dict(
            f_fixed=0.0, d_fixed=0.0, e_fixed=0.0,   # overridden by is_rh logic
        ),
    }

    multipath_raw: dict  = {}
    multipath_summary    = {}

    # Approximate test-period CPI for synthetic paths (use training mean)
    pi_train_mean = float(results["cpi"]["pi_monthly"]
                          .reindex(results["raw_train"].index)
                          .fillna(0.0).mean())
    pi_synthetic  = np.full(SIM_HORIZON, pi_train_mean, dtype=np.float64)

    for name, spec in policy_specs.items():
        is_rh = (name == "Regime Heuristic")
        print(f"  Simulating {name}...", end="", flush=True)
        traj = _run_vectorised_policy(
            r_eq        = scenarios["r_eq"],
            r_bond      = scenarios["r_bond"],
            r_L_MtM     = scenarios["r_L_MtM"],
            vstoxx      = scenarios["vstoxx"],
            policy_name = "regime_heuristic" if is_rh else name.lower().replace(" ", "_"),
            f_fixed     = spec["f_fixed"],
            d_fixed     = spec["d_fixed"],
            e_fixed     = spec["e_fixed"],
            p33_raw     = h_thresh["p33_raw"],
            p67_raw     = h_thresh["p67_raw"],
            env_cfg     = env_cfg,
            fr_init     = args.fr_init,
            b_init      = args.b_init,
        )
        multipath_raw[name] = traj
        m = compute_batch_metrics(traj, pi=pi_synthetic)
        multipath_summary[name] = summarise_batch_metrics(m)
        print(f"  done  (mean FR_T={multipath_summary[name]['terminal_FR_mean']:.4f}, "
              f"dep={multipath_summary[name]['buffer_depletion_freq_mean']:.3f})")

    # ---- 6. Historical path evaluation -------------------------------------
    print("\n[6/7] Running all policies on historical path (Jan 2018 - Dec 2025)...")
    hist_metrics, hist_trajs = run_all_historical(
        results, env_cfg, model_path,
        f_c     = calib["f_star"],
        d_c     = calib["d_star"],
        heuristic_thresholds = h_thresh,
        run_drl = run_drl,
        seed    = 0,
    )

    for name, m in hist_metrics.items():
        print(f"  {name:<20}  FR_T={m.get('fr_terminal', float('nan')):.4f}  "
              f"dist={m.get('total_distributions', float('nan')):.4f}  "
              f"dep={m.get('buffer_depletion_freq', float('nan')):.4f}  "
              f"MVEV_breaches={m.get('mvev_breach_count', '—')}")

    # ---- 7. Save outputs ---------------------------------------------------
    print("\n[7/7] Saving outputs...")

    # 7a. Multipath CSV
    mp_df = _multipath_to_df(multipath_summary)
    mp_path = RESULTS_DIR / "robustness_multipath.csv"
    mp_df.to_csv(mp_path, index=False)
    print(f"  Saved: {mp_path}")

    # 7b. Historical CSV
    hist_df = _historical_to_df(hist_metrics)
    hist_path = RESULTS_DIR / "robustness_historical.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"  Saved: {hist_path}")

    # 7c. Calibration JSON
    calib_json = {
        "f_star":            calib["f_star"],
        "d_star":            calib["d_star"],
        "depletion_budget":  calib["depletion_budget"],
        "n_feasible":        calib["n_feasible"],
        "n_total":           calib["n_total"],
        "constraint_binding": calib["constraint_binding"],
        "best_result":       calib["best_result"],
        "feasible_set":      calib["feasible_set"],
        "heuristic_thresholds": h_thresh,
        "var_lag_order":     var_fit["lag_order"],
        "swap20_beta":       var_fit["swap20_beta"],
        "swap20_intercept":  var_fit["swap20_intercept"],
        "n_scenarios":       args.n_paths,
        "sim_horizon":       SIM_HORIZON,
        "seed":              args.seed,
        "fr_init":           args.fr_init,
        "b_init":            args.b_init,
    }
    calib_path = RESULTS_DIR / "baseline_C_calibration.json"
    with open(calib_path, "w") as fh:
        json.dump(calib_json, fh, indent=2, default=float)
    print(f"  Saved: {calib_path}")

    # 7d. Markdown summary
    summary_md = _build_summary_md(
        multipath_summary, hist_metrics, calib, h_thresh
    )
    md_path = RESULTS_DIR / "robustness_summary.md"
    md_path.write_text(summary_md, encoding="utf-8")
    print(f"  Saved: {md_path}")

    print("\n" + "=" * 68)
    print("Done.")
    print("=" * 68)


if __name__ == "__main__":
    main()
