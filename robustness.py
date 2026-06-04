"""robustness.py — Out-of-sample robustness analysis for the Wtp DRL agent.

Checks implemented (all applied to the trained run_007 best_model.zip)
-----------------------------------------------------------------------
1. Initial conditions  : FR_init in {0.95, 1.00, 1.05, 1.10, 1.20},
                         B_init  in {0.02, 0.05, 0.10}
2. Transaction costs   : {0, 10, 25, 50} bps equity-turnover friction
3. Liability blend     : UFR-only (0%) / 50-50 / 70-30 (base) / MtM-only (100%)
4. DNB stress          : 6 stylised Besluit FTK Art. 23 shocks applied at
                         the start of the test period (Jan 2018)

Usage
-----
    py -3 robustness.py
    py -3 robustness.py --model-path src/models/run_007/best_model.zip
    py -3 robustness.py --skip-tc --skip-blend  # run only IC + DNB
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.data_pipeline   import run_pipeline
from src.environment     import WtpPensionEnv, EnvConfig, make_env_from_pipeline
from src.agent           import AgentConfig, WtpActorCriticPolicy
from src.baselines       import FixedRuleALM
from src.metrics         import compute_metrics, diebold_mariano, dm_losses, format_dm_table

try:
    from src.hoevenaars_alm import HoevenaarsALM as _HoevenaarsALM
except ImportError:
    _HoevenaarsALM = None

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required.  pip install stable-baselines3"
    ) from exc


# ---------------------------------------------------------------------------
# SB3 adapter
# ---------------------------------------------------------------------------

class _SB3Adapter:
    """Thin wrapper: model.predict() -> run_episode-compatible predict()."""

    def __init__(self, model) -> None:
        self._model = model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action


# ---------------------------------------------------------------------------
# Episode runner with optional transaction-cost deduction
# ---------------------------------------------------------------------------

def _run_episode(agent, env, tc_bps: float = 0.0,
                 fr_init: Optional[float] = None,
                 b_init:  Optional[float] = None) -> dict:
    """Run a full episode; optionally apply equity-turnover transaction costs.

    Args:
        agent:   Any object with ``predict(obs) -> action``.
        env:     :class:`WtpPensionEnv` instance.
        tc_bps:  Transaction cost in basis points per unit of equity-weight
                 turnover.  0 = no costs.
        fr_init: Override initial funding ratio (via reset options).
        b_init:  Override initial buffer level (via reset options).

    Returns:
        Trajectory dict (same schema as ``baselines.run_episode``).
    """
    tc_rate = tc_bps / 10_000
    options = {}
    if fr_init is not None:
        options["fr_init"] = fr_init
    if b_init is not None:
        options["b_init"] = b_init

    # Reset any stateful agent internals (e.g. Hoevenaars _step counter)
    # before starting a new episode.
    if hasattr(agent, "reset"):
        agent.reset()

    obs, _ = env.reset(options=options if options else None)

    prev_w_eq = env.cfg.w_eq_base

    dates, FR, B         = [], [], []
    w_eq_t, r_p_t, r_L_t = [], [], []
    f_tilde_t, d_tilde_t  = [], []
    rewards               = []

    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        w_eq = info["w_eq"]

        # ---- Apply equity-turnover transaction cost ---------------------- #
        if tc_rate > 0.0:
            turnover = abs(w_eq - prev_w_eq)
            tc       = tc_rate * turnover
            # Deduct from portfolio return and propagate to FR
            info["r_p"] -= tc
            env._fr     -= tc   # patch internal FR so next step starts correctly

        prev_w_eq = w_eq

        dates.append(info["date"])
        FR.append(env._fr)          # use (possibly TC-adjusted) FR
        B.append(info["B"])
        w_eq_t.append(w_eq)
        r_p_t.append(info["r_p"])
        r_L_t.append(info["r_L"])
        f_tilde_t.append(info["f_tilde"])
        d_tilde_t.append(info["d_tilde"])
        rewards.append(reward)

    return {
        "dates":        dates,
        "FR":           np.array(FR,        dtype=np.float64),
        "B":            np.array(B,         dtype=np.float64),
        "w_eq":         np.array(w_eq_t,    dtype=np.float64),
        "r_p":          np.array(r_p_t,     dtype=np.float64),
        "r_L":          np.array(r_L_t,     dtype=np.float64),
        "f_tilde":      np.array(f_tilde_t, dtype=np.float64),
        "d_tilde":      np.array(d_tilde_t, dtype=np.float64),
        "rewards":      np.array(rewards,   dtype=np.float64),
        "total_reward": float(np.sum(rewards)),
        "terminated":   terminated,
        "n_steps":      len(rewards),
    }


# ---------------------------------------------------------------------------
# Helper: build test environment with array overrides
# ---------------------------------------------------------------------------

def _make_test_env(
    results:              dict,
    cfg:                  Optional[EnvConfig] = None,
    r_eq_override:        Optional[np.ndarray] = None,
    r_bond_override:      Optional[np.ndarray] = None,
    r_L_MtM_override:     Optional[np.ndarray] = None,
    suppress_r_L_blended: bool = False,
    seed:                 int = 0,
) -> WtpPensionEnv:
    """Build a test-split WtpPensionEnv with optional market-data overrides.

    Passes through to the normal factory, but replaces individual return
    series before instantiation — used for liability-blend and DNB-stress
    checks.

    Args:
        results:          Pipeline output dict.
        cfg:              :class:`EnvConfig`; defaults to ``EnvConfig()``.
        r_eq_override:    Replacement equity return array ``(T,)``.
        r_bond_override:  Replacement bond return array ``(T,)``.
        r_L_MtM_override: Replacement MtM liability return array ``(T,)``.
        seed:             RNG seed.

    Returns:
        Configured :class:`WtpPensionEnv`.
    """
    env_cfg  = cfg or EnvConfig()
    z_scaled = results["z_test"].values
    z_raw    = results["z_test_raw"]
    dates    = results["z_test"].index
    cpi      = results["cpi"]
    pi_series = cpi["pi_monthly"].reindex(dates).fillna(0.0).values

    # Default market return arrays (same logic as make_env_from_pipeline)
    r_eq = (
        (np.exp(z_raw["mom_msci_1m"].values) - 1.0)
        .clip(env_cfg.r_eq_clip[0], env_cfg.r_eq_clip[1])
    )
    r_bond = (
        (-env_cfg.duration * z_raw["d_swap_10y"].values / 100.0)
        .clip(env_cfg.r_bond_clip[0], env_cfg.r_bond_clip[1])
    )
    if "d_rts_20y" in z_raw.columns:
        r_L_MtM = -env_cfg.duration * z_raw["d_rts_20y"].values / 100.0
    else:
        r_L_MtM = -env_cfg.duration * z_raw["d_swap_20y"].values / 100.0

    # Pre-computed DNB RTS blended liability return — must be passed through
    # so robustness env matches evaluate.py exactly (same liability dynamics).
    # Suppressed when: (a) caller sets suppress_r_L_blended=True (Check 3 blend
    # sensitivity, where we need the in-house blend to respond to
    # liability_mtm_weight), or (b) bond/liability arrays are overridden (DNB
    # stress, where shocked series must take precedence).
    r_L_blended = None
    if "r_L_blended" in results and not suppress_r_L_blended:
        r_L_blended = (
            results["r_L_blended"]
            .reindex(dates)
            .fillna(0.0)
            .values
        )

    # Apply overrides
    if r_eq_override    is not None: r_eq    = r_eq_override.copy()
    if r_bond_override  is not None: r_bond  = r_bond_override.copy()
    if r_L_MtM_override is not None: r_L_MtM = r_L_MtM_override.copy()

    # Clear blended series when shocked so stressed dynamics are used.
    if r_bond_override is not None or r_L_MtM_override is not None:
        r_L_blended = None

    return WtpPensionEnv(
        z_scaled    = z_scaled,
        r_eq        = r_eq,
        r_bond      = r_bond,
        r_L_MtM     = r_L_MtM,
        pi_monthly  = pi_series,
        dates       = dates,
        cfg         = env_cfg,
        seed        = seed,
        r_L_blended = r_L_blended,
    )


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_METRIC_KEYS = [
    ("fr_terminal",           "FR Terminal",  ".4f"),
    ("fr_mdd",                "FR MDD",       ".4f"),
    ("buffer_depletion_freq", "Buf Depl",     ".4f"),
    ("total_distributions",   "Total Dist",   ".4f"),
    ("calmar_ratio",          "Calmar",       ".4f"),
]


def _format_table(rows: list[tuple], col_headers: list[str], title: str) -> str:
    """Format a robustness result table.

    Args:
        rows:         List of ``(row_label, metrics_dict)`` tuples.
        col_headers:  Column header labels (one per metrics key).
        title:        Table title.

    Returns:
        Formatted multi-line string.
    """
    label_w  = max(len(r[0]) for r in rows) + 2
    col_w    = 10

    header   = f"  {'':>{label_w}}" + "".join(f"{h:>{col_w}}" for h in col_headers)
    sep      = "-" * len(header)

    lines = [title, "=" * len(header), header, sep]
    for label, m in rows:
        row = f"  {label:>{label_w}}"
        for key, _, fmt in _METRIC_KEYS:
            v = m.get(key, float("nan"))
            row += f"{v:{col_w}{fmt}}"
        lines.append(row)
    lines.append("=" * len(header))
    return "\n".join(lines)


def _metrics_row(traj: dict, pi_test: np.ndarray) -> dict:
    T = len(traj["FR"])
    pi = pi_test[:T] if len(pi_test) >= T else pi_test
    return compute_metrics(traj, pi_monthly=pi)


# ---------------------------------------------------------------------------
# Check 1: Initial condition sensitivity
# ---------------------------------------------------------------------------

def check_initial_conditions(
    agent,
    results:   dict,
    env_cfg:   EnvConfig,
    pi_test:   np.ndarray,
) -> dict:
    """Vary FR_init and B_init; report core metrics for each combination.

    Returns:
        Nested dict ``{label: metrics_dict}``.
    """
    print("\n" + "=" * 64)
    print("CHECK 1: Initial Condition Sensitivity")
    print("=" * 64)

    fr_grid = [0.95, 1.00, 1.05, 1.10, 1.20]
    b_fixed = 0.05

    # --- FR grid (B fixed at 0.05) ---------------------------------------- #
    print("\n  Varying FR_init  (B_init=0.05 fixed):")
    fr_rows = []
    fr_results = {}
    for fr in fr_grid:
        env = _make_test_env(results, cfg=env_cfg)
        traj = _run_episode(agent, env, fr_init=fr, b_init=b_fixed)
        m    = _metrics_row(traj, pi_test)
        label = f"FR={fr:.2f}"
        fr_rows.append((label, m))
        fr_results[label] = m
        print(f"    FR_init={fr:.2f}  term={m['fr_terminal']:.4f}  "
              f"mdd={m['fr_mdd']:.4f}  dep={m['buffer_depletion_freq']:.4f}  "
              f"dist={m['total_distributions']:.4f}  calmar={m['calmar_ratio']:.4f}")

    col_headers = [h for _, h, _ in _METRIC_KEYS]
    print("\n" + _format_table(fr_rows, col_headers, "FR_init sensitivity  (B_init=0.05)"))

    # --- B grid (FR fixed at 1.05) ---------------------------------------- #
    b_grid  = [0.01, 0.02, 0.05, 0.10, 0.15]
    fr_fixed = 1.05
    print(f"\n  Varying B_init  (FR_init={fr_fixed:.2f} fixed):")
    b_rows = []
    b_results = {}
    for b in b_grid:
        env = _make_test_env(results, cfg=env_cfg)
        traj = _run_episode(agent, env, fr_init=fr_fixed, b_init=b)
        m    = _metrics_row(traj, pi_test)
        label = f"B={b:.2f}"
        b_rows.append((label, m))
        b_results[label] = m
        print(f"    B_init={b:.2f}  term={m['fr_terminal']:.4f}  "
              f"mdd={m['fr_mdd']:.4f}  dep={m['buffer_depletion_freq']:.4f}  "
              f"dist={m['total_distributions']:.4f}  calmar={m['calmar_ratio']:.4f}")

    print("\n" + _format_table(b_rows, col_headers, f"B_init sensitivity  (FR_init={fr_fixed:.2f})"))

    return {"fr_grid": fr_results, "b_grid": b_results}


# ---------------------------------------------------------------------------
# Check 2: Transaction cost sensitivity
# ---------------------------------------------------------------------------

def check_transaction_costs(
    agent,
    results: dict,
    env_cfg: EnvConfig,
    pi_test: np.ndarray,
) -> dict:
    """Vary equity-turnover transaction costs via EnvConfig; measure impact.

    TC is applied inside the environment (same mechanism as training) rather
    than post-hoc.  The base env_cfg carries the training tc_bps, so the grid
    includes that level explicitly as the "training condition" baseline.

    Returns:
        Dict ``{tc_label: metrics_dict}``.
    """
    print("\n" + "=" * 64)
    print("CHECK 2: Transaction Cost Sensitivity")
    print("=" * 64)

    train_tc = env_cfg.tc_bps          # e.g. 10 for run_042, 0 for run_039
    tc_grid  = sorted({0, 10, 25, 50, int(train_tc)})   # always include training level
    rows     = []
    tc_results = {}

    for tc_bps in tc_grid:
        # Vary tc_bps via env config — no post-hoc deduction
        cfg_tc = replace(env_cfg, tc_bps=float(tc_bps))
        env    = _make_test_env(results, cfg=cfg_tc)
        traj   = _run_episode(agent, env, tc_bps=0.0)   # TC already in env
        m      = _metrics_row(traj, pi_test)
        suffix = " (training)" if tc_bps == train_tc and train_tc > 0 else ""
        label  = f"{tc_bps} bps{suffix}"
        rows.append((label, m))
        tc_results[label] = m
        print(f"  TC={tc_bps:>2d} bps{suffix} | term={m['fr_terminal']:.4f}  "
              f"mdd={m['fr_mdd']:.4f}  dep={m['buffer_depletion_freq']:.4f}  "
              f"dist={m['total_distributions']:.4f}  calmar={m['calmar_ratio']:.4f}")

    col_headers = [h for _, h, _ in _METRIC_KEYS]
    print("\n" + _format_table(rows, col_headers,
                               "Transaction cost sensitivity  (FR_init=1.05, B_init=0.05)"))
    return tc_results


# ---------------------------------------------------------------------------
# Check 3: Liability-blend sensitivity (DRL vs baselines)
# ---------------------------------------------------------------------------

def check_liability_blend(
    agent,
    fixed_agent,
    results:    dict,
    pi_test:    np.ndarray,
    mc_agent=None,
    hoev_agent=None,
    env_cfg:    Optional[EnvConfig] = None,
) -> dict:
    """Vary the MtM / UFR liability blend weight for DRL and all baselines.

    Economic motivation
    -------------------
    The liability discount rate used in Dutch pension accounting is contested.
    DNB prescribes a UFR-blended rate curve; pure MtM (swap curve) is the
    theoretically correct present value.  The blend weight directly shifts
    every agent's FR trajectory in absolute terms — UFR makes liabilities
    cheaper, lifting all FRs mechanically.

    The thesis robustness argument: the DRL's *relative* advantage over
    benchmarks must be stable across blend choices.  If the gap shrinks or
    reverses under a specific blend, the DRL advantage is an artifact of the
    liability assumption rather than of learning.  Including all baselines
    side-by-side makes the stability (or fragility) visible in one table.

    Blends tested:
      - 0.00 : Pure UFR  (r_UFR = 0.002711/month = 3.30% ann.)
      - 0.50 : 50-50
      - 0.70 : 70-30 base (training default, Art. 15 Wtp blended curve)
      - 1.00 : Pure MtM  (full interest-rate sensitivity, swap curve)

    Args:
        agent:       DRL agent (has ``predict(obs) -> action``).
        fixed_agent: Fixed-Rule ALM baseline.
        results:     Data pipeline output dict.
        pi_test:     Monthly CPI inflation array for the test period.
        mc_agent:    Optional Monte Carlo ALM baseline (slow; pass None to skip).
        hoev_agent:  Optional Hoevenaars ALM baseline (pass None to skip).

    Returns:
        Nested dict ``{blend_label: {"DRL": metrics, "Fixed-Rule": metrics,
        ["Monte Carlo": metrics], ["Hoevenaars": metrics],
        "gap_drl_fixed": metrics}}``.
    """
    print("\n" + "=" * 64)
    print("CHECK 3: Liability Blend Sensitivity  (DRL vs all baselines)")
    print("=" * 64)

    blend_grid = [
        (0.00, "Pure UFR  (0%)"),
        (0.50, "50-50"),
        (0.70, "70-30 (base)"),
        (1.00, "Pure MtM (100%)"),
    ]

    col_headers = [h for _, h, _ in _METRIC_KEYS]

    # Per-agent row lists for separate formatted tables.
    drl_rows    = []
    fixed_rows  = []
    mc_rows     = []
    hoev_rows   = []
    gap_rows    = []   # DRL − Fixed-Rule
    bl_results  = {}

    _base_w_mtm = (env_cfg or EnvConfig()).liability_mtm_weight

    for w_mtm, label in blend_grid:
        # For non-base blends: suppress the pre-baked pipeline RTS series so
        # the environment computes liability from liability_mtm_weight directly.
        # For the training-default blend (w_mtm == base): keep suppress=False
        # so the pipeline series is used — exactly matching evaluate.py and
        # making Table 5.5.1 consistent with Table 4.1.1 at the base row.
        suppress = abs(w_mtm - _base_w_mtm) > 1e-9
        cfg = replace(env_cfg or EnvConfig(), liability_mtm_weight=w_mtm)

        env_drl   = _make_test_env(results, cfg=cfg, suppress_r_L_blended=suppress)
        env_fixed = _make_test_env(results, cfg=cfg, suppress_r_L_blended=suppress)

        traj_drl   = _run_episode(agent,       env_drl)
        traj_fixed = _run_episode(fixed_agent, env_fixed)

        m_drl   = _metrics_row(traj_drl,   pi_test)
        m_fixed = _metrics_row(traj_fixed, pi_test)

        # Arithmetic gap: positive = DRL higher than Fixed.
        # For MDD and Buf Depl, *lower* is better, so gap < 0 favours DRL.
        # We record the raw difference and explain the sign in the summary.
        gap_m = {k: m_drl[k] - m_fixed[k] for k, _, _ in _METRIC_KEYS}

        entry: dict = {
            "DRL":          m_drl,
            "Fixed-Rule":   m_fixed,
            "gap_drl_fixed": gap_m,
        }

        if mc_agent is not None:
            env_mc  = _make_test_env(results, cfg=cfg, suppress_r_L_blended=suppress)
            traj_mc = _run_episode(mc_agent, env_mc)
            m_mc    = _metrics_row(traj_mc, pi_test)
            entry["Monte Carlo"] = m_mc
            mc_rows.append((label, m_mc))

        if hoev_agent is not None:
            env_hoev  = _make_test_env(results, cfg=cfg, suppress_r_L_blended=suppress)
            traj_hoev = _run_episode(hoev_agent, env_hoev)
            m_hoev    = _metrics_row(traj_hoev, pi_test)
            entry["Hoevenaars"] = m_hoev
            hoev_rows.append((label, m_hoev))

        bl_results[label] = entry
        drl_rows.append((label,   m_drl))
        fixed_rows.append((label, m_fixed))
        gap_rows.append((label,   gap_m))

        # Inline progress per blend
        def _pr(tag, m):
            print(f"  {label:<18}  {tag:<12}: "
                  f"term={m['fr_terminal']:.4f}  "
                  f"mdd={m['fr_mdd']:.4f}  "
                  f"dep={m['buffer_depletion_freq']:.4f}  "
                  f"dist={m['total_distributions']:.4f}  "
                  f"calmar={m['calmar_ratio']:.4f}")

        _pr("DRL",        m_drl)
        _pr("Fixed-Rule", m_fixed)
        if mc_agent   is not None: _pr("Monte Carlo",  entry["Monte Carlo"])
        if hoev_agent is not None: _pr("Hoevenaars",   entry["Hoevenaars"])
        print()

    # Stacked formatted tables (one per agent)
    print("\n" + _format_table(drl_rows,   col_headers, "DRL agent  (liability blend sensitivity)"))
    print("\n" + _format_table(fixed_rows, col_headers, "Fixed-Rule ALM  (liability blend sensitivity)"))
    if mc_rows:
        print("\n" + _format_table(mc_rows,   col_headers, "Monte Carlo ALM  (liability blend sensitivity)"))
    if hoev_rows:
        print("\n" + _format_table(hoev_rows, col_headers, "Hoevenaars ALM  (liability blend sensitivity)"))

    # Gap table: DRL minus Fixed-Rule.  col_headers are intentionally reused
    # (same 10-char column width) — sign convention is explained below.
    print("\n  Gap table sign convention:")
    print("    +FR Terminal / +Total Dist / +Calmar  => DRL better than Fixed-Rule")
    print("    -FR MDD      / -Buf Depl              => DRL better than Fixed-Rule")
    print("\n" + _format_table(gap_rows, col_headers,
                               "Gap: DRL minus Fixed-Rule"))

    # Stability summary: range of the DRL-Fixed gap across all blends.
    # A narrow span confirms the relative advantage is blend-invariant.
    print("\n  Gap stability across blends (DRL - Fixed-Rule):")
    for key, header, _ in _METRIC_KEYS:
        gaps  = [bl_results[lbl]["gap_drl_fixed"][key] for _, lbl in blend_grid]
        base  = gaps[2]   # index 2 = 70-30 base blend
        span  = max(gaps) - min(gaps)
        print(f"    {header:<12}  [{min(gaps):+.4f}, {max(gaps):+.4f}]  "
              f"base={base:+.4f}  span={span:.4f}")

    # ---- Diebold-Mariano test under each blend ----------------------------- #
    # Formally tests whether the DRL advantage is statistically significant
    # at every discount-rate assumption, closing the blend robustness argument.
    # We test DRL vs Fixed-Rule (primary) and DRL vs Hoevenaars (if available)
    # on both the buffer-depletion indicator and the FR stability loss.
    print("\n  Diebold-Mariano significance under each blend (HLN correction):")

    # Reconstruct per-step loss arrays stored during the loop above.
    # bl_results[label] already holds trajectories implicitly via the gap dicts;
    # we need the raw FR/B arrays, so re-run the episodes here at negligible cost
    # (deterministic policies, same env seeds).
    dm_by_blend: dict[str, dict] = {}

    for w_mtm, label in blend_grid:
        suppress  = abs(w_mtm - _base_w_mtm) > 1e-9
        cfg       = replace(env_cfg or EnvConfig(), liability_mtm_weight=w_mtm)
        env_drl_  = _make_test_env(results, cfg=cfg, suppress_r_L_blended=suppress)
        env_fx_   = _make_test_env(results, cfg=cfg, suppress_r_L_blended=suppress)

        traj_drl_ = _run_episode(agent,       env_drl_)
        traj_fx_  = _run_episode(fixed_agent, env_fx_)

        drl_l = dm_losses(traj_drl_)
        fx_l  = dm_losses(traj_fx_)
        T_dm  = len(drl_l["buf_depl"])

        entry: dict = {
            "DRL vs Fixed-Rule": {
                "Buf. Depl.":  diebold_mariano(fx_l["buf_depl"],   drl_l["buf_depl"]),
                "FR Stab.":    diebold_mariano(fx_l["fr_sq_dev"],  drl_l["fr_sq_dev"]),
            }
        }

        if hoev_agent is not None:
            env_hv_ = _make_test_env(results, cfg=cfg, suppress_r_L_blended=suppress)
            traj_hv = _run_episode(hoev_agent, env_hv_)
            hv_l    = dm_losses(traj_hv)
            entry["DRL vs Hoevenaars"] = {
                "Buf. Depl.": diebold_mariano(hv_l["buf_depl"],  drl_l["buf_depl"]),
                "FR Stab.":   diebold_mariano(hv_l["fr_sq_dev"], drl_l["fr_sq_dev"]),
            }

        dm_by_blend[label] = entry
        bl_results[label]["dm"] = entry

    # Per-blend DM summary table (DRL vs Fixed-Rule, two losses)
    print()
    _BLEND_COL_W = max(len(lbl) for _, lbl in blend_grid) + 2
    _CELL        = 22
    _loss_hdrs   = ["Buf. Depl.", "FR Stab."]
    _hdr_line    = (f"  {'Blend':<{_BLEND_COL_W}}"
                    + "".join(f"{h:^{_CELL}}" for h in _loss_hdrs))
    _sub_line    = (f"  {'':^{_BLEND_COL_W}}"
                    + "".join(f"{'Stat':>7}{'p-val':>8}{'':>7}"
                              for _ in _loss_hdrs))
    _sep         = "-" * len(_hdr_line)
    print(f"  DRL vs Fixed-Rule (HLN, T={T_dm}, h=1)")
    print("  " + "=" * (len(_hdr_line) - 2))
    print(_hdr_line)
    print(_sub_line)
    print(_sep)
    for _, label in blend_grid:
        row = f"  {label:<{_BLEND_COL_W}}"
        for lkey in _loss_hdrs:
            stat, p = dm_by_blend[label]["DRL vs Fixed-Rule"][lkey]
            p_str   = "<0.001" if p < 0.001 else f"{p:.3f}"
            sig     = ("**" if p < 0.01 else "*" if p < 0.05
                       else "." if p < 0.10 else "")
            row    += f"  {stat:>6.2f}  {p_str:>6}  {sig:<2}"
        print(row)
    print(_sep)
    print("  ** p<0.01   * p<0.05   . p<0.10")
    print("  " + "=" * (len(_hdr_line) - 2))

    return bl_results


# ---------------------------------------------------------------------------
# Check 4: DNB stress scenarios (Besluit FTK Art. 23)
# ---------------------------------------------------------------------------

# Six stylised shocks applied at month 0 of the test period (Jan 2018).
# Shock parameters (d = duration = 18 years):
#
#  S1 Rate Down   : All yields -100 bps
#                   r_bond[0] = clip(-d * (-1.0)/100, -5%, +5%) = +5%
#                   r_L_MtM[0] = -d * (-1.0)/100 = +18%   (no clip on liability)
#
#  S2 Rate Up     : All yields +200 bps
#                   r_bond[0] = clip(-d * (+2.0)/100, -5%, +5%) = -5%
#                   r_L_MtM[0] = -d * (+2.0)/100 = -36%
#
#  S3 Equity      : Instantaneous -40% equity return
#                   r_eq[0] = -0.40  (clips to -0.30 in env)
#
#  S4 Moderate Eq : Instantaneous -20% equity return
#                   r_eq[0] = -0.20
#
#  S5 Rate Down + Equity : S1 + S3 combined
#
#  S6 Rate Up   + Equity : S2 + S3 combined

def check_dnb_stress(
    agent,
    results:  dict,
    env_cfg:  EnvConfig,
    pi_test:  np.ndarray,
) -> dict:
    """Run 6 DNB Besluit FTK Art. 23 stress scenarios.

    Each scenario injects a one-month shock at t=0 (Jan 2018); the remaining
    test period uses historical market data unchanged.

    Returns:
        Dict ``{scenario_name: metrics_dict}``.
    """
    print("\n" + "=" * 64)
    print("CHECK 4: DNB Stress Scenarios  (Besluit FTK Art. 23)")
    print("=" * 64)

    d   = env_cfg.duration   # 18 years
    # The env resets to t = lb - 1, so the first executed step reads index [t0].
    t0  = env_cfg.lookback - 1   # = 11 for lb=12

    # Retrieve baseline arrays ------------------------------------------------
    z_raw  = results["z_test_raw"]
    dates  = results["z_test"].index

    r_eq_base = (
        (np.exp(z_raw["mom_msci_1m"].values) - 1.0)
        .clip(env_cfg.r_eq_clip[0], env_cfg.r_eq_clip[1])
    ).copy()

    r_bond_base = (
        (-d * z_raw["d_swap_10y"].values / 100.0)
        .clip(env_cfg.r_bond_clip[0], env_cfg.r_bond_clip[1])
    ).copy()

    col_20y = "d_rts_20y" if "d_rts_20y" in z_raw.columns else "d_swap_20y"
    r_L_MtM_base = (-d * z_raw[col_20y].values / 100.0).copy()

    print(f"  (Shocks injected at array index {t0} = {dates[t0].date()}, "
          f"the first executed step)")

    # Baseline (no shock) -----------------------------------------------------
    env_base = _make_test_env(results, cfg=env_cfg)
    traj_base = _run_episode(agent, env_base)
    m_base   = _metrics_row(traj_base, pi_test)

    # Scenario definitions ----------------------------------------------------
    scenarios = []

    # S0: Baseline (no shock)
    scenarios.append(("Baseline (no shock)", {}, {}, {}))

    # S1: Rate Down -100 bps
    r_bond_s1    = r_bond_base.copy()
    r_L_MtM_s1   = r_L_MtM_base.copy()
    r_bond_s1[t0]   = np.clip(-d * (-1.0) / 100.0,
                               env_cfg.r_bond_clip[0], env_cfg.r_bond_clip[1])
    r_L_MtM_s1[t0]  = -d * (-1.0) / 100.0
    scenarios.append(("S1 Rate Down -100bps",
                       {"r_bond_override": r_bond_s1,
                        "r_L_MtM_override": r_L_MtM_s1}, {}, {}))

    # S2: Rate Up +200 bps
    r_bond_s2    = r_bond_base.copy()
    r_L_MtM_s2   = r_L_MtM_base.copy()
    r_bond_s2[t0]   = np.clip(-d * 2.0 / 100.0,
                               env_cfg.r_bond_clip[0], env_cfg.r_bond_clip[1])
    r_L_MtM_s2[t0]  = -d * 2.0 / 100.0
    scenarios.append(("S2 Rate Up  +200bps",
                       {"r_bond_override": r_bond_s2,
                        "r_L_MtM_override": r_L_MtM_s2}, {}, {}))

    # S3: Equity crash -40%  (clips to -30% inside the env)
    r_eq_s3 = r_eq_base.copy()
    r_eq_s3[t0] = np.clip(-0.40, env_cfg.r_eq_clip[0], env_cfg.r_eq_clip[1])
    scenarios.append(("S3 Equity   -40%",
                       {"r_eq_override": r_eq_s3}, {}, {}))

    # S4: Moderate equity -20%
    r_eq_s4 = r_eq_base.copy()
    r_eq_s4[t0] = -0.20
    scenarios.append(("S4 Equity   -20%",
                       {"r_eq_override": r_eq_s4}, {}, {}))

    # S5: Rate Down + Equity crash (combined worst-case funding shock)
    # Note: r_eq_s3 / r_bond_s1 / r_L_MtM_s1 already have shock at t0
    scenarios.append(("S5 Rate Down + Equity",
                       {"r_eq_override":    r_eq_s3.copy(),
                        "r_bond_override":   r_bond_s1.copy(),
                        "r_L_MtM_override":  r_L_MtM_s1.copy()}, {}, {}))

    # S6: Rate Up + Equity crash (standard combined FTK shock)
    scenarios.append(("S6 Rate Up  + Equity",
                       {"r_eq_override":    r_eq_s3.copy(),
                        "r_bond_override":   r_bond_s2.copy(),
                        "r_L_MtM_override":  r_L_MtM_s2.copy()}, {}, {}))

    # Run all scenarios -------------------------------------------------------
    rows = []
    stress_results = {}

    for name, overrides, _unused1, _unused2 in scenarios:
        if name == "Baseline (no shock)":
            traj = traj_base
            m    = m_base
        else:
            env  = _make_test_env(results, cfg=env_cfg, **overrides)
            traj = _run_episode(agent, env)
            m    = _metrics_row(traj, pi_test)

        rows.append((name, m))
        stress_results[name] = m
        print(f"  {name:<26} | FR_1={traj['FR'][0]:.4f}  "
              f"term={m['fr_terminal']:.4f}  mdd={m['fr_mdd']:.4f}  "
              f"dep={m['buffer_depletion_freq']:.4f}  "
              f"calmar={m['calmar_ratio']:.4f}")

    col_headers = [h for _, h, _ in _METRIC_KEYS]
    print("\n" + _format_table(rows, col_headers, "DNB stress scenarios"))

    # Shock impact summary (delta from baseline) ------------------------------
    print("\n  Shock impact vs baseline  (delta FR Terminal | delta MDD):")
    base_term = m_base["fr_terminal"]
    base_mdd  = m_base["fr_mdd"]
    for name, m in stress_results.items():
        if name == "Baseline (no shock)":
            continue
        d_term = m["fr_terminal"] - base_term
        d_mdd  = m["fr_mdd"]      - base_mdd
        print(f"    {name:<26}  dFR_term={d_term:+.4f}  dMDD={d_mdd:+.4f}")

    return stress_results


# ---------------------------------------------------------------------------
# Check 5: Reward weight sensitivity (one-at-a-time ±50%)
# ---------------------------------------------------------------------------

def check_reward_weights(
    drl_agent,
    fixed_agent,
    results:  dict,
    pi_test:  np.ndarray,
    env_cfg:  Optional[EnvConfig] = None,
) -> dict:
    """Vary each reward weight ±50% (one-at-a-time) and compare DRL vs Fixed-Rule.

    Since the DRL policy is deterministic, the *physical* trajectory (FR, B,
    distributions) is identical across weight perturbations.  Only the total
    episode reward changes.  This confirms that the DRL advantage is not an
    artefact of the specific weight choice used for training.

    Base weights are read from ``env_cfg`` so this check automatically
    reflects the actual training configuration (including tc_bps).

    Returns:
        Dict ``{config_label: {"drl": total_reward, "fixed": total_reward,
        "advantage": drl - fixed}}``.
    """
    print("\n" + "=" * 64)
    print("CHECK 5: Reward Weight Sensitivity  (one-at-a-time +/-50%)")
    print("=" * 64)

    base_cfg = env_cfg or EnvConfig()

    # All reward-relevant scalar fields from the training config
    base = dict(
        alpha          = base_cfg.alpha,
        beta           = base_cfg.beta,
        gamma          = base_cfg.gamma,
        delta          = base_cfg.delta,
        fill_bonus     = base_cfg.fill_bonus,
        epsilon_equity = base_cfg.epsilon_equity,
        dist_weight    = base_cfg.dist_weight,
    )

    print(f"  Base weights: " +
          "  ".join(f"{k}={v}" for k, v in base.items()))

    # Build list of (label, overrides) -- one-at-a-time perturbations
    configs = [("base (no change)", {})]
    for weight in ("alpha", "beta", "gamma", "fill_bonus",
                   "epsilon_equity", "dist_weight"):
        for scale, tag in ((0.5, "-50%"), (1.5, "+50%")):
            configs.append((
                f"{weight} {tag}",
                {weight: base[weight] * scale},
            ))

    rows  = []
    rw_results = {}

    for label, overrides in configs:
        # Merge overrides into full env_cfg (preserves tc_bps and all other fields)
        cfg_kw = {**base, **overrides}
        perturbed_cfg = replace(
            base_cfg,
            alpha          = cfg_kw["alpha"],
            beta           = cfg_kw["beta"],
            gamma          = cfg_kw["gamma"],
            delta          = cfg_kw["delta"],
            fill_bonus     = cfg_kw["fill_bonus"],
            epsilon_equity = cfg_kw["epsilon_equity"],
            dist_weight    = cfg_kw["dist_weight"],
        )
        env_drl   = _make_test_env(results, cfg=perturbed_cfg)
        env_fixed = _make_test_env(results, cfg=perturbed_cfg)

        traj_drl   = _run_episode(drl_agent,   env_drl)
        traj_fixed = _run_episode(fixed_agent, env_fixed)

        r_drl   = traj_drl["total_reward"]
        r_fixed = traj_fixed["total_reward"]
        adv     = r_drl - r_fixed

        rows.append((label, r_drl, r_fixed, adv))
        rw_results[label] = {
            "drl_reward":   float(r_drl),
            "fixed_reward": float(r_fixed),
            "advantage":    float(adv),
        }
        sign = "+" if adv >= 0 else ""
        print(f"  {label:<28}  DRL={r_drl:+8.2f}  Fixed={r_fixed:+8.2f}  "
              f"adv={sign}{adv:.2f}")

    # Formatted table
    col_w  = 12
    header = f"  {'Config':<28}  {'DRL Reward':>{col_w}}  {'Fixed Reward':>{col_w}}  {'Advantage':>{col_w}}"
    sep    = "-" * len(header)
    lines  = ["\nReward weight sensitivity", "=" * len(header), header, sep]
    for label, r_drl, r_fixed, adv in rows:
        sign = "+" if adv >= 0 else ""
        lines.append(
            f"  {label:<28}  {r_drl:{col_w}.2f}  {r_fixed:{col_w}.2f}  "
            f"{sign}{adv:{col_w-1}.2f}"
        )
    lines.append("=" * len(header))
    print("\n".join(lines))

    n_pos = sum(1 for _, r, f, _ in rows if r > f)
    print(f"\n  DRL outscores Fixed-Rule in {n_pos}/{len(rows)} weight configurations.")
    return rw_results


# ---------------------------------------------------------------------------
# Check 6: Regime count K sensitivity
# ---------------------------------------------------------------------------

def check_regime_k(
    drl_agent,
    fixed_agent,
    results:  dict,
    pi_test:  np.ndarray,
    env_cfg:  Optional[EnvConfig] = None,
) -> dict:
    """Report regime-conditional metrics under K=2, K=3, K=4 VSTOXX splits.

    The trained model uses K=3 GMM gating; this check verifies that the
    DRL advantage holds regardless of how we partition VSTOXX into regimes.

    Threshold sets
    --------------
    K=2  :  Low (<25),             High (>=25)
    K=3  :  Low (<20), Med (20-30),  High (>=30)   [training default]
    K=4  :  VLow (<15), Low (15-25), Med (25-35), High (>=35)

    Returns:
        Nested dict ``{K_label: {regime: {agent: metrics}}}``.
    """
    print("\n" + "=" * 64)
    print("CHECK 6: Regime Count K Sensitivity  (K=2, 3, 4)")
    print("=" * 64)

    # Run both agents once on base env (preserves tc_bps and other training fields)
    base_cfg   = env_cfg or EnvConfig()
    env_drl    = _make_test_env(results, cfg=base_cfg)
    env_fixed  = _make_test_env(results, cfg=base_cfg)
    traj_drl   = _run_episode(drl_agent,   env_drl)
    traj_fixed = _run_episode(fixed_agent, env_fixed)

    test_dates = results["z_test"].index
    vstoxx = (
        results["z_test_raw"]["vstoxx_level"]
        .reindex(test_dates)
        .ffill().bfill()
    )

    # Align VSTOXX to trajectory dates (episodes start at lb-1, so 85 steps)
    traj_dates_idx = pd.DatetimeIndex(traj_drl["dates"])
    vstoxx_traj    = vstoxx.reindex(traj_dates_idx).ffill().bfill().values
    T              = len(traj_drl["FR"])

    # Regime threshold configurations
    k_configs = [
        (2, [25.0],       ["Low (<25)",     "High (>=25)"]),
        (3, [20.0, 30.0], ["Low (<20)",     "Med (20-30)",  "High (>=30)"]),
        (4, [15.0, 25.0, 35.0], ["VLow (<15)", "Low (15-25)", "Med (25-35)", "High (>=35)"]),
    ]

    all_k_results = {}

    for K, thresholds, labels in k_configs:
        print(f"\n  K={K}  thresholds={thresholds}")

        # Build masks
        edges = [-np.inf] + thresholds + [np.inf]
        masks = []
        for lo, hi, lbl in zip(edges[:-1], edges[1:], labels):
            mask = (vstoxx_traj >= lo) & (vstoxx_traj < hi)
            masks.append((lbl, mask))

        k_result = {}
        header_printed = False

        for lbl, mask in masks:
            n = int(mask.sum())
            if n < 2:
                print(f"    {lbl:<16} n={n:2d}  (too few months)")
                continue

            pi_sub = pi_test[: T][mask] if len(pi_test) >= T else pi_test[mask]

            def _sub_metrics(traj, mask=mask, pi_sub=pi_sub):
                sub = {
                    k: (np.asarray(v)[mask]
                        if isinstance(v, (np.ndarray, list)) else v)
                    for k, v in traj.items()
                    if k != "dates"
                }
                sub["dates"] = [d for d, m in zip(traj["dates"], mask) if m]
                return compute_metrics(sub, pi_monthly=pi_sub)

            m_drl   = _sub_metrics(traj_drl)
            m_fixed = _sub_metrics(traj_fixed)

            if not header_printed:
                print(f"    {'Regime':<16}  {'n':>4}  "
                      f"{'DRL MDD':>8}  {'FR MDD':>8}  "
                      f"{'DRL Dep':>8}  {'FR Dep':>8}  "
                      f"{'DRL Cal':>8}  {'FR Cal':>8}")
                print(f"    {'-'*16}  {'-'*4}  " + "  ".join(["-"*8]*6))
                header_printed = True

            print(f"    {lbl:<16}  {n:4d}  "
                  f"{m_drl['fr_mdd']:8.4f}  {m_fixed['fr_mdd']:8.4f}  "
                  f"{m_drl['buffer_depletion_freq']:8.4f}  "
                  f"{m_fixed['buffer_depletion_freq']:8.4f}  "
                  f"{m_drl['calmar_ratio']:8.4f}  "
                  f"{m_fixed['calmar_ratio']:8.4f}")

            k_result[lbl] = {"DRL": m_drl, "Fixed-Rule": m_fixed, "n_months": n}

        all_k_results[f"K={K}"] = k_result

    return all_k_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Robustness checks for Wtp DRL run_007",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", type=str,
                   default="src/models/run_007/best_model.zip")
    p.add_argument("--log-dir",    type=str,
                   default="src/models/run_007")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--skip-tc",    action="store_true",
                   help="Skip transaction-cost check")
    p.add_argument("--skip-blend", action="store_true",
                   help="Skip liability-blend check")
    p.add_argument("--only-reward-weights", action="store_true",
                   help="Run only the reward-weight sensitivity check")
    p.add_argument("--only-blend", action="store_true",
                   help="Run only the liability-blend sensitivity check")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args    = parse_args(argv)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Wtp DRL Pension Fund -- Robustness Analysis")
    print("=" * 64)
    print(f"  Model      : {args.model_path}")
    print(f"  Test period: Jan 2018 - Dec 2025")

    # ---- 1. Load data ------------------------------------------------------ #
    print("\n[1/3] Loading data pipeline...")
    results = run_pipeline()

    # Read reward weights and tc_bps from the model's train_config.json so the
    # robustness environment exactly matches the training configuration.
    train_cfg_path = Path(args.model_path).parent / "train_config.json"
    _env_overrides: dict = {}
    if train_cfg_path.exists():
        with open(train_cfg_path) as f:
            _tcfg = json.load(f)
        # Fields that map directly to EnvConfig attributes
        _field_map = {
            "tc_bps":          "tc_bps",
            "alpha":           "alpha",
            "beta":            "beta",
            "gamma_depletion": "gamma",      # YAML key → EnvConfig field
            "delta":           "delta",
            "fill_bonus":      "fill_bonus",
            "epsilon_equity":  "epsilon_equity",
            "dist_weight":     "dist_weight",
        }
        for yaml_key, cfg_field in _field_map.items():
            if yaml_key in _tcfg and _tcfg[yaml_key] is not None:
                _env_overrides[cfg_field] = float(_tcfg[yaml_key])
        if _env_overrides:
            print(f"  [train_config] Loaded env overrides: "
                  + "  ".join(f"{k}={v}" for k, v in _env_overrides.items()))

    env_cfg = EnvConfig(**_env_overrides)

    test_dates = results["z_test"].index
    pi_test = (
        results["cpi"]["pi_monthly"]
        .reindex(test_dates)
        .fillna(0.0)
        .values
    )

    # ---- 2. Load model ----------------------------------------------------- #
    print("\n[2/3] Loading DRL model...")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train.py first."
        )

    dummy_env = make_env_from_pipeline(results, split="test", cfg=env_cfg,
                                       seed=args.seed)
    drl_model = PPO.load(
        str(model_path),
        env=dummy_env,
        custom_objects={
            "policy_class":  WtpActorCriticPolicy,
            "policy_kwargs": {"wtp_cfg": AgentConfig()},
        },
    )
    drl_agent   = _SB3Adapter(drl_model)
    fixed_agent = FixedRuleALM()
    print(f"  Loaded: {model_path}")

    # Hoevenaars ALM — mirrors the loading logic in evaluate.py.
    # Uses the v4 headline policy calibrated on the training period.
    hoev_agent = None
    if _HoevenaarsALM is not None:
        _hoev_path = _ROOT / "results" / "hoevenaars_v4_headline_policy.json"
        if _hoev_path.exists():
            with open(_hoev_path) as _hf:
                _hl = json.load(_hf)["params"]
            _hoev_params = np.array([
                _hl["a"], _hl["h"], _hl["i"], _hl["c"],
                _hl["g"], _hl["e"], _hl["B_target"], _hl["B_min"],
            ], dtype=np.float64)
            _mu_vstoxx = float(np.percentile(
                results["z_train_raw"]["vstoxx_level"].values, 67
            ))
            hoev_agent = _HoevenaarsALM(
                params        = _hoev_params,
                mu_vstoxx     = _mu_vstoxx,
                vstoxx_series = results["z_test"]["vstoxx_level"].values,
            )
            print(f"  Hoevenaars ALM loaded  (mu_vstoxx={_mu_vstoxx:.2f})")
        else:
            print("  Hoevenaars ALM skipped "
                  "(results/hoevenaars_v4_headline_policy.json not found)")

    # ---- 3. Run checks ----------------------------------------------------- #
    print("\n[3/3] Running robustness checks...")

    all_results = {}

    if args.only_reward_weights:
        all_results["reward_weights"] = check_reward_weights(
            drl_agent, fixed_agent, results, pi_test, env_cfg=env_cfg
        )
    elif args.only_blend:
        all_results["liability_blend"] = check_liability_blend(
            drl_agent, fixed_agent, results, pi_test,
            hoev_agent=hoev_agent, env_cfg=env_cfg,
        )
    else:
        all_results["initial_conditions"] = check_initial_conditions(
            drl_agent, results, env_cfg, pi_test
        )

        if not args.skip_tc:
            all_results["transaction_costs"] = check_transaction_costs(
                drl_agent, results, env_cfg, pi_test
            )

        if not args.skip_blend:
            all_results["liability_blend"] = check_liability_blend(
                drl_agent, fixed_agent, results, pi_test,
                hoev_agent=hoev_agent, env_cfg=env_cfg,
            )

        all_results["dnb_stress"] = check_dnb_stress(
            drl_agent, results, env_cfg, pi_test
        )

        all_results["reward_weights"] = check_reward_weights(
            drl_agent, fixed_agent, results, pi_test, env_cfg=env_cfg
        )

        all_results["regime_k"] = check_regime_k(
            drl_agent, fixed_agent, results, pi_test, env_cfg=env_cfg
        )

    # ---- Save -------------------------------------------------------------- #
    def _serialise(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    out_path = log_dir / "robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_serialise)
    print(f"\n  Results saved: {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
