"""visualize.py — Visualise data features and out-of-sample evaluation results.

Produces 10 figure files saved to figures/:
  fig1_data_overview.png      — Market data: VSTOXX regimes, equity returns, yields
  fig2_fr_trajectories.png    — Funding ratio paths for all three agents
  fig3_buffer_dist.png        — Buffer level + cumulative distributions
  fig4_drl_actions.png        — DRL agent action breakdown over test period
  fig5_regime_metrics.png     — Regime-conditional bar charts
  fig6_cohort_rr.png          — Rolling 12-month real replacement rates per cohort
  fig7_3d_state_space.png     — 3D: FR x Buffer x Time state-space trajectory
  fig8_3d_policy_surface.png  — 3D: VSTOXX x FR -> equity weight (learned policy)
  fig9_3d_cohort_surface.png  — 3D: Time x Cohort -> rolling real replacement rate
  fig10_3d_regime_bars.png    — 3D: Regime x Agent -> key metrics bar chart

Usage
-----
    py -3 visualize.py
    py -3 visualize.py --out-dir results/figures
    py -3 visualize.py --no-mc          # skip Monte Carlo (faster)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.data_pipeline import run_pipeline
from src.environment   import make_env_from_pipeline, EnvConfig
from src.agent         import AgentConfig, WtpActorCriticPolicy
from src.baselines     import FixedRuleALM, MonteCarloALM, run_episode

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    raise ImportError("stable-baselines3 required: pip install stable-baselines3") from exc

# ── Colour palette ──────────────────────────────────────────────────────────
C = {
    "drl":    "#2563EB",   # blue
    "fixed":  "#DC2626",   # red
    "mc":     "#16A34A",   # green
    "low":    "#86EFAC",   # light green
    "med":    "#FCD34D",   # amber
    "high":   "#FCA5A5",   # light red
    "fr":     "#374151",   # dark grey
    "buf":    "#7C3AED",   # purple
    "dist":   "#EA580C",   # orange
}

AGENT_LABELS = {"DRL (PPO)": "DRL (PPO)", "Fixed-Rule": "Fixed-Rule ALM", "Monte Carlo": "Monte Carlo ALM"}
AGENT_COLORS = {"DRL (PPO)": C["drl"],    "Fixed-Rule": C["fixed"],        "Monte Carlo": C["mc"]}


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

class _SB3Adapter:
    def __init__(self, model) -> None:
        self._model = model
    def predict(self, obs):
        action, _ = self._model.predict(obs, deterministic=True)
        return action


def _shade_regimes(ax, vstoxx: pd.Series, dates: pd.DatetimeIndex) -> None:
    """Shade Low/Med/High volatility regimes in background."""
    thresholds = (20.0, 30.0)
    vs = vstoxx.reindex(dates).ffill().values
    in_regime = np.digitize(vs, thresholds)   # 0=Low, 1=Med, 2=High
    regime_colors = [C["low"], C["med"], C["high"]]
    i = 0
    while i < len(dates):
        j = i
        while j < len(dates) and in_regime[j] == in_regime[i]:
            j += 1
        ax.axvspan(dates[i], dates[min(j, len(dates)-1)],
                   color=regime_colors[in_regime[i]], alpha=0.25, lw=0)
        i = j


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ────────────────────────────────────────────────────────────────────────────
# Figure 1 — Data overview
# ────────────────────────────────────────────────────────────────────────────

def fig_data_overview(results: dict, out: Path) -> None:
    z_raw     = pd.concat([results["z_train_raw"], results["z_val_raw"], results["z_test_raw"]])
    test_start = results["z_test_raw"].index[0]
    val_start  = results["z_val_raw"].index[0]

    dates = z_raw.index

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Market Data Overview (Jan 2000 – Dec 2025)", fontsize=13, fontweight="bold")

    # ── Panel 1: VSTOXX ─────────────────────────────────────────────────────
    ax = axes[0]
    vstoxx = z_raw["vstoxx_level"]
    ax.fill_between(dates, vstoxx, alpha=0.4, color=C["high"])
    ax.plot(dates, vstoxx, color=C["high"], lw=0.8)
    ax.axhline(20, color="grey", ls="--", lw=0.8, label="Low/Med threshold (20)")
    ax.axhline(30, color="grey", ls=":",  lw=0.8, label="Med/High threshold (30)")
    ax.set_ylabel("VSTOXX")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("VSTOXX Volatility Index")

    # ── Panel 2: MSCI World equity return ───────────────────────────────────
    ax = axes[1]
    eq_ret = z_raw["mom_msci_1m"]
    ax.bar(dates, eq_ret * 100, width=20, color=np.where(eq_ret >= 0, C["drl"], C["fixed"]), alpha=0.7)
    ax.set_ylabel("Return (%)")
    ax.set_title("MSCI World 1M Momentum (Log Return)")

    # ── Panel 3: Swap curve slope ────────────────────────────────────────────
    ax = axes[2]
    slope_30_10 = z_raw["slope_30y_10y"]
    slope_10_2  = z_raw["slope_10y_2y"]
    ax.plot(dates, slope_30_10 * 100, color=C["mc"],    lw=1.0, label="Swap 30Y-10Y")
    ax.plot(dates, slope_10_2  * 100, color=C["fixed"], lw=1.0, label="Swap 10Y-2Y")
    ax.axhline(0, color="black", lw=0.6, ls="-")
    ax.set_ylabel("Spread (bps)")
    ax.legend(fontsize=8)
    ax.set_title("Swap Curve Slope")

    # ── Panel 4: Dutch CPI ───────────────────────────────────────────────────
    ax = axes[3]
    pi = results["cpi"]["pi_monthly"].reindex(dates).ffill() * 100
    ax.plot(dates, pi, color=C["dist"], lw=1.2)
    ax.set_ylabel("Monthly CPI (%)")
    ax.set_title("Dutch CPI (Monthly, Annualised Equivalent)")

    # ── Train/Val/Test shading ───────────────────────────────────────────────
    for ax in axes:
        ax.axvspan(val_start,  test_start, color="lightyellow", alpha=0.5, zorder=0)
        ax.axvspan(test_start, dates[-1],  color="lightcyan",   alpha=0.5, zorder=0)
        ax.axvline(val_start,  color="orange", lw=1.0, ls="--")
        ax.axvline(test_start, color="steelblue", lw=1.0, ls="--")

    # Legend for periods
    from matplotlib.patches import Patch
    period_patches = [
        Patch(color="white",      label="Training (2000-2015)"),
        Patch(color="lightyellow",label="Validation (2016-2017)"),
        Patch(color="lightcyan",  label="Test (2018-2025)"),
    ]
    axes[0].legend(handles=period_patches + axes[0].get_legend_handles_labels()[0],
                   fontsize=8, loc="upper left")

    fig.tight_layout()
    _save(fig, out)


# ────────────────────────────────────────────────────────────────────────────
# Figure 2 — FR trajectories
# ────────────────────────────────────────────────────────────────────────────

def fig_fr_trajectories(trajectories: dict, dates: pd.DatetimeIndex,
                        vstoxx: pd.Series, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Funding Ratio — Test Period (Jan 2018 – Dec 2025)",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: FR level ───────────────────────────────────────────────────
    ax = axes[0]
    _shade_regimes(ax, vstoxx, dates)
    for name, traj in trajectories.items():
        ax.plot(dates[:len(traj["FR"])], traj["FR"],
                color=AGENT_COLORS[name], lw=1.8, label=AGENT_LABELS[name])
    ax.axhline(1.043, color="black", ls=":",  lw=1.2, label="MVEV floor (1.043)")
    ax.axhline(1.050, color="grey",  ls="--", lw=1.0, label="FR target (1.05)")
    ax.axhline(1.000, color="black", ls="-",  lw=0.6)
    ax.set_ylabel("Funding Ratio")
    ax.legend(fontsize=9)
    ax.set_title("Funding Ratio Level")
    ax.set_ylim(0.85, 2.05)

    # ── Panel 2: FR drawdown ────────────────────────────────────────────────
    ax = axes[1]
    _shade_regimes(ax, vstoxx, dates)
    for name, traj in trajectories.items():
        fr  = np.array(traj["FR"])
        T   = len(fr)
        peak = np.maximum.accumulate(fr)
        dd   = (fr - peak) / peak
        ax.plot(dates[:T], dd * 100, color=AGENT_COLORS[name], lw=1.5,
                label=AGENT_LABELS[name])
    ax.fill_between(dates[:T], 0, -5, color="tomato", alpha=0.1)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.set_title("Funding Ratio Drawdown")

    fig.tight_layout()
    _save(fig, out)


# ────────────────────────────────────────────────────────────────────────────
# Figure 3 — Buffer level + cumulative distributions
# ────────────────────────────────────────────────────────────────────────────

def fig_buffer_dist(trajectories: dict, dates: pd.DatetimeIndex,
                    vstoxx: pd.Series, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Solidarity Buffer & Distributions — Test Period",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Buffer level ───────────────────────────────────────────────
    ax = axes[0]
    _shade_regimes(ax, vstoxx, dates)
    for name, traj in trajectories.items():
        buf = np.array(traj["B"])
        ax.plot(dates[:len(buf)], buf * 100, color=AGENT_COLORS[name],
                lw=1.8, label=AGENT_LABELS[name])
    ax.axhline(0,   color="black", lw=0.8)
    ax.axhline(15,  color="grey",  ls="--", lw=1.0, label="Buffer cap (15%)")
    ax.axhline(0.1, color="tomato", ls=":", lw=1.0, label="Depletion threshold (0.1%)")
    ax.set_ylabel("Buffer Level (%)")
    ax.legend(fontsize=9)
    ax.set_title("Solidarity Buffer Level (Art. 10d)")
    ax.set_ylim(-0.5, 17)

    # ── Panel 2: Cumulative distributions ───────────────────────────────────
    ax = axes[1]
    _shade_regimes(ax, vstoxx, dates)
    for name, traj in trajectories.items():
        dists = np.array(traj["d_tilde"])
        T = len(dists)
        ax.plot(dates[:T], np.cumsum(dists) * 100, color=AGENT_COLORS[name],
                lw=1.8, label=AGENT_LABELS[name])
    ax.set_ylabel("Cumulative Distributions (%)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.set_title("Cumulative Distributions to Members")

    fig.tight_layout()
    _save(fig, out)


# ────────────────────────────────────────────────────────────────────────────
# Figure 4 — DRL agent actions
# ────────────────────────────────────────────────────────────────────────────

def fig_drl_actions(traj: dict, dates: pd.DatetimeIndex,
                    vstoxx: pd.Series, out: Path) -> None:
    w_eq   = np.array(traj["w_eq"])
    f_til  = np.array(traj["f_tilde"])
    d_til  = np.array(traj["d_tilde"])
    T      = len(w_eq)
    d      = dates[:T]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("DRL Agent Policy — Test Period (Jan 2018 – Dec 2025)",
                 fontsize=13, fontweight="bold")

    series  = [w_eq,   f_til,  d_til]
    titles  = ["Equity Weight (w_eq)", "Fill Rate Applied (f̃_t)", "Distribution Rate Applied (d̃_t)"]
    colors  = [C["drl"], C["mc"], C["dist"]]
    refs    = [0.60,   0.03,   0.02]
    ref_lbl = ["Fixed-Rule w_eq=60%", "Fixed-Rule f=3%", "Fixed-Rule d=2%"]
    ylims   = [(0.28, 0.92), (-0.002, 0.11), (-0.002, 0.055)]

    for i, ax in enumerate(axes):
        _shade_regimes(ax, vstoxx, d)
        ax.plot(d, series[i], color=colors[i], lw=1.4, alpha=0.9)
        ax.axhline(refs[i], color="grey", ls="--", lw=0.9, label=ref_lbl[i])
        ax.set_ylabel(titles[i], fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(ylims[i])

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, out)


# ────────────────────────────────────────────────────────────────────────────
# Figure 5 — Regime-conditional bar charts
# ────────────────────────────────────────────────────────────────────────────

def fig_regime_metrics(trajectories: dict, vstoxx: pd.Series,
                       pi_test: np.ndarray, out: Path) -> None:
    from src.metrics import regime_conditional_metrics

    regime_data = {
        name: regime_conditional_metrics(traj, vstoxx, pi_monthly=pi_test)
        for name, traj in trajectories.items()
    }

    metrics_to_plot = [
        ("buffer_depletion_freq", "Buffer Depletion Freq",   "Frequency"),
        ("total_distributions",   "Total Distributions",     "Sum (fraction)"),
        ("fr_mdd",                "FR Max Drawdown",         "Drawdown"),
        ("calmar_ratio",          "Calmar Ratio",            "Calmar"),
    ]
    regimes   = ["Low", "Medium", "High"]
    agents    = list(trajectories.keys())
    n_agents  = len(agents)
    bar_w     = 0.25
    x         = np.arange(len(regimes))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Regime-Conditional Metrics (Test Period)",
                 fontsize=13, fontweight="bold")

    for ax, (key, title, ylabel) in zip(axes.flat, metrics_to_plot):
        for i, agent in enumerate(agents):
            vals = [regime_data[agent].get(r, {}).get(key, 0.0) for r in regimes]
            bars = ax.bar(x + i * bar_w, vals,
                          width=bar_w, color=AGENT_COLORS[agent],
                          label=AGENT_LABELS[agent], alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + bar_w * (n_agents - 1) / 2)
        ax.set_xticklabels(regimes)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

        # colour backgrounds per regime
        for j, color in enumerate([C["low"], C["med"], C["high"]]):
            ax.axvspan(j - 0.5, j + 0.5, color=color, alpha=0.12, zorder=0)

    fig.tight_layout()
    _save(fig, out)


# ────────────────────────────────────────────────────────────────────────────
# Figure 6 — Cohort rolling replacement rates
# ────────────────────────────────────────────────────────────────────────────

def fig_cohort_rr(trajectories: dict, dates: pd.DatetimeIndex,
                  pi_test: np.ndarray, vstoxx: pd.Series, out: Path) -> None:
    lookback = 12

    def rolling_rr(cohort_returns: np.ndarray, pi: np.ndarray) -> np.ndarray:
        T = len(cohort_returns)
        rr = np.full(T, np.nan)
        for t in range(lookback, T):
            window = cohort_returns[t - lookback:t]
            pi_w   = pi[t - lookback:t]
            rr[t]  = np.sum(np.log(1 + np.clip(window, -0.5, 0.5) - pi_w))
        return rr

    cohort_names = ["Young (i=1)", "Mid-career (i=2)", "Retired (i=3)"]
    cohort_colors = ["#1D4ED8", "#7C3AED", "#B45309"]

    fig, axes = plt.subplots(len(trajectories), 1,
                             figsize=(14, 4 * len(trajectories)), sharex=True)
    if len(trajectories) == 1:
        axes = [axes]
    fig.suptitle("Rolling 12-Month Real Replacement Rates by Cohort",
                 fontsize=13, fontweight="bold")

    for ax, (agent, traj) in zip(axes, trajectories.items()):
        _shade_regimes(ax, vstoxx, dates)
        T = min(len(traj["FR"]), len(dates))
        pi = pi_test[:T]

        # Derive cohort returns from trajectory per CLAUDE.md spec:
        #   Young:    r_p  (portfolio return)
        #   Mid:      (FR_t - FR_{t-1}) / FR_{t-1}
        #   Retired:  d_tilde / 3
        fr_arr = np.array(traj["FR"])[:T]
        rp_arr = np.array(traj["r_p"])[:T]
        dt_arr = np.array(traj["d_tilde"])[:T]
        fr_prev = np.concatenate([[fr_arr[0]], fr_arr[:-1]])
        cohort_returns_list = [
            rp_arr,
            (fr_arr - fr_prev) / np.maximum(fr_prev, 1e-6),
            dt_arr / 3.0,
        ]
        for ci, ret in enumerate(cohort_returns_list):
            rr  = rolling_rr(ret, pi)
            ax.plot(dates[:T], rr, color=cohort_colors[ci], lw=1.4,
                    label=cohort_names[ci])

        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(f"{AGENT_LABELS.get(agent, agent)}", fontweight="bold")
        ax.set_ylabel("Rolling Real RR (log)")
        ax.legend(fontsize=9)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, out)


# ────────────────────────────────────────────────────────────────────────────
# Figure 7 — 3D state-space trajectory: FR x Buffer x Time
# ────────────────────────────────────────────────────────────────────────────

def fig_3d_state_space(trajectories: dict, dates: pd.DatetimeIndex, out: Path) -> None:
    """3D line plot: each agent's path through (FR, Buffer, Time) state space.

    X = Funding Ratio, Y = Buffer Level, Z = time index.
    Colour gradient along the line shows time progression.
    """
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("3D State-Space Trajectory: FR x Buffer Level x Time",
                 fontsize=13, fontweight="bold")

    n_agents = len(trajectories)
    for idx, (name, traj) in enumerate(trajectories.items()):
        ax = fig.add_subplot(1, n_agents, idx + 1, projection="3d")

        fr  = np.array(traj["FR"])
        buf = np.array(traj["B"]) * 100   # convert to %
        T   = len(fr)
        t   = np.arange(T)

        # Draw line segments coloured by time
        from matplotlib.collections import LineCollection
        norm = plt.Normalize(0, T)
        cmap = plt.cm.viridis

        for i in range(T - 1):
            color = cmap(norm(i))
            ax.plot(fr[i:i+2], buf[i:i+2], t[i:i+2],
                    color=color, lw=1.4, alpha=0.85)

        # Mark start and end
        ax.scatter([fr[0]],  [buf[0]],  [t[0]],  color="green",  s=50,
                   zorder=5, label="Start")
        ax.scatter([fr[-1]], [buf[-1]], [t[-1]], color="red",    s=50,
                   zorder=5, label="End")

        # MVEV floor plane
        fr_plane  = np.array([1.043, 1.043])
        buf_plane = np.array([0, 15])
        t_plane   = np.array([0, T])
        FR_p, T_p = np.meshgrid(fr_plane, t_plane)
        B_p       = np.full_like(FR_p, 7.5)
        ax.plot_surface(FR_p, B_p * 0 + np.linspace(0, 15, 2)[np.newaxis, :],
                        T_p, alpha=0.0)   # invisible — just for reference

        ax.set_xlabel("Funding Ratio", fontsize=8, labelpad=6)
        ax.set_ylabel("Buffer (%)",    fontsize=8, labelpad=6)
        ax.set_zlabel("Month",         fontsize=8, labelpad=6)
        ax.set_title(AGENT_LABELS.get(name, name), fontweight="bold", fontsize=10)
        ax.axvlines = ax.plot([1.043, 1.043], [0, 15], [0, 0],
                              color="tomato", ls="--", lw=1.0, alpha=0.6,
                              label="MVEV 1.043")
        ax.legend(fontsize=7, loc="upper left")

        # Colour bar for time
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label("Month index", fontsize=7)
        tick_pos = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        cbar.set_ticks(tick_pos)
        cbar.set_ticklabels([str(dates[p].year) for p in tick_pos])

    fig.tight_layout()
    _save(fig, out)


# ────────────────────────────────────────────────────────────────────────────
# Figure 8 — 3D DRL policy surface: VSTOXX x FR -> equity weight
# ────────────────────────────────────────────────────────────────────────────

def fig_3d_policy_surface(traj_drl: dict, vstoxx_test: pd.Series,
                          dates: pd.DatetimeIndex, out: Path) -> None:
    """3D scatter + interpolated surface: learned policy w_eq = f(VSTOXX, FR).

    Shows how the DRL agent adapts equity allocation to the joint state of
    market volatility (VSTOXX) and pension fund solvency (FR).
    """
    from scipy.interpolate import griddata

    fr   = np.array(traj_drl["FR"])
    w_eq = np.array(traj_drl["w_eq"])
    T    = len(fr)
    vs   = vstoxx_test.reindex(dates[:T]).ffill().values

    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection="3d")
    fig.suptitle("3D DRL Policy Surface: VSTOXX x Funding Ratio -> Equity Weight",
                 fontsize=13, fontweight="bold")

    # Scatter actual observations, coloured by time
    sc = ax.scatter(vs, fr, w_eq, c=np.arange(T), cmap="plasma",
                    s=18, alpha=0.7, depthshade=True)

    # Interpolated surface over a regular grid
    vs_grid  = np.linspace(vs.min(),   vs.max(),   40)
    fr_grid  = np.linspace(fr.min(),   fr.max(),   40)
    VS, FR   = np.meshgrid(vs_grid, fr_grid)
    W        = griddata((vs, fr), w_eq, (VS, FR), method="linear")

    surf = ax.plot_surface(VS, FR, W, cmap="coolwarm", alpha=0.35,
                           linewidth=0, antialiased=True)

    # Reference planes
    ax.plot_surface(VS, FR, np.full_like(W, 0.55),
                    color="grey", alpha=0.10, linewidth=0)

    # Regime threshold lines on the base
    for vth, col, lbl in [(20, "green", "Low/Med (20)"),
                           (30, "orange", "Med/High (30)")]:
        ax.plot([vth, vth], [fr.min(), fr.max()], [w_eq.min(), w_eq.min()],
                color=col, ls="--", lw=1.5, label=lbl)

    ax.set_xlabel("VSTOXX",         fontsize=9, labelpad=8)
    ax.set_ylabel("Funding Ratio",  fontsize=9, labelpad=8)
    ax.set_zlabel("Equity Weight",  fontsize=9, labelpad=8)
    ax.set_title("Each point = one test month;\nsurface = interpolated policy",
                 fontsize=9)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.45, pad=0.12)
    cbar.set_label("Month index (time)", fontsize=8)
    tick_pos = [0, T // 3, 2 * T // 3, T - 1]
    cbar.set_ticks(tick_pos)

    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, out)


# ────────────────────────────────────────────────────────────────────────────
# Figure 9 — 3D cohort replacement rate surface
# ────────────────────────────────────────────────────────────────────────────

def fig_3d_cohort_surface(trajectories: dict, dates: pd.DatetimeIndex,
                          pi_test: np.ndarray, out: Path) -> None:
    """3D surface: Time x Cohort -> rolling 12M real replacement rate.

    For each agent, shows how intergenerational equity evolves over time —
    a flat surface means all cohorts earn equal real returns (ideal equity).
    """
    lookback = 12

    def rolling_rr(cohort_returns: np.ndarray, pi: np.ndarray) -> np.ndarray:
        T  = len(cohort_returns)
        rr = np.full(T, np.nan)
        for t in range(lookback, T):
            window = cohort_returns[t - lookback:t]
            pi_w   = pi[t - lookback:t]
            rr[t]  = np.sum(np.log(1 + np.clip(window, -0.5, 0.5) - pi_w))
        return rr

    cohort_labels = ["Young", "Mid-career", "Retired"]

    fig = plt.figure(figsize=(15, 5 * len(trajectories)))
    fig.suptitle("3D Intergenerational Equity Surface: Time x Cohort -> Real Replacement Rate",
                 fontsize=13, fontweight="bold")

    for idx, (name, traj) in enumerate(trajectories.items()):
        ax = fig.add_subplot(len(trajectories), 1, idx + 1, projection="3d")

        T      = min(len(traj["FR"]), len(dates))
        pi     = pi_test[:T]
        fr_arr = np.array(traj["FR"])[:T]
        rp_arr = np.array(traj["r_p"])[:T]
        dt_arr = np.array(traj["d_tilde"])[:T]
        fr_prev = np.concatenate([[fr_arr[0]], fr_arr[:-1]])

        cohort_rets = [
            rp_arr,
            (fr_arr - fr_prev) / np.maximum(fr_prev, 1e-6),
            dt_arr / 3.0,
        ]

        t_idx = np.arange(T)
        c_idx = np.array([0, 1, 2])
        T_grid, C_grid = np.meshgrid(t_idx, c_idx)

        RR = np.full((3, T), np.nan)
        for ci, ret in enumerate(cohort_rets):
            RR[ci] = rolling_rr(ret, pi)

        # Surface
        surf = ax.plot_surface(T_grid, C_grid, RR,
                               cmap="RdYlGn", alpha=0.80,
                               linewidth=0.2, edgecolor="none")

        # Zero plane
        ax.plot_surface(T_grid, C_grid, np.zeros_like(RR),
                        color="grey", alpha=0.12, linewidth=0)

        ax.set_xlabel("Month",              fontsize=8, labelpad=6)
        ax.set_ylabel("Cohort",             fontsize=8, labelpad=6)
        ax.set_zlabel("Real RR (log-sum)",  fontsize=8, labelpad=6)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(cohort_labels, fontsize=7)

        # X-axis: show years
        tick_months = [i for i in range(0, T, 12)]
        ax.set_xticks(tick_months)
        ax.set_xticklabels([str(dates[i].year) for i in tick_months], fontsize=7)

        ax.set_title(AGENT_LABELS.get(name, name), fontweight="bold", fontsize=10)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.12)
        cbar.set_label("Real RR", fontsize=7)

    fig.tight_layout()
    _save(fig, out)


# ────────────────────────────────────────────────────────────────────────────
# Figure 10 — 3D regime-conditional metric bar chart
# ────────────────────────────────────────────────────────────────────────────

def fig_3d_regime_bars(trajectories: dict, vstoxx_test: pd.Series,
                       pi_test: np.ndarray, out: Path) -> None:
    """3D grouped bar chart: Regime x Agent -> multiple metrics.

    Two sub-plots: Buffer Depletion Frequency and FR MDD.
    The 3D layout makes the joint regime/agent comparison immediately readable.
    """
    from src.metrics import regime_conditional_metrics

    regime_data = {
        name: regime_conditional_metrics(traj, vstoxx_test, pi_monthly=pi_test)
        for name, traj in trajectories.items()
    }

    regimes = ["Low", "Medium", "High"]
    agents  = list(trajectories.keys())
    metrics = [
        ("buffer_depletion_freq", "Buffer Depletion Frequency"),
        ("fr_mdd",                "FR Max Drawdown"),
    ]

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle("3D Regime-Conditional Performance: Regime x Agent x Metric",
                 fontsize=13, fontweight="bold")

    for plot_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = fig.add_subplot(1, 2, plot_idx + 1, projection="3d")

        bar_w  = 0.25
        bar_d  = 0.6
        colors = [AGENT_COLORS[a] for a in agents]

        for ai, (agent, color) in enumerate(zip(agents, colors)):
            for ri, regime in enumerate(regimes):
                val = regime_data[agent].get(regime, {}).get(metric_key, 0.0)
                x0  = ri - bar_d / 2
                y0  = ai * bar_w
                # Draw bar as a 3D box
                verts = [
                    [(x0,       y0,       0),
                     (x0+bar_d, y0,       0),
                     (x0+bar_d, y0+bar_w, 0),
                     (x0,       y0+bar_w, 0)],   # bottom
                    [(x0,       y0,       0),
                     (x0+bar_d, y0,       0),
                     (x0+bar_d, y0,       val),
                     (x0,       y0,       val)],  # front face
                    [(x0,       y0+bar_w, 0),
                     (x0+bar_d, y0+bar_w, 0),
                     (x0+bar_d, y0+bar_w, val),
                     (x0,       y0+bar_w, val)],  # back face
                    [(x0,       y0,       0),
                     (x0,       y0+bar_w, 0),
                     (x0,       y0+bar_w, val),
                     (x0,       y0,       val)],  # left face
                    [(x0+bar_d, y0,       0),
                     (x0+bar_d, y0+bar_w, 0),
                     (x0+bar_d, y0+bar_w, val),
                     (x0+bar_d, y0,       val)],  # right face
                    [(x0,       y0,       val),
                     (x0+bar_d, y0,       val),
                     (x0+bar_d, y0+bar_w, val),
                     (x0,       y0+bar_w, val)],  # top
                ]
                poly = Poly3DCollection(verts, alpha=0.75,
                                        facecolor=color, edgecolor="white", lw=0.4)
                ax.add_collection3d(poly)

                # Value label on top
                ax.text(x0 + bar_d / 2, y0 + bar_w / 2, val + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

        ax.set_xlim(-0.5, len(regimes) - 0.5)
        ax.set_ylim(-0.1, len(agents) * bar_w + 0.1)
        ax.set_zlim(0, None)

        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels(regimes, fontsize=8)
        ax.set_yticks([ai * bar_w + bar_w / 2 for ai in range(len(agents))])
        ax.set_yticklabels([AGENT_LABELS.get(a, a) for a in agents], fontsize=7)
        ax.set_zlabel(metric_label, fontsize=8, labelpad=6)
        ax.set_title(metric_label, fontweight="bold", fontsize=10)

        # Custom legend patches
        patches = [mpatches.Patch(color=AGENT_COLORS[a],
                                  label=AGENT_LABELS.get(a, a)) for a in agents]
        ax.legend(handles=patches, fontsize=7, loc="upper right")

    fig.tight_layout()
    _save(fig, out)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Visualise Wtp DRL results")
    p.add_argument("--model-path", default="src/models/run_001/best_model.zip")
    p.add_argument("--out-dir",    default="src/models/run_001/figures")
    p.add_argument("--no-mc",      action="store_true", help="Skip Monte Carlo ALM")
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args    = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Wtp DRL Pension Fund -- Visualisation")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────────── #
    print("\n[1/3] Loading data...")
    results  = run_pipeline()
    env_cfg  = EnvConfig()

    test_dates  = results["z_test"].index
    vstoxx_full = pd.concat([results["z_train_raw"], results["z_val_raw"],
                             results["z_test_raw"]])["vstoxx_level"]
    vstoxx_test = results["z_test_raw"]["vstoxx_level"]
    pi_test     = (
        results["cpi"]["pi_monthly"]
        .reindex(test_dates).fillna(0.0).values
    )

    # ── Fig 1: data overview (no episodes needed) ────────────────────────── #
    print("\n[2/3] Generating data overview figure...")
    fig_data_overview(results, out_dir / "fig1_data_overview.png")

    # ── Agents ──────────────────────────────────────────────────────────── #
    print("\n[3/3] Running agents on test set...")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run train.py first or pass --model-path."
        )

    env_drl   = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=args.seed)
    env_fixed = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=args.seed)

    drl_model = PPO.load(
        str(model_path), env=env_drl,
        custom_objects={"policy_class": WtpActorCriticPolicy,
                        "policy_kwargs": {"wtp_cfg": AgentConfig()}},
    )
    drl_agent   = _SB3Adapter(drl_model)
    fixed_agent = FixedRuleALM()

    print("  Running DRL agent...")
    traj_drl   = run_episode(drl_agent, env_drl)
    print("  Running Fixed-Rule ALM...")
    traj_fixed = run_episode(fixed_agent, env_fixed)

    trajectories = {"DRL (PPO)": traj_drl, "Fixed-Rule": traj_fixed}

    if not args.no_mc:
        env_mc   = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=args.seed)
        mc_agent = MonteCarloALM()
        print("  Fitting Monte Carlo ALM...")
        mc_agent.fit(results["z_train_raw"], results["raw_train"], results["cpi"], env_cfg)
        print("  Running Monte Carlo ALM...")
        traj_mc  = run_episode(mc_agent, env_mc)
        trajectories["Monte Carlo"] = traj_mc

    # ── Figures 2-6 (2D) ────────────────────────────────────────────────── #
    print("\n  Generating 2D result figures...")
    fig_fr_trajectories(trajectories, test_dates, vstoxx_test,
                        out_dir / "fig2_fr_trajectories.png")
    fig_buffer_dist(trajectories, test_dates, vstoxx_test,
                    out_dir / "fig3_buffer_dist.png")
    fig_drl_actions(traj_drl, test_dates, vstoxx_test,
                    out_dir / "fig4_drl_actions.png")
    fig_regime_metrics(trajectories, vstoxx_test, pi_test,
                       out_dir / "fig5_regime_metrics.png")
    fig_cohort_rr(trajectories, test_dates, pi_test, vstoxx_test,
                  out_dir / "fig6_cohort_rr.png")

    # ── Figures 7-10 (3D) ───────────────────────────────────────────────── #
    print("\n  Generating 3D figures...")
    fig_3d_state_space(trajectories, test_dates,
                       out_dir / "fig7_3d_state_space.png")
    fig_3d_policy_surface(traj_drl, vstoxx_test, test_dates,
                          out_dir / "fig8_3d_policy_surface.png")
    fig_3d_cohort_surface(trajectories, test_dates, pi_test,
                          out_dir / "fig9_3d_cohort_surface.png")
    fig_3d_regime_bars(trajectories, vstoxx_test, pi_test,
                       out_dir / "fig10_3d_regime_bars.png")

    print(f"\nAll figures saved to: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
