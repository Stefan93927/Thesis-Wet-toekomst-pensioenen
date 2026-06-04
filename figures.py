"""figures.py — Thesis-quality figures for the Wtp DRL pension fund evaluation.

Figures generated
-----------------
1. fig1_trajectories.pdf     — FR, buffer, cumulative distributions over time
                                (DRL vs Fixed-Rule vs MC ALM, VSTOXX shading)
2. fig2_regime_bars.pdf      — Regime-conditional MDD, depletion, Calmar
3. fig3_ic_sensitivity.pdf   — Initial condition sensitivity (FR_init, B_init)
4. fig4_dnb_stress.pdf       — DNB stress scenario FR impact
5. fig5_robustness_grid.pdf  — TC + liability blend side-by-side
6. fig6_reward_weights.pdf   — Reward weight sensitivity (DRL advantage)
7. fig7_regime_k.pdf         — Regime count K sensitivity

Usage
-----
    py -3 figures.py
    py -3 figures.py --model-path src/models/run_007/best_model.zip
    py -3 figures.py --no-show   # save only, do not open windows
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.data_pipeline import run_pipeline
from src.environment   import WtpPensionEnv, EnvConfig, make_env_from_pipeline
from src.agent         import AgentConfig, WtpActorCriticPolicy
from src.baselines     import FixedRuleALM, MonteCarloALM, run_episode

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    raise ImportError("stable-baselines3 required: pip install stable-baselines3") from exc

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    # Font
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "font.size":            11,
    "axes.titlesize":       12,
    "axes.labelsize":       11,
    "legend.fontsize":      10,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    # Axes
    "axes.grid":            True,
    "axes.grid.axis":       "y",
    "grid.alpha":           0.25,
    "grid.linestyle":       "--",
    "grid.color":           "#cccccc",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.8,
    # Figure
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
    # Legend
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "#cccccc",
    "legend.borderpad":     0.5,
})

# Professional muted palette (colorblind-friendly)
C_DRL   = "#2166AC"   # steel blue
C_FIXED = "#B2182B"   # brick red
C_MC    = "#1A7E3C"   # forest green
C_BASE  = "#888888"   # neutral grey

# Subtle regime shading (very light, doesn't compete with data)
REGIME_COLORS = {
    "Low":    "#EAF3FB",   # very light blue
    "Medium": "#FFF8E7",   # very light amber
    "High":   "#FDECEA",   # very light red
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SB3Adapter:
    def __init__(self, model):
        self._model = model
    def predict(self, obs):
        a, _ = self._model.predict(obs, deterministic=True)
        return a


def _detect_n_regimes(model_path: str) -> int:
    """Infer K (number of GMM regimes) from the gating layer shape in the zip."""
    import zipfile, torch, io as _io
    _zip = model_path if model_path.endswith(".zip") else model_path + ".zip"
    try:
        with zipfile.ZipFile(_zip) as zf:
            with zf.open("policy.pth") as f:
                state = torch.load(_io.BytesIO(f.read()), map_location="cpu",
                                   weights_only=False)
        # gating linear bias shape is (K,)
        bias = state.get("wtp_net.gating.linear.bias")
        if bias is not None:
            return int(bias.shape[0])
    except Exception:
        pass
    return 3  # default


def _load_model(model_path: str, results: dict, env_cfg: EnvConfig) -> _SB3Adapter:
    dummy = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)
    _n = _detect_n_regimes(str(model_path))
    _cfg = AgentConfig(n_regimes=_n, gmm_n_regimes=_n,
                       beta_bar=([0.70, 0.55, 0.40, 0.25] if _n == 4
                                 else [0.65, 0.55, 0.35]))
    model = PPO.load(
        str(model_path),
        env=dummy,
        custom_objects={
            "policy_class":  WtpActorCriticPolicy,
            "policy_kwargs": {"wtp_cfg": _cfg},
        },
    )
    return _SB3Adapter(model)


def _run_trajectories(results, drl_agent, env_cfg):
    """Run DRL, Fixed-Rule, MC ALM on test set; return trajectories + metadata."""
    test_env_drl   = make_env_from_pipeline(results, "test", env_cfg, seed=0)
    test_env_fixed = make_env_from_pipeline(results, "test", env_cfg, seed=0)
    test_env_mc    = make_env_from_pipeline(results, "test", env_cfg, seed=0)

    fixed_agent = FixedRuleALM()

    print("  Running DRL agent...")
    traj_drl = run_episode(drl_agent, test_env_drl)

    print("  Running Fixed-Rule ALM...")
    traj_fixed = run_episode(fixed_agent, test_env_fixed)

    print("  Fitting + running Monte Carlo ALM...")
    mc_agent = MonteCarloALM()
    mc_agent.fit(results["z_train_raw"], results["raw_train"], results["cpi"], env_cfg)
    traj_mc = run_episode(mc_agent, test_env_mc)

    # Dates and VSTOXX
    dates = pd.DatetimeIndex(traj_drl["dates"])
    test_dates = results["z_test"].index
    vstoxx = (
        results["z_test_raw"]["vstoxx_level"]
        .reindex(test_dates).ffill().bfill()
        .reindex(dates).ffill().bfill()
    )

    # CPI aligned to trajectory
    pi = (
        results["cpi"]["pi_monthly"]
        .reindex(test_dates).fillna(0.0)
        .reindex(dates).fillna(0.0)
        .values
    )

    return {
        "DRL (PPO)":  traj_drl,
        "Fixed-Rule": traj_fixed,
        "MC ALM":     traj_mc,
    }, dates, vstoxx, pi


def _add_vstoxx_shading(ax, dates, vstoxx, thresholds=(20, 30), alpha=0.15):
    """Shade background by VSTOXX regime."""
    lo, hi = thresholds
    x = dates
    for i in range(len(x)):
        v = vstoxx.iloc[i] if hasattr(vstoxx, "iloc") else vstoxx[i]
        if v < lo:
            color = REGIME_COLORS["Low"]
        elif v < hi:
            color = REGIME_COLORS["Medium"]
        else:
            color = REGIME_COLORS["High"]
        if i < len(x) - 1:
            ax.axvspan(x[i], x[i + 1], color=color, alpha=alpha, linewidth=0)


def _save(fig, path: Path, show: bool):
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    print(f"  Saved: {path.stem}.pdf / .png")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Time-series trajectories
# ---------------------------------------------------------------------------

def _bar_labels(ax, fmt=".3f", fontsize=9, padding=4):
    """Add value labels on top of bar containers."""
    for container in ax.containers:
        ax.bar_label(container, fmt=f"{{:{fmt}}}", fontsize=fontsize,
                     padding=padding, label_type="edge")


def fig1_trajectories(trajectories, dates, vstoxx, out_dir, show=False):
    """4-panel: FR, buffer, cumulative distributions, VSTOXX level."""
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2, 2, 1.5]})
    fig.subplots_adjust(hspace=0.06)

    styles = {
        "DRL (PPO)":  (C_DRL,   "-",  2.0),
        "Fixed-Rule": (C_FIXED, "--", 1.5),
        "MC ALM":     (C_MC,    ":",  1.5),
    }

    # ---- Panel 1: FR -------------------------------------------------------- #
    ax = axes[0]
    _add_vstoxx_shading(ax, dates, vstoxx)
    for name, traj in trajectories.items():
        c, ls, lw = styles[name]
        ax.plot(dates, traj["FR"], color=c, ls=ls, lw=lw, label=name)
    ax.axhline(1.05,  color="grey",  ls="--", lw=0.8, label="FR target (1.05)")
    ax.axhline(1.043, color="black", ls=":",  lw=0.8, label="MVEV floor (1.043)")
    ax.set_ylabel("Funding Ratio")
    ax.set_title("Out-of-Sample Performance: Dec 2018 – Dec 2025")
    ax.legend(loc="upper left", ncol=3, framealpha=0.9)
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    # Annotate terminal FR values
    for name, traj in trajectories.items():
        c, _, _ = styles[name]
        ax.annotate(f"{traj['FR'][-1]:.3f}", xy=(dates[-1], traj["FR"][-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=c, fontsize=8, va="center")

    # ---- Panel 2: Buffer ---------------------------------------------------- #
    ax = axes[1]
    _add_vstoxx_shading(ax, dates, vstoxx)
    for name, traj in trajectories.items():
        c, ls, lw = styles[name]
        ax.plot(dates, traj["B"], color=c, ls=ls, lw=lw)
    ax.axhline(0.001, color="black", ls=":", lw=0.8, label="Depletion (0.1%)")
    ax.axhline(0.15,  color="grey",  ls="--", lw=0.8, label="B_max (15%)")
    ax.set_ylabel("Solidarity Buffer B")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.set_ylim(bottom=-0.005)

    # ---- Panel 3: Cumulative distributions ---------------------------------- #
    ax = axes[2]
    _add_vstoxx_shading(ax, dates, vstoxx)
    for name, traj in trajectories.items():
        c, ls, lw = styles[name]
        cum_dist = np.cumsum(traj["d_tilde"])
        ax.plot(dates, cum_dist, color=c, ls=ls, lw=lw)
        # Annotate total at end
        ax.annotate(f"{cum_dist[-1]:.3f}", xy=(dates[-1], cum_dist[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=c, fontsize=8, va="center")
    ax.set_ylabel("Cumul. Distributions")

    # Regime legend
    patches = [
        mpatches.Patch(color=REGIME_COLORS["Low"],    alpha=0.6, label="Low VSTOXX (<20)"),
        mpatches.Patch(color=REGIME_COLORS["Medium"], alpha=0.6, label="Med (20–30)"),
        mpatches.Patch(color=REGIME_COLORS["High"],   alpha=0.6, label="High (>30)"),
    ]
    ax.legend(handles=patches, loc="upper left", ncol=3, framealpha=0.9)

    # ---- Panel 4: VSTOXX level ---------------------------------------------- #
    ax = axes[3]
    vstoxx_vals = vstoxx.values if hasattr(vstoxx, "values") else np.array(vstoxx)
    ax.fill_between(dates, vstoxx_vals, alpha=0.4, color=C_BASE, step="mid")
    ax.plot(dates, vstoxx_vals, color=C_BASE, lw=1.0)
    ax.axhline(20, color=REGIME_COLORS["Medium"][0:7], ls="--", lw=0.8, alpha=0.8)
    ax.axhline(30, color=REGIME_COLORS["High"][0:7],   ls="--", lw=0.8, alpha=0.8)
    ax.set_ylabel("VSTOXX")
    ax.set_xlabel("Date")
    ax.set_ylim(bottom=0)
    # Annotate thresholds
    ax.text(dates[-1], 20, " 20", va="center", fontsize=7, color="grey")
    ax.text(dates[-1], 30, " 30", va="center", fontsize=7, color="grey")

    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    _save(fig, out_dir / "fig1_trajectories", show)


# ---------------------------------------------------------------------------
# Figure 2: Regime-conditional bars
# ---------------------------------------------------------------------------

def fig2_regime_bars(eval_json: dict, out_dir, show=False):
    """Grouped bar chart: regime-conditional MDD, depletion, Calmar."""
    agents  = ["DRL (PPO)", "Fixed-Rule", "Monte Carlo"]
    regimes = ["Low", "Medium", "High"]
    metrics = [
        ("fr_mdd",                "FR Max Drawdown",      0.0,  0.20),
        ("buffer_depletion_freq", "Buffer Depletion Freq", 0.0,  1.0),
        ("calmar_ratio",          "Calmar Ratio",          0.0, 10.0),
    ]

    colors = [C_DRL, C_FIXED, C_MC]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Regime-Conditional Metrics  (Test Period Jan 2019 – Dec 2025)",
                 fontsize=11, y=1.01)

    x     = np.arange(len(regimes))
    width = 0.25

    regime_data = eval_json.get("regime_metrics", {})

    for ax, (key, label, ylo, yhi) in zip(axes, metrics):
        for i, (agent, color) in enumerate(zip(agents, colors)):
            vals = []
            for reg in regimes:
                v = (regime_data.get(agent, {})
                                .get(reg,   {})
                                .get(key, float("nan")))
                vals.append(float(v) if not np.isnan(v) else 0.0)
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=agent, color=color, alpha=0.85, edgecolor="white",
                          linewidth=0.5)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)

        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.set_title(label)
        ax.set_ylim(ylo, yhi * 1.15)   # headroom for rotated labels
        if ax == axes[0]:
            ax.set_ylabel("Value")

    axes[1].legend(loc="upper right", ncol=1)
    fig.tight_layout()
    _save(fig, out_dir / "fig2_regime_bars", show)


# ---------------------------------------------------------------------------
# Figure 3: Initial condition sensitivity
# ---------------------------------------------------------------------------

def fig3_ic_sensitivity(rob_json: dict, out_dir, show=False):
    """2-row grid: metrics vs FR_init (top) and B_init (bottom)."""
    ic = rob_json.get("initial_conditions", {})
    fr_data = ic.get("fr_grid", {})
    b_data  = ic.get("b_grid",  {})

    metrics = [
        ("fr_terminal",           "FR Terminal"),
        ("fr_mdd",                "FR MDD"),
        ("buffer_depletion_freq", "Buf. Depl. Freq"),
        ("calmar_ratio",          "Calmar Ratio"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("Initial Condition Sensitivity  (DRL Agent)", fontsize=12, y=1.01)

    fr_labels = sorted(fr_data.keys())
    b_labels  = sorted(b_data.keys())
    fr_vals   = [float(lbl.split("=")[1]) for lbl in fr_labels]
    b_vals    = [float(lbl.split("=")[1]) for lbl in b_labels]

    for col, (key, title) in enumerate(metrics):
        # --- Row 0: vs FR_init ---
        ax = axes[0, col]
        vals = [fr_data[lbl].get(key, np.nan) for lbl in fr_labels]
        ax.plot(fr_vals, vals, "o-", color=C_DRL, lw=1.8, ms=6)
        ax.set_title(title)
        ax.set_xlabel("Initial FR")
        if col == 0:
            ax.set_ylabel("Metric value\n(varying FR$_{init}$)")

        # --- Row 1: vs B_init ---
        ax = axes[1, col]
        vals = [b_data[lbl].get(key, np.nan) for lbl in b_labels]
        ax.plot(b_vals, vals, "s--", color=C_MC, lw=1.8, ms=6)
        ax.set_xlabel("Initial Buffer B")
        if col == 0:
            ax.set_ylabel("Metric value\n(varying B$_{init}$)")

    # Row labels
    for row, label in enumerate(["Varying FR$_{init}$", "Varying B$_{init}$"]):
        axes[row, 0].annotate(
            label, xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", va="center", ha="center", rotation=90,
        )

    fig.tight_layout()
    _save(fig, out_dir / "fig3_ic_sensitivity", show)


# ---------------------------------------------------------------------------
# Figure 4: DNB stress scenarios
# ---------------------------------------------------------------------------

def fig4_dnb_stress(rob_json: dict, out_dir, show=False):
    """Horizontal bars: FR after shock (month 1) and terminal FR."""
    stress = rob_json.get("dnb_stress", {})
    names  = list(stress.keys())

    term_vals  = [stress[n]["fr_terminal"] for n in names]
    mdd_vals   = [stress[n]["fr_mdd"]      for n in names]
    base_term  = term_vals[0]

    delta_term = [v - base_term for v in term_vals]

    colors_bar = []
    for d in delta_term:
        if abs(d) < 1e-6:
            colors_bar.append(C_BASE)
        elif d < 0:
            colors_bar.append(C_FIXED)
        else:
            colors_bar.append(C_DRL)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("DNB Stress Scenarios  (Besluit FTK Art. 23 — shock at Jan 2019)",
                 fontsize=11)

    y = np.arange(len(names))

    # ---- Panel 1: Terminal FR ----------------------------------------------- #
    ax = axes[0]
    bars = ax.barh(y, term_vals, color=colors_bar, alpha=0.85, edgecolor="white")
    ax.axvline(base_term, color="grey", ls="--", lw=1.2, label=f"Baseline ({base_term:.2f})")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("FR Terminal (Dec 2025)")
    ax.set_title("Terminal Funding Ratio")
    ax.legend()
    # Value labels
    for bar, v in zip(bars, term_vals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=8)

    # ---- Panel 2: delta FR terminal ----------------------------------------- #
    ax = axes[1]
    bars2 = ax.barh(y, delta_term, color=colors_bar, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Delta FR Terminal vs Baseline")
    ax.set_title("Shock Impact on Terminal FR")

    # Annotate delta value + MDD
    for i, (d, mdd) in enumerate(zip(delta_term, mdd_vals)):
        sign = "+" if d >= 0 else ""
        axes[1].text(d + (0.005 if d >= 0 else -0.005), i,
                     f"{sign}{d:.3f}  MDD={mdd:.3f}", va="center",
                     ha="left" if d >= 0 else "right", fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir / "fig4_dnb_stress", show)


# ---------------------------------------------------------------------------
# Figure 5: TC + Liability blend robustness grid
# ---------------------------------------------------------------------------

def fig5_robustness_grid(rob_json: dict, out_dir, show=False):
    """2-row grid: transaction costs (top) and liability blend (bottom)."""
    tc_data    = rob_json.get("transaction_costs", {})
    blend_data = rob_json.get("liability_blend",   {})

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    fig.suptitle("Sensitivity: Transaction Costs (top) and Liability Blend (bottom)",
                 fontsize=11)

    metrics = [
        ("fr_terminal",           "FR Terminal"),
        ("buffer_depletion_freq", "Buf. Depletion Freq"),
        ("calmar_ratio",          "Calmar Ratio"),
    ]

    # ---- Row 1: Transaction costs ------------------------------------------- #
    tc_labels  = ["0 bps", "10 bps", "25 bps", "50 bps"]
    x_tc = np.arange(len(tc_labels))
    for ax, (key, title) in zip(axes[0], metrics):
        vals = [tc_data.get(lbl, {}).get(key, np.nan) for lbl in tc_labels]
        bars = ax.bar(x_tc, vals, color=C_DRL, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
        ax.set_xticks(x_tc)
        ax.set_xticklabels(tc_labels, rotation=0)
        ax.set_title(title)
        ax.set_xlabel("Transaction Cost")
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.12)
        if ax == axes[0][0]:
            ax.set_ylabel("Value (DRL agent)")
        # Mark baseline
        ax.axhline(vals[0], color=C_BASE, ls="--", lw=0.8)

    # ---- Row 2: Liability blend --------------------------------------------- #
    blend_labels = ["Pure UFR  (0%)", "50-50", "70-30 (base)", "Pure MtM (100%)"]
    x_bl = np.arange(len(blend_labels))
    for ax, (key, title) in zip(axes[1], metrics):
        vals = [blend_data.get(lbl, {}).get(key, np.nan) for lbl in blend_labels]
        # Colour base config differently
        bar_colors = [C_BASE if lbl == "70-30 (base)" else C_DRL
                      for lbl in blend_labels]
        bars = ax.bar(x_bl, vals, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
        ax.set_xticks(x_bl)
        ax.set_xticklabels(["UFR", "50-50", "70-30\n(base)", "MtM"], rotation=0)
        ax.set_xlabel("Liability Blend")
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.12)
        if ax == axes[1][0]:
            ax.set_ylabel("Value (DRL agent)")

    fig.tight_layout()
    _save(fig, out_dir / "fig5_robustness_grid", show)


# ---------------------------------------------------------------------------
# Figure 6: Reward weight sensitivity
# ---------------------------------------------------------------------------

def fig6_reward_weights(rob_json: dict, out_dir, show=False):
    """Horizontal bar chart: DRL advantage over Fixed-Rule per weight config."""
    rw = rob_json.get("reward_weights", {})
    labels    = list(rw.keys())
    adv_vals  = [rw[lbl]["advantage"]    for lbl in labels]
    drl_vals  = [rw[lbl]["drl_reward"]   for lbl in labels]
    fixed_vals = [rw[lbl]["fixed_reward"] for lbl in labels]

    base_adv = adv_vals[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Reward Weight Sensitivity  (one-at-a-time ±50%)", fontsize=11)

    y = np.arange(len(labels))
    short_labels = [
        lbl.replace("(no change)", "(base)")
           .replace("alpha", "α")
           .replace("beta",  "β")
           .replace("gamma", "γ")
           .replace("delta", "δ")
        for lbl in labels
    ]

    # ---- Panel 1: DRL advantage --------------------------------------------- #
    ax = axes[0]
    colors = [C_DRL if a >= base_adv else C_FIXED for a in adv_vals]
    ax.barh(y, adv_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(base_adv, color="grey", ls="--", lw=1.2, label=f"Base advantage ({base_adv:.1f})")
    ax.set_yticks(y)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("DRL advantage over Fixed-Rule\n(total episode reward)")
    ax.set_title("Reward Advantage")
    ax.legend()

    # ---- Panel 2: Total rewards side by side -------------------------------- #
    ax = axes[1]
    width = 0.35
    ax.barh(y - width/2, drl_vals,   width, color=C_DRL,   alpha=0.85, label="DRL",        edgecolor="white")
    ax.barh(y + width/2, fixed_vals, width, color=C_FIXED, alpha=0.85, label="Fixed-Rule",  edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Total Episode Reward")
    ax.set_title("Total Reward Comparison")
    ax.legend()

    fig.tight_layout()
    _save(fig, out_dir / "fig6_reward_weights", show)


# ---------------------------------------------------------------------------
# Figure 7: Regime K sensitivity
# ---------------------------------------------------------------------------

def fig7_regime_k(rob_json: dict, out_dir, show=False):
    """Grouped bars: depletion and Calmar for DRL vs Fixed-Rule under K=2,3,4."""
    rk = rob_json.get("regime_k", {})

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Regime Count K Sensitivity  (VSTOXX threshold partitioning)",
                 fontsize=11)

    k_specs = [
        ("K=2", ["Low (<25)", "High (>=25)"]),
        ("K=3", ["Low (<20)", "Med (20-30)", "High (>=30)"]),
        ("K=4", ["VLow (<15)", "Low (15-25)", "Med (25-35)", "High (>=35)"]),
    ]

    for ax, (metric_key, metric_label) in zip(
        axes,
        [("buffer_depletion_freq", "Buffer Depletion Frequency"),
         ("calmar_ratio",          "Calmar Ratio")],
    ):
        x_offset = 0
        xtick_pos = []
        xtick_lbl = []
        group_centers = []
        group_labels  = []

        for k_label, regimes in k_specs:
            k_data = rk.get(k_label, {})
            n_reg  = len(regimes)
            group_start = x_offset
            x_reg = np.arange(n_reg) + x_offset
            width = 0.35

            drl_vals   = [k_data.get(r, {}).get("DRL",        {}).get(metric_key, np.nan) for r in regimes]
            fixed_vals = [k_data.get(r, {}).get("Fixed-Rule",  {}).get(metric_key, np.nan) for r in regimes]
            ns         = [k_data.get(r, {}).get("n_months", 0) for r in regimes]

            ax.bar(x_reg - width/2, drl_vals,   width, color=C_DRL,   alpha=0.85, edgecolor="white",
                   label="DRL"        if k_label == "K=2" else "_")
            ax.bar(x_reg + width/2, fixed_vals, width, color=C_FIXED, alpha=0.85, edgecolor="white",
                   label="Fixed-Rule" if k_label == "K=2" else "_")

            for i, (r, n) in enumerate(zip(regimes, ns)):
                short = r.split("(")[0].strip()
                xtick_pos.append(x_offset + i)
                xtick_lbl.append(f"{short}\n(n={n})")

            group_centers.append((group_start + x_offset + n_reg - 1) / 2)
            group_labels.append(k_label)
            x_offset += n_reg + 1   # gap between K groups

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lbl, fontsize=8)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)

        # K group labels above
        ylim = ax.get_ylim()
        for cx, clbl in zip(group_centers, group_labels):
            ax.text(cx, ylim[1] * 0.97, clbl, ha="center", va="top",
                    fontsize=9, fontweight="bold", color="dimgrey")

        # Vertical separators
        x_off = 0
        for _, regimes in k_specs[:-1]:
            x_off += len(regimes)
            ax.axvline(x_off + 0.5, color="lightgrey", lw=1.0)
            x_off += 1

    axes[0].legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir / "fig7_regime_k", show)


# ---------------------------------------------------------------------------
# Figure 8: Action statistics for DRL agent
# ---------------------------------------------------------------------------

def fig8_action_stats(trajectories, dates, vstoxx, out_dir, show=False):
    """3-panel: equity weight, portfolio vs liability return, fill/dist rates."""
    drl_traj   = trajectories["DRL (PPO)"]
    fixed_traj = trajectories["Fixed-Rule"]

    w_eq_drl   = np.array(drl_traj["w_eq"])
    w_eq_fixed = np.array(fixed_traj["w_eq"])
    r_p_drl    = np.array(drl_traj["r_p"])
    r_p_fixed  = np.array(fixed_traj["r_p"])
    r_L_drl    = np.array(drl_traj["r_L"])
    f_t = np.array(drl_traj["f_tilde"])
    d_t = np.array(drl_traj["d_tilde"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.suptitle("DRL Agent: Actions and Returns  (Test Period)", fontsize=11)
    fig.subplots_adjust(hspace=0.08)

    # ---- Panel 1: Equity weight --------------------------------------------- #
    ax = axes[0]
    _add_vstoxx_shading(ax, dates, vstoxx)
    ax.plot(dates, w_eq_drl,   color=C_DRL,   lw=1.5, label="DRL w_eq")
    ax.plot(dates, w_eq_fixed, color=C_FIXED, lw=1.0, ls="--", label="Fixed-Rule w_eq")
    ax.axhline(0.55, color="grey",  ls="--", lw=0.8, label="Strategic base (55%)")
    ax.axhline(0.30, color="black", ls=":",  lw=0.7)
    ax.axhline(0.90, color="black", ls=":",  lw=0.7, label="Bounds (30%/90%)")
    ax.set_ylabel("Equity Weight")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.set_ylim(0.25, 0.98)
    # Annotate mean
    ax.text(dates[5], 0.28,
            f"DRL mean={w_eq_drl.mean():.3f}  std={w_eq_drl.std():.3f}",
            fontsize=8, color=C_DRL)

    # ---- Panel 2: Monthly portfolio vs liability return --------------------- #
    ax = axes[1]
    _add_vstoxx_shading(ax, dates, vstoxx)
    ax.bar(dates, r_p_drl,  width=20, color=C_DRL,   alpha=0.6,
           label="DRL r_p", align="center")
    ax.plot(dates, r_L_drl, color="black", lw=1.2, ls="--",
            label="Liability return r_L")
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_ylabel("Monthly Return")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    # Annotate surplus months
    surplus = r_p_drl - r_L_drl
    n_pos = (surplus > 0).sum()
    ax.text(dates[5], ax.get_ylim()[0] * 0.85 if ax.get_ylim()[0] < 0 else 0.01,
            f"Outperforms liability in {n_pos}/{len(surplus)} months",
            fontsize=8, color=C_DRL)

    # ---- Panel 3: Fill and distribution rates -------------------------------- #
    ax = axes[2]
    _add_vstoxx_shading(ax, dates, vstoxx)
    ax.bar(dates, f_t, width=20, color=C_DRL,   alpha=0.7,
           label="Fill rate f\u0303_t",         align="center")
    ax.bar(dates, d_t, width=20, color=C_FIXED, alpha=0.7,
           label="Distribution rate d\u0303_t", align="center", bottom=f_t)
    ax.set_ylabel("Rate")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.text(dates[5], ax.get_ylim()[1] * 0.85 if len(ax.get_ylim()) else 0.04,
            f"Total fills={f_t.sum():.3f}  Total dist={d_t.sum():.3f}",
            fontsize=8, color="dimgrey")

    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    _save(fig, out_dir / "fig8_action_stats", show)


# ---------------------------------------------------------------------------
# Figure 9: Multi-seed reproducibility
# ---------------------------------------------------------------------------

def fig9_multiseed(ms_json: dict, out_dir: Path, show: bool = False):
    """4-panel bar chart: mean +/- std across seeds vs Fixed-Rule.

    Panels: FR Terminal, FR MDD, Calmar Ratio, Buffer Depletion Frequency.
    Error bars = +/- 1 std across seeds.  Fixed-Rule shown as dashed line.

    Args:
        ms_json:  Dict loaded from ``multiseed_results.json``.
        out_dir:  Output directory.
        show:     Display interactively.
    """
    agg       = ms_json["aggregate"]
    fixed_m   = ms_json.get("fixed_rule", {})
    n_seeds   = agg["n_seeds"]
    seeds     = agg["seeds"]

    panels = [
        ("fr_terminal",           "FR Terminal",              "higher is better"),
        ("fr_mdd",                "FR Max Drawdown",          "lower is better"),
        ("calmar_ratio",          "Calmar Ratio",             "higher is better"),
        ("buffer_depletion_freq", "Buffer Depletion Freq",    "lower is better"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    fig.suptitle(
        f"Multi-Seed Reproducibility  (n={n_seeds} seeds: {', '.join(str(s) for s in seeds)})",
        fontsize=11,
    )

    per_seed = agg.get("per_seed", {})

    for ax, (key, label, direction) in zip(axes, panels):
        # Per-seed values as scatter
        seed_vals = [per_seed[str(s)]["metrics"][key] for s in seeds if str(s) in per_seed]
        if not seed_vals:
            seed_vals = [per_seed[s]["metrics"][key] for s in per_seed]

        mean_val  = agg["mean"][key]
        std_val   = agg["std"][key]
        fixed_val = fixed_m.get(key, None)

        # Bar for mean
        ax.bar(
            [0], [mean_val],
            width=0.5,
            color=C_DRL, alpha=0.85, edgecolor="white",
            yerr=[[std_val], [std_val]],
            capsize=6,
            error_kw={"elinewidth": 1.5, "ecolor": "black"},
            label="DRL mean +/- std",
        )

        # Individual seed dots
        x_jitter = np.linspace(-0.08, 0.08, len(seed_vals))
        ax.scatter(
            x_jitter, seed_vals,
            color=C_DRL, edgecolors="white", s=40, zorder=5, alpha=0.9,
            label="Per-seed value",
        )

        # Fixed-Rule line
        if fixed_val is not None:
            ax.axhline(fixed_val, color=C_FIXED, ls="--", lw=1.5, label="Fixed-Rule")
            ax.text(
                0.97, fixed_val,
                f"Fixed: {fixed_val:.3f}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom",
                fontsize=8, color=C_FIXED,
            )

        # Mean annotation
        ax.text(
            0, mean_val + std_val * 1.15,
            f"{mean_val:.3f}\n(+/-{std_val:.3f})",
            ha="center", va="bottom", fontsize=8.5, color="black",
        )

        ax.set_title(label, fontsize=10)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks([0])
        ax.set_xticklabels([f"DRL\n(n={n_seeds})"])
        ax.text(
            0.98, 0.02, direction,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="grey", style="italic",
        )

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    _save(fig, out_dir / "fig9_multiseed", show)


# ---------------------------------------------------------------------------
# Figure 10: PPV trajectories (CPI-deflated)
# ---------------------------------------------------------------------------

def fig10_ppv_trajectories(trajectories, pi, out_dir, show=False):
    """3-panel: real PPV trajectories per cohort, all models overlaid."""
    cpi_index = np.cumprod(1.0 + pi)

    cohorts = [
        ("Young",   "ppv_young", "#2166AC"),
        ("Mid",     "ppv_mid",   "#FF9800"),
        ("Retired", "ppv_ret",   "#4CAF50"),
    ]
    line_styles = {
        "DRL (PPO)":  ("-",  2.0),
        "Fixed-Rule": ("--", 1.5),
        "MC ALM":     (":",  1.5),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(
        "Personal Pension Capital (PPV) — Real (CPI-Deflated)\nTest Period: Jan 2018 – Dec 2025",
        fontsize=12, fontweight="bold",
    )

    for ax, (cohort_name, ppv_key, color) in zip(axes, cohorts):
        for model_name, traj in trajectories.items():
            if ppv_key not in traj:
                continue
            T        = len(traj[ppv_key])
            ppv_real = np.asarray(traj[ppv_key]) / cpi_index[:T]
            dates    = pd.to_datetime(traj["dates"])
            ls, lw   = line_styles.get(model_name, ("-", 1.5))
            ax.plot(dates, ppv_real, color=color, ls=ls, lw=lw, label=model_name)
            # Annotate terminal value
            ax.annotate(
                f"{ppv_real[-1]:.3f}",
                xy=(dates[-1], ppv_real[-1]),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=8, va="center",
            )

        ax.axhline(1.0, color="grey", lw=0.8, ls=":", label="Initial (1.00)")
        ax.set_title(f"{cohort_name} Cohort", fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel("Real PPV")
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    fig.tight_layout()
    _save(fig, out_dir / "fig10_ppv_trajectories", show)


# ---------------------------------------------------------------------------
# Figure 11: PPV attribution decomposition
# ---------------------------------------------------------------------------

def fig11_ppv_attribution(trajectories, env_cfg, out_dir, show=False):
    """Stacked bar: log-additive attribution of ln(PPV_T) per cohort × model."""
    cohorts   = ["Young", "Mid", "Retired"]
    rp_keys   = ["r_p_young", "r_p_mid", "r_p_ret"]
    ppv_keys  = ["ppv_young", "ppv_mid", "ppv_ret"]

    model_names = list(trajectories.keys())
    n_m = len(model_names)

    invest = np.zeros((n_m, 3))
    fills  = np.zeros((n_m, 3))
    dists  = np.zeros((n_m, 3))

    for m_i, (mname, traj) in enumerate(trajectories.items()):
        T         = len(traj["FR"])
        f_fill    = np.asarray(traj.get("f_tilde",    np.zeros(T)), dtype=np.float64)
        d_tilde   = np.asarray(traj["d_tilde"],  dtype=np.float64)
        dec_exc   = np.asarray(traj.get("dec_excess", np.zeros(T)), dtype=np.float64)
        total_dist = d_tilde + dec_exc

        for c_i, rp_key in enumerate(rp_keys):
            r_inv = np.asarray(traj.get(rp_key, traj["r_p"]), dtype=np.float64)
            invest[m_i, c_i] = float(np.sum(np.log(np.maximum(1.0 + r_inv,      1e-8))))
            fills [m_i, c_i] = float(np.sum(np.log(np.maximum(1.0 - f_fill,     1e-8))))
            dists [m_i, c_i] = float(np.sum(np.log(np.maximum(1.0 + total_dist, 1e-8))))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle(
        "PPV Attribution Decomposition — Log-Additive Contributions to ln(PPV_T)\n"
        "Test Period: Jan 2018 – Dec 2025",
        fontsize=12, fontweight="bold",
    )

    x     = np.arange(n_m)
    width = 0.55

    for c_i, (cohort, ax) in enumerate(zip(cohorts, axes)):
        inv_c  = invest[:, c_i]
        fill_c = fills[:, c_i]
        dist_c = dists[:, c_i]

        b1 = ax.bar(x, inv_c,  width=width, label="Investment return",  color="#2196F3", alpha=0.85)
        b2 = ax.bar(x, fill_c, width=width, bottom=inv_c, label="Buffer fill (cost)", color="#F44336", alpha=0.85)
        b3 = ax.bar(x, dist_c, width=width, bottom=inv_c + fill_c,
                    label="Distributions", color="#4CAF50", alpha=0.85)

        # Terminal PPV annotation
        for m_i, (mname, traj) in enumerate(trajectories.items()):
            ppv_key = ppv_keys[c_i]
            ppv_T   = float(traj[ppv_key][-1]) if ppv_key in traj else float("nan")
            total_ln = inv_c[m_i] + fill_c[m_i] + dist_c[m_i]
            ax.text(m_i, total_ln + 0.02, f"PPV={ppv_T:.3f}",
                    ha="center", va="bottom", fontsize=8, color="black")

        ax.set_title(f"{cohort} Cohort", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("ln(PPV_T) contribution" if c_i == 0 else "")
        ax.axhline(0, color="black", lw=0.8)
        if c_i == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir / "fig11_ppv_attribution", show)


# ---------------------------------------------------------------------------
# Figure 12: Distribution pattern per model
# ---------------------------------------------------------------------------

def fig12_distributions(trajectories, out_dir, show=False):
    """Monthly distribution bars per model, stacked with December excess."""
    n = len(trajectories)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Monthly Solidarity Buffer Distributions\nTest Period: Jan 2018 – Dec 2025",
        fontsize=12, fontweight="bold",
    )

    for ax, (model_name, traj) in zip(axes, trajectories.items()):
        dates = pd.to_datetime(traj["dates"])
        dist  = np.asarray(traj["d_tilde"])
        dec   = np.asarray(traj.get("dec_excess", np.zeros(len(dist))))
        total = dist + dec
        n_nonzero = int((total > 1e-4).sum())

        ax.bar(dates, dist, width=20, color=C_DRL,   alpha=0.85, label="d_tilde (regular)")
        ax.bar(dates, dec,  width=20, color=C_MC,    alpha=0.70, label="Dec. cap excess",
               bottom=dist)
        ax.set_title(
            f"{model_name}  |  {n_nonzero}/{len(dist)} distributing months  |  "
            f"total={total.sum():.4f}",
            fontsize=10,
        )
        ax.set_ylabel("Distribution rate")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Date")
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    fig.tight_layout()
    _save(fig, out_dir / "fig12_distributions", show)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate thesis figures for Wtp DRL evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path",  type=str,
                   default="src/models/run_016/best_model.zip")
    p.add_argument("--results-dir", type=str,
                   default=None,
                   help="Directory with eval_results.json (default: model's parent dir)")
    p.add_argument("--out-dir",     type=str,
                   default="figures")
    p.add_argument("--lifecycle",   action="store_true", default=False,
                   help="Use lifecycle EnvConfig (must match training; required for run_011)")
    p.add_argument("--no-show",     action="store_true",
                   help="Save figures without displaying them")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args    = parse_args(argv)
    show    = not args.no_show
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_dir = Path(args.results_dir) if args.results_dir else Path(args.model_path).parent
    eval_json_path = res_dir / "eval_results.json"
    rob_json_path  = res_dir / "robustness_results.json"

    print("=" * 64)
    print("Wtp DRL Pension Fund -- Figure Generation")
    print("=" * 64)
    print(f"  Model      : {args.model_path}")
    print(f"  Results    : {res_dir}")
    print(f"  Output dir : {out_dir}/")

    # ---- Load JSON results -------------------------------------------------- #
    print("\n[1/4] Loading evaluation results...")
    if not eval_json_path.exists():
        raise FileNotFoundError(f"Run evaluate.py first: {eval_json_path}")

    with open(eval_json_path) as f: eval_json = json.load(f)

    rob_json = None
    if rob_json_path.exists():
        with open(rob_json_path) as f: rob_json = json.load(f)
    else:
        print(f"  Note: robustness_results.json not found — skipping figures 3-7")

    # ---- Load data + model -------------------------------------------------- #
    print("\n[2/4] Loading data pipeline and model...")
    results = run_pipeline()
    env_cfg = EnvConfig(use_lifecycle=args.lifecycle)
    print(f"  use_lifecycle: {env_cfg.use_lifecycle}")
    drl_agent = _load_model(args.model_path, results, env_cfg)

    # ---- Run trajectories --------------------------------------------------- #
    print("\n[3/4] Running test episodes for trajectory plots...")
    trajectories, dates, vstoxx, pi = _run_trajectories(results, drl_agent, env_cfg)

    # Align CPI to trajectory dates
    pi_aligned = (
        results["cpi"]["pi_monthly"]
        .reindex(results["z_test"].index).fillna(0.0)
        .reindex(dates).fillna(0.0)
        .values
    )

    # ---- Generate figures --------------------------------------------------- #
    print("\n[4/4] Generating figures...")

    print("\n  Figure 1: FR / buffer / distributions trajectory")
    fig1_trajectories(trajectories, dates, vstoxx, out_dir, show)

    print("  Figure 2: Regime-conditional metric bars")
    fig2_regime_bars(eval_json, out_dir, show)

    if rob_json is not None:
        print("  Figure 3: Initial condition sensitivity")
        fig3_ic_sensitivity(rob_json, out_dir, show)

        print("  Figure 4: DNB stress scenarios")
        fig4_dnb_stress(rob_json, out_dir, show)

        print("  Figure 5: TC + liability blend grid")
        fig5_robustness_grid(rob_json, out_dir, show)

        print("  Figure 6: Reward weight sensitivity")
        fig6_reward_weights(rob_json, out_dir, show)

        print("  Figure 7: Regime K sensitivity")
        fig7_regime_k(rob_json, out_dir, show)
    else:
        print("  Figures 3-7: skipped (no robustness_results.json)")

    print("  Figure 8: DRL action statistics")
    fig8_action_stats(trajectories, dates, vstoxx, out_dir, show)

    # Optional: multi-seed figure (only if multiseed_results.json exists)
    ms_path = res_dir / ".." / "multiseed_results.json"
    if not ms_path.exists():
        ms_path = Path(args.results_dir).parent / "multiseed_results.json"
    if ms_path.exists():
        print("  Figure 9: Multi-seed reproducibility")
        with open(ms_path) as _f:
            ms_json = json.load(_f)
        fig9_multiseed(ms_json, out_dir, show)
    else:
        print("  Figure 9: skipped (run multiseed.py first to generate multiseed_results.json)")

    # PPV framework figures (run_011+, lifecycle mode)
    if env_cfg.use_lifecycle:
        has_ppv = any("ppv_young" in traj for traj in trajectories.values())
        if has_ppv:
            print("\n  Figure 10: PPV trajectories (CPI-deflated)")
            fig10_ppv_trajectories(trajectories, pi_aligned, out_dir, show)

            print("  Figure 11: PPV attribution decomposition")
            fig11_ppv_attribution(trajectories, env_cfg, out_dir, show)
        else:
            print("  Figures 10-11: skipped (no PPV data in trajectories)")

    print("\n  Figure 12: Monthly distributions pattern")
    fig12_distributions(trajectories, out_dir, show)

    print(f"\n  All figures saved to: {out_dir}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
