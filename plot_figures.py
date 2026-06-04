"""plot_figures.py — Comprehensive thesis figures.

Figures produced
----------------
fig_fr_paths.pdf        FR paths + MVEV floor + target (main body)
fig_fr_drawdown.pdf     Running FR drawdown (main body)
fig_cohort_rr.pdf       Rolling 12M real RR per cohort, 3 panels (main body)
fig_rr_variance.pdf     Cross-cohort RR variance over time (main body)
fig_rr_boxplot.pdf      Cohort RR distribution box plots (main body)
fig_fan_chart.pdf       Multi-seed FR fan chart — active + degenerate (robustness)

Usage
-----
    py -3 plot_figures.py --model-path src/models/run_007/best_model.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.data_pipeline import run_pipeline
from src.environment   import make_env_from_pipeline, EnvConfig
from src.agent         import AgentConfig, WtpActorCriticPolicy
from src.baselines     import FixedRuleALM, MonteCarloALM, run_episode

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    raise ImportError("pip install stable-baselines3") from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MVEV_FLOOR  = 1.043
FR_TARGET   = 1.05
LOOKBACK    = 12
W_RET       = 0.45
DEGENERATE  = {2, 8, 12}

C_DRL   = "#1f77b4"
C_FIXED = "#ff7f0e"
C_MC    = "#2ca02c"
C_YOUNG = "#9467bd"
C_MID   = "#8c564b"
C_RET   = "#e377c2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SB3Adapter:
    def __init__(self, model) -> None:
        self._model = model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action


def load_drl(path: str, env, cfg=None):
    cfg = cfg or AgentConfig()
    model = PPO.load(
        path, env=env,
        custom_objects={"policy_class": WtpActorCriticPolicy,
                        "policy_kwargs": {"wtp_cfg": cfg}},
    )
    return _SB3Adapter(model)


def rolling_cohort_rr(traj: dict, pi: np.ndarray, lookback: int = LOOKBACK):
    """Return (dates, rr_young, rr_mid, rr_ret, rr_var) as arrays."""
    FR   = np.asarray(traj["FR"])
    r_p  = np.asarray(traj["r_p"])
    dist = np.asarray(traj["d_tilde"])
    T    = len(FR)

    FR_prev = np.concatenate([[FR[0]], FR[:-1]])
    r_young = r_p
    r_mid   = (FR - FR_prev) / np.maximum(FR_prev, 1e-8)
    r_ret   = dist / W_RET

    dates_out, rr_y, rr_m, rr_r, rr_v = [], [], [], [], []
    all_dates = pd.DatetimeIndex(traj["dates"])

    for t in range(lookback, T):
        vals = []
        for ret in (r_young, r_mid, r_ret):
            log_sum = sum(
                np.log(max(1.0 + ret[t - lookback + k] - pi[t - lookback + k], 1e-8))
                for k in range(lookback)
            )
            vals.append(log_sum)
        dates_out.append(all_dates[t])
        rr_y.append(vals[0])
        rr_m.append(vals[1])
        rr_r.append(vals[2])
        rr_v.append(float(np.var(vals)))

    return (pd.DatetimeIndex(dates_out),
            np.array(rr_y), np.array(rr_m),
            np.array(rr_r), np.array(rr_v))


def fmt_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def save(fig, out_dir: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / name}.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="src/models/run_007/best_model.zip")
    p.add_argument("--no-mc",      action="store_true")
    p.add_argument("--out-dir",    default="src/models/run_007")
    args    = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ---------------------------------------------------------------
    print("Loading data pipeline...")
    results = run_pipeline()
    env_cfg = EnvConfig()

    test_dates = results["z_test"].index
    pi_test = (results["cpi"]["pi_monthly"]
               .reindex(test_dates).fillna(0.0).values)

    # ---- Main agents --------------------------------------------------------
    print("Loading main agents...")
    env_drl   = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)
    env_fixed = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)
    env_mc    = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)

    drl_agent   = load_drl(args.model_path, env_drl)
    fixed_agent = FixedRuleALM()

    mc_agent = None
    if not args.no_mc:
        mc_agent = MonteCarloALM()
        mc_agent.fit(results["z_train_raw"], results["raw_train"],
                     results["cpi"], env_cfg)

    print("Running main episodes...")
    traj_drl   = run_episode(drl_agent,   env_drl)
    traj_fixed = run_episode(fixed_agent, env_fixed)
    traj_mc    = run_episode(mc_agent, env_mc) if mc_agent else None

    dates = pd.DatetimeIndex(traj_drl["dates"])
    FR_drl   = np.array(traj_drl["FR"])
    FR_fixed = np.array(traj_fixed["FR"])
    FR_mc    = np.array(traj_mc["FR"]) if traj_mc else None

    # =========================================================================
    # FIGURE 1 — FR paths
    # =========================================================================
    print("Building FR paths figure...")
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(dates, FR_drl,   color=C_DRL,   lw=2,   label="DRL (PPO)")
    ax.plot(dates, FR_fixed, color=C_FIXED, lw=2,   label="Fixed-Rule")
    if FR_mc is not None:
        ax.plot(dates, FR_mc, color=C_MC, lw=2, label="Monte Carlo")

    ax.axhline(MVEV_FLOOR, color="red",    lw=1.2, ls="--",
               label=f"MVEV floor ({MVEV_FLOOR})")
    ax.axhline(FR_TARGET,  color="grey",   lw=1.0, ls=":",
               label=f"FR target ({FR_TARGET})")

    ax.fill_between(dates, MVEV_FLOOR, ax.get_ylim()[0] if ax.get_ylim()[0] < MVEV_FLOOR else 0,
                    alpha=0.06, color="red")

    fmt_xaxis(ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Funding Ratio $\\mathrm{FR}_t$")
    ax.set_title("Funding Ratio — Test Period (Jan 2018 – Dec 2025)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, "fig_fr_paths")
    plt.close(fig)

    # =========================================================================
    # FIGURE 2 — FR drawdown
    # =========================================================================
    print("Building FR drawdown figure...")
    fig, ax = plt.subplots(figsize=(12, 4))

    for FR, label, color in [
        (FR_drl,   "DRL (PPO)",   C_DRL),
        (FR_fixed, "Fixed-Rule",  C_FIXED),
    ] + ([(FR_mc, "Monte Carlo", C_MC)] if FR_mc is not None else []):
        peak = np.maximum.accumulate(FR)
        dd   = (peak - FR) / np.maximum(peak, 1e-8)
        ax.plot(dates, dd, color=color, lw=2, label=label)

    fmt_xaxis(ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("FR Drawdown")
    ax.set_title("Funding Ratio Drawdown — Test Period (Jan 2018 – Dec 2025)")
    ax.invert_yaxis()
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, "fig_fr_drawdown")
    plt.close(fig)

    # =========================================================================
    # FIGURE 3 — Rolling cohort RR per model (3 panels)
    # =========================================================================
    print("Building rolling cohort RR figure...")

    agents_rr = [("DRL (PPO)", traj_drl, C_DRL)]
    if traj_mc:
        agents_rr += [("Fixed-Rule", traj_fixed, C_FIXED),
                      ("Monte Carlo", traj_mc,    C_MC)]
    else:
        agents_rr += [("Fixed-Rule", traj_fixed, C_FIXED)]

    ncols = len(agents_rr)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, (name, traj, _) in zip(axes, agents_rr):
        d, ry, rm, rr, _ = rolling_cohort_rr(traj, pi_test)
        ax.plot(d, ry, color=C_YOUNG, lw=1.8, label="Young")
        ax.plot(d, rm, color=C_MID,   lw=1.8, label="Mid-career")
        ax.plot(d, rr, color=C_RET,   lw=1.8, label="Retired")
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        fmt_xaxis(ax)
        ax.set_title(name)
        ax.set_xlabel("Date")
        if ax is axes[0]:
            ax.set_ylabel("Rolling 12M real RR (log)")
        ax.legend(framealpha=0.9, fontsize=9)

    fig.suptitle("Rolling 12-Month Real Replacement Rates by Cohort",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, out_dir, "fig_cohort_rr")
    plt.close(fig)

    # =========================================================================
    # FIGURE 4 — Cross-cohort RR variance over time
    # =========================================================================
    print("Building RR variance figure...")
    fig, ax = plt.subplots(figsize=(12, 4))

    for name, traj, color in agents_rr:
        d, _, _, _, rr_v = rolling_cohort_rr(traj, pi_test)
        ax.plot(d, rr_v, color=color, lw=2, label=name)

    fmt_xaxis(ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cross-cohort RR variance $E_t$")
    ax.set_title("Cross-Cohort RR Variance Over Time (lower = more equitable)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, "fig_rr_variance")
    plt.close(fig)

    # =========================================================================
    # FIGURE 5 — Cohort RR box plots
    # =========================================================================
    print("Building RR box plot figure...")
    fig, ax = plt.subplots(figsize=(10, 5))

    all_data    = []
    all_labels  = []
    all_colors  = []
    positions   = []
    pos = 1

    for name, traj, agent_color in agents_rr:
        _, ry, rm, rr, _ = rolling_cohort_rr(traj, pi_test)
        for cohort_data, cohort_label, cohort_color in [
            (ry, "Young",      C_YOUNG),
            (rm, "Mid",        C_MID),
            (rr, "Retired",    C_RET),
        ]:
            all_data.append(cohort_data)
            all_labels.append(f"{cohort_label}\n{name}")
            all_colors.append(cohort_color)
            positions.append(pos)
            pos += 1
        pos += 0.5   # gap between models

    bp = ax.boxplot(all_data, positions=positions, patch_artist=True,
                    widths=0.7, medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, fontsize=8)
    ax.set_ylabel("Rolling 12M real RR (log)")
    ax.set_title("Cohort Replacement Rate Distributions — Test Period")

    legend_patches = [
        mpatches.Patch(color=C_YOUNG, alpha=0.75, label="Young"),
        mpatches.Patch(color=C_MID,   alpha=0.75, label="Mid-career"),
        mpatches.Patch(color=C_RET,   alpha=0.75, label="Retired"),
    ]
    ax.legend(handles=legend_patches, framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, "fig_rr_boxplot")
    plt.close(fig)

    # =========================================================================
    # FIGURE 6 — Multi-seed FR fan chart
    # =========================================================================
    print("Building multi-seed fan chart...")

    seed_map = {s: f"src/models/run_007_s{s}/best_model.zip"
                for s in range(1, 15)}
    seed_map[42] = "src/models/run_007/best_model.zip"

    fig, ax = plt.subplots(figsize=(12, 5))

    fr_active_all = []

    for seed, model_path in seed_map.items():
        mp = Path(model_path)
        if not mp.exists():
            print(f"  Skipping seed {seed} — model not found")
            continue

        env_s = make_env_from_pipeline(results, split="test",
                                       cfg=env_cfg, seed=seed)
        agent_s = load_drl(str(mp), env_s)
        traj_s  = run_episode(agent_s, env_s)
        fr_s    = np.array(traj_s["FR"])
        dates_s = pd.DatetimeIndex(traj_s["dates"])

        is_degen = seed in DEGENERATE

        if is_degen:
            ax.plot(dates_s, fr_s, color="grey", lw=0.8,
                    ls="--", alpha=0.45, zorder=1)
        elif seed == 42:
            pass   # plot last so it's on top
        else:
            ax.plot(dates_s, fr_s, color=C_DRL, lw=0.9,
                    alpha=0.25, zorder=2)
            fr_active_all.append(fr_s)

        print(f"  Seed {seed} done  (degenerate={is_degen})")

    # Median band of active seeds
    if fr_active_all:
        stack = np.vstack(fr_active_all)
        ax.fill_between(dates_s,
                        np.percentile(stack, 25, axis=0),
                        np.percentile(stack, 75, axis=0),
                        color=C_DRL, alpha=0.15, zorder=3,
                        label="Active seeds IQR")

    # Seed 42 bold on top
    env_42  = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)
    agent_42 = load_drl("src/models/run_007/best_model.zip", env_42)
    traj_42  = run_episode(agent_42, env_42)
    ax.plot(pd.DatetimeIndex(traj_42["dates"]), np.array(traj_42["FR"]),
            color=C_DRL, lw=2.5, zorder=5, label="Seed 42 (primary)")

    # Baselines
    ax.plot(dates, FR_fixed, color=C_FIXED, lw=2, ls="-.",
            zorder=4, label="Fixed-Rule baseline")

    ax.axhline(MVEV_FLOOR, color="red",  lw=1.2, ls="--",
               label=f"MVEV floor ({MVEV_FLOOR})", zorder=6)
    ax.axhline(FR_TARGET,  color="grey", lw=1.0, ls=":",
               label=f"FR target ({FR_TARGET})", zorder=6)

    # Legend entries for degenerate
    degen_line = plt.Line2D([0], [0], color="grey", lw=1.2,
                            ls="--", alpha=0.7, label="Degenerate seeds (2,8,12)")
    active_line = plt.Line2D([0], [0], color=C_DRL, lw=1,
                             alpha=0.5, label="Active seeds (n=12)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [active_line, degen_line],
              framealpha=0.9, fontsize=9)

    fmt_xaxis(ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Funding Ratio $\\mathrm{FR}_t$")
    ax.set_title("Multi-Seed FR Fan Chart — 15 Seeds (3 degenerate shown dashed)")
    fig.tight_layout()
    save(fig, out_dir, "fig_fan_chart")
    plt.close(fig)

    print("\nAll figures complete.")


if __name__ == "__main__":
    main()
