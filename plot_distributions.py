"""plot_distributions.py — Distribution comparison across all three models.

Produces two figures:
  1. Monthly distribution rate over time (bar chart, stacked by model)
  2. Cumulative distributions over time (line chart)

Usage
-----
    py -3 plot_distributions.py --model-path src/models/run_007/best_model.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


class _SB3Adapter:
    def __init__(self, model) -> None:
        self._model = model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str,
                   default="src/models/run_007/best_model.zip")
    p.add_argument("--no-mc", action="store_true")
    p.add_argument("--out-dir", type=str, default="src/models/run_007")
    return p.parse_args(argv)


def main(argv=None):
    args    = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ---------------------------------------------------------------
    results = run_pipeline()
    env_cfg = EnvConfig()

    test_env_drl   = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)
    test_env_fixed = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)
    test_env_mc    = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)

    # ---- Agents -------------------------------------------------------------
    drl_model = PPO.load(
        args.model_path,
        env=test_env_drl,
        custom_objects={
            "policy_class":  WtpActorCriticPolicy,
            "policy_kwargs": {"wtp_cfg": AgentConfig()},
        },
    )
    drl_agent   = _SB3Adapter(drl_model)
    fixed_agent = FixedRuleALM()

    mc_agent = None
    if not args.no_mc:
        mc_agent = MonteCarloALM()
        mc_agent.fit(
            results["z_train_raw"],
            results["raw_train"],
            results["cpi"],
            env_cfg,
        )

    # ---- Trajectories -------------------------------------------------------
    print("Running episodes...")
    traj_drl   = run_episode(drl_agent,   test_env_drl)
    traj_fixed = run_episode(fixed_agent, test_env_fixed)
    traj_mc    = run_episode(mc_agent, test_env_mc) if mc_agent else None

    dates = pd.DatetimeIndex(traj_drl["dates"])
    dist_drl   = np.array(traj_drl["d_tilde"])
    dist_fixed = np.array(traj_fixed["d_tilde"])
    dist_mc    = np.array(traj_mc["d_tilde"]) if traj_mc else None

    # ---- Colours ------------------------------------------------------------
    c_drl   = "#1f77b4"   # blue
    c_fixed = "#ff7f0e"   # orange
    c_mc    = "#2ca02c"   # green

    # =========================================================================
    # Figure 1: Monthly distribution rate
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 4))

    width = 20          # bar width in days
    offsets = [-20, 0, 20] if dist_mc is not None else [-10, 10]

    ax1.bar(dates + pd.Timedelta(days=offsets[0]), dist_drl,
            width=width, color=c_drl,   alpha=0.85, label="DRL (PPO)")
    ax1.bar(dates + pd.Timedelta(days=offsets[1]), dist_fixed,
            width=width, color=c_fixed, alpha=0.85, label="Fixed-Rule")
    if dist_mc is not None:
        ax1.bar(dates + pd.Timedelta(days=offsets[2]), dist_mc,
                width=width, color=c_mc, alpha=0.85, label="Monte Carlo")

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Monthly distribution rate $\\tilde{d}_t$")
    ax1.set_title("Monthly Distributions — Test Period (Jan 2018 – Dec 2025)")
    ax1.legend(framealpha=0.9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
    fig1.tight_layout()
    path1 = out_dir / "dist_monthly.pdf"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    fig1.savefig(str(path1).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {path1}")

    # =========================================================================
    # Figure 2: Cumulative distributions
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 4))

    cum_drl   = np.cumsum(dist_drl)
    cum_fixed = np.cumsum(dist_fixed)

    ax2.plot(dates, cum_drl,   color=c_drl,   lw=2,   label=f"DRL (PPO)   total={cum_drl[-1]:.3f}")
    ax2.plot(dates, cum_fixed, color=c_fixed, lw=2,   label=f"Fixed-Rule  total={cum_fixed[-1]:.3f}")
    if dist_mc is not None:
        cum_mc = np.cumsum(dist_mc)
        ax2.plot(dates, cum_mc, color=c_mc, lw=2,
                 label=f"Monte Carlo total={cum_mc[-1]:.3f}")

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative distributions $\\sum \\tilde{d}_t$")
    ax2.set_title("Cumulative Distributions — Test Period (Jan 2018 – Dec 2025)")
    ax2.legend(framealpha=0.9)
    fig2.tight_layout()
    path2 = out_dir / "dist_cumulative.pdf"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    fig2.savefig(str(path2).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {path2}")

    # =========================================================================
    # Figure 3: Buffer level over time (context for when distributions stop)
    # =========================================================================
    fig3, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    b_drl   = np.array(traj_drl["B"])
    b_fixed = np.array(traj_fixed["B"])

    ax_b, ax_d = axes

    ax_b.plot(dates, b_drl,   color=c_drl,   lw=2, label="DRL (PPO)")
    ax_b.plot(dates, b_fixed, color=c_fixed, lw=2, label="Fixed-Rule")
    if traj_mc is not None:
        ax_b.plot(dates, np.array(traj_mc["B"]), color=c_mc, lw=2, label="Monte Carlo")
    ax_b.axhline(0.001, color="red", lw=1, ls="--", alpha=0.6, label="Depletion threshold")
    ax_b.set_ylabel("Buffer level $B_t$")
    ax_b.set_title("Buffer Level and Monthly Distributions — Test Period")
    ax_b.legend(framealpha=0.9, fontsize=9)

    ax_d.bar(dates - pd.Timedelta(days=10), dist_drl,
             width=18, color=c_drl,   alpha=0.85, label="DRL (PPO)")
    ax_d.bar(dates + pd.Timedelta(days=10), dist_fixed,
             width=18, color=c_fixed, alpha=0.85, label="Fixed-Rule")
    if dist_mc is not None:
        ax_d.bar(dates + pd.Timedelta(days=28), dist_mc,
                 width=18, color=c_mc, alpha=0.85, label="Monte Carlo")
    ax_d.set_ylabel("Monthly $\\tilde{d}_t$")
    ax_d.set_xlabel("Date")
    ax_d.legend(framealpha=0.9, fontsize=9)

    ax_d.xaxis.set_major_locator(mdates.YearLocator())
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig3.tight_layout()
    path3 = out_dir / "buffer_and_dist.pdf"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    fig3.savefig(str(path3).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {path3}")

    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
