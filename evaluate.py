"""evaluate.py — Out-of-sample evaluation of the Wtp DRL agent vs. baselines.

Pipeline
--------
1. Load data pipeline (test split Jan 2018 - Dec 2025).
2. Run three agents on the test environment:
   - DRL PPO agent (best_model.zip)
   - Fixed-Rule ALM baseline
   - Monte Carlo ALM baseline
3. Compute all evaluation metrics for each agent.
4. Compute regime-conditional metrics (Low / Med / High VSTOXX).
5. Print formatted comparison tables and save results to JSON.

Usage
-----
    # Evaluate best model from default run directory
    py -3 evaluate.py

    # Custom model path
    py -3 evaluate.py --model-path src/models/run_002/best_model.zip

    # Skip Monte Carlo (slow to fit)
    py -3 evaluate.py --no-mc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.data_pipeline import run_pipeline
from src.environment   import make_env_from_pipeline, EnvConfig
from src.agent         import AgentConfig, WtpActorCriticPolicy
import zipfile
from src.baselines      import FixedRuleALM, MonteCarloALM, run_episode
from src.hoevenaars_alm import HoevenaarsALM
from src.metrics       import (
    compute_metrics,
    regime_conditional_metrics,
    bootstrap_ci,
    format_metrics_table,
    format_regime_table,
    format_ci_table,
    diebold_mariano,
    dm_losses,
    format_dm_table,
)

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required.  Install with:  pip install stable-baselines3"
    ) from exc


# ---------------------------------------------------------------------------
# SB3 adapter (same as train.py)
# ---------------------------------------------------------------------------

class _SB3Adapter:
    """Thin wrapper so SB3 model.predict() matches run_episode's interface."""

    def __init__(self, model) -> None:
        self._model = model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate Wtp DRL agent vs. baselines on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", type=str,
                   default="src/models/run_001/best_model.zip",
                   help="Path to trained PPO model (.zip)")
    p.add_argument("--log-dir",    type=str,
                   default=None,
                   help="Directory to save evaluation results JSON (default: model's parent dir)")
    p.add_argument("--no-mc",      action="store_true",
                   help="Skip Monte Carlo ALM (faster, omits MC row)")
    p.add_argument("--lifecycle",  action="store_true", default=True,
                   help="Enable per-cohort lifecycle equity weights "
                        "(auto-loaded from train_config.json when available; "
                        "default True matches training default)")
    p.add_argument("--no-lifecycle", dest="lifecycle", action="store_false",
                   help="Force lifecycle off (legacy models only)")
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(argv=None) -> None:
    args    = parse_args(argv)
    model_path = Path(args.model_path)
    log_dir = Path(args.log_dir) if args.log_dir else model_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Wtp DRL Pension Fund -- Out-of-Sample Evaluation")
    print("=" * 64)
    print(f"  Model      : {args.model_path}")
    print(f"  Test period: Jan 2018 - Dec 2025")

    # ---- 1. Data --------------------------------------------------------- #
    print("\n[1/4] Loading data pipeline...")
    results = run_pipeline()

    # Auto-detect lifecycle setting from the model's saved train_config.json.
    # Falls back to args.lifecycle (default True) if key is absent.
    _tc_path = model_path.parent / "train_config.json"
    _lifecycle = args.lifecycle
    _tc_bps    = 0.0
    if _tc_path.exists():
        _tc = json.loads(_tc_path.read_text())
        if "lifecycle" in _tc:
            _lifecycle = bool(_tc["lifecycle"])
            print(f"  use_lifecycle: {_lifecycle}  (from train_config.json)")
        else:
            print(f"  use_lifecycle: {_lifecycle}  (train_config.json has no 'lifecycle' key; using --lifecycle default)")
        if "tc_bps" in _tc:
            _tc_bps = float(_tc["tc_bps"])
            print(f"  tc_bps: {_tc_bps}  (from train_config.json)")
        else:
            print(f"  tc_bps: 0.0  (train_config.json has no 'tc_bps' key; defaulting to 0)")
    else:
        print(f"  use_lifecycle: {_lifecycle}  (no train_config.json; using --lifecycle default)")
    env_cfg = EnvConfig(use_lifecycle=_lifecycle, tc_bps=_tc_bps)

    test_env_drl    = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=args.seed)
    test_env_fixed  = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=args.seed)
    test_env_mc     = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=args.seed)
    test_env_hoev   = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=args.seed)

    # Align test CPI
    test_dates = results["z_test"].index
    pi_test    = (
        results["cpi"]["pi_monthly"]
        .reindex(test_dates)
        .fillna(0.0)
        .values
    )
    vstoxx_test = results["z_test_raw"]["vstoxx_level"]

    # ---- 2. Load agents -------------------------------------------------- #
    print("\n[2/4] Loading agents...")

    # DRL agent
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run train.py first, or pass --model-path to a valid .zip file."
        )
    # Auto-detect n_regimes from gating layer shape in the zip
    _n_regimes = 3
    try:
        import torch, io as _io
        _zip_path = str(model_path) if str(model_path).endswith(".zip") else str(model_path) + ".zip"
        with zipfile.ZipFile(_zip_path) as _zf:
            with _zf.open("policy.pth") as _f:
                _state = torch.load(_io.BytesIO(_f.read()), map_location="cpu",
                                    weights_only=False)
        _bias = _state.get("wtp_net.gating.linear.bias")
        if _bias is not None:
            _n_regimes = int(_bias.shape[0])
    except Exception:
        pass
    _eval_cfg = AgentConfig(n_regimes=_n_regimes, gmm_n_regimes=_n_regimes,
                            beta_bar=([0.70, 0.55, 0.40, 0.25] if _n_regimes == 4
                                      else [0.65, 0.55, 0.35]))
    drl_model = PPO.load(
        str(model_path),
        env=test_env_drl,
        custom_objects={
            "policy_class":  WtpActorCriticPolicy,
            "policy_kwargs": {"wtp_cfg": _eval_cfg},
        },
    )
    drl_agent = _SB3Adapter(drl_model)
    print(f"  DRL agent loaded from {model_path}")

    # Fixed-Rule baseline
    fixed_agent = FixedRuleALM()
    print("  Fixed-Rule ALM ready")

    # Hoevenaars ALM baseline (v4 headline policy, fixed parameters)
    import json as _json
    _hoev_path = _ROOT / "results" / "hoevenaars_v4_headline_policy.json"
    if _hoev_path.exists():
        _hl = _json.loads(_hoev_path.read_text())["params"]
        _hoev_params = np.array([
            _hl["a"], _hl["h"], _hl["i"], _hl["c"],
            _hl["g"], _hl["e"], _hl["B_target"], _hl["B_min"],
        ], dtype=np.float64)
        _mu_vstoxx = float(np.percentile(
            results["z_train_raw"]["vstoxx_level"].values, 67
        ))
        hoev_agent = HoevenaarsALM(
            params        = _hoev_params,
            mu_vstoxx     = _mu_vstoxx,
            vstoxx_series = results["z_test"]["vstoxx_level"].values,
        )
        print(f"  Hoevenaars ALM ready  (mu_vstoxx={_mu_vstoxx:.2f})")
    else:
        hoev_agent = None
        print("  Hoevenaars ALM skipped (results/hoevenaars_v4_headline_policy.json not found)")

    # Monte Carlo baseline
    mc_agent = None
    if not args.no_mc:
        print("  Fitting Monte Carlo ALM (VAR + grid search)...")
        mc_agent = MonteCarloALM()
        mc_agent.fit(
            results["z_train_raw"],
            results["raw_train"],
            results["cpi"],
            env_cfg,
        )
        print(f"  Monte Carlo ALM ready  (f*={mc_agent.f_star:.3f}, d*={mc_agent.d_star:.3f})")

    # ---- 3. Run episodes ------------------------------------------------- #
    print("\n[3/4] Running test episodes...")

    trajectories = {}

    print("  Running DRL agent...")
    trajectories["DRL (PPO)"] = run_episode(drl_agent, test_env_drl)

    print("  Running Fixed-Rule ALM...")
    trajectories["Fixed-Rule"] = run_episode(fixed_agent, test_env_fixed)

    if mc_agent is not None:
        print("  Running Monte Carlo ALM...")
        trajectories["Monte Carlo"] = run_episode(mc_agent, test_env_mc)

    if hoev_agent is not None:
        print("  Running Hoevenaars ALM...")
        trajectories["Hoevenaars"] = run_episode(hoev_agent, test_env_hoev)

    # ---- 4. Metrics ------------------------------------------------------ #
    print("\n[4/4] Computing metrics...")

    metrics_by_agent  = {}
    regime_by_agent   = {}
    ci_by_agent       = {}

    for name, traj in trajectories.items():
        m = compute_metrics(traj, pi_monthly=pi_test)
        r = regime_conditional_metrics(traj, vstoxx_test, pi_monthly=pi_test)
        metrics_by_agent[name] = m
        regime_by_agent[name]  = r

    print("  Computing bootstrap CIs (1000 replications per agent)...")
    for name, traj in trajectories.items():
        ci_by_agent[name] = bootstrap_ci(traj, pi_monthly=pi_test, n_boot=1000, seed=0)

    # ---- Print results --------------------------------------------------- #
    print("\n" + "=" * 64)
    print("RESULTS -- Test period (Jan 2018 - Dec 2025)")
    print("=" * 64)

    print("\n" + format_metrics_table(metrics_by_agent, title="Core Metrics"))
    print("\n" + format_ci_table(metrics_by_agent, ci_by_agent,
                                 title="Core Metrics with 95% Bootstrap CI"))

    for metric_key, metric_label in [
        ("fr_mdd",                "FR Max Drawdown"),
        ("buffer_depletion_freq", "Buffer Depletion Freq"),
        ("total_distributions",   "Total Distributions"),
        ("calmar_ratio",          "Calmar Ratio"),
    ]:
        print("\n" + format_regime_table(regime_by_agent, metric_key, metric_label))

    # ---- Action statistics (DRL agent) ----------------------------------- #
    drl_traj = trajectories["DRL (PPO)"]
    if "actions" in drl_traj:
        actions = np.array(drl_traj["actions"])
        print("\nDRL Agent -- Action Statistics (test period):")
        labels = ["equity_tilt", "fill_rate ", "dist_rate "]
        for i, lab in enumerate(labels):
            a = actions[:, i]
            print(f"  {lab}: mean={a.mean():+.4f}  std={a.std():.4f}  "
                  f"min={a.min():+.4f}  max={a.max():+.4f}")

    # ---- Diebold-Mariano significance tests ------------------------------ #
    # Compare DRL against each baseline on two per-period loss functions:
    #   buf_depl  = 1(B_t <= 0.001)           — primary Wtp adequacy loss
    #   fr_sq_dev = (FR_t - 1.05)^2           — solvency stability loss
    # Positive HLN stat => DRL has lower expected loss (DRL wins).
    print("\n" + "=" * 64)
    print("Diebold-Mariano Tests")
    print("=" * 64)

    drl_losses = dm_losses(trajectories["DRL (PPO)"])
    T_dm       = len(drl_losses["buf_depl"])

    baselines_for_dm = [
        n for n in ("Fixed-Rule", "Monte Carlo", "Hoevenaars")
        if n in trajectories
    ]

    dm_results: dict[str, dict[str, tuple[float, float]]] = {}
    for bname in baselines_for_dm:
        bl_losses = dm_losses(trajectories[bname])
        dm_results[f"DRL vs {bname}"] = {
            "Buf. Depl. (1/0)":    diebold_mariano(bl_losses["buf_depl"],
                                                    drl_losses["buf_depl"]),
            "FR Stab. (sq. dev.)": diebold_mariano(bl_losses["fr_sq_dev"],
                                                    drl_losses["fr_sq_dev"]),
        }

    print(format_dm_table(dm_results, T=T_dm))

    # ---- Save JSON ------------------------------------------------------- #
    output = {
        "test_period": "2018-01 to 2025-12",
        "model_path":  str(args.model_path),
        "metrics":     metrics_by_agent,
        "bootstrap_ci_95": {
            agent: {k: list(v) for k, v in cis.items()}
            for agent, cis in ci_by_agent.items()
        },
        "regime_metrics": {
            agent: {
                regime: {k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in rm.items()}
                for regime, rm in rdict.items()
            }
            for agent, rdict in regime_by_agent.items()
        },
    }
    out_path = log_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n  Results saved: {out_path}")

    # ---- Save per-step trajectory arrays for downstream analysis ---------- #
    traj_keys = ["FR", "B", "d_tilde", "f_tilde", "w_eq", "r_p", "r_L",
                 "r_p_young", "r_p_mid", "r_p_ret",
                 "ppv_young", "ppv_mid", "ppv_ret", "rewards"]
    for agent_name, traj in trajectories.items():
        safe_name = agent_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        arrays = {k: traj[k] for k in traj_keys if k in traj}
        arrays["dates"] = np.array([str(d) for d in traj["dates"]])
        np.savez(log_dir / f"trajectory_{safe_name}.npz", **arrays)
    print(f"  Trajectories saved: {log_dir}/trajectory_*.npz")
    print("=" * 64)


if __name__ == "__main__":
    evaluate()
