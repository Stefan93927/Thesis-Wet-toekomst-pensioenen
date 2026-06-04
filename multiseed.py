"""multiseed.py — Multi-seed reproducibility analysis for the Wtp DRL agent.

Trains the PPO agent with multiple random seeds, evaluates each best-model
on the test set, and reports mean +/- std across all seeds to demonstrate
that results are not driven by a single lucky initialisation.

Seed layout
-----------
- Seed 42     : base run, already trained
- Seeds 1,2,3,4        : four additional replications (default)



Usage
-----
    # Train seeds 1,2,3,4 then evaluate all 5 (base + 4 new)
    py -3 multiseed.py

    # Evaluate only -- skip training (requires existing best_model.zip)
    py -3 multiseed.py --no-train

    # Custom seeds, fewer timesteps (faster)
    py -3 multiseed.py --seeds 1 2 3 --timesteps 500000

    # Override base run directory
    py -3 multiseed.py --base-dir src/models/run_007
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.data_pipeline import run_pipeline
from src.environment   import make_env_from_pipeline, EnvConfig
from src.agent         import fit_gmm, make_agent, AgentConfig, WtpActorCriticPolicy
from src.baselines     import FixedRuleALM, run_episode
from src.metrics       import compute_metrics, bootstrap_ci

try:
    import torch
    import random as _random
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback, CallbackList, CheckpointCallback,
    )
    from stable_baselines3.common.vec_env import SubprocVecEnv
except ImportError as exc:
    raise ImportError(
        "torch and stable-baselines3 are required.\n"
        "Install with:  pip install torch stable-baselines3"
    ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _SB3Adapter:
    def __init__(self, model) -> None:
        self._model = model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action


class _MinimalValCallback(BaseCallback):
    """Lightweight validation callback: saves best model only."""

    def __init__(self, val_env, eval_freq: int, log_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.val_env   = val_env
        self.eval_freq = eval_freq
        self.log_dir   = log_dir
        self.best_reward: float = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        adapter = _SB3Adapter(self.model)
        traj    = run_episode(adapter, self.val_env)
        if traj["total_reward"] > self.best_reward:
            self.best_reward = traj["total_reward"]
            self.model.save(str(self.log_dir / "best_model"))
            if self.verbose >= 1:
                print(f"    [seed] t={self.num_timesteps:>9,d}  "
                      f"new best reward={self.best_reward:.2f}")
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_seed(
    seed:       int,
    results:    dict,
    log_dir:    Path,
    timesteps:  int,
    eval_freq:  int,
    n_envs:     int,
) -> Path:
    """Train one seed and return path to best_model.zip."""
    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = log_dir / "best_model.zip"

    if best_model_path.exists():
        print(f"  Seed {seed}: best_model.zip already exists, skipping training.")
        return best_model_path

    print(f"\n  Seed {seed}: training {timesteps:,} steps -> {log_dir}")
    _set_seeds(seed)

    env_cfg   = EnvConfig()
    agent_cfg = AgentConfig(total_timesteps=timesteps)

    vstoxx_train    = results["z_train_raw"]["vstoxx_level"].values
    rts_slope_train = results["z_train_raw"]["rts_slope_30y_10y"].values
    gmm = fit_gmm(
        vstoxx_train    = vstoxx_train,
        rts_slope_train = rts_slope_train,
        n_regimes       = agent_cfg.gmm_n_regimes,
        seed            = agent_cfg.gmm_seed,
    )

    if n_envs > 1:
        def make_env(rank):
            def _init():
                return make_env_from_pipeline(results, "train", env_cfg, seed=seed + rank)
            return _init
        train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        train_env = make_env_from_pipeline(results, "train", env_cfg, seed=seed)

    val_env   = make_env_from_pipeline(results, "val", env_cfg, seed=seed)
    agent     = make_agent(train_env, cfg=agent_cfg, gmm=gmm, seed=seed)

    val_cb  = _MinimalValCallback(val_env, eval_freq, log_dir, verbose=1)
    ckpt_cb = CheckpointCallback(
        save_freq  = max(timesteps // 10, 50_000),
        save_path  = str(log_dir / "checkpoints"),
        name_prefix = "ppo_wtp",
        verbose    = 0,
    )

    t0 = time.time()
    agent.learn(
        total_timesteps     = timesteps,
        callback            = CallbackList([val_cb, ckpt_cb]),
        progress_bar        = True,
        reset_num_timesteps = True,
    )
    elapsed = time.time() - t0
    print(f"  Seed {seed}: done in {elapsed/60:.1f} min  "
          f"best val reward={val_cb.best_reward:.2f}")

    # Save config
    with open(log_dir / "train_config.json", "w") as f:
        json.dump({"seed": seed, "timesteps": timesteps, "eval_freq": eval_freq}, f)

    return best_model_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_seed(
    model_path: Path,
    results:    dict,
    pi_test:    np.ndarray,
    seed:       int,
) -> dict:
    """Evaluate one seed's best_model on the test set; return metrics + CIs."""
    env_cfg  = EnvConfig()
    test_env = make_env_from_pipeline(results, "test", env_cfg, seed=0)

    model = PPO.load(
        str(model_path),
        env=test_env,
        custom_objects={
            "policy_class":  WtpActorCriticPolicy,
            "policy_kwargs": {"wtp_cfg": AgentConfig()},
        },
    )
    drl_agent = _SB3Adapter(model)
    traj      = run_episode(drl_agent, test_env)
    metrics   = compute_metrics(traj, pi_monthly=pi_test)
    cis       = bootstrap_ci(traj, pi_monthly=pi_test, n_boot=1000, seed=0)

    return {"metrics": metrics, "bootstrap_ci_95": cis, "seed": seed}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(seed_results: list[dict]) -> dict:
    """Compute mean, std, min, max across seed results.

    Args:
        seed_results: List of dicts each containing ``"metrics"`` and ``"seed"``.

    Returns:
        Dict with ``"mean"``, ``"std"``, ``"min"``, ``"max"`` sub-dicts.
    """
    metric_keys = list(seed_results[0]["metrics"].keys())
    arrays      = {k: [r["metrics"][k] for r in seed_results] for k in metric_keys}

    return {
        "n_seeds": len(seed_results),
        "seeds":   [r["seed"] for r in seed_results],
        "mean":    {k: float(np.mean(arrays[k]))  for k in metric_keys},
        "std":     {k: float(np.std(arrays[k], ddof=1)) for k in metric_keys},
        "min":     {k: float(np.min(arrays[k]))   for k in metric_keys},
        "max":     {k: float(np.max(arrays[k]))   for k in metric_keys},
        "per_seed": {
            r["seed"]: {"metrics": r["metrics"], "bootstrap_ci_95": {
                k: list(v) for k, v in r["bootstrap_ci_95"].items()
            }}
            for r in seed_results
        },
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _print_aggregate_table(agg: dict, fixed_metrics: dict | None = None) -> None:
    keys = [
        ("fr_terminal",           "FR Terminal"),
        ("fr_mdd",                "FR MDD"),
        ("fr_vol_ann",            "FR Vol (ann)"),
        ("buffer_depletion_freq", "Buf Depl Freq"),
        ("total_distributions",   "Total Dist"),
        ("calmar_ratio",          "Calmar"),
        ("cohort_rr_var",         "Cohort RR Var"),
    ]
    seeds_str = ", ".join(str(s) for s in agg["seeds"])
    print(f"\nMulti-seed results  (n={agg['n_seeds']} seeds: {seeds_str})")
    print("=" * 72)
    header = f"  {'Metric':<22}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}"
    if fixed_metrics:
        header += f"{'Fixed-Rule':>12}"
    print(header)
    print("-" * len(header))
    for key, label in keys:
        row = (f"  {label:<22}"
               f"{agg['mean'][key]:>10.4f}"
               f"{agg['std'][key]:>10.4f}"
               f"{agg['min'][key]:>10.4f}"
               f"{agg['max'][key]:>10.4f}")
        if fixed_metrics:
            row += f"{fixed_metrics.get(key, float('nan')):>12.4f}"
        print(row)
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Multi-seed reproducibility check for Wtp DRL agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seeds",      type=int, nargs="+", default=[1, 2, 3, 4],
                   help="Additional seeds to train/evaluate")
    p.add_argument("--base-dir",   type=str, default="src/models/run_007",
                   help="Directory of the base run (seed=42)")
    p.add_argument("--base-seed",  type=int, default=42,
                   help="Seed used for the base run")
    p.add_argument("--seed-dir-prefix", type=str, default="src/models/run_007_s",
                   help="Prefix for per-seed log directories (seed appended)")
    p.add_argument("--timesteps",  type=int, default=1_000_000,
                   help="PPO timesteps per seed")
    p.add_argument("--eval-freq",  type=int, default=50_000,
                   help="Validation frequency per seed")
    p.add_argument("--n-envs",     type=int, default=4,
                   help="Parallel training envs per seed")
    p.add_argument("--no-train",   action="store_true",
                   help="Skip training; evaluate existing checkpoints only")
    p.add_argument("--out-dir",    type=str, default="src/models",
                   help="Directory to save multiseed_results.json")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args    = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Wtp DRL -- Multi-Seed Reproducibility Analysis")
    print("=" * 64)
    print(f"  Base run   : {args.base_dir}  (seed={args.base_seed})")
    print(f"  Extra seeds: {args.seeds}")
    print(f"  Timesteps  : {args.timesteps:,}")
    print(f"  No-train   : {args.no_train}")

    # ---- Data (shared across seeds) ---------------------------------------- #
    print("\n[1/3] Loading data pipeline...")
    results = run_pipeline()

    test_dates = results["z_test"].index
    pi_test    = (
        results["cpi"]["pi_monthly"]
        .reindex(test_dates).fillna(0.0).values
    )

    # ---- Train / locate each seed ------------------------------------------ #
    print("\n[2/3] Training / locating seed models...")
    seed_model_paths: dict[int, Path] = {}

    # Base run
    base_path = Path(args.base_dir) / "best_model.zip"
    if base_path.exists():
        seed_model_paths[args.base_seed] = base_path
        print(f"  Base seed {args.base_seed}: found {base_path}")
    else:
        print(f"  WARNING: base model not found at {base_path} -- skipping base seed")

    # Extra seeds
    for seed in args.seeds:
        log_dir = Path(f"{args.seed_dir_prefix}{seed}")
        if args.no_train:
            p = log_dir / "best_model.zip"
            if p.exists():
                seed_model_paths[seed] = p
                print(f"  Seed {seed}: found {p}")
            else:
                print(f"  Seed {seed}: no model at {p} -- skipping (run without --no-train)")
        else:
            p = train_seed(seed, results, log_dir, args.timesteps, args.eval_freq, args.n_envs)
            seed_model_paths[seed] = p

    if not seed_model_paths:
        print("ERROR: no models found or trained.  Exiting.")
        sys.exit(1)

    # ---- Evaluate each seed ------------------------------------------------ #
    print(f"\n[3/3] Evaluating {len(seed_model_paths)} seed(s) on test set...")
    seed_results: list[dict] = []

    for seed, model_path in sorted(seed_model_paths.items()):
        print(f"  Evaluating seed {seed}...")
        r = evaluate_seed(model_path, results, pi_test, seed)
        seed_results.append(r)
        m = r["metrics"]
        print(f"    FR_term={m['fr_terminal']:.4f}  MDD={m['fr_mdd']:.4f}  "
              f"Calmar={m['calmar_ratio']:.4f}  Depl={m['buffer_depletion_freq']:.3f}  "
              f"Dist={m['total_distributions']:.4f}")

    # ---- Aggregate --------------------------------------------------------- #
    agg = aggregate_seeds(seed_results)

    # Fixed-Rule baseline for comparison column
    fixed_env   = make_env_from_pipeline(results, "test", EnvConfig(), seed=0)
    fixed_traj  = run_episode(FixedRuleALM(), fixed_env)
    fixed_m     = compute_metrics(fixed_traj, pi_monthly=pi_test)

    _print_aggregate_table(agg, fixed_metrics=fixed_m)

    # ---- Save --------------------------------------------------------------- #
    out_path = out_dir / "multiseed_results.json"
    with open(out_path, "w") as f:
        json.dump({"aggregate": agg, "fixed_rule": fixed_m}, f, indent=2, default=float)
    print(f"\n  Results saved: {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
