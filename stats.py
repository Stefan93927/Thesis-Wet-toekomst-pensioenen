"""stats.py — Statistical significance tests for the Wtp DRL evaluation.

Tests implemented
-----------------
1. Diebold–Mariano (Harvey, Leybourne & Newbold 1997 small-sample correction)
   Compares per-step loss paths: DRL vs Fixed-Rule, DRL vs MC ALM.
   Loss function: squared funding-ratio shortfall below the FR target (1.05).

2. Multi-seed significance
   Paired t-test and Wilcoxon signed-rank test on per-seed Calmar ratios
   (DRL seeds vs Fixed-Rule constant).

3. Seed filtering protocol
   Active seeds: total_dist > DIST_THRESHOLD (0.05).
   Degenerate seeds converge to the no-distribution local optimum and are
   reported separately with full disclosure.

Usage
-----
    py -3 stats.py
    py -3 stats.py --no-dm      # skip DM test (faster)
    py -3 stats.py --no-seed    # skip seed analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FR_TARGET       = 1.05    # stability target
MVEV_FLOOR      = 1.043   # regulatory solvency floor
DIST_THRESHOLD  = 0.05    # minimum total distributions to classify as active seed
MODEL_PATH      = "src/models/run_007/best_model.zip"
MS_JSON_PATH    = "src/models/multiseed_results.json"


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def _shortfall_loss(fr_path: np.ndarray, target: float = FR_TARGET) -> np.ndarray:
    """Per-step squared shortfall below FR target.

    L_t = max(0, target - FR_t)^2

    Args:
        fr_path: FR trajectory array (T,).
        target:  FR target level.

    Returns:
        Loss array (T,).
    """
    return np.maximum(0.0, target - fr_path) ** 2


# ---------------------------------------------------------------------------
# Diebold–Mariano test (HLN 1997 small-sample correction)
# ---------------------------------------------------------------------------

def diebold_mariano(
    loss1: np.ndarray,
    loss2: np.ndarray,
    h: int = 1,
    alpha: float = 0.05,
) -> dict:
    """Diebold–Mariano test with Harvey–Leybourne–Newbold (1997) correction.

    Tests H0: E[d_t] = 0  vs  H1: E[d_t] != 0
    where d_t = loss1_t - loss2_t.

    Positive d̄ means model 2 has lower loss (is better).
    Negative d̄ means model 1 has lower loss (is better).

    Args:
        loss1:  Per-step losses from model 1 (T,).
        loss2:  Per-step losses from model 2 (T,).
        h:      Forecast horizon (1 for one-step-ahead).
        alpha:  Significance level.

    Returns:
        Dict with keys: d_bar, dm_stat, hlm_stat, p_value, reject_h0,
        se, T, h, alpha.
    """
    d = loss1 - loss2
    T = len(d)
    d_bar = d.mean()

    # HAC variance: Newey–West with lag = h-1
    gamma0 = np.mean((d - d_bar) ** 2)
    hac_var = gamma0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        hac_var += 2 * gamma_k

    se = np.sqrt(max(hac_var / T, 1e-12))

    # Raw DM statistic
    dm_stat = d_bar / se

    # HLN small-sample correction: multiply DM by sqrt((T+1-2h+h(h-1)/T)/T)
    hlm_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    hlm_stat = dm_stat * hlm_factor

    # Two-sided p-value from t(T-1)
    p_value = 2.0 * scipy_stats.t.sf(abs(hlm_stat), df=T - 1)

    return {
        "d_bar":     float(d_bar),
        "dm_stat":   float(dm_stat),
        "hlm_stat":  float(hlm_stat),
        "p_value":   float(p_value),
        "reject_h0": bool(p_value < alpha),
        "se":        float(se),
        "T":         int(T),
        "h":         int(h),
        "alpha":     float(alpha),
    }


def run_dm_tests(results: dict, env_cfg: EnvConfig, model_path: str) -> dict:
    """Run DM tests comparing DRL vs Fixed-Rule and DRL vs MC ALM.

    Args:
        results:    Data pipeline output.
        env_cfg:    EnvConfig instance.
        model_path: Path to DRL best_model.zip.

    Returns:
        Dict with DM test results for both comparisons.
    """
    print("\n[DM] Loading agents and running test episodes...")

    # Load DRL
    dummy = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)
    model = PPO.load(
        model_path,
        env=dummy,
        custom_objects={
            "policy_class":  WtpActorCriticPolicy,
            "policy_kwargs": {"wtp_cfg": AgentConfig()},
        },
    )
    class _Adapter:
        def predict(self, obs):
            a, _ = model.predict(obs, deterministic=True)
            return a

    drl_agent   = _Adapter()
    fixed_agent = FixedRuleALM()
    mc_agent    = MonteCarloALM()
    mc_agent.fit(results["z_train_raw"], results["raw_train"], results["cpi"], env_cfg)

    env_drl   = make_env_from_pipeline(results, "test", env_cfg, seed=0)
    env_fixed = make_env_from_pipeline(results, "test", env_cfg, seed=0)
    env_mc    = make_env_from_pipeline(results, "test", env_cfg, seed=0)

    print("[DM] Running trajectories...")
    traj_drl   = run_episode(drl_agent,   env_drl)
    traj_fixed = run_episode(fixed_agent, env_fixed)
    traj_mc    = run_episode(mc_agent,    env_mc)

    fr_drl   = traj_drl["FR"]
    fr_fixed = traj_fixed["FR"]
    fr_mc    = traj_mc["FR"]

    loss_drl   = _shortfall_loss(fr_drl)
    loss_fixed = _shortfall_loss(fr_fixed)
    loss_mc    = _shortfall_loss(fr_mc)

    print(f"  Mean loss  DRL={loss_drl.mean():.6f}  Fixed={loss_fixed.mean():.6f}  MC={loss_mc.mean():.6f}")

    dm_vs_fixed = diebold_mariano(loss_fixed, loss_drl)   # d>0 means DRL better
    dm_vs_mc    = diebold_mariano(loss_mc,    loss_drl)   # d>0 means DRL better

    return {
        "loss_function":    "squared shortfall below FR target (1.05)",
        "DRL_vs_FixedRule": dm_vs_fixed,
        "DRL_vs_MCALM":     dm_vs_mc,
        "mean_loss": {
            "DRL":        float(loss_drl.mean()),
            "Fixed-Rule": float(loss_fixed.mean()),
            "MC ALM":     float(loss_mc.mean()),
        },
    }


# ---------------------------------------------------------------------------
# Multi-seed significance
# ---------------------------------------------------------------------------

def run_seed_significance(ms_json: dict) -> dict:
    """Paired t-test and Wilcoxon test on per-seed Calmar ratios.

    Also applies the seed filtering protocol (active vs degenerate).

    Args:
        ms_json: Dict loaded from multiseed_results.json.

    Returns:
        Dict with test results and filtered aggregate statistics.
    """
    agg      = ms_json["aggregate"]
    per_seed = agg["per_seed"]
    fixed_calmar = ms_json["fixed_rule"]["calmar_ratio"]

    seeds    = list(per_seed.keys())
    calmar_vals = [per_seed[s]["metrics"]["calmar_ratio"]      for s in seeds]
    dist_vals   = [per_seed[s]["metrics"]["total_distributions"] for s in seeds]
    mdd_vals    = [per_seed[s]["metrics"]["fr_mdd"]              for s in seeds]
    dep_vals    = [per_seed[s]["metrics"]["buffer_depletion_freq"] for s in seeds]

    # ---- Seed filtering protocol ---------------------------------------- #
    active_idx     = [i for i, d in enumerate(dist_vals) if d >= DIST_THRESHOLD]
    degenerate_idx = [i for i, d in enumerate(dist_vals) if d < DIST_THRESHOLD]

    active_calmar  = [calmar_vals[i] for i in active_idx]
    active_seeds   = [seeds[i]       for i in active_idx]
    degen_seeds    = [seeds[i]       for i in degenerate_idx]

    # ---- All-seeds test -------------------------------------------------- #
    # One-sample t-test: H0: mean Calmar = Fixed-Rule Calmar
    diffs_all = [c - fixed_calmar for c in calmar_vals]
    t_stat_all, p_ttest_all = scipy_stats.ttest_1samp(calmar_vals, fixed_calmar)
    # Wilcoxon signed-rank (paired vs constant)
    try:
        w_stat_all, p_wilcox_all = scipy_stats.wilcoxon(diffs_all, alternative="greater")
    except Exception:
        w_stat_all, p_wilcox_all = float("nan"), float("nan")

    # ---- Active-seeds test ----------------------------------------------- #
    if len(active_calmar) >= 3:
        diffs_act = [c - fixed_calmar for c in active_calmar]
        t_stat_act, p_ttest_act = scipy_stats.ttest_1samp(active_calmar, fixed_calmar)
        try:
            w_stat_act, p_wilcox_act = scipy_stats.wilcoxon(diffs_act, alternative="greater")
        except Exception:
            w_stat_act, p_wilcox_act = float("nan"), float("nan")
    else:
        t_stat_act = p_ttest_act = w_stat_act = p_wilcox_act = float("nan")

    return {
        "seed_filtering": {
            "threshold":       DIST_THRESHOLD,
            "criterion":       f"total_distributions >= {DIST_THRESHOLD}",
            "active_seeds":    active_seeds,
            "degenerate_seeds": degen_seeds,
            "n_active":        len(active_idx),
            "n_degenerate":    len(degenerate_idx),
        },
        "all_seeds": {
            "n":           len(seeds),
            "calmar_vals": calmar_vals,
            "mean":        float(np.mean(calmar_vals)),
            "std":         float(np.std(calmar_vals, ddof=1)),
            "fixed_rule":  float(fixed_calmar),
            "t_stat":      float(t_stat_all),
            "p_ttest":     float(p_ttest_all),
            "w_stat":      float(w_stat_all) if not np.isnan(w_stat_all) else None,
            "p_wilcox":    float(p_wilcox_all) if not np.isnan(p_wilcox_all) else None,
        },
        "active_seeds": {
            "n":           len(active_idx),
            "calmar_vals": active_calmar,
            "mean":        float(np.mean(active_calmar)) if active_calmar else float("nan"),
            "std":         float(np.std(active_calmar, ddof=1)) if len(active_calmar) > 1 else float("nan"),
            "fixed_rule":  float(fixed_calmar),
            "t_stat":      float(t_stat_act),
            "p_ttest":     float(p_ttest_act),
            "w_stat":      float(w_stat_act) if not np.isnan(w_stat_act) else None,
            "p_wilcox":    float(p_wilcox_act) if not np.isnan(p_wilcox_act) else None,
        },
        "per_seed_detail": {
            s: {
                "calmar":  calmar_vals[i],
                "dist":    dist_vals[i],
                "mdd":     mdd_vals[i],
                "depl":    dep_vals[i],
                "active":  dist_vals[i] >= DIST_THRESHOLD,
            }
            for i, s in enumerate(seeds)
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_dm(res: dict) -> None:
    w = 70
    print("\n" + "=" * w)
    print("Diebold–Mariano Tests  (HLN 1997 small-sample correction)")
    print(f"Loss function: {res['loss_function']}")
    print("-" * w)
    print(f"  Mean loss — DRL: {res['mean_loss']['DRL']:.6f}  "
          f"Fixed-Rule: {res['mean_loss']['Fixed-Rule']:.6f}  "
          f"MC ALM: {res['mean_loss']['MC ALM']:.6f}")
    print()

    for label, key in [("DRL vs Fixed-Rule", "DRL_vs_FixedRule"),
                       ("DRL vs MC ALM",     "DRL_vs_MCALM")]:
        r = res[key]
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05
              else ("*" if r["p_value"] < 0.10 else "n.s."))
        direction = "DRL better" if r["d_bar"] > 0 else "baseline better"
        print(f"  {label:<25}  d_bar={r['d_bar']:+.6f}  "
              f"HLN-stat={r['hlm_stat']:+.3f}  "
              f"p={r['p_value']:.4f} {sig}  [{direction}]")

    print("-" * w)
    print("  Significance: *** p<0.01  ** p<0.05  * p<0.10  n.s. not significant")
    print("=" * w)


def _print_seed(res: dict) -> None:
    w = 70
    print("\n" + "=" * w)
    print("Multi-Seed Significance Analysis")
    print("-" * w)

    filt = res["seed_filtering"]
    print(f"  Seed filtering: total_dist >= {filt['threshold']}")
    print(f"  Active seeds    : {filt['active_seeds']}  (n={filt['n_active']})")
    print(f"  Degenerate seeds: {filt['degenerate_seeds']}  (n={filt['n_degenerate']})")
    print(f"  Criterion: seeds converging to near-zero distributions are excluded")
    print()

    for label, key in [("All seeds", "all_seeds"), ("Active seeds only", "active_seeds")]:
        r = res[key]
        if r["n"] < 2:
            print(f"  {label}: insufficient seeds (n={r['n']})")
            continue
        sig_t = ("***" if r["p_ttest"] < 0.01 else
                 ("**" if r["p_ttest"] < 0.05 else
                  ("*"  if r["p_ttest"] < 0.10 else "n.s.")))
        print(f"  {label} (n={r['n']})")
        print(f"    Calmar: mean={r['mean']:.4f}  std={r['std']:.4f}  "
              f"Fixed-Rule={r['fixed_rule']:.4f}")
        print(f"    t-test (H0: mean=Fixed-Rule): t={r['t_stat']:+.3f}  "
              f"p={r['p_ttest']:.4f} {sig_t}")
        if r["p_wilcox"] is not None:
            sig_w = ("***" if r["p_wilcox"] < 0.01 else
                     ("**" if r["p_wilcox"] < 0.05 else
                      ("*"  if r["p_wilcox"] < 0.10 else "n.s.")))
            print(f"    Wilcoxon (H1: DRL > Fixed): W={r['w_stat']:.1f}  "
                  f"p={r['p_wilcox']:.4f} {sig_w}")
        print()

    print("  Per-seed detail:")
    print(f"  {'Seed':<8} {'Calmar':>8} {'Dist':>8} {'MDD':>8} {'Depl':>8}  Status")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  ------")
    for s, d in res["per_seed_detail"].items():
        status = "active" if d["active"] else "DEGENERATE"
        print(f"  {s:<8} {d['calmar']:>8.4f} {d['dist']:>8.4f} "
              f"{d['mdd']:>8.4f} {d['depl']:>8.4f}  {status}")
    print("=" * w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Statistical tests for the Wtp DRL evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path",  default=MODEL_PATH)
    p.add_argument("--ms-json",     default=MS_JSON_PATH)
    p.add_argument("--results-dir", default="src/models/run_007")
    p.add_argument("--no-dm",       action="store_true", help="Skip DM tests")
    p.add_argument("--no-seed",     action="store_true", help="Skip seed analysis")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out  = {}

    print("=" * 70)
    print("Wtp DRL — Statistical Significance Tests")
    print("=" * 70)

    # ---- DM tests -------------------------------------------------------- #
    if not args.no_dm:
        print("\n[1/2] Diebold–Mariano tests...")
        results = run_pipeline()
        env_cfg = EnvConfig()
        dm_res  = run_dm_tests(results, env_cfg, args.model_path)
        _print_dm(dm_res)
        out["diebold_mariano"] = dm_res
    else:
        print("\n[1/2] Skipping DM tests (--no-dm)")

    # ---- Seed significance ----------------------------------------------- #
    if not args.no_seed:
        print("\n[2/2] Multi-seed significance tests...")
        ms_path = Path(args.ms_json)
        if not ms_path.exists():
            print(f"  WARNING: {ms_path} not found — run multiseed.py first")
        else:
            with open(ms_path) as f:
                ms_json = json.load(f)
            seed_res = run_seed_significance(ms_json)
            _print_seed(seed_res)
            out["seed_significance"] = seed_res
    else:
        print("\n[2/2] Skipping seed analysis (--no-seed)")

    # ---- Save ------------------------------------------------------------ #
    out_path = Path(args.results_dir) / "stats_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
