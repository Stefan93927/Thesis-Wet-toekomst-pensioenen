"""run_hoevenaars.py — Driver for the Hoevenaars multi-factor ALM baseline.

Usage
-----
    # Full run (calibrate + generate + optimise + evaluate):
    py run_hoevenaars.py

    # Skip expensive steps if outputs already exist:
    py run_hoevenaars.py --skip-scenarios --skip-bo

    # Adjust number of BO trials per lambda (default 50):
    py run_hoevenaars.py --trials 30

All outputs written to results/:
    hoevenaars_var_calibration.json
    hoevenaars_scenarios.npz
    hoevenaars_pareto_front.csv
    hoevenaars_pareto_front.png
    hoevenaars_coverage.png
    hoevenaars_headline_policy.json
    hoevenaars_multipath.csv
    hoevenaars_historical.csv
    hoevenaars_summary.md
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

from src.data_pipeline  import run_pipeline
from src.environment    import EnvConfig
from src.hoevenaars_alm import (
    calibrate_var,
    generate_scenarios,
    run_bo,
    extract_pareto,
    select_headline,
    evaluate_multipath,
    evaluate_historical,
    historical_coverage,
    inflation_coverage_diagnostic,
    plot_pareto,
    plot_coverage,
    plot_pareto_vs_drl,
    write_summary,
    save_calibration,
    save_scenarios,
    load_scenarios,
    N_TRIALS_PER_LAMBDA,
    LAMBDAS,
)

_RESULTS = _ROOT / "results"
_RESULTS.mkdir(exist_ok=True)

# Calibration and scenarios never change between v1/v2 — always use base names.
P_CALIB = _RESULTS / "hoevenaars_var_calibration.json"
P_SCEN  = _RESULTS / "hoevenaars_scenarios.npz"


def _make_output_paths(prefix: str) -> dict:
    """Return a dict of output Path objects keyed by short name."""
    r = _RESULTS
    return {
        "pareto":        r / f"{prefix}_pareto_front.csv",
        "pareto_png":    r / f"{prefix}_pareto_front.png",
        "coverage_png":  r / f"{prefix}_coverage.png",
        "headline":      r / f"{prefix}_headline_policy.json",
        "multipath":     r / f"{prefix}_multipath.csv",
        "historical":    r / f"{prefix}_historical.csv",
        "summary":       r / f"{prefix}_summary.md",
        "pvd_png":       r / f"{prefix}_pareto_vs_drl.png",
        "pvd_csv":       r / f"{prefix}_pareto_vs_drl.csv",
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-scenarios", action="store_true",
                   help="Load scenarios from existing .npz instead of regenerating.")
    p.add_argument("--skip-bo", action="store_true",
                   help="Load Pareto CSV from existing file instead of re-running BO.")
    p.add_argument("--trials", type=int, default=N_TRIALS_PER_LAMBDA,
                   help="BO trials per lambda value (default: 50).")
    p.add_argument("--mvev-tol", type=float, default=0.05,
                   help="P(MVEV_breach) upper bound for BO Run 2 (default: 0.05).")
    p.add_argument("--prefix", type=str, default="hoevenaars",
                   help="Output filename prefix (default: hoevenaars). "
                        "Use 'hoevenaars_v2' for the fixed-a specification.")
    p.add_argument("--a-fixed", type=float, default=None,
                   help="Fix strategic equity weight a at this value; "
                        "optimise only (b,c,g,e). Overrides --a-low/--a-high.")
    p.add_argument("--a-low", type=float, default=0.55,
                   help="Lower bound for a search (default: 0.55). "
                        "Ignored when --a-fixed is set.")
    p.add_argument("--a-high", type=float, default=0.80,
                   help="Upper bound for a search (default: 0.80). "
                        "Use 0.60 for v3 Dutch-industry range.")
    p.add_argument("--b-max", type=float, default=1.50,
                   help="Upper bound for de-risking slope b (default: 1.50).")
    p.add_argument("--v4", action="store_true",
                   help="Use v4 extended policy class: 8 params "
                        "(a, h, i, c, g, e, B_target, B_min) with VSTOXX tilt. "
                        "Requires --a-low 0.50 --a-high 0.58.")
    p.add_argument("--budget", type=int, default=None,
                   help="Total BO evaluations per run (overrides --trials). "
                        "E.g. --budget 400 gives 66 trials per lambda for v4.")
    return p.parse_args()


def _load_existing_robustness() -> list[dict]:
    """Load DRL, Fixed-Rule, Constrained MC metrics from robustness_historical.csv."""
    path = _RESULTS / "robustness_historical.csv"
    if not path.exists():
        print("  WARNING: robustness_historical.csv not found; "
              "comparison table will be incomplete.")
        return []
    df   = pd.read_csv(path)
    rows = []
    for policy in df["policy"].unique():
        sub  = df[df["policy"] == policy].set_index("metric")["value"].to_dict()
        sub["policy"] = policy
        rows.append(sub)
    return rows


def main() -> None:
    args = _parse_args()
    P    = _make_output_paths(args.prefix)

    # Trials per lambda: --budget takes precedence over --trials
    n_trials_per_lambda = (
        args.budget // len(LAMBDAS) if args.budget
        else args.trials
    )

    print("=" * 72)
    print(f"Hoevenaars Multi-Factor ALM Baseline  [prefix={args.prefix}]")
    if args.v4:
        spec_label = "v4 extended (VSTOXX tilt)"
        a_spec = f"a in [{args.a_low:.2f}, {args.a_high:.2f}]"
        print(f"  Specification: {a_spec}, h/i VSTOXX tilt, 8-param policy  ({spec_label})")
        print(f"  Budget: {n_trials_per_lambda} trials x {len(LAMBDAS)} lambdas = "
              f"{n_trials_per_lambda*len(LAMBDAS)} evaluations per run")
    elif args.a_fixed is not None:
        a_spec = f"a fixed={args.a_fixed:.2f}"
        spec_label = "v2 fixed-a"
        print(f"  Specification: {a_spec}, b in [0, {args.b_max:.2f}]  ({spec_label})")
    else:
        a_spec = f"a in [{args.a_low:.2f}, {args.a_high:.2f}]"
        spec_label = "v3 industry-range" if args.a_high <= 0.62 else "v1 wide-range"
        print(f"  Specification: {a_spec}, b in [0, {args.b_max:.2f}]  ({spec_label})")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # 1. Data pipeline                                                    #
    # ------------------------------------------------------------------ #
    print("\n[1/9] Running data pipeline...")
    results = run_pipeline()
    env_cfg = EnvConfig()

    z_train_raw = results["z_train_raw"]
    z_val_raw   = results["z_val_raw"]
    cpi         = results["cpi"]

    # mu_vstoxx: 67th percentile of training-period VSTOXX (raw, unscaled)
    mu_vstoxx = float(np.percentile(z_train_raw["vstoxx_level"].values, 67))
    if args.v4:
        print(f"  mu_VSTOXX (67th pct of training period, 2000-2015): {mu_vstoxx:.2f}")

    # ------------------------------------------------------------------ #
    # 2. VAR calibration (always shared — prefix-independent)            #
    # ------------------------------------------------------------------ #
    print("\n[2/9] Calibrating VAR(1)...")
    if P_CALIB.exists():
        print(f"  Loading existing calibration from {P_CALIB.name}")
        with open(P_CALIB) as f:
            calib_json = json.load(f)
        calib = dict(calib_json)
        calib["_np_coefs"]     = np.array(calib["var_coefs"])
        calib["_np_intercept"] = np.array(calib["intercept"])
        calib["_np_sigma"]     = np.array(calib["sigma_resid"])
    else:
        calib = calibrate_var(z_train_raw, cpi)
        save_calibration(calib, P_CALIB)
        print(f"  Saved: {P_CALIB.name}")

    _print_adf_summary(calib)

    # ------------------------------------------------------------------ #
    # 3. Scenario generation (always shared — prefix-independent)        #
    # ------------------------------------------------------------------ #
    print("\n[3/9] Generating 1,000 VAR scenarios (84 months each)...")
    if args.skip_scenarios and P_SCEN.exists():
        print(f"  Loading existing scenarios from {P_SCEN.name}")
        scenarios = load_scenarios(P_SCEN)
    else:
        scenarios = generate_scenarios(calib, z_val_raw, cpi)
        save_scenarios(scenarios, P_SCEN)
        print(f"  Saved: {P_SCEN.name}")

    if "neg_rate_frac" not in scenarios:
        scenarios["neg_rate_frac"] = float((scenarios["y_long"] < 0).mean())
    calib["neg_rate_frac_from_scenarios"] = scenarios["neg_rate_frac"]

    # ------------------------------------------------------------------ #
    # 4. Bayesian optimisation                                            #
    # ------------------------------------------------------------------ #
    print("\n[4/9] Bayesian optimisation (Run 1 + Run 2)...")
    if args.skip_bo and P["pareto"].exists():
        print(f"  Loading existing candidates from {P['pareto'].name}")
        candidates_df = pd.read_csv(P["pareto"])
    else:
        bo_kwargs = dict(
            n_trials_per_lambda = n_trials_per_lambda,
            a_fixed             = args.a_fixed,
            b_max               = args.b_max,
            a_low               = args.a_low,
            a_high              = args.a_high,
            use_v4              = args.v4,
            mu_vstoxx           = mu_vstoxx if args.v4 else None,
        )
        print("  Run 1: maximise E[total_dist] s.t. P(depletion) <= 0.10")
        df_r1 = run_bo(scenarios, run_type="run1", **bo_kwargs)
        print(f"  Run 2: maximise E[FR_T]        s.t. P(MVEV_breach) <= {args.mvev_tol:.2f}")
        df_r2 = run_bo(scenarios, run_type="run2",
                       mvev_constraint=args.mvev_tol, **bo_kwargs)
        candidates_df = pd.concat([df_r1, df_r2], ignore_index=True)
        candidates_df.to_csv(P["pareto"], index=False)
        print(f"  Saved: {P['pareto'].name}  ({len(candidates_df)} candidates)")

    # MVEV satisfiability diagnostic
    min_pmvev = candidates_df["P_mvev"].min()
    print(f"  MVEV satisfiability: min P_mvev across all candidates = {min_pmvev:.4f}"
          f"  ({'constraint INFEASIBLE' if min_pmvev > args.mvev_tol else 'feasible region exists'})")

    # ------------------------------------------------------------------ #
    # 5. Pareto front + headline selection                                #
    # ------------------------------------------------------------------ #
    print("\n[5/9] Extracting Pareto front and selecting headline policy...")
    pareto_df = extract_pareto(candidates_df)
    print(f"  Pareto front: {len(pareto_df)} non-dominated candidates")

    headline  = select_headline(pareto_df)
    params    = np.array(list(headline["params"].values()))
    if args.v4:
        p = headline["params"]
        print(f"  Headline: a={p['a']:.3f}, h={p['h']:.3f}, i={p['i']:.3f}, "
              f"c={p['c']:.3f}, g={p['g']:.3f}, e={p['e']:.3f}, "
              f"B_target={p['B_target']:.3f}, B_min={p['B_min']:.3f}")
    else:
        print(f"  Headline: a={params[0]:.3f}, b={params[1]:.3f}, "
              f"c={params[2]:.3f}, g={params[3]:.3f}, e={params[4]:.3f}")
    print(f"  E[total_dist]={headline['selected_dist']:.4f} "
          f"(target={headline['target_dist_drl']:.4f}, "
          f"gap={headline['dist_gap']:.4f})")

    with open(P["headline"], "w") as f:
        json.dump(headline, f, indent=2)
    print(f"  Saved: {P['headline'].name}")

    # ------------------------------------------------------------------ #
    # 6. Pareto plot                                                      #
    # ------------------------------------------------------------------ #
    print("\n[6/9] Plotting Pareto front...")
    plot_pareto(candidates_df, pareto_df, headline, P["pareto_png"])

    # ------------------------------------------------------------------ #
    # 7. Multi-path evaluation                                            #
    # ------------------------------------------------------------------ #
    print("\n[7/9] Evaluating headline policy on 1,000 scenarios...")
    multipath_df = evaluate_multipath(
        scenarios, params,
        mu_vstoxx=mu_vstoxx if args.v4 else None,
    )
    multipath_df.to_csv(P["multipath"], index=False)
    print(f"  Saved: {P['multipath'].name}")
    print(multipath_df.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 8. Historical-path evaluation                                       #
    # ------------------------------------------------------------------ #
    print("\n[8/9] Evaluating headline policy on Jan 2018 - Dec 2025...")
    hist_metrics, hist_FR = evaluate_historical(
        params, results, env_cfg,
        mu_vstoxx=mu_vstoxx if args.v4 else None,
    )

    hist_df = pd.DataFrame([
        {"policy": "Hoevenaars", "metric": k, "value": v}
        for k, v in hist_metrics.items()
    ])
    hist_df.to_csv(P["historical"], index=False)
    print(f"  Saved: {P['historical'].name}")
    _print_hist_row("Hoevenaars", hist_metrics)

    # ------------------------------------------------------------------ #
    # 9. Historical coverage diagnostic                                   #
    # ------------------------------------------------------------------ #
    print("\n[9/9] Coverage diagnostics...")
    coverage = historical_coverage(
        scenarios, params, hist_FR,
        mu_vstoxx=mu_vstoxx if args.v4 else None,
    )
    plot_coverage(coverage, hist_FR, P["coverage_png"])
    print(f"  FR path coverage (p5-p95 band): {coverage['coverage_p5_p95']*100:.1f}%")

    test_dates     = results["z_test"].index
    inflation_diag = inflation_coverage_diagnostic(scenarios, cpi, test_dates)
    peak_pct = inflation_diag["pi_realized_max_2022"] * 100
    p95_pct  = inflation_diag["pi_sim_p95_2022_mean"] * 100
    cov_pct  = inflation_diag["paths_covering_2022_peak"] * 100
    pi_band  = inflation_diag["pi_overall_coverage"] * 100
    covered  = cov_pct > 10
    print(f"  Inflation - 2022 realized peak: {peak_pct:.3f}%/month  "
          f"simulated p95: {p95_pct:.3f}%/month  "
          f"paths covering peak: {cov_pct:.1f}%  "
          f"{'OK' if covered else 'WARNING: 2022 peak NOT covered by simulated distribution'}")
    print(f"  Overall pi band coverage (test period): {pi_band:.1f}%")

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    comparison_rows = _load_existing_robustness()

    summary_md = write_summary(
        calib           = calib,
        scenarios       = scenarios,
        headline        = headline,
        multipath_df    = multipath_df,
        hist_metrics    = hist_metrics,
        coverage        = coverage,
        pareto_df       = pareto_df,
        comparison_rows = comparison_rows,
        inflation_diag  = inflation_diag,
    )
    P["summary"].write_text(summary_md, encoding="utf-8")
    print(f"\n  Summary saved: {P['summary'].name}")

    # ------------------------------------------------------------------ #
    # Option C: Pareto-vs-DRL comparison plot                            #
    # ------------------------------------------------------------------ #
    print("\n[Option C] Pareto-front comparison plot vs DRL/Fixed-Rule...")
    pareto_vs_drl_df = plot_pareto_vs_drl(
        candidates_df     = candidates_df,
        pareto_df         = pareto_df,
        headline          = headline,
        comparison_rows   = comparison_rows,
        hoev_hist_metrics = hist_metrics,
        out_path          = P["pvd_png"],
    )
    pareto_vs_drl_df.to_csv(P["pvd_csv"], index=False)
    print(f"  CSV saved: {P['pvd_csv'].name}  ({len(pareto_vs_drl_df)} rows)")

    # ------------------------------------------------------------------ #
    # Final comparison table                                              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("Comparison table - historical path (Jan 2018 - Dec 2025)")
    print("=" * 72)
    _print_comparison_table(comparison_rows, hist_metrics)

    print("\nDone.")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_adf_summary(calib: dict) -> None:
    print("  ADF stationarity decisions:")
    for r in calib.get("adf_tests", []):
        status = ("stationary"
                  if r["stationary_at_05"]
                  else "Unit root detected — series differenced for VAR")
        print(f"    {r['variable']:<25}  p={r['p_value']:.4f}  {status}")
    vstoxx_mode = "differenced" if calib.get("vstoxx_differenced") else "level"
    print(f"    VSTOXX enters VAR as: {vstoxx_mode}")


def _print_hist_row(name: str, m: dict) -> None:
    print(
        f"  {name:<22}  "
        f"FR_T={m.get('fr_terminal', float('nan')):.4f}  "
        f"MDD={m.get('fr_mdd', float('nan')):.4f}  "
        f"dep={m.get('buffer_depletion_freq', float('nan')):.4f}  "
        f"dist={m.get('total_distributions', float('nan')):.4f}  "
        f"Calmar={m.get('calmar_ratio', float('nan')):.4f}"
    )


def _print_comparison_table(comparison_rows: list[dict], hist_metrics: dict) -> None:
    metrics = [
        "fr_terminal", "fr_mdd", "buffer_depletion_freq",
        "total_distributions", "calmar_ratio", "mvev_breach_count",
    ]
    comp = {r["policy"]: r for r in comparison_rows}

    header = f"{'Metric':<28}  {'DRL':>8}  {'Fixed-Rule':>10}  {'MC':>8}  {'Hoevenaars':>12}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        drl   = comp.get("DRL (PPO)",     {}).get(m, float("nan"))
        fixed = comp.get("Fixed-Rule",    {}).get(m, float("nan"))
        mc    = comp.get("Constrained MC",{}).get(m, float("nan"))
        hoev  = hist_metrics.get(m, float("nan"))
        print(
            f"  {m:<26}  {drl:8.4f}  {fixed:10.4f}  {mc:8.4f}  {hoev:12.4f}"
        )


if __name__ == "__main__":
    main()
