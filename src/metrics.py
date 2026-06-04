"""metrics.py — Evaluation metrics for the Wtp DRL pension fund agent.

All functions operate on the trajectory dictionary returned by
``baselines.run_episode()`` and produce scalar summaries suitable for
reporting and regime-conditional analysis.

Metrics implemented
-------------------
- FR Terminal Level
- FR Maximum Drawdown (MDD)
- FR Annualised Volatility
- Buffer Depletion Frequency  P̂(B_t <= 0.001)
- Total Distributions  sum(d̃_t)
- Calmar Ratio  (annualised mean FR growth / |MDD|)
- Mean Cross-Cohort Real Replacement Rate Variance  E[Var(RR_1, RR_2, RR_3)]
- Per-cohort mean 12M real replacement rates  (Young, Mid-career, Retired)
- Regime-conditional breakdowns (Low / Medium / High VSTOXX)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    trajectory:  dict,
    pi_monthly:  Optional[np.ndarray] = None,
    lookback:    int = 12,
) -> dict:
    """Compute all evaluation metrics from a single trajectory.

    Args:
        trajectory:  Dict returned by ``baselines.run_episode()``.  Must
                     contain ``"FR"``, ``"B"``, ``"r_p"``, ``"d_tilde"``.
        pi_monthly:  Optional ``(T,)`` array of monthly CPI inflation aligned
                     to the trajectory.  Zeros assumed if not provided.
        lookback:    Rolling window (months) for replacement-rate calculation.

    Returns:
        Dictionary with keys:

        ``fr_terminal``, ``fr_mdd``, ``fr_vol_ann``,
        ``buffer_depletion_freq``, ``total_distributions``,
        ``calmar_ratio``, ``cohort_rr_var``,
        ``rr_mean_young``, ``rr_mean_mid``, ``rr_mean_ret``.
    """
    FR   = np.asarray(trajectory["FR"],      dtype=np.float64)
    B    = np.asarray(trajectory["B"],       dtype=np.float64)
    r_p  = np.asarray(trajectory["r_p"],     dtype=np.float64)
    dist = np.asarray(trajectory["d_tilde"], dtype=np.float64)
    T    = len(FR)

    pi = (
        np.asarray(pi_monthly, dtype=np.float64)
        if pi_monthly is not None and len(pi_monthly) == T
        else np.zeros(T, dtype=np.float64)
    )

    # ---- FR metrics ------------------------------------------------------ #
    fr_terminal    = float(FR[-1])
    running_max    = np.maximum.accumulate(FR)
    drawdowns      = (running_max - FR) / np.maximum(running_max, 1e-8)
    fr_mdd         = float(drawdowns.max())

    # Annualised volatility of monthly FR changes
    fr_changes     = np.diff(FR)
    fr_vol_ann     = float(fr_changes.std() * np.sqrt(12)) if len(fr_changes) > 1 else 0.0

    # ---- Buffer depletion ------------------------------------------------ #
    buffer_depletion_freq = float((B <= 0.001).mean())

    # ---- Distributions --------------------------------------------------- #
    total_distributions = float(dist.sum())

    # ---- Calmar ratio ---------------------------------------------------- #
    # Annualised mean monthly FR growth / max drawdown
    fr_monthly_growth = fr_changes / np.maximum(FR[:-1], 1e-8)
    fr_ann_growth     = float(fr_monthly_growth.mean() * 12) if len(fr_monthly_growth) > 0 else 0.0
    calmar_ratio      = fr_ann_growth / (fr_mdd + 1e-8)

    # ---- Intergenerational equity: cohort RR variance ------------------- #
    # PPV framework (run_010+, use_lifecycle=True):
    #   R_{i,t} = w_i^eq*r_eq + (1-w_i^eq)*r_bond - f̃_t + (d̃_t + dec_excess)
    #   All cohorts bear the same solidarity cost and receive the same benefit;
    #   variance comes purely from the lifecycle equity mix difference.
    # Legacy mode (use_lifecycle=False, run_007/008b):
    #   Young=r_p_agg, Mid=DeltaFR/FR, Retired=dist/0.45
    has_lifecycle = ("r_p_young" in trajectory and "r_p_mid" in trajectory
                     and "r_p_ret" in trajectory)
    if has_lifecycle:
        r_p_young  = np.asarray(trajectory["r_p_young"],              dtype=np.float64)
        r_p_mid    = np.asarray(trajectory["r_p_mid"],                dtype=np.float64)
        r_p_ret    = np.asarray(trajectory["r_p_ret"],                dtype=np.float64)
        f_fill     = np.asarray(trajectory.get("f_tilde",    np.zeros(T)), dtype=np.float64)
        dec_exc    = np.asarray(trajectory.get("dec_excess", np.zeros(T)), dtype=np.float64)
        total_dist = dist + dec_exc
        r_young = r_p_young - f_fill + total_dist   # PPV formula
        r_mid   = r_p_mid   - f_fill + total_dist
        r_ret   = r_p_ret   - f_fill + total_dist
    else:
        FR_prev = np.concatenate([[FR[0]], FR[:-1]])
        r_young = r_p
        r_mid   = (FR - FR_prev) / np.maximum(FR_prev, 1e-8)
        r_ret   = dist / 0.45

    rr_young_list: list[float] = []
    rr_mid_list:   list[float] = []
    rr_ret_list:   list[float] = []
    rr_vars:       list[float] = []

    for t in range(lookback, T):
        rr = []
        for cohort_ret in (r_young, r_mid, r_ret):
            log_sum = 0.0
            for k in range(lookback):
                idx     = t - lookback + k
                arg     = 1.0 + cohort_ret[idx] - pi[idx]
                log_sum += np.log(max(arg, 1e-8))
            rr.append(log_sum)
        rr_young_list.append(rr[0])
        rr_mid_list.append(rr[1])
        rr_ret_list.append(rr[2])
        rr_vars.append(float(np.var(rr)))

    cohort_rr_var = float(np.mean(rr_vars))       if rr_vars else 0.0
    rr_mean_young = float(np.mean(rr_young_list)) if rr_young_list else 0.0
    rr_mean_mid   = float(np.mean(rr_mid_list))   if rr_mid_list else 0.0
    rr_mean_ret   = float(np.mean(rr_ret_list))   if rr_ret_list else 0.0

    # ---- Terminal PPV (Personal Pension Capital) ------------------------- #
    # Available when use_lifecycle=True; PPV is updated in-environment each step.
    ppv_young_term = float(trajectory["ppv_young"][-1]) if "ppv_young" in trajectory else float("nan")
    ppv_mid_term   = float(trajectory["ppv_mid"][-1])   if "ppv_mid"   in trajectory else float("nan")
    ppv_ret_term   = float(trajectory["ppv_ret"][-1])   if "ppv_ret"   in trajectory else float("nan")

    return {
        "fr_terminal":           fr_terminal,
        "fr_mdd":                fr_mdd,
        "fr_vol_ann":            fr_vol_ann,
        "buffer_depletion_freq": buffer_depletion_freq,
        "total_distributions":   total_distributions,
        "calmar_ratio":          calmar_ratio,
        "cohort_rr_var":         cohort_rr_var,
        "rr_mean_young":         rr_mean_young,
        "rr_mean_mid":           rr_mean_mid,
        "rr_mean_ret":           rr_mean_ret,
        "ppv_young_term":        ppv_young_term,
        "ppv_mid_term":          ppv_mid_term,
        "ppv_ret_term":          ppv_ret_term,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    trajectory:  dict,
    pi_monthly:  Optional[np.ndarray] = None,
    n_boot:      int = 1000,
    ci:          float = 0.95,
    seed:        int = 0,
) -> dict[str, tuple[float, float]]:
    """Compute bootstrap confidence intervals for all core metrics.

    Resamples T timesteps with replacement; indices are sorted before
    applying so that the bootstrap path preserves temporal ordering
    (standard approach for financial time-series CIs).

    Args:
        trajectory:  Dict returned by ``run_episode()``.
        pi_monthly:  Optional monthly CPI array ``(T,)``.
        n_boot:      Number of bootstrap replications (default 1000).
        ci:          Confidence level (default 0.95 -> 95% CI).
        seed:        RNG seed for reproducibility.

    Returns:
        Dict ``{metric_key: (lower, upper)}`` with the same keys as
        :func:`compute_metrics`.
    """
    rng  = np.random.default_rng(seed)
    T    = len(trajectory["FR"])
    traj_keys = [k for k in ["FR", "B", "r_p", "d_tilde", "f_tilde", "dec_excess",
                              "r_p_young", "r_p_mid", "r_p_ret",
                              "ppv_young", "ppv_mid", "ppv_ret"]
                 if k in trajectory]
    pi   = (
        np.asarray(pi_monthly, dtype=np.float64)
        if pi_monthly is not None and len(pi_monthly) == T
        else np.zeros(T, dtype=np.float64)
    )

    metric_keys = [
        "fr_terminal", "fr_mdd", "fr_vol_ann",
        "buffer_depletion_freq", "total_distributions",
        "calmar_ratio", "cohort_rr_var",
        "rr_mean_young", "rr_mean_mid", "rr_mean_ret",
        "ppv_young_term", "ppv_mid_term", "ppv_ret_term",
    ]
    boot_samples: dict[str, list[float]] = {k: [] for k in metric_keys}

    for _ in range(n_boot):
        idx    = np.sort(rng.integers(0, T, size=T))   # sorted -> temporal order
        sub    = {k: np.asarray(trajectory[k])[idx] for k in traj_keys}
        sub_pi = pi[idx]
        m      = compute_metrics(sub, pi_monthly=sub_pi)
        for k in metric_keys:
            boot_samples[k].append(m[k])

    alpha = (1.0 - ci) / 2.0
    return {
        k: (
            float(np.percentile(boot_samples[k], 100 * alpha)),
            float(np.percentile(boot_samples[k], 100 * (1.0 - alpha))),
        )
        for k in metric_keys
    }


def format_ci_table(
    metrics_by_agent: dict[str, dict],
    ci_by_agent:      dict[str, dict[str, tuple[float, float]]],
    title:            str = "Metrics with 95% Bootstrap CI",
) -> str:
    """Format a metrics table with 95% CI brackets below each value.

    Args:
        metrics_by_agent: ``{agent: metrics_dict}`` from :func:`compute_metrics`.
        ci_by_agent:      ``{agent: {metric: (lo, hi)}}`` from :func:`bootstrap_ci`.
        title:            Table title.

    Returns:
        Multi-line string ready for printing.
    """
    display = [
        ("fr_terminal",           "FR Terminal"),
        ("fr_mdd",                "FR MDD"),
        ("fr_vol_ann",            "FR Vol (ann)"),
        ("buffer_depletion_freq", "Buf Depl Freq"),
        ("total_distributions",   "Total Dist"),
        ("calmar_ratio",          "Calmar"),
        ("cohort_rr_var",         "Cohort RR Var"),
        ("rr_mean_young",         "RR Mean Young"),
        ("rr_mean_mid",           "RR Mean Mid"),
        ("rr_mean_ret",           "RR Mean Ret"),
        ("ppv_young_term",        "PPV Young (T)"),
        ("ppv_mid_term",          "PPV Mid (T)"),
        ("ppv_ret_term",          "PPV Ret (T)"),
    ]

    agents = list(metrics_by_agent.keys())
    col_w  = 20

    lines = [title, "=" * (22 + col_w * len(agents))]
    header = f"  {'Metric':<20}" + "".join(f"{a:^{col_w}}" for a in agents)
    lines += [header, "-" * len(header)]

    for key, label in display:
        val_row = f"  {label:<20}"
        ci_row  = f"  {'':20}"
        for agent in agents:
            val = metrics_by_agent[agent].get(key, float("nan"))
            lo, hi = ci_by_agent.get(agent, {}).get(key, (float("nan"), float("nan")))
            val_row += f"{val:^{col_w}.4f}"
            ci_row  += f"{'['+f'{lo:.4f}, {hi:.4f}'+']':^{col_w}}"
        lines.append(val_row)
        lines.append(ci_row)

    lines.append("=" * (22 + col_w * len(agents)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regime-conditional analysis
# ---------------------------------------------------------------------------

def regime_conditional_metrics(
    trajectory:     dict,
    vstoxx_series:  pd.Series,
    pi_monthly:     Optional[np.ndarray] = None,
    thresholds:     tuple = (20.0, 30.0),
) -> dict:
    """Break trajectory metrics down by VSTOXX volatility regime.

    Regimes are defined by VSTOXX level:
    - Low    : VSTOXX < thresholds[0]
    - Medium : thresholds[0] <= VSTOXX < thresholds[1]
    - High   : VSTOXX >= thresholds[1]

    Args:
        trajectory:    Dict returned by ``run_episode()``.
        vstoxx_series: Monthly VSTOXX values aligned to the trajectory dates.
                       Index must be compatible with ``trajectory["dates"]``.
        pi_monthly:    Optional inflation array  ``(T,)``.
        thresholds:    ``(low_hi, mid_hi)`` VSTOXX cutoffs.

    Returns:
        Dict with keys ``"Low"``, ``"Medium"``, ``"High"``, each containing
        a metrics sub-dict (same keys as :func:`compute_metrics`) plus
        ``"n_months"``.  Returns an empty sub-dict for regimes with no data.
    """
    dates  = trajectory["dates"]
    T      = len(dates)
    lo, hi = thresholds

    # Align VSTOXX to trajectory dates
    vstoxx_aligned = (
        vstoxx_series.reindex(pd.DatetimeIndex(dates)).ffill().bfill().values
    )

    regime_masks = {
        "Low":    vstoxx_aligned < lo,
        "Medium": (vstoxx_aligned >= lo) & (vstoxx_aligned < hi),
        "High":   vstoxx_aligned >= hi,
    }

    results = {}
    for name, mask in regime_masks.items():
        n = int(mask.sum())
        if n < 2:
            results[name] = {"n_months": n}
            continue

        sub_traj = {
            k: (np.asarray(v)[mask] if isinstance(v, (np.ndarray, list)) else v)
            for k, v in trajectory.items()
            if k != "dates"
        }
        sub_traj["dates"] = [d for d, m in zip(dates, mask) if m]

        sub_pi = (
            np.asarray(pi_monthly)[mask]
            if pi_monthly is not None and len(pi_monthly) == T
            else None
        )

        m = compute_metrics(sub_traj, sub_pi)
        m["n_months"] = n
        results[name] = m

    return results


# ---------------------------------------------------------------------------
# Diebold-Mariano predictive accuracy test
# ---------------------------------------------------------------------------

def diebold_mariano(
    loss_ref:        np.ndarray,
    loss_challenger: np.ndarray,
    h:               int = 1,
) -> tuple[float, float]:
    """Harvey-Leybourne-Newbold modified Diebold-Mariano test (1997).

    Economic motivation
    -------------------
    The DM test answers: "Is the DRL agent's period-by-period loss
    significantly lower than the baseline's, accounting for serial
    correlation in the loss differential?"  For a pension fund this is
    more informative than comparing scalar metric averages, because it
    exploits all 84 monthly observations and gives a formal p-value
    suitable for thesis reporting.

    Test formulation
    ----------------
    Define the loss differential  d_t = L_t^ref - L_t^challenger.
    H0: E[d_t] = 0  (equal predictive accuracy).
    A positive HLN statistic means the challenger has lower expected
    loss (challenger wins).

    The Harvey-Leybourne-Newbold correction replaces the standard normal
    with a t_{T-1} distribution and rescales the DM statistic to account
    for small-sample bias — important here because T = 84 months.

    Newey-West HAC variance uses h-1 lags.  For monthly pension fund
    returns with h=1 (no multi-step aggregation) this reduces to the
    ordinary sample variance of d_t, which is appropriate.

    Args:
        loss_ref:        Per-period loss array for the reference model ``(T,)``.
        loss_challenger: Per-period loss array for the challenger model ``(T,)``.
        h:               Forecast horizon (1 for single-step monthly).

    Returns:
        ``(hln_stat, p_value)`` — two-sided p-value from a t_{T-1} distribution.
        Positive stat => challenger (DRL) has lower expected loss.
    """
    d    = np.asarray(loss_ref, dtype=np.float64) - np.asarray(loss_challenger, dtype=np.float64)
    T    = len(d)
    dbar = d.mean()

    # Newey-West HAC variance with h-1 lags (for h=1: plain sample variance)
    gamma0  = np.var(d, ddof=1)
    nw_var  = gamma0
    for k in range(1, h):
        gammak  = float(np.cov(d[k:], d[:-k])[0, 1])
        nw_var += 2.0 * (1.0 - k / h) * gammak

    dm_stat = dbar / np.sqrt(max(nw_var / T, 1e-14))

    # HLN rescaling factor (Harvey, Leybourne & Newbold 1997, eq. 4)
    hln_factor = np.sqrt((T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T)
    hln_stat   = dm_stat * hln_factor
    p_value    = float(2.0 * _scipy_stats.t.sf(abs(hln_stat), df=T - 1))

    return float(hln_stat), p_value


def dm_losses(trajectory: dict, fr_target: float = 1.05) -> dict[str, np.ndarray]:
    """Extract per-period loss arrays used in DM tests.

    Two loss functions are returned:
    - ``"buf_depl"``  : Binary indicator 1(B_t <= 0.001).  Directly measures
                        the Wtp Art. 10d buffer adequacy failure at each month.
                        This is the primary loss for the DM test because buffer
                        depletion is the core policy objective.
    - ``"fr_sq_dev"`` : (FR_t - fr_target)^2.  Penalises FR deviation from the
                        regulatory target, capturing both under- and over-funding.

    Args:
        trajectory: Dict from ``run_episode()`` containing ``"FR"`` and ``"B"``.
        fr_target:  Target funding ratio (default 1.05, DNB recommendation).

    Returns:
        Dict with keys ``"buf_depl"`` and ``"fr_sq_dev"``, each ``(T,)`` array.
    """
    FR = np.asarray(trajectory["FR"], dtype=np.float64)
    B  = np.asarray(trajectory["B"],  dtype=np.float64)
    return {
        "buf_depl":  (B  <= 0.001).astype(np.float64),
        "fr_sq_dev": (FR - fr_target) ** 2,
    }


def _sig_stars(p: float) -> str:
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    if p < 0.10: return "."
    return ""


def format_dm_table(
    dm_results: dict[str, dict[str, tuple[float, float]]],
    T:          int,
    h:          int = 1,
    title:      str = "Diebold-Mariano Predictive Accuracy Tests",
) -> str:
    """Format a Diebold-Mariano results table.

    Args:
        dm_results: Nested dict ``{comparison_label: {loss_label: (stat, p)}}``.
                    Positive stat means the challenger (DRL) has lower loss.
        T:          Number of test-period observations (printed in header).
        h:          Forecast horizon (printed in header).
        title:      Table title.

    Returns:
        Multi-line string ready for printing.
    """
    comparisons = list(dm_results.keys())
    loss_labels = list(next(iter(dm_results.values())).keys())

    comp_w  = max(len(c) for c in comparisons) + 2
    cell_w  = 22   # "  8.42  <0.001  **"

    # Header
    loss_header = "".join(f"{lbl:^{cell_w}}" for lbl in loss_labels)
    sub_header  = "".join(f"{'Stat':>7}{'p-val':>8}{'':>7}" for _ in loss_labels)
    sep         = "-" * (comp_w + len(loss_header))

    lines = [
        title,
        f"  H0: equal predictive accuracy  |  positive stat => DRL wins  "
        f"|  HLN correction, T={T}, h={h}",
        "=" * (comp_w + len(loss_header)),
        f"  {'Comparison':<{comp_w}}" + loss_header,
        f"  {'':^{comp_w}}" + sub_header,
        sep,
    ]

    for comp, losses in dm_results.items():
        row = f"  {comp:<{comp_w}}"
        for lbl in loss_labels:
            stat, p = losses.get(lbl, (float("nan"), float("nan")))
            p_str   = f"<0.001" if p < 0.001 else f"{p:.3f}"
            sig     = _sig_stars(p)
            row    += f"  {stat:>6.2f}  {p_str:>6}  {sig:<2}"
        lines.append(row)

    lines.append(sep)
    lines.append("  ** p<0.01   * p<0.05   . p<0.10")
    lines.append("=" * (comp_w + len(loss_header)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_metrics_table(
    metrics_by_agent: dict[str, dict],
    title: str = "Evaluation Metrics",
) -> str:
    """Format a comparison table of metrics across multiple agents.

    Args:
        metrics_by_agent: ``{agent_name: metrics_dict}`` where each
                          metrics_dict is the output of :func:`compute_metrics`.
        title:            Table title printed above the header row.

    Returns:
        Multi-line string ready for printing.
    """
    keys = [
        ("fr_terminal",           "FR Terminal",    ".4f"),
        ("fr_mdd",                "FR MDD",         ".4f"),
        ("fr_vol_ann",            "FR Vol (ann)",   ".4f"),
        ("buffer_depletion_freq", "Buf Depl Freq",  ".4f"),
        ("total_distributions",   "Total Dist",     ".4f"),
        ("calmar_ratio",          "Calmar",         ".4f"),
        ("cohort_rr_var",         "Cohort RR Var",  ".6f"),
        ("rr_mean_young",         "RR Mean Young",  ".4f"),
        ("rr_mean_mid",           "RR Mean Mid",    ".4f"),
        ("rr_mean_ret",           "RR Mean Ret",    ".4f"),
        ("ppv_young_term",        "PPV Young (T)",  ".4f"),
        ("ppv_mid_term",          "PPV Mid (T)",    ".4f"),
        ("ppv_ret_term",          "PPV Ret (T)",    ".4f"),
    ]

    agents = list(metrics_by_agent.keys())
    col_w  = max(len(a) for a in agents) + 2

    lines = []
    lines.append(title)
    lines.append("=" * (22 + col_w * len(agents)))

    # Header
    header = f"  {'Metric':<20}" + "".join(f"{a:>{col_w}}" for a in agents)
    lines.append(header)
    lines.append("-" * len(header))

    for key, label, fmt in keys:
        row = f"  {label:<20}"
        for agent in agents:
            val = metrics_by_agent[agent].get(key, float("nan"))
            row += f"{val:{col_w}{fmt}}"
        lines.append(row)

    lines.append("=" * (22 + col_w * len(agents)))
    return "\n".join(lines)


def format_regime_table(
    regime_metrics_by_agent: dict[str, dict],
    metric_key: str = "fr_mdd",
    metric_label: str = "FR MDD",
) -> str:
    """Format a regime-conditional breakdown table for one metric.

    Args:
        regime_metrics_by_agent: ``{agent: {regime: metrics_dict}}``.
        metric_key:   Key from :func:`compute_metrics` to display.
        metric_label: Human-readable label for the metric.

    Returns:
        Multi-line string ready for printing.
    """
    agents  = list(regime_metrics_by_agent.keys())
    regimes = ["Low", "Medium", "High"]
    col_w   = max(len(a) for a in agents) + 2

    lines = [
        f"Regime-conditional: {metric_label}",
        "=" * (14 + col_w * len(agents)),
        f"  {'Regime':<12}" + "".join(f"{a:>{col_w}}" for a in agents),
        "-" * (14 + col_w * len(agents)),
    ]

    for regime in regimes:
        row = f"  {regime:<12}"
        for agent in agents:
            val = (
                regime_metrics_by_agent[agent]
                .get(regime, {})
                .get(metric_key, float("nan"))
            )
            n = regime_metrics_by_agent[agent].get(regime, {}).get("n_months", 0)
            row += f"  {val:>{col_w-5}.4f}(n={n:2d})"
        lines.append(row)

    lines.append("=" * (14 + col_w * len(agents)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main — self-test using Fixed-Rule ALM on test set
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__.replace("src/metrics.py", "").rstrip("/\\")))

    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data_pipeline import run_pipeline
    from src.environment   import make_env_from_pipeline
    from src.baselines     import FixedRuleALM, run_episode

    print("=" * 64)
    print("Wtp DRL -- Metrics self-test (Fixed-Rule ALM on test set)")
    print("=" * 64)

    results   = run_pipeline()
    env       = make_env_from_pipeline(results, split="test", seed=0)
    agent     = FixedRuleALM()
    traj      = run_episode(agent, env)

    # Align pi_monthly to trajectory dates
    cpi       = results["cpi"]
    dates_idx = pd.DatetimeIndex(traj["dates"])
    pi        = cpi["pi_monthly"].reindex(dates_idx).fillna(0.0).values

    metrics = compute_metrics(traj, pi_monthly=pi)

    print("\nCore metrics (Fixed-Rule ALM, test 2018-2025):")
    for k, v in metrics.items():
        print(f"  {k:<28}: {v:.6f}")

    # Regime analysis
    vstoxx = results["z_test_raw"]["vstoxx_level"]
    regime = regime_conditional_metrics(traj, vstoxx, pi_monthly=pi)

    print("\nRegime-conditional FR MDD:")
    for r_name, r_metrics in regime.items():
        mdd = r_metrics.get("fr_mdd", float("nan"))
        n   = r_metrics.get("n_months", 0)
        print(f"  {r_name:<8}: MDD={mdd:.4f}  n_months={n}")

    print("\nFormatted comparison table:")
    table = format_metrics_table({"FixedRule": metrics}, title="Test period metrics")
    print(table)

    print("\nDone.")
    sys.exit(0)
