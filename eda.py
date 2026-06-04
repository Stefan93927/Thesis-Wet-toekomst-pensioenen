"""eda.py — Exploratory Data Analysis for the Wtp DRL pension fund project.

Produces figures and summary tables for the thesis data chapter.

Figures saved to eda_figures/:
  eda1_summary_stats.png      — Feature summary statistics (train / test)
  eda2_feature_distributions.png  — Boxplots of all 31 features by split
  eda3_correlation.png        — Pearson correlation heatmap (training period)
  eda4_raw_timeseries.png     — Key raw market variables (2000-2025)
  eda5_regime_breakdown.png   — VSTOXX regime composition + distributions
  eda6_dist_shift.png         — Train vs Test distribution shift (violin)

Usage:
    py -3 eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.data_pipeline import run_pipeline

OUT_DIR = Path("eda_figures")
OUT_DIR.mkdir(exist_ok=True)

# ── Colour scheme ────────────────────────────────────────────────────────────
SPLIT_COLORS = {"Train": "#2563EB", "Val": "#F59E0B", "Test": "#DC2626"}

FEATURE_GROUPS = {
    "Equity Momentum": [
        "mom_msci_1m","mom_msci_3m","mom_msci_12m",
        "mom_aex_1m", "mom_aex_3m", "mom_aex_12m",
        "mom_stoxx_1m","mom_stoxx_3m","mom_stoxx_12m",
        "mom_em_1m",  "mom_em_3m",  "mom_em_12m",
    ],
    "Volatility & Rates": [
        "vstoxx_level","d_vstoxx_1m","d_vstoxx_3m",
        "euribor_3m","euribor_1y_proxy","yield_us_10y","yield_de_10y",
    ],
    "Swap Curve": [
        "slope_10y_2y","slope_30y_10y",
        "swap_2y","swap_5y","swap_10y","swap_20y","swap_30y",
        "d_swap_10y","d_swap_20y",
    ],
    "Other": ["liab_proxy","gold_log_ret","oil_log_ret"],
}

FEATURE_LABELS = {
    "mom_msci_1m":     "MSCI 1M", "mom_msci_3m":  "MSCI 3M", "mom_msci_12m": "MSCI 12M",
    "mom_aex_1m":      "AEX 1M",  "mom_aex_3m":   "AEX 3M",  "mom_aex_12m":  "AEX 12M",
    "mom_stoxx_1m":    "SX50 1M", "mom_stoxx_3m":  "SX50 3M", "mom_stoxx_12m":"SX50 12M",
    "mom_em_1m":       "EM 1M",   "mom_em_3m":     "EM 3M",   "mom_em_12m":   "EM 12M",
    "vstoxx_level":    "VSTOXX",  "d_vstoxx_1m":   "dVST 1M", "d_vstoxx_3m":  "dVST 3M",
    "euribor_3m":      "EUR3M",   "euribor_1y_proxy":"EUR1Y", "yield_us_10y":  "USY10",
    "yield_de_10y":    "DEY10",   "liab_proxy":    "Liab",
    "slope_10y_2y":    "Sl10-2",  "slope_30y_10y": "Sl30-10",
    "swap_2y":         "SW2Y",    "swap_5y":       "SW5Y",    "swap_10y":      "SW10Y",
    "swap_20y":        "SW20Y",   "swap_30y":      "SW30Y",
    "d_swap_10y":      "dSW10",   "d_swap_20y":    "dSW20",
    "gold_log_ret":    "Gold",    "oil_log_ret":   "Oil",
}


def _save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ────────────────────────────────────────────────────────────────────────────
# EDA 1 — Summary statistics table
# ────────────────────────────────────────────────────────────────────────────

def eda1_summary_stats(splits: dict) -> None:
    """Print and plot a summary-statistics table for train and test splits."""
    train = splits["Train"]
    test  = splits["Test"]
    cols  = list(FEATURE_LABELS.keys())
    labels= [FEATURE_LABELS[c] for c in cols]

    rows = []
    for c, lab in zip(cols, labels):
        tr = train[c]
        te = test[c]
        rows.append({
            "Feature": lab,
            "Train Mean": f"{tr.mean():.4f}",
            "Train Std":  f"{tr.std():.4f}",
            "Train Min":  f"{tr.min():.4f}",
            "Train Max":  f"{tr.max():.4f}",
            "Test Mean":  f"{te.mean():.4f}",
            "Test Std":   f"{te.std():.4f}",
        })
    df = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print("FEATURE SUMMARY STATISTICS (unscaled, training vs test)")
    print("=" * 90)
    print(df.to_string(index=False))

    df.to_csv(OUT_DIR / "summary_stats.csv", index=False)
    print(f"\n  Stats saved: {OUT_DIR / 'summary_stats.csv'}")

    # Visual table
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis("off")
    col_labels = list(df.columns)
    tbl = ax.table(
        cellText=df.values,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.3)
    # Header colour
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1E3A5F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Alternate row colours
    for i in range(1, len(df) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor("#F0F4FF" if i % 2 == 0 else "white")

    ax.set_title("Feature Summary Statistics — Training vs Test Period",
                 fontsize=12, fontweight="bold", pad=10)
    _save(fig, "eda1_summary_stats.png")


# ────────────────────────────────────────────────────────────────────────────
# EDA 2 — Feature distributions (boxplots by group)
# ────────────────────────────────────────────────────────────────────────────

def eda2_feature_distributions(splits: dict) -> None:
    n_groups = len(FEATURE_GROUPS)
    fig, axes = plt.subplots(n_groups, 1, figsize=(16, 4 * n_groups))
    fig.suptitle("Feature Distributions by Group — Train vs Test (unscaled)",
                 fontsize=13, fontweight="bold")

    for ax, (group, feats) in zip(axes, FEATURE_GROUPS.items()):
        labels = [FEATURE_LABELS[f] for f in feats]
        x      = np.arange(len(feats))
        width  = 0.3

        for i, (split_name, color) in enumerate(
            [("Train", SPLIT_COLORS["Train"]), ("Test", SPLIT_COLORS["Test"])]
        ):
            df = splits[split_name]
            bp = ax.boxplot(
                [df[f].dropna().values for f in feats],
                positions=x + (i - 0.5) * width,
                widths=width * 0.85,
                patch_artist=True,
                medianprops=dict(color="white", lw=1.5),
                whiskerprops=dict(color=color, lw=1.0),
                capprops=dict(color=color, lw=1.0),
                flierprops=dict(marker=".", ms=2, alpha=0.3, color=color),
                boxprops=dict(facecolor=color, alpha=0.6, linewidth=0),
            )
            # Add invisible bar for legend
            ax.bar([-999], [0], color=color, alpha=0.6, label=split_name)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_title(group, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "eda2_feature_distributions.png")


# ────────────────────────────────────────────────────────────────────────────
# EDA 3 — Correlation heatmap (training period)
# ────────────────────────────────────────────────────────────────────────────

def eda3_correlation(splits: dict) -> None:
    train  = splits["Train"]
    cols   = list(FEATURE_LABELS.keys())
    labels = [FEATURE_LABELS[c] for c in cols]
    corr   = train[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.suptitle("Feature Correlation Matrix — Training Period (Jan 2000 – Dec 2015)",
                 fontsize=13, fontweight="bold")

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7.5)

    # Annotate cells with |r| > 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            if abs(v) > 0.5 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if abs(v) > 0.7 else "black")

    # Group separators
    groups_flat = list(FEATURE_GROUPS.values())
    pos = 0
    for g in groups_flat[:-1]:
        pos += len(g)
        ax.axhline(pos - 0.5, color="black", lw=1.2)
        ax.axvline(pos - 0.5, color="black", lw=1.2)

    fig.tight_layout()
    _save(fig, "eda3_correlation.png")


# ────────────────────────────────────────────────────────────────────────────
# EDA 4 — Raw time series (key variables 2000-2025)
# ────────────────────────────────────────────────────────────────────────────

def eda4_raw_timeseries(results: dict) -> None:
    all_raw = pd.concat([results["raw_train"], results["raw_val"], results["raw_test"]])
    z_all   = pd.concat([results["z_train_raw"], results["z_val_raw"], results["z_test_raw"]])

    val_start  = results["z_val_raw"].index[0]
    test_start = results["z_test_raw"].index[0]
    end        = z_all.index[-1]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Key Market Variables — Jan 2000 to Dec 2025",
                 fontsize=13, fontweight="bold")
    gs  = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    def shade(ax):
        ax.axvspan(val_start,  test_start, color="lightyellow", alpha=0.6, zorder=0)
        ax.axvspan(test_start, end,        color="lightcyan",   alpha=0.6, zorder=0)
        ax.axvline(val_start,  color="orange",    lw=0.9, ls="--")
        ax.axvline(test_start, color="steelblue", lw=0.9, ls="--")

    # ── Panel: MSCI World price (rebased) ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    msci = all_raw["Equity_World_MSCI"].dropna()
    rebased = msci / msci.iloc[0] * 100
    ax.plot(rebased.index, rebased, color="#2563EB", lw=1.2)
    shade(ax)
    ax.set_title("MSCI World (rebased 100)", fontsize=9)
    ax.set_ylabel("Index")

    # ── Panel: AEX ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    aex = all_raw["Equity_NL_AEX"].dropna()
    ax.plot(aex.index, aex, color="#7C3AED", lw=1.2)
    shade(ax)
    ax.set_title("AEX Index", fontsize=9)
    ax.set_ylabel("Points")

    # ── Panel: VSTOXX ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(z_all.index, z_all["vstoxx_level"],
                    color="#FCA5A5", alpha=0.6)
    ax.plot(z_all.index, z_all["vstoxx_level"], color="#DC2626", lw=0.8)
    ax.axhline(20, color="grey", ls="--", lw=0.8)
    ax.axhline(30, color="grey", ls=":",  lw=0.8)
    shade(ax)
    ax.set_title("VSTOXX Volatility Index", fontsize=9)
    ax.set_ylabel("VSTOXX")

    # ── Panel: Euribor 3M ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(z_all.index, z_all["euribor_3m"] * 100, color="#16A34A", lw=1.2)
    ax.axhline(0, color="black", lw=0.6)
    shade(ax)
    ax.set_title("Euribor 3M (%)", fontsize=9)
    ax.set_ylabel("Rate (%)")

    # ── Panel: Swap 10Y ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(z_all.index, z_all["swap_10y"] * 100, color="#0891B2", lw=1.2, label="10Y")
    ax.plot(z_all.index, z_all["swap_30y"] * 100, color="#0E7490", lw=1.0, ls="--", label="30Y")
    ax.axhline(0, color="black", lw=0.6)
    shade(ax)
    ax.set_title("EUR Swap Rates (%)", fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.legend(fontsize=7)

    # ── Panel: Swap slope ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(z_all.index, z_all["slope_10y_2y"]  * 100, color="#D97706", lw=1.2, label="10Y-2Y")
    ax.plot(z_all.index, z_all["slope_30y_10y"] * 100, color="#B45309", lw=1.0, ls="--", label="30Y-10Y")
    ax.axhline(0, color="black", lw=0.6)
    shade(ax)
    ax.set_title("Swap Curve Slope (%)", fontsize=9)
    ax.set_ylabel("Spread (%)")
    ax.legend(fontsize=7)

    # ── Panel: Gold & Oil monthly log returns ───────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    ax.bar(z_all.index, z_all["gold_log_ret"] * 100, width=20,
           color=np.where(z_all["gold_log_ret"] >= 0, "gold", "grey"), alpha=0.7)
    shade(ax)
    ax.set_title("Gold Monthly Log Return (%)", fontsize=9)
    ax.set_ylabel("Return (%)")

    ax = fig.add_subplot(gs[3, 1])
    ax.bar(z_all.index, z_all["oil_log_ret"] * 100, width=20,
           color=np.where(z_all["oil_log_ret"] >= 0, "#78716C", "#DC2626"), alpha=0.7)
    shade(ax)
    ax.set_title("Oil Monthly Log Return (%)", fontsize=9)
    ax.set_ylabel("Return (%)")

    # Legend for period shading
    from matplotlib.patches import Patch
    legend_els = [
        Patch(color="white",       label="Training (2000-2015)"),
        Patch(color="lightyellow", label="Validation (2016-2017)"),
        Patch(color="lightcyan",   label="Test (2018-2025)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

    _save(fig, "eda4_raw_timeseries.png")


# ────────────────────────────────────────────────────────────────────────────
# EDA 5 — VSTOXX regime breakdown
# ────────────────────────────────────────────────────────────────────────────

def eda5_regime_breakdown(splits: dict) -> None:
    thresholds = (20.0, 30.0)
    regime_labels = ["Low (<20)", "Medium (20-30)", "High (>30)"]
    regime_colors = ["#86EFAC", "#FCD34D", "#FCA5A5"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("VSTOXX Regime Breakdown by Split", fontsize=13, fontweight="bold")

    for ax, (split, color_key) in zip(axes, [("Train", "#2563EB"), ("Val", "#F59E0B"), ("Test", "#DC2626")]):
        vs = splits[split]["vstoxx_level"].dropna()
        regimes = np.digitize(vs.values, thresholds)
        counts  = [np.sum(regimes == i) for i in range(3)]
        total   = sum(counts)

        wedges, texts, autotexts = ax.pie(
            counts,
            labels=[f"{lab}\n(n={c})" for lab, c in zip(regime_labels, counts)],
            colors=regime_colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title(f"{split} Period\n(n={total} months)", fontweight="bold")

    fig.tight_layout()
    _save(fig, "eda5_regime_breakdown.png")


# ────────────────────────────────────────────────────────────────────────────
# EDA 6 — Distribution shift: Train vs Test (violin plots)
# ────────────────────────────────────────────────────────────────────────────

def eda6_dist_shift(splits: dict) -> None:
    # Pick 12 most interesting features
    focus_feats = [
        "mom_msci_1m", "mom_msci_12m", "vstoxx_level", "d_vstoxx_1m",
        "euribor_3m", "yield_de_10y", "slope_10y_2y", "swap_10y",
        "d_swap_10y", "liab_proxy", "gold_log_ret", "oil_log_ret",
    ]
    focus_labels = [FEATURE_LABELS[f] for f in focus_feats]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle("Train vs Test Distribution Shift (Key Features)",
                 fontsize=13, fontweight="bold")

    for ax, (feat, lab) in zip(axes.flat, zip(focus_feats, focus_labels)):
        tr_vals = splits["Train"][feat].dropna().values
        te_vals = splits["Test"][feat].dropna().values

        vp = ax.violinplot([tr_vals, te_vals], positions=[1, 2],
                           showmedians=True, showextrema=False)

        vp["bodies"][0].set_facecolor(SPLIT_COLORS["Train"])
        vp["bodies"][0].set_alpha(0.6)
        vp["bodies"][1].set_facecolor(SPLIT_COLORS["Test"])
        vp["bodies"][1].set_alpha(0.6)
        vp["cmedians"].set_color("white")
        vp["cmedians"].set_linewidth(2)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Train", "Test"], fontsize=8)
        ax.set_title(lab, fontweight="bold", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Annotate means
        ax.text(1, ax.get_ylim()[1] * 0.98,
                f"μ={tr_vals.mean():.3f}", ha="center", fontsize=6.5,
                color=SPLIT_COLORS["Train"])
        ax.text(2, ax.get_ylim()[1] * 0.98,
                f"μ={te_vals.mean():.3f}", ha="center", fontsize=6.5,
                color=SPLIT_COLORS["Test"])

    fig.tight_layout()
    _save(fig, "eda6_dist_shift.png")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Wtp DRL Pension Fund -- Exploratory Data Analysis")
    print("=" * 60)

    print("\nLoading data pipeline...")
    results = run_pipeline()

    splits = {
        "Train": results["z_train_raw"],
        "Val":   results["z_val_raw"],
        "Test":  results["z_test_raw"],
    }

    print(f"\nData shapes:")
    for name, df in splits.items():
        print(f"  {name:6s}: {df.shape[0]} months x {df.shape[1]} features "
              f"({df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')})")

    print(f"\nGenerating EDA figures -> {OUT_DIR}/")
    eda1_summary_stats(splits)
    eda2_feature_distributions(splits)
    eda3_correlation(splits)
    eda4_raw_timeseries(results)
    eda5_regime_breakdown(splits)
    eda6_dist_shift(splits)

    print(f"\nAll EDA figures saved to: {OUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
