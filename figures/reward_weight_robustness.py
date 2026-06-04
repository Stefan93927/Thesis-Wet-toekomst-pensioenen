"""
Reward-weight sensitivity figure and LaTeX table for the robustness section.
Reads src/models/run_042_robustness/robustness_results.json produced by
  python robustness.py --only-reward-weights
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "src/models/run_042_robustness/robustness_results.json"

with open(JSON) as f:
    raw = json.load(f)["reward_weights"]

base_adv = raw["base (no change)"]["advantage"]

# Parameters in display order with LaTeX labels and short descriptions
PARAMS = [
    ("alpha",          r"$\alpha$",      "Stability (FR target)"),
    ("beta",           r"$\beta$",       "Equity (distributions)"),
    ("gamma",          r"$\gamma$",      "Buffer depletion penalty"),
    ("fill_bonus",     "fill bonus",     "Fill incentive"),
    ("epsilon_equity", r"$\varepsilon$", "Cohort equity penalty"),
    ("dist_weight",    "dist weight",    "Distribution scale"),
]

low  = [raw[f"{k} -50%"]["advantage"] for k, *_ in PARAMS]
high = [raw[f"{k} +50%"]["advantage"] for k, *_ in PARAMS]
pct_lo = [(v - base_adv) / base_adv * 100 for v in low]
pct_hi = [(v - base_adv) / base_adv * 100 for v in high]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

fig, ax = plt.subplots(figsize=(7.5, 3.8))

n = len(PARAMS)
x = np.arange(n)
w = 0.32

col_lo = "#2166ac"   # blue  — weight halved
col_hi = "#d6604d"   # red   — weight 1.5×
col_base = "#555555"

bars_lo = ax.bar(x - w/2, low,  width=w, color=col_lo, alpha=0.85,
                 label=r"$-50\%$", zorder=3)
bars_hi = ax.bar(x + w/2, high, width=w, color=col_hi, alpha=0.85,
                 label=r"$+50\%$", zorder=3)

# Reference line at base
ax.axhline(base_adv, color=col_base, linewidth=1.1, linestyle="--",
           label=f"Base ({base_adv:.2f})", zorder=4)

# ±5% shaded band
ax.axhspan(base_adv * 0.95, base_adv * 1.05,
           color="grey", alpha=0.08, zorder=2, label=r"Base $\pm 5\%$")

# Annotate % deviation on bars (skip if <0.01%)
def _ann(rects, pcts):
    for rect, pct in zip(rects, pcts):
        if abs(pct) < 0.01:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() - 0.15,
                    "0%", ha="center", va="top", fontsize=7.5,
                    color="white", fontweight="bold")
        else:
            sign = "+" if pct > 0 else ""
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.08,
                    f"{sign}{pct:.1f}%", ha="center", va="bottom",
                    fontsize=7.5, color=col_base)

_ann(bars_lo, pct_lo)
_ann(bars_hi, pct_hi)

ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab, _ in PARAMS], fontsize=10)
ax.set_ylabel("DRL advantage  (reward$_{DRL}$ − reward$_{Fixed}$)", fontsize=9.5)
ax.set_ylim(16.5, 19.5)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))
ax.grid(axis="y", linewidth=0.4, color="grey", alpha=0.4, zorder=1)
ax.legend(framealpha=0.9, fontsize=9, loc="upper right",
          handlelength=1.5, handletextpad=0.5)
ax.set_title("Reward-weight sensitivity: DRL advantage over Fixed-Rule ALM",
             fontsize=10, pad=8)

fig.tight_layout()
out_pdf = Path(__file__).with_suffix(".pdf")
out_png = Path(__file__).with_suffix(".png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=180, bbox_inches="tight")
print(f"Saved {out_pdf.name}  and  {out_png.name}")

# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
LATEX = r"""\begin{table}[htbp]
  \centering
  \caption{Reward-weight sensitivity: DRL advantage over Fixed-Rule ALM
           under one-at-a-time $\pm 50\%$ perturbation of each reward
           coefficient.  Base advantage = \textbf{17.77}
           (DRL episode reward minus Fixed-Rule episode reward on the
           2018--2025 test period).}
  \label{tab:robustness_reward_weights}
  \begin{tabular}{llrrrr}
    \toprule
    Parameter & Role
      & $-50\%$ & Base & $+50\%$ & Max\,$|\Delta|$\,(\%) \\
    \midrule
"""
for (k, sym, desc), lo, hi in zip(PARAMS, low, high):
    max_dev = max(abs(lo - base_adv), abs(hi - base_adv)) / base_adv * 100
    LATEX += (
        f"    {sym:<22} & {desc:<30} & "
        f"{lo:6.2f} & {base_adv:6.2f} & {hi:6.2f} & "
        f"{max_dev:5.1f}\\% \\\\\n"
    )

LATEX += r"""    \bottomrule
  \end{tabular}
  \begin{tablenotes}\small
    \item \textit{Note:} $\alpha$ (stability), $\beta$ (equity),
      and dist\,weight are pure reward scalars; perturbing them shifts
      both agents' scores identically, leaving the advantage unchanged.
      $\gamma$ (buffer penalty), fill\,bonus, and $\varepsilon$ (cohort
      equity) alter the scored value of physical actions that differ
      between DRL and Fixed-Rule, producing the small but non-zero
      deviations shown above.  The DRL agent outperforms the Fixed-Rule
      baseline in all 13 tested configurations, with the advantage
      confined to a $\pm 1.8\%$ band around the base value.
  \end{tablenotes}
\end{table}
"""

print("\n" + "=" * 70)
print("LaTeX table")
print("=" * 70)
print(LATEX)

latex_path = Path(__file__).parent / "table_reward_weight_robustness.tex"
latex_path.write_text(LATEX)
print(f"Saved {latex_path.name}")
