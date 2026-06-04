# Managing the Solidarity Reserve
### A Regime-Aware Deep Reinforcement Learning Approach to the Wet toekomst pensioenen

MSc Finance Thesis — Stefan Bolt, University of Groningen, 2026

This repository contains the full code for a Deep Reinforcement Learning (DRL) agent that optimises equity allocation, solidarity reserve fill rates, and participant distributions for a Dutch pension fund operating under the Wet toekomst pensioenen (Wtp) solidarity premium contract (SPR).

The agent is trained with Proximal Policy Optimisation (PPO) and evaluated against three static baselines:

- **Fixed-Rule ALM** — static 55/45 equity/bond split with fixed fill and distribution rates
- **Monte Carlo ALM** — VAR(1)-optimised fill and distribution rates over 1,000 simulated scenarios
- **Hoevenaars-style ALM** — state-conditional, Bayesian-optimised multi-factor ALM (8-parameter)

All evaluation uses the out-of-sample test period January 2018–December 2025 on the actual historical path. No test-period information is used during training.

## Table of Contents
- Quick Start
- Installation
- Data Setup
- Reproducing the Main Results
- Expected Results
- Repository Structure
- Model Architecture
- Thesis Run Reference
- Citation

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data files in the project root (see Data Setup section)

# 3. Train the main model (~3–6 hours on a modern CPU, faster with GPU)
py -3 train.py --timesteps 2000000 --seed 42 --log-dir src/models/run_043 \
    --n-regimes 3 --lifecycle --bc-warmstart --bc-warmstart-steps 500000 \
    --bc-initial-weight 1.0 --bc-n-demos 10 --tc-bps 10.0 \
    --alpha 1.0 --beta 0.8 --delta 1000.0 --fill-bonus 3.0 \
    --gamma-depletion 100.0 --epsilon-equity 2.0 --zeta 1.0 \
    --norm-reward --vf-coef 0.5 --max-grad-norm 0.5 --weight-decay 0.0001 \
    --lr-warmup-steps 10000 --ent-coef 0.01 --lr 0.0003 \
    --eval-freq 50000 --checkpoint-freq 102400

# 4. Evaluate (skips Monte Carlo if --no-mc is passed)
py -3 evaluate.py --model-path src/models/run_043/best_model.zip

# 5. Robustness analysis
py -3 robustness.py --model-path src/models/run_043/best_model.zip
```

> On **Windows PowerShell**, the trailing `\` line-continuation does not work — put each command on a single line, or replace `\` with a backtick `` ` ``.

Pre-trained model artefacts for all thesis runs are included under `src/models/`. You can skip Step 3 and evaluate the saved model directly.

## Installation
### Prerequisites
| Requirement | Minimum version |
|---|---|
| Python | 3.10 |
| CUDA (opt.) | 11.8 |

### Steps
```bash
# Clone the repository
git clone https://github.com/Stefan93927/Thesis-Wet-toekomst-pensioenen.git
cd Thesis-Wet-toekomst-pensioenen

# Create a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

**Note on PyTorch:** the above installs the CPU-only build. For GPU training, install the correct CUDA build from pytorch.org before running `pip install -r requirements.txt`.

## Data Setup
The five market data files are not included in this repository (LSEG proprietary data). Place them in the project root before running any script:

| File | Description | Rows |
|---|---|---|
| `Financial_Data_30Y_English.xlsx` | 36 LSEG daily series (AEX, MSCI World, swaps, Euribor, VSTOXX, etc.). CSV-inside-single-column — the pipeline splits on commas automatically. | ~8,261 |
| `LSEG_Clean_Global_Indices.csv` | Global equity indices (SPX, NDX, DAX, CAC40, FTSE, Nikkei, HSI). | ~7,848 |
| `Agal_Indicators.csv` | VIX, VSTOXX, TED Spread, US yields. | ~7,796 |
| `CPI nederland.csv` | Dutch CPI from CBS Statline. Semicolon-separated. Date format YYYYMMnn. | Monthly |
| `Extra_Indicators.csv` | Clean Euribor 3M/1Y and VSTOXX from Jan 1999. Used instead of Excel Euribor (non-standard scale). | Monthly |

After placing the files, verify the data pipeline runs correctly:
```bash
py -3 -c "from src.data_pipeline import run_pipeline; r = run_pipeline(); print('OK:', r['z_train'].shape, r['z_test'].shape)"
# Expected: OK: (192, 31) (96, 31)
```
The pipeline fits and saves `src/scaler.joblib` (StandardScaler) and `src/gmm_k3.joblib` (K=3 GMM on VSTOXX) using training data only. These artefacts are already included in the repository so evaluation can run without the raw data.

## Reproducing the Main Results
### Run 043 — Main Result (DRL vs Fixed-Rule vs Monte Carlo vs Hoevenaars ALM)
This is the primary thesis result, Table 4.1.1.

**Training (~3–6h CPU):**
```bash
py -3 train.py \
    --timesteps 2000000 \
    --seed 42 \
    --log-dir src/models/run_043 \
    --n-regimes 3 \
    --lifecycle \
    --bc-warmstart \
    --bc-warmstart-steps 500000 \
    --bc-initial-weight 1.0 \
    --bc-n-demos 10 \
    --tc-bps 10.0 \
    --alpha 1.0 \
    --beta 0.8 \
    --delta 1000.0 \
    --fill-bonus 3.0 \
    --gamma-depletion 100.0 \
    --epsilon-equity 2.0 \
    --zeta 1.0 \
    --norm-reward \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --weight-decay 0.0001 \
    --lr-warmup-steps 10000 \
    --ent-coef 0.01 \
    --lr 0.0003 \
    --eval-freq 50000 \
    --checkpoint-freq 102400
```

**Evaluation (uses saved `best_model.zip`):**
```bash
# Calibrate Hoevenaars ALM (Bayesian optimisation, ~30–60 min)
py -3 run_hoevenaars.py

# Evaluate with all three baselines
py -3 evaluate.py --model-path src/models/run_043/best_model.zip
```
Results are saved to `src/models/run_043/eval_results.json`.

### Run 040 — Ablation: Scalar Reward Micro-Distribution Exploit
Demonstrates that a distribution-incentivised reward without the lexicographic ordering causes the agent to exploit micro-distributions. Uses `fill_bonus=1.5` instead of 3.0:
```bash
py -3 train.py \
    --timesteps 2000000 \
    --seed 42 \
    --log-dir src/models/run_040 \
    --n-regimes 3 \
    --lifecycle \
    --bc-warmstart \
    --bc-warmstart-steps 500000 \
    --bc-initial-weight 1.0 \
    --bc-n-demos 10 \
    --alpha 1.0 \
    --beta 0.8 \
    --delta 1000.0 \
    --fill-bonus 1.5 \
    --gamma-depletion 100.0 \
    --epsilon-equity 2.0 \
    --zeta 1.0 \
    --norm-reward \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --weight-decay 0.0001 \
    --lr-warmup-steps 10000 \
    --ent-coef 0.01 \
    --lr 0.0003 \
    --eval-freq 50000 \
    --checkpoint-freq 102400
```

### Run 007 — Pre-Lifecycle Proof of Concept (legacy)
Simpler single-step environment without the PPV lifecycle framework. 1M steps, no BC warmstart:
```bash
py -3 train.py \
    --timesteps 1000000 \
    --seed 42 \
    --log-dir src/models/run_007 \
    --no-lifecycle \
    --ent-coef 0.05
```

### Robustness Analysis
```bash
# Main robustness suite (initial conditions, transaction costs, liability blend, DNB stress)
py -3 robustness.py --model-path src/models/run_043/best_model.zip

# Extended robustness with Hoevenaars baseline + multi-path VAR simulation
py -3 evaluate_robustness.py --model-path src/models/run_043/best_model.zip
```

## Expected Results
Results from `src/models/run_043/eval_results.json` — test period Jan 2018–Dec 2025:

| Metric | DRL (PPO) | Fixed-Rule ALM | Hoevenaars ALM |
|---|---|---|---|
| FR Terminal | 1.867 | 1.651 | 1.488 |
| FR Max Drawdown | 15.28% | 14.71% | 14.85% |
| FR Annualised Vol | 10.36% | 8.81% | 8.15% |
| Buffer Depletion Freq | 3.53% | 94.12% | 3.53% |
| Total Distributions | 13.85% | 16.30% | 16.47% |
| Calmar Ratio | 0.600 | 0.507 | 0.399 |
| Cohort RR Variance | 0.00171 | 0.00186 | 0.00189 |
| PPV Young (terminal) | 2.171 | 1.992 | 1.818 |
| PPV Mid-career (terminal) | 1.911 | 1.734 | 1.580 |
| PPV Retired (terminal) | 1.294 | 1.173 | 1.064 |

**Note on reproducibility:** due to PyTorch non-determinism (especially on GPU), the exact numbers may vary by ±0.5% even with `--seed 42`. The pre-trained `best_model.zip` in this repository reproduces the thesis numbers exactly.

## Repository Structure
```
data/                               ← project root
├── README.md
├── requirements.txt
│
├── train.py                        ← main training script
├── evaluate.py                     ← out-of-sample evaluation vs baselines
├── robustness.py                   ← robustness checks (IC, TC, liability blend, DNB stress)
├── evaluate_robustness.py          ← extended robustness with multi-path VAR simulation
├── run_hoevenaars.py               ← Hoevenaars ALM calibration + evaluation driver
│
├── src/
│   ├── data_pipeline.py            ← data loading, feature engineering, train/val/test split
│   ├── environment.py              ← Gymnasium env with all Art. 10d Wtp rules
│   ├── agent.py                    ← LSTM + GMM gating + PPO policy (WtpActorCriticPolicy)
│   ├── baselines.py                ← FixedRuleALM, MonteCarloALM, run_episode()
│   ├── hoevenaars_alm.py           ← Hoevenaars-style multi-factor ALM baseline
│   ├── metrics.py                  ← evaluation metrics, bootstrap CI, Diebold-Mariano test
│   ├── scaler.joblib               ← pre-fitted StandardScaler (train period only)
│   └── models/
│       ├── run_043/                ← main result: PPO vs Fixed-Rule vs Monte Carlo vs Hoevenaars
│       │   ├── best_model.zip      ← best checkpoint (by val composite score)
│       │   ├── final_model.zip     ← model at end of training
│       │   ├── train_config.json   ← exact hyperparameters used
│       │   ├── eval_results.json   ← test-period metrics + bootstrap CI + regime breakdown
│       │   ├── val_history.json    ← validation curve (model selection only)
│       │   └── trajectory_*.npz    ← saved trajectory arrays for figure generation
│       ├── run_040/                ← ablation: scalar reward → micro-distribution exploit
│       └── run_007/                ← pre-lifecycle PoC (legacy, no PPV framework)
```

## Model Architecture
The DRL agent has four end-to-end trained components:
```
z_{t-11:t}  ──► LSTM(31 → 256)  ──► h_t
                                     │
                              GMM gating (K=3)
                              VSTOXX regimes:
                              Low (<20):  β̄=0.65
                              Med (20-30): β̄=0.55   ──► β_t (risk budget)
                              High (≥30): β̄=0.35
                                     │
                       Deterministic risk-budget allocation
                       (constraint-enforcing action masking;
                        every action clipped to Art. 10d bounds)
                                     │
                              PPO Head  ──► [e_t, f_t, d_t]
```
**State space:** `[FR_t, B_t, z_{t-11}, ..., z_t]` — funding ratio, buffer, and 12-month lookback of 31 market features.

**Action space (3-dim continuous):**
- `e_t ∈ [−0.25, +0.25]` — equity tilt → `w_eq = clip(0.55 + e_t, 0.30, 0.80)`
- `f_t ∈ [0, 0.10]` — fill rate (fund → buffer transfer)
- `d_t ∈ [0, 0.05]` — distribution rate (buffer → participants transfer)

**Reward (lexicographic priority):**
1. MVEV floor penalty (δ=1000 if FR < 1.043)
2. Buffer depletion penalty (γ=100 when B_t near zero)
3. Safe-zone composite: FR stability (α=1.0) + fill bonus (ϕ=3.0) + distribution incentive (β=0.8) − cohort equity variance (ε=2.0)

**Training:** 500k steps behavioural cloning warmstart (BC weight decays 1.0→0.0) + 1.5M steps PPO. VecNormalize on rewards only.

**Key hyperparameters:** lr=3×10⁻⁴, n_steps=2048, batch=64, γ=0.99, clip=0.2, GAE λ=0.95, ent_coef=0.01, tc=10bps.

## Wtp Regulatory Context
The environment enforces three hard constraints from Art. 10d Pensioenwet (solidarity reserve):

| Constraint | Legal basis | Implementation |
|---|---|---|
| Annual fill cap: max 10% of cumulative positive overrendement per calendar year | Art. 10d lid 2 | Reset O⁺ and cumulative fills on 1 January |
| Distribution gate: only when FR_t ≥ 1.00 and B_t > 0 | Art. 10d lid 4 | d̃_t = min(d_t, B_t) if FR≥1, else 0 |
| Buffer bounds: B_t ∈ [0, 0.15]; excess on 31 Dec distributed to participants | Art. 10d lid 1 | Hard clip + year-end sweep |

These are implemented as explicit, testable functions in `src/environment.py` — not as soft reward penalties.

## Thesis Run Reference
| Run | Purpose | Key configuration |
|---|---|---|
| `run_043` | Institutional benchmark — three ALM baselines (Fixed-Rule, Monte Carlo, Hoevenaars) | Full lexicographic reward + PPV lifecycle; 2M steps; BC warmstart |
| `run_040` | Ablation — scalar reward exploit | `fill_bonus=1.5` (lower threshold → micro-distribution exploit) |
| `run_007` | Pre-lifecycle proof of concept | `timesteps=1M`, no BC warmstart, no lifecycle PPV, `ent_coef=0.05` |

## Citation
```bibtex
@mastersthesis{bolt2026wtp,
  author    = {Stefan Bolt},
  title     = {Managing the Solidarity Reserve: A Regime-Aware Deep Reinforcement
               Learning Approach to the Wet toekomst pensioenen},
  school    = {University of Groningen},
  year      = {2026},
  type      = {MSc Finance Thesis}
}
```

## License
This repository is shared for academic reproducibility. The market data files are proprietary (LSEG) and are not included. All code is © Stefan Bolt 2026.
