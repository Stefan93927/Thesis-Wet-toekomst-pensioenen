"""environment.py — Custom Gymnasium environment for the Wtp SPR pension fund.

The environment simulates a solidarity premium pension fund (solidariteitsreserve,
SPR) under Wet toekomst pensioenen (Wtp), Art. 10d.  It wraps processed monthly
market data from data_pipeline.py and exposes a standard Gymnasium interface.

State space  (374-dim) : [FR_t, B_t,  z_{t-11}, …, z_t]
  FR_t  ∈ [0, 5]        : funding ratio (assets / liabilities)
  B_t   ∈ [0, 0.15]     : solidarity buffer (fraction of liabilities)
  z_*   : 12 × 31 = 372  scaled monthly feature vectors (LSTM lookback)

Action space (3-dim continuous) :
  e_t ∈ [-0.25, +0.25]  : equity tilt  →  w_eq = clip(0.55 + e_t, 0.30, 0.80)
  f_t ∈ [0, 0.10]       : fill rate (transfer from fund to buffer)
  d_t ∈ [0, 0.05]       : distribution rate (transfer from buffer to participants)

Transition dynamics follow the liability-blended scheme described in CLAUDE.md.
All three Art. 10d hard constraints are enforced as explicit, testable functions.

Usage
-----
    from src.data_pipeline import run_pipeline
    from src.environment   import make_env_from_pipeline

    results = run_pipeline()
    env     = make_env_from_pipeline(results, split="train", seed=42)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required.  Install with:  pip install gymnasium"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """All tunable constants for the Wtp pension environment.

    Keep every magic number here, never hard-code inside logic.
    """

    # --- Liability modelling ----------------------------------------------- #
    duration:              float = 18.0   # years (mid-point of 17–20 range)
    r_ufr:                 float = 0.002711 # per month (3.30 % annualised, DNB/EIOPA UFR from Jan 2025)
    liability_mtm_weight:  float = 0.70   # weight on MtM, rest on UFR

    # --- Action space bounds ---------------------------------------------- #
    w_eq_base:      float = 0.55   # strategic equity weight (aggregate / legacy)
    w_eq_min:       float = 0.30
    w_eq_max:       float = 0.80
    eq_tilt_min:    float = -0.25
    eq_tilt_max:    float = +0.25
    fill_rate_max:  float = 0.10
    dist_rate_max:  float = 0.05

    # --- Lifecycle / PPV framework 
    use_lifecycle:   bool  = True   
    w_eq_young_base: float = 0.85   # young:      long horizon, high equity
    w_eq_mid_base:   float = 0.70   # mid-career: transitioning
    w_eq_ret_base:   float = 0.30   # retired:    capital preservation
    w_eq_young_min:  float = 0.60;  w_eq_young_max: float = 1.00
    w_eq_mid_min:    float = 0.45;  w_eq_mid_max:   float = 0.95
    w_eq_ret_min:    float = 0.05;  w_eq_ret_max:   float = 0.55
    # Cohort liability / participation shares
    w_young: float = 0.20
    w_mid:   float = 0.35
    w_ret:   float = 0.45

    # --- Art. 10d buffer constraints -------------------------------------- #
    b_max:                 float = 0.15   # lid 1: maximum buffer
    annual_fill_cap_frac:  float = 0.10   # lid 2: max fill = 10 % × annual O+
    fr_dist_threshold:     float = 1.00   # lid 4: FR must be >= this to distribute

    # --- Return clipping -------------------------------------------------- #
    r_eq_clip:   Tuple[float, float] = (-0.30, +0.30)
    r_bond_clip: Tuple[float, float] = (-0.05, +0.05)

    # --- Reward structure (lexicographic hierarchy) -------------- #
    # Implements the institutional mandate priority ordering as a conditional
    # structure rather than competing weighted terms:
    #   Priority 1: MVEV compliance   (Besluit FTK Art. 11a) — non-negotiable
    #   Priority 2: Buffer solvency   (Art. 10d lid 1)       — must maintain capacity
    #   Priority 3: Optimisation      (solvency + dist + equity) — normal ops only

    # Priority 1 weight (MVEV quadratic penalty): large to dominate all else
    delta:       float = 1000

    # Priority 2 weight (buffer depletion quadratic penalty)
    gamma:       float = 100

    # Priority 3 weights (safe-zone optimisation)
    alpha:       float = 1.0     # stability: one-sided penalty below FR_target
    beta:        float = 0.8     # distribution incentive Q_t
    fill_bonus:  float = 3.0     # direct fill reward: R_fill = fill_bonus * f_tilde
    epsilon_equity: float = 2.0  # cross-cohort RR variance penalty

    # Distribution incentive scaling
    dist_weight:             float = 5.0   # multiplicative scale on Q_t
    dist_reward_buffer_gate: float = 0.05  # B threshold: full Q_t above, partial below

    # --- Transaction costs ------------------------------------------------ #
    tc_bps:      float = 0.1   # one-way equity turnover cost in basis points
                               # (deducted from r_p_t each step as a drag)

    # Legacy fields kept for backward compatibility with older configs/tests
    zeta:        float = 0.0   # FR change penalty (unused in run_037+)
    log_dist:    bool  = True  # log1p(d/scale) utility in Q_t (unused in run_037+)
    log_dist_scale: float = 0.005
    lambda_smooth:  float = 0.0

    # --- FR thresholds ---------------------------------------------------- #
    fr_target:      float = 1.05    # stability reward centred here
    fr_mvev:        float = 1.043   # MVEV floor — insolvency if FR < this
    fr_catastrophe: float = 0.50    # terminate episode early if FR falls here

    # --- Observation / episode ------------------------------------------- #
    lookback:        int = 12   # months of feature history in state
    n_features:      int = 31   # feature dimension per month
    n_extra_solvency: int = 3   

    # --- Initial conditions (can be overridden via reset(options=...) ) --- #
    fr_init: float = 1.05
    b_init:  float = 0.05


# ---------------------------------------------------------------------------
# Art. 10d constraint functions  (pure, side-effect-free → easily testable)
# ---------------------------------------------------------------------------

def apply_annual_fill_cap(
    f_t: float,
    annual_o_plus: float,
    annual_fill_used: float,
    cfg: EnvConfig,
) -> float:
    """Enforce Art. 10d lid 2: annual fill cap.

    The total fills in a calendar year may not exceed
    ``cfg.annual_fill_cap_frac`` (10 %) of the cumulative positive
    overrendement in that year.

    Args:
        f_t:              Requested fill rate from the agent.
        annual_o_plus:    Cumulative positive overrendement so far this year.
        annual_fill_used: Fills already executed this calendar year.
        cfg:              EnvConfig.

    Returns:
        Effective fill rate f̃_t ∈ [0, f_t].
    """
    budget    = cfg.annual_fill_cap_frac * annual_o_plus
    remaining = max(0.0, budget - annual_fill_used)
    return min(f_t, remaining)


def apply_distribution_rule(
    d_t: float,
    b_t: float,
    fr_t: float,
    cfg: EnvConfig,
) -> float:
    """Enforce Art. 10d lid 4: distribution eligibility rule.

    Distributions are only allowed when FR_t >= threshold and the buffer
    holds sufficient assets.  The amount is further capped at the available
    buffer.

    Args:
        d_t:  Requested distribution rate from the agent.
        b_t:  Current buffer level (fraction of liabilities).
        fr_t: Current funding ratio.
        cfg:  EnvConfig.

    Returns:
        Effective distribution rate d̃_t ∈ [0, d_t].
    """
    if fr_t < cfg.fr_dist_threshold or b_t <= 0.0:
        return 0.0
    return min(d_t, b_t)


def apply_buffer_bounds(
    fr: float,
    b: float,
    is_december: bool,
    cfg: EnvConfig,
) -> Tuple[float, float, float]:
    """Enforce Art. 10d lid 1: buffer bounds B ∈ [0, b_max].

    Rules applied in order:
    1. If B < 0: shortfall absorbed by FR; B floored at 0.
    2. On 31 Dec: if B > b_max, cap at b_max and record excess for
       distribution to participants.

    Args:
        fr:          Funding ratio before bound enforcement.
        b:           Buffer level before bound enforcement.
        is_december: True when the current step is in December.
        cfg:         EnvConfig.

    Returns:
        ``(fr, b, dec_excess)`` where ``dec_excess`` is the December cap
        surplus (distributed to participants; does not affect FR).
    """
    dec_excess = 0.0

    # Lid 1a: negative buffer shortfall absorbed by FR
    if b < 0.0:
        fr = fr + b   # fr -= |b|
        b  = 0.0

    # Lid 1b: December cap — distribute excess above b_max
    if is_december and b > cfg.b_max:
        dec_excess = b - cfg.b_max
        b = cfg.b_max

    return fr, b, dec_excess


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class WtpPensionEnv(gym.Env):
    """Gymnasium environment for Wtp solidarity premium pension optimisation.

    The environment steps through monthly market data.  At each step the agent
    chooses an equity tilt, a fill rate, and a distribution rate.  The
    environment applies all three Art. 10d hard constraints, updates FR and B,
    tracks three participant cohorts for intergenerational equity, and returns
    a composite reward.

    Args:
        z_scaled:    ``(T, 31)`` array of *scaled* monthly feature vectors
                     (output of data_pipeline StandardScaler).
        r_eq:        ``(T,)`` MSCI World simple monthly return, already clipped
                     to ``cfg.r_eq_clip``.
        r_bond:      ``(T,)`` Bond return proxy, already clipped.
        r_L_MtM:     ``(T,)`` Mark-to-market liability return
                     (``-duration * DeltaSwap20Y / 100``).
        pi_monthly:  ``(T,)`` Dutch monthly CPI inflation.
        dates:       ``(T,)`` Month-end dates aligned with the above arrays.
        cfg:         Optional :class:`EnvConfig`; defaults to ``EnvConfig()``.
        seed:        Optional integer seed for reproducibility.
        r_L_blended: Optional ``(T,)`` pre-computed blended liability return
                     from DNB RTS (70% MtM + 30% UFR).  When provided, this
                     replaces the internally computed blend and makes the
                     simulation consistent with official DNB methodology.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        z_scaled:    np.ndarray,
        r_eq:        np.ndarray,
        r_bond:      np.ndarray,
        r_L_MtM:     np.ndarray,
        pi_monthly:  np.ndarray,
        dates,
        cfg:         Optional[EnvConfig] = None,
        seed:        Optional[int] = None,
        r_L_blended: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        self.cfg  = cfg or EnvConfig()
        self.z    = np.asarray(z_scaled,   dtype=np.float32)
        self.r_eq    = np.asarray(r_eq,    dtype=np.float64)
        self.r_bond  = np.asarray(r_bond,  dtype=np.float64)
        self.r_L_MtM = np.asarray(r_L_MtM,dtype=np.float64)
        self.r_L_blended = (
            np.asarray(r_L_blended, dtype=np.float64)
            if r_L_blended is not None else None
        )
        self.pi      = np.asarray(pi_monthly, dtype=np.float64)
        self.dates   = pd.DatetimeIndex(dates)
        self.T       = len(self.dates)

        lb = self.cfg.lookback
        n  = self.cfg.n_features

        # ---- Gymnasium spaces ------------------------------------------- #
        n_sol    = 2 + self.cfg.n_extra_solvency            # 5 solvency scalars
        obs_dim  = n_sol + lb * n                           # 5 + 372 = 377
        obs_low  = np.full(obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(obs_dim,  np.inf, dtype=np.float32)
        obs_low[0],  obs_high[0]  = 0.0, 5.0               # FR
        obs_low[1],  obs_high[1]  = 0.0, self.cfg.b_max    # B
        obs_low[2],  obs_high[2]  = 0.0, np.inf            # annual_o_plus (cumul. overrendement)
        obs_low[3],  obs_high[3]  = 0.0, self.cfg.b_max    # annual_fill_used (capped at B_max)
        obs_low[4],  obs_high[4]  = 0.0, 1.0               # month_norm (Jan=0, Dec=1)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        act_low  = np.array(
            [self.cfg.eq_tilt_min, 0.0, 0.0], dtype=np.float32
        )
        act_high = np.array(
            [self.cfg.eq_tilt_max, self.cfg.fill_rate_max, self.cfg.dist_rate_max],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        # ---- Mutable state (set properly in reset()) -------------------- #
        self._t:   int   = 0
        self._fr:  float = self.cfg.fr_init
        self._b:   float = self.cfg.b_init

        # Annual Art. 10d lid 2 trackers
        self._annual_o_plus:    float = 0.0
        self._annual_fill_used: float = 0.0

        # Rolling cohort return and inflation histories  (shape: (3, lb) and (lb,))
        self._cohort_hist = np.zeros((3, lb), dtype=np.float64)
        self._pi_hist     = np.zeros(lb,      dtype=np.float64)

        # Seed RNG (Gymnasium convention)
        if seed is not None:
            self.reset(seed=seed)

    # ---------------------------------------------------------------------- #
    # Gymnasium interface                                                     #
    # ---------------------------------------------------------------------- #

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment to the start of the data period.

        Args:
            seed:    Optional RNG seed (passed to gymnasium).
            options: Optional dict with keys ``"fr_init"`` and/or
                     ``"b_init"`` to override the default initial conditions.

        Returns:
            ``(obs, info)`` where obs is the initial 374-dim observation and
            info contains ``"t"`` and ``"date"``.
        """
        super().reset(seed=seed)

        cfg = self.cfg
        lb  = cfg.lookback

        # Start at the first index with a full lookback window
        self._t  = lb - 1   # index 11 → month 12 of the data (Jan 2001 for train)
        self._fr = (options or {}).get("fr_init", cfg.fr_init)
        self._b  = (options or {}).get("b_init",  cfg.b_init)

        self._annual_o_plus    = 0.0
        self._annual_fill_used = 0.0
        self._cohort_hist      = np.zeros((3, lb), dtype=np.float64)
        self._pi_hist          = np.zeros(lb,      dtype=np.float64)
        self._ppv              = np.ones(3,         dtype=np.float64)  # PPV[young, mid, ret]
        self._prev_d_tilde:    float = 0.0   # previous month's effective distribution (run_012)
        self._prev_w_eq:       float = cfg.w_eq_base   # previous month's aggregate equity weight (TC tracking)

        obs  = self._get_obs()
        info = {"t": self._t, "date": str(self.dates[self._t].date())}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the simulation by one month.

        Steps
        -----
        1.  Clip action to valid ranges.
        2.  Compute equity weight w_eq from equity tilt e_t.
        3.  Compute portfolio return r_p and liability return r_L.
        4.  Accumulate annual overrendement O+ (Art. 10d lid 2).
        5.  Apply annual fill cap → effective fill f̃_t.
        6.  Update FR and B (both scale with the funding return).
        7.  Apply distribution rule → effective distribution d̃_t (lid 4).
        8.  Apply buffer bounds; handle December cap (lid 1).
        9.  Reset annual trackers when January starts.
        10. Update three-cohort return histories.
        11. Compute composite reward.
        12. Advance time index; check termination.

        Args:
            action: ``(3,)`` array [equity_tilt, fill_rate, dist_rate].

        Returns:
            ``(obs, reward, terminated, truncated, info)``
        """
        t   = self._t
        cfg = self.cfg

        # ---- 1. Clip actions to valid ranges ---------------------------- #
        e_t = float(np.clip(action[0], cfg.eq_tilt_min,   cfg.eq_tilt_max))
        f_t = float(np.clip(action[1], 0.0,               cfg.fill_rate_max))
        d_t = float(np.clip(action[2], 0.0,               cfg.dist_rate_max))

        # ---- 2. Equity weight(s) --------------------------------------- #
        r_eq_t   = float(self.r_eq[t])
        r_bond_t = float(self.r_bond[t])

        if cfg.use_lifecycle:
            # Per-cohort portfolio returns (Wtp SPR personal share model)
            w_eq_young = float(np.clip(cfg.w_eq_young_base + e_t,
                                       cfg.w_eq_young_min, cfg.w_eq_young_max))
            w_eq_mid   = float(np.clip(cfg.w_eq_mid_base   + e_t,
                                       cfg.w_eq_mid_min,   cfg.w_eq_mid_max))
            w_eq_ret   = float(np.clip(cfg.w_eq_ret_base   + e_t,
                                       cfg.w_eq_ret_min,   cfg.w_eq_ret_max))
            r_p_young = w_eq_young * r_eq_t + (1.0 - w_eq_young) * r_bond_t
            r_p_mid   = w_eq_mid   * r_eq_t + (1.0 - w_eq_mid)   * r_bond_t
            r_p_ret   = w_eq_ret   * r_eq_t + (1.0 - w_eq_ret)   * r_bond_t
            # Aggregate return for FR dynamics (liability-share weighted)
            r_p_t = (cfg.w_young * r_p_young
                   + cfg.w_mid   * r_p_mid
                   + cfg.w_ret   * r_p_ret)
            w_eq  = r_p_t / (r_eq_t - r_bond_t + 1e-8) if abs(r_eq_t - r_bond_t) > 1e-8 \
                    else cfg.w_eq_base   # implied aggregate weight (info only)
        else:
            # Legacy single equity weight
            w_eq  = float(np.clip(cfg.w_eq_base + e_t, cfg.w_eq_min, cfg.w_eq_max))
            r_p_t = w_eq * r_eq_t + (1.0 - w_eq) * r_bond_t
            r_p_young = r_p_mid = r_p_ret = r_p_t

        # ---- 2b. Transaction cost drag (equity turnover) ------------------- #
        # Implied aggregate equity weight for TC purposes.
        # recompute directly from tilt so TC reflects the actual tilt magnitude.
        if cfg.use_lifecycle:
            w_eq_agg = (cfg.w_young * float(np.clip(cfg.w_eq_young_base + e_t,
                                                     cfg.w_eq_young_min, cfg.w_eq_young_max))
                      + cfg.w_mid   * float(np.clip(cfg.w_eq_mid_base   + e_t,
                                                     cfg.w_eq_mid_min,   cfg.w_eq_mid_max))
                      + cfg.w_ret   * float(np.clip(cfg.w_eq_ret_base   + e_t,
                                                     cfg.w_eq_ret_min,   cfg.w_eq_ret_max)))
        else:
            w_eq_agg = w_eq
        if cfg.tc_bps > 0.0:
            turnover   = abs(w_eq_agg - self._prev_w_eq)
            tc_drag    = (cfg.tc_bps / 10_000.0) * turnover
            r_p_t      -= tc_drag
            r_p_young  -= tc_drag
            r_p_mid    -= tc_drag
            r_p_ret    -= tc_drag
        self._prev_w_eq = w_eq_agg

        # ---- 3. Liability return ---------------------------------------- #

        r_L_MtM_t = float(self.r_L_MtM[t])
        if self.r_L_blended is not None:
            # Use pre-computed DNB RTS blended return
            r_L_t = float(self.r_L_blended[t])
        else:
            # Fallback: blend MtM with UFR in-house
            r_L_t = (
                cfg.liability_mtm_weight       * r_L_MtM_t
                + (1.0 - cfg.liability_mtm_weight) * cfg.r_ufr
            )

        # ---- 4. Annual overrendement accumulation ----------------------- #
        over_t = r_p_t - r_L_t
        self._annual_o_plus += max(0.0, over_t)

        # ---- 5. Effective fill (Art. 10d lid 2) ------------------------- #
        f_tilde = apply_annual_fill_cap(
            f_t, self._annual_o_plus, self._annual_fill_used, cfg
        )
        self._annual_fill_used += f_tilde

        # ---- 6. FR and B update ---------------------------------------- #
        # Both the main fund and the buffer earn the same funding return.
        # Fills move assets from the main fund into the buffer.
        fr_old = self._fr
        b_old  = self._b
        growth = (1.0 + r_p_t) / max(1.0 + r_L_t, 1e-8)

        fr_new = fr_old * growth - f_tilde
        b_new  = b_old  * growth + f_tilde

        # ---- 7. Effective distribution (Art. 10d lid 4) ---------------- #
        d_tilde = apply_distribution_rule(d_t, b_new, fr_new, cfg)
        b_new  -= d_tilde

        # ---- 8. Buffer bounds (Art. 10d lid 1) ------------------------- #
        is_dec = (self.dates[t].month == 12)
        fr_new, b_new, dec_excess = apply_buffer_bounds(fr_new, b_new, is_dec, cfg)

        # ---- 9. Annual reset when January of the next step arrives ------ #
        if t + 1 < self.T and self.dates[t + 1].month == 1:
            self._annual_o_plus    = 0.0
            self._annual_fill_used = 0.0

        # ---- 10. Cohort return histories -------------------------------- #
        pi_t = float(self.pi[t]) if not np.isnan(self.pi[t]) else 0.0
        self._update_cohort_history(
            r_p_young, r_p_mid, r_p_ret, d_tilde, dec_excess, f_tilde, fr_old, fr_new, pi_t
        )

        # ---- 10b. PPV update (Personal Pension Capital, PPV framework) -- #
        if cfg.use_lifecycle:
            total_dist_t = d_tilde + dec_excess
            self._ppv[0] = max(self._ppv[0] * (1.0 + r_p_young - f_tilde + total_dist_t), 1e-8)
            self._ppv[1] = max(self._ppv[1] * (1.0 + r_p_mid   - f_tilde + total_dist_t), 1e-8)
            self._ppv[2] = max(self._ppv[2] * (1.0 + r_p_ret   - f_tilde + total_dist_t), 1e-8)

        # ---- 11. Reward ------------------------------------------------- #
        reward = self._compute_reward(fr_new, b_new, d_tilde, f_tilde)

        # ---- 12. Advance; check termination ----------------------------- #
        self._fr = fr_new
        self._b  = b_new
        self._t += 1

        terminated = bool(fr_new < cfg.fr_catastrophe)
        truncated  = bool(self._t >= self.T)

        obs  = self._get_obs()
        info = {
            "t":          t,
            "date":       str(self.dates[t].date()),
            "FR":         fr_new,
            "B":          b_new,
            "w_eq":       w_eq,
            "r_p":        r_p_t,
            "r_p_young":  r_p_young,
            "r_p_mid":    r_p_mid,
            "r_p_ret":    r_p_ret,
            "r_L":        r_L_t,
            "over":       over_t,
            "f_tilde":    f_tilde,
            "d_tilde":    d_tilde,
            "dec_excess": dec_excess,
            "ppv_young":  float(self._ppv[0]),
            "ppv_mid":    float(self._ppv[1]),
            "ppv_ret":    float(self._ppv[2]),
        }
        return obs, float(reward), terminated, truncated, info

    # ---------------------------------------------------------------------- #
    # Private helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _get_obs(self) -> np.ndarray:
        """Build the 377-dim observation [FR, B, o_plus, fill_used, month, z_flat].
""" 
      
              t  = self._t
        lb = self.cfg.lookback

        # Slice lookback window; zero-pad on the left if at episode start
        start  = max(0, t - lb + 1)
        window = self.z[start : t + 1]          # up to (lb, 31)

        if len(window) < lb:
            pad    = np.zeros((lb - len(window), self.cfg.n_features), dtype=np.float32)
            window = np.vstack([pad, window])

        t_safe     = min(t, self.T - 1)
        month_norm = (self.dates[t_safe].month - 1) / 11.0
        solvency = np.array(
            [self._fr, self._b, self._annual_o_plus, self._annual_fill_used, month_norm],
            dtype=np.float32,
        )
        return np.concatenate([solvency, window.flatten()])

    def _update_cohort_history(
        self,
        r_p_young:  float,
        r_p_mid:    float,
        r_p_ret:    float,
        d_tilde:    float,
        dec_excess: float,
        f_tilde:    float,
        fr_old:     float,
        fr_new:     float,
        pi_t:       float,
    ) -> None:
        """Shift rolling cohort-return and inflation histories; append new period.

        PPV framework (use_lifecycle=True, run_010+):
          R_{i,t} = w_i^eq * r_eq + (1-w_i^eq) * r_bond - f̃_t + (d̃_t + dec_excess)
          All cohorts bear the same solidarity cost (f̃_t) and receive the same
          distribution benefit; only the lifecycle equity mix differs.

        Legacy mode (use_lifecycle=False, run_007 / run_008b):
          Young      (i=0) : R_1 = r_p_agg
          Mid-career (i=1) : R_2 = (FR_new - FR_old) / FR_old
          Retired    (i=2) : R_3 = d̃_t / 0.45
        """
        if self.cfg.use_lifecycle:
            total_dist = d_tilde + dec_excess
            r_young = r_p_young - f_tilde + total_dist   # PPV formula
            r_mid   = r_p_mid   - f_tilde + total_dist
            r_ret   = r_p_ret   - f_tilde + total_dist
        else:
            r_young = r_p_young                               # == r_p_agg in legacy
            r_mid   = (fr_new - fr_old) / max(fr_old, 1e-8)
            r_ret   = d_tilde / 0.45

        # Roll left (oldest month out, newest month in at index -1)
        self._cohort_hist = np.roll(self._cohort_hist, -1, axis=1)
        self._cohort_hist[:, -1] = [r_young, r_mid, r_ret]

        self._pi_hist = np.roll(self._pi_hist, -1)
        self._pi_hist[-1] = pi_t

    def _compute_equity_term(self) -> float:
        """Compute E_t = cross-cohort variance of 12M real replacement rates.

        For each cohort i:
            RR_{i,t} = sum_{k=t-11}^{t}  ln(1 + R_{i,k} - pi_k)

        Returns:
            Var(RR_1, RR_2, RR_3)  — cross-cohort variance.
        """
        pi  = self._pi_hist         # (lb,)
        rr  = np.empty(3)

        for i in range(3):
            log_sum = 0.0
            for k in range(self.cfg.lookback):
                arg      = 1.0 + self._cohort_hist[i, k] - pi[k]
                log_sum += np.log(max(arg, 1e-8))    # guard log(0)
            rr[i] = log_sum

        return float(np.var(rr))   # Var of 3 values = cross-cohort spread

    def set_gamma(self, gamma: float) -> None:
        """Update the buffer-depletion penalty weight in-place (used by curriculum callback)."""
        self.cfg.gamma = gamma

    def _compute_reward(
        self,
        fr:      float,
        b:       float,
        d_tilde: float,
        f_tilde: float,
    ) -> float:
        """Lexicographic reward: hard priority ordering over three regions.

        Implements the institutional mandate hierarchy as a conditional
        structure rather than competing weighted terms:

        Priority 1 — MVEV compliance (Besluit FTK Art. 11a)
            If FR < 1.043: return -delta * shortfall²
            Nothing else matters; agent must restore solvency first.

        Priority 2 — Buffer preservation (Art. 10d lid 1)
            If B < 0.01: return -gamma * shortfall²
            Must rebuild buffer before distributing.

        Priority 3 — Safe-zone optimisation (normal operations)
            Reached only when FR ≥ 1.043 AND B ≥ 0.01.
            Optimise: solvency stability + distributions + fills − equity variance.

        Args:
            fr:      Funding ratio at time t.
            b:       Buffer level (fraction of liabilities) at time t.
            d_tilde: Realised distribution after Art. 10d lid 4.
            f_tilde: Realised fill after Art. 10d lid 2 annual cap.

        Returns:
            Scalar reward.
        """
        cfg = self.cfg

        # ---- Priority 1: MVEV floor ------------------------------------ #
        if fr < cfg.fr_mvev:
            shortfall = cfg.fr_mvev - fr
            return -cfg.delta * (shortfall ** 2)

        # ---- Priority 2: Buffer near-depletion ------------------------- #
        buffer_crisis_threshold = 0.01
        if b < buffer_crisis_threshold:
            shortfall = buffer_crisis_threshold - b
            return -cfg.gamma * (shortfall ** 2)

        # ---- Priority 3: Safe-zone optimisation ------------------------ #

        # Solvency stability: one-sided shortfall below FR_target
        S_t = -max(0.0, cfg.fr_target - fr) ** 2

        # Distribution incentive: piecewise conservative buffer-health scaling
        # Zone 1 (Danger)      : B < 0.05  → circuit breaker, no dist reward
        # Zone 2 (Rebuilding)  : 0.05 <= B < 0.08 → ramps 0.0 → 1.0
        # Zone 3 (Optimisation): 0.08 <= B < 0.15 → ramps 1.0 → 1.5
        # Zone 4 (Cap)         : B >= 0.15 → capped at 1.5
        if b < 0.05:
            buffer_scaling = 0.0
        elif b < 0.08:
            buffer_scaling = (b - 0.05) / (0.08 - 0.05)          # 0.0 → 1.0
        elif b < 0.15:
            buffer_scaling = 1.0 + 0.5 * (b - 0.08) / (0.15 - 0.08)  # 1.0 → 1.5
        else:
            buffer_scaling = 1.5
        Q_t = cfg.dist_weight * d_tilde * buffer_scaling

        # Fill reward: direct gradient signal for buffer building
        F_t = cfg.fill_bonus * f_tilde

        # Intergenerational equity term
        E_t = self._compute_equity_term() if cfg.epsilon_equity > 0.0 else 0.0

        return (cfg.alpha * S_t
                + cfg.beta * Q_t
                + F_t
                - cfg.epsilon_equity * E_t)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_env_from_pipeline(
    results: dict,
    split:   str = "train",
    cfg:     Optional[EnvConfig] = None,
    seed:    Optional[int] = None,
) -> WtpPensionEnv:
    """Build a :class:`WtpPensionEnv` directly from pipeline results.

    Extracts the market return series needed for transition dynamics from the
    *unscaled* feature splits (which contain d_swap_10y, d_swap_20y, and the
    MSCI log-return), then instantiates the environment.

    Args:
        results: Dictionary returned by ``data_pipeline.run_pipeline()``.
        split:   One of ``"train"``, ``"val"``, or ``"test"``.
        cfg:     Optional :class:`EnvConfig`.
        seed:    Optional random seed.

    Returns:
        Configured :class:`WtpPensionEnv` instance.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

    env_cfg  = cfg or EnvConfig()
    z_scaled = results[f"z_{split}"].values      # (T, 31) scaled
    z_raw    = results[f"z_{split}_raw"]          # (T, 31) unscaled — for dynamics
    cpi      = results["cpi"]
    dates    = results[f"z_{split}"].index        # month-end DatetimeIndex

    # --- CPI aligned to feature dates ------------------------------------ #
    pi_series = cpi["pi_monthly"].reindex(dates).fillna(0.0).values

    # --- Market return series from unscaled features --------------------- #
    # r_eq:   MSCI World simple return = exp(log-return) - 1, then clip
    r_eq = (
        (np.exp(z_raw["mom_msci_1m"].values) - 1.0)
        .clip(env_cfg.r_eq_clip[0], env_cfg.r_eq_clip[1])
    )

    # r_bond: -Duration × DeltaSwap_10Y / 100, then clip
    r_bond = (
        (-env_cfg.duration * z_raw["d_swap_10y"].values / 100.0)
        .clip(env_cfg.r_bond_clip[0], env_cfg.r_bond_clip[1])
    )

    # r_L_MtM: from RTS file if available, else swap-based fallback
    if "d_rts_20y" in z_raw.columns:
        r_L_MtM = -env_cfg.duration * z_raw["d_rts_20y"].values / 100.0
    else:
        r_L_MtM = -env_cfg.duration * z_raw["d_swap_20y"].values / 100.0

    # r_L_blended: pre-computed DNB RTS blended return (official methodology)
    r_L_blended = None
    if "r_L_blended" in results:
        r_L_blended = (
            results["r_L_blended"]
            .reindex(dates)
            .fillna(0.0)
            .values
        )

    return WtpPensionEnv(
        z_scaled    = z_scaled,
        r_eq        = r_eq,
        r_bond      = r_bond,
        r_L_MtM     = r_L_MtM,
        pi_monthly  = pi_series,
        dates       = dates,
        cfg         = env_cfg,
        seed        = seed,
        r_L_blended = r_L_blended,
    )


# ---------------------------------------------------------------------------
# Main — sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_pipeline import run_pipeline

    print("=" * 64)
    print("Wtp DRL Pension Fund -- Environment Sanity Check")
    print("=" * 64)

    print("\n[1/3] Running data pipeline...")
    results = run_pipeline()

    print("[2/3] Building training environment...")
    env = make_env_from_pipeline(results, split="train", seed=42)

    print(f"      Observation space : {env.observation_space.shape}  (expected (377,))")
    print(f"      Action space      : {env.action_space.shape}  (expected (3,))")
    print(f"      Data steps        : {env.T}")

    print("\n[3/3] Running 24 steps with random actions...\n")
    obs, info = env.reset(seed=42)
    assert obs.shape == (377,), f"Unexpected obs shape: {obs.shape}"

    header = (
        f"  {'Step':>4}  {'Date':<12}  {'FR':>7}  {'B':>7}  "
        f"{'w_eq':>6}  {'r_p%':>7}  {'r_L%':>7}  "
        f"{'f~':>6}  {'d~':>6}  {'reward':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_reward = 0.0
    for step in range(24):
        action              = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward       += reward

        print(
            f"  {step+1:4d}  {info['date']:<12}  {info['FR']:7.4f}  "
            f"{info['B']:7.4f}  {info['w_eq']:6.3f}  "
            f"{info['r_p']*100:7.3f}  {info['r_L']*100:7.3f}  "
            f"{info['f_tilde']:6.4f}  {info['d_tilde']:6.4f}  "
            f"{reward:8.4f}"
        )

        if terminated:
            print(f"\n  *** Episode terminated early (FR < {env.cfg.fr_catastrophe}) ***")
            break
        if truncated:
            print("\n  *** Data exhausted ***")
            break

    print(f"\n  Total reward (24 steps): {total_reward:.4f}")

    # Verify no NaN leaked into observations
    nan_count = int(np.isnan(obs).sum())
    status    = "OK" if nan_count == 0 else f"WARNING: {nan_count} NaN(s)"
    print(f"  Final obs NaN check    : {status}")

    print("\nDone.")
    sys.exit(0)
