"""baselines.py — Fixed-Rule ALM and Monte Carlo ALM baseline models.

Both baselines implement the same Wtp SPR action space as the DRL agent
(equity tilt, fill rate, distribution rate) and are subject to identical
Art. 10d constraints enforced by the environment.

Fixed-Rule ALM
--------------
Static 60 % equity allocation with fixed fill rate 0.03 and distribution
rate 0.02.  No state-dependent logic.

Monte Carlo ALM
---------------
1. Fit a VAR(P) model on [r_eq, DeltaSwap_10Y, pi_monthly] from training
   data.  Lag P is selected by BIC (up to max_lag).
2. A supplementary OLS regression derives DeltaSwap_20Y from DeltaSwap_10Y
   for the liability calculation.
3. Generate n_scenarios random paths by bootstrapping VAR residuals.
4. Grid-search over (f, d) pairs to minimise buffer depletion frequency
   P(B_t <= depletion_threshold) using a fast vectorised NumPy simulation.
5. Apply the optimal (f*, d*) as a fixed deterministic policy at test time.

Usage
-----
    from src.data_pipeline  import run_pipeline
    from src.environment    import make_env_from_pipeline, EnvConfig
    from src.baselines      import FixedRuleALM, MonteCarloALM, run_episode

    results  = run_pipeline()
    env_cfg  = EnvConfig()

    fixed    = FixedRuleALM()
    mc       = MonteCarloALM()
    mc.fit(results["z_train_raw"], results["raw_train"], results["cpi"], env_cfg)

    env      = make_env_from_pipeline(results, split="test")
    traj     = run_episode(fixed, env)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.api import VAR
except ImportError as exc:
    raise ImportError(
        "statsmodels is required.  Install with:  pip install statsmodels"
    ) from exc


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

@dataclass
class FixedRuleConfig:
    """Parameters for the Fixed-Rule ALM baseline."""
    equity_tilt: float = 0.00    # -> w_eq = 0.55 + 0.00 = 0.55
    fill_rate:   float = 0.03
    dist_rate:   float = 0.02


@dataclass
class MonteCarloConfig:
    """Parameters for the Monte Carlo ALM baseline."""

    # VAR fitting
    max_lag:     int   = 6       # BIC search up to this lag
    var_vars:    list  = field(default_factory=lambda: [
        "r_eq", "d_swap_10y", "pi_monthly"
    ])

    # Simulation
    n_scenarios:  int   = 1_000
    sim_horizon:  int   = 180    # months per simulated path
    seed:         int   = 42

    # Grid search
    f_grid: list = field(default_factory=lambda: [
        round(v, 2) for v in np.arange(0.00, 0.11, 0.01).tolist()
    ])
    d_grid: list = field(default_factory=lambda: [
        round(v, 2) for v in np.arange(0.00, 0.06, 0.01).tolist()
    ])

    # Depletion threshold used in grid search objective
    depletion_threshold: float = 0.001   # 0.1 % of liabilities

    # Grid search objective: maximise expected total participant distributions.
    # Under Wtp Art. 10d the annual fill cap is too small to prevent year-end
    # buffer depletion for any d > 0, so pure depletion-minimisation always
    # collapses to the trivial d*=0 solution.  We therefore find the policy
    # that maximises E[sum_t d_tilde_t] over the simulated VAR paths, which
    # represents the maximum sustainable payout rate consistent with the
    # simulated return dynamics.
    # (depletion_budget retained for reference but not used in optimisation.)
    depletion_budget: float = 0.30

    # Liability / fund dynamics (mirrors EnvConfig defaults)
    duration:             float = 18.0
    r_ufr:                float = 0.002711  # (1.033)^(1/12)-1 = 3.30% ann., DNB/EIOPA UFR
    liability_mtm_weight: float = 0.70
    w_eq:                 float = 0.55   # no equity tilt for MC baseline
    r_eq_clip:            tuple = (-0.30, +0.30)
    r_bond_clip:          tuple = (-0.05, +0.05)
    fr_init:              float = 1.05
    b_init:               float = 0.05
    b_max:                float = 0.15
    annual_fill_cap_frac: float = 0.10
    fr_dist_threshold:    float = 1.00


# ---------------------------------------------------------------------------
# Fixed-Rule ALM
# ---------------------------------------------------------------------------

class FixedRuleALM:
    """Static 55/45 rebalancing policy with fixed fill and distribution rates.

    The agent always returns the same action regardless of the observed
    state.  The environment enforces all Art. 10d constraints.

    Args:
        cfg: Optional :class:`FixedRuleConfig`.
    """

    def __init__(self, cfg: Optional[FixedRuleConfig] = None) -> None:
        self.cfg    = cfg or FixedRuleConfig()
        self._action = np.array(
            [self.cfg.equity_tilt, self.cfg.fill_rate, self.cfg.dist_rate],
            dtype=np.float32,
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return the fixed action, ignoring the observation.

        Args:
            obs: 374-dim observation (ignored).

        Returns:
            ``(3,)`` action array [equity_tilt, fill_rate, dist_rate].
        """
        return self._action.copy()

    def __repr__(self) -> str:
        c = self.cfg
        return (
            f"FixedRuleALM(w_eq={0.55 + c.equity_tilt:.2f}, "
            f"f={c.fill_rate}, d={c.dist_rate})"
        )


# ---------------------------------------------------------------------------
# Monte Carlo ALM — fast vectorised simulation
# ---------------------------------------------------------------------------

def _fit_var(
    series: pd.DataFrame,
    max_lag: int,
) -> tuple:
    """Fit VAR(P) model; select lag P by BIC.

    Args:
        series:  DataFrame of stationary endogenous variables (T, K).
        max_lag: Maximum lag to consider.

    Returns:
        ``(fitted_model, lag_order)``
    """
    model   = VAR(series)
    results = model.fit(maxlags=max_lag, ic="bic", trend="c")
    return results, results.k_ar


def _bootstrap_scenarios(
    var_result,
    n_scenarios: int,
    sim_horizon: int,
    seed: int,
) -> np.ndarray:
    """Generate scenarios by bootstrapping VAR residuals.

    Draws with replacement from the fitted residuals to preserve the
    cross-sectional correlation structure without normality assumptions.

    Args:
        var_result:   Fitted statsmodels VARResults object.
        n_scenarios:  Number of paths to generate.
        sim_horizon:  Length of each path in months.
        seed:         NumPy random seed.

    Returns:
        Array of shape ``(sim_horizon, n_scenarios, K)`` where K is the
        number of VAR variables.
    """
    rng       = np.random.default_rng(seed)
    residuals = var_result.resid.values          # (T_fit - lag, K)
    coefs     = var_result.coefs                 # (lag, K, K)
    intercept = var_result.intercept             # (K,)
    lag       = var_result.k_ar
    K         = residuals.shape[1]

    # Initialise each scenario from the last `lag` observations of training
    history_init = var_result.model.endog[-lag:, :]   # (lag, K)

    # Shape: (n_scenarios, lag, K)
    history = np.tile(history_init, (n_scenarios, 1, 1)).astype(np.float64)

    T_res    = residuals.shape[0]
    out      = np.empty((sim_horizon, n_scenarios, K), dtype=np.float64)

    for t in range(sim_horizon):
        # Deterministic part: sum over lags
        forecast = intercept.copy()               # (K,)
        for p in range(lag):
            # history[:, lag-1-p, :] is the p+1 lag for each scenario
            forecast = forecast + (history[:, lag - 1 - p, :] @ coefs[p].T)

        # Stochastic part: bootstrap residual
        idx      = rng.integers(0, T_res, size=n_scenarios)
        shocks   = residuals[idx, :]              # (n_scenarios, K)

        step = forecast + shocks                  # (n_scenarios, K)
        out[t] = step

        # Shift history window
        history = np.roll(history, -1, axis=1)
        history[:, -1, :] = step

    return out   # (sim_horizon, n_scenarios, K)


def _vectorised_simulation(
    r_eq:     np.ndarray,    # (T, N) equity return
    r_bond:   np.ndarray,    # (T, N) bond return proxy
    r_L_MtM:  np.ndarray,    # (T, N) liability MtM return
    f:        float,
    d:        float,
    cfg:      MonteCarloConfig,
) -> dict:
    """Run all N scenarios in parallel for a fixed (f, d) policy.

    Mirrors the WtpPensionEnv transition dynamics exactly.  Calendar month
    is estimated from position within the sim_horizon (month 0 = January).

    Args:
        r_eq:    Equity return scenarios ``(T, N)``.
        r_bond:  Bond return scenarios ``(T, N)``.
        r_L_MtM: Liability MtM return scenarios ``(T, N)``.
        f:       Fixed fill rate.
        d:       Fixed distribution rate.
        cfg:     :class:`MonteCarloConfig`.

    Returns:
        Dict with ``"depletion_prob"``, ``"FR_terminal"``, ``"B_terminal"``.
    """
    T, N = r_eq.shape

    FR = np.full(N, cfg.fr_init,  dtype=np.float64)
    B  = np.full(N, cfg.b_init,   dtype=np.float64)
    annual_o_plus    = np.zeros(N, dtype=np.float64)
    annual_fill_used = np.zeros(N, dtype=np.float64)
    # Depletion counted only at year-end (Dec 31) so mid-year distributions
    # that temporarily empty the buffer do not falsely trigger depletion.
    year_end_depleted = np.zeros(N, dtype=bool)
    total_dist        = np.zeros(N, dtype=np.float64)

    w_eq = cfg.w_eq

    for t in range(T):
        month = (t % 12) + 1   # 1-indexed calendar month estimate

        r_p = w_eq * r_eq[t] + (1.0 - w_eq) * r_bond[t]
        r_L = (
            cfg.liability_mtm_weight       * r_L_MtM[t]
            + (1.0 - cfg.liability_mtm_weight) * cfg.r_ufr
        )

        over = r_p - r_L
        annual_o_plus += np.maximum(0.0, over)

        # Art. 10d lid 2: annual fill cap
        budget    = cfg.annual_fill_cap_frac * annual_o_plus
        remaining = np.maximum(0.0, budget - annual_fill_used)
        f_tilde   = np.minimum(f, remaining)
        annual_fill_used += f_tilde

        # FR and B update (both earn funding return)
        growth = (1.0 + r_p) / np.maximum(1.0 + r_L, 1e-8)
        FR_new = FR * growth - f_tilde
        B_new  = B  * growth + f_tilde

        # Art. 10d lid 4: distribution rule
        can_dist  = (FR_new >= cfg.fr_dist_threshold) & (B_new > 0.0)
        d_tilde   = np.where(can_dist, np.minimum(d, B_new), 0.0)
        B_new    -= d_tilde
        total_dist += d_tilde

        # Art. 10d lid 1: negative buffer absorbed by FR
        shortfall = B_new < 0.0
        FR_new    = np.where(shortfall, FR_new + B_new, FR_new)
        B_new     = np.maximum(B_new, 0.0)

        # Art. 10d lid 1: December cap + year-end depletion check
        if month == 12:
            B_new = np.minimum(B_new, cfg.b_max)
            year_end_depleted |= (B_new <= cfg.depletion_threshold)
            annual_o_plus    = np.zeros(N, dtype=np.float64)
            annual_fill_used = np.zeros(N, dtype=np.float64)

        FR = FR_new
        B  = B_new

    return {
        "depletion_prob":    float(year_end_depleted.mean()),
        "mean_total_dist":   float(total_dist.mean()),
        "FR_terminal":       FR,
        "B_terminal":        B,
    }


class MonteCarloALM:
    """VAR(P)-based Monte Carlo optimisation for fill and distribution rates.

    After calling :meth:`fit`, :meth:`predict` returns the optimal fixed
    policy ``(0, f*, d*)`` (no equity tilt; equity weight = w_eq = 0.55).

    Args:
        cfg: Optional :class:`MonteCarloConfig`.
    """

    def __init__(self, cfg: Optional[MonteCarloConfig] = None) -> None:
        self.cfg          = cfg or MonteCarloConfig()
        self.var_result   = None
        self.lag_order    = None
        self._swap20_beta = None    # OLS coefficient: d_swap_20y ~ d_swap_10y
        self._swap20_intercept = 0.0
        self.f_star       = self.cfg.f_grid[0]
        self.d_star       = self.cfg.d_grid[0]
        self._is_fitted   = False

    # ------------------------------------------------------------------ #
    # Fitting                                                             #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        z_raw_train:  pd.DataFrame,
        raw_train:    pd.DataFrame,
        cpi:          pd.DataFrame,
        env_cfg,
    ) -> "MonteCarloALM":
        """Fit VAR(P) and find optimal (f*, d*) via Monte Carlo grid search.

        Args:
            z_raw_train: Unscaled training features from pipeline.
            raw_train:   Raw monthly training DataFrame from pipeline.
            cpi:         Monthly CPI DataFrame from pipeline.
            env_cfg:     EnvConfig (used for duration, r_ufr, etc.).

        Returns:
            self (for chaining).
        """
        cfg = self.cfg

        # ---- 1. Build VAR input series from RAW (unscaled) data ----------- #
        r_eq_s = raw_train["Equity_World_MSCI"].pct_change().values
        r_eq_s = np.clip(r_eq_s, cfg.r_eq_clip[0], cfg.r_eq_clip[1])

        d_swap_10y_s = raw_train["Swap_10Y"].diff().values
        col_20y = "RTS_20Y" if "RTS_20Y" in raw_train.columns else "Swap_20Y"
        d_swap_20y_s = raw_train[col_20y].diff().values

        pi_s = (
            cpi["pi_monthly"]
            .reindex(raw_train.index)
            .fillna(0.0)
            .values
        )

        var_data = pd.DataFrame(
            {
                "r_eq":       r_eq_s,
                "d_swap_10y": d_swap_10y_s,
                "pi_monthly": pi_s,
            },
            index=raw_train.index,
        ).dropna()

        print(f"  Fitting VAR (max_lag={cfg.max_lag}) on {len(var_data)} training months...")
        self.var_result, self.lag_order = _fit_var(var_data, cfg.max_lag)
        print(f"  VAR lag selected by BIC: P={self.lag_order}")

        # ---- 2. OLS: d_swap_20y ~ d_swap_10y (for liability simulation) - #
        x = d_swap_10y_s
        y = d_swap_20y_s
        mask = np.isfinite(x) & np.isfinite(y)
        x_m, y_m = x[mask], y[mask]
        beta, intercept = np.polyfit(x_m, y_m, 1)
        self._swap20_beta      = float(beta)
        self._swap20_intercept = float(intercept)
        print(f"  OLS d_swap_20y ~ d_swap_10y: beta={beta:.4f}, intercept={intercept:.6f}")

        # ---- 3. Simulate scenarios --------------------------------------- #
        print(f"  Simulating {cfg.n_scenarios} scenarios (horizon={cfg.sim_horizon} months)...")
        scenarios = _bootstrap_scenarios(
            self.var_result,
            n_scenarios = cfg.n_scenarios,
            sim_horizon = cfg.sim_horizon,
            seed        = cfg.seed,
        )
        # scenarios: (T, N, 3)  cols: [r_eq, d_swap_10y, pi]

        # Clip and derive return series
        r_eq_scen  = np.clip(
            scenarios[:, :, 0], cfg.r_eq_clip[0], cfg.r_eq_clip[1]
        )
        d_10y_scen = scenarios[:, :, 1]

        # Bond return from d_swap_10y
        r_bond_scen = np.clip(
            -env_cfg.duration * d_10y_scen / 100.0,
            cfg.r_bond_clip[0], cfg.r_bond_clip[1],
        )

        # Liability MtM from OLS-derived d_swap_20y
        d_20y_scen = self._swap20_intercept + self._swap20_beta * d_10y_scen
        r_L_MtM_scen = -env_cfg.duration * d_20y_scen / 100.0

        # ---- 4. Grid search --------------------------------------------- #
        print(
            f"  Grid search over {len(cfg.f_grid)} x {len(cfg.d_grid)} "
            f"= {len(cfg.f_grid)*len(cfg.d_grid)} (f, d) combinations..."
        )

        # Objective: argmax E[total distributions] over simulated VAR paths.
        # Pure depletion-minimisation degenerates to d*=0 under Wtp Art. 10d
        # because the annual fill cap cannot prevent year-end buffer depletion
        # for any d > 0.  Maximising expected payouts identifies the policy
        # with the highest sustainable distribution rate.
        results_grid = []
        for f in cfg.f_grid:
            for d in cfg.d_grid:
                res = _vectorised_simulation(
                    r_eq_scen, r_bond_scen, r_L_MtM_scen, f, d, cfg
                )
                results_grid.append((f, d, res["depletion_prob"], res["mean_total_dist"]))

        best = max(results_grid, key=lambda x: x[3])   # argmax mean_total_dist
        best_f, best_d, best_dep, best_dist = best
        print(
            f"  Optimal policy (max E[dist]): "
            f"f*={best_f:.2f}, d*={best_d:.2f}  "
            f"(mean_dist={best_dist:.4f}, depletion={best_dep:.4f})"
        )

        self.f_star = best_f
        self.d_star = best_d
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict(self, _obs: np.ndarray) -> np.ndarray:
        """Return the fixed optimal policy, ignoring the observation.

        Args:
            _obs: 374-dim observation (ignored).

        Returns:
            ``(3,)`` action ``[0.0, f*, d*]`` (no equity tilt).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        return np.array([0.0, self.f_star, self.d_star], dtype=np.float32)

    def __repr__(self) -> str:
        if self._is_fitted:
            return (
                f"MonteCarloALM(lag={self.lag_order}, "
                f"f*={self.f_star}, d*={self.d_star})"
            )
        return "MonteCarloALM(unfitted)"


# ---------------------------------------------------------------------------
# Demonstration policy for behavioral cloning warmstart (run_035)
# ---------------------------------------------------------------------------

class DemonstrationPolicy:
    """Hand-designed rule-based policy for behavioral cloning warmstart.

    Produces (obs, action) demonstrations that teach the agent the
    "fill buffer first, then distribute" sequence the PPO agent failed to
    discover through pure exploration (runs 033-034 equity-collapse diagnosis).

    Phase 1 (months 0-35): Aggressive accumulation
      - Equity tilt +0.10 (65% equity), fill rate 5%, zero distributions.
      - Goal: build buffer from ~0.05 to 0.08-0.12 before first distribution.

    Phase 2 (months 36+): Balanced SPR operation
      - Neutral equity tilt (0.00, strategic 55%), fill rate 3%.
      - Progressive dist_rate mirrors run_041 buffer-health zones:
        B < 0.05 → 0%, B in [0.05, 0.10) → 1%, B in [0.10, 0.15) → 2%,
        B >= 0.15 → 3%.

    Interface matches ``run_episode``'s ``agent.predict(obs)`` convention,
    so DemonstrationPolicy can be passed directly to ``run_episode``.
    """

    def __init__(self) -> None:
        self.month_counter: int = 0

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Map flat observation to demonstration action.

        Args:
            obs: Flat observation array (shape 377).  obs[0]=FR, obs[1]=B.

        Returns:
            Action array ``[equity_tilt, fill_rate, dist_rate]``.
        """
        B  = float(obs[1])
        self.month_counter += 1

        if self.month_counter <= 36:
            # Phase 1: build buffer aggressively
            return np.array([0.10, 0.05, 0.00], dtype=np.float32)

        # Phase 2: progressive distribution — mirrors run_041 buffer-health zones
        if B < 0.05:
            dist_rate = 0.00
        elif B < 0.10:
            dist_rate = 0.01
        elif B < 0.15:
            dist_rate = 0.02
        else:
            dist_rate = 0.03
        return np.array([0.00, 0.03, dist_rate], dtype=np.float32)

    def reset(self) -> None:
        """Reset month counter at episode start."""
        self.month_counter = 0


# ---------------------------------------------------------------------------
# Episode runner (works with any agent that has predict(obs) -> action)
# ---------------------------------------------------------------------------

def run_episode(agent, env) -> dict:
    """Run a full episode and collect the trajectory.

    The agent must implement ``predict(obs: np.ndarray) -> np.ndarray``.

    Args:
        agent: Any object with a ``predict`` method (FixedRuleALM,
               MonteCarloALM, or trained DRL policy).
        env:   A :class:`~src.environment.WtpPensionEnv` instance.

    Returns:
        Dictionary with trajectory arrays and summary statistics:

        - ``dates``         : month-end dates of each step.
        - ``FR``            : funding ratio trajectory.
        - ``B``             : buffer trajectory.
        - ``w_eq``          : equity weight trajectory.
        - ``r_p``           : portfolio return trajectory.
        - ``r_L``           : liability return trajectory.
        - ``f_tilde``       : effective fill trajectory.
        - ``d_tilde``       : effective distribution trajectory.
        - ``rewards``       : per-step rewards.
        - ``total_reward``  : sum of rewards.
        - ``terminated``    : whether episode ended early (FR catastrophe).
        - ``n_steps``       : number of steps completed.
    """
    obs, info = env.reset()

    dates, FR, B              = [], [], []
    w_eq_t, r_p_t, r_L_t     = [], [], []
    r_p_young_t, r_p_mid_t, r_p_ret_t = [], [], []
    f_tilde_t, d_tilde_t, dec_excess_t = [], [], []
    ppv_young_t, ppv_mid_t, ppv_ret_t  = [], [], []
    rewards                   = []

    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        dates.append(info["date"])
        FR.append(info["FR"])
        B.append(info["B"])
        w_eq_t.append(info["w_eq"])
        r_p_t.append(info["r_p"])
        r_p_young_t.append(info.get("r_p_young", info["r_p"]))
        r_p_mid_t.append(info.get("r_p_mid",     info["r_p"]))
        r_p_ret_t.append(info.get("r_p_ret",     info["r_p"]))
        r_L_t.append(info["r_L"])
        f_tilde_t.append(info["f_tilde"])
        d_tilde_t.append(info["d_tilde"])
        dec_excess_t.append(info.get("dec_excess", 0.0))
        ppv_young_t.append(info.get("ppv_young",   1.0))
        ppv_mid_t.append(info.get("ppv_mid",       1.0))
        ppv_ret_t.append(info.get("ppv_ret",       1.0))
        rewards.append(reward)

    return {
        "dates":        dates,
        "FR":           np.array(FR),
        "B":            np.array(B),
        "w_eq":         np.array(w_eq_t),
        "r_p":          np.array(r_p_t),
        "r_p_young":    np.array(r_p_young_t),
        "r_p_mid":      np.array(r_p_mid_t),
        "r_p_ret":      np.array(r_p_ret_t),
        "r_L":          np.array(r_L_t),
        "f_tilde":      np.array(f_tilde_t),
        "d_tilde":      np.array(d_tilde_t),
        "dec_excess":   np.array(dec_excess_t),
        "ppv_young":    np.array(ppv_young_t),
        "ppv_mid":      np.array(ppv_mid_t),
        "ppv_ret":      np.array(ppv_ret_t),
        "rewards":      np.array(rewards),
        "total_reward": float(np.sum(rewards)),
        "terminated":   terminated,
        "n_steps":      len(rewards),
    }


# ---------------------------------------------------------------------------
# Main — fit and compare both baselines on the test split
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data_pipeline import run_pipeline
    from src.environment   import make_env_from_pipeline, EnvConfig

    print("=" * 64)
    print("Wtp DRL Pension Fund -- Baselines")
    print("=" * 64)

    print("\n[1/4] Running data pipeline...")
    results = run_pipeline()
    env_cfg = EnvConfig()

    # ---- Fixed-Rule ALM ------------------------------------------------- #
    print("\n[2/4] Fixed-Rule ALM (55/45, f=0.03, d=0.02)...")
    fixed = FixedRuleALM()
    env   = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)
    traj_fixed = run_episode(fixed, env)

    # ---- Monte Carlo ALM ------------------------------------------------ #
    print("\n[3/4] Monte Carlo ALM — fitting VAR and grid search...")
    mc = MonteCarloALM()
    mc.fit(
        z_raw_train = results["z_train_raw"],
        raw_train   = results["raw_train"],
        cpi         = results["cpi"],
        env_cfg     = env_cfg,
    )

    print("\n[4/4] Running Monte Carlo ALM on test set...")
    env       = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)
    traj_mc   = run_episode(mc, env)

    # ---- Summary table -------------------------------------------------- #
    def summarise(name: str, traj: dict) -> None:
        FR   = traj["FR"]
        B    = traj["B"]
        dist = traj["d_tilde"]

        fr_mdd = float(
            (np.maximum.accumulate(FR) - FR).max()
            / max(np.maximum.accumulate(FR).max(), 1e-8)
        )
        dep_freq = float((B <= 0.001).mean())

        print(
            f"  {name:<22}  "
            f"FR_terminal={FR[-1]:.4f}  "
            f"FR_MDD={fr_mdd:.4f}  "
            f"FR_vol={FR.std():.4f}  "
            f"B_dep_freq={dep_freq:.4f}  "
            f"sum_dist={dist.sum():.4f}  "
            f"total_reward={traj['total_reward']:9.2f}  "
            f"steps={traj['n_steps']}"
        )

    print()
    print("-" * 120)
    summarise(str(fixed),  traj_fixed)
    summarise(str(mc),     traj_mc)
    print("-" * 120)

    print("\nDone.")
    sys.exit(0)
