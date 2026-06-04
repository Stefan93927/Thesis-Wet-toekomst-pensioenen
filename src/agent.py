"""agent.py — LSTM + GMM Regime Gating + Risk Parity + PPO policy.

Architecture (end-to-end differentiable)
-----------------------------------------
1. LSTM Temporal Feature Extractor
   Input  : 12-month lookback window of 31 scaled features  (12 × 31)
   Output : hidden state h_t ∈ R^256
   Details: single-layer LSTM, LayerNorm on h_t, Dropout(0.3)

2. GMM Liability-Aware Regime Gating  (run_005: 2D bivariate GMM)
   Input  : [h_t, VSTOXX_t, RTS_Slope_30Y_10Y_t, FR_t]  (259-dim)
   Output : regime weights γ_t ∈ Δ^4  (softmax), risk budget β_t ∈ (0,1)
   Details: bivariate GMM fitted offline on [VSTOXX, RTS_Slope_30Y_10Y];
            K=4 regimes capturing both equity-vol and rate-environment state.
            β̄ = [0.70, 0.55, 0.40, 0.25] ordered by combined stress level.
            Gating also receives FR_t to couple Wtp regulatory thresholds.

3. Differentiable Risk-Budgeting Layer
   Input  : β_t, conditional volatilities (σ_eq, σ_bond) from covariance head
   Output : risk-parity equity weight w_eq* (closed form, natively differentiable)
   Formula: w_eq* = β_t * σ_bond / (β_t * σ_bond + (1−β_t) * σ_eq)
   Note   : exact closed-form solution for the 2-asset risk-parity problem;
            no cvxpylayers required.  Gradients flow back to LSTM via autograd.

4. PPO (Stable Baselines 3)
   - Custom ActorCriticPolicy wrapping WtpPolicyNetwork
   - Action mean: [e_t = clip(w_eq* − 0.55 + δe, −0.25, +0.25), f_t, d_t]
   - Learned diagonal action noise (log_std parameter)
   - Separate value head: Linear([h_t, FR, B]) → scalar

Usage
-----
    from src.data_pipeline import run_pipeline
    from src.environment   import make_env_from_pipeline
    from src.agent         import fit_gmm, make_agent, AgentConfig

    results = run_pipeline()
    env     = make_env_from_pipeline(results, split="train", seed=0)

    vstoxx  = results["z_train_raw"]["vstoxx_level"].values
    gmm     = fit_gmm(vstoxx)

    agent   = make_agent(env, gmm=gmm)
    agent.learn(total_timesteps=2_000_000)
    agent.save("src/models/ppo_wtp")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Type

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
except ImportError as exc:
    raise ImportError(
        "PyTorch is required.  Install with:  pip install torch"
    ) from exc

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.type_aliases import Schedule
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    import gymnasium as gym
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 and gymnasium are required.\n"
        "Install with:  pip install stable-baselines3 gymnasium"
    ) from exc

try:
    from sklearn.mixture import GaussianMixture
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required.  Install with:  pip install scikit-learn"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """All tunable constants for the DRL agent architecture and training."""

    # --- LSTM ------------------------------------------------------------ #
    lstm_hidden:   int   = 256
    lstm_dropout:  float = 0.3
    lookback:      int   = 12
    n_features:    int   = 31

    # --- GMM Regime Gating: 2D bivariate on [VSTOXX, RTS_Slope_30Y_10Y] --- #
    n_regimes:              int   = 3
    # Risk budget per regime: ordered Low -> Medium -> High VSTOXX stress
    # Regime 1: Low vol    (VSTOXX < 20)  -> accumulate buffer, equity room
    # Regime 2: Medium vol (VSTOXX 20-30) -> maintain strategic allocation
    # Regime 3: High vol   (VSTOXX >= 30) -> protect FR, suspend distributions
    # Values are logit(beta_target) so that sigmoid recovers the intended budgets:
    #   sigmoid(0.619)=0.65, sigmoid(0.201)=0.55, sigmoid(-0.619)=0.35
    beta_bar:               list  = field(default_factory=lambda: [0.619, 0.201, -0.619])
    vstoxx_feature_idx:     int   = 14   # index of vstoxx_level in z_t
    rts_slope_feature_idx:  int   = 13   # index of rts_slope_30y_10y in z_t

    # --- Covariance / Risk parity ---------------------------------------- #
    cov_eps: float = 1e-6   # numerical floor for softplus volatilities

    # --- Action heads (equity correction range) -------------------------- #
    equity_correction_scale: float = 0.05   # tanh correction around w_eq*
    w_eq_base:    float = 0.55
    w_eq_min:     float = 0.30
    w_eq_max:     float = 0.90
    eq_tilt_min:  float = -0.25   # run_007: widened from -0.10 to match CLAUDE.md
    eq_tilt_max:  float = +0.25   # run_007: widened from +0.10 to match CLAUDE.md
    fill_max:     float = 0.10
    dist_max:     float = 0.05

    # --- Observation layout ---------------------------------------------- #
    # Must be consistent with EnvConfig: 2 (FR, B) + n_extra_solvency (3) = 5
    n_solvency_vars: int = 5   # : FR, B, annual_o_plus, fill_used, month_norm

    # --- PPO hyperparameters --------------- #
    total_timesteps: int   = 2_000_000
    learning_rate:   float = 3e-4
    lr_warmup_steps: int   = 10_000   # ramp LR from 0 to learning_rate over this many steps
    max_grad_norm:   float = 0.5      # gradient clip norm (passed to PPO constructor)
    weight_decay:    float = 1e-4     # L2 regularisation on all policy parameters
    n_steps:         int   = 2_048
    batch_size:      int   = 64
    gamma:           float = 0.99
    clip_range:      float = 0.20
    gae_lambda:      float = 0.95
    ent_coef:        float = 0.05
    vf_coef:         float = 0.5
    # Per-dimension log_std bounds — sized to each action's range so that
    # σ_init ≈ 0.25 × range (healthy exploration) and σ_max ≤ 0.50 × range
    # (prevents Gaussian mass from leaking outside the clipping boundary).
    # Ranges: e_t [-0.25,+0.25]=0.50, f_t [0,0.10]=0.10, d_t [0,0.05]=0.05
    log_std_init:    tuple = (-2.08, -3.69, -4.38)   # ln(0.125), ln(0.025), ln(0.0125)
    log_std_max:     tuple = (-1.39, -3.00, -3.69)   # ln(0.25),  ln(0.05),  ln(0.025)
    log_std_min:     tuple = (-6.0,  -6.0,  -6.0)    # floor: prevents full determinism

    # --- Behavioral cloning warmstart ------------------------------------ #
    bc_enabled:         bool  = True    # activate BC warmstart in train.py
    bc_warmstart_steps: int   = 500_000 # total steps for BC annealing
    bc_initial_weight:  float = 1.0     # BC loss weight at step 0
    bc_final_weight:    float = 0.0     # BC loss weight at bc_warmstart_steps

    # --- Paths ----------------------------------------------------------- #
    model_dir: Path = field(default_factory=lambda: Path("src") / "models")

    # --- GMM fitting ----------------------------------------------------- #
    gmm_n_regimes: int = 3
    gmm_seed:      int = 42


# ---------------------------------------------------------------------------
# Module 2: GMM Regime Gating
# ---------------------------------------------------------------------------

class GMMRegimeGating(nn.Module):
    """Differentiable softmax gating over K liability-aware regimes.

    Converts [h_t, VSTOXX_t, RTS_Slope_t, FR_t] into a weighted combination
    of K per-regime risk budgets β̄_k, yielding a dynamic scalar risk budget β_t.

    The 2D regime structure (VSTOXX × RTS slope) captures both equity-market
    stress and the interest-rate environment, which together drive pension fund
    funding ratio dynamics.  FR_t couples the Wtp regulatory thresholds directly
    into the risk budget.

    Args:
        h_dim:     Dimension of the LSTM hidden state.
        n_regimes: Number of regimes K (default 3: Low/Medium/High VSTOXX).
        beta_bar:  Target equity risk budget per regime (length K).
    """

    def __init__(
        self,
        h_dim:     int,
        n_regimes: int,
        beta_bar:  list,
    ) -> None:
        super().__init__()
        # +3 for VSTOXX scalar, RTS slope scalar, FR scalar
        self.linear = nn.Linear(h_dim + 3, n_regimes)
        # run_008: learnable risk budgets — PPO adapts per-regime equity allocation
        # sigmoid in forward keeps each beta in (0,1); initialised from domain priors
        self.beta_bar = nn.Parameter(
            torch.tensor(beta_bar, dtype=torch.float32)
        )

    def forward(
        self,
        h_t:         Tensor,
        vstoxx_t:    Tensor,
        rts_slope_t: Tensor,
        fr_t:        Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute regime weights and dynamic risk budget.

        Args:
            h_t:          LSTM hidden state       (B, h_dim).
            vstoxx_t:     Scaled VSTOXX level     (B, 1).
            rts_slope_t:  Scaled RTS 30Y-10Y slope (B, 1).
            fr_t:         Funding ratio            (B, 1).

        Returns:
            ``(gamma_t, beta_t)`` where gamma_t has shape (B, K) and
            beta_t has shape (B, 1).
        """
        gating_in = torch.cat([h_t, vstoxx_t, rts_slope_t, fr_t], dim=1)  # (B, h_dim+3)
        gamma_t   = torch.softmax(self.linear(gating_in), dim=1)            # (B, K)
        beta_t    = (gamma_t * torch.sigmoid(self.beta_bar)).sum(dim=1, keepdim=True)  # (B, 1)
        return gamma_t, beta_t


# ---------------------------------------------------------------------------
# Module 1+3: LSTM + Covariance head + Risk Parity (differentiable)
# ---------------------------------------------------------------------------

class WtpPolicyNetwork(nn.Module):
    """Complete LSTM → GMM Gating → Risk Parity policy network.

    Processes the flat 374-dim observation, extracts h_t via LSTM, computes
    a differentiable risk-parity equity weight, and returns action means and
    state value for PPO.

    Args:
        cfg: :class:`AgentConfig`.
    """

    def __init__(self, cfg: AgentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.lstm_hidden

        # Module 1: LSTM temporal extractor
        self.lstm       = nn.LSTM(cfg.n_features, H, batch_first=True)
        self.layer_norm = nn.LayerNorm(H)
        self.dropout    = nn.Dropout(cfg.lstm_dropout)

        # Module 2: GMM regime gating
        self.gating = GMMRegimeGating(H, cfg.n_regimes, cfg.beta_bar)

        # Module 3: Conditional covariance head (2 asset volatilities)
        # Projects h_t → [log_var_eq, log_var_bond]; softplus ensures positivity
        self.cov_head = nn.Linear(H, 2)

        # Action head A: small equity correction around risk-parity mean
        self.equity_correction = nn.Linear(H, 1)

        # Action head B: fill rate (0 → fill_max)
        self.fill_head = nn.Linear(H, 1)

        # Action head C: distribution rate (0 → dist_max)
        self.dist_head = nn.Linear(H, 1)

        # Value head: takes [h_t, FR, B, annual_o_plus, fill_used, month_norm]
        self.value_head = nn.Linear(H + cfg.n_solvency_vars, 1)

    # ------------------------------------------------------------------ #
    # Forward helpers                                                     #
    # ------------------------------------------------------------------ #

    def _parse_obs(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split flat observation into solvency state and feature sequence.

        run_007 layout: [FR, B, annual_o_plus, fill_used, month_norm, z_flat]
        Solvency block size = cfg.n_solvency_vars (default 5).

        Args:
            obs: Flat observation (B, 377).

        Returns:
            ``(FR, B_buf, solvency, z_seq)`` with shapes
            (B,1), (B,1), (B, n_solvency_vars), (B,12,31).
        """
        n_sol = self.cfg.n_solvency_vars
        FR       = obs[:, 0:1]
        B_buf    = obs[:, 1:2]
        solvency = obs[:, :n_sol]
        z_seq    = obs[:, n_sol:].reshape(-1, self.cfg.lookback, self.cfg.n_features)
        return FR, B_buf, solvency, z_seq

    def _risk_parity_weight(
        self,
        beta_t:     Tensor,
        sigma_eq:   Tensor,
        sigma_bond: Tensor,
    ) -> Tensor:
        """Closed-form 2-asset risk parity equity weight (differentiable).

        Solves: risk_contribution(equity) = beta_t * total_portfolio_risk
        Solution: w_eq* = beta_t * sigma_bond
                          / (beta_t * sigma_bond + (1 - beta_t) * sigma_eq)

        Args:
            beta_t:     Target equity risk fraction (B, 1), clamped to (0,1).
            sigma_eq:   Equity volatility (B, 1), strictly positive.
            sigma_bond: Bond volatility   (B, 1), strictly positive.

        Returns:
            Risk-parity equity weight (B, 1), clipped to [w_eq_min, w_eq_max].
        """
        # Correct risk-contribution parity: w_eq s.t. w_eq*sigma_eq contributes
        # fraction beta of total risk.  Closed-form (uncorrelated assets):
        #   w_eq = sqrt(beta)*sigma_bond / (sqrt(beta)*sigma_bond + sqrt(1-beta)*sigma_eq)
        b  = beta_t.clamp(1e-4, 1.0 - 1e-4)
        w  = (torch.sqrt(b) * sigma_bond
              / (torch.sqrt(b) * sigma_bond + torch.sqrt(1.0 - b) * sigma_eq + 1e-8))
        return w.clamp(self.cfg.w_eq_min, self.cfg.w_eq_max)

    # ------------------------------------------------------------------ #
    # Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, obs: Tensor) -> dict:
        """Forward pass: observation → action means, value, and internals.

        Args:
            obs: Flat observation tensor (B, 374).

        Returns:
            Dictionary with keys:

            - ``action_mean``   : (B, 3) — [equity_tilt, fill, dist]
            - ``value``         : (B, 1) — state value estimate
            - ``w_eq_rp``       : (B, 1) — risk-parity equity weight
            - ``beta_t``        : (B, 1) — dynamic risk budget
            - ``gamma_t``       : (B, K) — regime weights
        """
        cfg = self.cfg

        # ---- Parse observation ---------------------------------------- #
        FR, B_buf, solvency, z_seq = self._parse_obs(obs)  # z_seq: (B, 12, 31)

        # ---- Module 1: LSTM ------------------------------------------- #
        _, (h_n, _) = self.lstm(z_seq)                # h_n: (1, B, H)
        h_t = self.layer_norm(h_n.squeeze(0))         # (B, H)
        h_t = self.dropout(h_t)

        # ---- Module 2: GMM regime gating ------------------------------ #
        # Extract most-recent scalars from the feature sequence
        vstoxx_t    = z_seq[:, -1, cfg.vstoxx_feature_idx    : cfg.vstoxx_feature_idx    + 1]
        rts_slope_t = z_seq[:, -1, cfg.rts_slope_feature_idx : cfg.rts_slope_feature_idx + 1]
        # FR_t is already available from _parse_obs; pass directly as regime input
        gamma_t, beta_t = self.gating(h_t, vstoxx_t, rts_slope_t, FR)  # (B,K), (B,1)

        # ---- Module 3: Risk parity ------------------------------------ #
        log_vars   = self.cov_head(h_t)                # (B, 2)
        sigmas     = F.softplus(log_vars) + cfg.cov_eps
        sigma_eq   = sigmas[:, 0:1]
        sigma_bond = sigmas[:, 1:2]

        w_eq_rp = self._risk_parity_weight(beta_t, sigma_eq, sigma_bond)  # (B,1)

        # ---- Action means --------------------------------------------- #
        # Equity tilt: risk-parity recommendation + small learnable correction.
        # run_036a: removed .clamp() — environment clips sampled actions to
        # [-0.25, +0.25] but the mean can float freely, so gradients flow even
        # when the risk-parity signal pushes toward a boundary.
        delta_e = torch.tanh(self.equity_correction(h_t)) * cfg.equity_correction_scale
        e_mean  = w_eq_rp - cfg.w_eq_base + delta_e                          # (B,1)

        # Fill rate: tanh rescaled to [0, fill_max].
        # run_036a: replaces sigmoid which saturated to 0 at init causing
        # zero-gradient lock.  tanh derivative (1-tanh²) > 0 everywhere.
        f_mean = 0.5 * (torch.tanh(self.fill_head(h_t)) + 1.0) * cfg.fill_max  # (B,1)

        # Distribution rate: tanh rescaled to [0, dist_max].
        # Same motivation as fill rate — prevents sigmoid saturation.
        d_mean = 0.5 * (torch.tanh(self.dist_head(h_t)) + 1.0) * cfg.dist_max  # (B,1)

        action_mean = torch.cat([e_mean, f_mean, d_mean], dim=1)            # (B,3)

        # ---- Value head (receives full solvency block incl. fill credit) #
        value = self.value_head(torch.cat([h_t, solvency], dim=1))          # (B,1)

        return {
            "action_mean": action_mean,
            "value":       value,
            "w_eq_rp":     w_eq_rp,
            "beta_t":      beta_t,
            "gamma_t":     gamma_t,
        }


# ---------------------------------------------------------------------------
# SB3 ActorCriticPolicy wrapper
# ---------------------------------------------------------------------------

class WtpActorCriticPolicy(ActorCriticPolicy):
    """Stable Baselines 3 ActorCriticPolicy wrapping WtpPolicyNetwork.

    Overrides only the four methods the PPO training loop calls:
    ``_build``, ``forward``, ``evaluate_actions``, and ``predict_values`` /
    ``get_distribution``.  Everything else (rollout collection, PPO update,
    logging) is handled by SB3.

    Pass ``wtp_cfg`` via ``policy_kwargs`` when constructing PPO::

        PPO(WtpActorCriticPolicy, env,
            policy_kwargs={"wtp_cfg": cfg}, ...)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space:      gym.Space,
        lr_schedule:       Schedule,
        wtp_cfg:           Optional[AgentConfig] = None,
        **kwargs,
    ) -> None:
        self.wtp_cfg = wtp_cfg or AgentConfig()
        # Pass minimal kwargs to parent; suppress default MLP construction
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Disable SB3's default feature extractor and MLP
            net_arch     = [],
            ortho_init   = False,
            **{k: v for k, v in kwargs.items()
               if k not in ("net_arch", "ortho_init")},
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """Replace SB3's default MLP with WtpPolicyNetwork."""
        cfg = self.wtp_cfg

        self.wtp_net = WtpPolicyNetwork(cfg)

        # Learned diagonal action noise: per-dimension, sized to each action range
        n_actions = self.action_space.shape[0]   # 3
        assert len(cfg.log_std_init) == n_actions, (
            f"log_std_init length {len(cfg.log_std_init)} != n_actions {n_actions}"
        )
        self.log_std = nn.Parameter(
            torch.tensor(list(cfg.log_std_init), dtype=torch.float32)
        )

        # Single optimizer over all parameters.
        # run_038: explicit weight_decay + eps to prevent weight explosion.
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr_schedule(1),
            eps=1e-5,
            weight_decay=cfg.weight_decay,
        )

    def _forward_policy(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Shared forward; returns (action_mean, value)."""
        out        = self.wtp_net(obs)
        return out["action_mean"], out["value"]

    def _make_distribution(self, mean_actions: Tensor) -> DiagGaussianDistribution:
        """SB3-compatible diagonal Gaussian from mean and per-dimension clamped log_std.

        run_032: per-dimension clamp prevents action-space saturation (sigma >> range)
        that caused ~98% of dist_rate samples to clip to zero in run_031.
        Each dimension's std is bounded to [exp(log_std_min_k), exp(log_std_max_k)].
        """
        lo = torch.tensor(list(self.wtp_cfg.log_std_min), dtype=self.log_std.dtype,
                          device=self.log_std.device)
        hi = torch.tensor(list(self.wtp_cfg.log_std_max), dtype=self.log_std.dtype,
                          device=self.log_std.device)
        log_std_clamped = torch.max(torch.min(self.log_std, hi), lo)
        dist = DiagGaussianDistribution(action_dim=self.action_space.shape[0])
        dist.proba_distribution(mean_actions, log_std_clamped)
        return dist

    # ------------------------------------------------------------------ #
    # SB3 interface                                                       #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        obs:           Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Called during rollout collection."""
        mean_actions, values = self._forward_policy(obs)
        dist      = self._make_distribution(mean_actions)
        actions   = dist.get_actions(deterministic=deterministic)
        log_prob  = dist.log_prob(actions)
        return actions, values.squeeze(-1), log_prob

    def evaluate_actions(
        self,
        obs:     Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Called during the PPO update step."""
        mean_actions, values = self._forward_policy(obs)
        dist     = self._make_distribution(mean_actions)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return values.squeeze(-1), log_prob, entropy

    def predict_values(self, obs: Tensor) -> Tensor:
        """Called for value bootstrapping at end of rollout."""
        _, values = self._forward_policy(obs)
        return values.squeeze(-1)

    def get_distribution(self, obs: Tensor) -> DiagGaussianDistribution:
        """Called by SB3's predict() during evaluation."""
        mean_actions, _ = self._forward_policy(obs)
        return self._make_distribution(mean_actions)


# ---------------------------------------------------------------------------
# Offline GMM fitting
# ---------------------------------------------------------------------------

def fit_gmm(
    vstoxx_train:    np.ndarray,
    rts_slope_train: np.ndarray,
    n_regimes:       int = 4,
    seed:            int = 42,
) -> GaussianMixture:
    """Fit a bivariate GMM on [VSTOXX, RTS_Slope_30Y_10Y] from training data.

    The GMM is used OFFLINE only — to confirm the number of regimes and
    validate the β̄ assignments.  The online gating network is trained
    end-to-end and does not call the GMM at runtime.

    The 2D feature space captures four distinct pension fund environments:
      - Low vol + steep curve  : expansion / reflation  (most equity room)
      - Low vol + flat curve   : late cycle / rate risk  (cautious)
      - High vol + steep curve : equity crash, rates intact
      - High vol + flat curve  : combined stress (e.g. 2020 COVID)

    Args:
        vstoxx_train:    1-D array of monthly VSTOXX values (training period).
        rts_slope_train: 1-D array of monthly RTS 30Y-10Y slope (training period).
        n_regimes:       Number of mixture components K (default 4).
        seed:            Random seed for reproducibility.

    Returns:
        Fitted :class:`~sklearn.mixture.GaussianMixture`.
    """
    X = np.column_stack([vstoxx_train, rts_slope_train])   # (T, 2)
    gmm = GaussianMixture(
        n_components    = n_regimes,
        covariance_type = "full",
        random_state    = seed,
        n_init          = 10,
    )
    gmm.fit(X)

    labels      = gmm.predict(X)
    vstoxx_med  = np.median(vstoxx_train)
    slope_med   = np.median(rts_slope_train)

    print(f"  GMM regimes (K={n_regimes}) fitted on {len(X)} training months "
          f"[VSTOXX, RTS_Slope_30Y_10Y]:")
    print(f"  (medians: VSTOXX={vstoxx_med:.1f}, RTS slope={slope_med:.3f})")

    beta_bar_target = [0.70, 0.55, 0.40, 0.25]
    # Sort regimes by VSTOXX mean for interpretable display
    order = np.argsort(gmm.means_[:, 0])
    for rank, k in enumerate(order):
        mask       = labels == k
        v_mean     = gmm.means_[k, 0]
        s_mean     = gmm.means_[k, 1]
        v_label    = "Low " if v_mean < vstoxx_med else "High"
        s_label    = "Steep" if s_mean > slope_med  else "Flat "
        beta_rank  = rank if rank < len(beta_bar_target) else -1
        print(
            f"    Regime {rank+1} [VSTOXX {v_label} / Slope {s_label}]  "
            f"VSTOXX={v_mean:.1f}  slope={s_mean:.3f}  "
            f"n={mask.sum()}  beta_bar={beta_bar_target[beta_rank]}"
        )
    return gmm


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_agent(
    env,
    cfg:  Optional[AgentConfig] = None,
    gmm:  Optional[GaussianMixture] = None,
    seed: int = 42,
) -> PPO:
    """Create a Stable Baselines 3 PPO agent with the Wtp custom policy.

    Args:
        env:  A :class:`~src.environment.WtpPensionEnv` instance.
        cfg:  Optional :class:`AgentConfig`.
        gmm:  Optional fitted :class:`~sklearn.mixture.GaussianMixture`
              (informational only; gating is trained end-to-end).
        seed: Random seed for SB3 reproducibility.

    Returns:
        Configured :class:`~stable_baselines3.PPO` instance.
    """
    cfg = cfg or AgentConfig()

    agent = PPO(
        policy          = WtpActorCriticPolicy,
        env             = env,
        learning_rate   = cfg.learning_rate,
        n_steps         = cfg.n_steps,
        batch_size      = cfg.batch_size,
        gamma           = cfg.gamma,
        clip_range      = cfg.clip_range,
        gae_lambda      = cfg.gae_lambda,
        ent_coef        = cfg.ent_coef,
        vf_coef         = cfg.vf_coef,
        max_grad_norm   = cfg.max_grad_norm,
        verbose         = 1,
        seed            = seed,
        policy_kwargs   = {"wtp_cfg": cfg},
    )
    return agent


# ---------------------------------------------------------------------------
# Parameter count utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main — architecture summary and forward-pass sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data_pipeline import run_pipeline
    from src.environment   import make_env_from_pipeline

    print("=" * 64)
    print("Wtp DRL Pension Fund -- Agent Architecture")
    print("=" * 64)

    torch.manual_seed(42)
    np.random.seed(42)
    cfg = AgentConfig()

    # ---- Load data ------------------------------------------------------- #
    print("\n[1/4] Loading pipeline data...")
    results = run_pipeline()

    # ---- Fit bivariate GMM on training [VSTOXX, RTS slope] -------------- #
    print("\n[2/4] Fitting bivariate GMM on training [VSTOXX, RTS_Slope_30Y_10Y]...")
    vstoxx_train    = results["z_train_raw"]["vstoxx_level"].values
    rts_slope_train = results["z_train_raw"]["rts_slope_30y_10y"].values
    gmm = fit_gmm(
        vstoxx_train    = vstoxx_train,
        rts_slope_train = rts_slope_train,
        n_regimes       = cfg.gmm_n_regimes,
        seed            = cfg.gmm_seed,
    )

    # ---- Build agent ----------------------------------------------------- #
    print("\n[3/4] Building PPO agent with custom WtpActorCriticPolicy...")
    env   = make_env_from_pipeline(results, split="train", seed=0)
    agent = make_agent(env, cfg=cfg, gmm=gmm, seed=42)

    policy = agent.policy
    net    = policy.wtp_net

    total = count_parameters(policy)
    print(f"\n  Policy architecture:")
    print(f"    LSTM          : {cfg.n_features}-dim input, {cfg.lstm_hidden}-dim hidden")
    print(f"    GMM gating    : {cfg.lstm_hidden}+3 -> {cfg.n_regimes} regimes  [h_t, VSTOXX, RTS_slope, FR]")
    print(f"    Cov head      : {cfg.lstm_hidden} -> 2 volatilities (softplus)")
    print(f"    Equity head   : {cfg.lstm_hidden} -> 1 (risk parity + correction)")
    print(f"    Fill head     : {cfg.lstm_hidden} -> 1 (sigmoid * {cfg.fill_max})")
    print(f"    Dist head     : {cfg.lstm_hidden} -> 1 (sigmoid * {cfg.dist_max})")
    print(f"    Value head    : {cfg.lstm_hidden}+2 -> 1")
    print(f"    Log-std param : (3,)  init={list(cfg.log_std_init)}")
    print(f"  Total trainable parameters: {total:,}")

    # ---- Forward pass sanity check --------------------------------------- #
    print("\n[4/4] Forward pass sanity check (batch of 4 obs)...")
    obs_np   = np.stack([env.reset()[0] for _ in range(4)])
    obs_t    = torch.as_tensor(obs_np, dtype=torch.float32)

    net.eval()
    with torch.no_grad():
        out = net(obs_t)

    am    = out["action_mean"]
    val   = out["value"]
    w_rp  = out["w_eq_rp"]
    beta  = out["beta_t"]
    gamma = out["gamma_t"]

    print(f"  action_mean shape : {tuple(am.shape)}   (expected (4, 3))")
    print(f"  value shape       : {tuple(val.shape)}  (expected (4, 1))")
    print(f"  action_mean sample (first obs):")
    print(f"    e_t (equity tilt)  = {am[0,0].item():+.4f}  "
          f"-> w_eq = {0.55 + am[0,0].item():.4f}")
    print(f"    f_t (fill rate)    = {am[0,1].item():.4f}")
    print(f"    d_t (dist rate)    = {am[0,2].item():.4f}")
    print(f"  Risk parity weight  = {w_rp[0,0].item():.4f}")
    print(f"  Dynamic risk budget = {beta[0,0].item():.4f}  "
          f"(regime weights: {gamma[0].tolist()})")
    print(f"  Value estimate      = {val[0,0].item():.4f}")

    # Verify action range constraints
    e_ok  = bool((am[:, 0].abs() <= 0.10 + 1e-4).all())
    f_ok  = bool(((am[:, 1] >= 0) & (am[:, 1] <= cfg.fill_max + 1e-4)).all())
    d_ok  = bool(((am[:, 2] >= 0) & (am[:, 2] <= cfg.dist_max + 1e-4)).all())
    print(f"\n  Action range checks:")
    print(f"    e_t in [-0.25, +0.25]: {'OK' if e_ok else 'FAIL'}")
    print(f"    f_t in [0, {cfg.fill_max}]    : {'OK' if f_ok else 'FAIL'}")
    print(f"    d_t in [0, {cfg.dist_max}]    : {'OK' if d_ok else 'FAIL'}")

    # Quick rollout test (evaluate_actions)
    print("\n  evaluate_actions check (2 obs)...")
    obs2  = torch.as_tensor(obs_np[:2], dtype=torch.float32)
    act2  = torch.zeros(2, 3, dtype=torch.float32)
    policy.set_training_mode(True)
    vals2, lp2, ent2 = policy.evaluate_actions(obs2, act2)
    print(f"    values    : {vals2.tolist()}")
    print(f"    log_probs : {lp2.tolist()}")
    print(f"    entropy   : {ent2.tolist()}")

    print("\nDone.")
    sys.exit(0)
