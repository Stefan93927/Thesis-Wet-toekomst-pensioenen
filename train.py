"""train.py — Main training script for the Wtp DRL pension fund agent.

Pipeline
--------
1. Set global random seeds (numpy, torch, gymnasium).
2. Run data pipeline (loads, cleans, features, splits).
3. Fit GMM offline on training VSTOXX.
4. Build training and validation Gymnasium environments.
5. Instantiate SB3 PPO agent with WtpActorCriticPolicy.
6. Train with two callbacks:
   - WtpValidationCallback : runs a full validation episode every
     ``--eval-freq`` timesteps; saves the best checkpoint (by val reward).
   - CheckpointCallback    : saves periodic checkpoints to disk.
7. Save the final model.
8. Print a brief training summary.

Usage (terminal / VS Code terminal)
------------------------------------
    # Default: 2M steps, seed 42, logs to src/models/run_001/
    py -3 train.py

    # Custom
    py -3 train.py --timesteps 500000 --seed 0 --log-dir src/models/run_002

    # Quick smoke test (~5 episodes)
    py -3 train.py --timesteps 1000 --eval-freq 500

Usage (Jupyter notebook)
-------------------------
    %run train.py --timesteps 500000

Notes
-----
- Validation-period metrics are NEVER reported in final results (only used
  for checkpoint selection and early-convergence monitoring).
- The best model (by cumulative val reward) is saved to
  ``<log_dir>/best_model.zip``.
- Training logs (val metrics history) are saved to
  ``<log_dir>/val_history.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import io
import zipfile
import numpy as np

# ---- project imports -------------------------------------------------------
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.data_pipeline import run_pipeline
from src.environment   import make_env_from_pipeline, EnvConfig
from src.agent         import fit_gmm, make_agent, AgentConfig, WtpActorCriticPolicy
from src.baselines     import run_episode, DemonstrationPolicy
from src.metrics       import compute_metrics

try:
    import torch
    import torch.nn.functional as F
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
    )
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
except ImportError as exc:
    raise ImportError(
        "torch and stable-baselines3 are required.\n"
        "Install with:  pip install torch stable-baselines3"
    ) from exc


# ---------------------------------------------------------------------------
# Minimal flat-YAML config loader (no external dependency)
# ---------------------------------------------------------------------------

def _parse_simple_yaml(path: str) -> dict:
    """Parse a flat key: value YAML file (no nesting, no anchors required).

    Supports scalar types: bool, int, float, str.  Lines that start with
    ``section_name:`` (value is empty) are treated as section headers and
    skipped so the file can have optional section labels for readability.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary mapping arg dest names to Python values.
    """
    result: dict = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#")[0].strip()          # strip comments
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            if val == "":                             # section header — skip
                continue
            # Type coercion
            if val.lower() == "true":
                result[key] = True
            elif val.lower() == "false":
                result[key] = False
            elif val.lower() in ("null", "none", "~"):
                result[key] = None
            else:
                try:
                    result[key] = int(val)
                except ValueError:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val
    return result


# ---------------------------------------------------------------------------
# Fine-tune loader: load policy weights only, fresh optimizer
# ---------------------------------------------------------------------------

def load_for_finetune(ft_path: Path, train_env, agent_cfg: AgentConfig, seed: int):
    """Load policy weights from a saved zip into a fresh PPO agent.

    SB3's PPO.load() fails when the saved optimizer has a different parameter
    group size (e.g. K=4 vs K=3 regimes).  This helper reads the policy weights
    directly from the zip and loads them into a freshly initialised agent, giving
    a warm-started policy with a fresh Adam optimiser — ideal for fine-tuning.

    Args:
        ft_path:    Path to the saved ``.zip`` model.
        train_env:  Training VecEnv (passed to the new PPO).
        agent_cfg:  AgentConfig for the NEW agent; must have matching architecture
                    (n_regimes, lstm_hidden, etc.) to the saved weights.
        seed:       Random seed.

    Returns:
        :class:`~stable_baselines3.PPO` with weights loaded, fresh optimiser.
    """
    import torch
    from stable_baselines3 import PPO as _PPO

    zip_path = str(ft_path)
    if not zip_path.endswith(".zip"):
        zip_path += ".zip"

    # Read the policy state dict directly from the zip
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("policy.pth") as f:
            policy_state = torch.load(io.BytesIO(f.read()), map_location="cpu",
                                      weights_only=False)

    # Build a fresh agent with the matching architecture
    fresh = _PPO(
        policy        = WtpActorCriticPolicy,
        env           = train_env,
        learning_rate = agent_cfg.learning_rate,
        n_steps       = agent_cfg.n_steps,
        batch_size    = agent_cfg.batch_size,
        gamma         = agent_cfg.gamma,
        clip_range    = agent_cfg.clip_range,
        gae_lambda    = agent_cfg.gae_lambda,
        ent_coef      = agent_cfg.ent_coef,
        vf_coef       = agent_cfg.vf_coef,
        verbose       = 1,
        seed          = seed,
        policy_kwargs = {"wtp_cfg": agent_cfg},
    )

    # Inject the saved weights (policy only — fresh optimiser)
    missing, unexpected = fresh.policy.load_state_dict(policy_state, strict=False)
    if missing:
        print(f"  WARNING: missing keys in loaded state_dict: {missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys in loaded state_dict: {unexpected}")
    print(f"  Policy weights loaded from {ft_path} (fresh optimiser)")
    return fresh


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set global random seeds for numpy, torch, and Python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# SB3 adapter so run_episode() can call agent.predict()
# ---------------------------------------------------------------------------

class _SB3Adapter:
    """Thin wrapper so SB3 model.predict() matches run_episode's interface."""

    def __init__(self, model) -> None:
        self._model = model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action


# ---------------------------------------------------------------------------
# Validation callback
# ---------------------------------------------------------------------------

class WtpValidationCallback(BaseCallback):
    """Run a full validation episode at regular intervals during training.

    Saves the model whenever the cumulative validation reward improves.
    Writes a JSON log of all validation checkpoints to ``log_dir``.

    NOTE: Validation metrics are only used for model selection — they are
    NEVER reported in the final evaluation (test period only).

    Args:
        val_env:    Validation :class:`~src.environment.WtpPensionEnv`.
        pi_val:     Monthly CPI inflation aligned to validation dates ``(T,)``.
        eval_freq:  Run validation every this many PPO timesteps.
        log_dir:    Directory to save best model and history JSON.
        verbose:    Print validation results to stdout (0=silent, 1=print).
    """

    def __init__(
        self,
        val_env,
        pi_val:    np.ndarray,
        eval_freq: int,
        log_dir:   Path,
        verbose:   int = 1,
    ) -> None:
        super().__init__(verbose)
        self.val_env   = val_env
        self.pi_val    = pi_val
        self.eval_freq = eval_freq
        self.log_dir   = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.best_reward: float = -np.inf
        self.best_score:  float = -np.inf   # dist × (1 - dep_freq)
        self.history:     list  = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Run one deterministic val episode
        adapter  = _SB3Adapter(self.model)
        traj     = run_episode(adapter, self.val_env)
        metrics  = compute_metrics(traj, pi_monthly=self.pi_val)

        # Composite score: rewards distributing while penalising depletion.
        # score = dist × (1 - dep_freq) — picks the balanced model, not
        # the aggressive-dist (dep≈1) or collapsed-d=0 (dep≈0, dist≈0) extremes.
        score = (metrics["total_distributions"]
                 * (1.0 - metrics["buffer_depletion_freq"]))

        entry = {
            "timestep":    self.num_timesteps,
            "val_reward":  traj["total_reward"],
            "val_score":   score,
            **metrics,
        }
        self.history.append(entry)

        if self.verbose >= 1:
            print(
                f"  [val] t={self.num_timesteps:>9,d} | "
                f"reward={traj['total_reward']:8.2f} | "
                f"FR_term={metrics['fr_terminal']:.4f} | "
                f"MDD={metrics['fr_mdd']:.4f} | "
                f"dep={metrics['buffer_depletion_freq']:.3f} | "
                f"dist={metrics['total_distributions']:.4f} | "
                f"score={score:.4f}"
            )

        # Save best checkpoint by composite score (not raw reward)
        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.log_dir / "best_model"))
            if self.verbose >= 1:
                print(f"    -> New best model saved (score={self.best_score:.4f})")

        return True

    def _on_training_end(self) -> None:
        history_path = self.log_dir / "val_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        if self.verbose >= 1:
            print(f"\n  Validation history saved: {history_path}")


# ---------------------------------------------------------------------------
# Gamma curriculum callback
# ---------------------------------------------------------------------------

class GammaCurriculumCallback(BaseCallback):
    """Linearly anneal the buffer-depletion penalty (gamma) from 0 to target_gamma.

    During the first ``curriculum_steps`` timesteps gamma rises from 0 to
    ``target_gamma``.  After that it stays fixed.  This prevents the agent from
    collapsing to d=0 early: it first learns to distribute freely (gamma≈0),
    then gradually learns when depletion risk is too high to justify distributions.

    Args:
        target_gamma:      Final gamma value (e.g. 6.0).
        curriculum_steps:  Number of timesteps over which gamma is annealed.
        verbose:           Print gamma value every 50 k steps.
    """

    def __init__(self, target_gamma: float, curriculum_steps: int, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.target_gamma     = target_gamma
        self.curriculum_steps = curriculum_steps
        self._last_log        = -1

    def _on_step(self) -> bool:
        t = self.num_timesteps
        frac     = min(1.0, t / max(1, self.curriculum_steps))
        gamma_t  = self.target_gamma * frac

        # Push updated gamma into every sub-environment
        self.training_env.env_method("set_gamma", gamma_t)

        log_every = 50_000
        if self.verbose >= 1 and (t // log_every) != (self._last_log // log_every):
            print(f"  [curriculum] t={t:>9,d} | gamma={gamma_t:.3f}/{self.target_gamma:.1f}")
            self._last_log = t

        return True


# ---------------------------------------------------------------------------
# Weight norm monitoring callback
# ---------------------------------------------------------------------------

class WeightMonitorCallback(BaseCallback):
    """Log LSTM and LayerNorm weight norms each rollout; warn on explosion.

    Detects weight explosion early (LayerNorm norm > 10.0 indicates the
    normalisation layer has become an amplifier, which caused saturation
    in run_035-037).

    Args:
        log_freq: Print norms every this many timesteps.
        warn_threshold: Print a WARNING when LayerNorm weight norm exceeds this.
    """

    def __init__(self, log_freq: int = 50_000, warn_threshold: float = 10.0,
                 verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_freq       = log_freq
        self.warn_threshold = warn_threshold
        self._last_log      = -1

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if (t // self.log_freq) == (self._last_log // self.log_freq):
            return True
        self._last_log = t

        net = self.model.policy.wtp_net
        with torch.no_grad():
            lstm_norm = sum(p.norm().item() for p in net.lstm.parameters())
            ln_w_norm = net.layer_norm.weight.norm().item()
            ln_b_norm = net.layer_norm.bias.norm().item()
            vh_norm   = net.value_head.weight.norm().item()

        self.logger.record("weights/lstm_total_norm",    lstm_norm)
        self.logger.record("weights/layer_norm_weight",  ln_w_norm)
        self.logger.record("weights/layer_norm_bias",    ln_b_norm)
        self.logger.record("weights/value_head_norm",    vh_norm)

        if self.verbose >= 1 and ln_w_norm > self.warn_threshold:
            print(f"  [weights] t={t:>9,d} | lstm={lstm_norm:.1f} "
                  f"ln_w={ln_w_norm:.2f} ln_b={ln_b_norm:.2f} vh={vh_norm:.2f}"
                  f"  ** WARNING: LayerNorm exploding **")
        elif self.verbose >= 1:
            print(f"  [weights] t={t:>9,d} | lstm={lstm_norm:.1f} "
                  f"ln_w={ln_w_norm:.2f} ln_b={ln_b_norm:.2f} vh={vh_norm:.2f}")

        return True


# ---------------------------------------------------------------------------
# Behavioral cloning warmstart callback
# ---------------------------------------------------------------------------

class BCWarmstartCallback(BaseCallback):
    """Add a BC MSE gradient step after each PPO rollout.

    Linearly anneals the BC loss weight from ``bc_initial_weight`` to
    ``bc_final_weight`` over ``bc_warmstart_steps`` timesteps.  After that
    the callback is a no-op (weight=0).

    The BC gradient step uses the same optimizer as PPO so the policy
    weights are shared.  Gradients are clipped to max_norm=0.5.

    Args:
        demo_obs:             (N, obs_dim) float32 array of demo observations.
        demo_actions:         (N, act_dim) float32 array of demo actions.
        bc_initial_weight:    Starting BC loss coefficient.
        bc_final_weight:      Ending BC loss coefficient (0 = pure PPO).
        bc_warmstart_steps:   Total timesteps over which BC is active.
        log_freq:             Print BC diagnostic every this many timesteps.
        verbose:              0=silent, 1=print at log_freq.
    """

    def __init__(
        self,
        demo_obs:           np.ndarray,
        demo_actions:       np.ndarray,
        bc_initial_weight:  float = 1.0,
        bc_final_weight:    float = 0.0,
        bc_warmstart_steps: int   = 500_000,
        log_freq:           int   = 50_000,
        verbose:            int   = 1,
    ) -> None:
        super().__init__(verbose)
        self._demo_obs     = torch.as_tensor(demo_obs,    dtype=torch.float32)
        self._demo_acts    = torch.as_tensor(demo_actions, dtype=torch.float32)
        self.bc_initial_weight  = bc_initial_weight
        self.bc_final_weight    = bc_final_weight
        self.bc_warmstart_steps = bc_warmstart_steps
        self.log_freq           = log_freq
        self._last_log          = -1

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t >= self.bc_warmstart_steps:
            return True

        # Only fire once per PPO rollout (every n_steps env steps per env)
        # n_calls is incremented every step; n_steps is per-env rollout length
        n_steps = self.model.n_steps
        if self.n_calls % n_steps != 0:
            return True

        # Linear weight annealing
        frac      = max(0.0, 1.0 - t / self.bc_warmstart_steps)
        bc_weight = (self.bc_initial_weight * frac
                     + self.bc_final_weight * (1.0 - frac))
        if bc_weight <= 0.0:
            return True

        device = self.model.device
        demo_obs  = self._demo_obs.to(device)
        demo_acts = self._demo_acts.to(device)

        # Sample a minibatch from the demo buffer
        n = demo_obs.shape[0]
        idx = np.random.choice(n, size=min(64, n), replace=False)

        policy = self.model.policy
        policy.set_training_mode(True)

        mean_actions, _ = policy._forward_policy(demo_obs[idx])
        bc_loss = F.mse_loss(mean_actions, demo_acts[idx])
        total   = bc_weight * bc_loss

        policy.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        policy.optimizer.step()

        if (self.verbose >= 1
                and (t // self.log_freq) != (self._last_log // self.log_freq)):
            print(f"  [BC] t={t:>9,d} | weight={bc_weight:.3f} | "
                  f"bc_loss={bc_loss.item():.5f}")
            self._last_log = t

        return True


# ---------------------------------------------------------------------------
# Demonstration data generation
# ---------------------------------------------------------------------------

def generate_demonstration_data(
    results:    dict,
    env_cfg:    EnvConfig,
    seed:       int,
    n_episodes: int = 10,
) -> tuple:
    """Collect (obs, action) pairs from the DemonstrationPolicy.

    Runs ``n_episodes`` full training episodes using the hand-designed rule
    policy and returns stacked observation and action arrays suitable for
    the BCWarmstartCallback.

    Args:
        results:    Pipeline output dict from ``run_pipeline()``.
        env_cfg:    Environment config (same as training env).
        seed:       Random seed for environment reset.
        n_episodes: Number of full episodes to collect.

    Returns:
        ``(demo_obs, demo_actions)`` as float32 numpy arrays of shapes
        ``(N, obs_dim)`` and ``(N, act_dim)``.
    """
    demo_policy = DemonstrationPolicy()
    obs_list:  list = []
    act_list:  list = []

    for ep in range(n_episodes):
        env = make_env_from_pipeline(results, split="train", cfg=env_cfg,
                                     seed=seed + ep)
        obs, _ = env.reset()
        demo_policy.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action = demo_policy.predict(obs)
            obs_list.append(obs.copy())
            act_list.append(action.copy())
            obs, _, terminated, truncated, _ = env.step(action)

        env.close()

    demo_obs     = np.array(obs_list,  dtype=np.float32)
    demo_actions = np.array(act_list,  dtype=np.float32)
    print(f"  Generated {len(demo_obs):,} demo transitions "
          f"from {n_episodes} episodes  "
          f"(obs={demo_obs.shape}, acts={demo_actions.shape})")
    return demo_obs, demo_actions


# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------

def _print_summary(
    log_dir:     Path,
    elapsed:     float,
    timesteps:   int,
    best_reward: float,
) -> None:
    print("\n" + "=" * 64)
    print("Training complete")
    print("=" * 64)
    print(f"  Timesteps      : {timesteps:,}")
    print(f"  Elapsed        : {elapsed/60:.1f} min")
    print(f"  Best val reward: {best_reward:.2f}")
    print(f"  Log directory  : {log_dir}")
    print(f"  Files saved    :")
    for f in sorted(log_dir.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(log_dir)}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train Wtp DRL pension fund agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Optional YAML config file — values override argparse defaults but
    # can still be overridden by explicit CLI flags.
    p.add_argument("--config", type=str, default=None,
                   help="Path to a flat-key YAML config file (optional; "
                        "keys must match CLI arg dest names)")
    # Defaults are set to the best validated configuration found in run_028.
    # Changing any of these requires a conscious override via CLI flag.
    # run_028 key result: dep=0.000, dist=0.031, Calmar=0.613, FR_term=1.25 (val).
    p.add_argument("--timesteps",       type=int,   default=2_000_000,
                   help="Total PPO training timesteps")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--eval-freq",       type=int,   default=50_000,
                   help="Validation episode frequency (timesteps)")
    p.add_argument("--checkpoint-freq", type=int,   default=102_400,
                   help="Checkpoint save frequency (timesteps)")
    p.add_argument("--log-dir",         type=str,   default="src/models/run_007",
                   help="Directory for checkpoints and logs")
    p.add_argument("--lr",              type=float, default=3e-4,
                   help="PPO learning rate")
    p.add_argument("--n-steps",         type=int,   default=2_048,
                   help="PPO rollout length per update")
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--ent-coef",        type=float, default=0.05,
                   help="PPO entropy coefficient")
    p.add_argument("--n-envs",          type=int,   default=4,
                   help="Number of parallel training environments")
    p.add_argument("--dist-weight",     type=float, default=5.0,
                   help="Distribution incentive weight in Q_t = dist_weight*d_util - E_t"
                        " (run_028: 5.0)")
    p.add_argument("--log-dist",        action="store_true", default=True,
                   help="Use log1p(d/scale) utility instead of linear d in Q_t"
                        " (run_028: True; disable with --no-log-dist)")
    p.add_argument("--no-log-dist",     dest="log_dist", action="store_false",
                   help="Disable log1p utility (fall back to linear d in Q_t)")
    p.add_argument("--log-dist-scale",  type=float, default=0.005,
                   help="Scale for log utility: log1p(d / scale) (run_028: 0.005)")
    p.add_argument("--n-regimes",       type=int,   default=3,
                   help="Number of GMM regimes K (run_007 used K=4)")
    p.add_argument("--lambda-smooth",   type=float, default=0.0,
                   help="Smoothness penalty weight (0 to disable; run_028: 0.0)")
    p.add_argument("--zeta",            type=float, default=1.0,
                   help="FR change penalty weight in S_t (run_028: 1.0)")
    p.add_argument("--epsilon-equity",  type=float, default=0.3,
                   help="Separate cohort-variance penalty weight (run_028: 0.3)")
    p.add_argument("--gamma-depletion", type=float, default=8.0,
                   help="Buffer depletion penalty weight in reward (run_028: 8.0)")
    p.add_argument("--lifecycle",       action="store_true", default=True,
                   help="Enable per-cohort lifecycle equity weights (Wtp SPR personal share)"
                        " (run_028: True; disable with --no-lifecycle)")
    p.add_argument("--no-lifecycle",    dest="lifecycle", action="store_false",
                   help="Disable lifecycle weights (legacy aggregate mode)")
    p.add_argument("--curriculum-gamma", action="store_true", default=False,
                   help="Anneal gamma from 0 to --gamma-depletion over --curriculum-steps")
    p.add_argument("--curriculum-steps", type=int,  default=500_000,
                   help="Timesteps over which gamma is annealed (default 500k)")
    p.add_argument("--vf-coef",         type=float, default=0.5,
                   help="PPO value function loss coefficient (default 0.5; lower=slower VF fit)")
    p.add_argument("--norm-reward",     action="store_true", default=False,
                   help="Wrap training env with VecNormalize (reward normalisation only)")
    p.add_argument("--no-progress-bar", action="store_true",
                   help="Disable tqdm progress bar")
    p.add_argument("--finetune-from",   type=str,   default=None,
                   help="Path to a saved model zip to fine-tune from (e.g. src/models/run_007/best_model.zip)")
    # Reward weight overrides (env_cfg fields not previously exposed as CLI args)
    p.add_argument("--alpha",           type=float, default=None,
                   help="Stability term weight (EnvConfig.alpha; default from EnvConfig)")
    p.add_argument("--beta",            type=float, default=None,
                   help="Distribution incentive weight (EnvConfig.beta; default from EnvConfig)")
    p.add_argument("--delta",           type=float, default=None,
                   help="MVEV insolvency penalty weight (EnvConfig.delta; default from EnvConfig)")
    p.add_argument("--fill-bonus",      type=float, default=None,
                   help="Direct fill reward coefficient (EnvConfig.fill_bonus; run_037: 3.0)")
    p.add_argument("--max-grad-norm",   type=float, default=0.5,
                   help="Gradient clip norm for PPO updates (default 0.5)")
    p.add_argument("--weight-decay",    type=float, default=1e-4,
                   help="Adam weight decay / L2 regularisation (default 1e-4)")
    p.add_argument("--lr-warmup-steps", type=int,   default=10_000,
                   help="Ramp LR from 0 to --lr over this many timesteps (default 10k)")
    # Behavioral cloning warmstart
    p.add_argument("--bc-warmstart",    action="store_true", default=False,
                   help="Enable behavioral cloning warmstart (DemonstrationPolicy)")
    p.add_argument("--bc-warmstart-steps", type=int, default=500_000,
                   help="Timesteps over which BC weight is annealed 1.0->0.0")
    p.add_argument("--bc-initial-weight",  type=float, default=1.0,
                   help="Starting BC loss coefficient")
    p.add_argument("--bc-n-demos",      type=int,   default=10,
                   help="Number of demonstration episodes to generate for BC")
    p.add_argument("--tc-bps",          type=float, default=0.0,
                   help="One-way equity turnover transaction cost in basis points "
                        "(deducted from r_p each step; 0=disabled)")

    # --- Apply YAML config overrides BEFORE final parse ------------------- #
    # Pre-parse just --config to detect the file, then set_defaults.
    pre, _ = p.parse_known_args(argv)
    if pre.config is not None:
        yaml_overrides = _parse_simple_yaml(pre.config)
        # Remap any YAML keys that differ from argparse dest names
        _key_map = {"gamma": "gamma_depletion", "learning_rate": "lr"}
        remapped = {_key_map.get(k, k): v for k, v in yaml_overrides.items()}
        p.set_defaults(**{k: v for k, v in remapped.items()
                          if k != "config"})
        print(f"  [config] Loaded overrides from {pre.config}: "
              f"{list(remapped.keys())}")

    return p.parse_args(argv)


def train(argv=None) -> None:
    """Entry point: parse args, run full training pipeline."""
    args    = parse_args(argv)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- Reproducibility ------------------------------------------------- #
    set_seeds(args.seed)

    print("=" * 64)
    print("Wtp DRL Pension Fund -- Training")
    print("=" * 64)
    print(f"  Timesteps    : {args.timesteps:,}")
    print(f"  Seed         : {args.seed}")
    print(f"  Eval freq    : {args.eval_freq:,} steps")
    print(f"  N envs       : {args.n_envs}")
    print(f"  Log dir      : {log_dir}")
    if args.bc_warmstart:
        print(f"  BC warmstart : enabled  ({args.bc_warmstart_steps:,} steps, "
              f"w0={args.bc_initial_weight:.2f}, n_demos={args.bc_n_demos})")
    if args.finetune_from:
        print(f"  Fine-tune    : {args.finetune_from}")

    # ---- 1. Data pipeline ------------------------------------------------ #
    print("\n[1/5] Running data pipeline...")
    results = run_pipeline()

    # ---- 2. GMM fitting -------------------------------------------------- #
    print("\n[2/5] Fitting bivariate GMM on training [VSTOXX, RTS_Slope_30Y_10Y]...")
    _beta_bar = ([0.70, 0.55, 0.40, 0.25] if args.n_regimes == 4
                 else [0.65, 0.55, 0.35])
    agent_cfg = AgentConfig(
        total_timesteps = args.timesteps,
        learning_rate   = args.lr,
        lr_warmup_steps = args.lr_warmup_steps,
        max_grad_norm   = args.max_grad_norm,
        weight_decay    = args.weight_decay,
        n_steps         = args.n_steps,
        batch_size      = args.batch_size,
        ent_coef        = args.ent_coef,
        vf_coef         = args.vf_coef,
        n_regimes       = args.n_regimes,
        gmm_n_regimes   = args.n_regimes,
        beta_bar        = _beta_bar,
    )
    vstoxx_train    = results["z_train_raw"]["vstoxx_level"].values
    rts_slope_train = results["z_train_raw"]["swap_slope_30y_10y"].values
    gmm = fit_gmm(
        vstoxx_train    = vstoxx_train,
        rts_slope_train = rts_slope_train,
        n_regimes       = agent_cfg.gmm_n_regimes,
        seed            = agent_cfg.gmm_seed,
    )

    # ---- 3. Environments ------------------------------------------------- #
    print("\n[3/5] Building environments...")
    # Build env_cfg; alpha/beta/delta fall back to EnvConfig defaults when
    # the corresponding CLI arg is None (not supplied).
    _env_kwargs: dict = dict(
        dist_weight    = args.dist_weight,
        use_lifecycle  = args.lifecycle,
        log_dist       = args.log_dist,
        log_dist_scale = args.log_dist_scale,
        lambda_smooth  = args.lambda_smooth,
        zeta           = args.zeta,
        epsilon_equity = args.epsilon_equity,
        gamma          = args.gamma_depletion,
        tc_bps         = args.tc_bps,
    )
    if args.alpha is not None:
        _env_kwargs["alpha"] = args.alpha
    if args.beta is not None:
        _env_kwargs["beta"] = args.beta
    if args.delta is not None:
        _env_kwargs["delta"] = args.delta
    if args.fill_bonus is not None:
        _env_kwargs["fill_bonus"] = args.fill_bonus

    env_cfg = EnvConfig(**_env_kwargs)
    print(f"  alpha={env_cfg.alpha}  beta={env_cfg.beta}  gamma={env_cfg.gamma}"
          f"  delta={env_cfg.delta}  fill_bonus={env_cfg.fill_bonus}")
    print(f"  dist_weight={env_cfg.dist_weight}  epsilon_equity={env_cfg.epsilon_equity}"
          f"  use_lifecycle={env_cfg.use_lifecycle}  tc_bps={env_cfg.tc_bps}")

    if args.n_envs > 1:
        def make_train_env(rank):
            def _init():
                return make_env_from_pipeline(results, split="train", cfg=env_cfg, seed=args.seed + rank)
            return _init
        train_env = SubprocVecEnv([make_train_env(i) for i in range(args.n_envs)])
    else:
        train_env = make_env_from_pipeline(results, split="train", cfg=env_cfg, seed=args.seed)

    if args.norm_reward:
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        print("  VecNormalize reward normalisation: ON (clip_reward=10.0)")

    # Val env uses a separate copy of env_cfg so the curriculum callback
    # (which modifies env_cfg.gamma in the training env) never touches it.
    val_env_cfg = copy.copy(env_cfg)
    val_env = make_env_from_pipeline(results, split="val", cfg=val_env_cfg, seed=args.seed)

    # Align validation CPI for metrics computation inside callback
    val_dates = results["z_val"].index
    pi_val    = (
        results["cpi"]["pi_monthly"]
        .reindex(val_dates)
        .fillna(0.0)
        .values
    )

    # ---- 4. Agent -------------------------------------------------------- #
    print("\n[4/5] Building PPO agent...")
    # LR warmup schedule: ramp from 0 → base_lr over lr_warmup_steps.
    # SB3 passes progress_remaining ∈ [1.0, 0.0] to the schedule callable.
    _base_lr      = args.lr
    _warmup_steps = args.lr_warmup_steps
    _total_steps  = args.timesteps

    def _lr_schedule(progress_remaining: float) -> float:
        current_step = (1.0 - progress_remaining) * _total_steps
        warmup_factor = min(1.0, current_step / max(1, _warmup_steps))
        return _base_lr * warmup_factor

    if args.finetune_from:
        ft_path = Path(args.finetune_from)
        # Use the current run's agent_cfg so architecture matches exactly.
        # (Legacy note: run_007 used K=4 regimes; same-run resumes use agent_cfg.)
        agent = load_for_finetune(ft_path, train_env, agent_cfg, seed=args.seed)
    else:
        agent = make_agent(train_env, cfg=agent_cfg, gmm=gmm, seed=args.seed)

    # Apply LR warmup schedule (overrides the constant LR set in make_agent)
    agent.lr_schedule = _lr_schedule
    print(f"  LR warmup: 0 -> {_base_lr} over {_warmup_steps:,} steps")

    # Save training config for reproducibility
    config_path = log_dir / "train_config.json"
    config_dict = {
        "timesteps":        args.timesteps,
        "seed":             args.seed,
        "eval_freq":        args.eval_freq,
        "checkpoint_freq":  args.checkpoint_freq,
        "learning_rate":    args.lr,
        "n_steps":          args.n_steps,
        "batch_size":       args.batch_size,
        "ent_coef":         args.ent_coef,
        "dist_weight":      args.dist_weight,
        "log_dist":         args.log_dist,
        "log_dist_scale":   args.log_dist_scale,
        "n_regimes":        args.n_regimes,
        "lambda_smooth":    args.lambda_smooth,
        "zeta":             args.zeta,
        "epsilon_equity":   args.epsilon_equity,
        "gamma_depletion":  args.gamma_depletion,
        "alpha":            env_cfg.alpha,
        "beta":             env_cfg.beta,
        "delta":            env_cfg.delta,
        "fill_bonus":       env_cfg.fill_bonus,
        "max_grad_norm":    args.max_grad_norm,
        "weight_decay":     args.weight_decay,
        "lr_warmup_steps":  args.lr_warmup_steps,
        "vf_coef":          args.vf_coef,
        "norm_reward":      args.norm_reward,
        "curriculum_gamma": args.curriculum_gamma,
        "curriculum_steps": args.curriculum_steps,
        "finetune_from":    args.finetune_from,
        "lifecycle":        args.lifecycle,
        "bc_warmstart":     args.bc_warmstart,
        "bc_warmstart_steps": args.bc_warmstart_steps,
        "bc_initial_weight":  args.bc_initial_weight,
        "bc_n_demos":       args.bc_n_demos,
        "tc_bps":           env_cfg.tc_bps,
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Config saved: {config_path}")

    # ---- 4b. Behavioral cloning demonstrations (optional) ---------------- #
    demo_obs = demo_actions = None
    if args.bc_warmstart:
        print(f"\n[4b/5] Generating BC demonstrations ({args.bc_n_demos} episodes)...")
        demo_obs, demo_actions = generate_demonstration_data(
            results    = results,
            env_cfg    = env_cfg,
            seed       = args.seed,
            n_episodes = args.bc_n_demos,
        )

    # ---- 5. Train -------------------------------------------------------- #
    print(f"\n[5/5] Training ({args.timesteps:,} timesteps)...\n")
    val_cb   = WtpValidationCallback(
        val_env   = val_env,
        pi_val    = pi_val,
        eval_freq = args.eval_freq,
        log_dir   = log_dir,
        verbose   = 1,
    )
    ckpt_cb  = CheckpointCallback(
        save_freq      = args.checkpoint_freq,
        save_path      = str(log_dir / "checkpoints"),
        name_prefix    = "ppo_wtp",
        save_replay_buffer = False,
        verbose        = 0,
    )

    wm_cb = WeightMonitorCallback(log_freq=50_000, warn_threshold=10.0, verbose=1)
    callbacks = [val_cb, ckpt_cb, wm_cb]
    if args.bc_warmstart and demo_obs is not None:
        bc_cb = BCWarmstartCallback(
            demo_obs            = demo_obs,
            demo_actions        = demo_actions,
            bc_initial_weight   = args.bc_initial_weight,
            bc_final_weight     = 0.0,
            bc_warmstart_steps  = args.bc_warmstart_steps,
            verbose             = 1,
        )
        callbacks.append(bc_cb)
        print(f"  BC warmstart callback: weight {args.bc_initial_weight:.2f} -> 0.0 "
              f"over {args.bc_warmstart_steps:,} steps")
    if args.curriculum_gamma:
        curr_cb = GammaCurriculumCallback(
            target_gamma     = args.gamma_depletion,
            curriculum_steps = args.curriculum_steps,
            verbose          = 1,
        )
        callbacks.append(curr_cb)
        print(f"  Gamma curriculum: 0 -> {args.gamma_depletion} over {args.curriculum_steps:,} steps")

    t0 = time.time()
    agent.learn(
        total_timesteps = args.timesteps,
        callback        = CallbackList(callbacks),
        progress_bar    = not args.no_progress_bar,
        reset_num_timesteps = True,
    )
    elapsed = time.time() - t0

    # ---- Save final model ----------------------------------------------- #
    final_path = log_dir / "final_model"
    agent.save(str(final_path))
    print(f"\n  Final model saved: {final_path}.zip")

    # Save VecNormalize stats so evaluate.py can reproduce training conditions
    if args.norm_reward and isinstance(train_env, VecNormalize):
        norm_path = log_dir / "vec_normalize.pkl"
        train_env.save(str(norm_path))
        print(f"  VecNormalize stats saved: {norm_path}")

    _print_summary(log_dir, elapsed, args.timesteps, val_cb.best_score)


if __name__ == "__main__":
    train()
