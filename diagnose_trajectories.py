"""Trajectory decomposition diagnostic: discretionary vs mechanical distributions."""
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.data_pipeline import run_pipeline
from src.environment   import make_env_from_pipeline, EnvConfig
from src.agent         import AgentConfig, WtpActorCriticPolicy
from src.baselines     import FixedRuleALM
import zipfile
from stable_baselines3 import PPO

def run_episode_detailed(agent, env):
    """Run one episode, return per-step trajectory dict."""
    obs, _ = env.reset()
    done = False
    rows = []
    while not done:
        if hasattr(agent, 'predict'):
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = agent.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append({
            'FR':        info['FR'],
            'B':         info['B'],
            'f_tilde':   info['f_tilde'],
            'd_tilde':   info['d_tilde'],
            'dec_excess':info['dec_excess'],
        })
    return rows

def analyse(rows, name):
    FR          = np.array([r['FR']         for r in rows])
    B           = np.array([r['B']          for r in rows])
    d_tilde     = np.array([r['d_tilde']    for r in rows])
    dec_excess  = np.array([r['dec_excess'] for r in rows])
    f_tilde     = np.array([r['f_tilde']    for r in rows])

    total_dist        = d_tilde.sum() + dec_excess.sum()
    disc_dist         = d_tilde.sum()
    mech_dist         = dec_excess.sum()
    fill_total        = f_tilde.sum()
    # Art. 10d lid 4: distributions allowed when FR >= 1.00 AND B > 0.
    # Use B > 0 (not B > 0.005) to match the actual rule in apply_distribution_rule().
    months_eligible   = int(((FR >= 1.00) & (B > 0.0)).sum())
    months_dist       = int((d_tilde > 0.001).sum())
    participation     = months_dist / max(months_eligible, 1)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Total distributions  : {total_dist:.4f}")
    print(f"    Discretionary (d~) : {disc_dist:.4f}  ({100*disc_dist/max(total_dist,1e-9):.1f}%)")
    print(f"    Mechanical (dec+)  : {mech_dist:.4f}  ({100*mech_dist/max(total_dist,1e-9):.1f}%)")
    print(f"  Total fills          : {fill_total:.4f}")
    print(f"  Eligible months      : {months_eligible} / {len(rows)}")
    print(f"  Months distributed   : {months_dist}")
    print(f"  Participation rate   : {participation:.1%}")
    print(f"  Buf Depl Freq        : {(B < 0.005).mean():.3f}")
    print(f"  FR terminal          : {FR[-1]:.4f}")

if __name__ == "__main__":
    print("Running data pipeline (test split)...")
    results  = run_pipeline()
    env_cfg  = EnvConfig(use_lifecycle=True)
    test_env = make_env_from_pipeline(results, split="test", cfg=env_cfg, seed=0)

    # --- DRL run_028 ---
    model_path = _ROOT / "src/models/run_028/best_model.zip"
    with zipfile.ZipFile(model_path) as zf:
        with zf.open("policy.pkl") as f:
            import io, pickle
            policy = pickle.load(f)
    agent_drl = PPO.load(str(model_path), env=test_env, device="cpu",
                         custom_objects={"policy_class": WtpActorCriticPolicy,
                                         "lr_schedule": lambda _: 3e-4})
    rows_drl = run_episode_detailed(agent_drl, test_env)
    analyse(rows_drl, "DRL run_028 (best_model)")

    # --- Fixed-Rule ---
    fixed_cfg  = EnvConfig(use_lifecycle=True)
    fixed_env  = make_env_from_pipeline(results, split="test", cfg=fixed_cfg, seed=0)
    fixed_rule = FixedRuleALM()

    class _FixedAdapter:
        def predict(self, obs, deterministic=True):
            return fixed_rule.act(fixed_env), None

    rows_fixed = run_episode_detailed(_FixedAdapter(), fixed_env)
    analyse(rows_fixed, "Fixed-Rule ALM")
