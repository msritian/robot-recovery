"""Evaluate trained policies per real-world scenario, and optionally record videos.

Policies act deterministically (PPO: Gaussian mean; SAC: tanh of the mean). Every policy sees the
same episode seeds in every scenario, so results are paired across algorithms and seeds.

    python scripts/evaluate.py --models results/<batch>/runs/*/*.cleanrl_model --episodes 100 --out results/<batch>/eval.csv
    python scripts/evaluate.py --models <one model> --scenarios train_mix hard --episodes 5 --video-dir videos/best
"""

import argparse
import glob
import os
import re
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algos"))
import robot_recovery  # noqa: E402,F401
from ppo_continuous_action import Agent  # noqa: E402
from sac_continuous_action import Actor  # noqa: E402

# name -> (env id, overrides). Single-factor stressors isolate one real-world failure mode each.
SCENARIOS = {
    "train_mix": ("AntRecovery-v0", {}),
    "flat": ("AntRecovery-v0", {"scenario_probs": {"flat": 1.0}}),
    "rough": ("AntRecovery-v0", {"scenario_probs": {"rough": 1.0}}),
    "slope": ("AntRecovery-v0", {"scenario_probs": {"slope": 1.0}}),
    "strong_push": ("AntRecovery-v0", {"push_prob": 1.0, "push_dv": (1.0, 2.0), "push_spin": 4.0}),
    "high_drop": ("AntRecovery-v0", {"drop_prob": 1.0, "drop_height": (1.0, 1.5), "drop_spin": 5.0}),
    "low_friction": ("AntRecovery-v0", {"friction_scale": (0.3, 0.5)}),
    "heavy_payload": ("AntRecovery-v0", {"torso_mass_scale": (1.5, 2.0)}),
    "weak_motors": ("AntRecovery-v0", {"motor_scale": (0.6, 0.7)}),
    "limp_leg": ("AntRecovery-v0", {"leg_fault_prob": 1.0, "leg_fault_strength": 0.0}),
    "rubble": ("AntRecovery-v0", {"scenario_probs": {"rubble": 1.0}}),  # unseen debris field (never trained on)
    "hard": ("AntRecoveryHard-v0", {}),
}


class Baseline:
    """Reference policies: "zero" (do nothing) and "random" (uniform random torques)."""

    def __init__(self, kind, seed=0):
        self.kind = kind
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return np.zeros(8) if self.kind == "zero" else self.rng.uniform(-1.0, 1.0, size=8)


class Policy:
    def __init__(self, path, env):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # PPO checkpoints hold numpy obs stats
        venv = gym.vector.SyncVectorEnv([lambda: env])  # the CleanRL nets read spaces from a vector env
        if "agent" in ckpt:
            self.kind = "ppo"
            self.net = Agent(venv, hidden_size=ckpt["agent"]["actor_mean.0.weight"].shape[0])
            self.net.load_state_dict(ckpt["agent"])
            self.mean, self.var = ckpt["obs_mean"], ckpt["obs_var"]
        else:
            self.kind = "sac"
            self.net = Actor(venv)
            self.net.load_state_dict(ckpt["actor"])
        self.net.eval()

    @torch.no_grad()
    def act(self, obs):
        if self.kind == "ppo":
            obs = np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)
            a = self.net.actor_mean(torch.as_tensor(obs, dtype=torch.float32)[None])
        else:
            _, _, a = self.net.get_action(torch.as_tensor(obs, dtype=torch.float32)[None])
        return np.clip(a[0].numpy(), -1.0, 1.0)


def run(model_path, scenario, episodes, seed0, video_dir=None):
    env_id, overrides = SCENARIOS[scenario]
    env = gym.make(env_id, render_mode="rgb_array" if video_dir else None, **overrides)
    if video_dir:
        env = gym.wrappers.RecordVideo(env, os.path.join(video_dir, scenario), episode_trigger=lambda e: True, disable_logger=True)
    policy = Baseline(model_path) if model_path in ("zero", "random") else Policy(model_path, gym.make(env_id, **overrides))
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(policy.act(obs))
            ret += r
            done = term or trunc
        rows.append(
            {"episode": ep, "return": ret, "success": info["success"], "recovered": info["recovered"],
             "time_to_recover": info["time_to_recover"], "terrain": info["scenario"]}
        )
    env.close()
    return pd.DataFrame(rows), policy.kind


def describe(path):
    if path in ("zero", "random"):
        return f"baseline_{path}", 0
    run_name = os.path.basename(os.path.dirname(path))
    m = re.match(r"(?P<env>.+?)__(?P<exp>.+)__(?P<seed>\d+)__\d+$", run_name)
    return (m["exp"], int(m["seed"])) if m else (run_name, -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help='paths or globs to .cleanrl_model files, or "zero" / "random" baselines')
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000, help="first evaluation episode seed (disjoint from training seeds)")
    p.add_argument("--out", help="CSV of per-episode results")
    p.add_argument("--video-dir", help="record every evaluation episode here")
    a = p.parse_args()

    paths = sorted({f for pat in a.models for f in (glob.glob(pat) if pat not in ("zero", "random") else [pat])})
    frames = []
    for path in paths:
        exp, seed = describe(path)
        for sc in a.scenarios:
            df, kind = run(path, sc, a.episodes, a.seed, a.video_dir)
            df = df.assign(exp=exp, algo=kind, seed=seed, scenario=sc, model=path)
            frames.append(df)
            print(f"{exp:40s} seed {seed}  {sc:14s} success {df.success.mean():.2f}  recovered {df.recovered.mean():.2f}  "
                  f"t_recover {df.time_to_recover.mean():5.2f}s  return {df['return'].mean():7.1f}", flush=True)
    if a.out and frames:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pd.concat(frames).to_csv(a.out, index=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
