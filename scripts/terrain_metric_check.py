"""Check of the success metric on sloped and bumpy ground.

The metric measures "upright" against the ground direction at the start point. If the Ant slides off a slope
onto flatter ground and stands up there, it is standing but tilted relative to the start slope, and at 30 deg it
cannot reach the 0.9 threshold (cos 30 deg = 0.87). This replays the terrain episodes of the stress tests and the
main evaluation (same seeds) and scores each twice: with the original metric (must match stress.csv) and with the ground direction measured
under the torso at every step. Training and the reported results are unchanged; this is an extra check.

    python scripts/terrain_metric_check.py --models "results/final_sac/runs/*/*.cleanrl_model" "results/final_ppo/runs/*/*.cleanrl_model" \
        --out results/final_eval/terrain_metric_check.csv
    python scripts/plot_terrain_check.py results/final_eval/terrain_metric_check.csv results/final_eval/terrain_check.png

(I ran one process per policy in parallel and concatenated the CSVs; the result is the same.)
"""

import argparse
import glob
import os
import sys

import gymnasium as gym
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import evaluate  # noqa: E402
import stress_test  # noqa: E402
import robot_recovery  # noqa: E402,F401

HOLD = 50  # last 2.5 s, as in the metric


def episode(env, policy, seed):
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    cfg, hold, hold_local, done = u.cfg, 0, 0, False
    while not done:
        obs, _, term, trunc, info = env.step(policy.act(obs))
        done = term or trunc
        x, y = u.data.qpos[:2]
        up = u._torso_rot()[:, 2]
        h, w = info["height"], float(np.linalg.norm(u.data.qvel[3:6]))
        ok = h >= cfg.success_height and w <= cfg.success_angvel
        hold = hold + 1 if ok and info["upright_cos"] >= cfg.success_cos else 0
        hold_local = hold_local + 1 if ok and float(up @ u.terrain.normal(x, y)) >= cfg.success_cos else 0
    x, y = u.data.qpos[:2]
    tilt = np.degrees(np.arccos(np.clip(u.terrain.normal(x, y)[2], -1, 1)))
    return info["success"], float(hold >= HOLD), float(hold_local >= HOLD), tilt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--factors", nargs="+", default=["slope", "rough"], help="stress-test ladders to replay")
    p.add_argument("--scenarios", nargs="+", default=["train_mix", "rough", "slope", "hard"], help="main evaluation conditions to replay")
    p.add_argument("--episodes", type=int, default=50, help="per stress level (the main evaluation always uses 100)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    jobs = [(f, lvl, "AntRecovery-v0", kw, a.episodes, 20_000) for f in a.factors for lvl, kw in stress_test.LADDERS[f][1]]
    jobs += [("eval", sc, *evaluate.SCENARIOS[sc], 100, 10_000) for sc in a.scenarios]
    rows = []
    for path in sorted(f for pat in a.models for f in glob.glob(pat)):
        exp, seed = evaluate.describe(path)
        for factor, level, env_id, kw, n, seed0 in jobs:
            env = gym.make(env_id, **kw)
            policy = evaluate.Policy(path, gym.make(env_id, **kw))
            for ep in range(n):
                s, s_orig, s_local, tilt = episode(env, policy, seed0 + ep)
                rows.append({"algo": policy.kind, "seed": seed, "factor": factor, "level": level, "episode": ep, "success": s,
                             "success_recomputed": s_orig, "success_local": s_local, "end_ground_tilt": tilt})
            env.close()
            g = pd.DataFrame(rows[-n:])
            print(f"seed {seed} {factor}={level}  original {g.success.mean():.2f}  local {g.success_local.mean():.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    assert (df.success == df.success_recomputed).all(), "recomputed metric does not match the environment's"
    print(df.groupby(["factor", "level"], sort=False)[["success", "success_local"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
