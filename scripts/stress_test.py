"""Stress ladders: turn up one real-world factor at a time until recovery breaks.

Each ladder starts from the training distribution and pins a single factor to increasing levels
(e.g. slope angle 5 -> 35 deg); everything else is randomised as in training. Every policy sees the
same episode seeds at every level. Output: per-episode CSV + one panel per factor (success vs level).

    python scripts/stress_test.py --models "results/final_*/runs/*/*.cleanrl_model" --episodes 50 --out results/stress
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import evaluate  # noqa: E402

# factor -> (axis label, [(level value, overrides)])
LADDERS = {
    "slope": ("Slope angle (deg)", [(d, {"scenario_probs": {"slope": 1.0}, "slope_degs": (d, d, d)}) for d in [5, 10, 15, 20, 25, 30, 35]]),
    "rough": ("Bump amplitude (cm)", [(a * 100, {"scenario_probs": {"rough": 1.0}, "rough_amps": (a, a, a)}) for a in [0.03, 0.06, 0.10, 0.15, 0.20]]),
    "rubble": ("Debris pieces (unseen in training)", [(n, {"scenario_probs": {"rubble": 1.0}, "rubble_count": n}) for n in [20, 70, 120, 200]]),
    "push": ("Shove speed change (m/s)", [(v, {"push_prob": 1.0, "push_dv": (v, v), "push_spin": 2.0 * v}) for v in [0.5, 1.0, 2.0, 3.0, 4.0]]),
    "friction": ("Friction (x nominal)", [(f, {"friction_scale": (f, f)}) for f in [1.0, 0.6, 0.4, 0.25, 0.15]]),
    "payload": ("Torso mass (x nominal)", [(m, {"torso_mass_scale": (m, m)}) for m in [1.0, 1.5, 2.0, 3.0, 4.0]]),
    "motors": ("Motor strength (x nominal)", [(k, {"motor_scale": (k, k)}) for k in [1.0, 0.7, 0.5, 0.4, 0.3]]),
    "drop": ("Drop height (m)", [(h, {"drop_prob": 1.0, "drop_height": (h, h), "drop_spin": 4.0}) for h in [0.5, 1.0, 2.0, 3.0]]),
    "dead_legs": ("Dead legs", [(0, {"leg_fault_prob": 0.0}), (1, {"leg_fault_prob": 1.0, "leg_fault_count": 1}), (2, {"leg_fault_prob": 1.0, "leg_fault_count": 2})]),
}
COLORS = {"ppo": "#1f77b4", "sac": "#d62728"}


def plot(df, out_png):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (factor, (label, _)) in zip(axes.flat, LADDERS.items()):
        sub = df[df.factor == factor]
        for algo, g in sub.groupby("algo"):
            per_seed = g.groupby(["seed", "level"]).success.mean().unstack(0)  # level x seed
            for s in per_seed.columns:
                ax.plot(per_seed.index, per_seed[s], color=COLORS.get(algo), alpha=0.25, lw=0.8)
            n = per_seed.shape[1]
            mean = per_seed.mean(axis=1)
            if n > 1:
                half = stats.t.ppf(0.975, n - 1) * per_seed.std(axis=1, ddof=1) / np.sqrt(n)
                ax.fill_between(per_seed.index, mean - half, mean + half, color=COLORS.get(algo), alpha=0.2, lw=0)
            ax.plot(per_seed.index, mean, "o-", color=COLORS.get(algo), lw=2, ms=4, label=f"{algo.upper()} (n={n})")
        ax.set_title(factor.replace("_", " "))
        ax.set_xlabel(label)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Recovery success")
    axes.flat[0].legend(loc="lower left")
    fig.suptitle("Stress ladders: success vs. one factor at a time (mean ± t-95% CI over seeds; thin = seeds)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_png.replace(".png", ".pdf"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--factors", nargs="+", default=list(LADDERS), choices=list(LADDERS))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--out", required=True, help="output directory")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    paths = sorted({f for pat in a.models for f in (glob.glob(pat) if pat not in ("zero", "random") else [pat])})
    frames = []
    for path in paths:
        exp, seed = evaluate.describe(path)
        for factor in a.factors:
            for level, overrides in LADDERS[factor][1]:
                name = f"{factor}={level}"
                evaluate.SCENARIOS[name] = ("AntRecovery-v0", overrides)
                df, kind = evaluate.run(path, name, a.episodes, a.seed)
                frames.append(df.assign(exp=exp, algo=kind, seed=seed, factor=factor, level=level))
                print(f"{exp:24s} seed {seed}  {name:22s} success {df.success.mean():.2f}  t_recover {df.time_to_recover.mean():5.2f}s", flush=True)
    df = pd.concat(frames)
    df.to_csv(os.path.join(a.out, "stress.csv"), index=False)
    plot(df, os.path.join(a.out, "stress.png"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
