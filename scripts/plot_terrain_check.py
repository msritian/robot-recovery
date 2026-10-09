"""Figure for terrain_metric_check.py: success on slopes and bumps, scored with the ground direction at the
start point (the metric used everywhere else) and with the ground direction under the torso.

    python scripts/plot_terrain_check.py results/final_eval/terrain_metric_check.csv results/final_eval/terrain_check.png
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

COLORS = {"ppo": "#1f77b4", "sac": "#d62728"}
PANELS = {"slope": "Slope angle (deg)", "rough": "Bump amplitude (cm)"}


def main(csv, out):
    df = pd.read_csv(csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (factor, label) in zip(axes, PANELS.items()):
        sub = df[df.factor == factor].assign(level=lambda d: d.level.astype(float))
        for algo, g in sub.groupby("algo"):
            for col, style, name in [("success", "-", "start-point ground (metric)"), ("success_local", "--", "ground under torso")]:
                per_seed = g.groupby(["level", "seed"])[col].mean().unstack()
                n, mean = per_seed.shape[1], per_seed.mean(axis=1)
                half = stats.t.ppf(0.975, n - 1) * per_seed.std(axis=1, ddof=1) / np.sqrt(n)
                ax.fill_between(mean.index, mean - half, mean + half, color=COLORS[algo], alpha=0.12, lw=0)
                ax.plot(mean.index, mean, style, marker="o", ms=3, color=COLORS[algo], label=f"{algo.upper()}, {name}")
        ax.set_xlabel(label)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recovery success")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Same episodes scored two ways (5 seeds, 50 episodes per level, mean ± t-95%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    main(*sys.argv[1:3])
