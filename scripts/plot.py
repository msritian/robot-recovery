"""Learning curves for PPO vs SAC, as required by the assignment (Section 5.4 / 6).

For each run, the per-episode scalar (return or success) is smoothed over a trailing window of
episodes *within that seed*, then linearly interpolated onto a common environment-step grid.
Across seeds we plot the mean and the Student-t 95% interval  mean +- t_{0.975,n-1} * s / sqrt(n),
with every individual seed drawn as a thin line behind it.

    python scripts/plot.py --runs results/<batch>/runs --out results/<batch>/plots
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

COLORS = {"ppo": "#1f77b4", "sac": "#d62728"}
METRICS = {
    "charts/episodic_return": ("Episodic return", "return"),
    "charts/success": ("Success rate (upright & steady for final 2.5 s)", "success"),
    "charts/time_to_recover": ("Time to recover (s, 20 = never)", "time_to_recover"),
}


def load_runs(run_dir):
    """{algo: {seed: {tag: (steps, values)}}} from CleanRL TensorBoard logs."""
    runs = defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(run_dir, "*"))):
        m = re.match(r".+__(?P<exp>.+)__(?P<seed>\d+)__\d+$", os.path.basename(d))
        if not m:
            continue
        algo = "ppo" if "ppo" in m["exp"] else "sac" if "sac" in m["exp"] else m["exp"]
        acc = EventAccumulator(d, size_guidance={"scalars": 0})
        acc.Reload()
        runs[algo][int(m["seed"])] = {
            tag: (np.array([e.step for e in acc.Scalars(tag)]), np.array([e.value for e in acc.Scalars(tag)]))
            for tag in METRICS
            if tag in acc.Tags()["scalars"]
        }
    return runs


def seed_curve(steps, values, grid, window):
    smoothed = pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()
    out = np.interp(grid, steps, smoothed)
    out[grid < steps[0]] = np.nan  # no extrapolation before the first finished episode
    return out


def plot_metric(runs, tag, grid, window, ax):
    title, _ = METRICS[tag]
    summary = {}
    for algo in sorted(runs):
        curves = np.array([seed_curve(*runs[algo][s][tag], grid, window) for s in sorted(runs[algo]) if tag in runs[algo][s]])
        if len(curves) == 0:
            continue
        n = len(curves)
        for c in curves:
            ax.plot(grid, c, color=COLORS.get(algo), alpha=0.25, lw=0.8)
        mean = np.nanmean(curves, axis=0)
        if n > 1:
            half = stats.t.ppf(0.975, n - 1) * np.nanstd(curves, axis=0, ddof=1) / np.sqrt(n)
            ax.fill_between(grid, mean - half, mean + half, color=COLORS.get(algo), alpha=0.2, lw=0)
        ax.plot(grid, mean, color=COLORS.get(algo), lw=2, label=f"{algo.upper()} (n={n})")
        summary[algo] = (mean[-1], half[-1] if n > 1 else np.nan, curves[:, -1])
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=50, help="trailing smoothing window, in episodes, per seed")
    p.add_argument("--grid-points", type=int, default=200)
    a = p.parse_args()

    runs = load_runs(a.runs)
    max_step = min(max(v[0][-1] for s in seeds.values() for v in s.values()) for seeds in runs.values())
    grid = np.linspace(0, max_step, a.grid_points)
    os.makedirs(a.out, exist_ok=True)
    for tag, (title, short) in METRICS.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        summary = plot_metric(runs, tag, grid, a.window, ax)
        ax.set_title(f"AntRecovery-v0: {title.split(' (')[0]}  (mean ± t-95% CI, thin = seeds)")
        if short == "success":
            ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{short}.png"), dpi=200)
        fig.savefig(os.path.join(a.out, f"{short}.pdf"))
        plt.close(fig)
        for algo, (m, h, finals) in summary.items():
            print(f"{short:16s} {algo}: final {m:.3f} ± {h:.3f}   per-seed {np.round(finals, 3).tolist()}")


if __name__ == "__main__":
    main()
