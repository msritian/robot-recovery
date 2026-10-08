"""Learning curves for PPO vs SAC, as required by the assignment (Section 5.4 / 6).

For each run, the per-episode scalar (return or success) is smoothed over a trailing window of
episodes *within that seed*, then linearly interpolated onto a common environment-step grid.
Across seeds we plot the mean and the Student-t 95% interval  mean +- t_{0.975,n-1} * s / sqrt(n),
with every individual seed drawn as a thin line behind it.

    python scripts/plot.py --runs results/final_ppo/runs results/final_sac/runs --out results/final_plots
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


def load_runs(run_dirs):
    """{algo: {seed: {tag: (steps, values)}}} from CleanRL TensorBoard logs."""
    runs = defaultdict(dict)
    for d in sorted(d for r in run_dirs for d in glob.glob(os.path.join(r, "*"))):
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
    out[(grid < steps[0]) | (grid > steps[-1])] = np.nan  # no extrapolation outside the run
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
        # only plot the mean where every seed has data, so n is the same at every point
        full = ~np.isnan(curves).any(axis=0)
        mean = np.where(full, curves.mean(axis=0), np.nan)
        half = np.full_like(mean, np.nan)
        if n > 1:
            half = np.where(full, stats.t.ppf(0.975, n - 1) * curves.std(axis=0, ddof=1) / np.sqrt(n), np.nan)
            ax.fill_between(grid, mean - half, mean + half, color=COLORS.get(algo), alpha=0.2, lw=0)
        ax.plot(grid, mean, color=COLORS.get(algo), lw=2, label=f"{algo.upper()} (n={n})")
        last = np.where(full)[0][-1]
        summary[algo] = (grid[last], mean[last], half[last], curves[:, last], curves)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="one or more CleanRL runs/ directories")
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=50, help="trailing smoothing window, in episodes, per seed")
    p.add_argument("--grid-points", type=int, default=200)
    p.add_argument("--env-label", default="AntRecoveryLowCost-v0")
    p.add_argument("--xmax", type=float, help="only plot up to this many steps (e.g. 1e6 for an equal-budget view)")
    p.add_argument("--suffix", default="", help="appended to output file names")
    p.add_argument("--compare-at", nargs="*", type=float, default=[], help="also report every algorithm at these steps")
    a = p.parse_args()

    runs = load_runs(a.runs)
    max_step = a.xmax or max(v[0][-1] for seeds in runs.values() for s in seeds.values() for v in s.values())
    grid = np.linspace(0, max_step, a.grid_points)
    os.makedirs(a.out, exist_ok=True)
    for tag, (title, short) in METRICS.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        summary = plot_metric(runs, tag, grid, a.window, ax)
        ax.set_title(f"{a.env_label}: {title.split(' (')[0]}  (mean ± t-95% CI, thin = seeds)", fontsize=10)
        if short == "success":
            ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{short}{a.suffix}.png"), dpi=200)
        fig.savefig(os.path.join(a.out, f"{short}{a.suffix}.pdf"))
        plt.close(fig)
        for algo, (step, m, h, finals, curves) in summary.items():
            print(f"{short:16s} {algo} @ {step / 1e6:.2f}M (end): {m:.3f} ± {h:.3f}   per-seed {np.round(finals, 3).tolist()}")
            for at in a.compare_at:
                i = np.argmin(np.abs(grid - at))
                c = curves[:, i]
                if not np.isnan(c).any():
                    hh = stats.t.ppf(0.975, len(c) - 1) * c.std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else np.nan
                    print(f"{'':16s} {algo} @ {grid[i] / 1e6:.2f}M: {c.mean():.3f} ± {hh:.3f}   per-seed {np.round(c, 3).tolist()}")


if __name__ == "__main__":
    main()
