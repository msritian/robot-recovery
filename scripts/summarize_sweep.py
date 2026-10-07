"""Table of a tuning sweep: success rate per configuration (tag) and seed at several step budgets.

    python scripts/summarize_sweep.py results/<batch>/runs [--at 1e6 3e6 5e6] [--plot out.png]

Success at budget B = mean success over the 100 training episodes that ended just before B.
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--at", nargs="+", type=float, default=[1e6, 2e6, 3e6, 5e6])
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--plot")
    a = p.parse_args()

    curves = defaultdict(dict)  # tag -> seed -> (steps, success)
    for root in a.run_dirs:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            m = re.match(r".+__(?P<tag>.+)__(?P<seed>\d+)__\d+$", os.path.basename(d))
            if not m:
                continue
            acc = EventAccumulator(d, size_guidance={"scalars": 0})
            acc.Reload()
            if "charts/success" not in acc.Tags()["scalars"]:
                continue
            ev = acc.Scalars("charts/success")
            curves[m["tag"]][int(m["seed"])] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))

    header = f"{'config':45s} " + " ".join(f"{b / 1e6:>4.0f}M mean (per seed)".rjust(26) for b in a.at)
    print(header)
    for tag in sorted(curves):
        cells = []
        for b in a.at:
            vals = []
            for seed in sorted(curves[tag]):
                steps, succ = curves[tag][seed]
                done = succ[steps <= b]
                vals.append(done[-a.window :].mean() if len(done) and steps[-1] >= 0.98 * b else np.nan)
            cells.append(f"{np.nanmean(vals) if not np.all(np.isnan(vals)) else float('nan'):.2f} ({' '.join(f'{v:.2f}' for v in vals)})".rjust(26))
        print(f"{tag:45s} " + " ".join(cells))

    if a.plot:
        import matplotlib.pyplot as plt
        import pandas as pd

        fig, ax = plt.subplots(figsize=(9, 5))
        cmap = plt.get_cmap("tab10")
        for i, tag in enumerate(sorted(curves)):
            for j, seed in enumerate(sorted(curves[tag])):
                steps, succ = curves[tag][seed]
                sm = pd.Series(succ).rolling(a.window, min_periods=1).mean()
                ax.plot(steps, sm, color=cmap(i % 10), alpha=0.8, lw=1, label=tag if j == 0 else None)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(f"Success (rolling {a.window} episodes)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print("wrote", a.plot)


if __name__ == "__main__":
    main()
