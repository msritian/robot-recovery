"""Tile evenly spaced frames of one or more videos into a single PNG (one row per video)."""
import sys
import imageio.v2 as imageio
import numpy as np

out, paths = sys.argv[1], sys.argv[2:]
rows = []
for p in paths:
    frames = imageio.mimread(p, memtest=False)
    idx = np.linspace(0, len(frames) - 1, 8).astype(int)
    rows.append(np.concatenate([frames[i][::2, ::2] for i in idx], axis=1))
imageio.imwrite(out, np.concatenate(rows, axis=0))
print(out, [len(r) for r in rows])
