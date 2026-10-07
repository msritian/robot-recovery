"""Procedural terrain for recovery experiments.

The world is a flat plane (spawn area around the origin) plus a strip of
heightfield tiles along +x. Each tile is 4 m x 4 m and is either:
  - "rough": smooth random bumps (rubble / mine floor), or
  - "slope": a four-sided pyramid, so the robot can land on a slope facing any direction.
Every tile tapers to zero height at its border, so there are no cliffs between tiles
or at the edge of the strip.

The heightfield layout must match the <hfield> declared in assets/ant_recovery.xml.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom

# Must match assets/ant_recovery.xml
HFIELD_CENTER_X = 16.0
HFIELD_HALF_X = 12.0  # strip spans x in [4, 28]
HFIELD_HALF_Y = 2.0  # strip spans y in [-2, 2]
HFIELD_MAX_ELEV = 1.5  # metres; data in [0, 1] is scaled by this
HFIELD_NCOL = 481  # samples along x (5 cm resolution)
HFIELD_NROW = 81  # samples along y
TILE_SIZE = 4.0
N_TILES = 6


@dataclass
class Tile:
    kind: str  # "rough" or "slope"
    center_x: float
    param: float  # rough: bump amplitude (m); slope: face angle (deg)


class Terrain:
    def __init__(self, rough_amps, slope_degs, seed):
        """rough_amps / slope_degs: one entry per rough / slope tile (3 + 3 = 6 tiles)."""
        assert len(rough_amps) + len(slope_degs) == N_TILES
        rng = np.random.default_rng(seed)
        self.xs = np.linspace(HFIELD_CENTER_X - HFIELD_HALF_X, HFIELD_CENTER_X + HFIELD_HALF_X, HFIELD_NCOL)
        self.ys = np.linspace(-HFIELD_HALF_Y, HFIELD_HALF_Y, HFIELD_NROW)
        self.heights = np.zeros((HFIELD_NROW, HFIELD_NCOL))  # [row=y, col=x]

        kinds = [("rough", a) for a in rough_amps] + [("slope", d) for d in slope_degs]
        x0 = HFIELD_CENTER_X - HFIELD_HALF_X
        self.tiles = []
        for i, (kind, param) in enumerate(kinds):
            cx = x0 + TILE_SIZE * (i + 0.5)
            self.tiles.append(Tile(kind, cx, float(param)))
            cols = np.where(np.abs(self.xs - cx) <= TILE_SIZE / 2)[0]
            dx = self.xs[cols][None, :] - cx
            dy = self.ys[:, None]
            if kind == "slope":
                # pyramid: height = tan(theta) * (2 - max(|dx|, |dy|))
                h = np.tan(np.deg2rad(param)) * (TILE_SIZE / 2 - np.maximum(np.abs(dx), np.abs(dy)))
            else:
                # low-frequency bumps (~0.5 m wavelength) + fine gravel noise, tapered to 0 at the border
                coarse = rng.uniform(-1, 1, size=(9, 9))
                h = zoom(coarse, (HFIELD_NROW / 9, len(cols) / 9), order=3)[:HFIELD_NROW, : len(cols)]
                h = h + 0.15 * rng.uniform(-1, 1, size=h.shape)
                h = param * (h - h.min()) / (h.max() - h.min() + 1e-9) * 2.0  # range [0, 2*amp]
                taper = np.clip((TILE_SIZE / 2 - np.maximum(np.abs(dx), np.abs(dy))) / 0.5, 0, 1)
                h = h * taper
            self.heights[:, cols] = np.maximum(self.heights[:, cols], h)
        assert self.heights.max() <= HFIELD_MAX_ELEV, "raise HFIELD_MAX_ELEV in terrain.py and the XML"

    def hfield_data(self):
        """Normalised, flattened data for model.hfield_data (row-major, row 0 at y = -half_y)."""
        return (self.heights / HFIELD_MAX_ELEV).astype(np.float32).ravel()

    def height(self, x, y):
        """Terrain height at world (x, y); bilinear on the heightfield, 0 off the strip."""
        if not (self.xs[0] <= x <= self.xs[-1] and self.ys[0] <= y <= self.ys[-1]):
            return 0.0
        fx = (x - self.xs[0]) / (self.xs[1] - self.xs[0])
        fy = (y - self.ys[0]) / (self.ys[1] - self.ys[0])
        c0, r0 = min(int(fx), HFIELD_NCOL - 2), min(int(fy), HFIELD_NROW - 2)
        tx, ty = fx - c0, fy - r0
        h = self.heights
        return float(
            (1 - ty) * ((1 - tx) * h[r0, c0] + tx * h[r0, c0 + 1]) + ty * ((1 - tx) * h[r0 + 1, c0] + tx * h[r0 + 1, c0 + 1])
        )

    def max_height_near(self, x, y, radius):
        """Highest terrain point within a square of half-width `radius` (for safe spawning)."""
        cols = np.abs(self.xs - x) <= radius
        rows = np.abs(self.ys - y) <= radius
        if not cols.any() or not rows.any():
            return 0.0
        return float(self.heights[np.ix_(rows, cols)].max())

    def normal(self, x, y, eps=0.15):
        """Unit surface normal, from finite differences over +-eps (smooths out gravel)."""
        dhdx = (self.height(x + eps, y) - self.height(x - eps, y)) / (2 * eps)
        dhdy = (self.height(x, y + eps) - self.height(x, y - eps)) / (2 * eps)
        n = np.array([-dhdx, -dhdy, 1.0])
        return n / np.linalg.norm(n)

    def sample_spawn(self, kind, rng):
        """Random (x, y) on a tile of the given kind, away from tile borders."""
        tile = self.tiles[rng.choice([i for i, t in enumerate(self.tiles) if t.kind == kind])]
        if kind == "slope":
            # land on a pyramid face, between 0.6 m and 1.4 m from the apex
            d = rng.uniform(0.6, 1.4)
            lat = rng.uniform(-0.6, 0.6) * d
            sign = rng.choice([-1.0, 1.0])
            dx, dy = (sign * d, lat) if rng.random() < 0.5 else (lat, sign * d)
        else:
            dx, dy = rng.uniform(-1.0, 1.0, size=2)
        return tile.center_x + dx, dy, tile
