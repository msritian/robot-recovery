"""Join video parts into one file, with a short title card before each part.

    python scripts/combine_videos.py --out videos/best_policy_sac_seed4.mp4 \
        --part "Part 1: 8 random episodes in a row (not hand-picked)" videos/parts/typical_episodes_sac_seed4.mp4 \
        --part "Part 2: harder conditions than training, and two failures" videos/highlight_reel_sac_seed4.mp4
"""

import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from make_reel import font

FPS = 20


def card(lines, size, seconds=2.0):
    img = Image.new("RGB", size, (20, 20, 20))
    d = ImageDraw.Draw(img)
    y = size[1] // 2 - 22 * len(lines)
    for i, text in enumerate(lines):
        size_pt = 26 if i == 0 else 19
        while size_pt > 10 and d.textlength(text, font=font(size_pt)) > size[0] - 40:  # shrink to fit
            size_pt -= 1
        f = font(size_pt)
        d.text(((size[0] - d.textlength(text, font=f)) / 2, y), text, font=f, fill=(255, 255, 255) if i == 0 else (200, 200, 200))
        y += 44
    return [np.asarray(img)] * int(seconds * FPS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Best policy: SAC seed 4, actions without randomness")
    p.add_argument("--part", nargs=2, action="append", metavar=("CAPTION", "VIDEO"), required=True)
    a = p.parse_args()
    writer = imageio.get_writer(a.out, fps=FPS, macro_block_size=16)
    n = 0
    for i, (caption, path) in enumerate(a.part):
        reader = imageio.get_reader(path)
        first = reader.get_data(0)
        for frame in card(([a.title] if i == 0 else []) + [caption], (first.shape[1], first.shape[0])):
            writer.append_data(frame)
            n += 1
        for frame in reader:  # one frame at a time; whole clips do not fit in memory
            writer.append_data(frame)
            n += 1
        reader.close()
    writer.close()
    print("wrote", a.out, f"({n / FPS:.0f} s)")

if __name__ == "__main__":
    main()
