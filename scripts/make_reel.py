"""Labelled highlight reel: one short clip per condition, from a single trained policy.

For each condition we run evaluation episodes (same seeds as evaluate.py) and keep the first episode whose
outcome matches what we want to show (success, or failure for the "limits" clips), so every clip is a real,
unedited rollout; the caption states the outcome measured by the true metric.

    python scripts/make_reel.py --model <best.cleanrl_model> --out videos/highlight_reel.mp4
"""

import argparse
import os
import sys

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import evaluate  # noqa: E402

# (caption, overrides, want_success)
CLIPS = [
    ("Upside down on flat ground", {"scenario_probs": {"flat": 1.0}, "tilt_deg": (170.0, 180.0), "drop_prob": 0.0, "push_prob": 0.0}, True),
    ("Bumpy ground (up to 14 cm)", {"scenario_probs": {"rough": 1.0}, "push_prob": 0.0}, True),
    ("Landing on a 15° slope", {"scenario_probs": {"slope": 1.0}, "slope_degs": (15.0, 15.0, 15.0), "push_prob": 0.0}, True),
    ("Rubble field (never seen in training)", {"scenario_probs": {"rubble": 1.0}, "push_prob": 0.0}, True),
    ("Dropped from 2 m while spinning", {"scenario_probs": {"flat": 1.0}, "drop_prob": 1.0, "drop_height": (2.0, 2.0), "drop_spin": 4.0, "push_prob": 0.0}, True),
    ("Shoved at 3 m/s at t = 5 s", {"scenario_probs": {"flat": 1.0}, "push_prob": 1.0, "push_dv": (3.0, 3.0), "push_spin": 6.0, "push_step": (100, 101), "drop_prob": 0.0}, True),
    ("Icy ground (friction x0.2)", {"scenario_probs": {"flat": 1.0}, "friction_scale": (0.2, 0.2), "push_prob": 0.0}, True),
    ("3x payload on the torso", {"scenario_probs": {"flat": 1.0}, "torso_mass_scale": (3.0, 3.0), "push_prob": 0.0}, True),
    ("One leg dead (no torque)", {"scenario_probs": {"flat": 1.0}, "leg_fault_prob": 1.0, "push_prob": 0.0}, True),
    ("Limit: 30° slope", {"scenario_probs": {"slope": 1.0}, "slope_degs": (30.0, 30.0, 30.0), "push_prob": 0.0}, False),
    ("Limit: two legs dead", {"scenario_probs": {"flat": 1.0}, "leg_fault_prob": 1.0, "leg_fault_count": 2, "push_prob": 0.0}, False),
]


def font(size):
    for f in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def caption(frame, title, sub, color):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, img.width, 62], fill=(0, 0, 0, 150))
    size = 24
    while size > 12 and d.textlength(title, font=font(size)) > img.width - 28:  # shrink long titles to fit
        size -= 1
    d.text((14, 8), title, font=font(size), fill=(255, 255, 255))
    d.text((14, 36), sub, font=font(17), fill=color)
    return np.asarray(img)


def rollout(policy, overrides, seed, max_frames, size, tint_ice):
    env = gym.make("AntRecovery-v0", render_mode="rgb_array", width=size[0], height=size[1], **overrides)
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    if tint_ice:
        floor = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        u.model.geom_rgba[floor] = [0.55, 0.8, 1.0, 1.0]  # the floor's own rgba overrides its material
    frames, done, info, t = [], False, {}, 0
    rollout.setup = (u.scenario, u._push_step, u.data.qpos[2] - u.terrain.height(*u.data.qpos[:2]))
    while not done:
        obs, _, term, trunc, info = env.step(policy.act(obs))
        if t < max_frames:
            frames.append((env.render(), info["upright_cos"] >= 0.9 and info["height"] >= 0.32))
        t += 1
        done = term or trunc
    env.close()
    return frames, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=10.0, help="length of each clip (the episode itself is 20 s)")
    p.add_argument("--tries", type=int, default=20, help="episodes to search for the wanted outcome")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--train-mix", type=int, default=0,
                   help="instead of the condition clips, record this many consecutive training-distribution episodes")
    a = p.parse_args()

    probe = gym.make("AntRecovery-v0")
    policy = evaluate.Policy(a.model, probe)
    label = f"{policy.kind.upper()} policy, deterministic, true-metric outcome shown"
    max_frames = int(a.seconds * 20)
    out_frames = []
    clips = CLIPS
    if a.train_mix:  # unfiltered: consecutive episodes, whatever the outcome
        clips = [(None, {}, None)] * a.train_mix
    for i, (title, overrides, want) in enumerate(clips):
        if want is None:
            frames, info = rollout(policy, overrides, a.seed + i, max_frames, (640, 480), False)
            terrain, push, h0 = rollout.setup
            title = f"Episode {i + 1}: {terrain} ground" + (f", dropped (torso starts {h0:.1f} m up)" if h0 > 0.6 else "")
            title += f", shoved at {push * 0.05:.1f} s" if 0 <= push < max_frames else ""
            k = i
        for k in range(a.tries if want is not None else 0):
            frames, info = rollout(policy, overrides, a.seed + k, max_frames, (640, 480), "friction_scale" in overrides)
            if bool(info["success"]) == want:
                break
        outcome = (f"RECOVERED in {info['time_to_recover']:.1f} s, standing at 20 s" if info["success"]
                   else "FAILED: not standing at the end of the 20 s episode")
        color = (120, 255, 120) if info["success"] else (255, 120, 120)
        for frame, up in frames:
            out_frames.append(caption(frame, title, outcome + ("   [upright]" if up else ""), color))
        print(f"{title:42s} seed {a.seed + k}  success={info['success']:.0f}  t={info['time_to_recover']:.1f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    imageio.mimsave(a.out, out_frames, fps=20, macro_block_size=16)
    print("wrote", a.out, f"({len(out_frames) / 20:.0f} s) —", label)


if __name__ == "__main__":
    main()
