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
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import evaluate  # noqa: E402

# (caption, overrides, want_success, where its measured success rate comes from)
CLIPS = [
    ("Upside down on flat ground", {"scenario_probs": {"flat": 1.0}, "tilt_deg": (170.0, 180.0), "drop_prob": 0.0, "push_prob": 0.0}, True, ("eval", "flat")),
    ("Bumpy ground, bumps up to 20 cm", {"scenario_probs": {"rough": 1.0}, "rough_amps": (0.10, 0.10, 0.10), "push_prob": 0.0}, True, ("rough", 10.0)),
    ("Landing on a 20° slope", {"scenario_probs": {"slope": 1.0}, "slope_degs": (20.0, 20.0, 20.0), "push_prob": 0.0}, True, ("slope", 20.0)),
    ("Dense rubble field, 200 pieces (never seen in training)", {"scenario_probs": {"rubble": 1.0}, "rubble_count": 200, "push_prob": 0.0}, True, ("rubble", 200.0)),
    ("Dropped from 3 m while spinning", {"scenario_probs": {"flat": 1.0}, "drop_prob": 1.0, "drop_height": (3.0, 3.0), "drop_spin": 4.0, "push_prob": 0.0}, True, ("drop", 3.0)),
    ("Shoved at 4 m/s at t = 5 s", {"scenario_probs": {"flat": 1.0}, "push_prob": 1.0, "push_dv": (4.0, 4.0), "push_spin": 8.0, "push_step": (100, 101), "drop_prob": 0.0}, True, ("push", 4.0)),
    ("Ice: friction x0.15", {"scenario_probs": {"flat": 1.0}, "friction_scale": (0.15, 0.15), "push_prob": 0.0}, True, ("friction", 0.15)),
    ("4x payload on the torso", {"scenario_probs": {"flat": 1.0}, "torso_mass_scale": (4.0, 4.0), "push_prob": 0.0}, True, ("payload", 4.0)),
    ("One leg dead (no torque)", {"scenario_probs": {"flat": 1.0}, "leg_fault_prob": 1.0, "push_prob": 0.0}, True, ("dead_legs", 1.0)),
    ("Limit: motors at 30% strength", {"scenario_probs": {"flat": 1.0}, "motor_scale": (0.3, 0.3), "push_prob": 0.0}, False, ("motors", 0.3)),
    ("Limit: two legs dead", {"scenario_probs": {"flat": 1.0}, "leg_fault_prob": 1.0, "leg_fault_count": 2, "push_prob": 0.0}, False, ("dead_legs", 2.0)),
]


def font(size):
    for f in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def caption(frame, title, sub, color, note=""):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, img.width, 84 if note else 62], fill=(0, 0, 0, 150))
    if note:
        d.text((14, 60), note, font=font(15), fill=(220, 220, 220))
    size = 24
    while size > 12 and d.textlength(title, font=font(size)) > img.width - 28:  # shrink long titles to fit
        size -= 1
    d.text((14, 8), title, font=font(size), fill=(255, 255, 255))
    d.text((14, 36), sub, font=font(17), fill=color)
    return np.asarray(img)


def rollout(policy, overrides, seed, max_frames, size, tint_ice):
    """Full episode; returns frames up to ~1.5 s after the Ant is finally steady (or max_frames if it never is)."""
    env = gym.make("AntRecovery-v0", render_mode="rgb_array", width=size[0], height=size[1], **overrides)
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    # free camera that follows the torso sideways but not vertically, so drops and slopes read correctly
    cam = u.mujoco_renderer._get_viewer("rgb_array").cam
    x0, y0, z0 = u.data.qpos[:3]
    ground = u.terrain.height(x0, y0)
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_FREE, -1
    cam.distance = 4.0 + 0.8 * max(0.0, z0 - ground - 0.5)
    cam.elevation, cam.azimuth = -14.0, 125.0
    look = np.array([x0, y0, ground + 0.45 + 0.4 * max(0.0, z0 - ground - 0.5)])

    def frame():
        x, y = u.data.qpos[:2]
        look[:2] += 0.15 * (np.array([x, y]) - look[:2])
        cam.lookat[:] = look
        return u.mujoco_renderer.render("rgb_array", camera_id=-1)

    if tint_ice:
        floor = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        u.model.geom_rgba[floor] = [0.55, 0.8, 1.0, 1.0]  # the floor's own rgba overrides its material
    frames, flags, done, info, t = [frame()], [], False, {}, 0
    rollout.setup = (u.scenario, u._push_step, u.data.qpos[2] - u.terrain.height(*u.data.qpos[:2]))
    while not done:
        obs, _, term, trunc, info = env.step(policy.act(obs))
        frames.append(frame())
        flags.append(info["upright_cos"] >= 0.9 and info["height"] >= 0.32)
        t += 1
        done = term or trunc
    env.close()
    if info["success"]:  # end 1.5 s after the final standing period starts (it then stands until 20 s)
        last_start = max(i for i in range(len(flags)) if flags[i] and (i == 0 or not flags[i - 1]))
        end = min(len(frames), max(last_start + 30, 60))
    else:
        end = max_frames
    return list(zip(frames[:end], [False] + flags[:end - 1])), info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=8.0, help="clip length for failures; successes end 1.5 s after the Ant is steady")
    p.add_argument("--tries", type=int, default=20, help="episodes to search for the wanted outcome")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--rates", nargs="*", default=["results/final_eval/stress_sac/stress.csv", "results/final_eval/eval_sac.csv"],
                   help="CSV files to look up each clip's measured success rate for the same policy")
    p.add_argument("--train-mix", type=int, default=0,
                   help="instead of the condition clips, record this many consecutive training-distribution episodes")
    a = p.parse_args()

    probe = gym.make("AntRecovery-v0")
    policy = evaluate.Policy(a.model, probe)
    label = f"{policy.kind.upper()} policy, deterministic, true-metric outcome shown"
    max_frames = int(a.seconds * 20)
    out_frames = []
    exp, model_seed = evaluate.describe(a.model)
    tables = [pd.read_csv(f) for f in a.rates if os.path.exists(f)]

    def rate(source):
        for t in tables:
            t = t[t.seed == model_seed]
            if source[0] == "eval":
                sel = t[t.scenario == source[1]] if "scenario" in t else t.iloc[:0]
            else:
                sel = t[(t.factor == source[0]) & (t.level == source[1])] if "factor" in t else t.iloc[:0]
            if len(sel):
                return f"This setting, same policy: {sel.success.mean() * 100:.0f}% success over {len(sel)} test episodes"
        return ""

    clips = CLIPS
    if a.train_mix:  # unfiltered: consecutive episodes, whatever the outcome
        clips = [(None, {}, None, None)] * a.train_mix
    for i, (title, overrides, want, source) in enumerate(clips):
        note = rate(source) if source else ""
        if want is None:
            frames, info = rollout(policy, overrides, a.seed + i, max_frames, (640, 480), False)
            terrain, push, h0 = rollout.setup
            title = f"Episode {i + 1}: {terrain} ground" + (f", dropped (torso starts {h0:.1f} m up)" if h0 > 0.6 else "")
            title += f", shoved at {push * 0.05:.1f} s" if 0 <= push < len(frames) else ""
            k = i
        for k in range(a.tries if want is not None else 0):
            frames, info = rollout(policy, overrides, a.seed + k, max_frames, (640, 480), "friction_scale" in overrides)
            if bool(info["success"]) == want:
                break
        outcome = (f"RECOVERED in {info['time_to_recover']:.1f} s, standing at 20 s" if info["success"]
                   else "FAILED: not standing at the end of the 20 s episode")
        color = (120, 255, 120) if info["success"] else (255, 120, 120)
        for frame, up in frames:
            out_frames.append(caption(frame, title, outcome + ("   [upright]" if up else ""), color, note))
        print(f"{title:42s} seed {a.seed + k}  success={info['success']:.0f}  t={info['time_to_recover']:.1f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    imageio.mimsave(a.out, out_frames, fps=20, macro_block_size=16)
    print("wrote", a.out, f"({len(out_frames) / 20:.0f} s) —", label)


if __name__ == "__main__":
    main()
