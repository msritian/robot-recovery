"""AntRecovery: the MuJoCo Ant must get back on its feet after a fall and stay there.

Each episode the Ant starts tipped over (120-180 deg by default, i.e. on its side-to-back),
possibly dropped from a height with spin, on flat ground, rough ground or a slope, and may be
shoved mid-episode. Friction, payload and motor strength are randomised. It is rewarded for
getting upright, standing tall and being still; the true metric (`success`) is whether it is
standing upright and steady for the last 2.5 s of the episode.
"""

import os
import tempfile
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from robot_recovery.terrain import Terrain

XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "ant_recovery.xml")
FLAT_SPAWN_HALF = 2.0  # flat spawns land in [-2, 2]^2 around the origin, away from the terrain strip (x >= 4)
RUBBLE_CENTER = (-10.0, 0.0)  # test-only debris field, away from all training spawn areas
RUBBLE_RADIUS = 2.5
RUBBLE_SPAWN_RADIUS = 1.2

# leg index -> actuator indices (actuator order in the XML: hip_4, ankle_4, hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3)
LEG_ACTUATORS = {"front_left": (2, 3), "front_right": (4, 5), "back_left": (6, 7), "back_right": (0, 1)}


@dataclass
class RecoveryConfig:
    # scenario mix
    scenario_probs: dict = field(default_factory=lambda: {"flat": 0.5, "rough": 0.25, "slope": 0.25})
    rough_amps: tuple = (0.03, 0.05, 0.07)  # bump amplitude per rough tile (m)
    slope_degs: tuple = (5.0, 10.0, 15.0)  # face angle per pyramid tile
    terrain_seed: int = 0
    rubble_count: int = 70  # debris pieces, only built when "rubble" is in scenario_probs (evaluation only)
    rubble_seed: int = 0
    # initial state
    tilt_deg: tuple = (120.0, 180.0)  # angle between torso up-axis and world up
    drop_prob: float = 0.3
    drop_height: tuple = (0.3, 1.0)  # extra height above the ground (m)
    drop_spin: float = 3.0  # max initial angular speed per axis when dropped (rad/s)
    # disturbances / domain randomisation
    push_prob: float = 0.5  # probability of one shove per episode
    push_dv: tuple = (0.3, 1.0)  # horizontal velocity change of the shove (m/s)
    push_spin: float = 2.0  # max angular velocity change of the shove per axis (rad/s)
    friction_scale: tuple = (0.8, 1.2)
    torso_mass_scale: tuple = (0.9, 1.3)  # payload
    motor_scale: tuple = (0.9, 1.1)  # actuator strength / wear
    leg_fault_prob: float = 0.0  # probability that one random leg is disabled for the episode
    leg_fault_strength: float = 0.0  # torque multiplier of the faulty leg (0 = limp)
    leg_fault_count: int = 1  # how many legs fail when a fault occurs (evaluation stress tests use 2)
    # episode
    episode_steps: int = 400  # 20 s at dt = 0.05
    # reward weights
    w_upright: float = 1.0
    w_height: float = 0.5
    w_still: float = 0.5
    w_ctrl: float = 0.01
    w_action_rate: float = 0.02
    # true-metric thresholds
    success_cos: float = 0.9  # torso up-axis . terrain normal (~26 deg)
    success_height: float = 0.32  # torso centre above terrain (m); lying on the torso gives ~0.25
    success_angvel: float = 1.0  # rad/s
    success_hold_steps: int = 50  # last 2.5 s
    recovered_hold_steps: int = 20  # 1 s of continuous standing counts as "recovered" (for time_to_recover)


TRAIN = RecoveryConfig()
FLAT = replace(TRAIN, scenario_probs={"flat": 1.0}, push_prob=0.0)
HARD = replace(
    TRAIN,
    rough_amps=(0.08, 0.10, 0.12),
    slope_degs=(15.0, 20.0, 25.0),
    terrain_seed=1,  # unseen terrain
    tilt_deg=(90.0, 180.0),
    drop_prob=0.5,
    drop_height=(0.5, 1.5),
    drop_spin=5.0,
    push_prob=1.0,
    push_dv=(1.0, 2.0),
    push_spin=4.0,
    friction_scale=(0.4, 1.5),
    torso_mass_scale=(1.0, 1.8),
    motor_scale=(0.7, 1.1),
)


def _random_tilt_quat(rng, tilt_range):
    """Random yaw, then tilt by an angle in tilt_range about a random horizontal axis."""
    yaw = rng.uniform(-np.pi, np.pi)
    q_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    tilt = np.deg2rad(rng.uniform(*tilt_range))
    phi = rng.uniform(-np.pi, np.pi)
    q_tilt = np.array([np.cos(tilt / 2), np.sin(tilt / 2) * np.cos(phi), np.sin(tilt / 2) * np.sin(phi), 0.0])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_tilt, q_yaw)
    return q


class AntRecoveryEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 20}

    def __init__(self, config: RecoveryConfig = TRAIN, render_mode=None, width=480, height=480, **overrides):
        utils.EzPickle.__init__(self, config, render_mode, width, height, **overrides)
        self.cfg = replace(config, **overrides)
        self.terrain = Terrain(self.cfg.rough_amps, self.cfg.slope_degs, self.cfg.terrain_seed)
        obs_space = Box(-np.inf, np.inf, shape=(33,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            self._model_path(),
            frame_skip=5,
            observation_space=obs_space,
            render_mode=render_mode,
            width=width,
            height=height,
            default_camera_config={"trackbodyid": 1, "distance": 3.5, "elevation": -20.0},
        )
        self.model.hfield_data[:] = self.terrain.hfield_data()

        m = self.model
        self._torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._robot_geoms = np.where(m.geom_bodyid != 0)[0]
        self._base_friction = m.geom_friction[:, 0].copy()
        self._base_torso_mass = m.body_mass[self._torso]
        self._base_torso_inertia = m.body_inertia[self._torso].copy()
        self._scenarios = list(self.cfg.scenario_probs)
        self._scenario_p = np.array([self.cfg.scenario_probs[s] for s in self._scenarios], dtype=float)
        self._scenario_p /= self._scenario_p.sum()

        self._motor = np.ones(8)
        self._prev_action = np.zeros(8)
        self._normal = np.array([0.0, 0.0, 1.0])
        self._t = 0
        self._hold = 0
        self._time_to_recover = None
        self._push_step = -1

    # ------------------------------------------------------------------ helpers
    def _model_path(self):
        """The standard model, or (for rubble tests) a copy with static debris blocks added."""
        if "rubble" not in self.cfg.scenario_probs:
            return XML_PATH
        rng = np.random.default_rng(self.cfg.rubble_seed)
        geoms, self._rubble_top = [], 0.0
        for i in range(self.cfg.rubble_count):
            r, a = RUBBLE_RADIUS * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
            x, y = RUBBLE_CENTER[0] + r * np.cos(a), RUBBLE_CENTER[1] + r * np.sin(a)
            if rng.random() < 0.25:  # flat slab (broken concrete)
                hx, hy, hz = rng.uniform(0.12, 0.30), rng.uniform(0.08, 0.20), rng.uniform(0.02, 0.05)
            else:  # brick / rock
                hx, hy, hz = rng.uniform(0.04, 0.12, size=3)
            tilt = rng.uniform(-20, 20, size=2)  # degrees (the base XML compiles angles in degrees)
            euler = f"{tilt[0]:.1f} {tilt[1]:.1f} {rng.uniform(0, 360):.1f}"
            z = hz * rng.uniform(0.3, 1.0)  # partly sunk into the ground
            self._rubble_top = max(self._rubble_top, z + np.sqrt(hx**2 + hy**2 + hz**2))
            grey = rng.uniform(0.35, 0.6)
            geoms.append(
                f'<geom name="rubble_{i}" type="box" size="{hx:.3f} {hy:.3f} {hz:.3f}" pos="{x:.3f} {y:.3f} {z:.3f}" '
                f'euler="{euler}" conaffinity="1" condim="3" rgba="{grey:.2f} {grey * 0.92:.2f} {grey * 0.85:.2f} 1"/>'
            )
        xml = open(XML_PATH).read()
        xml = xml.replace("</worldbody>", "\n    " + "\n    ".join(geoms) + "\n  </worldbody>")
        path = os.path.join(tempfile.gettempdir(), f"ant_rubble_{self.cfg.rubble_seed}_{self.cfg.rubble_count}_{os.getpid()}.xml")
        with open(path, "w") as f:
            f.write(xml)
        return path

    def _torso_rot(self):
        return self.data.xmat[self._torso].reshape(3, 3)

    def _height_above_terrain(self):
        x, y, z = self.data.qpos[0:3]
        return z - self.terrain.height(x, y)

    def _standing(self, upright_cos, height, angspeed):
        c = self.cfg
        return upright_cos >= c.success_cos and height >= c.success_height and angspeed <= c.success_angvel

    def _get_obs(self):
        rot = self._torso_rot()
        gravity_body = rot.T @ np.array([0.0, 0.0, -1.0])  # what an IMU reports
        angvel_body = self.data.qvel[3:6]  # free-joint angular velocity is already in the body frame
        linvel_body = rot.T @ self.data.qvel[0:3]
        joint_pos = self.data.qpos[7:15]
        joint_vel = 0.1 * self.data.qvel[6:14]
        return np.concatenate([gravity_body, angvel_body, linvel_body, joint_pos, joint_vel, self._prev_action])

    # ------------------------------------------------------------------ reset
    def reset_model(self):
        c, rng, m = self.cfg, self.np_random, self.model

        # domain randomisation (MuJoCo combines contact friction with max(), so scale robot and ground together)
        m.geom_friction[:, 0] = self._base_friction * rng.uniform(*c.friction_scale)
        s = rng.uniform(*c.torso_mass_scale)
        m.body_mass[self._torso] = self._base_torso_mass * s
        m.body_inertia[self._torso] = self._base_torso_inertia * s
        self._motor = np.full(8, rng.uniform(*c.motor_scale))
        self._fault_legs = []
        if rng.random() < c.leg_fault_prob:
            self._fault_legs = list(rng.choice(list(LEG_ACTUATORS), size=c.leg_fault_count, replace=False))
            for leg in self._fault_legs:
                self._motor[list(LEG_ACTUATORS[leg])] *= c.leg_fault_strength

        # where: flat ground, rough tile or slope tile
        self.scenario = self._scenarios[rng.choice(len(self._scenarios), p=self._scenario_p)]
        if self.scenario == "flat":
            x, y = rng.uniform(-FLAT_SPAWN_HALF, FLAT_SPAWN_HALF, size=2)
        elif self.scenario == "rubble":
            r, a = RUBBLE_SPAWN_RADIUS * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
            x, y = RUBBLE_CENTER[0] + r * np.cos(a), RUBBLE_CENTER[1] + r * np.sin(a)
        else:
            x, y, _ = self.terrain.sample_spawn(self.scenario, rng)
        self._normal = self.terrain.normal(x, y)

        # pose: tipped over, random joint angles, placed just above the ground (or dropped)
        qpos = self.init_qpos.copy()
        qvel = np.zeros(m.nv)
        qpos[3:7] = _random_tilt_quat(rng, c.tilt_deg)
        lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
        qpos[7:15] = rng.uniform(lo, hi)
        qpos[0:3] = [x, y, 0.0]
        self.set_state(qpos, qvel)
        lowest = self._lowest_point()
        ground = self._rubble_top if self.scenario == "rubble" else self.terrain.max_height_near(x, y, radius=1.2)
        qpos[2] = ground - lowest + 0.02
        if rng.random() < c.drop_prob:
            qpos[2] += rng.uniform(*c.drop_height)
            qvel[3:6] = rng.uniform(-c.drop_spin, c.drop_spin, size=3)
        qvel[6:14] = 0.1 * rng.standard_normal(8)
        self.set_state(qpos, qvel)

        self._push_step = rng.integers(40, c.episode_steps - 100) if rng.random() < c.push_prob else -1
        self._prev_action = np.zeros(8)
        self._t = 0
        self._hold = 0
        self._time_to_recover = None
        return self._get_obs()

    def _lowest_point(self):
        """Lowest z of any robot geom surface, relative to the torso origin (geoms are spheres / capsules)."""
        d, m = self.data, self.model
        z0 = d.qpos[2]
        low = np.inf
        for g in self._robot_geoms:
            r = m.geom_size[g, 0]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                axis_z = d.geom_xmat[g].reshape(3, 3)[2, 2]
                low = min(low, d.geom_xpos[g, 2] - abs(axis_z) * m.geom_size[g, 1] - r)
            else:
                low = min(low, d.geom_xpos[g, 2] - r)
        return low - z0

    # ------------------------------------------------------------------ step
    def step(self, action):
        c = self.cfg
        action = np.clip(action, -1.0, 1.0)
        if self._t == self._push_step:
            ang = self.np_random.uniform(-np.pi, np.pi)
            dv = self.np_random.uniform(*c.push_dv)
            qvel = self.data.qvel.copy()
            qvel[0:2] += dv * np.array([np.cos(ang), np.sin(ang)])
            qvel[3:6] += self.np_random.uniform(-c.push_spin, c.push_spin, size=3)
            self.set_state(self.data.qpos.copy(), qvel)

        self.do_simulation(action * self._motor, self.frame_skip)
        self._t += 1

        up = self._torso_rot()[:, 2]
        upright_cos = float(up @ self._normal)
        height = self._height_above_terrain()
        angspeed = float(np.linalg.norm(self.data.qvel[3:6]))

        # ---- reward (shaping; NOT the success metric)
        r_upright = ((upright_cos + 1.0) / 2.0) ** 2
        r_height = np.clip((height - 0.25) / 0.30, 0.0, 1.0) * float(upright_cos > 0.7)
        r_still = float(upright_cos > c.success_cos) * np.exp(-0.5 * angspeed**2)
        ctrl_cost = c.w_ctrl * float(np.sum(action**2))
        rate_cost = c.w_action_rate * float(np.sum((action - self._prev_action) ** 2))
        reward = c.w_upright * r_upright + c.w_height * r_height + c.w_still * r_still - ctrl_cost - rate_cost
        self._prev_action = action

        # ---- true metric bookkeeping
        self._hold = self._hold + 1 if self._standing(upright_cos, height, angspeed) else 0
        if self._time_to_recover is None and self._hold >= c.recovered_hold_steps:
            self._time_to_recover = (self._t - c.recovered_hold_steps) * self.dt
        terminated = not np.isfinite(self.state_vector()).all()

        info = {
            "success": float(self._hold >= c.success_hold_steps),  # read at the final step of the episode
            "time_to_recover": self._time_to_recover if self._time_to_recover is not None else c.episode_steps * self.dt,
            "recovered": float(self._time_to_recover is not None),
            "upright_cos": upright_cos,
            "height": height,
            "scenario": self.scenario,
            "r_upright": r_upright,
            "r_height": r_height,
            "r_still": r_still,
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info
