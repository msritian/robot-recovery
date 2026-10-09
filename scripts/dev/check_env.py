import time, numpy as np, mujoco, gymnasium as gym, robot_recovery
env = gym.make("AntRecovery-v0"); u = env.unwrapped
# 1) heightfield orientation: ray-cast straight down vs Terrain.height (bilinear)
rng = np.random.default_rng(0); errs = []
for _ in range(300):
    x, y = rng.uniform(4.2, 27.8), rng.uniform(-1.9, 1.9)
    gid = np.array([-1], dtype=np.int32)
    d = mujoco.mj_ray(u.model, u.data, np.array([x, y, 5.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
    errs.append(abs((5.0 - d) - u.terrain.height(x, y)))
print(f"hfield max |ray - height()| = {max(errs):.4f} m (terrain max {u.terrain.heights.max():.2f} m)")
# 2) episodes with random actions: metric + speed
obs, _ = env.reset(seed=0); print("obs shape", obs.shape)
t0 = time.time(); n = 0; stats = {}
for ep in range(30):
    obs, _ = env.reset(seed=ep); ret = 0; done = False
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample()); ret += r; n += 1; done = term or trunc
    stats.setdefault(info["scenario"], []).append((ret, info["success"], info["recovered"]))
print(f"steps/s {n/(time.time()-t0):.0f}")
for k, v in stats.items():
    v = np.array(v); print(f"  random policy {k:5s} n={len(v):2d} return {v[:,0].mean():6.1f} success {v[:,1].mean():.2f} recovered {v[:,2].mean():.2f}")
# 3) zero-action baseline
succ = []
for ep in range(30):
    env.reset(seed=100+ep); done = False
    while not done:
        _, r, term, trunc, info = env.step(np.zeros(8)); done = term or trunc
    succ.append(info["success"])
print("zero policy success", np.mean(succ))
