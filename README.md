# robot-recovery: learning fall recovery for the MuJoCo Ant

CS 839 programming assignment. Custom Gymnasium environment in which the MuJoCo Ant must get back on its feet
after a fall and stay standing, trained with CleanRL's PPO and SAC.

## Layout

| Path | Contents |
|---|---|
| `robot_recovery/ant_recovery.py` | the environment (MDP: reset distribution, observation, reward, success metric) |
| `robot_recovery/terrain.py` | heightfield terrain (bumpy tiles, pyramid slopes) |
| `robot_recovery/assets/ant_recovery.xml` | modified MuJoCo XML: unchanged Gymnasium `ant.xml` body + heightfield |
| `robot_recovery/__init__.py` | registers `AntRecovery-v0`, `AntRecoveryLowCost-v0` (used for all final results), `AntRecoveryFlat-v0`, `AntRecoveryHard-v0` |
| `algos/` | my copies of CleanRL `ppo_continuous_action.py` / `sac_continuous_action.py` (upstream commit in `algos/UPSTREAM_COMMIT`) and CleanRL's unchanged replay buffer |
| `patches/cleanrl.diff` | every change to the CleanRL files (all marked `# [robot-recovery]`) |
| `scripts/` | plotting, evaluation, stress tests, videos, CHTC launcher |
| `chtc/` | container build and HTCondor job files; `chtc/batches/*.txt` are the exact job lists |

## Setup

```bash
pip install uv
uv sync            # Python 3.10, gymnasium 0.29.1, mujoco 2.3.3, torch 2.4.1 (see pyproject.toml / uv.lock)
```

Gymnasium 0.29.1 has no `Ant-v5`, so the environment builds on the `Ant-v4` assets (the fallback the assignment allows).

## Changes to CleanRL (`patches/cleanrl.diff`)

1. `import robot_recovery` at the top of both scripts (registers the environments).
2. When an episode ends, also log the true metric: `charts/success`, `charts/recovered`, `charts/time_to_recover`.
3. `--save-model`: SAC gained the option (saves the actor); PPO now also saves its observation-normalisation
   statistics and no longer calls CleanRL's eval/upload helpers (not part of this repo).
4. PPO only: `--hidden-size` (default 64 = CleanRL's architecture); used only in tuning batch 2.

No algorithmic code was changed.

## Reproducing the results

Every run below is one command; `N` is the seed. Seeds 1-5 are paired across algorithms.

**Final runs** (`chtc/batches/final_sac.txt`, `chtc/batches/final_ppo.txt`):

```bash
python algos/sac_continuous_action.py --env-id AntRecoveryLowCost-v0 --seed N --total-timesteps 1000000 --save-model
python algos/ppo_continuous_action.py --env-id AntRecoveryLowCost-v0 --seed N --total-timesteps 5000000 --save-model
```

Add `--capture-video` to record training episodes locally (needs an OpenGL context).

**PPO tuning** (`chtc/batches/ppotune1.txt`, `chtc/batches/ppotune2.txt`): each line is
`<tag> <algo> <env-id> <seed> <timesteps> [extra CleanRL flags]`, run as
`python algos/<algo>_continuous_action.py --env-id <env-id> --seed <seed> --total-timesteps <timesteps> --exp-name <tag> --save-model [extra flags]`.

I ran these on CHTC (one CPU core per run) with `scripts/chtc.sh submit <batch> chtc/batches/<batch>.txt`, which
ships the committed code to `chtc/run_job.sh` inside the container built by `chtc/build_container.sh`;
`scripts/chtc.sh fetch <batch>` downloads `results/<batch>/runs/`. Running the commands locally gives the same runs.

**Figures, evaluation and videos** (after placing runs in `results/final_{ppo,sac}/runs/`):

```bash
# learning curves (5 seeds, t-95% CI, per-seed lines); full axis and the equal-budget 0-1M view
python scripts/plot.py --runs results/final_ppo/runs results/final_sac/runs --out results/final_plots --compare-at 1e6
python scripts/plot.py --runs results/final_ppo/runs results/final_sac/runs --out results/final_plots --xmax 1e6 --suffix _1M

# per-condition evaluation (100 episodes each, deterministic policies, identical episode seeds) and baselines
python scripts/evaluate.py --models "results/final_sac/runs/*/*.cleanrl_model" --episodes 100 --out results/final_eval/eval_sac.csv
python scripts/evaluate.py --models "results/final_ppo/runs/*/*.cleanrl_model" --episodes 100 --out results/final_eval/eval_ppo.csv
python scripts/evaluate.py --models zero random --episodes 100 --out results/final_eval/eval_baselines.csv

# stress ladders (one factor at a time)
python scripts/stress_test.py --models "results/final_sac/runs/*/*.cleanrl_model" --episodes 50 --out results/final_eval/stress_sac
python scripts/stress_test.py --models "results/final_ppo/runs/*/*.cleanrl_model" --episodes 50 --out results/final_eval/stress_ppo

# videos of the best policy (SAC seed 4: highest mean success over all evaluation conditions)
python scripts/make_reel.py --model results/final_sac/runs/AntRecoveryLowCost-v0__final_sac__4__*/final_sac.cleanrl_model --train-mix 8 --out videos/best_policy_sac_seed4.mp4
python scripts/make_reel.py --model results/final_sac/runs/AntRecoveryLowCost-v0__final_sac__4__*/final_sac.cleanrl_model --out videos/highlight_reel_sac_seed4.mp4
```
