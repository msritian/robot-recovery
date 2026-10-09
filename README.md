# robot-recovery: teaching the MuJoCo Ant to get back up after a fall

CS 839 programming assignment. The Ant starts fallen over and has to get back on its feet and stay standing.
I trained it with CleanRL's PPO and SAC. The report is `report.pdf`.

## What is where

| Path | What it is |
|---|---|
| `robot_recovery/ant_recovery.py` | The environment: start states, observation, reward and the success metric |
| `robot_recovery/terrain.py` | The bumpy and sloped ground |
| `robot_recovery/assets/ant_recovery.xml` | The modified MuJoCo model (standard Ant plus the terrain) |
| `robot_recovery/__init__.py` | Registers the environments. All final results use `AntRecoveryLowCost-v0` |
| `algos/` | My copies of CleanRL's PPO and SAC scripts, plus CleanRL's replay buffer (unchanged) |
| `patches/cleanrl.diff` | Every change I made to the CleanRL scripts |
| `scripts/` | Plots, tests, videos and the cluster launcher |
| `chtc/` | Files for running on the CHTC cluster. `chtc/batches/*.txt` list every run |
| `videos/` | `best_policy_sac_seed4.mp4` is the main video. `highlight_reel_sac_seed4.mp4` is its second part on its own. `recordvideo_sac_seed4/` has plain clips from Gymnasium's `RecordVideo` |
| `results/` | `final_plots/` has Figure 1. `final_eval/` has Table 2, Figures 2 and 3 and the raw test results. `final_sac/` and `final_ppo/` have the 10 trained models |

## Setup

```bash
pip install uv
uv sync
```

This installs Python 3.10, gymnasium 0.29.1, mujoco 2.3.3 and torch 2.4.1. Gymnasium 0.29.1 has no `Ant-v5`, so the
environment uses the `Ant-v4` model, as the assignment allows.

## Changes to CleanRL

1. Import `robot_recovery` at the top of both scripts, so CleanRL can find the environments.
2. Log the success metric at the end of every episode (`charts/success`, `charts/recovered`, `charts/time_to_recover`).
3. Save the trained model. For PPO this also saves the observation normalization, and I removed CleanRL's
   evaluation and upload step, which is not part of this repo.
4. A `--hidden-size` option for PPO (default 64, the same as CleanRL). I only used it in tuning.

No algorithm code was changed.

## Training

`N` is the seed. I used seeds 1 to 5 for both algorithms.

```bash
python algos/sac_continuous_action.py --env-id AntRecoveryLowCost-v0 --seed N --total-timesteps 1000000 --save-model
python algos/ppo_continuous_action.py --env-id AntRecoveryLowCost-v0 --seed N --total-timesteps 5000000 --save-model
```

Add `--capture-video` to record training episodes. This needs a display.

PPO tuning runs are listed in `chtc/batches/ppotune1.txt` and `ppotune2.txt`. Each line is
`<name> <algo> <env-id> <seed> <steps> [extra flags]` and runs as:

```bash
python algos/<algo>_continuous_action.py --env-id <env-id> --seed <seed> --total-timesteps <steps> --exp-name <name> --save-model [extra flags]
```

I ran all training on the CHTC cluster, one CPU core per run, with `scripts/chtc.sh submit <batch> chtc/batches/<batch>.txt`
and downloaded the results with `scripts/chtc.sh fetch <batch>`. The same commands work on a normal computer, but
the numbers can differ slightly between machines.

## Figures, tables and videos

The trained models are included, so everything below except the learning curves can be rerun without training.
The learning curves need the training logs, which are not in the zip because they are 52 MB.

```bash
# Figure 1: learning curves, full length and the first 1M steps
python scripts/plot.py --runs results/final_ppo/runs results/final_sac/runs --out results/final_plots --compare-at 1e6
python scripts/plot.py --runs results/final_ppo/runs results/final_sac/runs --out results/final_plots --xmax 1e6 --suffix _1M

# Table 1: PPO tuning
python scripts/summarize_sweep.py results/ppotune1/runs
python scripts/summarize_sweep.py results/ppotune2/runs

# Table 2: tests in 12 conditions, 100 episodes each, plus the do-nothing and random baselines
python scripts/evaluate.py --models "results/final_sac/runs/*/*.cleanrl_model" --episodes 100 --out results/final_eval/eval_sac.csv
python scripts/evaluate.py --models "results/final_ppo/runs/*/*.cleanrl_model" --episodes 100 --out results/final_eval/eval_ppo.csv
python scripts/evaluate.py --models zero random --episodes 100 --out results/final_eval/eval_baselines.csv

# Figure 2: stress tests, one factor at a time
python scripts/stress_test.py --models "results/final_sac/runs/*/*.cleanrl_model" --episodes 50 --out results/final_eval/stress_sac
python scripts/stress_test.py --models "results/final_ppo/runs/*/*.cleanrl_model" --episodes 50 --out results/final_eval/stress_ppo
python scripts/stress_test.py --plot-csv results/final_eval/stress_sac/stress.csv results/final_eval/stress_ppo/stress.csv --out results/final_eval/stress_combined.png

# Figure 3: the slope and bump episodes scored with the ground direction under the Ant
python scripts/terrain_metric_check.py --models "results/final_sac/runs/*/*.cleanrl_model" "results/final_ppo/runs/*/*.cleanrl_model" --out results/final_eval/terrain_metric_check.csv
python scripts/plot_terrain_check.py results/final_eval/terrain_metric_check.csv results/final_eval/terrain_check.png

# Videos of the best policy (SAC seed 4)
python scripts/make_reel.py --model results/final_sac/runs/AntRecoveryLowCost-v0__final_sac__4__*/final_sac.cleanrl_model --train-mix 8 --out videos/parts/typical_episodes_sac_seed4.mp4
python scripts/make_reel.py --model results/final_sac/runs/AntRecoveryLowCost-v0__final_sac__4__*/final_sac.cleanrl_model --out videos/highlight_reel_sac_seed4.mp4
python scripts/combine_videos.py --out videos/best_policy_sac_seed4.mp4 \
    --part "Part 1: 8 random episodes in a row (not hand-picked)" videos/parts/typical_episodes_sac_seed4.mp4 \
    --part "Part 2: harder conditions than training, and two failures" videos/highlight_reel_sac_seed4.mp4
python scripts/evaluate.py --models results/final_sac/runs/AntRecoveryLowCost-v0__final_sac__4__*/final_sac.cleanrl_model --scenarios train_mix --episodes 3 --video-dir videos/recordvideo_sac_seed4
```
