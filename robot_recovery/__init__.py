"""Registers the recovery environments with Gymnasium on import.

  AntRecovery-v0      training distribution: mild mix of flat / rough / slope, drops, light shoves, mild randomisation
  AntRecoveryFlat-v0  flat ground only, no shoves (ablation / sanity check)
  AntRecoveryHard-v0  held-out stress test: unseen rougher terrain, steeper slopes, bigger drops and shoves
  AntRecoveryLowCost-v0  training distribution with halved control / action-rate costs (reward-tuning variant)
"""

from dataclasses import replace

from gymnasium.envs.registration import register

from robot_recovery.ant_recovery import FLAT, HARD, TRAIN, AntRecoveryEnv, RecoveryConfig  # noqa: F401

LOW_COST = replace(TRAIN, w_ctrl=TRAIN.w_ctrl / 2, w_action_rate=TRAIN.w_action_rate / 2)

for _id, _cfg in [
    ("AntRecovery-v0", TRAIN),
    ("AntRecoveryFlat-v0", FLAT),
    ("AntRecoveryHard-v0", HARD),
    ("AntRecoveryLowCost-v0", LOW_COST),
]:
    register(
        id=_id,
        entry_point="robot_recovery.ant_recovery:AntRecoveryEnv",
        max_episode_steps=_cfg.episode_steps,
        kwargs={"config": _cfg},
    )
