"""SO-101-specific EE teleoperation adapter and HIL factory."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from gym_hil.controllers.so101_scripted import solve_position_ik
from gym_hil.envs.so101_pick_place_env import SO101PickPlaceEnv
from gym_hil.wrappers.hil_wrappers import InputsControlWrapper


class SO101EEActionWrapper(gym.ActionWrapper):
    """Map XYZ-plus-gripper teleoperation commands to normalized six-joint deltas."""

    def __init__(self, env: gym.Env, ee_step_size_m: float = 0.015) -> None:
        super().__init__(env)
        if ee_step_size_m <= 0:
            raise ValueError("ee_step_size_m must be positive")
        self.ee_step_size_m = ee_step_size_m
        self.action_space = gym.spaces.Box(
            low=np.asarray((-1.0, -1.0, -1.0, 0.0), dtype=np.float32),
            high=np.asarray((1.0, 1.0, 1.0, 2.0), dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,):
            raise ValueError(f"Expected teleoperation action shape (4,), got {action.shape}")
        base: SO101PickPlaceEnv = self.unwrapped
        tcp_target = base.data.site("gripperframe").xpos + np.clip(action[:3], -1, 1) * self.ee_step_size_m
        target_joints = solve_position_ik(base, tcp_target, keep_gripper_vertical=False)
        joint_action = np.clip(
            (target_joints - base.joint_positions) / base.joint_delta_scales,
            -1,
            1,
        )
        joint_action[-1] = np.clip(action[-1] - 1.0, -1, 1)
        return joint_action.astype(np.float32)


def make_so101_hil_env(
    *,
    use_gamepad: bool,
    controller_config_path: str | None = None,
    auto_reset: bool = False,
    ee_step_size_m: float = 0.015,
    **kwargs,
) -> gym.Env:
    """Create an SO-101 environment with upstream intervention metadata handling."""

    env = SO101PickPlaceEnv(**kwargs)
    env = SO101EEActionWrapper(env, ee_step_size_m=ee_step_size_m)
    return InputsControlWrapper(
        env,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        use_gripper=True,
        auto_reset=auto_reset,
        use_gamepad=use_gamepad,
        controller_config_path=controller_config_path,
    )
