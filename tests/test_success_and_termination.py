import gymnasium as gym
import mujoco
import numpy as np
import pytest

import gym_hil  # noqa: F401
from gym_hil.rewards import TaskConfig

ENV_ID = "gym_hil/SO101PickPlaceBase-v0"


def test_out_of_bounds_failure_terminates() -> None:
    env = gym.make(ENV_ID, observation_mode="state")
    try:
        env.reset(seed=0)
        cube_joint = env.unwrapped.model.joint("cube_joint")
        address = env.unwrapped.model.jnt_qposadr[cube_joint.id]
        env.unwrapped.data.qpos[address : address + 3] = (0.60, 0.0, 0.02)
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert terminated and not truncated
        assert info["termination_reason"] == "out_of_bounds"
        assert not info["is_success"]
    finally:
        env.close()


def test_success_requires_release_and_stable_hold() -> None:
    task_config = TaskConfig(stable_hold_s=0.3, control_hz=10, invalid_collision_limit=100)
    env = gym.make(
        ENV_ID,
        observation_mode="state",
        randomize_reset=False,
        task_config=task_config,
        control_hz=10,
    )
    try:
        env.reset(seed=0)
        target = env.unwrapped.data.body("target").xpos.copy()
        cube_joint = env.unwrapped.model.joint("cube_joint")
        cube_address = env.unwrapped.model.jnt_qposadr[cube_joint.id]
        env.unwrapped.data.qpos[cube_address : cube_address + 3] = (
            target[0],
            target[1],
            target[2] + task_config.cube_half_extent_m,
        )
        env.unwrapped.data.qvel[env.unwrapped.model.jnt_dofadr[cube_joint.id] :][:6] = 0
        gripper_address = env.unwrapped._qpos_addresses[-1]
        env.unwrapped.data.qpos[gripper_address] = 1.2
        env.unwrapped.data.ctrl[env.unwrapped._actuator_ids[-1]] = 1.2
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

        for expected_stable_steps in (1, 2):
            _, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
            assert info["is_released"]
            assert info["stable_steps"] == expected_stable_steps
            assert not info["is_stable"]
            assert not terminated and not truncated
            assert reward == 0.0

        _, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert terminated and not truncated
        assert info["is_success"] and info["is_stable"]
        assert info["termination_reason"] == "success"
        assert reward == pytest.approx(1.0)
    finally:
        env.close()


def test_recoverable_table_drop_does_not_terminate() -> None:
    env = gym.make(ENV_ID, observation_mode="state", randomize_reset=False)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert not terminated and not truncated
        assert info["termination_reason"] is None
    finally:
        env.close()
