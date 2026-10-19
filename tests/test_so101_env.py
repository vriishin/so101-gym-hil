import gymnasium as gym
import numpy as np
import pytest

import gym_hil  # noqa: F401

ENV_ID = "gym_hil/SO101PickPlaceBase-v0"


def test_action_and_observation_spaces() -> None:
    env = gym.make(ENV_ID, observation_mode="pixels_state", image_size=(64, 80))
    try:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        assert observation["pixels"]["front"].shape == (64, 80, 3)
        assert observation["pixels"]["wrist"].shape == (64, 80, 3)
        assert env.action_space.contains(env.action_space.sample())
        privileged = env.unwrapped.get_privileged_state()
        assert {"cube_position", "target_position", "tcp_position"} <= privileged.keys()
        assert "cube_position" not in observation
    finally:
        env.close()


def test_twenty_seeded_resets_are_deterministic_valid_and_variable() -> None:
    env = gym.make(ENV_ID, observation_mode="state", randomize_reset=True)
    try:
        cube_positions = []
        for seed in range(20):
            first_obs, first_info = env.reset(seed=seed)
            second_obs, second_info = env.reset(seed=seed)
            np.testing.assert_array_equal(first_obs["state"], second_obs["state"])
            np.testing.assert_array_equal(first_info["cube_position"], second_info["cube_position"])
            np.testing.assert_array_equal(first_info["target_position"], second_info["target_position"])

            cube_xy = first_info["cube_position"][:2]
            target_xy = first_info["target_position"][:2]
            bounds = env.unwrapped.cube_sampling_bounds
            target_bounds = env.unwrapped.target_sampling_bounds
            assert np.all(cube_xy >= bounds[0]) and np.all(cube_xy <= bounds[1])
            assert np.all(target_xy >= target_bounds[0]) and np.all(target_xy <= target_bounds[1])
            assert np.linalg.norm(cube_xy - target_xy) >= 0.10
            assert np.isfinite(first_obs["state"]).all()
            cube_positions.append(cube_xy)
        assert np.unique(np.asarray(cube_positions), axis=0).shape[0] > 1
    finally:
        env.close()


def test_action_clipping_and_joint_targets_remain_in_limits() -> None:
    env = gym.make(ENV_ID, observation_mode="state", randomize_reset=False)
    try:
        env.reset(seed=0)
        for _ in range(100):
            _, _, _, _, info = env.step(np.full(6, 50.0, dtype=np.float32))
            targets = env.unwrapped.data.ctrl[env.unwrapped._actuator_ids]
            limits = env.unwrapped.joint_control_ranges
            assert np.all(targets >= limits[:, 0]) and np.all(targets <= limits[:, 1])
            assert np.all(np.abs(info["executed_action"]) <= 1.0 + 1e-6)
    finally:
        env.close()


def test_one_thousand_random_steps_are_finite() -> None:
    env = gym.make(ENV_ID, observation_mode="state", randomize_reset=True)
    rng = np.random.default_rng(2026)
    try:
        env.reset(seed=0)
        for step in range(1000):
            observation, reward, terminated, truncated, _ = env.step(
                rng.uniform(-1, 1, size=6).astype(np.float32)
            )
            assert np.isfinite(observation["state"]).all()
            assert np.isfinite(env.unwrapped.data.qpos).all()
            assert np.isfinite(env.unwrapped.data.qvel).all()
            assert np.isfinite(reward)
            if terminated or truncated:
                env.reset(seed=step + 1)
    finally:
        env.close()


def test_internal_time_limit_is_truncation_only() -> None:
    env = gym.make(ENV_ID, observation_mode="state", episode_length_s=0.2, control_hz=10)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert not terminated and not truncated
        _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert not terminated and truncated
        assert info["termination_reason"] == "time_limit"
    finally:
        env.close()


def test_invalid_configuration_and_action_shape_fail_clearly() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        gym.make(ENV_ID, control_hz=12, physics_hz=500)
    env = gym.make(ENV_ID, observation_mode="state")
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError, match="Expected action shape"):
            env.step(np.zeros(5, dtype=np.float32))
    finally:
        env.close()


def test_repeated_create_render_close() -> None:
    for seed in range(3):
        env = gym.make(ENV_ID, observation_mode="pixels_state", image_size=(32, 32))
        observation, _ = env.reset(seed=seed)
        assert observation["pixels"]["front"].dtype == np.uint8
        env.close()


def test_small_initial_joint_perturbation_is_applied_and_validated() -> None:
    env = gym.make(ENV_ID, observation_mode="state", easy_reset=True, randomize_reset=False)
    try:
        _, baseline = env.reset(seed=4)
        perturbation = np.asarray((0.01, -0.01, 0.005, -0.005, 0.002, 0.0))
        _, perturbed = env.reset(seed=4, options={"initial_joint_perturbation": perturbation})
        np.testing.assert_allclose(
            perturbed["initial_joint_positions"] - baseline["initial_joint_positions"],
            perturbation,
            atol=1e-9,
        )
        with pytest.raises(ValueError, match="shape"):
            env.reset(options={"initial_joint_perturbation": np.zeros(5)})
        with pytest.raises(ValueError, match="within"):
            env.reset(options={"initial_joint_perturbation": np.full(6, 0.1)})
    finally:
        env.close()
