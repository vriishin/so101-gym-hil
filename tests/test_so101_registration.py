import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401


def test_so101_base_registration_and_headless_step() -> None:
    env = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        render_mode="rgb_array",
        observation_mode="pixels_state",
        randomize_reset=False,
    )
    try:
        observation, info = env.reset(seed=0)
        assert env.unwrapped.num_actuated_joints == 6
        assert env.observation_space.contains(observation)
        assert info["seed"] == 0

        observation, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert env.observation_space.contains(observation)
        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert info["executed_action"].shape == (6,)
        assert np.isfinite(observation["state"]).all()
        assert observation["pixels"]["front"].shape == (128, 128, 3)
        assert observation["pixels"]["front"].dtype == np.uint8
    finally:
        env.close()


def test_seeded_reset_is_deterministic() -> None:
    env = gym.make("gym_hil/SO101PickPlaceBase-v0", observation_mode="state")
    try:
        first_obs, first_info = env.reset(seed=42)
        second_obs, second_info = env.reset(seed=42)
        np.testing.assert_array_equal(first_obs["state"], second_obs["state"])
        np.testing.assert_array_equal(first_info["cube_position"], second_info["cube_position"])
    finally:
        env.close()
