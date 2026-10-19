import gymnasium as gym

import gym_hil  # noqa: F401
from gym_hil.controllers.so101_scripted import ScriptedPickPlacePolicy


def test_scripted_policy_succeeds_on_at_least_sixteen_of_twenty_easy_seeds() -> None:
    successes = 0
    for seed in range(20):
        env = gym.make(
            "gym_hil/SO101PickPlaceBase-v0",
            observation_mode="state",
            randomize_reset=True,
            episode_length_s=100,
            joint_delta_scale=0.06,
            gripper_delta_scale=0.03,
        )
        try:
            env.reset(seed=seed)
            successes += int(ScriptedPickPlacePolicy(env.unwrapped).run()["success"])
        finally:
            env.close()
    assert successes >= 16
