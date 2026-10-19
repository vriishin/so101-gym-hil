import gymnasium as gym
import numpy as np
import pytest
import torch

import gym_hil  # noqa: F401
from gym_hil.training.visual_policy import VisualActor
from gym_hil.training.visual_sac import VisualSACAgent
from gym_hil.wrappers.placement_curriculum import FullTaskCurriculum
from gym_hil.wrappers.transport_curriculum import CompositeGraspLiftTransportCurriculum
from gym_hil.wrappers.visual_policy_wrapper import (
    VISUAL_FEATURE_KEYS,
    VisualObservationAdapter,
    assert_no_privileged_leakage,
)


def make_visual_env():
    base = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="pixels_state",
        image_size=(64, 64),
        easy_reset=True,
        randomize_reset=False,
    )
    return VisualObservationAdapter(FullTaskCurriculum(CompositeGraspLiftTransportCurriculum(base)))


def test_visual_adapter_contains_only_images_and_robot_state() -> None:
    env = make_visual_env()
    try:
        observation, _ = env.reset(seed=0)
        assert set(observation) == set(VISUAL_FEATURE_KEYS)
        assert observation["observation.images.front"].shape == (64, 64, 3)
        assert observation["observation.images.wrist"].shape == (64, 64, 3)
        assert observation["observation.state"].shape == (12,)
        assert observation["observation.images.front"].dtype == np.uint8
        assert observation["observation.images.wrist"].dtype == np.uint8
        assert np.std(observation["observation.images.front"]) > 1.0
        assert np.std(observation["observation.images.wrist"]) > 1.0
        assert env.action_space.shape == (6,)
    finally:
        env.close()


@pytest.mark.parametrize(
    "leaked_key",
    ("cube_position", "target_position", "tcp_distance", "contact_flags", "task_phase"),
)
def test_visual_leakage_assertion_rejects_privileged_keys(leaked_key: str) -> None:
    observation = {
        "observation.images.front": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.state": np.zeros(12, dtype=np.float32),
        leaked_key: np.zeros(1),
    }
    with pytest.raises(AssertionError):
        assert_no_privileged_leakage(observation)


def test_visual_sac_has_six_actions_fresh_twin_critics_and_frozen_encoders() -> None:
    settings = {
        "actor_learning_rate": 1e-6,
        "critic_learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "entropy_coefficient": 0.001,
    }
    agent = VisualSACAgent(VisualActor(), settings)
    features = torch.zeros((3, 96))
    actions, log_probability = agent.sample_actions(features)
    q1, q2 = agent.critic(features, actions)
    assert actions.shape == (3, 6)
    assert log_probability.shape == (3, 1)
    assert q1.shape == q2.shape == (3, 1)
    assert agent.critic.q1 is not agent.critic.q2
    assert all(not parameter.requires_grad for parameter in agent.actor.front_encoder.parameters())
