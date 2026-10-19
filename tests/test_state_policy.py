import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401
from gym_hil.wrappers.grasp_curriculum import CompositeGraspLiftCurriculum
from gym_hil.wrappers.placement_curriculum import DescentCurriculum, FullTaskCurriculum
from gym_hil.wrappers.state_policy_wrapper import (
    PRIVILEGED_FEATURE_NAMES,
    FullTaskStateWrapper,
    PlacementStateWrapper,
    PrivilegedStateWrapper,
    UnifiedGraspLiftStateWrapper,
    UnifiedGraspLiftTransportStateWrapper,
)
from gym_hil.wrappers.transport_curriculum import CompositeGraspLiftTransportCurriculum


def test_privileged_state_wrapper_is_flat_finite_and_keeps_action_contract() -> None:
    base = gym.make("gym_hil/SO101PickPlaceBase-v0", observation_mode="state", reward_mode="potential")
    env = PrivilegedStateWrapper(base)
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (len(PRIVILEGED_FEATURE_NAMES),)
        assert env.observation_space.contains(observation)
        assert np.isfinite(observation).all()
        assert env.action_space.shape == (6,)
        next_observation, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
    finally:
        env.close()


def test_unified_grasp_lift_state_exposes_phase_and_aperture() -> None:
    base = gym.make("gym_hil/SO101PickPlaceBase-v0", observation_mode="state", easy_reset=True)
    curriculum = CompositeGraspLiftCurriculum(base)
    env = UnifiedGraspLiftStateWrapper(curriculum)
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (len(PRIVILEGED_FEATURE_NAMES) + 4,)
        assert np.isfinite(observation).all()
        assert observation[-2] == 1.0
        assert observation[-1] == 0.0
        assert env.action_space.shape == (6,)
    finally:
        env.close()


def test_unified_transport_state_is_phase_aware_and_keeps_six_actions() -> None:
    base = gym.make("gym_hil/SO101PickPlaceBase-v0", observation_mode="state", easy_reset=True)
    curriculum = CompositeGraspLiftTransportCurriculum(base)
    env = UnifiedGraspLiftTransportStateWrapper(curriculum)
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (35,)
        assert env.observation_space.contains(observation)
        np.testing.assert_array_equal(observation[-3:], (1.0, 0.0, 0.0))
        assert env.action_space.shape == (6,)
    finally:
        env.close()


def test_placement_and_full_task_state_shapes_are_finite() -> None:
    base = gym.make("gym_hil/SO101PickPlaceBase-v0", observation_mode="state", easy_reset=True)
    transport = CompositeGraspLiftTransportCurriculum(base)
    descent = DescentCurriculum(transport, transport_policy=None)
    placement = PlacementStateWrapper(descent)
    try:
        transport.reset(seed=0)
        descent._initialize_waypoint()
        placement_observation = placement.observation({})
        assert placement_observation.shape == (38,)
        assert np.isfinite(placement_observation).all()
    finally:
        placement.close()

    base = gym.make("gym_hil/SO101PickPlaceBase-v0", observation_mode="state", easy_reset=True)
    transport = CompositeGraspLiftTransportCurriculum(base)
    full = FullTaskStateWrapper(FullTaskCurriculum(transport))
    try:
        observation, _ = full.reset(seed=0)
        assert observation.shape == (42,)
        assert full.observation_space.contains(observation)
        assert np.isfinite(observation).all()
    finally:
        full.close()
