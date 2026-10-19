from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

import gym_hil  # noqa: F401
from gym_hil.rewards import PickPlaceReward, RewardConfig
from tests.test_task_state import state

ZERO_ACTION = np.zeros(6, dtype=np.float32)
EXPECTED_INFO_KEYS = {
    "is_success",
    "task_stage",
    "reward_mode",
    "reward_terms",
    "reach_distance",
    "target_distance",
    "object_lift_height",
    "is_grasped",
    "is_lifted",
    "object_in_target",
    "is_released",
    "is_stable",
    "stable_steps",
    "invalid_collision_count",
    "executed_action",
    "termination_reason",
}


@pytest.mark.parametrize("mode", ["sparse", "naive_dense", "staged_dense", "potential"])
def test_all_reward_modes_are_finite_scalars(mode: str) -> None:
    reward = PickPlaceReward(RewardConfig(mode=mode), stable_steps_required=5)
    before = state()
    after = replace(before, reach_distance=0.01)
    reward.reset(before)
    result = reward.compute(before, after, ZERO_ACTION, ZERO_ACTION, False, False)
    assert isinstance(result.total, float)
    assert np.isfinite(result.total)
    assert all(np.isfinite(value) for value in result.terms.values())


def test_sparse_is_strict_and_pays_success_only() -> None:
    reward = PickPlaceReward(RewardConfig(mode="sparse"), stable_steps_required=5)
    before = state()
    reward.reset(before)
    assert (
        reward.compute(before, replace(before, is_grasped=True), ZERO_ACTION, ZERO_ACTION, False, False).total
        == 0
    )
    successful = replace(before, is_released=True, is_stable=True, success=True)
    result = reward.compute(before, successful, ZERO_ACTION, ZERO_ACTION, True, False)
    assert result.total == 1.0
    assert result.terms == {"success": 1.0}


def test_naive_transport_is_ungated_but_staged_transport_is_gated() -> None:
    before = state(target_distance=0.30)
    after = replace(before, target_distance=0.02)

    naive = PickPlaceReward(RewardConfig(mode="naive_dense"), stable_steps_required=5)
    naive.reset(before)
    naive_result = naive.compute(before, after, ZERO_ACTION, ZERO_ACTION, False, False)
    assert naive_result.terms["transport"] > 0

    staged = PickPlaceReward(RewardConfig(mode="staged_dense"), stable_steps_required=5)
    staged.reset(before)
    staged_result = staged.compute(before, after, ZERO_ACTION, ZERO_ACTION, False, False)
    assert staged_result.terms["transport"] == 0

    potential = PickPlaceReward(RewardConfig(mode="potential"), stable_steps_required=5)
    assert potential.potential(before) == pytest.approx(potential.potential(after))


def test_stability_is_inactive_outside_target() -> None:
    reward = PickPlaceReward(RewardConfig(mode="potential"), stable_steps_required=5)
    outside = state(stable_steps=0, object_in_target=False)
    outside_with_counter = replace(outside, stable_steps=5)
    assert reward.potential(outside) == pytest.approx(reward.potential(outside_with_counter))


def test_potential_closed_cycle_has_zero_shaping_when_gamma_one() -> None:
    reward = PickPlaceReward(RewardConfig(mode="potential", gamma=1.0), stable_steps_required=5)
    first = state(reach_distance=0.20)
    second = replace(first, reach_distance=0.02)
    reward.reset(first)
    forward = reward.compute(first, second, ZERO_ACTION, ZERO_ACTION, False, False)
    backward = reward.compute(second, first, ZERO_ACTION, ZERO_ACTION, False, False)
    assert forward.terms["potential_shaping"] + backward.terms["potential_shaping"] == pytest.approx(0.0)


def test_penalty_signs_and_terminal_potential() -> None:
    config = RewardConfig(mode="potential")
    reward = PickPlaceReward(config, stable_steps_required=5)
    before = state(is_grasped=True)
    failed = replace(before, is_grasped=False, out_of_bounds=True)
    reward.reset(before)
    result = reward.compute(before, failed, np.ones(6), ZERO_ACTION, True, False)
    assert result.terms["action_penalty"] < 0
    assert result.terms["jerk_penalty"] < 0
    assert result.terms["drop_penalty"] < 0
    assert result.terms["out_of_bounds_penalty"] < 0
    assert result.potential_current == 0.0


def test_successful_trajectory_outreturns_equivalent_failure() -> None:
    reward = PickPlaceReward(RewardConfig(mode="sparse"), stable_steps_required=5)
    before = state()
    reward.reset(before)
    failed = reward.compute(before, before, ZERO_ACTION, ZERO_ACTION, True, False).total
    succeeded = reward.compute(
        before,
        replace(before, is_released=True, is_stable=True, success=True),
        ZERO_ACTION,
        ZERO_ACTION,
        True,
        False,
    ).total
    assert succeeded > failed


@pytest.mark.parametrize("mode", ["sparse", "naive_dense", "staged_dense", "potential"])
def test_environment_reward_modes_and_info_contract(mode: str) -> None:
    env = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="state",
        reward_mode=mode,
        randomize_reset=False,
    )
    try:
        env.reset(seed=0)
        _, reward, _, _, info = env.step(ZERO_ACTION)
        assert np.isfinite(reward)
        assert info.keys() >= EXPECTED_INFO_KEYS
        assert info["reward_mode"] == mode
        assert all(np.isfinite(value) for value in info["reward_terms"].values())
        if mode == "sparse":
            assert reward == 0.0
        if mode == "potential":
            assert {"potential_previous", "potential_current"} <= info.keys()
    finally:
        env.close()


def test_reward_configuration_mode_must_match_environment() -> None:
    with pytest.raises(ValueError, match="must equal"):
        gym.make(
            "gym_hil/SO101PickPlaceBase-v0",
            observation_mode="state",
            reward_mode="sparse",
            reward_config=RewardConfig(mode="potential"),
        )
