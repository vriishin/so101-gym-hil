import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401
from gym_hil.wrappers.grasp_curriculum import (
    AlignmentGraspCurriculum,
    ArmLiftCurriculum,
    CompositeGraspLiftCurriculum,
    GraspLiftCurriculum,
    GripperOnlyGraspCurriculum,
    ResidualLiftCurriculum,
    ScalarLiftCurriculum,
)
from gym_hil.wrappers.placement_curriculum import DescentCurriculum, ReleaseCurriculum
from gym_hil.wrappers.transport_curriculum import CompositeGraspLiftTransportCurriculum


def make_curriculum():
    base = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="state",
        reward_mode="sparse",
        easy_reset=True,
        randomize_reset=False,
        episode_length_s=6.0,
    )
    return GripperOnlyGraspCurriculum(base, sustained_steps=5)


def test_arm_actions_are_masked_and_six_dimensional_contract_is_preserved() -> None:
    env = make_curriculum()
    try:
        env.reset(seed=0)
        initial_arm = env.unwrapped.joint_positions[:5]
        _, _, _, _, info = env.step(np.ones(6, dtype=np.float32))
        assert env.action_space.shape == (6,)
        np.testing.assert_array_equal(info["masked_action"][:5], np.zeros(5))
        np.testing.assert_allclose(env.unwrapped.joint_positions[:5], initial_arm, atol=0.01)
    finally:
        env.close()


def test_forced_close_reaches_sustained_grasp_once() -> None:
    env = make_curriculum()
    try:
        env.reset(seed=0)
        terminal_rewards = 0
        info = {}
        for _ in range(30):
            _, reward, terminated, truncated, info = env.step(
                np.asarray((0, 0, 0, 0, 0, -1), dtype=np.float32)
            )
            terminal_rewards += int(info["reward_terms"]["terminal_grasp_success"] > 0)
            if terminated or truncated:
                break
        assert info["grasp_only_success"]
        assert info["sustained_grasp_steps"] >= 5
        assert terminal_rewards == 1
    finally:
        env.close()


def test_alignment_curriculum_scales_only_unlocked_arm_dimensions() -> None:
    base = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="state",
        easy_reset=True,
        easy_reset_perturbation_m=0.01,
        randomize_reset=False,
    )
    env = AlignmentGraspCurriculum(base, unlocked_arm_indices=(0, 1), arm_action_scale=0.15)
    try:
        env.reset(seed=7)
        _, _, _, _, info = env.step(np.ones(6, dtype=np.float32))
        np.testing.assert_allclose(info["masked_action"][:2], 0.15)
        np.testing.assert_array_equal(info["masked_action"][2:5], np.zeros(3))
        assert "alignment_progress" in info["reward_terms"]
    finally:
        env.close()


def test_lift_curriculum_does_not_terminate_at_grasp() -> None:
    base = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="state",
        easy_reset=True,
        randomize_reset=False,
        episode_length_s=10.0,
    )
    env = GraspLiftCurriculum(base, sustained_steps=5, terminal_bonus=0.0)
    try:
        env.reset(seed=0)
        info = {}
        for _ in range(20):
            _, _, terminated, truncated, info = env.step(np.asarray((0, 0, 0, 0, 0, -1), dtype=np.float32))
            if info["stable_grasp_achieved"]:
                break
        assert info["stable_grasp_achieved"]
        assert not terminated
        assert not truncated
        assert not info["grasp_lift_success"]
        assert info["reward_terms"]["lift_progress"] >= 0
    finally:
        env.close()


def test_scalar_lift_projection_reaches_five_centimeters() -> None:
    base = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="state",
        easy_reset=True,
        randomize_reset=False,
        episode_length_s=10.0,
    )
    direction = (-0.0000171062, -1.0, -0.23467704, 0.58487764, 0.000555088)
    env = ScalarLiftCurriculum(base, lift_direction=direction, episode_steps=60)
    try:
        env.reset(seed=0)
        info = {}
        for _ in range(60):
            _, _, terminated, truncated, info = env.step(np.asarray((1, 0, 0, 0, 0, 0), dtype=np.float32))
            if terminated or truncated:
                break
        assert info["scalar_lift_success"]
        assert info["lift_height"] >= 0.05
        assert info["grasp_retained"]
        assert info["projected_action"][5] == -1
    finally:
        env.close()


def test_arm_lift_masks_everything_except_minimum_lift_joints() -> None:
    base = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="state",
        easy_reset=True,
        randomize_reset=False,
        episode_length_s=10.0,
    )
    env = ArmLiftCurriculum(base, arm_indices=(1, 2, 3), arm_action_scale=0.3)
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.ones(6, dtype=np.float32))
        np.testing.assert_allclose(info["projected_action"], (0, 0.3, 0.3, 0.3, 0, -1))
    finally:
        env.close()


def test_residual_lift_reproduces_validated_direction_and_masks_other_joints() -> None:
    base = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="state",
        easy_reset=True,
        randomize_reset=False,
        episode_length_s=10.0,
    )
    raw_direction = np.asarray((-1.0, -0.23467704, 0.58487764), dtype=np.float32)
    direction_norm = float(np.linalg.norm(raw_direction))
    env = ResidualLiftCurriculum(
        base,
        lift_direction=tuple(raw_direction),
        lift_action_scale=0.30 * direction_norm,
        residual_action_scale=0.05,
    )
    try:
        env.reset(seed=0)
        action = np.asarray((1, 0, 0, 0, 1, 1), dtype=np.float32)
        _, _, _, _, info = env.step(action)
        expected = np.asarray((0, -0.30, -0.30 * 0.23467704, 0.30 * 0.58487764, 0, -1))
        np.testing.assert_allclose(info["projected_action"], expected, atol=1e-6)
        np.testing.assert_array_equal(info["residual_action"], np.zeros(3))
        assert info["reward_terms"]["residual_penalty"] == 0.0
        assert info["reward_terms"]["orthogonal_motion_penalty"] == 0.0
    finally:
        env.close()


def test_residual_lift_conversion_and_scale_limits() -> None:
    base = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="state",
        easy_reset=True,
        randomize_reset=False,
        episode_length_s=10.0,
    )
    direction = np.asarray((-1.0, -0.23467704, 0.58487764), dtype=np.float32)
    env = ResidualLiftCurriculum(
        base,
        lift_direction=tuple(direction),
        lift_action_scale=0.30 * float(np.linalg.norm(direction)),
        residual_action_scale=0.05,
    )
    try:
        env.reset(seed=0)
        desired = 0.30 * direction
        converted, error = env.encode_arm_action(desired)
        np.testing.assert_allclose(converted, (1, 0, 0, 0, 0, 0), atol=1e-6)
        assert error < 1e-6
        env.set_residual_scale(0.10)
        assert env.residual_action_scale == 0.10
        with np.testing.assert_raises(ValueError):
            env.set_residual_scale(0.21)
    finally:
        env.close()


def test_composite_curriculum_preserves_both_action_interpretations() -> None:
    base = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="state",
        easy_reset=True,
        randomize_reset=False,
        episode_length_s=20.0,
    )
    env = CompositeGraspLiftCurriculum(base)
    try:
        env.reset(seed=0)
        requested = np.ones(6, dtype=np.float32)
        np.testing.assert_allclose(
            env.project_action(requested, "grasp"),
            (0.15, 0.15, 0.15, 0.15, 0.0, 1.0),
        )
        lift = env.project_action(np.asarray((1, 0, 0, 0, 1, 1), dtype=np.float32), "lift")
        assert lift[0] == 0 and lift[4] == 0 and lift[5] == -1
        expected = 0.3546039552 * env.lift_direction_3d
        np.testing.assert_allclose(lift[1:4], expected, atol=1e-6)
        assert env.phase == "grasp"
        assert env.sustained_grasp_count == 0
    finally:
        env.close()


def test_composite_transport_preserves_prior_and_locks_gripper() -> None:
    base = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="state",
        easy_reset=True,
        randomize_reset=False,
        episode_length_s=30.0,
    )
    env = CompositeGraspLiftTransportCurriculum(base, transport_residual_action_scale=0.10)
    try:
        env.reset(seed=0)
        env.phase = "transport"
        env._transport_joint_target[:5] = (
            env.unwrapped.joint_positions[:5] + env.unwrapped.joint_delta_scales[:5]
        )
        action = np.asarray((0.5, 1, 0, 0, 0, 0), dtype=np.float32)
        projected = env.project_action(action)
        np.testing.assert_allclose(projected[:5], (0.6, 0.5, 0.5, 0.5, 0.5), atol=1e-6)
        assert projected[-1] == -1.0
        assert env.action_space.shape == (6,)
    finally:
        env.close()


def test_descent_prior_preserves_six_actions_and_locks_gripper() -> None:
    base = gym.make("gym_hil/SO101PickPlaceBase-v0", observation_mode="state", easy_reset=True)
    transport = CompositeGraspLiftTransportCurriculum(base)
    env = DescentCurriculum(transport, transport_policy=None, residual_action_scale=0.05)
    try:
        transport.reset(seed=0)
        env._release_joint_target[:5] = (
            env.unwrapped.joint_positions[:5] + env.unwrapped.joint_delta_scales[:5]
        )
        transformed, prior, residual = env.transform_action(
            np.asarray((0.5, 1, 0, 0, 0, 0), dtype=np.float32)
        )
        np.testing.assert_allclose(prior, np.ones(5), atol=1e-6)
        np.testing.assert_allclose(transformed[:5], (0.55, 0.5, 0.5, 0.5, 0.5), atol=1e-6)
        np.testing.assert_allclose(residual, (0.05, 0, 0, 0, 0), atol=1e-6)
        assert transformed[-1] == -1.0
        assert env.action_space.shape == (6,)
    finally:
        env.close()


def test_release_curriculum_freezes_every_arm_dimension() -> None:
    base = gym.make("gym_hil/SO101PickPlaceBase-v0", observation_mode="state", easy_reset=True)
    transport = CompositeGraspLiftTransportCurriculum(base)
    descent = DescentCurriculum(transport, transport_policy=None)
    env = ReleaseCurriculum(descent)
    try:
        transport.reset(seed=0)
        _, _, _, _, info = env.step(np.ones(6, dtype=np.float32))
        np.testing.assert_array_equal(info["transformed_action"][:5], np.zeros(5))
        assert info["transformed_action"][-1] == 1.0
    finally:
        env.close()
