from dataclasses import replace

import numpy as np

from gym_hil.rewards import TaskConfig, TaskState
from gym_hil.rewards.task_state import (
    grasped,
    in_target,
    instantaneously_stable,
    lifted,
    out_of_bounds,
    released,
)

CONFIG = TaskConfig()


def state(**changes) -> TaskState:
    base = TaskState(
        tcp_position=np.zeros(3),
        tcp_velocity=np.zeros(3),
        object_position=np.asarray((0.20, -0.30, 0.02)),
        object_velocity=np.zeros(3),
        object_angular_velocity=np.zeros(3),
        target_position=np.asarray((0.20, -0.30, 0.002)),
        gripper_position=0.5,
        left_finger_contact=False,
        right_finger_contact=False,
        object_in_target=False,
        object_lift_height=0.0,
        reach_distance=0.02,
        target_distance=0.10,
        is_grasped=False,
        is_lifted=False,
        is_released=False,
        is_stable=False,
        stable_steps=0,
        success=False,
        invalid_collision_count=0,
        out_of_bounds=False,
    )
    return replace(base, **changes)


def test_near_cube_and_one_finger_are_not_grasps() -> None:
    assert not grasped(False, False, 0.4, 0.01, CONFIG)
    assert not grasped(True, False, 0.4, 0.01, CONFIG)


def test_grasp_requires_contacts_closed_gripper_and_proximity() -> None:
    assert grasped(True, True, 0.4, 0.01, CONFIG)
    assert not grasped(True, True, 1.2, 0.01, CONFIG)
    assert not grasped(True, True, 0.4, 0.20, CONFIG)


def test_lift_target_release_and_stability_predicates() -> None:
    assert lifted(0.05, CONFIG)
    assert not lifted(0.03, CONFIG)
    object_position = np.asarray((0.20, -0.30, 0.02))
    target_position = np.asarray((0.20, -0.30, 0.002))
    assert in_target(object_position, target_position, CONFIG)
    assert not in_target(object_position + np.asarray((0.04, 0, 0)), target_position, CONFIG)
    assert not released(True, 0.5, False, False, CONFIG)
    assert not released(True, 1.2, True, False, CONFIG)
    assert released(True, 1.2, False, False, CONFIG)
    assert instantaneously_stable(True, np.zeros(3), np.zeros(3), CONFIG)
    assert not instantaneously_stable(True, np.asarray((0.1, 0, 0)), np.zeros(3), CONFIG)
    assert not instantaneously_stable(False, np.zeros(3), np.zeros(3), CONFIG)


def test_out_of_bounds_predicate() -> None:
    assert not out_of_bounds(np.asarray((0.2, -0.2, 0.02)), CONFIG)
    assert out_of_bounds(np.asarray((0.5, -0.2, 0.02)), CONFIG)
    assert out_of_bounds(np.asarray((0.2, -0.2, -0.04)), CONFIG)
