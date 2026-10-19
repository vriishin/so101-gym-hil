"""Pure task predicates and immutable task-state representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gym_hil.rewards.config import TaskConfig


def grasped(
    left_finger_contact: bool,
    right_finger_contact: bool,
    gripper_position: float,
    reach_distance: float,
    config: TaskConfig,
) -> bool:
    """Require bilateral contact, a closed gripper, and proximity to the pinch site."""

    return bool(
        left_finger_contact
        and right_finger_contact
        and gripper_position <= config.gripper_closed_threshold
        and reach_distance <= config.grasp_distance_m
    )


def lifted(object_lift_height: float, config: TaskConfig) -> bool:
    """Return whether the cube exceeds the configured lift height."""

    return bool(object_lift_height >= config.lift_threshold_m)


def in_target(object_position: np.ndarray, target_position: np.ndarray, config: TaskConfig) -> bool:
    """Require the complete cube footprint and its center height to fit the target region."""

    usable_radius = config.target_radius_m - config.cube_half_extent_m
    xy_inside = np.linalg.norm(object_position[:2] - target_position[:2]) <= usable_radius
    expected_z = target_position[2] + config.cube_half_extent_m
    height_inside = abs(float(object_position[2] - expected_z)) <= config.target_height_tolerance_m
    return bool(xy_inside and height_inside)


def released(
    object_in_target: bool,
    gripper_position: float,
    left_finger_contact: bool,
    right_finger_contact: bool,
    config: TaskConfig,
) -> bool:
    """Require placement, an open gripper, and ended finger contacts."""

    return bool(
        object_in_target
        and gripper_position >= config.gripper_open_threshold
        and not left_finger_contact
        and not right_finger_contact
    )


def instantaneously_stable(
    object_in_target: bool,
    object_velocity: np.ndarray,
    object_angular_velocity: np.ndarray,
    config: TaskConfig,
) -> bool:
    """Check the non-temporal part of the stability predicate."""

    return bool(
        object_in_target
        and np.linalg.norm(object_velocity) <= config.stable_linear_velocity_m_s
        and np.linalg.norm(object_angular_velocity) <= config.stable_angular_velocity_rad_s
    )


def out_of_bounds(object_position: np.ndarray, config: TaskConfig) -> bool:
    """Detect a cube outside the recoverable workspace."""

    return bool(
        np.any(np.abs(object_position[:2]) > config.workspace_xy_limit_m)
        or object_position[2] < config.minimum_object_z_m
        or object_position[2] > config.maximum_object_z_m
    )


@dataclass(frozen=True)
class TaskState:
    """Privileged simulator state used only for reward and evaluation logic."""

    tcp_position: np.ndarray
    tcp_velocity: np.ndarray
    object_position: np.ndarray
    object_velocity: np.ndarray
    object_angular_velocity: np.ndarray
    target_position: np.ndarray
    gripper_position: float
    left_finger_contact: bool
    right_finger_contact: bool
    object_in_target: bool
    object_lift_height: float
    reach_distance: float
    target_distance: float
    is_grasped: bool
    is_lifted: bool
    is_released: bool
    is_stable: bool
    stable_steps: int
    success: bool
    invalid_collision_count: int
    out_of_bounds: bool
