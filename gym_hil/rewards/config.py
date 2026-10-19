"""Typed and validated task/reward configuration."""

from dataclasses import dataclass
from typing import Literal

RewardMode = Literal["sparse", "naive_dense", "staged_dense", "potential"]


@dataclass(frozen=True)
class RewardConfig:
    """Weights and normalizers shared by every pick-and-place reward mode."""

    mode: RewardMode = "sparse"
    gamma: float = 0.99
    success_bonus: float = 1.0
    shaping_scale: float = 0.25
    reach_weight: float = 0.10
    grasp_weight: float = 0.15
    lift_weight: float = 0.20
    transport_weight: float = 0.25
    place_weight: float = 0.15
    release_weight: float = 0.05
    stability_weight: float = 0.10
    action_penalty_weight: float = 0.001
    jerk_penalty_weight: float = 0.001
    drop_penalty: float = 0.20
    out_of_bounds_penalty: float = 0.50
    reach_sigma_m: float = 0.08
    target_sigma_m: float = 0.10
    desired_lift_height_m: float = 0.06

    def __post_init__(self) -> None:
        if self.mode not in {"sparse", "naive_dense", "staged_dense", "potential"}:
            raise ValueError(f"Unsupported reward mode: {self.mode}")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if min(self.reach_sigma_m, self.target_sigma_m, self.desired_lift_height_m) <= 0:
            raise ValueError("reward feature scales must be positive")
        weights = (
            self.success_bonus,
            self.shaping_scale,
            self.reach_weight,
            self.grasp_weight,
            self.lift_weight,
            self.transport_weight,
            self.place_weight,
            self.release_weight,
            self.stability_weight,
            self.action_penalty_weight,
            self.jerk_penalty_weight,
            self.drop_penalty,
            self.out_of_bounds_penalty,
        )
        if min(weights) < 0:
            raise ValueError("reward weights and penalties must be non-negative magnitudes")


@dataclass(frozen=True)
class TaskConfig:
    """Geometric, stability, and failure thresholds for task predicates."""

    lift_threshold_m: float = 0.04
    grasp_distance_m: float = 0.05
    gripper_closed_threshold: float = 0.80
    gripper_open_threshold: float = 1.05
    target_radius_m: float = 0.045
    cube_half_extent_m: float = 0.018
    target_height_tolerance_m: float = 0.025
    stable_linear_velocity_m_s: float = 0.02
    stable_angular_velocity_rad_s: float = 0.20
    stable_hold_s: float = 0.50
    control_hz: int = 10
    workspace_xy_limit_m: float = 0.43
    minimum_object_z_m: float = -0.03
    maximum_object_z_m: float = 0.60
    invalid_collision_limit: int = 25
    terminate_on_out_of_bounds: bool = True
    grasp_assist: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.lift_threshold_m,
            self.grasp_distance_m,
            self.target_radius_m,
            self.cube_half_extent_m,
            self.target_height_tolerance_m,
            self.stable_linear_velocity_m_s,
            self.stable_angular_velocity_rad_s,
            self.stable_hold_s,
            self.workspace_xy_limit_m,
            self.maximum_object_z_m,
        )
        if min(positive) <= 0 or self.control_hz <= 0 or self.invalid_collision_limit <= 0:
            raise ValueError("task thresholds, control_hz, and collision limit must be positive")
        if self.gripper_closed_threshold >= self.gripper_open_threshold:
            raise ValueError("closed gripper threshold must be below open threshold")
        if self.target_radius_m <= self.cube_half_extent_m:
            raise ValueError("target radius must exceed the cube half extent")

    @property
    def stable_steps_required(self) -> int:
        """Number of consecutive control steps required for stability."""

        return max(1, round(self.stable_hold_s * self.control_hz))
