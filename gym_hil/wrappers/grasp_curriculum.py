"""Minimal gripper-only curriculum used to isolate sustained grasp learning."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class GripperOnlyGraspCurriculum(gym.Wrapper):
    """Mask arm actions and replace full-task reward with grasp-only shaping."""

    def __init__(
        self,
        env: gym.Env,
        *,
        sustained_steps: int = 5,
        alignment_radius_m: float = 0.04,
        terminal_bonus: float = 1.0,
        closure_progress_scale: float = 0.2,
        opening_penalty_scale: float = 0.01,
        terminate_on_grasp: bool = True,
    ) -> None:
        super().__init__(env)
        self.sustained_steps = sustained_steps
        self.alignment_radius_m = alignment_radius_m
        self.terminal_bonus = terminal_bonus
        self.closure_progress_scale = closure_progress_scale
        self.opening_penalty_scale = opening_penalty_scale
        self.terminate_on_grasp = terminate_on_grasp
        self._sustained_count = 0
        self._previous_gripper_position = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._sustained_count = 0
        self._previous_gripper_position = float(self.unwrapped._task_state.gripper_position)
        info = {**info, "curriculum": "gripper_only_grasp", "sustained_grasp_steps": 0}
        return observation, info

    def _transform_action(self, requested: np.ndarray) -> np.ndarray:
        masked = requested.copy()
        masked[:5] = 0.0
        return masked

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32)
        if requested.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {requested.shape}")
        masked = self._transform_action(requested)
        observation, _base_reward, terminated, truncated, info = self.env.step(masked)
        state = self.unwrapped._task_state
        aligned = bool(state.reach_distance <= self.alignment_radius_m)
        bilateral = bool(state.left_finger_contact and state.right_finger_contact)
        maintained = bool(aligned and bilateral and state.is_grasped)
        self._sustained_count = self._sustained_count + 1 if maintained else 0
        success = self._sustained_count >= self.sustained_steps

        # Smaller aperture values mean a more closed gripper. Reward only closure
        # progress while alignment remains valid; never pay a repeated grasp term.
        closure_delta = self._previous_gripper_position - float(state.gripper_position)
        closure_progress = self.closure_progress_scale * max(closure_delta, 0.0) if aligned else 0.0
        opening_penalty = -self.opening_penalty_scale * max(-closure_delta, 0.0) if aligned else 0.0
        terminal_success = self.terminal_bonus if success else 0.0
        reward = float(closure_progress + opening_penalty + terminal_success)
        self._previous_gripper_position = float(state.gripper_position)
        terminated = bool(
            (success and self.terminate_on_grasp) or (terminated and not info.get("is_success", False))
        )
        info = {
            **info,
            "is_success": success,
            "grasp_only_success": success,
            "bilateral_contact": bilateral,
            "valid_pregrasp_alignment": aligned,
            "sustained_grasp_steps": self._sustained_count,
            "requested_action": requested.copy(),
            "masked_action": masked.copy(),
            "reward_terms": {
                "closure_progress": float(closure_progress),
                "opening_penalty": float(opening_penalty),
                "terminal_grasp_success": float(terminal_success),
            },
            "termination_reason": "sustained_grasp" if success else info.get("termination_reason"),
        }
        return observation, reward, terminated, truncated, info


class AlignmentGraspCurriculum(GripperOnlyGraspCurriculum):
    """Unlock small arm corrections and add potential-based alignment progress."""

    def __init__(
        self,
        env: gym.Env,
        *,
        unlocked_arm_indices: tuple[int, ...] = (0, 1, 2, 3),
        arm_action_scale: float = 0.15,
        alignment_progress_scale: float = 0.1,
        alignment_sigma_m: float = 0.03,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.unlocked_arm_indices = unlocked_arm_indices
        self.arm_action_scale = arm_action_scale
        self.alignment_progress_scale = alignment_progress_scale
        self.alignment_sigma_m = alignment_sigma_m
        self._previous_alignment_potential = 0.0

    def _alignment_potential(self) -> float:
        return float(np.exp(-self.unwrapped._task_state.reach_distance / self.alignment_sigma_m))

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self._previous_alignment_potential = self._alignment_potential()
        return observation, info

    def _transform_action(self, requested: np.ndarray) -> np.ndarray:
        transformed = requested.copy()
        transformed[:5] = 0.0
        transformed[list(self.unlocked_arm_indices)] = (
            requested[list(self.unlocked_arm_indices)] * self.arm_action_scale
        )
        return transformed

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        current_potential = self._alignment_potential()
        alignment_progress = self.alignment_progress_scale * (
            current_potential - self._previous_alignment_potential
        )
        self._previous_alignment_potential = current_potential
        reward += alignment_progress
        info["reward_terms"]["alignment_progress"] = float(alignment_progress)
        return observation, float(reward), terminated, truncated, info


class GraspLiftCurriculum(AlignmentGraspCurriculum):
    """Require a stable grasp followed by a gated vertical lift."""

    def __init__(
        self,
        env: gym.Env,
        *,
        lift_height_m: float = 0.05,
        lift_progress_scale: float = 2.0,
        lift_terminal_bonus: float = 5.0,
        drop_penalty: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(env, terminate_on_grasp=False, **kwargs)
        self.lift_height_m = lift_height_m
        self.lift_progress_scale = lift_progress_scale
        self.lift_terminal_bonus = lift_terminal_bonus
        self.drop_penalty = drop_penalty
        self._stable_grasp_achieved = False
        self._previous_lift_height = 0.0

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self._stable_grasp_achieved = False
        self._previous_lift_height = float(self.unwrapped._task_state.object_lift_height)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        state = self.unwrapped._task_state
        if info["sustained_grasp_steps"] >= self.sustained_steps:
            self._stable_grasp_achieved = True
        lift_delta = float(state.object_lift_height - self._previous_lift_height)
        lift_progress = self.lift_progress_scale * max(lift_delta, 0.0) if state.is_grasped else 0.0
        dropped = bool(self._stable_grasp_achieved and not state.is_grasped)
        drop = -self.drop_penalty if dropped else 0.0
        success = bool(
            self._stable_grasp_achieved
            and state.is_grasped
            and state.object_lift_height >= self.lift_height_m
        )
        terminal_lift = self.lift_terminal_bonus if success else 0.0
        reward = float(reward + lift_progress + drop + terminal_lift)
        self._previous_lift_height = float(state.object_lift_height)
        terminated = bool(success or terminated)
        info.update(
            {
                "is_success": success,
                "grasp_lift_success": success,
                "stable_grasp_achieved": self._stable_grasp_achieved,
                "lift_height": float(state.object_lift_height),
                "dropped_after_grasp": dropped,
                "termination_reason": "grasp_and_lift" if success else info.get("termination_reason"),
            }
        )
        info["reward_terms"].update(
            {
                "lift_progress": float(lift_progress),
                "drop_penalty": float(drop),
                "terminal_lift_success": float(terminal_lift),
            }
        )
        return observation, reward, terminated, truncated, info


class ScalarLiftCurriculum(gym.Wrapper):
    """Start grasped and project one policy scalar onto a validated lift direction."""

    def __init__(
        self,
        env: gym.Env,
        *,
        lift_direction: tuple[float, ...],
        action_index: int = 0,
        lift_action_scale: float = 0.30,
        episode_steps: int = 60,
        lift_height_m: float = 0.05,
        stable_lift_steps: int = 3,
        height_progress_scale: float = 1.0,
        height_potential_exponent: float = 1.0,
        bonus_1cm: float = 0.2,
        bonus_3cm: float = 0.5,
        terminal_bonus: float = 5.0,
        drop_penalty: float = 1.0,
    ) -> None:
        super().__init__(env)
        direction = np.asarray(lift_direction, dtype=np.float32)
        if direction.shape != (5,):
            raise ValueError("lift_direction must contain five arm values")
        self.lift_direction = direction
        self.action_index = action_index
        self.lift_action_scale = lift_action_scale
        self.episode_steps = episode_steps
        self.lift_height_m = lift_height_m
        self.stable_lift_steps = stable_lift_steps
        self.height_progress_scale = height_progress_scale
        self.height_potential_exponent = height_potential_exponent
        self.bonus_1cm = bonus_1cm
        self.bonus_3cm = bonus_3cm
        self.terminal_bonus = terminal_bonus
        self.drop_penalty = drop_penalty
        self._initial_cube_z = 0.0
        self._previous_potential = 0.0
        self._awarded_1cm = False
        self._awarded_3cm = False
        self._lift_stable_count = 0
        self._policy_steps = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        sustained = 0
        for _ in range(25):
            close = np.zeros(6, dtype=np.float32)
            close[-1] = -1.0
            observation, _, terminated, truncated, info = self.env.step(close)
            state = self.unwrapped._task_state
            sustained = (
                sustained + 1
                if state.is_grasped and state.left_finger_contact and state.right_finger_contact
                else 0
            )
            if sustained >= 5:
                break
            if terminated or truncated:
                raise RuntimeError("Base environment ended during fixed grasp acquisition")
        if sustained < 5:
            raise RuntimeError("Fixed grasp acquisition did not reach five sustained steps")
        self._initial_cube_z = float(self.unwrapped._task_state.object_position[2])
        self._previous_potential = 0.0
        self._awarded_1cm = self._awarded_3cm = False
        self._lift_stable_count = self._policy_steps = 0
        return observation, {**info, "curriculum": "scalar_lift", "fixed_grasp_acquired": True}

    def _project_action(self, requested: np.ndarray) -> np.ndarray:
        scalar = float(requested[self.action_index])
        projected = np.zeros(6, dtype=np.float32)
        projected[:5] = scalar * self.lift_direction * self.lift_action_scale
        projected[-1] = -1.0
        return projected

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32)
        if requested.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {requested.shape}")
        projected = self._project_action(requested)
        observation, _base_reward, terminated, _base_truncated, info = self.env.step(projected)
        self._policy_steps += 1
        state = self.unwrapped._task_state
        grasped = bool(state.is_grasped and state.left_finger_contact and state.right_finger_contact)
        height = float(state.object_position[2] - self._initial_cube_z)
        potential = float(np.clip(height / self.lift_height_m, 0.0, 1.0) ** self.height_potential_exponent)
        height_progress = (
            self.height_progress_scale * (potential - self._previous_potential) if grasped else 0.0
        )
        bonus_1cm = self.bonus_1cm if grasped and height >= 0.01 and not self._awarded_1cm else 0.0
        bonus_3cm = self.bonus_3cm if grasped and height >= 0.03 and not self._awarded_3cm else 0.0
        self._awarded_1cm |= bonus_1cm > 0
        self._awarded_3cm |= bonus_3cm > 0
        dropped = not grasped
        drop = -self.drop_penalty if dropped else 0.0
        self._lift_stable_count = (
            self._lift_stable_count + 1 if grasped and height >= self.lift_height_m else 0
        )
        success = self._lift_stable_count >= self.stable_lift_steps
        terminal_lift = self.terminal_bonus if success else 0.0
        reward = float(height_progress + bonus_1cm + bonus_3cm + drop + terminal_lift)
        self._previous_potential = potential
        truncated = self._policy_steps >= self.episode_steps
        terminated = bool(success or dropped or terminated)
        info = {
            **info,
            "is_success": success,
            "scalar_lift_success": success,
            "grasp_retained": grasped,
            "dropped_after_grasp": dropped,
            "lift_height": height,
            "lift_stable_steps": self._lift_stable_count,
            "requested_action": requested.copy(),
            "projected_action": projected.copy(),
            "reward_terms": {
                "height_progress": float(height_progress),
                "bonus_1cm": float(bonus_1cm),
                "bonus_3cm": float(bonus_3cm),
                "drop_penalty": float(drop),
                "terminal_lift_success": float(terminal_lift),
            },
            "termination_reason": "stable_5cm_lift"
            if success
            else ("drop" if dropped else ("time_limit" if truncated else info.get("termination_reason"))),
        }
        return observation, reward, terminated, truncated, info


class ArmLiftCurriculum(ScalarLiftCurriculum):
    """Restore the minimum direct joint actions needed for coordinated lifting."""

    def __init__(self, env: gym.Env, *, arm_indices=(1, 2, 3), arm_action_scale=0.30, **kwargs):
        super().__init__(env, lift_direction=(0, 0, 0, 0, 0), **kwargs)
        self.arm_indices = tuple(arm_indices)
        self.arm_action_scale = arm_action_scale

    def _project_action(self, requested: np.ndarray) -> np.ndarray:
        projected = np.zeros(6, dtype=np.float32)
        projected[list(self.arm_indices)] = requested[list(self.arm_indices)] * self.arm_action_scale
        projected[-1] = -1.0
        return projected


class ResidualLiftCurriculum(ScalarLiftCurriculum):
    """Lift along a validated direction with small, explicitly bounded residuals.

    The public action remains six-dimensional. Action 0 is ``alpha`` and
    actions 1-3 are residual coordinates for shoulder lift, elbow, and wrist
    flex. Joints 0 and 4 are masked and the gripper is locked closed.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        lift_direction: tuple[float, float, float],
        lift_action_scale: float,
        residual_action_scale: float = 0.05,
        residual_penalty_scale: float = 0.02,
        orthogonal_penalty_scale: float = 0.04,
        reverse_penalty_scale: float = 0.05,
        arm_indices: tuple[int, int, int] = (1, 2, 3),
        **kwargs,
    ) -> None:
        # ScalarLiftCurriculum owns grasped reset, height shaping, and gates.
        super().__init__(env, lift_direction=(0, 0, 0, 0, 0), **kwargs)
        direction = np.asarray(lift_direction, dtype=np.float32)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("lift_direction must contain three finite values")
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            raise ValueError("lift_direction must be non-zero")
        if residual_action_scale < 0 or residual_action_scale > 0.20:
            raise ValueError("residual_action_scale must be in [0, 0.20]")
        self.lift_direction_3d = direction / norm
        self.lift_action_scale = float(lift_action_scale)
        self.residual_action_scale = float(residual_action_scale)
        self.residual_penalty_scale = float(residual_penalty_scale)
        self.orthogonal_penalty_scale = float(orthogonal_penalty_scale)
        self.reverse_penalty_scale = float(reverse_penalty_scale)
        self.arm_indices = tuple(arm_indices)
        if len(self.arm_indices) != 3:
            raise ValueError("arm_indices must contain three joint indices")
        self._positive_lift_progress = False

    def set_residual_scale(self, scale: float) -> None:
        """Set a curriculum scale without allowing unrestricted residual control."""

        scale = float(scale)
        if scale < 0 or scale > 0.20:
            raise ValueError("residual scale must be in [0, 0.20]")
        self.residual_action_scale = scale

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self._positive_lift_progress = False
        return observation, {
            **info,
            "curriculum": "residual_lift",
            "residual_action_scale": self.residual_action_scale,
        }

    def _project_action(self, requested: np.ndarray) -> np.ndarray:
        alpha = float(requested[0])
        residual_command = requested[1:4]
        arm_action = (
            alpha * self.lift_action_scale * self.lift_direction_3d
            + residual_command * self.residual_action_scale
        )
        projected = np.zeros(6, dtype=np.float32)
        projected[list(self.arm_indices)] = arm_action
        projected[-1] = -1.0
        return projected

    def encode_arm_action(self, arm_action: np.ndarray) -> tuple[np.ndarray, float]:
        """Convert a desired three-joint command into alpha-plus-residual form."""

        desired = np.asarray(arm_action, dtype=np.float32)
        if desired.shape != (3,):
            raise ValueError("arm_action must contain three values")
        alpha = float(np.dot(desired, self.lift_direction_3d) / self.lift_action_scale)
        alpha = float(np.clip(alpha, -1.0, 1.0))
        remainder = desired - alpha * self.lift_action_scale * self.lift_direction_3d
        if self.residual_action_scale > 0:
            residual = np.clip(remainder / self.residual_action_scale, -1.0, 1.0)
        else:
            residual = np.zeros(3, dtype=np.float32)
        requested = np.zeros(6, dtype=np.float32)
        requested[0] = alpha
        requested[1:4] = residual
        reconstructed = self._project_action(requested)[list(self.arm_indices)]
        return requested, float(np.linalg.norm(reconstructed - desired))

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32)
        if requested.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {requested.shape}")
        projected = self._project_action(requested)
        arm_action = projected[list(self.arm_indices)]
        base_component = float(requested[0]) * self.lift_action_scale * self.lift_direction_3d
        residual = arm_action - base_component
        parallel_residual = float(np.dot(residual, self.lift_direction_3d))
        orthogonal_residual = residual - parallel_residual * self.lift_direction_3d
        had_positive_progress = self._positive_lift_progress

        observation, reward, terminated, truncated, info = super().step(requested)
        height_progress = float(info["reward_terms"]["height_progress"])
        self._positive_lift_progress |= height_progress > 0.0

        residual_penalty = -self.residual_penalty_scale * float(np.linalg.norm(residual))
        orthogonal_penalty = -self.orthogonal_penalty_scale * float(np.linalg.norm(orthogonal_residual))
        signed_lift_action = float(np.dot(arm_action, self.lift_direction_3d))
        reverse_penalty = (
            -self.reverse_penalty_scale * max(-signed_lift_action, 0.0) if had_positive_progress else 0.0
        )
        reward = float(reward + residual_penalty + orthogonal_penalty + reverse_penalty)
        info.update(
            {
                "residual_lift_success": bool(info["scalar_lift_success"]),
                "alpha": float(requested[0]),
                "residual_command": requested[1:4].copy(),
                "residual_action": residual.copy(),
                "orthogonal_residual_magnitude": float(np.linalg.norm(orthogonal_residual)),
                "signed_lift_action": signed_lift_action,
                "residual_action_scale": self.residual_action_scale,
            }
        )
        info["reward_terms"].update(
            {
                "residual_penalty": float(residual_penalty),
                "orthogonal_motion_penalty": float(orthogonal_penalty),
                "reverse_after_progress_penalty": float(reverse_penalty),
            }
        )
        return observation, reward, terminated, truncated, info


class CompositeGraspLiftCurriculum(gym.Wrapper):
    """Phase-aware grasp then residual-lift task without transport or placement."""

    def __init__(
        self,
        env: gym.Env,
        *,
        sustained_grasp_steps: int = 5,
        grasp_arm_indices: tuple[int, ...] = (0, 1, 2, 3),
        grasp_arm_action_scale: float = 0.15,
        alignment_sigma_m: float = 0.03,
        alignment_progress_scale: float = 0.1,
        sustained_grasp_bonus: float = 1.0,
        lift_direction: tuple[float, float, float] = (-1.0, -0.23467704, 0.58487764),
        lift_action_scale: float = 0.3546039552,
        residual_action_scale: float = 0.20,
        lift_arm_indices: tuple[int, int, int] = (1, 2, 3),
        lift_height_m: float = 0.05,
        stable_lift_steps: int = 3,
        height_progress_scale: float = 1.0,
        height_potential_exponent: float = 4.0,
        bonus_1cm: float = 0.2,
        bonus_3cm: float = 0.5,
        terminal_bonus: float = 5.0,
        drop_penalty: float = 1.0,
        residual_penalty_scale: float = 0.02,
        orthogonal_penalty_scale: float = 0.04,
        reverse_penalty_scale: float = 0.05,
        episode_steps: int = 150,
    ) -> None:
        super().__init__(env)
        direction = np.asarray(lift_direction, dtype=np.float32)
        if direction.shape != (3,) or float(np.linalg.norm(direction)) <= 0:
            raise ValueError("lift_direction must contain three non-zero values")
        self.lift_direction_3d = direction / np.linalg.norm(direction)
        self.sustained_grasp_steps = int(sustained_grasp_steps)
        self.grasp_arm_indices = tuple(grasp_arm_indices)
        self.grasp_arm_action_scale = float(grasp_arm_action_scale)
        self.alignment_sigma_m = float(alignment_sigma_m)
        self.alignment_progress_scale = float(alignment_progress_scale)
        self.sustained_grasp_bonus = float(sustained_grasp_bonus)
        self.lift_action_scale = float(lift_action_scale)
        self.residual_action_scale = float(residual_action_scale)
        self.lift_arm_indices = tuple(lift_arm_indices)
        self.lift_height_m = float(lift_height_m)
        self.stable_lift_steps = int(stable_lift_steps)
        self.height_progress_scale = float(height_progress_scale)
        self.height_potential_exponent = float(height_potential_exponent)
        self.bonus_1cm = float(bonus_1cm)
        self.bonus_3cm = float(bonus_3cm)
        self.terminal_bonus = float(terminal_bonus)
        self.drop_penalty = float(drop_penalty)
        self.residual_penalty_scale = float(residual_penalty_scale)
        self.orthogonal_penalty_scale = float(orthogonal_penalty_scale)
        self.reverse_penalty_scale = float(reverse_penalty_scale)
        self.episode_steps = int(episode_steps)
        self.phase = "grasp"
        self.sustained_grasp_count = 0
        self._policy_steps = 0
        self._previous_alignment_potential = 0.0
        self._lift_start_cube_z = 0.0
        self._previous_height_potential = 0.0
        self._awarded_1cm = False
        self._awarded_3cm = False
        self._lift_stable_count = 0
        self._positive_lift_progress = False

    def _alignment_potential(self) -> float:
        return float(np.exp(-self.unwrapped._task_state.reach_distance / self.alignment_sigma_m))

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.phase = "grasp"
        self.sustained_grasp_count = 0
        self._policy_steps = 0
        self._previous_alignment_potential = self._alignment_potential()
        self._lift_start_cube_z = float(self.unwrapped._task_state.object_position[2])
        self._previous_height_potential = 0.0
        self._awarded_1cm = self._awarded_3cm = False
        self._lift_stable_count = 0
        self._positive_lift_progress = False
        return observation, {
            **info,
            "curriculum": "composite_grasp_lift",
            "policy_phase": self.phase,
            "sustained_grasp_steps": 0,
        }

    def _grasp_action(self, requested: np.ndarray) -> np.ndarray:
        projected = requested.copy()
        projected[:5] = 0.0
        indices = list(self.grasp_arm_indices)
        projected[indices] = requested[indices] * self.grasp_arm_action_scale
        return projected

    def _lift_action(self, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha = float(requested[0])
        residual = requested[1:4] * self.residual_action_scale
        arm_action = alpha * self.lift_action_scale * self.lift_direction_3d + residual
        projected = np.zeros(6, dtype=np.float32)
        projected[list(self.lift_arm_indices)] = arm_action
        projected[-1] = -1.0
        parallel = float(np.dot(residual, self.lift_direction_3d)) * self.lift_direction_3d
        return projected, residual, residual - parallel

    def project_action(self, action: np.ndarray, phase: str | None = None) -> np.ndarray:
        """Expose the phase action transform for diagnostics and demonstrations."""

        requested = np.asarray(action, dtype=np.float32)
        selected_phase = self.phase if phase is None else phase
        if selected_phase == "grasp":
            return self._grasp_action(requested)
        return self._lift_action(requested)[0]

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32)
        if requested.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {requested.shape}")
        phase_before = self.phase
        residual = np.zeros(3, dtype=np.float32)
        orthogonal = np.zeros(3, dtype=np.float32)
        if phase_before == "grasp":
            projected = self._grasp_action(requested)
        else:
            projected, residual, orthogonal = self._lift_action(requested)
        observation, _base_reward, base_terminated, _base_truncated, info = self.env.step(projected)
        self._policy_steps += 1
        state = self.unwrapped._task_state
        bilateral = bool(state.left_finger_contact and state.right_finger_contact)
        maintained = bool(bilateral and state.is_grasped)
        phase_switched = False
        grasp_bonus = 0.0
        alignment_progress = 0.0
        if phase_before == "grasp":
            current_alignment = self._alignment_potential()
            alignment_progress = self.alignment_progress_scale * (
                current_alignment - self._previous_alignment_potential
            )
            self._previous_alignment_potential = current_alignment
            self.sustained_grasp_count = self.sustained_grasp_count + 1 if maintained else 0
            if self.sustained_grasp_count >= self.sustained_grasp_steps:
                self.phase = "lift"
                phase_switched = True
                grasp_bonus = self.sustained_grasp_bonus
                self._lift_start_cube_z = float(state.object_position[2])

        height = max(float(state.object_position[2] - self._lift_start_cube_z), 0.0)
        height_progress = bonus_1cm = bonus_3cm = terminal_lift = drop = 0.0
        residual_penalty = orthogonal_penalty = reverse_penalty = 0.0
        if phase_before == "lift":
            potential = float(
                np.clip(height / self.lift_height_m, 0.0, 1.0) ** self.height_potential_exponent
            )
            height_progress = (
                self.height_progress_scale * (potential - self._previous_height_potential)
                if maintained
                else 0.0
            )
            bonus_1cm = self.bonus_1cm if maintained and height >= 0.01 and not self._awarded_1cm else 0.0
            bonus_3cm = self.bonus_3cm if maintained and height >= 0.03 and not self._awarded_3cm else 0.0
            self._awarded_1cm |= bonus_1cm > 0
            self._awarded_3cm |= bonus_3cm > 0
            drop = -self.drop_penalty if not maintained else 0.0
            self._lift_stable_count = (
                self._lift_stable_count + 1 if maintained and height >= self.lift_height_m else 0
            )
            residual_penalty = -self.residual_penalty_scale * float(np.linalg.norm(residual))
            orthogonal_penalty = -self.orthogonal_penalty_scale * float(np.linalg.norm(orthogonal))
            signed_lift = float(np.dot(projected[list(self.lift_arm_indices)], self.lift_direction_3d))
            reverse_penalty = (
                -self.reverse_penalty_scale * max(-signed_lift, 0.0) if self._positive_lift_progress else 0.0
            )
            self._positive_lift_progress |= height_progress > 0.0
            self._previous_height_potential = potential
        success = self._lift_stable_count >= self.stable_lift_steps
        terminal_lift = self.terminal_bonus if success else 0.0
        reward = float(
            alignment_progress
            + grasp_bonus
            + height_progress
            + bonus_1cm
            + bonus_3cm
            + terminal_lift
            + drop
            + residual_penalty
            + orthogonal_penalty
            + reverse_penalty
        )
        dropped = bool(phase_before == "lift" and not maintained)
        terminated = bool(success or dropped or base_terminated)
        truncated = self._policy_steps >= self.episode_steps
        info = {
            **info,
            "is_success": success,
            "grasp_lift_success": success,
            "policy_phase_before": phase_before,
            "policy_phase": self.phase,
            "phase_switched": phase_switched,
            "bilateral_contact": bilateral,
            "sustained_grasp_steps": self.sustained_grasp_count,
            "sustained_grasp_achieved": self.phase == "lift",
            "grasp_to_lift_transition": phase_switched,
            "grasp_retained": maintained,
            "dropped_after_grasp": dropped,
            "lift_height": height,
            "lift_stable_steps": self._lift_stable_count,
            "requested_action": requested.copy(),
            "projected_action": projected.copy(),
            "reward_terms": {
                "alignment_progress": float(alignment_progress),
                "sustained_grasp_bonus": float(grasp_bonus),
                "height_progress": float(height_progress),
                "bonus_1cm": float(bonus_1cm),
                "bonus_3cm": float(bonus_3cm),
                "terminal_lift_success": float(terminal_lift),
                "drop_penalty": float(drop),
                "residual_penalty": float(residual_penalty),
                "orthogonal_motion_penalty": float(orthogonal_penalty),
                "reverse_after_progress_penalty": float(reverse_penalty),
            },
            "termination_reason": (
                "stable_5cm_lift"
                if success
                else "drop"
                if dropped
                else "time_limit"
                if truncated
                else info.get("termination_reason")
            ),
        }
        return observation, reward, terminated, truncated, info
