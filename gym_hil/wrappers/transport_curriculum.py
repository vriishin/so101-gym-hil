"""Structured transport-only curriculum initialized from a stable lifted grasp."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from gym_hil.controllers.so101_scripted import solve_position_ik
from gym_hil.wrappers.grasp_curriculum import CompositeGraspLiftCurriculum
from gym_hil.wrappers.state_policy_wrapper import unified_grasp_lift_state_vector


class TransportCurriculum(gym.Wrapper):
    """Auto-acquire Phase 3C lift, then follow a waypoint prior plus residuals."""

    def __init__(
        self,
        env: gym.Env,
        *,
        grasp_lift_policy,
        residual_action_scale: float = 0.05,
        episode_steps: int = 75,
        target_tolerance_m: float = 0.025,
        safe_lift_height_m: float = 0.04,
        sustained_target_steps: int = 5,
        distance_progress_scale: float = 1.0,
        low_height_penalty_scale: float = 2.0,
        drop_penalty: float = 5.0,
        terminal_bonus: float = 5.0,
        residual_penalty_scale: float = 0.02,
    ) -> None:
        super().__init__(env)
        if not 0 <= residual_action_scale <= 0.20:
            raise ValueError("residual_action_scale must be in [0, 0.20]")
        self.grasp_lift_policy = grasp_lift_policy
        self.residual_action_scale = float(residual_action_scale)
        self.episode_steps = int(episode_steps)
        self.target_tolerance_m = float(target_tolerance_m)
        self.safe_lift_height_m = float(safe_lift_height_m)
        self.sustained_target_steps = int(sustained_target_steps)
        self.distance_progress_scale = float(distance_progress_scale)
        self.low_height_penalty_scale = float(low_height_penalty_scale)
        self.drop_penalty = float(drop_penalty)
        self.terminal_bonus = float(terminal_bonus)
        self.residual_penalty_scale = float(residual_penalty_scale)
        self._transport_joint_target = np.zeros(6, dtype=np.float64)
        self._initial_distance = 1.0
        self._previous_distance = 1.0
        self._target_stable_count = 0
        self._policy_steps = 0
        self._safe_cube_z = 0.0
        self._minimum_cube_height = np.inf
        self._minimum_clearance = np.inf
        self._lift_acquisition_steps = 0

    @property
    def safe_height_margin(self) -> float:
        return float(self.unwrapped._task_state.object_position[2] - self._safe_cube_z)

    @property
    def transport_joint_target(self) -> np.ndarray:
        return self._transport_joint_target.copy()

    def set_residual_scale(self, scale: float) -> None:
        scale = float(scale)
        if not 0 <= scale <= 0.20:
            raise ValueError("residual scale must be in [0, 0.20]")
        self.residual_action_scale = scale

    def _prior_action(self) -> np.ndarray:
        error = self._transport_joint_target[:5] - self.unwrapped.joint_positions[:5]
        return np.clip(error / self.unwrapped.joint_delta_scales[:5], -1.0, 1.0).astype(np.float32)

    def transform_action(self, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action = np.asarray(requested, dtype=np.float32)
        if action.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {action.shape}")
        prior = self._prior_action()
        residual = action[1:6] * self.residual_action_scale
        transformed = np.zeros(6, dtype=np.float32)
        transformed[:5] = float(action[0]) * prior + residual
        transformed[-1] = -1.0
        return transformed, prior, residual

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        lift_info: dict = {}
        for step in range(150):
            policy_observation = unified_grasp_lift_state_vector(self.env)
            action = self.grasp_lift_policy.predict(policy_observation, deterministic=True)[0]
            observation, _, terminated, truncated, lift_info = self.env.step(action)
            if lift_info.get("grasp_lift_success", False):
                self._lift_acquisition_steps = step + 1
                break
            if terminated or truncated:
                raise RuntimeError("Phase 3C policy failed while initializing transport")
        else:
            raise RuntimeError("Phase 3C policy did not produce a stable lifted grasp")

        state = self.unwrapped._task_state
        tcp_target = state.tcp_position.copy()
        tcp_target[:2] += state.target_position[:2] - state.object_position[:2]
        self._transport_joint_target = solve_position_ik(
            self.unwrapped, tcp_target, keep_gripper_vertical=False
        )
        self._transport_joint_target[-1] = self.unwrapped.joint_positions[-1]
        self._initial_distance = max(
            float(np.linalg.norm(state.object_position[:2] - state.target_position[:2])), 1e-6
        )
        self._previous_distance = self._initial_distance
        self._target_stable_count = 0
        self._policy_steps = 0
        self._safe_cube_z = float(self.unwrapped._initial_cube_height + self.safe_lift_height_m)
        self._minimum_cube_height = float(state.object_position[2] - self.unwrapped._initial_cube_height)
        self._minimum_clearance = float(
            state.object_position[2] - self.unwrapped.task_config.cube_half_extent_m
        )
        return observation, {
            **info,
            **lift_info,
            "curriculum": "transport_only",
            "stable_lift_initialized": True,
            "lift_acquisition_steps": self._lift_acquisition_steps,
            "transport_joint_target": self._transport_joint_target.copy(),
            "residual_action_scale": self.residual_action_scale,
        }

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32)
        transformed, prior, residual = self.transform_action(requested)
        # Bypass the completed grasp-lift task wrapper while retaining the same base env.
        observation, _base_reward, base_terminated, _base_truncated, info = self.env.env.step(transformed)
        self._policy_steps += 1
        state = self.unwrapped._task_state
        grasped = bool(state.is_grasped and state.left_finger_contact and state.right_finger_contact)
        distance = float(np.linalg.norm(state.object_position[:2] - state.target_position[:2]))
        progress = (
            self.distance_progress_scale * (self._previous_distance - distance) / self._initial_distance
            if grasped and self.safe_height_margin >= 0
            else 0.0
        )
        height_penalty = -self.low_height_penalty_scale * max(-self.safe_height_margin, 0.0)
        dropped = not grasped
        drop = -self.drop_penalty if dropped else 0.0
        residual_penalty = -self.residual_penalty_scale * float(np.linalg.norm(residual))
        target_condition = bool(
            grasped and self.safe_height_margin >= 0 and distance <= self.target_tolerance_m
        )
        self._target_stable_count = self._target_stable_count + 1 if target_condition else 0
        success = self._target_stable_count >= self.sustained_target_steps
        terminal = self.terminal_bonus if success else 0.0
        reward = float(progress + height_penalty + drop + residual_penalty + terminal)
        self._previous_distance = distance
        cube_height = float(state.object_position[2] - self.unwrapped._initial_cube_height)
        clearance = float(state.object_position[2] - self.unwrapped.task_config.cube_half_extent_m)
        self._minimum_cube_height = min(self._minimum_cube_height, cube_height)
        self._minimum_clearance = min(self._minimum_clearance, clearance)
        prior_norm = float(np.linalg.norm(prior))
        arm_norm = float(np.linalg.norm(transformed[:5]))
        alignment = (
            float(np.dot(transformed[:5], prior) / (arm_norm * prior_norm))
            if arm_norm > 1e-8 and prior_norm > 1e-8
            else 1.0
        )
        terminated = bool(success or dropped or base_terminated)
        truncated = self._policy_steps >= self.episode_steps
        info = {
            **info,
            "is_success": success,
            "transport_success": success,
            "target_entry_condition": target_condition,
            "target_stable_steps": self._target_stable_count,
            "grasp_retained": grasped,
            "dropped_after_lift": dropped,
            "horizontal_target_error": distance,
            "cube_height": cube_height,
            "safe_height_margin": self.safe_height_margin,
            "minimum_cube_height": self._minimum_cube_height,
            "minimum_table_clearance": self._minimum_clearance,
            "requested_action": requested.copy(),
            "transformed_action": transformed.copy(),
            "transport_prior_action": prior.copy(),
            "transport_residual": residual.copy(),
            "transport_prior_alignment": alignment,
            "reward_terms": {
                "transport_progress": float(progress),
                "low_height_penalty": float(height_penalty),
                "drop_penalty": float(drop),
                "residual_penalty": float(residual_penalty),
                "terminal_transport_success": float(terminal),
            },
            "termination_reason": (
                "sustained_target_entry"
                if success
                else "drop"
                if dropped
                else "time_limit"
                if truncated
                else info.get("termination_reason")
            ),
        }
        return observation, reward, terminated, truncated, info


class CompositeGraspLiftTransportCurriculum(CompositeGraspLiftCurriculum):
    """Phase-aware grasp, lift, and structured-prior transport curriculum.

    The action meaning stays identical to Phase 3C before transport and to
    Phase 4.1 after a stable 5 cm lift.  Descent, release, and placement are
    deliberately absent.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        transport_residual_action_scale: float = 0.20,
        target_tolerance_m: float = 0.025,
        safe_lift_height_m: float = 0.04,
        sustained_target_steps: int = 5,
        transport_progress_scale: float = 1.0,
        low_height_penalty_scale: float = 2.0,
        transport_drop_penalty: float = 5.0,
        transport_terminal_bonus: float = 5.0,
        transport_residual_penalty_scale: float = 0.02,
        **grasp_lift_kwargs,
    ) -> None:
        super().__init__(env, **grasp_lift_kwargs)
        self.transport_residual_action_scale = float(transport_residual_action_scale)
        self.target_tolerance_m = float(target_tolerance_m)
        self.safe_lift_height_m = float(safe_lift_height_m)
        self.sustained_target_steps = int(sustained_target_steps)
        self.transport_progress_scale = float(transport_progress_scale)
        self.low_height_penalty_scale = float(low_height_penalty_scale)
        self.transport_drop_penalty = float(transport_drop_penalty)
        self.transport_terminal_bonus = float(transport_terminal_bonus)
        self.transport_residual_penalty_scale = float(transport_residual_penalty_scale)
        self._transport_joint_target = np.zeros(6, dtype=np.float64)
        self._transport_initial_distance = 1.0
        self._transport_previous_distance = 1.0
        self._transport_target_stable_count = 0
        self._transport_safe_cube_z = 0.0
        self._transport_minimum_height = np.inf
        self._transport_start_step = -1
        self._stable_lift_achieved = False

    @property
    def safe_height_margin(self) -> float:
        return float(self.unwrapped._task_state.object_position[2] - self._transport_safe_cube_z)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self._transport_joint_target.fill(0.0)
        self._transport_initial_distance = self._transport_previous_distance = 1.0
        self._transport_target_stable_count = 0
        self._transport_safe_cube_z = 0.0
        self._transport_minimum_height = np.inf
        self._transport_start_step = -1
        self._stable_lift_achieved = False
        info["curriculum"] = "composite_grasp_lift_transport"
        return observation, info

    def _begin_transport(self) -> None:
        state = self.unwrapped._task_state
        tcp_target = state.tcp_position.copy()
        tcp_target[:2] += state.target_position[:2] - state.object_position[:2]
        self._transport_joint_target = solve_position_ik(
            self.unwrapped, tcp_target, keep_gripper_vertical=False
        )
        self._transport_joint_target[-1] = self.unwrapped.joint_positions[-1]
        distance = float(np.linalg.norm(state.object_position[:2] - state.target_position[:2]))
        self._transport_initial_distance = max(distance, 1e-6)
        self._transport_previous_distance = distance
        self._transport_target_stable_count = 0
        self._transport_safe_cube_z = float(self.unwrapped._initial_cube_height + self.safe_lift_height_m)
        self._transport_minimum_height = float(state.object_position[2] - self.unwrapped._initial_cube_height)
        self._transport_start_step = self._policy_steps
        self._stable_lift_achieved = True
        self.phase = "transport"

    def transport_prior_action(self) -> np.ndarray:
        error = self._transport_joint_target[:5] - self.unwrapped.joint_positions[:5]
        return np.clip(error / self.unwrapped.joint_delta_scales[:5], -1.0, 1.0).astype(np.float32)

    def _transport_action(self, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prior = self.transport_prior_action()
        residual = requested[1:6] * self.transport_residual_action_scale
        transformed = np.zeros(6, dtype=np.float32)
        transformed[:5] = float(requested[0]) * prior + residual
        transformed[-1] = -1.0
        return transformed, prior, residual

    def project_action(self, action: np.ndarray, phase: str | None = None) -> np.ndarray:
        selected_phase = self.phase if phase is None else phase
        requested = np.asarray(action, dtype=np.float32)
        if selected_phase == "transport":
            return self._transport_action(requested)[0]
        return super().project_action(requested, selected_phase)

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32)
        if self.phase != "transport":
            observation, reward, terminated, truncated, info = super().step(requested)
            if info.get("grasp_lift_success", False):
                self._begin_transport()
                terminated = False
                truncated = False
                info.update(
                    {
                        "is_success": False,
                        "transport_success": False,
                        "stable_lift_achieved": True,
                        "transport_phase_switched": True,
                        "policy_phase": "transport",
                        "transport_joint_target": self._transport_joint_target.copy(),
                        "termination_reason": None,
                    }
                )
            else:
                info["transport_phase_switched"] = False
                info["stable_lift_achieved"] = self._stable_lift_achieved
                info["transport_success"] = False
            return observation, reward, terminated, truncated, info

        transformed, prior, residual = self._transport_action(requested)
        observation, _base_reward, base_terminated, _base_truncated, info = self.env.step(transformed)
        self._policy_steps += 1
        state = self.unwrapped._task_state
        grasped = bool(state.is_grasped and state.left_finger_contact and state.right_finger_contact)
        distance = float(np.linalg.norm(state.object_position[:2] - state.target_position[:2]))
        progress = (
            self.transport_progress_scale
            * (self._transport_previous_distance - distance)
            / self._transport_initial_distance
            if grasped and self.safe_height_margin >= 0.0
            else 0.0
        )
        height_penalty = -self.low_height_penalty_scale * max(-self.safe_height_margin, 0.0)
        dropped = not grasped
        drop = -self.transport_drop_penalty if dropped else 0.0
        residual_penalty = -self.transport_residual_penalty_scale * float(np.linalg.norm(residual))
        condition = bool(grasped and self.safe_height_margin >= 0.0 and distance <= self.target_tolerance_m)
        self._transport_target_stable_count = self._transport_target_stable_count + 1 if condition else 0
        success = self._transport_target_stable_count >= self.sustained_target_steps
        terminal = self.transport_terminal_bonus if success else 0.0
        reward = float(progress + height_penalty + drop + residual_penalty + terminal)
        self._transport_previous_distance = distance
        cube_height = float(state.object_position[2] - self.unwrapped._initial_cube_height)
        self._transport_minimum_height = min(self._transport_minimum_height, cube_height)
        prior_norm = float(np.linalg.norm(prior))
        arm_norm = float(np.linalg.norm(transformed[:5]))
        alignment = (
            float(np.dot(transformed[:5], prior) / (arm_norm * prior_norm))
            if arm_norm > 1e-8 and prior_norm > 1e-8
            else 1.0
        )
        terminated = bool(success or dropped or base_terminated)
        truncated = self._policy_steps >= self.episode_steps
        info = {
            **info,
            "is_success": success,
            "grasp_lift_success": self._stable_lift_achieved,
            "stable_lift_achieved": self._stable_lift_achieved,
            "transport_success": success,
            "policy_phase_before": "transport",
            "policy_phase": "transport",
            "transport_phase_switched": False,
            "target_entry_condition": condition,
            "target_stable_steps": self._transport_target_stable_count,
            "grasp_retained": grasped,
            "dropped_after_grasp": dropped,
            "horizontal_target_error": distance,
            "cube_height": cube_height,
            "safe_height_margin": self.safe_height_margin,
            "minimum_cube_height": self._transport_minimum_height,
            "requested_action": requested.copy(),
            "projected_action": transformed.copy(),
            "transport_prior_action": prior.copy(),
            "transport_residual": residual.copy(),
            "transport_prior_alignment": alignment,
            "reward_terms": {
                "transport_progress": float(progress),
                "low_height_penalty": float(height_penalty),
                "drop_penalty": float(drop),
                "residual_penalty": float(residual_penalty),
                "terminal_transport_success": float(terminal),
            },
            "termination_reason": (
                "sustained_target_entry"
                if success
                else "drop"
                if dropped
                else "time_limit"
                if truncated
                else info.get("termination_reason")
            ),
        }
        return observation, reward, terminated, truncated, info
