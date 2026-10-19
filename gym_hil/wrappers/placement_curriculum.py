"""Isolated descent, release, and stable-placement curricula."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from gym_hil.controllers.so101_scripted import solve_position_ik
from gym_hil.wrappers.state_policy_wrapper import unified_grasp_lift_transport_state_vector


class DescentCurriculum(gym.Wrapper):
    """Initialize above the target and descend along an IK waypoint prior."""

    def __init__(
        self,
        env: gym.Env,
        *,
        transport_policy,
        residual_action_scale: float = 0.05,
        episode_steps: int = 75,
        horizontal_tolerance_m: float = 0.027,
        release_height_tolerance_m: float = 0.006,
        maximum_downward_velocity_m_s: float = 0.15,
        sustained_descent_steps: int = 3,
        vertical_progress_scale: float = 1.0,
        horizontal_progress_scale: float = 0.25,
        horizontal_drift_penalty_scale: float = 1.0,
        unsafe_velocity_penalty_scale: float = 1.0,
        collision_drop_penalty: float = 5.0,
        terminal_bonus: float = 5.0,
        residual_penalty_scale: float = 0.02,
    ) -> None:
        super().__init__(env)
        self.transport_policy = transport_policy
        self.residual_action_scale = float(residual_action_scale)
        self.episode_steps = int(episode_steps)
        self.horizontal_tolerance_m = float(horizontal_tolerance_m)
        self.release_height_tolerance_m = float(release_height_tolerance_m)
        self.maximum_downward_velocity_m_s = float(maximum_downward_velocity_m_s)
        self.sustained_descent_steps = int(sustained_descent_steps)
        self.vertical_progress_scale = float(vertical_progress_scale)
        self.horizontal_progress_scale = float(horizontal_progress_scale)
        self.horizontal_drift_penalty_scale = float(horizontal_drift_penalty_scale)
        self.unsafe_velocity_penalty_scale = float(unsafe_velocity_penalty_scale)
        self.collision_drop_penalty = float(collision_drop_penalty)
        self.terminal_bonus = float(terminal_bonus)
        self.residual_penalty_scale = float(residual_penalty_scale)
        self.phase = "descent"
        self._release_joint_target = np.zeros(6, dtype=np.float64)
        self._previous_vertical_error = 1.0
        self._previous_horizontal_error = 1.0
        self._descent_stable_count = 0
        self._policy_steps = 0
        self._initial_collision_count = 0
        self._transport_initialization_steps = 0

    @property
    def release_height(self) -> float:
        state = self.unwrapped._task_state
        return float(state.target_position[2] + self.unwrapped.task_config.cube_half_extent_m)

    @property
    def release_height_error(self) -> float:
        return float(self.unwrapped._task_state.object_position[2] - self.release_height)

    @property
    def descent_stable_fraction(self) -> float:
        return min(self._descent_stable_count / self.sustained_descent_steps, 1.0)

    def set_residual_scale(self, scale: float) -> None:
        if not 0 <= float(scale) <= 0.20:
            raise ValueError("residual scale must be in [0, 0.20]")
        self.residual_action_scale = float(scale)

    def _initialize_waypoint(self) -> None:
        state = self.unwrapped._task_state
        desired_cube = state.target_position.copy()
        desired_cube[2] += self.unwrapped.task_config.cube_half_extent_m
        tcp_target = state.tcp_position + (desired_cube - state.object_position)
        self._release_joint_target = solve_position_ik(
            self.unwrapped, tcp_target, keep_gripper_vertical=False
        )
        self._release_joint_target[-1] = self.unwrapped.joint_positions[-1]
        self._previous_vertical_error = abs(self.release_height_error)
        self._previous_horizontal_error = float(
            np.linalg.norm(state.object_position[:2] - state.target_position[:2])
        )
        self._descent_stable_count = 0
        self._policy_steps = 0
        self._initial_collision_count = state.invalid_collision_count
        self.phase = "descent"

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        transport_info = {}
        for step in range(300):
            policy_observation = unified_grasp_lift_transport_state_vector(self.env)
            action = self.transport_policy.predict(policy_observation, deterministic=True)[0]
            observation, _, _, truncated, transport_info = self.env.step(action)
            if transport_info.get("transport_success", False):
                self._transport_initialization_steps = step + 1
                break
            if truncated:
                raise RuntimeError("Phase 4.2 policy truncated while initializing descent")
        else:
            raise RuntimeError("Phase 4.2 policy failed to initialize a transported grasp")
        self._initialize_waypoint()
        return observation, {
            **info,
            **transport_info,
            "curriculum": "descent_only",
            "policy_phase": self.phase,
            "transport_initialization_steps": self._transport_initialization_steps,
            "release_joint_target": self._release_joint_target.copy(),
        }

    def prior_action(self) -> np.ndarray:
        error = self._release_joint_target[:5] - self.unwrapped.joint_positions[:5]
        return np.clip(error / self.unwrapped.joint_delta_scales[:5], -1.0, 1.0).astype(np.float32)

    def transform_action(self, requested: np.ndarray):
        requested = np.asarray(requested, dtype=np.float32)
        if requested.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {requested.shape}")
        prior = self.prior_action()
        residual = requested[1:6] * self.residual_action_scale
        transformed = np.zeros(6, dtype=np.float32)
        transformed[:5] = float(requested[0]) * prior + residual
        transformed[-1] = -1.0
        return transformed, prior, residual

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32)
        transformed, prior, residual = self.transform_action(requested)
        observation, _, base_terminated, _, info = self.unwrapped.step(transformed)
        self._policy_steps += 1
        state = self.unwrapped._task_state
        horizontal = float(np.linalg.norm(state.object_position[:2] - state.target_position[:2]))
        vertical = abs(self.release_height_error)
        grasped = bool(state.is_grasped and state.left_finger_contact and state.right_finger_contact)
        downward_speed = max(-float(state.object_velocity[2]), 0.0)
        new_collisions = state.invalid_collision_count - self._initial_collision_count
        valid = bool(
            horizontal <= self.horizontal_tolerance_m
            and vertical <= self.release_height_tolerance_m
            and grasped
            and downward_speed <= self.maximum_downward_velocity_m_s
            and new_collisions == 0
        )
        self._descent_stable_count = self._descent_stable_count + 1 if valid else 0
        success = self._descent_stable_count >= self.sustained_descent_steps
        vertical_progress = self.vertical_progress_scale * (self._previous_vertical_error - vertical)
        horizontal_progress = self.horizontal_progress_scale * (self._previous_horizontal_error - horizontal)
        drift = -self.horizontal_drift_penalty_scale * max(horizontal - self.horizontal_tolerance_m, 0.0)
        unsafe = -self.unsafe_velocity_penalty_scale * max(
            downward_speed - self.maximum_downward_velocity_m_s, 0.0
        )
        failure = -self.collision_drop_penalty if (not grasped or new_collisions > 0) else 0.0
        residual_penalty = -self.residual_penalty_scale * float(np.linalg.norm(residual))
        terminal = self.terminal_bonus if success else 0.0
        reward = float(
            vertical_progress + horizontal_progress + drift + unsafe + failure + residual_penalty + terminal
        )
        self._previous_vertical_error = vertical
        self._previous_horizontal_error = horizontal
        terminated = bool(success or not grasped or new_collisions > 0 or base_terminated)
        truncated = self._policy_steps >= self.episode_steps
        info = {
            **info,
            "is_success": success,
            "descent_success": success,
            "valid_release_pose": valid,
            "policy_phase": self.phase,
            "horizontal_target_error": horizontal,
            "release_height_error": self.release_height_error,
            "downward_velocity": downward_speed,
            "grasp_retained": grasped,
            "new_invalid_collisions": new_collisions,
            "descent_stable_steps": self._descent_stable_count,
            "requested_action": requested.copy(),
            "transformed_action": transformed.copy(),
            "descent_prior_action": prior.copy(),
            "descent_residual": residual.copy(),
            "reward_terms": {
                "vertical_progress": float(vertical_progress),
                "horizontal_progress": float(horizontal_progress),
                "horizontal_drift_penalty": float(drift),
                "unsafe_velocity_penalty": float(unsafe),
                "collision_drop_penalty": float(failure),
                "residual_penalty": float(residual_penalty),
                "terminal_descent_success": float(terminal),
            },
            "termination_reason": (
                "valid_descent"
                if success
                else "drop"
                if not grasped
                else "collision"
                if new_collisions > 0
                else "time_limit"
                if truncated
                else info.get("termination_reason")
            ),
        }
        return observation, reward, terminated, truncated, info


class ReleaseCurriculum(gym.Wrapper):
    """Initialize at the release pose, freeze the arm, and learn opening/stability."""

    def __init__(
        self,
        env: DescentCurriculum,
        *,
        episode_steps: int = 50,
        opening_progress_scale: float = 1.0,
        release_bonus: float = 1.0,
        stability_progress_scale: float = 1.0,
        invalid_release_penalty: float = 5.0,
        target_exit_penalty: float = 5.0,
    ) -> None:
        super().__init__(env)
        self.episode_steps = int(episode_steps)
        self.opening_progress_scale = float(opening_progress_scale)
        self.release_bonus = float(release_bonus)
        self.stability_progress_scale = float(stability_progress_scale)
        self.invalid_release_penalty = float(invalid_release_penalty)
        self.target_exit_penalty = float(target_exit_penalty)
        self.phase = "release"
        self._policy_steps = 0
        self._previous_aperture = 0.0
        self._previous_stable_steps = 0
        self._release_awarded = False
        self._released_once = False
        self._target_exit_after_release = False
        self._post_release_steps = 0

    @property
    def release_height_error(self) -> float:
        return self.env.release_height_error

    @property
    def descent_stable_fraction(self) -> float:
        return 1.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        descent_info = {}
        for _ in range(self.env.episode_steps):
            observation, _, _, truncated, descent_info = self.env.step(
                np.asarray((1, 0, 0, 0, 0, 0), dtype=np.float32)
            )
            if descent_info.get("descent_success", False):
                break
            if truncated:
                raise RuntimeError("Teacher descent truncated while initializing release")
        else:
            raise RuntimeError("Teacher descent failed to initialize release pose")
        self.phase = "release"
        self._policy_steps = 0
        self._previous_aperture = float(self.unwrapped._task_state.gripper_position)
        self._previous_stable_steps = 0
        self._release_awarded = False
        self._released_once = False
        self._target_exit_after_release = False
        self._post_release_steps = 0
        return observation, {**info, **descent_info, "curriculum": "release_only", "policy_phase": self.phase}

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32)
        transformed = np.zeros(6, dtype=np.float32)
        transformed[-1] = requested[-1]
        observation, _, base_terminated, _, info = self.unwrapped.step(transformed)
        self._policy_steps += 1
        state = self.unwrapped._task_state
        valid_pose = bool(
            np.linalg.norm(state.object_position[:2] - state.target_position[:2])
            <= self.env.horizontal_tolerance_m
            and abs(self.release_height_error) <= self.env.release_height_tolerance_m
        )
        aperture_progress = (
            self.opening_progress_scale * max(state.gripper_position - self._previous_aperture, 0.0)
            if valid_pose and not self._released_once
            else 0.0
        )
        release_condition = bool(
            state.gripper_position >= self.unwrapped.task_config.gripper_open_threshold
            and not state.is_grasped
            and state.object_in_target
        )
        newly_released = bool(release_condition and not self._release_awarded)
        release_reward = self.release_bonus if newly_released else 0.0
        self._release_awarded |= newly_released
        self._released_once |= release_condition
        self._post_release_steps = self._post_release_steps + 1 if self._released_once else 0
        self.phase = "stability" if self._released_once else "release"
        stability_progress = (
            self.stability_progress_scale
            * max(state.stable_steps - self._previous_stable_steps, 0)
            / self.unwrapped.task_config.stable_steps_required
        )
        premature = -self.invalid_release_penalty if (not valid_pose and not state.is_grasped) else 0.0
        exited = bool(self._released_once and not state.object_in_target)
        self._target_exit_after_release |= exited
        exit_penalty = -self.target_exit_penalty if exited else 0.0
        reward = float(aperture_progress + release_reward + stability_progress + premature + exit_penalty)
        self._previous_aperture = float(state.gripper_position)
        self._previous_stable_steps = state.stable_steps
        success = bool(
            self._released_once
            and self._post_release_steps >= 5
            and state.object_in_target
            and not self._target_exit_after_release
        )
        terminated = bool(success or exited or premature < 0 or (base_terminated and not success))
        truncated = self._policy_steps >= self.episode_steps
        info = {
            **info,
            "is_success": success,
            "release_success": self._released_once,
            "stable_placement_success": success,
            "policy_phase": self.phase,
            "valid_release_pose": valid_pose,
            "cube_retained_in_target": bool(self._released_once and state.object_in_target),
            "target_exit_after_release": self._target_exit_after_release,
            "invalid_release": premature < 0,
            "requested_action": requested.copy(),
            "transformed_action": transformed.copy(),
            "reward_terms": {
                "opening_progress": float(aperture_progress),
                "release_bonus": float(release_reward),
                "stability_progress": float(stability_progress),
                "premature_release_penalty": float(premature),
                "target_exit_penalty": float(exit_penalty),
            },
            "termination_reason": (
                "stable_placement"
                if success
                else "target_exit"
                if exited
                else "invalid_release"
                if premature < 0
                else "time_limit"
                if truncated
                else info.get("termination_reason")
            ),
        }
        return observation, reward, terminated, truncated, info


class CompositePlacementCurriculum(ReleaseCurriculum):
    """Start transported, then expose descent followed by release and stability."""

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.phase = "descent"
        self._policy_steps = 0
        self._previous_aperture = float(self.unwrapped._task_state.gripper_position)
        self._previous_stable_steps = 0
        self._release_awarded = False
        self._released_once = False
        self._target_exit_after_release = False
        self._post_release_steps = 0
        return observation, {**info, "curriculum": "composite_placement", "policy_phase": self.phase}

    def start_from_transported_state(self) -> None:
        """Activate placement without resetting an already transported base state."""

        self.env._initialize_waypoint()
        self.phase = "descent"
        self._policy_steps = 0
        self._previous_aperture = float(self.unwrapped._task_state.gripper_position)
        self._previous_stable_steps = 0
        self._release_awarded = False
        self._released_once = False
        self._target_exit_after_release = False
        self._post_release_steps = 0

    def step(self, action):
        if self.phase == "descent":
            observation, reward, terminated, truncated, info = self.env.step(action)
            if info.get("descent_success", False):
                self.phase = "release"
                terminated = truncated = False
                info.update(
                    {"is_success": False, "placement_phase_switched": True, "policy_phase": self.phase}
                )
            else:
                info["placement_phase_switched"] = False
            return observation, reward, terminated, truncated, info
        if self.phase == "release":
            observation, reward, terminated, truncated, info = super().step(action)
            if info.get("release_success", False):
                self._initialize_retreat()
                terminated = truncated = False
                info.update(
                    {"is_success": False, "placement_phase_switched": True, "policy_phase": "retreat"}
                )
            else:
                info["placement_phase_switched"] = False
            return observation, reward, terminated, truncated, info
        if self.phase == "retreat":
            observation, reward, reached, _, info = self._retreat_step(action)
            if reached:
                self.phase = "stability"
            info.update(
                {"is_success": False, "placement_phase_switched": reached, "policy_phase": self.phase}
            )
            return observation, reward, False, False, info

        requested = np.asarray(action, dtype=np.float32)
        transformed = np.zeros(6, dtype=np.float32)
        transformed[-1] = 1.0
        observation, reward, base_terminated, _, info = self.unwrapped.step(transformed)
        self._policy_steps += 1
        state = self.unwrapped._task_state
        success = bool(state.success)
        exited = not state.object_in_target
        terminated = bool(success or exited or (base_terminated and not success))
        truncated = self._policy_steps >= self.episode_steps
        info.update(
            {
                "is_success": success,
                "release_success": True,
                "stable_placement_success": success,
                "cube_retained_in_target": state.object_in_target,
                "target_exit_after_release": exited,
                "policy_phase": "stability",
                "placement_phase_switched": False,
                "requested_action": requested.copy(),
                "transformed_action": transformed.copy(),
                "termination_reason": "stable_placement"
                if success
                else "target_exit"
                if exited
                else "time_limit"
                if truncated
                else info.get("termination_reason"),
            }
        )
        info["placement_phase_switched"] = False
        return observation, reward, terminated, truncated, info

    def _initialize_retreat(self, height_m: float = 0.06) -> None:
        state = self.unwrapped._task_state
        tcp_target = state.tcp_position.copy()
        tcp_target[2] += float(height_m)
        self._retreat_joint_target = solve_position_ik(
            self.unwrapped, tcp_target, keep_gripper_vertical=False
        )
        self._retreat_joint_target[-1] = self.unwrapped.joint_positions[-1]
        self.phase = "retreat"

    def _retreat_step(self, action: np.ndarray):
        requested = np.asarray(action, dtype=np.float32)
        prior = np.clip(
            (self._retreat_joint_target[:5] - self.unwrapped.joint_positions[:5])
            / self.unwrapped.joint_delta_scales[:5],
            -1.0,
            1.0,
        ).astype(np.float32)
        transformed = np.zeros(6, dtype=np.float32)
        transformed[:5] = float(requested[0]) * prior
        transformed[-1] = 1.0
        observation, reward, _, _, info = self.unwrapped.step(transformed)
        reached = bool(
            np.max(np.abs(self._retreat_joint_target[:5] - self.unwrapped.joint_positions[:5])) <= 0.025
        )
        info.update(
            {
                "retreat_success": reached,
                "policy_phase": "retreat",
                "requested_action": requested.copy(),
                "transformed_action": transformed.copy(),
                "retreat_prior_action": prior.copy(),
            }
        )
        return observation, reward, reached, False, info

    def begin_retreat(self, height_m: float = 0.06) -> None:
        self._initialize_retreat(height_m)

    def retreat_step(self, action: np.ndarray):
        return self._retreat_step(action)


class FullTaskCurriculum(gym.Wrapper):
    """Reconnect Phase 4.2 and placement while preserving both action priors."""

    def __init__(self, env: gym.Env, *, placement_episode_steps: int = 80, **descent_kwargs):
        super().__init__(env)
        self.descent = DescentCurriculum(env, transport_policy=None, **descent_kwargs)
        self.placement = CompositePlacementCurriculum(self.descent, episode_steps=placement_episode_steps)
        self.placement_active = False

    @property
    def phase(self) -> str:
        return self.placement.phase if self.placement_active else self.env.phase

    @property
    def release_height_error(self) -> float:
        state = self.unwrapped._task_state
        release_height = state.target_position[2] + self.unwrapped.task_config.cube_half_extent_m
        return float(state.object_position[2] - release_height)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.placement_active = False
        return observation, {**info, "curriculum": "full_state_task", "policy_phase": self.phase}

    def step(self, action):
        if not self.placement_active:
            observation, reward, terminated, truncated, info = self.env.step(action)
            if info.get("transport_success", False):
                self.placement_active = True
                self.placement.start_from_transported_state()
                terminated = truncated = False
                info.update(
                    {
                        "is_success": False,
                        "full_task_phase_switched": True,
                        "policy_phase": "descent",
                        "termination_reason": None,
                    }
                )
            else:
                info["full_task_phase_switched"] = False
            return observation, reward, terminated, truncated, info
        observation, reward, terminated, truncated, info = self.placement.step(action)
        info["full_task_phase_switched"] = bool(info.get("placement_phase_switched", False))
        return observation, reward, terminated, truncated, info
