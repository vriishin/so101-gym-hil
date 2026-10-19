"""SO-101 MuJoCo pick-and-place environment.

This first implementation establishes the stable Gymnasium API and normalized joint-delta control.
Task predicates and configurable reward shaping are added in the next milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.rewards import PickPlaceReward, RewardConfig, TaskConfig, TaskState
from gym_hil.rewards.task_state import (
    grasped,
    in_target,
    instantaneously_stable,
    lifted,
    out_of_bounds,
    released,
)

SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

SO101_HOME = np.asarray((0.0, -0.55, 0.95, 0.35, 0.0, 0.65), dtype=np.float64)
SO101_EASY_PREGRASP = np.asarray((-0.000045, 0.374799, -0.186313, 1.328483, -0.000065, 1.0), dtype=np.float64)
CUBE_SAMPLING_BOUNDS = np.asarray(((0.18, -0.220), (0.235, -0.155)), dtype=np.float64)
TARGET_SAMPLING_BOUNDS = np.asarray(((0.18, -0.355), (0.235, -0.290)), dtype=np.float64)
MIN_CUBE_TARGET_DISTANCE = 0.10
MAX_RESET_ATTEMPTS = 100


@dataclass(frozen=True)
class SO101EnvConfig:
    """Validated simulation and control settings for the SO-101 task."""

    image_size: tuple[int, int] = (128, 128)
    control_hz: int = 10
    physics_hz: int = 500
    episode_length_s: float = 10.0
    joint_delta_scale: float = 0.05
    gripper_delta_scale: float = 0.10
    min_cube_target_distance_m: float = MIN_CUBE_TARGET_DISTANCE
    max_reset_attempts: int = MAX_RESET_ATTEMPTS

    def __post_init__(self) -> None:
        if self.control_hz <= 0 or self.physics_hz <= 0:
            raise ValueError("control_hz and physics_hz must be positive")
        if self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be an integer multiple of control_hz")
        if self.episode_length_s <= 0:
            raise ValueError("episode_length_s must be positive")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain two positive integers")
        if self.joint_delta_scale <= 0 or self.gripper_delta_scale <= 0:
            raise ValueError("action delta scales must be positive")
        if self.min_cube_target_distance_m <= 0:
            raise ValueError("min_cube_target_distance_m must be positive")
        if self.max_reset_attempts <= 0:
            raise ValueError("max_reset_attempts must be positive")


class SO101PickPlaceEnv(gym.Env[dict[str, Any], np.ndarray]):
    """Gymnasium environment for normalized SO-101 joint-delta control."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(
        self,
        render_mode: Literal["rgb_array", "human"] | None = None,
        observation_mode: Literal["pixels_state", "state"] = "pixels_state",
        reward_mode: str = "sparse",
        control_mode: Literal["joint_delta"] = "joint_delta",
        image_size: tuple[int, int] = (128, 128),
        control_hz: int = 10,
        physics_hz: int = 500,
        episode_length_s: float = 10.0,
        joint_delta_scale: float = 0.05,
        gripper_delta_scale: float = 0.10,
        randomize_reset: bool = True,
        easy_reset: bool = False,
        easy_reset_perturbation_m: float = 0.0,
        easy_target_perturbation_m: float = 0.0,
        domain_randomization: bool = False,
        seed: int | None = None,
        reward_config: RewardConfig | None = None,
        task_config: TaskConfig | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "rgb_array", "human"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if observation_mode not in {"pixels_state", "state"}:
            raise ValueError(f"Unsupported observation_mode: {observation_mode}")
        if control_mode != "joint_delta":
            raise ValueError("Milestone 0 supports only control_mode='joint_delta'")
        if reward_mode not in {"sparse", "naive_dense", "staged_dense", "potential"}:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")
        if domain_randomization:
            raise ValueError("Domain randomization is not implemented yet")

        self.config = SO101EnvConfig(
            image_size=image_size,
            control_hz=control_hz,
            physics_hz=physics_hz,
            episode_length_s=episode_length_s,
            joint_delta_scale=joint_delta_scale,
            gripper_delta_scale=gripper_delta_scale,
        )
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.randomize_reset = randomize_reset
        self.easy_reset = easy_reset
        self.easy_reset_perturbation_m = float(easy_reset_perturbation_m)
        self.easy_target_perturbation_m = float(easy_target_perturbation_m)
        if self.easy_reset_perturbation_m < 0 or self.easy_target_perturbation_m < 0:
            raise ValueError("easy reset perturbations must be non-negative")
        self._constructor_seed = seed
        self._step_count = 0
        self._max_steps = round(episode_length_s * control_hz)
        self._previous_action = np.zeros(6, dtype=np.float32)
        self._human_viewer: Any | None = None
        self.task_config = task_config or TaskConfig(control_hz=control_hz)
        if self.task_config.control_hz != control_hz:
            raise ValueError("task_config.control_hz must equal environment control_hz")
        self.reward_config = reward_config or RewardConfig(mode=reward_mode)
        if self.reward_config.mode != reward_mode:
            raise ValueError("reward_config.mode must equal reward_mode")
        self._reward = PickPlaceReward(self.reward_config, self.task_config.stable_steps_required)
        self._task_state: TaskState | None = None
        self._initial_cube_height = 0.0
        self._stable_steps = 0
        self._invalid_collision_count = 0

        scene_path = Path(__file__).parents[1] / "assets" / "so101" / "scene_pick_place.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.model.opt.timestep = 1.0 / physics_hz
        self.data = mujoco.MjData(self.model)
        self._substeps = physics_hz // control_hz
        self._joint_ids = np.asarray([self.model.joint(name).id for name in SO101_JOINT_NAMES])
        self._qpos_addresses = np.asarray([self.model.jnt_qposadr[joint_id] for joint_id in self._joint_ids])
        self._actuator_ids = np.asarray([self.model.actuator(name).id for name in SO101_JOINT_NAMES])
        self._control_ranges = self.model.actuator_ctrlrange[self._actuator_ids].copy()
        self._camera_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in ("front", "wrist")
        }
        self._cube_geom_id = self.model.geom("cube_geom").id
        self._table_geom_id = self.model.geom("table").id
        self._moving_jaw_body_id = self.model.body("moving_jaw_so101_v1").id
        self._gripper_body_id = self.model.body("gripper").id
        self._robot_body_ids = {
            self.model.body(name).id
            for name in (
                "base",
                "shoulder",
                "upper_arm",
                "lower_arm",
                "wrist",
                "gripper",
                "moving_jaw_so101_v1",
            )
        }
        self._grasp_weld_id = self.model.equality("grasp_assist").id
        self._renderers: dict[str, mujoco.Renderer] = {}
        self.metadata = {**type(self).metadata, "render_fps": control_hz}

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        state_space = spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        if observation_mode == "pixels_state":
            height, width = image_size
            image_space = spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8)
            self.observation_space = spaces.Dict(
                {"pixels": spaces.Dict({"front": image_space, "wrist": image_space}), "state": state_space}
            )
        else:
            self.observation_space = spaces.Dict({"state": state_space})

    @property
    def num_actuated_joints(self) -> int:
        """Number of policy-controlled SO-101 joints."""

        return len(self._actuator_ids)

    @property
    def joint_control_ranges(self) -> np.ndarray:
        """Return a defensive copy of actuator target limits."""

        return self._control_ranges.copy()

    @property
    def cube_sampling_bounds(self) -> np.ndarray:
        """Axis-aligned XY bounds used for cube reset sampling."""

        return CUBE_SAMPLING_BOUNDS.copy()

    @property
    def target_sampling_bounds(self) -> np.ndarray:
        """Axis-aligned XY bounds used for target reset sampling."""

        return TARGET_SAMPLING_BOUNDS.copy()

    @property
    def joint_positions(self) -> np.ndarray:
        """Return the six current actuator joint positions."""

        return self.data.qpos[self._qpos_addresses].copy()

    @property
    def joint_delta_scales(self) -> np.ndarray:
        """Return per-action joint delta scales in MJCF coordinates."""

        scales = np.full(6, self.config.joint_delta_scale, dtype=np.float64)
        scales[-1] = self.config.gripper_delta_scale
        return scales

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset robot, task bodies, counters, and deterministic RNG state."""

        effective_seed = self._constructor_seed if seed is None else seed
        super().reset(seed=effective_seed, options=options)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_addresses] = np.clip(
            SO101_HOME, self._control_ranges[:, 0], self._control_ranges[:, 1]
        )
        if self.easy_reset:
            self.data.qpos[self._qpos_addresses] = np.clip(
                SO101_EASY_PREGRASP, self._control_ranges[:, 0], self._control_ranges[:, 1]
            )
        if options and "initial_joint_perturbation" in options:
            perturbation = np.asarray(options["initial_joint_perturbation"], dtype=np.float64)
            if perturbation.shape != (6,):
                raise ValueError("initial_joint_perturbation must have shape (6,)")
            if not np.all(np.isfinite(perturbation)) or np.any(np.abs(perturbation) > 0.05):
                raise ValueError("initial_joint_perturbation must be finite and within +/-0.05")
            self.data.qpos[self._qpos_addresses] = np.clip(
                self.data.qpos[self._qpos_addresses] + perturbation,
                self._control_ranges[:, 0],
                self._control_ranges[:, 1],
            )
        self.data.ctrl[self._actuator_ids] = self.data.qpos[self._qpos_addresses]

        cube_xy, target_xy = self._sample_reset_positions()
        cube_qpos_adr = self.model.jnt_qposadr[self.model.joint("cube_joint").id]
        self.data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = (*cube_xy, 0.018)
        self.data.mocap_pos[self.model.body("target").mocapid] = (*target_xy, 0.002)

        self._step_count = 0
        self._previous_action.fill(0)
        self._stable_steps = 0
        self._invalid_collision_count = 0
        self.data.eq_active[self._grasp_weld_id] = 0
        mujoco.mj_forward(self.model, self.data)
        self._initial_cube_height = float(self.data.body("cube").xpos[2])
        self._task_state = self._extract_task_state(update_counters=False)
        self._reward.reset(self._task_state)
        observation = self._observation()
        info = {
            "seed": effective_seed,
            "cube_position": self.data.body("cube").xpos.copy(),
            "target_position": self.data.body("target").xpos.copy(),
            "initial_joint_positions": self.data.qpos[self._qpos_addresses].copy(),
            "task_stage": self._reward.task_stage,
        }
        return observation, info

    def _sample_reset_positions(self) -> tuple[np.ndarray, np.ndarray]:
        if self.easy_reset:
            # Valid curriculum start: cube is immediately below the home TCP and
            # the target remains well separated on the table.
            cube = np.asarray((0.2965, 0.0))
            if self.easy_reset_perturbation_m:
                cube += self.np_random.uniform(
                    -self.easy_reset_perturbation_m,
                    self.easy_reset_perturbation_m,
                    size=2,
                )
            target = np.asarray((0.20, -0.18))
            if self.easy_target_perturbation_m:
                target += self.np_random.uniform(
                    -self.easy_target_perturbation_m,
                    self.easy_target_perturbation_m,
                    size=2,
                )
            return cube, target
        if not self.randomize_reset:
            return np.asarray((0.22, -0.18)), np.asarray((0.20, -0.32))
        for _ in range(self.config.max_reset_attempts):
            cube_xy = self.np_random.uniform(CUBE_SAMPLING_BOUNDS[0], CUBE_SAMPLING_BOUNDS[1])
            target_xy = self.np_random.uniform(TARGET_SAMPLING_BOUNDS[0], TARGET_SAMPLING_BOUNDS[1])
            if np.linalg.norm(cube_xy - target_xy) >= self.config.min_cube_target_distance_m:
                return cube_xy, target_xy
        raise RuntimeError(f"Could not sample a valid reset in {self.config.max_reset_attempts} attempts")

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply a clipped normalized joint delta and advance MuJoCo."""

        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}")
        normalized = np.clip(action, self.action_space.low, self.action_space.high)
        scales = np.full(6, self.config.joint_delta_scale, dtype=np.float64)
        scales[-1] = self.config.gripper_delta_scale
        previous_targets = self.data.ctrl[self._actuator_ids].copy()
        targets = previous_targets + normalized * scales
        executed_targets = np.clip(targets, self._control_ranges[:, 0], self._control_ranges[:, 1])
        self.data.ctrl[self._actuator_ids] = executed_targets
        mujoco.mj_step(self.model, self.data, nstep=self._substeps)
        self._update_grasp_assist()

        self._step_count += 1
        if self._task_state is None:
            raise RuntimeError("step() called before reset()")
        previous_state = self._task_state
        state = self._extract_task_state(update_counters=True)
        self._task_state = state
        failure_reason = self._failure_reason(state)
        terminated = bool(state.success or failure_reason is not None)
        truncated = self._step_count >= self._max_steps
        executed_action = ((executed_targets - previous_targets) / scales).astype(np.float32)
        reward_result = self._reward.compute(
            previous_state,
            state,
            executed_action,
            self._previous_action,
            terminated,
            truncated,
        )
        self._previous_action = executed_action
        info = {
            "is_success": state.success,
            "task_stage": self._reward.task_stage,
            "reward_mode": self.reward_mode,
            "reward_terms": reward_result.terms,
            "reach_distance": state.reach_distance,
            "target_distance": state.target_distance,
            "object_lift_height": state.object_lift_height,
            "is_grasped": state.is_grasped,
            "is_lifted": state.is_lifted,
            "object_in_target": state.object_in_target,
            "is_released": state.is_released,
            "is_stable": state.is_stable,
            "stable_steps": state.stable_steps,
            "invalid_collision_count": state.invalid_collision_count,
            "executed_action": executed_action,
            "termination_reason": "success"
            if state.success
            else failure_reason or ("time_limit" if truncated else None),
        }
        if reward_result.potential_previous is not None:
            info["potential_previous"] = reward_result.potential_previous
            info["potential_current"] = reward_result.potential_current
        return self._observation(), reward_result.total, terminated, truncated, info

    def _contact_flags(self) -> tuple[bool, bool]:
        if self.task_config.grasp_assist and self.data.eq_active[self._grasp_weld_id]:
            return True, True
        moving_jaw_contact = False
        fixed_jaw_contact = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self._cube_geom_id not in {contact.geom1, contact.geom2}:
                continue
            other_geom = contact.geom2 if contact.geom1 == self._cube_geom_id else contact.geom1
            other_body = int(self.model.geom_bodyid[other_geom])
            moving_jaw_contact |= other_body == self._moving_jaw_body_id
            fixed_jaw_contact |= other_body == self._gripper_body_id
        return bool(moving_jaw_contact), bool(fixed_jaw_contact)

    def _update_grasp_assist(self) -> None:
        if not self.task_config.grasp_assist:
            return
        gripper_position = float(self.data.qpos[self._qpos_addresses[-1]])
        if self.data.eq_active[self._grasp_weld_id]:
            if gripper_position >= self.task_config.gripper_open_threshold:
                self.data.eq_active[self._grasp_weld_id] = 0
                mujoco.mj_forward(self.model, self.data)
            return

        left_contact, right_contact = self._contact_flags()
        reach_distance = float(
            np.linalg.norm(self.data.body("cube").xpos - self.data.site("gripperframe").xpos)
        )
        if not grasped(
            left_contact,
            right_contact,
            gripper_position,
            reach_distance,
            self.task_config,
        ):
            return

        inverse_position = np.zeros(3, dtype=np.float64)
        inverse_quaternion = np.zeros(4, dtype=np.float64)
        relative_position = np.zeros(3, dtype=np.float64)
        relative_quaternion = np.zeros(4, dtype=np.float64)
        gripper = self.data.body("gripper")
        cube = self.data.body("cube")
        mujoco.mju_negPose(inverse_position, inverse_quaternion, gripper.xpos, gripper.xquat)
        mujoco.mju_mulPose(
            relative_position,
            relative_quaternion,
            inverse_position,
            inverse_quaternion,
            cube.xpos,
            cube.xquat,
        )
        # MuJoCo weld equality data stores anchor[0:3], relpose position[3:6],
        # relpose quaternion[6:10], and torque scale[10].
        self.model.eq_data[self._grasp_weld_id, 3:6] = relative_position
        self.model.eq_data[self._grasp_weld_id, 6:10] = relative_quaternion
        self.data.eq_active[self._grasp_weld_id] = 1
        mujoco.mj_forward(self.model, self.data)

    def _current_invalid_collisions(self) -> int:
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self._table_geom_id not in {contact.geom1, contact.geom2}:
                continue
            other_geom = contact.geom2 if contact.geom1 == self._table_geom_id else contact.geom1
            if int(self.model.geom_bodyid[other_geom]) in self._robot_body_ids:
                count += 1
        return count

    def _extract_task_state(self, *, update_counters: bool) -> TaskState:
        object_position = self.data.body("cube").xpos.copy()
        target_position = self.data.body("target").xpos.copy()
        tcp_position = self.data.site("gripperframe").xpos.copy()
        cube_cvel = self.data.body("cube").cvel.copy()
        gripper_cvel = self.data.body("gripper").cvel.copy()
        left_contact, right_contact = self._contact_flags()
        gripper_position = float(self.data.qpos[self._qpos_addresses[-1]])
        reach_distance = float(np.linalg.norm(object_position - tcp_position))
        object_lift_height = float(object_position[2] - self._initial_cube_height)
        object_in_target = in_target(object_position, target_position, self.task_config)
        is_grasped = grasped(
            left_contact,
            right_contact,
            gripper_position,
            reach_distance,
            self.task_config,
        )
        is_lifted = lifted(object_lift_height, self.task_config)
        is_released = released(
            object_in_target,
            gripper_position,
            left_contact,
            right_contact,
            self.task_config,
        )
        stable_now = instantaneously_stable(
            object_in_target,
            cube_cvel[3:],
            cube_cvel[:3],
            self.task_config,
        )
        stable_after_release = stable_now and is_released
        if update_counters:
            self._stable_steps = self._stable_steps + 1 if stable_after_release else 0
            self._invalid_collision_count += self._current_invalid_collisions()
        is_stable = stable_after_release and self._stable_steps >= self.task_config.stable_steps_required
        is_out_of_bounds = out_of_bounds(object_position, self.task_config)
        return TaskState(
            tcp_position=tcp_position,
            tcp_velocity=gripper_cvel[3:].copy(),
            object_position=object_position,
            object_velocity=cube_cvel[3:].copy(),
            object_angular_velocity=cube_cvel[:3].copy(),
            target_position=target_position,
            gripper_position=gripper_position,
            left_finger_contact=left_contact,
            right_finger_contact=right_contact,
            object_in_target=object_in_target,
            object_lift_height=object_lift_height,
            reach_distance=reach_distance,
            target_distance=float(np.linalg.norm(object_position - target_position)),
            is_grasped=is_grasped,
            is_lifted=is_lifted,
            is_released=is_released,
            is_stable=is_stable,
            stable_steps=self._stable_steps,
            success=bool(is_released and is_stable),
            invalid_collision_count=self._invalid_collision_count,
            out_of_bounds=is_out_of_bounds,
        )

    def _failure_reason(self, state: TaskState) -> str | None:
        if self.task_config.terminate_on_out_of_bounds and state.out_of_bounds:
            return "out_of_bounds"
        if state.invalid_collision_count >= self.task_config.invalid_collision_limit:
            return "invalid_collision"
        return None

    def render(self) -> np.ndarray | None:
        """Render the front camera or synchronize a passive human viewer."""

        if self.render_mode == "human":
            if self._human_viewer is None:
                import mujoco.viewer

                self._human_viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._human_viewer.sync()
            return None
        return self._render_camera("front")

    def _render_camera(self, camera: str) -> np.ndarray:
        renderer = self._renderers.get(camera)
        if renderer is None:
            height, width = self.config.image_size
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[camera] = renderer
        renderer.update_scene(self.data, camera=self._camera_ids[camera])
        return np.asarray(renderer.render(), dtype=np.uint8)

    def _observation(self) -> dict[str, Any]:
        qpos = self.data.qpos[self._qpos_addresses]
        dof_addresses = np.asarray([self.model.jnt_dofadr[joint_id] for joint_id in self._joint_ids])
        state = np.concatenate((qpos, self.data.qvel[dof_addresses])).astype(np.float32)
        observation: dict[str, Any] = {"state": state}
        if self.observation_mode == "pixels_state":
            observation["pixels"] = {
                "front": self._render_camera("front"),
                "wrist": self._render_camera("wrist"),
            }
        return observation

    def get_privileged_state(self) -> dict[str, np.ndarray]:
        """Return simulator-only task state for debugging and state-policy smoke tests."""

        return {
            "joint_positions": self.data.qpos[self._qpos_addresses].copy(),
            "joint_velocities": self.data.qvel[
                [self.model.jnt_dofadr[joint_id] for joint_id in self._joint_ids]
            ].copy(),
            "tcp_position": self.data.site("gripperframe").xpos.copy(),
            "cube_position": self.data.body("cube").xpos.copy(),
            "cube_velocity": self.data.body("cube").cvel.copy(),
            "target_position": self.data.body("target").xpos.copy(),
        }

    def close(self) -> None:
        """Release all MuJoCo renderer contexts."""

        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        if self._human_viewer is not None:
            self._human_viewer.close()
            self._human_viewer = None
