"""Privileged waypoint controller used to validate the SO-101 task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np

from gym_hil.envs.so101_pick_place_env import SO101PickPlaceEnv


@dataclass(frozen=True)
class ScriptedPolicyConfig:
    """Waypoint offsets and convergence tolerances for scripted validation."""

    approach_height_m: float = 0.08
    grasp_height_m: float = 0.030
    transport_height_m: float = 0.09
    place_height_m: float = 0.030
    grasp_tcp_offset_m: tuple[float, float] = (-0.025, 0.0)
    pregrasp_gripper_position: float = 1.00
    open_gripper_position: float = 1.35
    closed_gripper_position: float = 0.00
    joint_tolerance: float = 0.025
    max_steps_per_waypoint: int = 120
    settle_steps: int = 8
    alignment_iterations: int = 4
    alignment_tolerance_m: float = 0.004
    release_compensation_m: tuple[float, float] = (-0.005, 0.022)


def solve_position_ik(
    env: SO101PickPlaceEnv,
    target_position: np.ndarray,
    *,
    max_iterations: int = 250,
    tolerance_m: float = 0.002,
    damping: float = 1e-4,
    keep_gripper_vertical: bool = True,
    vertical_tolerance: float = 0.35,
) -> np.ndarray:
    """Solve SO-101 TCP position IK with a damped site Jacobian.

    Gym-HIL's existing EE controller is coupled to the seven-joint Panda and a mocap target. This
    model-specific solver is intentionally limited to offline scripted validation and never changes
    the policy action contract.
    """

    target_position = np.asarray(target_position, dtype=np.float64)
    if target_position.shape != (3,):
        raise ValueError("target_position must have shape (3,)")

    qpos_backup = env.data.qpos.copy()
    qvel_backup = env.data.qvel.copy()
    ctrl_backup = env.data.ctrl.copy()
    arm_qpos_addresses = env._qpos_addresses[:5]
    arm_dof_addresses = np.asarray(
        [env.model.jnt_dofadr[joint_id] for joint_id in env._joint_ids[:5]], dtype=int
    )
    limits = env.joint_control_ranges[:5]
    site_id = env.model.site("gripperframe").id
    jacobian_position = np.zeros((3, env.model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, env.model.nv), dtype=np.float64)
    gripper_body_id = env.model.body("gripper").id
    try:
        for _ in range(max_iterations):
            mujoco.mj_forward(env.model, env.data)
            position_error = target_position - env.data.site("gripperframe").xpos
            gripper_z = env.data.body("gripper").xmat.reshape(3, 3)[:, 2]
            orientation_error = np.cross(gripper_z, np.asarray((0.0, 0.0, 1.0)))
            if np.linalg.norm(position_error) <= tolerance_m and (
                not keep_gripper_vertical or np.linalg.norm(orientation_error) <= vertical_tolerance
            ):
                return env.data.qpos[env._qpos_addresses].copy()
            jacobian_position.fill(0)
            jacobian_rotation.fill(0)
            mujoco.mj_jacSite(env.model, env.data, jacobian_position, None, site_id)
            if keep_gripper_vertical:
                body_jacobian_position = np.zeros((3, env.model.nv), dtype=np.float64)
                mujoco.mj_jacBody(
                    env.model,
                    env.data,
                    body_jacobian_position,
                    jacobian_rotation,
                    gripper_body_id,
                )
                orientation_weight = 0.12
                jacobian = np.vstack(
                    (
                        jacobian_position[:, arm_dof_addresses],
                        orientation_weight * jacobian_rotation[:, arm_dof_addresses],
                    )
                )
                error = np.concatenate((position_error, orientation_weight * orientation_error))
            else:
                jacobian = jacobian_position[:, arm_dof_addresses]
                error = position_error
            lhs = jacobian.T @ jacobian + damping * np.eye(jacobian.shape[1])
            delta = np.linalg.solve(lhs, jacobian.T @ error)
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 0.12:
                delta *= 0.12 / delta_norm
            env.data.qpos[arm_qpos_addresses] = np.clip(
                env.data.qpos[arm_qpos_addresses] + 0.7 * delta,
                limits[:, 0],
                limits[:, 1],
            )
        raise RuntimeError(
            f"IK did not converge: target={target_position}, "
            f"error={np.linalg.norm(target_position - env.data.site('gripperframe').xpos):.4f} m"
        )
    finally:
        env.data.qpos[:] = qpos_backup
        env.data.qvel[:] = qvel_backup
        env.data.ctrl[:] = ctrl_backup
        mujoco.mj_forward(env.model, env.data)


class ScriptedPickPlacePolicy:
    """Finite-state privileged controller for deterministic scene validation."""

    def __init__(
        self,
        env: SO101PickPlaceEnv,
        config: ScriptedPolicyConfig | None = None,
        frame_callback: Callable[[SO101PickPlaceEnv], None] | None = None,
        transition_callback: Callable[[np.ndarray, np.ndarray, float, np.ndarray, bool, dict[str, Any]], None]
        | None = None,
    ) -> None:
        self.env = env
        self.config = config or ScriptedPolicyConfig()
        self.frame_callback = frame_callback
        self.transition_callback = transition_callback
        self.current_phase = "open"

    def _step(self, action: np.ndarray):
        if self.transition_callback is not None:
            from gym_hil.wrappers.state_policy_wrapper import phase_one_hot, privileged_state_vector

            observation = np.concatenate(
                (privileged_state_vector(self.env), phase_one_hot(self.current_phase))
            )
        transition = self.env.step(action)
        if self.frame_callback is not None:
            self.frame_callback(self.env)
        if self.transition_callback is not None:
            next_observation = np.concatenate(
                (privileged_state_vector(self.env), phase_one_hot(self.current_phase))
            )
            _, reward, terminated, truncated, info = transition
            self.transition_callback(
                observation,
                np.asarray(info["executed_action"], dtype=np.float32),
                float(reward),
                next_observation,
                bool(terminated or truncated),
                info,
            )
        return transition

    def _move_joints(self, target: np.ndarray) -> tuple[bool, dict]:
        info: dict = {}
        for _ in range(self.config.max_steps_per_waypoint):
            error = target - self.env.joint_positions
            if np.max(np.abs(error)) <= self.config.joint_tolerance:
                return True, info
            action = np.clip(error / self.env.joint_delta_scales, -1, 1).astype(np.float32)
            _, _, terminated, truncated, info = self._step(action)
            if terminated or truncated:
                return bool(info.get("is_success", False)), info
        return False, info

    def _pose_target(self, xyz: np.ndarray, gripper: float, *, vertical: bool = False) -> np.ndarray:
        target = solve_position_ik(self.env, xyz, keep_gripper_vertical=vertical)
        target[-1] = gripper
        return target

    def _align_cube_to_target(self, gripper: float) -> tuple[bool, dict]:
        """Correct model-relative grasp offsets before release using privileged state."""

        info: dict = {}
        for _ in range(self.config.alignment_iterations):
            state = self.env.get_privileged_state()
            desired_cube = state["target_position"].copy()
            desired_cube[:2] += np.asarray(self.config.release_compensation_m)
            desired_cube[2] += self.env.task_config.cube_half_extent_m
            error = desired_cube - state["cube_position"]
            if np.linalg.norm(error) <= self.config.alignment_tolerance_m:
                return True, info
            tcp_target = state["tcp_position"] + error
            reached, info = self._move_joints(self._pose_target(tcp_target, gripper))
            if not reached:
                return False, info
        state = self.env.get_privileged_state()
        target = state["target_position"].copy()
        target[:2] += np.asarray(self.config.release_compensation_m)
        target[2] += self.env.task_config.cube_half_extent_m
        return bool(np.linalg.norm(target - state["cube_position"]) <= 0.012), info

    def _release_and_wait(self) -> tuple[bool, dict]:
        """Open only until release, then hold still for the post-release stability window."""

        info: dict = {}
        for _ in range(self.config.max_steps_per_waypoint):
            action = np.zeros(6, dtype=np.float32)
            if not info.get("is_released", False):
                action[-1] = 1.0
            _, _, terminated, truncated, info = self._step(action)
            if info.get("is_success", False):
                return True, info
            if terminated or truncated:
                return False, info
            if info.get("is_released", False) and not info.get("object_in_target", False):
                return False, info
        return False, info

    def run(self) -> dict:
        """Execute approach, grasp, lift, transport, release, retreat, and stability phases."""

        cfg = self.config
        privileged = self.env.get_privileged_state()
        cube = privileged["cube_position"].copy()
        target = privileged["target_position"].copy()
        cube_tcp_xy = cube[:2] + np.asarray(cfg.grasp_tcp_offset_m)
        target_tcp_xy = target[:2] + np.asarray(cfg.grasp_tcp_offset_m)
        phases = (
            ("open", self.env.joint_positions[:5], cfg.pregrasp_gripper_position),
            ("approach", np.asarray((*cube_tcp_xy, cfg.approach_height_m)), cfg.pregrasp_gripper_position),
            ("descend", np.asarray((*cube_tcp_xy, cfg.grasp_height_m)), cfg.pregrasp_gripper_position),
            ("grasp", None, cfg.closed_gripper_position),
            ("lift", np.asarray((*cube_tcp_xy, cfg.transport_height_m)), cfg.closed_gripper_position),
            ("transport", np.asarray((*target_tcp_xy, cfg.transport_height_m)), cfg.closed_gripper_position),
            ("place", np.asarray((*target_tcp_xy, cfg.place_height_m)), cfg.closed_gripper_position),
            ("align", None, cfg.closed_gripper_position),
            ("release", None, cfg.open_gripper_position),
            ("retreat", np.asarray((*target_tcp_xy, cfg.transport_height_m)), cfg.open_gripper_position),
        )
        phase_results: dict[str, bool] = {}
        last_target = self.env.joint_positions.copy()
        info: dict = {}
        for name, xyz_or_joints, gripper in phases:
            self.current_phase = name if name != "retreat" else "release"
            if name == "align":
                reached, info = self._align_cube_to_target(gripper)
                phase_results[name] = reached
                if not reached:
                    return {"success": False, "phase_results": phase_results, "info": info}
                continue
            if name == "release":
                reached, info = self._release_and_wait()
                phase_results[name] = reached
                return {
                    "success": bool(info.get("is_success", False)),
                    "phase_results": phase_results,
                    "info": info,
                }
            if name == "open":
                joint_target = np.concatenate((np.asarray(xyz_or_joints), (gripper,)))
            elif xyz_or_joints is None:
                joint_target = last_target.copy()
                joint_target[-1] = gripper
            else:
                joint_target = self._pose_target(
                    np.asarray(xyz_or_joints),
                    gripper,
                    vertical=name == "descend",
                )
            reached, info = self._move_joints(joint_target)
            phase_results[name] = reached
            last_target = joint_target
            if info.get("is_success", False):
                return {"success": True, "phase_results": phase_results, "info": info}
            if not reached:
                return {"success": False, "phase_results": phase_results, "info": info}

        for _ in range(cfg.settle_steps):
            _, _, terminated, truncated, info = self._step(np.zeros(6, dtype=np.float32))
            if terminated or truncated:
                break
        return {"success": bool(info.get("is_success", False)), "phase_results": phase_results, "info": info}
