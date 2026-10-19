"""Privileged flat-state adapter for environment/reward learning smoke tests."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

PRIVILEGED_FEATURE_NAMES = (
    *(f"joint_position_{index}" for index in range(6)),
    *(f"joint_velocity_{index}" for index in range(6)),
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "cube_x",
    "cube_y",
    "cube_z",
    "cube_vx",
    "cube_vy",
    "cube_vz",
    "target_x",
    "target_y",
    "target_z",
    "left_finger_contact",
    "right_finger_contact",
)

POLICY_PHASES = ("open", "approach", "descend", "grasp", "lift", "transport", "place", "align", "release")


class PrivilegedStateWrapper(gym.ObservationWrapper):
    """Flatten explicit privileged state for the state-policy SAC smoke test."""

    def __init__(self, env: gym.Env, *, normalize: bool = True) -> None:
        super().__init__(env)
        self.normalize = normalize
        self.feature_names = PRIVILEGED_FEATURE_NAMES
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_names),),
            dtype=np.float32,
        )

    def observation(self, observation: dict) -> np.ndarray:
        del observation
        vector = privileged_state_vector(self.unwrapped)
        if not self.normalize:
            return vector
        scales = np.asarray(
            (
                *([3.2] * 6),
                *([10.0] * 6),
                *([0.5] * 3),
                *([0.5] * 3),
                *([2.0] * 3),
                *([0.5] * 3),
                1.0,
                1.0,
            ),
            dtype=np.float32,
        )
        return vector / scales


def privileged_state_vector(base_env) -> np.ndarray:
    """Build the canonical privileged vector directly from an unwrapped environment."""

    state = base_env._task_state
    if state is None:
        raise RuntimeError("Task state is unavailable before reset")
    privileged = base_env.get_privileged_state()
    flat = np.concatenate(
        (
            privileged["joint_positions"],
            privileged["joint_velocities"],
            state.tcp_position,
            state.object_position,
            state.object_velocity,
            state.target_position,
            np.asarray((state.left_finger_contact, state.right_finger_contact)),
        )
    )
    return flat.astype(np.float32)


def phase_one_hot(phase: str) -> np.ndarray:
    """Encode the privileged controller phase used only by state-policy experiments."""

    encoded = np.zeros(len(POLICY_PHASES), dtype=np.float32)
    encoded[POLICY_PHASES.index(phase)] = 1.0
    return encoded


class PhaseAwarePrivilegedStateWrapper(gym.Wrapper):
    """Add a closed-loop task phase to privileged state to remove imitation ambiguity."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.phase = "open"
        self.feature_names = PRIVILEGED_FEATURE_NAMES + tuple(f"phase_{p}" for p in POLICY_PHASES)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_names),), dtype=np.float32
        )

    def _advance(self) -> None:
        base = self.unwrapped
        state = base._task_state
        if state is None:
            return
        cfg = base.task_config
        tcp = state.tcp_position
        cube = state.object_position
        target = state.target_position
        offset = np.asarray((-0.025, 0.0))
        if self.phase == "open" and base.joint_positions[-1] >= 0.90:
            self.phase = "approach"
        elif self.phase == "approach" and np.linalg.norm(tcp - np.r_[cube[:2] + offset, 0.08]) < 0.050:
            self.phase = "descend"
        elif self.phase == "descend" and np.linalg.norm(tcp - np.r_[cube[:2] + offset, 0.03]) < 0.035:
            self.phase = "grasp"
        elif self.phase == "grasp" and (state.is_grasped or bool(base.data.eq_active[base._grasp_weld_id])):
            self.phase = "lift"
        elif self.phase == "lift" and state.object_lift_height >= 0.045:
            self.phase = "transport"
        elif self.phase == "transport" and np.linalg.norm(cube[:2] - target[:2]) < 0.035:
            self.phase = "place"
        elif self.phase == "place" and abs(cube[2] - (target[2] + cfg.cube_half_extent_m)) < 0.025:
            self.phase = "align"
        elif self.phase == "align" and np.linalg.norm(cube[:2] - target[:2]) < 0.018:
            self.phase = "release"

    def _observation(self) -> np.ndarray:
        self._advance()
        return np.concatenate((privileged_state_vector(self.unwrapped), phase_one_hot(self.phase)))

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self.phase = "open"
        return self._observation(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(), reward, terminated, truncated, info


class UnifiedGraspLiftStateWrapper(gym.ObservationWrapper):
    """Privileged state plus explicit aperture, sustained-grasp, and phase state."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.feature_names = PRIVILEGED_FEATURE_NAMES + (
            "gripper_aperture",
            "sustained_grasp_fraction",
            "phase_grasp",
            "phase_lift",
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_names),),
            dtype=np.float32,
        )

    def observation(self, observation: dict) -> np.ndarray:
        del observation
        return unified_grasp_lift_state_vector(self.env)


def unified_grasp_lift_state_vector(curriculum) -> np.ndarray:
    """Build the exact 30-value Phase 3C observation without a wrapper dependency."""

    base = privileged_state_vector(curriculum.unwrapped)
    scales = np.asarray(
        (
            *([3.2] * 6),
            *([10.0] * 6),
            *([0.5] * 3),
            *([0.5] * 3),
            *([2.0] * 3),
            *([0.5] * 3),
            1.0,
            1.0,
        ),
        dtype=np.float32,
    )
    normalized = base / scales
    fraction = float(
        min(curriculum.sustained_grasp_count, curriculum.sustained_grasp_steps)
        / curriculum.sustained_grasp_steps
    )
    phase = curriculum.phase
    extras = np.asarray(
        (
            curriculum.unwrapped._task_state.gripper_position,
            fraction,
            float(phase == "grasp"),
            float(phase == "lift"),
        ),
        dtype=np.float32,
    )
    return np.concatenate((normalized, extras)).astype(np.float32)


class TransportStateWrapper(gym.ObservationWrapper):
    """Privileged transport state with target displacement and safe-height margin."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.feature_names = PRIVILEGED_FEATURE_NAMES + (
            "cube_to_target_x",
            "cube_to_target_y",
            "grasp_stable",
            "phase_transport",
            "safe_height_margin",
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_names),), dtype=np.float32
        )

    def observation(self, observation: dict) -> np.ndarray:
        del observation
        return transport_state_vector(self.env)


def transport_state_vector(curriculum) -> np.ndarray:
    """Build the exact 31-value Phase 4.1 transport policy observation."""

    base = privileged_state_vector(curriculum.unwrapped)
    scales = np.asarray(
        (
            *([3.2] * 6),
            *([10.0] * 6),
            *([0.5] * 3),
            *([0.5] * 3),
            *([2.0] * 3),
            *([0.5] * 3),
            1.0,
            1.0,
        ),
        dtype=np.float32,
    )
    normalized = base / scales
    state = curriculum.unwrapped._task_state
    displacement = (state.target_position[:2] - state.object_position[:2]) / 0.5
    extras = np.asarray(
        (
            displacement[0],
            displacement[1],
            float(state.is_grasped and state.left_finger_contact and state.right_finger_contact),
            1.0,
            curriculum.safe_height_margin / 0.10,
        ),
        dtype=np.float32,
    )
    return np.concatenate((normalized, extras)).astype(np.float32)


def unified_grasp_lift_transport_state_vector(curriculum) -> np.ndarray:
    """Build the 35-value phase-aware Phase 4.2 policy observation."""

    base = privileged_state_vector(curriculum.unwrapped)
    scales = np.asarray(
        (
            *([3.2] * 6),
            *([10.0] * 6),
            *([0.5] * 3),
            *([0.5] * 3),
            *([2.0] * 3),
            *([0.5] * 3),
            1.0,
            1.0,
        ),
        dtype=np.float32,
    )
    normalized = base / scales
    state = curriculum.unwrapped._task_state
    displacement = (state.target_position[:2] - state.object_position[:2]) / 0.5
    phase = curriculum.phase
    fraction = float(
        min(curriculum.sustained_grasp_count, curriculum.sustained_grasp_steps)
        / curriculum.sustained_grasp_steps
    )
    safe_margin = curriculum.safe_height_margin / 0.10 if phase == "transport" else 0.0
    extras = np.asarray(
        (
            state.gripper_position,
            fraction,
            float(curriculum._stable_lift_achieved),
            displacement[0],
            displacement[1],
            safe_margin,
            float(phase == "grasp"),
            float(phase == "lift"),
            float(phase == "transport"),
        ),
        dtype=np.float32,
    )
    return np.concatenate((normalized, extras)).astype(np.float32)


class UnifiedGraspLiftTransportStateWrapper(gym.ObservationWrapper):
    """Phase-aware privileged observation for unified Phase 4.2 training."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)

    def observation(self, observation: dict) -> np.ndarray:
        del observation
        return unified_grasp_lift_transport_state_vector(self.env)


def placement_state_vector(curriculum) -> np.ndarray:
    """Build the 38-value phase-aware placement observation."""

    base = privileged_state_vector(curriculum.unwrapped)
    scales = np.asarray(
        (
            *([3.2] * 6),
            *([10.0] * 6),
            *([0.5] * 3),
            *([0.5] * 3),
            *([2.0] * 3),
            *([0.5] * 3),
            1.0,
            1.0,
        ),
        dtype=np.float32,
    )
    normalized = base / scales
    state = curriculum.unwrapped._task_state
    displacement = (state.target_position - state.object_position) / 0.5
    phase = curriculum.phase
    extras = np.asarray(
        (
            displacement[0],
            displacement[1],
            displacement[2],
            state.gripper_position,
            float(state.object_in_target),
            float(state.is_grasped),
            curriculum.release_height_error / 0.10,
            state.stable_steps / curriculum.unwrapped.task_config.stable_steps_required,
            float(phase == "descent"),
            float(phase == "release"),
            float(phase == "retreat"),
            float(phase == "stability"),
        ),
        dtype=np.float32,
    )
    return np.concatenate((normalized, extras)).astype(np.float32)


class PlacementStateWrapper(gym.ObservationWrapper):
    """Privileged phase-aware observation for isolated placement learning."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32)

    def observation(self, observation: dict) -> np.ndarray:
        del observation
        return placement_state_vector(self.env)


def full_task_state_vector(curriculum) -> np.ndarray:
    """Build the 42-value phase-aware full-task privileged observation."""

    base = privileged_state_vector(curriculum.unwrapped)
    scales = np.asarray(
        (
            *([3.2] * 6),
            *([10.0] * 6),
            *([0.5] * 3),
            *([0.5] * 3),
            *([2.0] * 3),
            *([0.5] * 3),
            1.0,
            1.0,
        ),
        dtype=np.float32,
    )
    normalized = base / scales
    state = curriculum.unwrapped._task_state
    displacement = (state.target_position - state.object_position) / 0.5
    source = curriculum.env
    grasp_fraction = float(
        min(source.sustained_grasp_count, source.sustained_grasp_steps) / source.sustained_grasp_steps
    )
    phase = curriculum.phase
    phases = ("grasp", "lift", "transport", "descent", "release", "retreat", "stability")
    extras = np.asarray(
        (
            displacement[0],
            displacement[1],
            displacement[2],
            state.gripper_position,
            grasp_fraction,
            float(source._stable_lift_achieved),
            float(curriculum.placement_active),
            curriculum.release_height_error / 0.10,
            state.stable_steps / curriculum.unwrapped.task_config.stable_steps_required,
            *(float(phase == name) for name in phases),
        ),
        dtype=np.float32,
    )
    return np.concatenate((normalized, extras)).astype(np.float32)


class FullTaskStateWrapper(gym.ObservationWrapper):
    """Privileged phase-aware state for unified full-task learning."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32)

    def observation(self, observation: dict) -> np.ndarray:
        del observation
        return full_task_state_vector(self.env)
