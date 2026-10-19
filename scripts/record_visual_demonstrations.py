"""Record leakage-safe visual demonstrations from the frozen state policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import SAC

import gym_hil  # noqa: F401
from gym_hil.wrappers.placement_curriculum import FullTaskCurriculum
from gym_hil.wrappers.state_policy_wrapper import full_task_state_vector
from gym_hil.wrappers.transport_curriculum import CompositeGraspLiftTransportCurriculum
from gym_hil.wrappers.visual_policy_wrapper import VisualObservationAdapter, assert_no_privileged_leakage


def make_visual_env(config: dict, seed: int, perturbation: float | None = None):
    settings = config["env"]
    size = tuple(map(int, config["image_size"]))
    reset_perturbation = float(config["reset_perturbation_m"] if perturbation is None else perturbation)
    base = gym.make(
        "gym_hil/SO101PickPlaceBase-v0",
        observation_mode="pixels_state",
        image_size=size,
        reward_mode="sparse",
        easy_reset=True,
        easy_reset_perturbation_m=reset_perturbation,
        easy_target_perturbation_m=reset_perturbation,
        randomize_reset=False,
        episode_length_s=80.0,
        joint_delta_scale=0.05,
        gripper_delta_scale=0.10,
    )
    transport = CompositeGraspLiftTransportCurriculum(
        base,
        episode_steps=int(settings["episode_steps"]),
        lift_direction=tuple(settings["lift_direction"]),
        lift_action_scale=float(settings["lift_action_scale"]),
        residual_action_scale=0.20,
    )
    curriculum = FullTaskCurriculum(
        transport,
        placement_episode_steps=int(settings["placement_episode_steps"]),
        residual_action_scale=float(settings["residual_action_scale"]),
        episode_steps=int(settings["descent_episode_steps"]),
        horizontal_tolerance_m=float(settings["horizontal_tolerance_m"]),
        release_height_tolerance_m=float(settings["release_height_tolerance_m"]),
        maximum_downward_velocity_m_s=float(settings["maximum_downward_velocity_m_s"]),
        sustained_descent_steps=int(settings["sustained_descent_steps"]),
    )
    env = VisualObservationAdapter(curriculum)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env, curriculum


def episode_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "front": np.asarray([row["front"] for row in rows], dtype=np.uint8),
        "wrist": np.asarray([row["wrist"] for row in rows], dtype=np.uint8),
        "proprio": np.asarray([row["proprio"] for row in rows], dtype=np.float32),
        "policy_actions": np.asarray([row["policy_action"] for row in rows], dtype=np.float32),
        "executed_actions": np.asarray([row["executed_action"] for row in rows], dtype=np.float32),
        "rewards": np.asarray([row["reward"] for row in rows], dtype=np.float32),
        "dones": np.asarray([row["done"] for row in rows], dtype=np.uint8),
        "diagnostic_stages": np.asarray([row["stage"] for row in rows], dtype="U12"),
        "reached_cube": np.asarray([row["reached"] for row in rows], dtype=np.uint8),
        "grasped_cube": np.asarray([row["grasped"] for row in rows], dtype=np.uint8),
        "lifted_cube": np.asarray([row["lifted"] for row in rows], dtype=np.uint8),
        "entered_target": np.asarray([row["entered"] for row in rows], dtype=np.uint8),
        "released_in_target": np.asarray([row["released"] for row in rows], dtype=np.uint8),
        "stable_success": np.asarray([row["success"] for row in rows], dtype=np.uint8),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/visual_milestone5.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["dataset_dir"])
    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    teacher = SAC.load(config["state_teacher_checkpoint"], device="cpu")
    env, curriculum = make_visual_env(config, int(config["seed"]))
    requested = int(config["demonstration_episodes"])
    successes = attempts = transitions = 0
    lengths: list[int] = []
    phase_counts: Counter[str] = Counter()
    min_front_std = min_wrist_std = float("inf")
    try:
        while successes < requested and attempts < requested * 2:
            observation, _ = env.reset(seed=18500 + attempts)
            rows: list[dict] = []
            terminated = truncated = False
            while not (terminated or truncated):
                assert_no_privileged_leakage(observation)
                stage = curriculum.phase
                state_observation = full_task_state_vector(curriculum)
                policy_action = teacher.predict(state_observation, deterministic=True)[0]
                next_observation, reward, terminated, truncated, info = env.step(policy_action)
                state = curriculum.unwrapped._task_state
                front = observation["observation.images.front"]
                wrist = observation["observation.images.wrist"]
                min_front_std = min(min_front_std, float(np.std(front)))
                min_wrist_std = min(min_wrist_std, float(np.std(wrist)))
                rows.append(
                    {
                        "front": front.copy(),
                        "wrist": wrist.copy(),
                        "proprio": observation["observation.state"].copy(),
                        "policy_action": np.asarray(policy_action, dtype=np.float32),
                        "executed_action": np.asarray(info["executed_action"], dtype=np.float32),
                        "reward": float(reward),
                        "done": bool(terminated or truncated),
                        "stage": stage,
                        "reached": state.reach_distance <= curriculum.unwrapped.task_config.grasp_distance_m,
                        "grasped": bool(info.get("sustained_grasp_achieved", False)),
                        "lifted": bool(info.get("stable_lift_achieved", False)),
                        "entered": bool(info.get("transport_success", False)),
                        "released": bool(info.get("release_success", False)),
                        "success": bool(info.get("stable_placement_success", False)),
                    }
                )
                observation = next_observation
            attempts += 1
            if not rows[-1]["success"]:
                continue
            path = episodes_dir / f"episode_{successes:03d}.npz"
            np.savez_compressed(path, **episode_arrays(rows))
            phase_counts.update(row["stage"] for row in rows)
            transitions += len(rows)
            lengths.append(len(rows))
            successes += 1
    finally:
        env.close()
    if successes < requested:
        raise RuntimeError(f"Recorded only {successes}/{requested} successful visual episodes")
    schema = {
        "policy_observation_keys": [
            "observation.images.front",
            "observation.images.wrist",
            "observation.state",
        ],
        "front": {"shape": [*config["image_size"], 3], "dtype": "uint8"},
        "wrist": {"shape": [*config["image_size"], 3], "dtype": "uint8"},
        "proprio": {
            "shape": [12],
            "dtype": "float32",
            "contents": "six joint positions followed by six joint velocities",
        },
        "policy_actions": {"shape": [6], "purpose": "BC target before phase action mapping"},
        "executed_actions": {"shape": [6], "purpose": "actual bounded base-environment action"},
        "diagnostic_only": [
            "diagnostic_stages",
            "reached_cube",
            "grasped_cube",
            "lifted_cube",
            "entered_target",
            "released_in_target",
            "stable_success",
            "rewards",
            "dones",
        ],
        "privileged_policy_features": [],
    }
    report = {
        "successful_episodes": successes,
        "attempts": attempts,
        "transitions": transitions,
        "mean_episode_length": float(np.mean(lengths)),
        "phase_transition_counts": dict(phase_counts),
        "minimum_front_pixel_std": min_front_std,
        "minimum_wrist_pixel_std": min_wrist_std,
        "all_actions_six_dimensional": True,
        "schema": schema,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
