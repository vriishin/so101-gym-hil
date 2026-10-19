"""Run the privileged SO-101 waypoint policy for task validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np

import gym_hil  # noqa: F401
from gym_hil.controllers.so101_scripted import ScriptedPickPlacePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Open the MuJoCo human viewer")
    parser.add_argument("--video", type=Path, help="Save the first episode's front-camera MP4")
    parser.add_argument("--fixed-reset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    frames = []
    successes = 0
    metric_names = (
        "reached_cube",
        "grasped_cube",
        "lifted_cube",
        "entered_target",
        "released_in_target",
        "stable_success",
    )
    stage_counts = {name: 0 for name in metric_names}
    maximum_stages = []
    reward_totals: dict[str, float] = {}
    for episode in range(args.episodes):
        env = gym.make(
            "gym_hil/SO101PickPlaceBase-v0",
            observation_mode="state",
            render_mode="human" if args.render else "rgb_array",
            randomize_reset=not args.fixed_reset,
            episode_length_s=100,
            joint_delta_scale=0.06,
            gripper_delta_scale=0.03,
        )
        try:
            seed = args.seed_start + episode
            env.reset(seed=seed)
            episode_metrics = {name: False for name in metric_names}
            episode_reward_terms: dict[str, float] = {}
            max_stage = 0

            def record(
                _observation,
                _action,
                _reward,
                _next_observation,
                _done,
                info,
                metrics=episode_metrics,
                reward_terms=episode_reward_terms,
                base_env=env.unwrapped,
            ):
                nonlocal max_stage
                metrics["reached_cube"] |= info["reach_distance"] <= base_env.task_config.grasp_distance_m
                metrics["grasped_cube"] |= bool(info["is_grasped"])
                metrics["lifted_cube"] |= bool(info["is_lifted"])
                metrics["entered_target"] |= bool(info["object_in_target"])
                metrics["released_in_target"] |= bool(info["object_in_target"] and info["is_released"])
                metrics["stable_success"] |= bool(info["is_success"])
                stage_order = {
                    "reach": 0,
                    "grasp": 1,
                    "lift": 2,
                    "transport": 3,
                    "release": 4,
                    "stability": 5,
                    "success": 6,
                }
                max_stage = max(max_stage, stage_order[info["task_stage"]])
                for name, value in info["reward_terms"].items():
                    reward_terms[name] = reward_terms.get(name, 0.0) + float(value)

            def capture(base_env, current_episode=episode):
                if args.render:
                    base_env.render()
                if args.video is not None and current_episode == 0:
                    frames.append(base_env._render_camera("front").copy())

            result = ScriptedPickPlacePolicy(
                env.unwrapped, frame_callback=capture, transition_callback=record
            ).run()
            successes += int(result["success"])
            for name, achieved in episode_metrics.items():
                stage_counts[name] += int(achieved)
            maximum_stages.append(max_stage)
            for name, value in episode_reward_terms.items():
                reward_totals[name] = reward_totals.get(name, 0.0) + value
            print(
                f"episode={episode} seed={seed} success={result['success']} "
                f"reason={result['info'].get('termination_reason')}"
            )
        finally:
            env.close()

    if args.video is not None and frames:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.video, frames, fps=10)
        print(f"video={args.video}")
    print(f"successes={successes}/{args.episodes} rate={successes / args.episodes:.3f}")
    print(
        "stage_rates=" + " ".join(f"{name}={stage_counts[name] / args.episodes:.3f}" for name in metric_names)
    )
    print(f"average_maximum_stage={np.mean(maximum_stages):.3f}/6")
    print(
        "mean_reward_term_totals="
        + " ".join(f"{name}={value / args.episodes:.6f}" for name, value in sorted(reward_totals.items()))
    )
    return 0 if successes >= 0.8 * args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
