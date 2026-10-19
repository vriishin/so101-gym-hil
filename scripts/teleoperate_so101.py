"""Teleoperate SO-101 with Gym-HIL keyboard or gamepad intervention controls."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("keyboard", "gamepad"), default="keyboard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--controller-config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suffix = "Keyboard" if args.control == "keyboard" else "Gamepad"
    env = gym.make(
        f"gym_hil/SO101PickPlace{suffix}-v0",
        render_mode="human",
        observation_mode="state",
        episode_length_s=max(10, args.max_steps / 10),
        controller_config_path=args.controller_config,
    )
    try:
        env.reset(seed=args.seed)
        neutral_action = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        for _ in range(args.max_steps):
            _, _, terminated, truncated, info = env.step(neutral_action)
            env.unwrapped.render()
            if terminated or truncated:
                print(f"episode ended: {info.get('termination_reason')}")
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
