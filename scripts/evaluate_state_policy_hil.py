"""Evaluate a state policy with a delayed scripted HIL recovery intervention."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import SAC

from gym_hil.controllers.so101_scripted import ScriptedPickPlacePolicy
from gym_hil.training.state_sac import make_state_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--takeover-step", type=int, default=40)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    env = make_state_env(config, args.seed_start)
    model = SAC.load(args.checkpoint, env=env, device=str(config["sac"]["device"]))
    autonomous = assisted = 0
    try:
        for episode in range(args.episodes):
            seed = args.seed_start + episode
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            info = {}
            for _ in range(args.takeover_step):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            was_autonomous = bool(info.get("is_success", False))
            if was_autonomous:
                autonomous += 1
                assisted += 1
                result = {"success": True}
            elif terminated or truncated:
                result = {"success": False}
            else:
                try:
                    result = ScriptedPickPlacePolicy(env.unwrapped).run()
                except RuntimeError as error:
                    result = {"success": False, "error": str(error)}
                assisted += int(result["success"])
            print(
                {
                    "episode": episode,
                    "seed": seed,
                    "autonomous_success": was_autonomous,
                    "assisted_success": bool(result["success"]),
                    "intervened": not was_autonomous,
                }
            )
    finally:
        env.close()
    print(
        {
            "episodes": args.episodes,
            "autonomous_success_rate": autonomous / args.episodes,
            "hil_assisted_success_rate": assisted / args.episodes,
            "takeover_step": args.takeover_step,
        }
    )
    minimum = float(config.get("demonstrations", {}).get("minimum_success_rate", 0.5))
    return 0 if assisted / args.episodes >= minimum else 1


if __name__ == "__main__":
    raise SystemExit(main())
