"""Render deterministic state-policy evaluation episodes to MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import yaml
from stable_baselines3 import SAC

from gym_hil.training.state_sac import make_state_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    env = make_state_env(config, args.seeds[0], render_mode="rgb_array")
    model = SAC.load(args.checkpoint, env=env, device=str(config["sac"]["device"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for seed in args.seeds:
            observation, _ = env.reset(seed=seed)
            frames = [env.unwrapped._render_camera("front").copy()]
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                frames.append(env.unwrapped._render_camera("front").copy())
            path = args.output_dir / f"seed_{seed}.mp4"
            iio.imwrite(path, frames, fps=env.unwrapped.metadata["render_fps"])
            print(path, "success=", info.get("is_success"), "stage=", info.get("task_stage"))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
