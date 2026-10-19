"""Reload and deterministically evaluate a state SAC checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import SAC

from gym_hil.training.state_sac import evaluate, make_state_env, summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=2000)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    env = make_state_env(config, args.seed_start)
    try:
        model = SAC.load(args.checkpoint, env=env, device=str(config["sac"]["device"]))
        rows = evaluate(
            env,
            episodes=args.episodes,
            seeds=list(range(args.seed_start, args.seed_start + args.episodes)),
            model=model,
        )
        for row in rows:
            print(row)
        print(summarize(rows))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
