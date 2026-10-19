"""Train and validate a privileged-state SAC policy for SO-101."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import stable_baselines3
import torch
import yaml
from stable_baselines3 import SAC

from gym_hil.training.state_sac import (
    LocalEvaluationCallback,
    append_eval_csv,
    behavior_clone_actor,
    collect_scripted_demonstrations,
    evaluate,
    make_state_env,
    seed_everything,
    seed_replay_buffer,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reward-mode", choices=("sparse", "naive_dense", "staged_dense", "potential"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def git_metadata() -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    return {"commit": run("git", "rev-parse", "HEAD"), "status": run("git", "status", "--short")}


def main() -> int:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.reward_mode is not None:
        config["reward_mode"] = args.reward_mode
    if args.seed is not None:
        config["seed"] = args.seed
    if args.total_steps is not None:
        config["total_steps"] = args.total_steps
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)

    output_dir = Path(config["output_dir"]) / config["reward_mode"] / f"seed_{config['seed']}"
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    metadata = {
        "python": platform.python_version(),
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git": git_metadata(),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    seed_everything(int(config["seed"]))
    train_env = make_state_env(config, int(config["seed"]))
    eval_env = make_state_env(config, int(config["seed"]) + 10_000)
    eval_seeds = [int(seed) for seed in config["evaluation"]["seeds"]]
    baseline_rows = evaluate(
        eval_env,
        episodes=int(config["evaluation"]["episodes"]),
        seeds=eval_seeds,
        random_policy=True,
    )
    append_eval_csv(output_dir / "eval_metrics.csv", baseline_rows, step=0, policy="random")
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 0, "policy": "random", **summarize(baseline_rows)}) + "\n")

    sac = config["sac"]
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=float(sac["learning_rate"]),
        buffer_size=int(sac["buffer_size"]),
        learning_starts=int(sac["learning_starts"]),
        batch_size=int(sac["batch_size"]),
        gamma=float(sac["gamma"]),
        tau=float(sac["tau"]),
        ent_coef=sac.get("ent_coef", "auto"),
        train_freq=int(sac["train_freq"]),
        gradient_steps=int(sac["gradient_steps"]),
        policy_kwargs={"net_arch": list(sac["net_arch"])},
        seed=int(config["seed"]),
        device=str(sac["device"]),
        verbose=1,
    )
    demo_config = config.get("demonstrations")
    if demo_config and int(demo_config.get("episodes", 0)) > 0:
        demo_observations, demo_actions, demo_transitions = collect_scripted_demonstrations(config)
        bc_losses = behavior_clone_actor(
            model,
            demo_observations,
            demo_actions,
            updates=int(demo_config["behavior_cloning_updates"]),
            batch_size=int(demo_config["behavior_cloning_batch_size"]),
            seed=int(config["seed"]),
        )
        if bool(demo_config.get("replay_seed", True)):
            seed_replay_buffer(model, demo_transitions)
        bc_rows = evaluate(
            eval_env,
            episodes=int(config["evaluation"]["episodes"]),
            seeds=eval_seeds,
            model=model,
        )
        append_eval_csv(output_dir / "eval_metrics.csv", bc_rows, step=0, policy="behavior_cloned")
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": 0,
                        "policy": "behavior_cloned",
                        "demo_transitions": len(demo_transitions),
                        "bc_final_loss": bc_losses[-1],
                        **summarize(bc_rows),
                    }
                )
                + "\n"
            )
    started = time.time()
    callback = LocalEvaluationCallback(config, output_dir)
    model.learn(total_timesteps=int(config["total_steps"]), callback=callback, progress_bar=False)
    model.save(checkpoints / "final")

    reloaded = SAC.load(checkpoints / "final", env=eval_env, device=str(sac["device"]))
    final_rows = evaluate(
        eval_env,
        episodes=int(config["evaluation"]["episodes"]),
        seeds=eval_seeds,
        model=reloaded,
    )
    append_eval_csv(
        output_dir / "eval_metrics.csv", final_rows, step=int(config["total_steps"]), policy="reloaded_sac"
    )
    final_summary = summarize(final_rows)
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "step": int(config["total_steps"]),
                    "policy": "reloaded_sac",
                    "elapsed_seconds": time.time() - started,
                    **final_summary,
                }
            )
            + "\n"
        )
    print({"random": summarize(baseline_rows), "reloaded_sac": final_summary})
    train_env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
