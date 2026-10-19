"""Run the Milestone 5 visual SAC/RLPD pipeline and diagnostic experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from record_visual_demonstrations import make_visual_env
from train_visual_bc import closed_loop_evaluation

from gym_hil.training.visual_sac import MixedVisualReplay, VisualSACAgent
from gym_hil.wrappers.visual_policy_wrapper import assert_no_privileged_leakage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/visual_milestone5.yaml"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--run-name", default="diagnostic_10k")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = config["rlpd"]
    total_steps = int(args.steps or settings["total_steps"])
    output = Path(config["output_dir"]) / "rlpd" / args.run_name
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    bc_path = Path(config["output_dir"]) / "behavior_cloning" / "best.pt"
    agent = VisualSACAgent.from_bc(bc_path, settings, settings["device"])
    replay = MixedVisualReplay(
        Path(config["dataset_dir"]),
        int(settings["replay_capacity"]),
        float(settings["demonstration_fraction"]),
        int(config["seed"]),
    )
    seeds = list(map(int, settings["evaluation_seeds"]))
    initial = closed_loop_evaluation(config, agent, seeds, settings["device"])
    agent.save(checkpoints / "initial.pt", {"step": 0, "evaluation": initial})
    best_success = float(initial["stable_success_rate"])
    best_path = checkpoints / "best.pt"
    selected_path = checkpoints / "selected.pt"
    agent.save(best_path, {"step": 0, "evaluation": initial})

    env, _ = make_visual_env(config, 20000)
    observation, _ = env.reset(seed=20000)
    assert_no_privileged_leakage(observation)
    latest_losses: dict[str, float] = {}
    episode = 0
    try:
        for step in range(1, total_steps + 1):
            action = agent.explore(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            assert_no_privileged_leakage(next_observation)
            done = bool(terminated or truncated)
            replay.add(observation, action, reward, next_observation, done)
            observation = next_observation
            if step % int(settings["update_every"]) == 0:
                latest_losses = agent.update(replay, int(settings["batch_size"]))
            if done:
                episode += 1
                observation, _ = env.reset(seed=20000 + episode)
            if step % int(settings["evaluation_interval"]) == 0 or step == total_steps:
                evaluation = closed_loop_evaluation(config, agent, seeds, settings["device"])
                row = {
                    "step": step,
                    "optimization_steps": agent.updates,
                    "replay_size": len(replay),
                    "online_replay_size": len(replay.online),
                    "demonstration_size": replay.demonstration_size,
                    "sampled_demonstration_fraction": replay.last_sampled_demonstration_fraction,
                    **latest_losses,
                    **evaluation,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                agent.save(checkpoints / f"step_{step}.pt", row)
                if evaluation["stable_success_rate"] > best_success:
                    best_success = float(evaluation["stable_success_rate"])
                    agent.save(best_path, row)
                print(json.dumps(row))
    finally:
        env.close()
    agent.save(checkpoints / "final.pt", {"step": total_steps, **latest_losses})

    selected = VisualSACAgent.load(best_path, settings, settings["device"])
    selected.save(
        selected_path,
        {"selected_from": str(best_path), "best_success_rate": best_success},
    )
    selected = VisualSACAgent.load(selected_path, settings, settings["device"])
    reloaded = closed_loop_evaluation(config, selected, seeds, settings["device"])
    healthy = bool(agent.updates > 0 and len(replay.online) == total_steps)
    report = {
        "steps": total_steps,
        "pipeline_healthy": healthy,
        "optimization_steps": agent.updates,
        "replay_size": len(replay),
        "online_replay_size": len(replay.online),
        "demonstration_size": replay.demonstration_size,
        "initial_bc_evaluation": initial,
        "best_success_rate": best_success,
        "best_checkpoint": str(best_path),
        "selected_checkpoint": str(selected_path),
        "final_checkpoint": str(checkpoints / "final.pt"),
        "reloaded_selected_evaluation": reloaded,
    }
    (output / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if healthy and reloaded["stable_success_rate"] > 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
