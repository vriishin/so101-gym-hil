"""Train and closed-loop evaluate a leakage-safe dual-camera visual actor."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from record_visual_demonstrations import make_visual_env
from torch.utils.data import DataLoader

from gym_hil.training.visual_policy import (
    VisualActor,
    VisualDemonstrationDataset,
    load_visual_actor,
    save_visual_actor,
)


def validation_metrics(model, loader, device):
    errors: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for front, wrist, proprio, actions, stages in loader:
            prediction = model(front.to(device), wrist.to(device), proprio.to(device)).cpu().numpy()
            absolute = np.abs(prediction - actions.numpy())
            for index, stage in enumerate(stages):
                errors.setdefault(stage, []).append(absolute[index])
    report = {}
    for stage, values in errors.items():
        array = np.asarray(values)
        report[stage] = {
            "samples": len(array),
            "mae": float(array.mean()),
            "rmse": float(np.sqrt(np.mean(array**2))),
            "per_action_mae": array.mean(0).tolist(),
        }
    all_values = np.concatenate([np.asarray(values) for values in errors.values()])
    report["all"] = {
        "samples": len(all_values),
        "mae": float(all_values.mean()),
        "rmse": float(np.sqrt(np.mean(all_values**2))),
        "per_action_mae": all_values.mean(0).tolist(),
    }
    return report


def closed_loop_evaluation(config, model, seeds, device="cpu"):
    env, curriculum = make_visual_env(config, seeds[0])
    stage_totals = Counter()
    failures = Counter()
    drops = 0
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            reached = {name: False for name in ("reach", "grasp", "lift", "target", "release", "success")}
            info = {}
            while not (terminated or truncated):
                action = model.predict(observation, device=device)
                observation, _, terminated, truncated, info = env.step(action)
                state = curriculum.unwrapped._task_state
                reached["reach"] |= state.reach_distance <= curriculum.unwrapped.task_config.grasp_distance_m
                reached["grasp"] |= bool(info.get("sustained_grasp_achieved", False))
                reached["lift"] |= bool(info.get("stable_lift_achieved", False))
                reached["target"] |= bool(info.get("transport_success", False))
                reached["release"] |= bool(info.get("release_success", False))
                reached["success"] |= bool(info.get("stable_placement_success", False))
                drops += int(bool(info.get("dropped_after_grasp", False)))
            for name, value in reached.items():
                stage_totals[name] += int(value)
            if reached["success"]:
                failures["success"] += 1
            else:
                first_missing = next(name for name in reached if not reached[name])
                failures[first_missing] += 1
    finally:
        env.close()
    count = len(seeds)
    return {
        "episodes": count,
        "reach_rate": stage_totals["reach"] / count,
        "grasp_rate": stage_totals["grasp"] / count,
        "lift_rate": stage_totals["lift"] / count,
        "target_entry_rate": stage_totals["target"] / count,
        "release_rate": stage_totals["release"] / count,
        "stable_success_rate": stage_totals["success"] / count,
        "drop_rate": drops / count,
        "failure_distribution": dict(failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training/visual_milestone5.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = config["behavior_cloning"]
    output = Path(config["output_dir"]) / "behavior_cloning"
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    torch.manual_seed(int(settings["seed"]))
    np.random.seed(int(settings["seed"]))
    episode_count = int(config["demonstration_episodes"])
    validation_count = round(episode_count * float(settings["validation_fraction"]))
    train_indices = list(range(episode_count - validation_count))
    validation_indices = list(range(episode_count - validation_count, episode_count))
    dataset_dir = Path(config["dataset_dir"])
    train_dataset = VisualDemonstrationDataset(dataset_dir, train_indices)
    validation_dataset = VisualDemonstrationDataset(dataset_dir, validation_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=int(settings["batch_size"]), shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=int(settings["batch_size"]), shuffle=False, num_workers=0
    )
    device = settings["device"]
    model = VisualActor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    action_loss_weights = torch.as_tensor(
        settings.get("action_loss_weights", [1.0] * 6), dtype=torch.float32, device=device
    )
    if action_loss_weights.shape != (6,) or torch.any(action_loss_weights <= 0):
        raise ValueError("behavior_cloning.action_loss_weights must contain six positive values")
    metrics_path = output / "training_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_loss = float("inf")
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        losses = []
        for front, wrist, proprio, actions, _ in train_loader:
            prediction = model(front.to(device), wrist.to(device), proprio.to(device))
            squared_error = (prediction - actions.to(device)) ** 2
            loss = torch.mean(squared_error * action_loss_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.item()))
        phase_metrics = validation_metrics(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
            "validation": phase_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if phase_metrics["all"]["rmse"] < best_loss:
            best_loss = phase_metrics["all"]["rmse"]
            save_visual_actor(
                output / "best.pt",
                model,
                {
                    "epoch": epoch,
                    "validation": phase_metrics,
                    "observation_keys": [
                        "observation.images.front",
                        "observation.images.wrist",
                        "observation.state",
                    ],
                    "action_shape": [6],
                    "action_loss_weights": action_loss_weights.cpu().tolist(),
                },
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_mse": row["train_mse"],
                    "validation_rmse": phase_metrics["all"]["rmse"],
                }
            )
        )
    save_visual_actor(output / "final.pt", model, {"epoch": int(settings["epochs"])})
    selected, metadata = load_visual_actor(output / "best.pt", device)
    seeds = list(map(int, settings["evaluation_seeds"]))
    closed_loop = closed_loop_evaluation(config, selected, seeds, device)
    reloaded, _ = load_visual_actor(output / "best.pt", device)
    reloaded_evaluation = closed_loop_evaluation(config, reloaded, seeds, device)
    report = {
        "train_episodes": len(train_indices),
        "validation_episodes": len(validation_indices),
        "train_transitions": len(train_dataset),
        "validation_transitions": len(validation_dataset),
        "best_checkpoint_metadata": metadata,
        "closed_loop": closed_loop,
        "reloaded_closed_loop": reloaded_evaluation,
    }
    (output / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
