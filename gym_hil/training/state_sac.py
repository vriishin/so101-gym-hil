"""Shared construction, evaluation, and logging helpers for state SAC."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as torch_functional
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import gym_hil  # noqa: F401
from gym_hil.controllers.so101_scripted import ScriptedPickPlacePolicy
from gym_hil.rewards import RewardConfig
from gym_hil.wrappers.state_policy_wrapper import PrivilegedStateWrapper


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_state_env(config: dict[str, Any], seed: int, *, render_mode: str | None = None) -> gym.Env:
    """Build the privileged training adapter around the unchanged base action API."""

    env_config = config["env"]
    reward_mode = config["reward_mode"]
    env = gym.make(
        config["env_id"],
        render_mode=render_mode,
        observation_mode="state",
        reward_mode=reward_mode,
        reward_config=RewardConfig(
            mode=reward_mode,
            gamma=float(config["sac"]["gamma"]),
            **config.get("reward", {}),
        ),
        randomize_reset=bool(env_config["randomize_reset"]),
        easy_reset=bool(env_config.get("easy_reset", False)),
        episode_length_s=float(env_config["episode_length_s"]),
        joint_delta_scale=float(env_config["joint_delta_scale"]),
        gripper_delta_scale=float(env_config["gripper_delta_scale"]),
    )
    env = PrivilegedStateWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def evaluate(
    env: gym.Env,
    *,
    episodes: int,
    seeds: list[int],
    model: SAC | None = None,
    random_policy: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate a random policy or deterministic SAC policy on fixed seeds."""

    if not random_policy and model is None:
        raise ValueError("model is required unless random_policy=True")
    rows = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seeds[episode % len(seeds)])
        terminated = truncated = False
        episode_return = 0.0
        length = 0
        final_info: dict[str, Any] = {}
        stage_flags = {
            "reached_cube": False,
            "grasped_cube": False,
            "lifted_cube": False,
            "entered_target": False,
            "released_in_target": False,
            "stable_success": False,
        }
        reward_term_totals: dict[str, float] = {}
        maximum_stage = 0
        stage_order = {
            "reach": 0,
            "grasp": 1,
            "lift": 2,
            "transport": 3,
            "release": 4,
            "stability": 5,
            "success": 6,
        }
        while not (terminated or truncated):
            if random_policy:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, final_info = env.step(action)
            episode_return += float(reward)
            length += 1
            stage_flags["reached_cube"] |= (
                final_info["reach_distance"] <= env.unwrapped.task_config.grasp_distance_m
            )
            stage_flags["grasped_cube"] |= bool(final_info["is_grasped"])
            stage_flags["lifted_cube"] |= bool(final_info["is_lifted"])
            stage_flags["entered_target"] |= bool(final_info["object_in_target"])
            stage_flags["released_in_target"] |= bool(
                final_info["object_in_target"] and final_info["is_released"]
            )
            stage_flags["stable_success"] |= bool(final_info["is_success"])
            maximum_stage = max(maximum_stage, stage_order[final_info["task_stage"]])
            for name, value in final_info["reward_terms"].items():
                reward_term_totals[name] = reward_term_totals.get(name, 0.0) + float(value)
        rows.append(
            {
                "episode": episode,
                "seed": seeds[episode % len(seeds)],
                "return": episode_return,
                "length": length,
                "success": bool(final_info.get("is_success", False)),
                "final_stage": final_info.get("task_stage", "unknown"),
                "termination_reason": final_info.get("termination_reason"),
                "maximum_stage": maximum_stage,
                **stage_flags,
                **{f"reward_total_{name}": value for name, value in reward_term_totals.items()},
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate evaluation rows without hiding per-episode data."""

    return {
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_length": float(np.mean([row["length"] for row in rows])),
        "average_maximum_stage": float(np.mean([row["maximum_stage"] for row in rows])),
        **{
            f"{name}_rate": float(np.mean([row[name] for row in rows]))
            for name in (
                "reached_cube",
                "grasped_cube",
                "lifted_cube",
                "entered_target",
                "released_in_target",
                "stable_success",
            )
        },
    }


def collect_scripted_demonstrations(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, dict]]]:
    """Collect successful privileged scripted transitions on the training distribution."""

    demo_config = config["demonstrations"]
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    transitions = []
    successes = 0
    for episode in range(int(demo_config["episodes"])):
        seed = int(demo_config["seed_start"]) + episode % int(demo_config["seed_pool_size"])
        base = gym.make(
            config["env_id"],
            observation_mode="state",
            reward_mode=config["reward_mode"],
            reward_config=RewardConfig(mode=config["reward_mode"], gamma=float(config["sac"]["gamma"])),
            randomize_reset=bool(config["env"]["randomize_reset"]),
            episode_length_s=100,
            joint_delta_scale=0.06,
            gripper_delta_scale=0.03,
        )
        episode_transitions = []

        def record(
            observation,
            action,
            reward,
            next_observation,
            done,
            info,
            destination=episode_transitions,
        ):
            destination.append((observation, action, reward, next_observation, done, info.copy()))

        try:
            base.reset(seed=seed)
            result = ScriptedPickPlacePolicy(base.unwrapped, transition_callback=record).run()
            if result["success"]:
                successes += 1
                transitions.extend(episode_transitions)
                observations.extend(item[0] for item in episode_transitions)
                actions.extend(item[1] for item in episode_transitions)
        finally:
            base.close()
    if not observations:
        raise RuntimeError("Scripted demonstration collection produced no successful episodes")
    success_rate = successes / int(demo_config["episodes"])
    if success_rate < 0.8:
        raise RuntimeError(f"Scripted demonstration success rate too low: {success_rate:.3f}")
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32), transitions


def behavior_clone_actor(
    model: SAC,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    updates: int,
    batch_size: int,
    seed: int,
) -> list[float]:
    """Initialize the SAC actor with supervised scripted actions."""

    rng = np.random.default_rng(seed)
    device = model.device
    losses = []
    model.actor.train()
    for _ in range(updates):
        indices = rng.integers(0, len(observations), size=min(batch_size, len(observations)))
        observation_tensor = torch.as_tensor(observations[indices], device=device)
        action_tensor = torch.as_tensor(actions[indices], device=device)
        predicted_actions = model.actor(observation_tensor, deterministic=True)
        loss = torch_functional.mse_loss(predicted_actions, action_tensor)
        model.actor.optimizer.zero_grad()
        loss.backward()
        model.actor.optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def seed_replay_buffer(model: SAC, transitions: list[tuple]) -> None:
    """Insert scripted transitions into the SB3 replay buffer before online learning."""

    for observation, action, reward, next_observation, done, info in transitions:
        model.replay_buffer.add(
            observation[None, :],
            next_observation[None, :],
            action[None, :],
            np.asarray([reward], dtype=np.float32),
            np.asarray([done], dtype=np.float32),
            [info],
        )


def append_eval_csv(path: Path, rows: list[dict[str, Any]], *, step: int, policy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "step",
        "policy",
        "episode",
        "seed",
        "return",
        "length",
        "success",
        "final_stage",
        "termination_reason",
        "maximum_stage",
        "reached_cube",
        "grasped_cube",
        "lifted_cube",
        "entered_target",
        "released_in_target",
        "stable_success",
    ]
    reward_fields = sorted({key for row in rows for key in row if key.startswith("reward_total_")})
    fieldnames = base_fields + reward_fields
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({"step": step, "policy": policy, **row})


class LocalEvaluationCallback(BaseCallback):
    """Periodically checkpoint and write evaluation summaries to local files."""

    def __init__(self, config: dict[str, Any], output_dir: Path) -> None:
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.eval_interval = int(config["evaluation"]["interval_steps"])
        self.checkpoint_interval = int(config["checkpoint_interval_steps"])
        self.eval_env = make_state_env(config, int(config["seed"]) + 10_000)
        self.eval_seeds = [int(seed) for seed in config["evaluation"]["seeds"]]

    def _write_summary(self, payload: dict[str, Any]) -> None:
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step % self.checkpoint_interval == 0:
            self.model.save(self.output_dir / "checkpoints" / f"step_{step}")
        if step % self.eval_interval == 0:
            rows = evaluate(
                self.eval_env,
                episodes=int(self.config["evaluation"]["episodes"]),
                seeds=self.eval_seeds,
                model=self.model,
            )
            append_eval_csv(self.output_dir / "eval_metrics.csv", rows, step=step, policy="sac")
            self._write_summary({"step": step, "policy": "sac", **summarize(rows)})
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()
