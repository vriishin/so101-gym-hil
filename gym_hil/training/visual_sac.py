"""Leakage-safe visual SAC/RLPD utilities for Milestone 5.

The image encoders are initialized by behavior cloning and frozen during the
short diagnostic run. SAC fine-tunes the stochastic action head against fresh
twin critics while retaining the demonstrated visual representation.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gym_hil.training.visual_policy import VisualActor, load_visual_actor


class VisualTwinCritic(nn.Module):
    def __init__(self, feature_dim: int = 96, hidden_dim: int = 128) -> None:
        super().__init__()

        def network() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim + 6, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = network()
        self.q2 = network()

    def forward(self, features: torch.Tensor, actions: torch.Tensor):
        inputs = torch.cat((features, actions), dim=1)
        return self.q1(inputs), self.q2(inputs)


class MixedVisualReplay:
    """A compact demonstration buffer plus bounded online transition buffer."""

    _KEYS = ("front", "wrist", "proprio", "policy_actions", "rewards", "dones")

    def __init__(
        self, dataset_dir: Path, capacity: int, demonstration_fraction: float, seed: int = 0
    ) -> None:
        if not 0.0 < demonstration_fraction < 1.0:
            raise ValueError("demonstration_fraction must be between zero and one")
        self.demonstration_fraction = float(demonstration_fraction)
        self.online: deque[tuple] = deque(maxlen=int(capacity))
        self.rng = np.random.default_rng(seed)
        arrays: dict[str, list[np.ndarray]] = {key: [] for key in self._KEYS}
        next_indices: list[np.ndarray] = []
        offset = 0
        for path in sorted((dataset_dir / "episodes").glob("episode_*.npz")):
            with np.load(path, allow_pickle=False) as episode:
                length = len(episode["policy_actions"])
                for key in arrays:
                    arrays[key].append(episode[key])
                next_indices.append(np.minimum(np.arange(length) + 1, length - 1) + offset)
                offset += length
        if not next_indices:
            raise FileNotFoundError(f"No visual demonstrations found under {dataset_dir}")
        self.demo = {key: np.concatenate(values) for key, values in arrays.items()}
        self.demo_next_indices = np.concatenate(next_indices)
        self.last_sampled_demonstration_fraction = 0.0

    @property
    def demonstration_size(self) -> int:
        return len(self.demo["policy_actions"])

    def __len__(self) -> int:
        return self.demonstration_size + len(self.online)

    def add(
        self, observation: dict, action: np.ndarray, reward: float, next_observation: dict, done: bool
    ) -> None:
        self.online.append(
            (
                observation["observation.images.front"].copy(),
                observation["observation.images.wrist"].copy(),
                observation["observation.state"].copy(),
                np.asarray(action, dtype=np.float32).copy(),
                np.float32(reward),
                next_observation["observation.images.front"].copy(),
                next_observation["observation.images.wrist"].copy(),
                next_observation["observation.state"].copy(),
                np.float32(done),
            )
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if not self.online:
            demo_count, online_count = batch_size, 0
        else:
            demo_count = int(round(batch_size * self.demonstration_fraction))
            online_count = batch_size - demo_count
        indices = self.rng.integers(0, self.demonstration_size, size=demo_count)
        result = {
            "front": [self.demo["front"][indices]],
            "wrist": [self.demo["wrist"][indices]],
            "proprio": [self.demo["proprio"][indices]],
            "actions": [self.demo["policy_actions"][indices]],
            "rewards": [self.demo["rewards"][indices]],
            "next_front": [self.demo["front"][self.demo_next_indices[indices]]],
            "next_wrist": [self.demo["wrist"][self.demo_next_indices[indices]]],
            "next_proprio": [self.demo["proprio"][self.demo_next_indices[indices]]],
            "dones": [self.demo["dones"][indices].astype(np.float32)],
        }
        if online_count:
            online_indices = self.rng.integers(0, len(self.online), size=online_count)
            rows = [self.online[int(index)] for index in online_indices]
            for key, column in zip(result, zip(*rows, strict=False), strict=True):
                result[key].append(np.asarray(column))
        self.last_sampled_demonstration_fraction = demo_count / batch_size
        return {key: np.concatenate(values, axis=0) for key, values in result.items()}


class VisualSACAgent:
    """Small SAC agent with a BC actor and fresh latent-space twin critics."""

    def __init__(self, actor: VisualActor, settings: dict, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        for encoder in (self.actor.front_encoder, self.actor.wrist_encoder, self.actor.proprio_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
        self.log_std = nn.Parameter(torch.full((6,), -3.0, device=self.device))
        self.critic = VisualTwinCritic().to(self.device)
        self.target_critic = VisualTwinCritic().to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            [*self.actor.action_head.parameters(), self.log_std],
            lr=float(settings["actor_learning_rate"]),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(settings["critic_learning_rate"])
        )
        self.gamma = float(settings["gamma"])
        self.tau = float(settings["tau"])
        self.alpha = float(settings["entropy_coefficient"])
        self.updates = 0

    def _tensor(self, value: np.ndarray, dtype=None) -> torch.Tensor:
        return torch.as_tensor(value, device=self.device, dtype=dtype)

    def encode(self, front: np.ndarray, wrist: np.ndarray, proprio: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self.actor.encode(self._tensor(front), self._tensor(wrist), self._tensor(proprio))

    def _distribution(self, features: torch.Tensor):
        logits = self.actor.action_head[:-1](features)
        std = self.log_std.clamp(-5.0, -2.0).exp().expand_as(logits)
        return torch.distributions.Normal(logits, std)

    def sample_actions(self, features: torch.Tensor, deterministic: bool = False):
        distribution = self._distribution(features)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(pre_tanh)
        correction = torch.log(1.0 - actions.square() + 1e-6)
        log_probability = (distribution.log_prob(pre_tanh) - correction).sum(dim=1, keepdim=True)
        return actions, log_probability

    @torch.no_grad()
    def predict(self, observation: dict[str, np.ndarray], device: str | None = None) -> np.ndarray:
        del device
        features = self.encode(
            observation["observation.images.front"][None],
            observation["observation.images.wrist"][None],
            observation["observation.state"][None],
        )
        return self.sample_actions(features, deterministic=True)[0][0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def explore(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        features = self.encode(
            observation["observation.images.front"][None],
            observation["observation.images.wrist"][None],
            observation["observation.state"][None],
        )
        return self.sample_actions(features, deterministic=False)[0][0].cpu().numpy().astype(np.float32)

    def update(self, replay: MixedVisualReplay, batch_size: int) -> dict[str, float]:
        batch = replay.sample(batch_size)
        features = self.encode(batch["front"], batch["wrist"], batch["proprio"])
        next_features = self.encode(batch["next_front"], batch["next_wrist"], batch["next_proprio"])
        actions = self._tensor(batch["actions"], torch.float32)
        rewards = self._tensor(batch["rewards"], torch.float32).unsqueeze(1)
        dones = self._tensor(batch["dones"], torch.float32).unsqueeze(1)
        with torch.no_grad():
            next_actions, next_log_probability = self.sample_actions(next_features)
            next_q1, next_q2 = self.target_critic(next_features, next_actions)
            target = rewards + self.gamma * (1.0 - dones) * (
                torch.minimum(next_q1, next_q2) - self.alpha * next_log_probability
            )
        q1, q2 = self.critic(features, actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        policy_actions, log_probability = self.sample_actions(features)
        policy_q1, policy_q2 = self.critic(features, policy_actions)
        actor_loss = (self.alpha * log_probability - torch.minimum(policy_q1, policy_q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_([*self.actor.action_head.parameters(), self.log_std], 10.0)
        self.actor_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)
        self.updates += 1
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "entropy": float((-log_probability).mean().detach().cpu()),
            "alpha": self.alpha,
        }

    def save(self, path: Path, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "actor": self.actor.state_dict(),
                "log_std": self.log_std.detach().cpu(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "updates": self.updates,
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def from_bc(cls, path: Path, settings: dict, device: str = "cpu") -> "VisualSACAgent":
        actor, _ = load_visual_actor(path, device)
        return cls(actor, settings, device)

    @classmethod
    def load(cls, path: Path, settings: dict, device: str = "cpu") -> "VisualSACAgent":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        actor = VisualActor()
        actor.load_state_dict(checkpoint["actor"])
        agent = cls(actor, settings, device)
        agent.log_std.data.copy_(checkpoint["log_std"].to(agent.device))
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.target_critic.load_state_dict(checkpoint["target_critic"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        agent.updates = int(checkpoint["updates"])
        return agent
