"""SAC utilities for demonstration-seeded grasp-and-lift training."""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer


class DemonstrationReplayBuffer(ReplayBuffer):
    """Sample a fixed demonstration fraction while retaining online transitions."""

    def __init__(self, *args, demonstration_fraction: float = 0.35, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0.25 <= demonstration_fraction <= 0.50:
            raise ValueError("demonstration_fraction must be in [0.25, 0.50]")
        self.demonstration_fraction = float(demonstration_fraction)
        self.demonstration_size = 0
        self.last_sampled_demonstration_fraction = 0.0

    def mark_demonstrations(self) -> None:
        if self.full:
            raise RuntimeError("Cannot mark demonstrations after replay wraps")
        self.demonstration_size = int(self.pos)

    def sample(self, batch_size: int, env=None):
        upper = self.buffer_size if self.full else self.pos
        if self.demonstration_size <= 0 or upper <= self.demonstration_size:
            self.last_sampled_demonstration_fraction = 1.0 if self.demonstration_size else 0.0
            return super().sample(batch_size, env=env)
        demonstration_count = int(round(batch_size * self.demonstration_fraction))
        online_count = batch_size - demonstration_count
        demonstration_indices = np.random.randint(0, self.demonstration_size, size=demonstration_count)
        online_indices = np.random.randint(self.demonstration_size, upper, size=online_count)
        indices = np.concatenate((demonstration_indices, online_indices))
        np.random.shuffle(indices)
        self.last_sampled_demonstration_fraction = demonstration_count / batch_size
        return self._get_samples(indices, env=env)


class UnifiedGraspLiftSAC(SAC):
    """SAC with a persistent lower actor LR and capped entropy coefficient."""

    actor_learning_rate: float
    entropy_coefficient_cap: float | None = None
    last_actor_loss: float | None = None
    last_critic_loss: float | None = None
    last_entropy_coefficient: float | None = None

    def set_actor_learning_rate(self, learning_rate: float) -> None:
        self.actor_learning_rate = float(learning_rate)
        for group in self.actor.optimizer.param_groups:
            group["lr"] = self.actor_learning_rate

    def _update_learning_rate(self, optimizers) -> None:
        super()._update_learning_rate(optimizers)
        if hasattr(self, "actor_learning_rate"):
            for group in self.actor.optimizer.param_groups:
                group["lr"] = self.actor_learning_rate

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps, batch_size)
        if self.log_ent_coef is not None and self.entropy_coefficient_cap is not None:
            with torch.no_grad():
                maximum = torch.log(
                    torch.as_tensor(self.entropy_coefficient_cap, device=self.log_ent_coef.device)
                )
                self.log_ent_coef.clamp_(max=maximum)
        values = self.logger.name_to_value
        self.last_actor_loss = _optional_float(values.get("train/actor_loss"))
        self.last_critic_loss = _optional_float(values.get("train/critic_loss"))
        self.last_entropy_coefficient = (
            float(torch.exp(self.log_ent_coef.detach()).item())
            if self.log_ent_coef is not None
            else _optional_float(values.get("train/ent_coef"))
        )
        self.logger.record("train/ent_coef", self.last_entropy_coefficient)


def _optional_float(value) -> float | None:
    return None if value is None else float(value)
