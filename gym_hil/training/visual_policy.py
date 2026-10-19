"""Small dual-camera visual actor and local demonstration dataset utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

PROPRIO_SCALES = np.asarray((*([3.2] * 6), *([10.0] * 6)), dtype=np.float32)


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class VisualActor(nn.Module):
    """Predict the normalized six-dimensional action without privileged inputs."""

    def __init__(self, image_feature_dim: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        self.front_encoder = ImageEncoder(image_feature_dim)
        self.wrist_encoder = ImageEncoder(image_feature_dim)
        self.proprio_encoder = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.action_head = nn.Sequential(
            nn.Linear(2 * image_feature_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Tanh(),
        )
        self.register_buffer("proprio_scales", torch.as_tensor(PROPRIO_SCALES))

    @staticmethod
    def _prepare_images(images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        return images.float() / 255.0

    def encode(self, front: torch.Tensor, wrist: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        front_features = self.front_encoder(self._prepare_images(front))
        wrist_features = self.wrist_encoder(self._prepare_images(wrist))
        proprio_features = self.proprio_encoder(proprio.float() / self.proprio_scales)
        return torch.cat((front_features, wrist_features, proprio_features), dim=1)

    def forward(self, front: torch.Tensor, wrist: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.encode(front, wrist, proprio))

    @torch.no_grad()
    def predict(self, observation: dict[str, np.ndarray], device: str = "cpu") -> np.ndarray:
        self.eval()
        front = torch.as_tensor(observation["observation.images.front"], device=device)
        wrist = torch.as_tensor(observation["observation.images.wrist"], device=device)
        proprio = torch.as_tensor(observation["observation.state"], device=device)
        return self(front, wrist, proprio).squeeze(0).cpu().numpy().astype(np.float32)


class VisualDemonstrationDataset(Dataset):
    """Load episode-level visual demonstrations into compact uint8 arrays."""

    def __init__(self, dataset_dir: Path, episode_indices: list[int]) -> None:
        arrays: dict[str, list[np.ndarray]] = {
            key: [] for key in ("front", "wrist", "proprio", "policy_actions", "diagnostic_stages")
        }
        self.episode_lengths: list[int] = []
        for episode_index in episode_indices:
            path = dataset_dir / "episodes" / f"episode_{episode_index:03d}.npz"
            with np.load(path, allow_pickle=False) as episode:
                for key in arrays:
                    arrays[key].append(episode[key])
                self.episode_lengths.append(len(episode["policy_actions"]))
        self.front = np.concatenate(arrays["front"])
        self.wrist = np.concatenate(arrays["wrist"])
        self.proprio = np.concatenate(arrays["proprio"])
        self.actions = np.concatenate(arrays["policy_actions"])
        self.stages = np.concatenate(arrays["diagnostic_stages"])

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.front[index]),
            torch.from_numpy(self.wrist[index]),
            torch.from_numpy(self.proprio[index]),
            torch.from_numpy(self.actions[index]),
            str(self.stages[index]),
        )


def save_visual_actor(path: Path, model: VisualActor, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_visual_actor(path: Path, device: str = "cpu") -> tuple[VisualActor, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = VisualActor()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint["metadata"]
