"""Leakage-safe visual observation adapter with LeRobot-compatible feature keys."""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import numpy as np

VISUAL_FEATURE_KEYS = (
    "observation.images.front",
    "observation.images.wrist",
    "observation.state",
)

PRIVILEGED_KEY_FRAGMENTS = (
    "cube",
    "object",
    "target",
    "tcp",
    "contact",
    "phase",
    "stage",
    "success",
    "reward",
    "distance",
    "grasp",
    "lift",
    "release",
)


def assert_no_privileged_leakage(observation: Mapping[str, object]) -> None:
    """Reject unexpected or diagnostic keys before they can reach a visual policy."""

    keys = tuple(observation.keys())
    if set(keys) != set(VISUAL_FEATURE_KEYS):
        raise AssertionError(f"Visual policy keys must be exactly {VISUAL_FEATURE_KEYS}; received {keys}")
    for key in keys:
        lowered = key.lower()
        leaked = [fragment for fragment in PRIVILEGED_KEY_FRAGMENTS if fragment in lowered]
        if leaked:
            raise AssertionError(f"Privileged feature name leaked through {key!r}: {leaked}")


class VisualObservationAdapter(gym.ObservationWrapper):
    """Expose only front/wrist RGB and robot joint position/velocity state."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base_space = self.unwrapped.observation_space
        if not isinstance(base_space, gym.spaces.Dict) or "pixels" not in base_space.spaces:
            raise ValueError("VisualObservationAdapter requires pixels_state observations")
        pixels = base_space["pixels"]
        state = base_space["state"]
        self.observation_space = gym.spaces.Dict(
            {
                "observation.images.front": pixels["front"],
                "observation.images.wrist": pixels["wrist"],
                "observation.state": state,
            }
        )
        self.feature_schema = {
            "observation.images.front": {
                "type": "VISUAL",
                "shape": list(pixels["front"].shape),
                "dtype": "uint8",
            },
            "observation.images.wrist": {
                "type": "VISUAL",
                "shape": list(pixels["wrist"].shape),
                "dtype": "uint8",
            },
            "observation.state": {
                "type": "STATE",
                "shape": list(state.shape),
                "names": [
                    *(f"joint_position_{index}" for index in range(6)),
                    *(f"joint_velocity_{index}" for index in range(6)),
                ],
            },
        }

    def observation(self, observation: dict) -> dict[str, np.ndarray]:
        adapted = {
            "observation.images.front": np.asarray(observation["pixels"]["front"], dtype=np.uint8),
            "observation.images.wrist": np.asarray(observation["pixels"]["wrist"], dtype=np.uint8),
            "observation.state": np.asarray(observation["state"], dtype=np.float32),
        }
        assert_no_privileged_leakage(adapted)
        if not self.observation_space.contains(adapted):
            raise AssertionError("Adapted visual observation violates its declared space")
        return adapted
