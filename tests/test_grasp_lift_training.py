import gymnasium as gym
import numpy as np

from gym_hil.training.grasp_lift_sac import DemonstrationReplayBuffer


def test_demonstration_replay_samples_configured_fraction() -> None:
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    replay = DemonstrationReplayBuffer(
        100,
        observation_space,
        action_space,
        device="cpu",
        n_envs=1,
        demonstration_fraction=0.35,
    )
    for _ in range(20):
        replay.add(
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            [{}],
        )
    replay.mark_demonstrations()
    for _ in range(20):
        replay.add(
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            [{}],
        )
    sample = replay.sample(100)
    demonstration_count = int((sample.observations[:, 0] == 0).sum().item())
    assert demonstration_count == 35
    assert replay.demonstration_size == 20
    assert replay.last_sampled_demonstration_fraction == 0.35
