import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401
from gym_hil.wrappers.so101_wrappers import SO101EEActionWrapper


def test_teleop_adapter_maps_to_six_normalized_joint_deltas() -> None:
    base = gym.make("gym_hil/SO101PickPlaceBase-v0", observation_mode="state")
    wrapper = SO101EEActionWrapper(base)
    try:
        wrapper.reset(seed=0)
        converted = wrapper.action(np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32))
        assert converted.shape == (6,)
        assert np.all(np.abs(converted) <= 1.0)
        np.testing.assert_allclose(converted[-1], 0.0)
    finally:
        wrapper.close()


def test_hil_registrations_use_so101_factory() -> None:
    for suffix in ("Keyboard", "Gamepad"):
        spec = gym.spec(f"gym_hil/SO101PickPlace{suffix}-v0")
        assert spec.entry_point == "gym_hil.wrappers.so101_wrappers:make_so101_hil_env"
