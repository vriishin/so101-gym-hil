"""Sparse, dense, staged, and potential-based pick-and-place rewards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gym_hil.rewards.config import RewardConfig
from gym_hil.rewards.task_state import TaskState


@dataclass(frozen=True)
class RewardResult:
    """Scalar reward and term-level diagnostics."""

    total: float
    terms: dict[str, float]
    potential_previous: float | None = None
    potential_current: float | None = None


class PickPlaceReward:
    """Stateful episode reward calculator with monotonic stage hysteresis."""

    STAGES = ("reach", "grasp", "lift", "transport", "release", "stability", "success")

    def __init__(self, config: RewardConfig, stable_steps_required: int) -> None:
        self.config = config
        self.stable_steps_required = stable_steps_required
        self._stage_index = 0

    @property
    def task_stage(self) -> str:
        return self.STAGES[self._stage_index]

    def reset(self, initial_state: TaskState) -> None:
        """Reset episode-local stage memory."""

        self._stage_index = 0
        self._advance_stage(initial_state)

    def _scores(self, state: TaskState) -> dict[str, float]:
        cfg = self.config
        return {
            "reach": float(np.exp(-state.reach_distance / cfg.reach_sigma_m)),
            "grasp": float(state.is_grasped),
            "lift": float(np.clip(state.object_lift_height / cfg.desired_lift_height_m, 0, 1)),
            "transport": float(np.exp(-state.target_distance / cfg.target_sigma_m)),
            "place": float(state.object_in_target),
            "release": float(state.is_released),
            "stability": float(state.object_in_target)
            * float(np.clip(state.stable_steps / self.stable_steps_required, 0, 1)),
        }

    def _advance_stage(self, state: TaskState) -> None:
        achieved = 0
        if state.is_grasped:
            achieved = 1
        if state.is_lifted:
            achieved = 2
        if state.is_lifted and state.target_distance <= self.config.target_sigma_m:
            achieved = 3
        if state.object_in_target:
            achieved = 4
        if state.is_released:
            achieved = 5
        if state.success:
            achieved = 6
        self._stage_index = max(self._stage_index, achieved)

    def potential(self, state: TaskState) -> float:
        """Compute the bounded shaping potential for a non-absorbing state."""

        s = self._scores(state)
        cfg = self.config
        lift_gate = float(state.is_lifted)
        grasp_gate = float(state.is_grasped or state.is_lifted)
        return float(
            cfg.reach_weight * s["reach"]
            + cfg.grasp_weight * s["grasp"]
            + cfg.lift_weight * grasp_gate * s["lift"]
            + cfg.transport_weight * lift_gate * s["transport"]
            + cfg.place_weight * s["place"]
            + cfg.release_weight * s["release"]
            + cfg.stability_weight * s["stability"]
        )

    def compute(
        self,
        previous_state: TaskState,
        state: TaskState,
        action: np.ndarray,
        previous_action: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> RewardResult:
        """Compute one transition reward and expose every component."""

        cfg = self.config
        stage_before = self._stage_index
        self._advance_stage(state)
        if cfg.mode == "sparse":
            success = cfg.success_bonus if state.success else 0.0
            return RewardResult(total=float(success), terms={"success": float(success)})

        previous_scores = self._scores(previous_state)
        scores = self._scores(state)
        action_penalty = -cfg.action_penalty_weight * float(np.dot(action, action))
        delta_action = action - previous_action
        jerk_penalty = -cfg.jerk_penalty_weight * float(np.dot(delta_action, delta_action))
        drop_penalty = (
            -cfg.drop_penalty
            if previous_state.is_grasped and not state.is_grasped and not state.object_in_target
            else 0.0
        )
        bounds_penalty = -cfg.out_of_bounds_penalty if state.out_of_bounds else 0.0
        success = cfg.success_bonus if state.success else 0.0

        if cfg.mode == "naive_dense":
            terms = {
                "reach": cfg.reach_weight * scores["reach"],
                "grasp": cfg.grasp_weight * scores["grasp"],
                "lift": cfg.lift_weight * scores["lift"],
                "transport": cfg.transport_weight * scores["transport"],
                "place": cfg.place_weight * scores["place"],
                "success": success,
                "action_penalty": action_penalty,
                "jerk_penalty": jerk_penalty,
                "drop_penalty": drop_penalty,
                "out_of_bounds_penalty": bounds_penalty,
            }
            return RewardResult(float(sum(terms.values())), terms)

        if cfg.mode == "staged_dense":
            progress_terms = {
                name: 0.0 for name in ("reach", "grasp", "lift", "transport", "place", "release", "stability")
            }
            if stage_before == 0:
                progress_terms["reach"] = cfg.reach_weight * (scores["reach"] - previous_scores["reach"])
            elif stage_before == 1:
                progress_terms["grasp"] = cfg.grasp_weight * (scores["grasp"] - previous_scores["grasp"])
                progress_terms["lift"] = cfg.lift_weight * (scores["lift"] - previous_scores["lift"])
            elif stage_before in {2, 3}:
                progress_terms["transport"] = cfg.transport_weight * (
                    scores["transport"] - previous_scores["transport"]
                )
            elif stage_before == 4:
                progress_terms["place"] = cfg.place_weight * scores["place"]
                progress_terms["release"] = cfg.release_weight * (
                    scores["release"] - previous_scores["release"]
                )
            elif stage_before >= 5:
                progress_terms["stability"] = cfg.stability_weight * (
                    scores["stability"] - previous_scores["stability"]
                )
            terms = {
                **{name: cfg.shaping_scale * value for name, value in progress_terms.items()},
                "success": success,
                "action_penalty": action_penalty,
                "jerk_penalty": jerk_penalty,
                "drop_penalty": drop_penalty,
                "out_of_bounds_penalty": bounds_penalty,
            }
            return RewardResult(float(sum(terms.values())), terms)

        potential_previous = self.potential(previous_state)
        potential_current = 0.0 if terminated else self.potential(state)
        shaping = cfg.shaping_scale * (cfg.gamma * potential_current - potential_previous)
        terms = {
            "potential_shaping": shaping,
            "success": success,
            "action_penalty": action_penalty,
            "jerk_penalty": jerk_penalty,
            "drop_penalty": drop_penalty,
            "out_of_bounds_penalty": bounds_penalty,
        }
        return RewardResult(
            float(sum(terms.values())),
            terms,
            potential_previous=potential_previous,
            potential_current=potential_current,
        )
