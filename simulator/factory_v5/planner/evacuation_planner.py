"""Evaluate all registered exits and choose a deterministic safe candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from planner.exit_evaluator import ExitEvaluation


class EvacuationFailureReason(Enum):
    NO_EXITS_REGISTERED = "no_exits_registered"
    NO_SAFE_EXIT = "no_safe_exit"
    INVALID_START_POSITION = "invalid_start_position"


@dataclass(frozen=True)
class ExitSelectionConfig:
    prefer_confirmed_usable_exit: bool = True
    fallback_to_shortest_reachable_exit: bool = True
    primary_key: str = "path_length_m"
    secondary_key: str = "accumulated_risk_cost"
    final_tie_breaker: str = "exit_id"
    float_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if isinstance(self.float_tolerance, bool) or not isinstance(
            self.float_tolerance, (int, float)
        ):
            raise TypeError("float_tolerance must be numeric")
        for name in (
            "prefer_confirmed_usable_exit",
            "fallback_to_shortest_reachable_exit",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        expected = ("path_length_m", "accumulated_risk_cost", "exit_id")
        actual = (self.primary_key, self.secondary_key, self.final_tie_breaker)
        if actual != expected:
            raise ValueError(f"unsupported exit selection order: {actual}")
        if self.float_tolerance <= 0:
            raise ValueError("float_tolerance must be positive")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "ExitSelectionConfig":
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown exit_selection settings: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class EvacuationPlan:
    success: bool
    start_position_world: tuple[float, float]
    selected_exit_id: str | None
    selected_exit_position_world: tuple[float, float] | None
    selected_approach_position_world: tuple[float, float] | None
    path_world: tuple[tuple[float, float], ...]
    path_grid: tuple[tuple[int, int], ...]
    selected_evaluation: ExitEvaluation | None
    all_evaluations: tuple[ExitEvaluation, ...]
    failure_reason: EvacuationFailureReason | None
    selection_reason: str | None
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["start_position_world"] = list(self.start_position_world)
        for key in ("selected_exit_position_world", "selected_approach_position_world"):
            result[key] = None if result[key] is None else list(result[key])
        result["path_world"] = [list(item) for item in self.path_world]
        result["path_grid"] = [list(item) for item in self.path_grid]
        result["selected_evaluation"] = (
            None if self.selected_evaluation is None
            else self.selected_evaluation.to_dict()
        )
        result["all_evaluations"] = [item.to_dict() for item in self.all_evaluations]
        result["failure_reason"] = (
            None if self.failure_reason is None else self.failure_reason.value
        )
        return result


class EvacuationPlanner:
    def __init__(self, evaluator, config: ExitSelectionConfig | None = None) -> None:
        self.evaluator = evaluator
        self.config = config or ExitSelectionConfig()

    def plan(
        self, exits, start_position_world, *, cost_map,
        static_obstacle_map, dynamic_obstacle_map, estimated_fire_map,
        created_at: float, risk_first: bool = False,
    ) -> EvacuationPlan:
        exits = tuple(exits)
        start = (float(start_position_world[0]), float(start_position_world[1]))
        if not exits:
            return self._failure(
                start, tuple(), EvacuationFailureReason.NO_EXITS_REGISTERED,
                created_at,
            )
        evaluations = tuple(self.evaluator.evaluate(
            item, start,
            cost_map=cost_map,
            static_obstacle_map=static_obstacle_map,
            dynamic_obstacle_map=dynamic_obstacle_map,
            estimated_fire_map=estimated_fire_map,
            evaluated_at=created_at,
        ) for item in sorted(exits, key=lambda item: item.exit_id))
        accepted = [item for item in evaluations if item.accepted]
        if not accepted:
            return self._failure(
                start, evaluations, EvacuationFailureReason.NO_SAFE_EXIT,
                created_at,
            )
        tolerance = self.config.float_tolerance

        def bucket(value):
            return int(round(float(value) / tolerance))

        confirmed = [item for item in accepted if item.exit_status == "usable"]
        if self.config.prefer_confirmed_usable_exit and confirmed:
            candidates = confirmed
        elif (
            self.config.prefer_confirmed_usable_exit
            and not self.config.fallback_to_shortest_reachable_exit
        ):
            return self._failure(
                start, evaluations, EvacuationFailureReason.NO_SAFE_EXIT,
                created_at,
            )
        else:
            candidates = accepted
        primary = "accumulated_risk_cost" if risk_first else self.config.primary_key
        secondary = "path_length_m" if risk_first else self.config.secondary_key
        candidates.sort(key=lambda item: (
            bucket(getattr(item, primary)),
            bucket(getattr(item, secondary)),
            item.exit_id,
        ))
        selected = candidates[0]
        reason = (
            (
                "confirmed usable exits preferred; lowest accumulated path risk; "
                "ties resolved by path length then exit_id"
                if risk_first else
                "confirmed usable exits preferred; shortest cost-aware A* path; "
                "ties resolved by accumulated risk cost then exit_id"
            )
        )
        return EvacuationPlan(
            True, start, selected.exit_id, selected.exit_position_world,
            selected.approach_position_world, selected.path_world,
            selected.path_grid, selected, evaluations, None, reason,
            float(created_at),
        )

    @staticmethod
    def _failure(start, evaluations, reason, created_at):
        return EvacuationPlan(
            False, start, None, None, None, tuple(), tuple(), None,
            tuple(evaluations), reason, None, float(created_at),
        )
