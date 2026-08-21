"""Explicit mission state machine for search and evacuation orchestration.

``SEARCH_EXITS`` retains the design name even though the robot observes both
exits and victims while searching.  This module owns transition policy only;
sensor interpretation, planning, motion, speech, and ROS reporting remain with
their respective adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any


class MissionState(Enum):
    """Internal states; enum values are stable log/display labels."""

    SEARCH_EXITS = "Searching exits and victims"
    EXPLORATION_STALLED = "Exit exploration stalled"
    EXPLORATION_COMPLETE = "All exits explored"
    APPROACH_VICTIM = "Approaching victim"
    ANNOUNCE_EVACUATION = "Announcing evacuation guidance"
    PLAN_EVACUATION = "Planning evacuation"
    EVALUATING_HAZARD_INFORMATION = "Evaluating sensor hazard knowledge"
    EVALUATING_EXITS = "Evaluating registered exits"
    ESCORT_VICTIM = "Escorting victim"
    FOLLOW_WAIT = "Waiting for escorted victim to catch up"
    FOLLOW_FAILED = "Victim following failed"
    EVACUATION_COMPLETE = "Evacuation complete"
    ROBOT_EVACUATED = "Robot evacuated through exit"
    REPORT_IMMOBILE_VICTIM = "Reporting immobile victim"
    REPLAN = "Replanning evacuation route"
    NO_SAFE_EXIT = "No safe exit"
    WAITING_FOR_VICTIM = "Waiting for victim"
    PLAN_RETURN_BY_HISTORY = "Planning return from travel history"
    RETURNING_BY_HISTORY = "Returning by recorded travel history"
    RETURN_PATH_BLOCKED = "Recorded return path blocked"
    RETURN_FAILED = "Recorded return failed"
    REPLANNING_TO_ENTRANCE = "Replanning to original entrance"
    RETURNING_TO_ENTRANCE = "Returning to original entrance by A*"
    NO_SAFE_ROUTE = "No safe evacuation route"
    MISSION_ABORTED = "Mission aborted"
    ERROR = "Mission error"


class MissionEvent(Enum):
    VICTIM_DETECTED = "victim_detected"
    VICTIM_REACHED = "victim_reached"
    ANNOUNCEMENT_FINISHED = "announcement_finished"
    VICTIM_STARTED_MOVING = "victim_started_moving"
    VICTIM_NOT_MOVING = "victim_not_moving"
    PATH_PLANNED = "path_planned"
    PATH_BLOCKED = "path_blocked"
    PATH_PLANNING_FAILED = "path_planning_failed"
    EXIT_UNSAFE = "exit_unsafe"
    EXIT_REACHED = "exit_reached"
    EVACUATION_CONFIRMED = "evacuation_confirmed"
    RETRY_REQUESTED = "retry_requested"
    SEARCH_RESUMED = "search_resumed"
    RETURN_REQUESTED = "return_requested"
    RETURN_PATH_CREATED = "return_path_created"
    RETURN_PATH_INVALIDATED = "return_path_invalidated"
    RETURN_COMPLETED = "return_completed"
    RETURN_FAILED = "return_failed"
    EXIT_EVALUATION_REQUESTED = "exit_evaluation_requested"
    SAFE_EXIT_SELECTED = "safe_exit_selected"
    NO_SAFE_EXIT_FOUND = "no_safe_exit_found"
    EVACUATION_PLAN_CREATED = "evacuation_plan_created"
    EVACUATION_PLAN_FAILED = "evacuation_plan_failed"
    VICTIM_READY_FOR_EVACUATION = "victim_ready_for_evacuation"
    HAZARD_INFORMATION_AVAILABLE = "hazard_information_available"
    NO_HAZARD_INFORMATION = "no_hazard_information"
    ACTIVE_PATH_INVALIDATED = "active_path_invalidated"
    REPLAN_REQUESTED = "replan_requested"
    ENTRANCE_ROUTE_CREATED = "entrance_route_created"
    NO_SAFE_ROUTE_FOUND = "no_safe_route_found"
    MISSION_ABORTED = "mission_aborted"
    ERROR_OCCURRED = "error_occurred"
    EXPLORATION_TARGET_SELECTED = "exploration_target_selected"
    EXIT_CHECK_COMPLETED = "exit_check_completed"
    EXPLORATION_STALLED = "exploration_stalled"
    EXPLORATION_COMPLETED = "exploration_completed"
    VICTIM_LAGGING = "victim_lagging"
    VICTIM_CAUGHT_UP = "victim_caught_up"
    VICTIM_FOLLOW_FAILED = "victim_follow_failed"
    ROBOT_EVACUATED = "robot_evacuated"


class InvalidTransitionError(RuntimeError):
    """Raised when an event is not legal in the current mission state."""


@dataclass(frozen=True)
class StateTransition:
    previous_state: MissionState
    next_state: MissionState
    event: MissionEvent
    reason: str
    sim_time: float | None
    victim_id: str | None
    exit_id: str | None


# Static rules live in one table. Context-dependent destinations are resolved
# in _resolve_next_state without distributing policy among callers.
_TRANSITIONS: dict[MissionState, dict[MissionEvent, MissionState]] = {
    MissionState.SEARCH_EXITS: {
        MissionEvent.VICTIM_DETECTED: MissionState.APPROACH_VICTIM,
        MissionEvent.EXPLORATION_TARGET_SELECTED: MissionState.SEARCH_EXITS,
        MissionEvent.EXIT_CHECK_COMPLETED: MissionState.SEARCH_EXITS,
        MissionEvent.EXPLORATION_STALLED: MissionState.EXPLORATION_STALLED,
        MissionEvent.EXPLORATION_COMPLETED: MissionState.EXPLORATION_COMPLETE,
        MissionEvent.ROBOT_EVACUATED: MissionState.ROBOT_EVACUATED,
    },
    MissionState.EXPLORATION_STALLED: {
        MissionEvent.RETRY_REQUESTED: MissionState.SEARCH_EXITS,
        MissionEvent.VICTIM_DETECTED: MissionState.APPROACH_VICTIM,
        MissionEvent.ROBOT_EVACUATED: MissionState.ROBOT_EVACUATED,
    },
    MissionState.EXPLORATION_COMPLETE: {
        MissionEvent.SEARCH_RESUMED: MissionState.SEARCH_EXITS,
        MissionEvent.ROBOT_EVACUATED: MissionState.ROBOT_EVACUATED,
    },
    MissionState.APPROACH_VICTIM: {
        MissionEvent.VICTIM_REACHED: MissionState.ANNOUNCE_EVACUATION,
        MissionEvent.VICTIM_NOT_MOVING: MissionState.REPORT_IMMOBILE_VICTIM,
    },
    MissionState.ANNOUNCE_EVACUATION: {
        MissionEvent.ANNOUNCEMENT_FINISHED: MissionState.WAITING_FOR_VICTIM,
        MissionEvent.VICTIM_NOT_MOVING: MissionState.REPORT_IMMOBILE_VICTIM,
    },
    MissionState.WAITING_FOR_VICTIM: {
        MissionEvent.VICTIM_STARTED_MOVING: MissionState.PLAN_EVACUATION,
        MissionEvent.VICTIM_NOT_MOVING: MissionState.REPORT_IMMOBILE_VICTIM,
    },
    MissionState.REPORT_IMMOBILE_VICTIM: {
        MissionEvent.SEARCH_RESUMED: MissionState.SEARCH_EXITS,
    },
    MissionState.PLAN_EVACUATION: {
        MissionEvent.VICTIM_READY_FOR_EVACUATION: MissionState.EVALUATING_HAZARD_INFORMATION,
        MissionEvent.PATH_PLANNED: MissionState.ESCORT_VICTIM,
        MissionEvent.PATH_PLANNING_FAILED: MissionState.REPLAN,
        MissionEvent.EXIT_UNSAFE: MissionState.REPLAN,
        MissionEvent.RETURN_REQUESTED: MissionState.PLAN_RETURN_BY_HISTORY,
        MissionEvent.EXIT_EVALUATION_REQUESTED: MissionState.EVALUATING_EXITS,
        MissionEvent.EVACUATION_PLAN_CREATED: MissionState.ESCORT_VICTIM,
    },
    MissionState.ESCORT_VICTIM: {
        MissionEvent.ACTIVE_PATH_INVALIDATED: MissionState.REPLAN,
        MissionEvent.PATH_BLOCKED: MissionState.REPLAN,
        MissionEvent.EXIT_UNSAFE: MissionState.REPLAN,
        MissionEvent.EXIT_REACHED: MissionState.EVACUATION_COMPLETE,
        MissionEvent.EVACUATION_CONFIRMED: MissionState.EVACUATION_COMPLETE,
        MissionEvent.VICTIM_LAGGING: MissionState.FOLLOW_WAIT,
        MissionEvent.VICTIM_FOLLOW_FAILED: MissionState.FOLLOW_FAILED,
    },
    MissionState.FOLLOW_WAIT: {
        MissionEvent.VICTIM_CAUGHT_UP: MissionState.ESCORT_VICTIM,
        MissionEvent.VICTIM_FOLLOW_FAILED: MissionState.FOLLOW_FAILED,
        MissionEvent.ACTIVE_PATH_INVALIDATED: MissionState.REPLAN,
    },
    MissionState.FOLLOW_FAILED: {
        MissionEvent.RETRY_REQUESTED: MissionState.FOLLOW_WAIT,
    },
    MissionState.REPLAN: {
        MissionEvent.RETRY_REQUESTED: MissionState.PLAN_EVACUATION,
        MissionEvent.PATH_PLANNING_FAILED: MissionState.NO_SAFE_EXIT,
        MissionEvent.RETURN_REQUESTED: MissionState.PLAN_RETURN_BY_HISTORY,
        MissionEvent.REPLAN_REQUESTED: MissionState.REPLANNING_TO_ENTRANCE,
        MissionEvent.EXIT_EVALUATION_REQUESTED: MissionState.EVALUATING_EXITS,
        MissionEvent.NO_SAFE_ROUTE_FOUND: MissionState.NO_SAFE_ROUTE,
    },
    MissionState.NO_SAFE_EXIT: {
        MissionEvent.RETRY_REQUESTED: MissionState.REPLAN,
        MissionEvent.RETURN_REQUESTED: MissionState.PLAN_RETURN_BY_HISTORY,
    },
    MissionState.EVACUATION_COMPLETE: {
        MissionEvent.SEARCH_RESUMED: MissionState.SEARCH_EXITS,
    },
    MissionState.EVALUATING_EXITS: {
        MissionEvent.SAFE_EXIT_SELECTED: MissionState.PLAN_EVACUATION,
        MissionEvent.NO_SAFE_EXIT_FOUND: MissionState.NO_SAFE_EXIT,
        MissionEvent.EVACUATION_PLAN_FAILED: MissionState.NO_SAFE_EXIT,
        MissionEvent.NO_SAFE_ROUTE_FOUND: MissionState.NO_SAFE_ROUTE,
    },
    MissionState.PLAN_RETURN_BY_HISTORY: {
        MissionEvent.RETURN_PATH_CREATED: MissionState.RETURNING_BY_HISTORY,
        MissionEvent.RETURN_FAILED: MissionState.RETURN_FAILED,
        MissionEvent.NO_SAFE_ROUTE_FOUND: MissionState.NO_SAFE_ROUTE,
    },
    MissionState.RETURNING_BY_HISTORY: {
        MissionEvent.RETURN_PATH_INVALIDATED: MissionState.RETURN_PATH_BLOCKED,
        MissionEvent.RETURN_COMPLETED: MissionState.EVACUATION_COMPLETE,
    },
    MissionState.RETURN_PATH_BLOCKED: {
        MissionEvent.RETURN_REQUESTED: MissionState.PLAN_RETURN_BY_HISTORY,
        MissionEvent.RETURN_FAILED: MissionState.RETURN_FAILED,
        MissionEvent.REPLAN_REQUESTED: MissionState.REPLANNING_TO_ENTRANCE,
        MissionEvent.NO_SAFE_ROUTE_FOUND: MissionState.NO_SAFE_ROUTE,
    },
    MissionState.RETURN_FAILED: {
        MissionEvent.RETRY_REQUESTED: MissionState.PLAN_RETURN_BY_HISTORY,
    },
    MissionState.EVALUATING_HAZARD_INFORMATION: {
        MissionEvent.HAZARD_INFORMATION_AVAILABLE: MissionState.EVALUATING_EXITS,
        MissionEvent.NO_HAZARD_INFORMATION: MissionState.EVALUATING_EXITS,
        MissionEvent.NO_SAFE_ROUTE_FOUND: MissionState.NO_SAFE_ROUTE,
    },
    MissionState.REPLANNING_TO_ENTRANCE: {
        MissionEvent.ENTRANCE_ROUTE_CREATED: MissionState.RETURNING_TO_ENTRANCE,
        MissionEvent.EXIT_EVALUATION_REQUESTED: MissionState.EVALUATING_EXITS,
        MissionEvent.NO_SAFE_ROUTE_FOUND: MissionState.NO_SAFE_ROUTE,
    },
    MissionState.RETURNING_TO_ENTRANCE: {
        MissionEvent.ACTIVE_PATH_INVALIDATED: MissionState.REPLAN,
        MissionEvent.RETURN_COMPLETED: MissionState.EVACUATION_COMPLETE,
    },
    MissionState.NO_SAFE_ROUTE: {
        MissionEvent.RETRY_REQUESTED: MissionState.REPLAN,
    },
}

_DEFAULT_REASONS = {
    MissionEvent.VICTIM_DETECTED: "victim detected",
    MissionEvent.VICTIM_REACHED: "robot reached the victim approach distance",
    MissionEvent.ANNOUNCEMENT_FINISHED: "evacuation announcement completed",
    MissionEvent.VICTIM_STARTED_MOVING: "victim started following the robot",
    MissionEvent.VICTIM_NOT_MOVING: "victim did not start or cannot continue moving",
    MissionEvent.PATH_PLANNED: "safe exit and evacuation path selected",
    MissionEvent.PATH_BLOCKED: "current path became blocked",
    MissionEvent.PATH_PLANNING_FAILED: "no evacuation path was generated",
    MissionEvent.EXIT_UNSAFE: "selected exit is no longer safe",
    MissionEvent.EXIT_REACHED: "robot and victim reached the exit",
    MissionEvent.EVACUATION_CONFIRMED: "victim evacuation confirmed",
    MissionEvent.RETRY_REQUESTED: "new information requested another planning attempt",
    MissionEvent.SEARCH_RESUMED: "search resumed for remaining victims",
    MissionEvent.RETURN_REQUESTED: "travel-history return requested",
    MissionEvent.RETURN_PATH_CREATED: "validated travel-history return path created",
    MissionEvent.RETURN_PATH_INVALIDATED: "travel-history return path became unsafe",
    MissionEvent.RETURN_COMPLETED: "robot returned to the recorded entry position",
    MissionEvent.RETURN_FAILED: "no safe travel-history return path is available",
    MissionEvent.EXIT_EVALUATION_REQUESTED: "victim ready; evaluating all exits",
    MissionEvent.SAFE_EXIT_SELECTED: "safe exit selected from complete evaluation",
    MissionEvent.NO_SAFE_EXIT_FOUND: "all registered exits were rejected",
    MissionEvent.EVACUATION_PLAN_CREATED: "selected evacuation path activated",
    MissionEvent.EVACUATION_PLAN_FAILED: "evacuation planning failed",
    MissionEvent.VICTIM_READY_FOR_EVACUATION: "victim is ready for final route selection",
    MissionEvent.HAZARD_INFORMATION_AVAILABLE: "sensor-derived fire information is available",
    MissionEvent.NO_HAZARD_INFORMATION: "no sensor-derived fire information exists",
    MissionEvent.ACTIVE_PATH_INVALIDATED: "active route became unsafe",
    MissionEvent.REPLAN_REQUESTED: "safe replanning requested after stopping movement",
    MissionEvent.ENTRANCE_ROUTE_CREATED: "safe A* route to original entrance created",
    MissionEvent.NO_SAFE_ROUTE_FOUND: "entry and registered exits have no safe route",
    MissionEvent.MISSION_ABORTED: "mission aborted by caller",
    MissionEvent.ERROR_OCCURRED: "mission error reported by caller",
    MissionEvent.EXPLORATION_TARGET_SELECTED: "next unchecked exit selected",
    MissionEvent.EXIT_CHECK_COMPLETED: "exit visit and safety check completed",
    MissionEvent.EXPLORATION_STALLED: "unchecked exits currently have no safe path",
    MissionEvent.VICTIM_LAGGING: "victim exceeded the maximum following distance",
    MissionEvent.VICTIM_CAUGHT_UP: "victim returned within the following resume distance",
    MissionEvent.VICTIM_FOLLOW_FAILED: "victim failed to catch up after repeated guidance",
    MissionEvent.ROBOT_EVACUATED: "robot exited after post-evacuation exploration became unavailable",
    MissionEvent.EXPLORATION_COMPLETED: "all exits have been directly checked",
}


def _serializable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    if hasattr(value, "tolist"):
        return _serializable(value.tolist())
    return value


class MissionManager:
    """Validate transitions and retain mission context and audit history."""

    def __init__(
        self,
        *,
        victim_reached_distance_m: float = 1.0,
        exit_reached_distance_m: float = 1.0,
        victim_wait_timeout_s: float = 10.0,
        max_replan_count: int = 5,
    ) -> None:
        if victim_reached_distance_m <= 0 or exit_reached_distance_m <= 0:
            raise ValueError("mission reach distances must be positive")
        if victim_wait_timeout_s <= 0:
            raise ValueError("victim_wait_timeout_s must be positive")
        if max_replan_count < 0:
            raise ValueError("max_replan_count must be non-negative")
        self.victim_reached_distance_m = float(victim_reached_distance_m)
        self.exit_reached_distance_m = float(exit_reached_distance_m)
        self.victim_wait_timeout_s = float(victim_wait_timeout_s)
        self.max_replan_count = int(max_replan_count)
        self.reset()

    def reset(self) -> None:
        self.current_state = MissionState.SEARCH_EXITS
        self.previous_state: MissionState | None = None
        self.selected_victim_id: str | None = None
        self.selected_victim_position: tuple[float, float] | None = None
        self.selected_exit_id: str | None = None
        self.selected_exit_position: tuple[float, float] | None = None
        self.selected_exit_reason: str | None = None
        self.current_path: Any = None
        self.replan_count = 0
        self.waiting_started_at: float | None = None
        self.transition_history: list[StateTransition] = []
        self.last_error: str | None = None
        self.failed_exits: dict[str, str] = {}
        self.immobile_victims: list[dict[str, Any]] = []

    def get_state(self) -> MissionState:
        return self.current_state

    def can_transition(self, event: MissionEvent) -> bool:
        if event in (MissionEvent.MISSION_ABORTED, MissionEvent.ERROR_OCCURRED):
            return self.current_state not in (
                MissionState.MISSION_ABORTED, MissionState.ERROR
            )
        return event in _TRANSITIONS.get(self.current_state, {})

    def _resolve_next_state(
        self, event: MissionEvent, context: dict[str, Any]
    ) -> MissionState:
        if event is MissionEvent.MISSION_ABORTED:
            return MissionState.MISSION_ABORTED
        if event is MissionEvent.ERROR_OCCURRED:
            return MissionState.ERROR
        proposed = _TRANSITIONS[self.current_state][event]
        if proposed is MissionState.REPLAN:
            if self.replan_count >= self.max_replan_count:
                return (
                    MissionState.NO_SAFE_ROUTE
                    if event is MissionEvent.ACTIVE_PATH_INVALIDATED
                    else MissionState.NO_SAFE_EXIT
                )
            self.replan_count += 1
        if event in (
            MissionEvent.EXIT_REACHED,
            MissionEvent.EVACUATION_CONFIRMED,
            MissionEvent.RETURN_COMPLETED,
        ):
            if bool(context.get("remaining_victims", False)):
                return MissionState.SEARCH_EXITS
        return proposed

    def handle_event(self, event: MissionEvent, **context: Any) -> StateTransition:
        if not isinstance(event, MissionEvent):
            raise TypeError("event must be a MissionEvent")
        if not self.can_transition(event):
            message = f"event {event.name} is not allowed in {self.current_state.name}"
            self.last_error = message
            raise InvalidTransitionError(message)

        previous = self.current_state
        victim_id = context.get("victim_id", self.selected_victim_id)
        exit_id = context.get("exit_id", self.selected_exit_id)
        if event is MissionEvent.VICTIM_DETECTED:
            if not victim_id or context.get("victim_position") is None:
                raise ValueError("VICTIM_DETECTED requires victim_id and victim_position")
            self.selected_victim_id = str(victim_id)
            self.selected_victim_position = tuple(context["victim_position"])
        if context.get("exit_id") is not None:
            self.selected_exit_id = str(context["exit_id"])
        if context.get("exit_position") is not None:
            self.selected_exit_position = tuple(context["exit_position"])
        if context.get("selection_reason") is not None:
            self.selected_exit_reason = str(context["selection_reason"])
        if context.get("path") is not None:
            self.current_path = context["path"]
        if context.get("failed_exits"):
            self.failed_exits.update({str(k): str(v) for k, v in context["failed_exits"].items()})
        if event is MissionEvent.VICTIM_NOT_MOVING:
            self.immobile_victims.append({
                "id": self.selected_victim_id,
                "position": self.selected_victim_position,
                "reason": context.get("reason", _DEFAULT_REASONS[event]),
            })
        if event is MissionEvent.ERROR_OCCURRED:
            self.last_error = str(context.get("reason", _DEFAULT_REASONS[event]))

        next_state = self._resolve_next_state(event, context)
        self.previous_state = previous
        self.current_state = next_state
        sim_time = context.get("sim_time")
        if sim_time is not None and not math.isfinite(float(sim_time)):
            raise ValueError("sim_time must be finite")
        if next_state is MissionState.WAITING_FOR_VICTIM:
            self.waiting_started_at = None if sim_time is None else float(sim_time)
        elif previous is MissionState.WAITING_FOR_VICTIM:
            self.waiting_started_at = None
        if next_state is MissionState.SEARCH_EXITS:
            self.selected_victim_id = None
            self.selected_victim_position = None
            self.selected_exit_id = None
            self.selected_exit_position = None
            self.selected_exit_reason = None
            self.current_path = None

        transition = StateTransition(
            previous_state=previous,
            next_state=next_state,
            event=event,
            reason=str(context.get("reason", _DEFAULT_REASONS[event])),
            sim_time=None if sim_time is None else float(sim_time),
            victim_id=None if victim_id is None else str(victim_id),
            exit_id=None if exit_id is None else str(exit_id),
        )
        self.transition_history.append(transition)
        return transition

    def check_wait_timeout(self, sim_time: float) -> StateTransition | None:
        if self.current_state is not MissionState.WAITING_FOR_VICTIM:
            return None
        if self.waiting_started_at is None:
            self.waiting_started_at = float(sim_time)
            return None
        if float(sim_time) - self.waiting_started_at < self.victim_wait_timeout_s:
            return None
        return self.handle_event(
            MissionEvent.VICTIM_NOT_MOVING,
            sim_time=sim_time,
            reason=f"victim wait timeout exceeded {self.victim_wait_timeout_s:.1f} s",
        )

    def get_transition_history(self) -> tuple[StateTransition, ...]:
        return tuple(self.transition_history)

    def to_dict(self) -> dict[str, Any]:
        return _serializable({
            "current_state": self.current_state,
            "previous_state": self.previous_state,
            "selected_victim_id": self.selected_victim_id,
            "selected_victim_position": self.selected_victim_position,
            "selected_exit_id": self.selected_exit_id,
            "selected_exit_position": self.selected_exit_position,
            "selected_exit_reason": self.selected_exit_reason,
            "current_path": self.current_path,
            "replan_count": self.replan_count,
            "max_replan_count": self.max_replan_count,
            "waiting_started_at": self.waiting_started_at,
            "transition_history": [asdict(item) for item in self.transition_history],
            "last_error": self.last_error,
            "failed_exits": self.failed_exits,
            "immobile_victims": self.immobile_victims,
        })
