import json

from planner.evacuation_planner import (
    EvacuationFailureReason, EvacuationPlanner, ExitSelectionConfig,
)
from planner.exit_evaluator import ExitEvaluation


def evaluation(exit_id, risk, length, unknown, accepted=True):
    return ExitEvaluation(
        exit_id, "unknown", (0, 0), (1, 1), (1, 1), accepted, accepted,
        ((0, 0), (1, 1)) if accepted else tuple(),
        ((0, 0), (1, 1)) if accepted else tuple(),
        length if accepted else None, risk if accepted else None,
        20.0, 0.0, 20.0, 0.0, unknown if accepted else None,
        tuple(), 1.0,
    )


class StubEvaluator:
    def __init__(self, values):
        self.values = values

    def evaluate(self, item, *_args, **_kwargs):
        return self.values[item.exit_id]


class Candidate:
    def __init__(self, exit_id):
        self.exit_id = exit_id


def run(values):
    planner = EvacuationPlanner(StubEvaluator(values))
    return planner.plan(
        [Candidate(item) for item in reversed(tuple(values))], (0, 0),
        cost_map=None, static_obstacle_map=None, dynamic_obstacle_map=None,
        estimated_fire_map=None, created_at=2,
    )


def test_priority_risk_then_length_then_unknown_then_id():
    values = {
        "risk": evaluation("risk", 0.5, 10, 0.5),
        "short": evaluation("short", 1.0, 3, 0.5),
    }
    assert run(values).selected_exit_id == "risk"
    values = {
        "long": evaluation("long", 1, 5, 0.1),
        "short": evaluation("short", 1, 4, 0.9),
    }
    assert run(values).selected_exit_id == "short"
    values = {
        "unknown-high": evaluation("unknown-high", 1, 4, 0.5),
        "unknown-low": evaluation("unknown-low", 1, 4, 0.1),
    }
    assert run(values).selected_exit_id == "unknown-low"
    values = {"B": evaluation("B", 1, 4, 0.1), "A": evaluation("A", 1, 4, 0.1)}
    assert run(values).selected_exit_id == "A"


def test_deterministic_all_results_and_serialization():
    values = {"B": evaluation("B", 1, 4, 0.1), "A": evaluation("A", 1, 4, 0.1)}
    plans = [run(values) for _ in range(5)]
    assert all(item.selected_exit_id == "A" for item in plans)
    assert [item.exit_id for item in plans[0].all_evaluations] == ["A", "B"]
    assert plans[0].path_grid[0] == (0, 0)
    json.dumps(plans[0].to_dict())


def test_all_rejected_and_no_exits_never_select_default():
    failed = run({"A": evaluation("A", 0, 0, 0, accepted=False)})
    assert not failed.success
    assert failed.selected_exit_id is None
    assert failed.failure_reason is EvacuationFailureReason.NO_SAFE_EXIT
    empty = EvacuationPlanner(StubEvaluator({})).plan(
        [], (0, 0), cost_map=None, static_obstacle_map=None,
        dynamic_obstacle_map=None, estimated_fire_map=None, created_at=1,
    )
    assert empty.failure_reason is EvacuationFailureReason.NO_EXITS_REGISTERED


def test_selection_config_rejects_changed_order():
    try:
        ExitSelectionConfig(primary_key="path_length_m")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid sort order was accepted")
    try:
        ExitSelectionConfig(float_tolerance="small")
    except TypeError:
        pass
    else:
        raise AssertionError("invalid tolerance type was accepted")
