"""Actual-travel recording and sensor-belief return planning."""

from .return_path_planner import (
    ReturnFailureReason, ReturnPathPlan, ReturnPathPlanner,
)
from .travel_history import (
    TravelHistory, TravelHistoryConfig, TravelPoint, TravelRecordReason,
)
from .evacuation_strategy_selector import (
    EvacuationRouteDecision, EvacuationRouteSelectionConfig,
    EvacuationStrategy, EvacuationStrategySelector,
    HazardKnowledgeConfig, HazardKnowledgeDecision, HazardKnowledgeState,
    HazardKnowledgeTracker,
    PathValidationConfig, ReplanningConfig, RouteFailureReason,
)
from .path_simplifier import (
    PathSimplificationConfig, PathSimplificationResult, PathRiskConfig,
    PathValidationSettings, SafePathSimplifier, SegmentRejectionReason,
    SegmentSafetyResult, cells_touched_by_segment,
    extract_direction_change_points,
)
from .exit_switching import (
    CostTrendDecision, ExitSwitchingConfig, RouteCostSample,
    RouteCostTrendMonitor, current_direction_world, evaluate_path_cost,
    is_opposite_direction,
)
from .exploration_manager import (
    ExplorationConfig, ExplorationManager, ExplorationPhase, ExplorationPlan,
)

__all__ = [
    "ReturnFailureReason", "ReturnPathPlan", "ReturnPathPlanner",
    "TravelHistory", "TravelHistoryConfig", "TravelPoint",
    "TravelRecordReason",
    "EvacuationRouteDecision", "EvacuationRouteSelectionConfig",
    "EvacuationStrategy", "EvacuationStrategySelector",
    "HazardKnowledgeConfig", "HazardKnowledgeDecision", "HazardKnowledgeState",
    "HazardKnowledgeTracker", "PathValidationConfig", "ReplanningConfig",
    "RouteFailureReason",
    "PathSimplificationConfig", "PathSimplificationResult", "PathRiskConfig",
    "PathValidationSettings", "SafePathSimplifier",
    "SegmentRejectionReason", "SegmentSafetyResult",
    "cells_touched_by_segment", "extract_direction_change_points",
    "CostTrendDecision", "ExitSwitchingConfig", "RouteCostSample",
    "RouteCostTrendMonitor", "current_direction_world", "evaluate_path_cost",
    "is_opposite_direction",
    "ExplorationConfig", "ExplorationManager", "ExplorationPhase",
    "ExplorationPlan",
]
