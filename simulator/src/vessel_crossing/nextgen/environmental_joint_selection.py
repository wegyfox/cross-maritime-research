"""Unified environmental JOINT selection facade with an optional exact backend.

The historical materialized A104/A105/A106 pipeline remains the default.  Callers may
explicitly opt into the exact cached/branch-and-bound selector when a bounded Cartesian
materialization would be wasteful.  The exact backend returns the same selected
``RankedEnvironmentalJointTuple`` contract; its arbitration adapter intentionally contains
only that selected row because no full ranked grid is materialized.
"""
from __future__ import annotations

from dataclasses import dataclass

from .environmental_joint_arbiter import (
    EnvironmentalJointArbitrationResult,
    arbitrate_environmental_joint_tuples,
)
from .environmental_joint_exact_search import (
    EnvironmentalJointExactSearchConfig,
    EnvironmentalJointExactSearchResult,
    select_environmental_joint_exact,
)
from .environmental_joint_semantics import (
    EnvironmentalJointSemanticConfig,
    EnvironmentalJointSemanticGrid,
    evaluate_environmental_joint_semantics,
)
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .joint_arbiter import JointArbitrationConfig
from .multi_intent import ComponentIntentAggregation


@dataclass(frozen=True, slots=True)
class EnvironmentalJointSelectionResult:
    """Selection evidence for either the legacy grid or the exact backend."""

    semantic_grid: EnvironmentalJointSemanticGrid | None
    exact_search: EnvironmentalJointExactSearchResult | None
    arbitration: EnvironmentalJointArbitrationResult

    @property
    def used_exact_search(self) -> bool:
        return self.exact_search is not None


def select_environmental_joint(
    context: EnvironmentalAvoidancePlanningContext,
    aggregation: ComponentIntentAggregation,
    *,
    semantic_config: EnvironmentalJointSemanticConfig,
    arbitration_config: JointArbitrationConfig,
    exact_search_config: EnvironmentalJointExactSearchConfig | None = None,
) -> EnvironmentalJointSelectionResult:
    """Select one JOINT tuple without changing the historical default backend."""

    if exact_search_config is None:
        grid = evaluate_environmental_joint_semantics(
            context,
            aggregation,
            config=semantic_config,
        )
        arbitration = arbitrate_environmental_joint_tuples(
            context,
            grid,
            config=arbitration_config,
        )
        return EnvironmentalJointSelectionResult(grid, None, arbitration)

    if not isinstance(exact_search_config, EnvironmentalJointExactSearchConfig):
        raise TypeError("exact_search_config must be EnvironmentalJointExactSearchConfig or None")
    exact = select_environmental_joint_exact(
        context,
        aggregation,
        semantic_config=semantic_config,
        arbitration_config=arbitration_config,
        search_config=exact_search_config,
    )
    arbitration = EnvironmentalJointArbitrationResult(
        arbitration_config,
        (exact.selected,),
        exact.selected,
    )
    return EnvironmentalJointSelectionResult(None, exact, arbitration)
