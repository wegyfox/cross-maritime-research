"""Explicit coordination of the optional Rule-13 past-progress contract.

Rule 13 responsibility itself remains pairwise and persistent until the lifecycle says
past-and-clear.  The a55 progress contract is an additional engineering liveness check:
it can require an overtaker to have positive longitudinal progress toward becoming
``past``.  In a connected multi-vessel component, requiring that extra condition on
every persisted overtaking edge simultaneously can be physically impossible (for
example A overtakes C while C overtakes B and both A/C are already at the same class
speed cap).

This module keeps those concerns separate.  It never releases or reassigns Rule-13
responsibility.  It only states which persisted Rule-13 edges must satisfy the optional
positive-progress contract *at this planning instant*.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .encounter_registry import PairKey
from .multi_intent import ComponentIntentAggregation


class Rule13ProgressCoordinationError(RuntimeError):
    """Base error for invalid Rule-13 progress-dependency structures."""


class Rule13ProgressDependencyCycle(Rule13ProgressCoordinationError):
    """Raised when persisted Rule-13 progress dependencies contain a directed cycle."""


class Rule13ProgressCoordinationPolicy(str, Enum):
    """How the optional positive-progress contract is applied inside one component."""

    STRICT_ALL_DUTIES = "strict_all_duties"
    FRONTIER_DEPENDENCIES = "frontier_dependencies"


@dataclass(frozen=True, slots=True)
class Rule13ProgressEdge:
    pair_key: PairKey
    overtaker_id: str
    overtaken_id: str
    progress_active: bool


@dataclass(frozen=True, slots=True)
class Rule13ProgressPlan:
    component_id: str
    policy: Rule13ProgressCoordinationPolicy
    edges: tuple[Rule13ProgressEdge, ...]

    @property
    def active_pair_keys(self) -> tuple[PairKey, ...]:
        return tuple(edge.pair_key for edge in self.edges if edge.progress_active)

    @property
    def deferred_pair_keys(self) -> tuple[PairKey, ...]:
        return tuple(edge.pair_key for edge in self.edges if not edge.progress_active)

    def is_progress_active(self, pair_key: PairKey) -> bool:
        return any(edge.pair_key == pair_key and edge.progress_active for edge in self.edges)

    def is_progress_deferred(self, pair_key: PairKey) -> bool:
        return any(edge.pair_key == pair_key and not edge.progress_active for edge in self.edges)

    def active_counterparts_for(self, vessel_id: str) -> tuple[str, ...]:
        return tuple(
            edge.overtaken_id
            for edge in self.edges
            if edge.overtaker_id == vessel_id and edge.progress_active
        )

    def deferred_counterparts_for(self, vessel_id: str) -> tuple[str, ...]:
        return tuple(
            edge.overtaken_id
            for edge in self.edges
            if edge.overtaker_id == vessel_id and not edge.progress_active
        )


def _rule13_progress_edges(
    aggregation: ComponentIntentAggregation,
) -> tuple[tuple[PairKey, str, str], ...]:
    edges: list[tuple[PairKey, str, str]] = []
    for pair_item in aggregation.pair_intents:
        required = tuple(
            intent
            for intent in (pair_item.intents.first, pair_item.intents.second)
            if intent.rule13_past_progress_required
        )
        if not required:
            continue
        if len(required) != 1:
            raise Rule13ProgressCoordinationError(
                f"pair {pair_item.key!r} must have exactly one Rule-13 progress-requiring overtaker"
            )
        intent = required[0]
        edges.append((pair_item.key, intent.vessel_id, intent.counterpart_id))
    return tuple(sorted(edges, key=lambda item: item[0]))


def _assert_acyclic(edges: tuple[tuple[PairKey, str, str], ...]) -> None:
    graph: dict[str, set[str]] = {}
    for _, overtaker, overtaken in edges:
        graph.setdefault(overtaker, set()).add(overtaken)
        graph.setdefault(overtaken, set())

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise Rule13ProgressDependencyCycle(
                "persisted Rule-13 progress dependencies contain a directed cycle"
            )
        visiting.add(node)
        for nxt in sorted(graph.get(node, ())):
            visit(nxt)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(graph):
        visit(node)


def build_rule13_progress_plan(
    aggregation: ComponentIntentAggregation,
    *,
    policy: Rule13ProgressCoordinationPolicy = Rule13ProgressCoordinationPolicy.STRICT_ALL_DUTIES,
) -> Rule13ProgressPlan:
    """Build one auditable progress-contract plan from pairwise Rule-13 intents.

    ``STRICT_ALL_DUTIES`` reproduces a55 behavior: every Rule-13 edge that is not yet
    modeled past must show positive progress when the screen is enabled.

    ``FRONTIER_DEPENDENCIES`` first validates that the directed overtaking graph is
    acyclic.  An edge is progress-active only when its *overtaken* vessel is not itself
    currently an overtaker that still requires past-progress.  Thus in A->C->B, C->B is
    the current frontier and A->C is deferred.  The deferred edge keeps all Rule-13
    responsibility, keep-clear semantics and physical safety screening; only the extra
    positive-progress-now requirement is postponed until the front dependency clears.
    """

    if not isinstance(policy, Rule13ProgressCoordinationPolicy):
        raise TypeError("policy must be Rule13ProgressCoordinationPolicy")

    raw_edges = _rule13_progress_edges(aggregation)
    if policy is Rule13ProgressCoordinationPolicy.STRICT_ALL_DUTIES:
        return Rule13ProgressPlan(
            aggregation.component_id,
            policy,
            tuple(
                Rule13ProgressEdge(pair_key, overtaker, overtaken, True)
                for pair_key, overtaker, overtaken in raw_edges
            ),
        )

    _assert_acyclic(raw_edges)
    active_overtakers = {overtaker for _, overtaker, _ in raw_edges}
    return Rule13ProgressPlan(
        aggregation.component_id,
        policy,
        tuple(
            Rule13ProgressEdge(
                pair_key,
                overtaker,
                overtaken,
                progress_active=(overtaken not in active_overtakers),
            )
            for pair_key, overtaker, overtaken in raw_edges
        ),
    )
