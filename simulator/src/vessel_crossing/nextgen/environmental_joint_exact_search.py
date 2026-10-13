"""Exact cached/branch-and-bound selection for environmental JOINT planning.

This is an alternative pure selection backend for cases where the explicit A104
Cartesian grid is unnecessarily expensive to materialize.  It reuses the same A104
per-vessel transitions / pair clearance certificates, the same A105 pair semantics,
and the same A106 criterion evaluator.  Only the search strategy differs:

- unary and pair-local evidence is computed once;
- incompatible partial assignments are rejected immediately;
- optimistic lower bounds on locally decomposable A106 criteria may prune branches;
- non-decomposable criteria are evaluated only at complete safe assignments;
- the selected tuple retains the canonical Cartesian ``command_index``.

The legacy materialized evaluator remains unchanged and is still the default runtime
backend.  This module has its own explicit resource contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import inf
from typing import Mapping

from .dynamics import signed_heading_delta
from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_joint_arbiter import (
    EnvironmentalJointArbitrationScore,
    NoEnvironmentalJointTuple,
    RankedEnvironmentalJointTuple,
    _criterion_value,
)
from .environmental_joint_physics import (
    EnvironmentalJointPairClearance,
    EnvironmentalJointPhysicsUnavailable,
    EnvironmentalJointResourceEstimate,
    EnvironmentalJointTupleEvaluation,
    EnvironmentalJointVesselCommandEvaluation,
    EnvironmentalJointVesselRejection,
    _post_transition_pair_clearance,
    build_environmental_candidate_transition,
    estimate_environmental_joint_resources,
)
from .environmental_joint_semantics import (
    EnvironmentalJointPairSemanticEvidence,
    EnvironmentalJointSemanticConfig,
    EnvironmentalJointSemanticTuple,
    _semantic_pair_evidence,
    generate_environmental_joint_command_options,
)
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .joint_arbiter import JointArbitrationConfig, JointArbitrationCriterion
from .multi_intent import ComponentIntentAggregation
from .rule13_progress_coordination import Rule13ProgressPlan, build_rule13_progress_plan

_EPS = 1e-10


class EnvironmentalJointExactSearchUnavailable(RuntimeError):
    """Raised when the explicit exact-search resource contract is exhausted."""

    def __init__(
        self,
        message: str,
        *,
        resource_estimate: EnvironmentalJointResourceEstimate | None = None,
        branch_nodes_visited: int | None = None,
        max_pair_evidence: int | None = None,
        max_branch_nodes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.resource_estimate = resource_estimate
        self.branch_nodes_visited = branch_nodes_visited
        self.max_pair_evidence = max_pair_evidence
        self.max_branch_nodes = max_branch_nodes


@dataclass(frozen=True, slots=True)
class EnvironmentalJointExactSearchConfig:
    """Independent bounded-work contract for the exact cached selector."""

    max_pair_evidence: int = 10_000
    max_branch_nodes: int = 10_000

    def __post_init__(self) -> None:
        for name in ("max_pair_evidence", "max_branch_nodes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class EnvironmentalJointExactSearchResult:
    semantic_config: EnvironmentalJointSemanticConfig
    arbitration_config: JointArbitrationConfig
    search_config: EnvironmentalJointExactSearchConfig
    progress_plan: Rule13ProgressPlan
    command_options: Mapping[str, tuple[WaterRelativeManeuverCommand, ...]]
    resource_estimate: EnvironmentalJointResourceEstimate
    selected: RankedEnvironmentalJointTuple
    branch_nodes_visited: int
    complete_safe_leaves_scored: int
    pruned_by_constraint: int
    pruned_by_lower_bound: int
    search_order: tuple[str, ...]

    @property
    def selected_command_index(self) -> int:
        return self.selected.evaluation.source_evaluation.command_index


@dataclass(frozen=True, slots=True)
class _LocalOption:
    evaluation: EnvironmentalJointVesselCommandEvaluation
    course_change_deg: float
    speed_change_fraction: float


@dataclass(slots=True)
class _SearchCounters:
    nodes: int = 0
    complete: int = 0
    pruned_constraint: int = 0
    pruned_bound: int = 0


def _canonical_command_index(
    vessel_ids: tuple[str, ...],
    command_options: Mapping[str, tuple[WaterRelativeManeuverCommand, ...]],
    assignment: Mapping[str, int],
) -> int:
    multiplier = 1
    result = 0
    for vessel_id in reversed(vessel_ids):
        result += assignment[vessel_id] * multiplier
        multiplier *= len(command_options[vessel_id])
    return result


def _local_option_table(
    context: EnvironmentalAvoidancePlanningContext,
    command_options: Mapping[str, tuple[WaterRelativeManeuverCommand, ...]],
    physics_config: EnvironmentalJointPhysicsConfig,
) -> tuple[
    dict[str, tuple[_LocalOption, ...]],
    dict[str, tuple[int, ...]],
]:
    table: dict[str, tuple[_LocalOption, ...]] = {}
    valid: dict[str, tuple[int, ...]] = {}
    for vessel_id in context.vessel_ids:
        inputs = context.for_vessel(vessel_id)
        following = context.following_constraint_for(vessel_id)
        rows: list[_LocalOption] = []
        accepted_indices: list[int] = []
        for option_index, command in enumerate(command_options[vessel_id]):
            reasons: list[EnvironmentalJointVesselRejection] = []
            if (
                inputs.profile is not None
                and inputs.profile.speed_envelope is not None
                and not inputs.profile.speed_envelope.permits(command.target_through_water_speed)
            ):
                reasons.append(EnvironmentalJointVesselRejection.SPEED_ENVELOPE)
            if (
                following is not None
                and command.target_through_water_speed
                > following.max_follower_speed_nm_per_min + _EPS
            ):
                reasons.append(EnvironmentalJointVesselRejection.SAME_ROUTE_FOLLOWING_SPEED)
            transition = build_environmental_candidate_transition(
                inputs.state,
                inputs.environmental_drift_velocity,
                command,
                inputs.maneuver_limits,
                motion_backend=context.motion_backend,
                motion_state=inputs.motion_state,
                evaluation_horizon_min=physics_config.pair_clearance.evaluation_horizon_min,
            )
            if not transition.reachable:
                reasons.append(EnvironmentalJointVesselRejection.PHYSICALLY_UNREACHABLE)
            evaluation = EnvironmentalJointVesselCommandEvaluation(
                vessel_id,
                command,
                transition,
                tuple(dict.fromkeys(reasons)),
            )
            course = abs(
                signed_heading_delta(
                    inputs.state.through_water_heading_deg,
                    command.target_through_water_heading_deg,
                )
            )
            initial_speed = inputs.state.through_water_speed
            speed = (
                abs(command.target_through_water_speed)
                if initial_speed == 0.0
                else abs(command.target_through_water_speed / initial_speed - 1.0)
            )
            rows.append(_LocalOption(evaluation, course, speed))
            if evaluation.accepted:
                accepted_indices.append(option_index)
        table[vessel_id] = tuple(rows)
        valid[vessel_id] = tuple(accepted_indices)
    return table, valid


def _pair_key(
    canonical_index: Mapping[str, int],
    first_id: str,
    second_id: str,
    first_option: int,
    second_option: int,
) -> tuple[str, str, int, int]:
    if canonical_index[first_id] < canonical_index[second_id]:
        return first_id, second_id, first_option, second_option
    return second_id, first_id, second_option, first_option


def _optimistic_score(
    criteria: tuple[JointArbitrationCriterion, ...],
    *,
    current_total_course: float,
    current_total_speed: float,
    current_max_course: float,
    current_max_speed: float,
    remaining: tuple[str, ...],
    minimum_local: Mapping[str, tuple[float, float]],
    assignment: Mapping[str, int],
    vessel_ids: tuple[str, ...],
    valid_indices: Mapping[str, tuple[int, ...]],
    command_options: Mapping[str, tuple[WaterRelativeManeuverCommand, ...]],
) -> tuple[float, ...]:
    min_course_remaining = tuple(minimum_local[v][0] for v in remaining)
    min_speed_remaining = tuple(minimum_local[v][1] for v in remaining)
    values: list[float] = []
    for criterion in criteria:
        if criterion is JointArbitrationCriterion.MIN_TOTAL_COURSE_CHANGE:
            values.append(current_total_course + sum(min_course_remaining))
        elif criterion is JointArbitrationCriterion.MIN_TOTAL_SPEED_CHANGE:
            values.append(current_total_speed + sum(min_speed_remaining))
        elif criterion is JointArbitrationCriterion.MIN_MAX_COURSE_CHANGE:
            values.append(max((current_max_course, *min_course_remaining), default=0.0))
        elif criterion is JointArbitrationCriterion.MIN_MAX_SPEED_CHANGE:
            values.append(max((current_max_speed, *min_speed_remaining), default=0.0))
        elif criterion in {
            JointArbitrationCriterion.MAX_PREDICTED_MIN_SEPARATION,
            JointArbitrationCriterion.MAX_MIN_CERTIFIED_DOMAIN_MARGIN,
        }:
            # No safe finite optimistic bound is available without completing all pair
            # objective evidence.  -inf disables pruning at this and later tie levels.
            values.append(-inf)
        elif criterion is JointArbitrationCriterion.GRID_ORDER:
            optimistic_assignment = dict(assignment)
            for vessel_id in remaining:
                optimistic_assignment[vessel_id] = min(valid_indices[vessel_id])
            values.append(
                float(
                    _canonical_command_index(
                        vessel_ids,
                        command_options,
                        optimistic_assignment,
                    )
                )
            )
        else:  # pragma: no cover - exhaustive enum guard
            raise AssertionError(f"unhandled arbitration criterion {criterion!r}")
    return tuple(values)


def select_environmental_joint_exact(
    context: EnvironmentalAvoidancePlanningContext,
    aggregation: ComponentIntentAggregation,
    *,
    semantic_config: EnvironmentalJointSemanticConfig,
    arbitration_config: JointArbitrationConfig,
    search_config: EnvironmentalJointExactSearchConfig = EnvironmentalJointExactSearchConfig(),
) -> EnvironmentalJointExactSearchResult:
    """Select the exact A104/A105/A106 winner without materializing the full grid."""

    if not isinstance(context, EnvironmentalAvoidancePlanningContext):
        raise TypeError("context must be EnvironmentalAvoidancePlanningContext")
    if not isinstance(aggregation, ComponentIntentAggregation):
        raise TypeError("aggregation must be ComponentIntentAggregation")
    if not isinstance(semantic_config, EnvironmentalJointSemanticConfig):
        raise TypeError("semantic_config must be EnvironmentalJointSemanticConfig")
    if not isinstance(arbitration_config, JointArbitrationConfig):
        raise TypeError("arbitration_config must be JointArbitrationConfig")
    if not isinstance(search_config, EnvironmentalJointExactSearchConfig):
        raise TypeError("search_config must be EnvironmentalJointExactSearchConfig")
    if not aggregation.compatible:
        raise NoEnvironmentalJointTuple("environmental JOINT aggregation is not compatible")

    progress_plan = build_rule13_progress_plan(
        aggregation,
        policy=semantic_config.rule13_progress_coordination,
    )
    command_options = generate_environmental_joint_command_options(
        context,
        aggregation,
        policy=semantic_config.candidate_policy,
        progress_plan=progress_plan,
    )
    resource_estimate = estimate_environmental_joint_resources(
        context.vessel_ids,
        command_options,
    )
    if resource_estimate.pair_command_evidence_count > search_config.max_pair_evidence:
        raise EnvironmentalJointExactSearchUnavailable(
            "environmental JOINT exact search pair evidence exceeds configured resources",
            resource_estimate=resource_estimate,
            max_pair_evidence=search_config.max_pair_evidence,
            max_branch_nodes=search_config.max_branch_nodes,
        )

    vessel_ids = context.vessel_ids
    canonical_index = {vessel_id: index for index, vessel_id in enumerate(vessel_ids)}
    local_options, valid_indices = _local_option_table(
        context, command_options, semantic_config.physics
    )
    if any(not valid_indices[vessel_id] for vessel_id in vessel_ids):
        raise NoEnvironmentalJointTuple("environmental JOINT exact search has a vessel with no admissible option")

    pair_clearance: dict[tuple[str, str, int, int], EnvironmentalJointPairClearance] = {}
    pair_semantics: dict[tuple[str, str, int, int], EnvironmentalJointPairSemanticEvidence] = {}
    constraint_degree = {vessel_id: 0 for vessel_id in vessel_ids}

    for first_id, second_id in combinations(vessel_ids, 2):
        restrictive = False
        for first_option in valid_indices[first_id]:
            for second_option in valid_indices[second_id]:
                first_eval = local_options[first_id][first_option].evaluation
                second_eval = local_options[second_id][second_option].evaluation
                clearance = _post_transition_pair_clearance(
                    context,
                    first_id,
                    second_id,
                    first_eval.transition,
                    second_eval.transition,
                    semantic_config.physics,
                )
                key = (first_id, second_id, first_option, second_option)
                pair_clearance[key] = clearance
                if not clearance.accepted:
                    restrictive = True
                    continue
                source = EnvironmentalJointTupleEvaluation(
                    0,
                    (first_eval, second_eval),
                    (clearance,),
                    True,
                    clearance.predicted_min_separation_nm,
                )
                evidence = _semantic_pair_evidence(
                    context,
                    aggregation,
                    progress_plan,
                    source,
                    first_id,
                    second_id,
                    semantic_config,
                )
                pair_semantics[key] = evidence
                if not evidence.accepted:
                    restrictive = True
        if restrictive:
            constraint_degree[first_id] += 1
            constraint_degree[second_id] += 1

    # Most-constrained-first affects work only. Canonical grid order is reconstructed at
    # each leaf and remains the final exact A106 tie-break.
    search_order = tuple(
        sorted(
            vessel_ids,
            key=lambda vessel_id: (-constraint_degree[vessel_id], canonical_index[vessel_id]),
        )
    )
    minimum_local = {
        vessel_id: (
            min(local_options[vessel_id][i].course_change_deg for i in valid_indices[vessel_id]),
            min(local_options[vessel_id][i].speed_change_fraction for i in valid_indices[vessel_id]),
        )
        for vessel_id in vessel_ids
    }

    assignment: dict[str, int] = {}
    counters = _SearchCounters()
    best: RankedEnvironmentalJointTuple | None = None

    def cached_pair(
        first_id: str,
        second_id: str,
        first_option: int,
        second_option: int,
    ) -> tuple[EnvironmentalJointPairClearance, EnvironmentalJointPairSemanticEvidence | None]:
        key = _pair_key(
            canonical_index,
            first_id,
            second_id,
            first_option,
            second_option,
        )
        clearance = pair_clearance[key]
        return clearance, pair_semantics.get(key)

    def recurse(
        depth: int,
        total_course: float,
        total_speed: float,
        max_course: float,
        max_speed: float,
    ) -> None:
        nonlocal best
        counters.nodes += 1
        if counters.nodes > search_config.max_branch_nodes:
            raise EnvironmentalJointExactSearchUnavailable(
                "environmental JOINT exact search branch-node budget exceeded",
                resource_estimate=resource_estimate,
                branch_nodes_visited=counters.nodes,
                max_pair_evidence=search_config.max_pair_evidence,
                max_branch_nodes=search_config.max_branch_nodes,
            )

        if best is not None:
            lower = _optimistic_score(
                arbitration_config.criteria,
                current_total_course=total_course,
                current_total_speed=total_speed,
                current_max_course=max_course,
                current_max_speed=max_speed,
                remaining=search_order[depth:],
                minimum_local=minimum_local,
                assignment=assignment,
                vessel_ids=vessel_ids,
                valid_indices=valid_indices,
                command_options=command_options,
            )
            if lower > best.score.criterion_values:
                counters.pruned_bound += 1
                return

        if depth == len(search_order):
            pair_rows: list[EnvironmentalJointPairClearance] = []
            semantic_rows: list[EnvironmentalJointPairSemanticEvidence] = []
            predicted: list[float] = []
            for first_id, second_id in combinations(vessel_ids, 2):
                clearance, evidence = cached_pair(
                    first_id,
                    second_id,
                    assignment[first_id],
                    assignment[second_id],
                )
                if not clearance.accepted or evidence is None or not evidence.accepted:
                    raise AssertionError("branch admitted an unsafe complete assignment")
                pair_rows.append(clearance)
                semantic_rows.append(evidence)
                if clearance.predicted_min_separation_nm is not None:
                    predicted.append(clearance.predicted_min_separation_nm)
            vessel_rows = tuple(
                local_options[vessel_id][assignment[vessel_id]].evaluation
                for vessel_id in vessel_ids
            )
            command_index = _canonical_command_index(vessel_ids, command_options, assignment)
            source = EnvironmentalJointTupleEvaluation(
                command_index,
                vessel_rows,
                tuple(pair_rows),
                True,
                min(predicted) if predicted else None,
            )
            semantic = EnvironmentalJointSemanticTuple(
                source,
                tuple(semantic_rows),
                True,
            )
            values = tuple(
                _criterion_value(criterion, context, semantic)
                for criterion in arbitration_config.criteria
            )
            ranked = RankedEnvironmentalJointTuple(
                semantic,
                EnvironmentalJointArbitrationScore(command_index, values),
            )
            counters.complete += 1
            if best is None or ranked.score.criterion_values < best.score.criterion_values:
                best = ranked
            return

        vessel_id = search_order[depth]
        for option_index in valid_indices[vessel_id]:
            compatible = True
            for assigned_id, assigned_option in assignment.items():
                clearance, evidence = cached_pair(
                    assigned_id,
                    vessel_id,
                    assigned_option,
                    option_index,
                )
                if not clearance.accepted or evidence is None or not evidence.accepted:
                    counters.pruned_constraint += 1
                    compatible = False
                    break
            if not compatible:
                continue
            assignment[vessel_id] = option_index
            local = local_options[vessel_id][option_index]
            recurse(
                depth + 1,
                total_course + local.course_change_deg,
                total_speed + local.speed_change_fraction,
                max(max_course, local.course_change_deg),
                max(max_speed, local.speed_change_fraction),
            )
            del assignment[vessel_id]

    recurse(0, 0.0, 0.0, 0.0, 0.0)
    if best is None:
        raise NoEnvironmentalJointTuple("environmental JOINT exact search contains no safe tuple")

    return EnvironmentalJointExactSearchResult(
        semantic_config,
        arbitration_config,
        search_config,
        progress_plan,
        command_options,
        resource_estimate,
        best,
        counters.nodes,
        counters.complete,
        counters.pruned_constraint,
        counters.pruned_bound,
        search_order,
    )
