'''Physical-only JOINT candidate evaluation for environmental water-relative commands.

A104 is the first environmental multi-vessel physical backend.  It deliberately stops
below COLREG intent generation, Rule-13 coordination, ranking, ownership execution and
review loops.  The caller supplies an ordered command option list for every vessel in a
JOINT a83 planning context.

Every Cartesian command tuple is evaluated on one common environmental world model:
- each vessel receives an exact a84 water-relative bounded transition;
- every unordered vessel pair is screened through a85 continuous transition clearance,
  including pairs that do not currently carry an active responsibility edge;
- after the latest command completion, pair safety continues on constant ground motion
  to the configured horizon, with optional hull-oriented domain certification;
- one unreachable/speed-envelope-invalid vessel or one unsafe pair rejects the tuple.

No command is applied and the historical ground-referenced JOINT backend is not used.
'''

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, product
from math import isfinite
from typing import Mapping

from ..cpa import ClosestApproach
from .domain_clearance import DomainClearanceCertification
from .environmental_candidate_transition import (
    EnvironmentalCandidateTransition,
    EnvironmentalCommandTrajectory,
    build_environmental_candidate_transition,
)
from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_pair_clearance import (
    EnvironmentalPairTransitionClearance,
    EnvironmentalPairTransitionConfig,
)
from .environmental_pair_command_clearance import evaluate_environmental_pair_command_clearance
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .fleet_supervision import FleetSupervisorBackend

_EPS = 1e-10


class EnvironmentalJointPhysicsError(RuntimeError):
    '''Base error for invalid environmental JOINT physical evaluations.'''


class EnvironmentalJointPhysicsUnavailable(EnvironmentalJointPhysicsError):
    '''Raised when the explicit Cartesian product exceeds configured resources.'''

    def __init__(
        self,
        message: str,
        *,
        estimate: 'EnvironmentalJointResourceEstimate | None' = None,
        max_combinations: int | None = None,
    ) -> None:
        super().__init__(message)
        self.estimate = estimate
        self.max_combinations = max_combinations


class EnvironmentalJointVesselRejection(str, Enum):
    SPEED_ENVELOPE = 'speed_envelope'
    PHYSICALLY_UNREACHABLE = 'physically_unreachable'
    SAME_ROUTE_FOLLOWING_SPEED = 'same_route_following_speed'


class EnvironmentalJointPairRejection(str, Enum):
    DOMAIN_EVIDENCE_INCOMPLETE = 'domain_evidence_incomplete'
    TRANSITION_CLEARANCE = 'transition_clearance'
    POST_TRANSITION_CLEARANCE = 'post_transition_clearance'


@dataclass(frozen=True, slots=True)
class EnvironmentalJointResourceEstimate:
    '''Exact finite-work cardinalities for one explicit JOINT option product.

    ``cartesian_tuple_count`` is the number of rows that the exact evaluator must
    materialize/rank.  The other counts are the unique memoizable physical/semantic
    evidence units after pair-local caching; none of these quantities is a physical
    feasibility result.
    '''

    vessel_ids: tuple[str, ...]
    option_counts: tuple[tuple[str, int], ...]
    cartesian_tuple_count: int
    vessel_transition_evidence_count: int
    pair_command_evidence_count: int


def estimate_environmental_joint_resources(
    vessel_ids: tuple[str, ...],
    command_options: Mapping[str, tuple[WaterRelativeManeuverCommand, ...]],
) -> EnvironmentalJointResourceEstimate:
    '''Return exact cardinalities without evaluating or pruning any command tuple.'''

    ids = tuple(vessel_ids)
    if not ids:
        raise ValueError('vessel_ids must be non-empty')
    expected = set(ids)
    if len(expected) != len(ids):
        raise ValueError('vessel_ids must be unique')
    _require_exact_keys('command_options', set(command_options), expected)
    counts: list[tuple[str, int]] = []
    for vessel_id in ids:
        count = len(command_options[vessel_id])
        if count < 1:
            raise ValueError(f'command_options for {vessel_id!r} must be non-empty')
        counts.append((vessel_id, count))
    product_count = 1
    for _, count in counts:
        product_count *= count
    pair_count = sum(
        first_count * second_count
        for (_, first_count), (_, second_count) in combinations(counts, 2)
    )
    return EnvironmentalJointResourceEstimate(
        vessel_ids=ids,
        option_counts=tuple(counts),
        cartesian_tuple_count=product_count,
        vessel_transition_evidence_count=sum(count for _, count in counts),
        pair_command_evidence_count=pair_count,
    )


@dataclass(frozen=True, slots=True)
class EnvironmentalJointPhysicsConfig:
    pair_clearance: EnvironmentalPairTransitionConfig
    max_combinations: int = 10000

    def __post_init__(self) -> None:
        if not isinstance(self.pair_clearance, EnvironmentalPairTransitionConfig):
            raise TypeError('pair_clearance must be EnvironmentalPairTransitionConfig')
        if not isinstance(self.max_combinations, int) or isinstance(self.max_combinations, bool):
            raise ValueError('max_combinations must be an integer')
        if self.max_combinations < 1:
            raise ValueError('max_combinations must be at least 1')


@dataclass(frozen=True, slots=True)
class EnvironmentalJointVesselCommandEvaluation:
    vessel_id: str
    command: WaterRelativeManeuverCommand
    transition: EnvironmentalCommandTrajectory
    rejection_reasons: tuple[EnvironmentalJointVesselRejection, ...]

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons


@dataclass(frozen=True, slots=True)
class EnvironmentalJointPairClearance:
    first_id: str
    second_id: str
    transition: EnvironmentalPairTransitionClearance | None
    post_transition_cpa: ClosestApproach | None
    post_transition_domain_clearance: DomainClearanceCertification | None
    predicted_min_separation_nm: float | None
    rejection_reasons: tuple[EnvironmentalJointPairRejection, ...]
    same_route_following_delegated: bool = False
    following_speed_ceiling_nm_per_min: float | None = None

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons

    @property
    def minimum_certified_domain_margin_nm(self) -> float | None:
        margins = []
        if self.transition is not None and self.transition.domain is not None:
            margins.append(self.transition.domain.conservative_lower_bound_nm)
        if self.post_transition_domain_clearance is not None:
            margins.append(self.post_transition_domain_clearance.conservative_lower_bound_nm)
        return min(margins) if margins else None


@dataclass(frozen=True, slots=True)
class EnvironmentalJointTupleEvaluation:
    command_index: int
    vessel_commands: tuple[EnvironmentalJointVesselCommandEvaluation, ...]
    pair_clearances: tuple[EnvironmentalJointPairClearance, ...]
    accepted: bool
    predicted_min_separation_nm: float | None

    def command_for(self, vessel_id: str) -> WaterRelativeManeuverCommand:
        for item in self.vessel_commands:
            if item.vessel_id == vessel_id:
                return item.command
        raise KeyError(vessel_id)

    def vessel_evaluation(self, vessel_id: str) -> EnvironmentalJointVesselCommandEvaluation:
        for item in self.vessel_commands:
            if item.vessel_id == vessel_id:
                return item
        raise KeyError(vessel_id)

    def pair_clearance(self, first_id: str, second_id: str) -> EnvironmentalJointPairClearance:
        key = tuple(sorted((first_id, second_id)))
        for item in self.pair_clearances:
            if (item.first_id, item.second_id) == key:
                return item
        raise KeyError(key)

    @property
    def minimum_certified_domain_margin_nm(self) -> float | None:
        if not self.pair_clearances:
            return None
        margins = tuple(item.minimum_certified_domain_margin_nm for item in self.pair_clearances)
        if any(value is None for value in margins):
            return None
        return min(value for value in margins if value is not None)


@dataclass(frozen=True, slots=True)
class EnvironmentalJointPhysicsGrid:
    component_id: str
    vessel_ids: tuple[str, ...]
    evaluations: tuple[EnvironmentalJointTupleEvaluation, ...]
    resource_estimate: EnvironmentalJointResourceEstimate

    @property
    def safe_tuples(self) -> tuple[EnvironmentalJointTupleEvaluation, ...]:
        return tuple(item for item in self.evaluations if item.accepted)

    @property
    def has_safe_tuple(self) -> bool:
        return any(item.accepted for item in self.evaluations)


@dataclass(frozen=True, slots=True)
class _DomainPose:
    position: tuple[float, float]
    heading_deg: float


def _require_exact_keys(name: str, actual: set[str], expected: set[str]) -> None:
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f'{name} keys must match context vessels; missing={missing}, extra={extra}')


def _post_transition_pair_clearance(
    context: EnvironmentalAvoidancePlanningContext,
    first_id: str,
    second_id: str,
    first_transition: EnvironmentalCommandTrajectory,
    second_transition: EnvironmentalCommandTrajectory,
    config: EnvironmentalJointPhysicsConfig,
) -> EnvironmentalJointPairClearance:
    first = context.for_vessel(first_id)
    second = context.for_vessel(second_id)
    reasons: list[EnvironmentalJointPairRejection] = []
    if context.same_route_pair_delegated(first_id, second_id):
        follower_constraint = context.following_constraint_for(first_id)
        if follower_constraint is None or follower_constraint.leader_id != second_id:
            candidate = context.following_constraint_for(second_id)
            if candidate is not None and candidate.leader_id == first_id:
                follower_constraint = candidate
            else:
                follower_constraint = None
        return EnvironmentalJointPairClearance(
            first_id,
            second_id,
            None,
            None,
            None,
            None,
            (),
            True,
            None if follower_constraint is None else follower_constraint.max_follower_speed_nm_per_min,
        )
    first_domain = first.domain
    second_domain = second.domain
    domains_complete = (first_domain is None) == (second_domain is None)
    if not domains_complete:
        reasons.append(EnvironmentalJointPairRejection.DOMAIN_EVIDENCE_INCOMPLETE)

    if not first_transition.reachable or not second_transition.reachable:
        return EnvironmentalJointPairClearance(
            first_id, second_id, None, None, None, None, tuple(reasons)
        )

    evidence = evaluate_environmental_pair_command_clearance(
        first_transition,
        second_transition,
        config.pair_clearance,
        first_domain=first_domain if domains_complete else None,
        second_domain=second_domain if domains_complete else None,
    )
    if not evidence.transition_certified_clear:
        reasons.append(EnvironmentalJointPairRejection.TRANSITION_CLEARANCE)
    if not evidence.post_transition_certified_clear:
        reasons.append(EnvironmentalJointPairRejection.POST_TRANSITION_CLEARANCE)

    return EnvironmentalJointPairClearance(
        first_id,
        second_id,
        evidence.transition,
        evidence.post_transition_cpa,
        evidence.post_transition_domain_clearance,
        evidence.predicted_min_separation_nm,
        tuple(dict.fromkeys(reasons)),
    )


def evaluate_environmental_joint_command_options(
    context: EnvironmentalAvoidancePlanningContext,
    *,
    command_options: Mapping[str, tuple[WaterRelativeManeuverCommand, ...]],
    config: EnvironmentalJointPhysicsConfig,
) -> EnvironmentalJointPhysicsGrid:
    """Evaluate an explicit environmental JOINT Cartesian command product physically.

    Per-vessel transitions and per-pair clearances are memoized within this pure
    evaluation. Those quantities depend only on their own command option(s), context,
    and immutable config, so reusing them across Cartesian tuples is exactly equivalent
    to recomputing them for every tuple while avoiding combinatorial duplicate work.
    """

    if not isinstance(context, EnvironmentalAvoidancePlanningContext):
        raise TypeError('context must be EnvironmentalAvoidancePlanningContext')
    if not isinstance(config, EnvironmentalJointPhysicsConfig):
        raise TypeError('config must be EnvironmentalJointPhysicsConfig')
    if context.backend is not FleetSupervisorBackend.JOINT or len(context.vessel_ids) < 3:
        raise ValueError('environmental JOINT physics requires a JOINT context with at least three vessels')

    vessel_ids = context.vessel_ids
    expected = set(vessel_ids)
    _require_exact_keys('command_options', set(command_options), expected)
    for vessel_id in vessel_ids:
        if not command_options[vessel_id]:
            raise ValueError(f'command_options for {vessel_id!r} must be non-empty')
        if any(not isinstance(item, WaterRelativeManeuverCommand) for item in command_options[vessel_id]):
            raise TypeError('command_options must contain WaterRelativeManeuverCommand values')

    option_lists = tuple(command_options[vessel_id] for vessel_id in vessel_ids)
    resource_estimate = estimate_environmental_joint_resources(vessel_ids, command_options)
    combination_count = resource_estimate.cartesian_tuple_count
    if combination_count > config.max_combinations:
        raise EnvironmentalJointPhysicsUnavailable(
            f'environmental JOINT candidate product has {combination_count} combinations, '
            f'exceeding max_combinations={config.max_combinations}',
            estimate=resource_estimate,
            max_combinations=config.max_combinations,
        )

    # A transition and its vessel-level admissibility depend on one vessel option only,
    # not on the other members of the Cartesian tuple. Compute each exactly once.
    vessel_option_evaluations: dict[
        str, tuple[EnvironmentalJointVesselCommandEvaluation, ...]
    ] = {}
    transitions_by_option: dict[str, tuple[EnvironmentalCommandTrajectory, ...]] = {}
    for vessel_id, options in zip(vessel_ids, option_lists, strict=True):
        inputs = context.for_vessel(vessel_id)
        following = context.following_constraint_for(vessel_id)
        evaluations_for_vessel: list[EnvironmentalJointVesselCommandEvaluation] = []
        transitions_for_vessel: list[EnvironmentalCommandTrajectory] = []
        for command in options:
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
                evaluation_horizon_min=config.pair_clearance.evaluation_horizon_min,
            )
            if not transition.reachable:
                reasons.append(EnvironmentalJointVesselRejection.PHYSICALLY_UNREACHABLE)
            transitions_for_vessel.append(transition)
            evaluations_for_vessel.append(
                EnvironmentalJointVesselCommandEvaluation(
                    vessel_id,
                    command,
                    transition,
                    tuple(dict.fromkeys(reasons)),
                )
            )
        vessel_option_evaluations[vessel_id] = tuple(evaluations_for_vessel)
        transitions_by_option[vessel_id] = tuple(transitions_for_vessel)

    # Pair clearance depends only on the two participating option indices. Lazily cache
    # it because tuples rejected at vessel level need no pair calculation at all.
    pair_cache: dict[tuple[str, str, int, int], EnvironmentalJointPairClearance] = {}
    pair_ids = tuple(combinations(vessel_ids, 2))
    vessel_index = {vessel_id: index for index, vessel_id in enumerate(vessel_ids)}

    evaluations: list[EnvironmentalJointTupleEvaluation] = []
    option_index_lists = tuple(tuple(range(len(options))) for options in option_lists)
    for index, option_indices in enumerate(product(*option_index_lists)):
        vessel_evaluations = tuple(
            vessel_option_evaluations[vessel_id][option_index]
            for vessel_id, option_index in zip(vessel_ids, option_indices, strict=True)
        )

        pair_clearances: list[EnvironmentalJointPairClearance] = []
        if all(item.accepted for item in vessel_evaluations):
            for first_id, second_id in pair_ids:
                first_option_index = option_indices[vessel_index[first_id]]
                second_option_index = option_indices[vessel_index[second_id]]
                key = (first_id, second_id, first_option_index, second_option_index)
                clearance = pair_cache.get(key)
                if clearance is None:
                    clearance = _post_transition_pair_clearance(
                        context,
                        first_id,
                        second_id,
                        transitions_by_option[first_id][first_option_index],
                        transitions_by_option[second_id][second_option_index],
                        config,
                    )
                    pair_cache[key] = clearance
                pair_clearances.append(clearance)

        accepted = (
            all(item.accepted for item in vessel_evaluations)
            and all(item.accepted for item in pair_clearances)
            and len(pair_clearances) == len(pair_ids)
        )
        predicted_values = tuple(
            item.predicted_min_separation_nm
            for item in pair_clearances
            if item.predicted_min_separation_nm is not None
        )
        evaluations.append(
            EnvironmentalJointTupleEvaluation(
                command_index=index,
                vessel_commands=vessel_evaluations,
                pair_clearances=tuple(pair_clearances),
                accepted=accepted,
                predicted_min_separation_nm=(
                    min(predicted_values) if predicted_values else None
                ),
            )
        )

    return EnvironmentalJointPhysicsGrid(
        component_id=context.component_id,
        vessel_ids=vessel_ids,
        evaluations=tuple(evaluations),
        resource_estimate=resource_estimate,
    )
