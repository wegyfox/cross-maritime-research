"""Profile-aware avoidance planning helpers for one snapshot-bound command owner.

A44 deliberately captures physical/profile evidence but does not choose how that evidence
is consumed by pair or joint planning.  This module is the narrow bridge from an
``AvoidancePlanningContext`` + ``FleetCommandOwner`` to the existing a42 physical
screens and a43 explicit arbiters.

No safety fallback is implicit.  The caller chooses either profile-domain geometry or
the legacy circular-separation geometry.  Profile-domain mode requires complete domain
evidence for every owned vessel and fails closed otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .avoidance import AvoidanceCandidateConfig
from .coupled_avoidance import (
    CoupledAvoidanceConfig,
    CoupledAvoidanceGrid,
    evaluate_coupled_avoidance_candidates,
)
from .dynamics import ManeuverCommand
from .fleet_supervision import FleetCommandOwner, FleetSupervisorBackend
from .joint_arbiter import JointArbitrationConfig, JointArbitrationResult, arbitrate_joint_tuples
from .joint_avoidance import JointAvoidanceConfig, JointAvoidanceGrid, evaluate_joint_avoidance_candidates
from .maneuver_intent import PairManeuverIntents, StandOnStage, select_pair_maneuver_intents
from .multi_intent import ComponentIntentAggregation, aggregate_component_intents
from .pair_arbiter import PairArbitrationConfig, PairArbitrationResult, arbitrate_coupled_pairs
from .planning_context import AvoidancePlanningContext
from .ship_domain import ShipDomain
from .rule13_progress_coordination import (
    Rule13ProgressCoordinationPolicy,
    Rule13ProgressPlan,
    build_rule13_progress_plan,
)


class PlanningHelperError(RuntimeError):
    """Base error for owner/context/config mismatches in planning helpers."""


class PlanningOwnerMismatch(PlanningHelperError):
    """Raised when the supplied owner is not the owner captured by the context."""


class MissingPlanningDomainEvidence(PlanningHelperError):
    """Raised when profile-domain planning is requested without complete domains."""


class PlanningSafetyGeometry(str, Enum):
    """Explicit geometry used by physical candidate safety screening."""

    PROFILE_DOMAINS_REQUIRED = "profile_domains_required"
    CIRCULAR_SEPARATION = "circular_separation"


@dataclass(frozen=True, slots=True)
class PairAvoidancePlanningResult:
    context: AvoidancePlanningContext
    intents: PairManeuverIntents
    grid: CoupledAvoidanceGrid
    arbitration: PairArbitrationResult
    safety_geometry: PlanningSafetyGeometry

    @property
    def selected_commands(self) -> dict[str, ManeuverCommand]:
        selected = self.arbitration.selected.evaluation
        first_id, second_id = self.context.vessel_ids
        return {
            first_id: selected.first_candidate.command,
            second_id: selected.second_candidate.command,
        }


@dataclass(frozen=True, slots=True)
class JointAvoidancePlanningResult:
    context: AvoidancePlanningContext
    aggregation: ComponentIntentAggregation
    grid: JointAvoidanceGrid
    arbitration: JointArbitrationResult
    safety_geometry: PlanningSafetyGeometry
    rule13_progress_plan: Rule13ProgressPlan | None = None

    @property
    def selected_commands(self) -> dict[str, ManeuverCommand]:
        selected = self.arbitration.selected.evaluation
        return {
            item.vessel_id: item.command
            for item in selected.vessel_commands
        }


def _validate_owner_context(
    context: AvoidancePlanningContext,
    owner: FleetCommandOwner,
    *,
    backend: FleetSupervisorBackend,
) -> None:
    if context.backend is not backend or owner.backend is not backend:
        raise PlanningOwnerMismatch(
            f"planning helper requires {backend.value!r} backend"
        )
    if (
        context.component_id != owner.component_id
        or context.vessel_ids != owner.vessel_ids
        or context.pair_keys != owner.pair_keys
    ):
        raise PlanningOwnerMismatch("planning context does not match supplied command owner")


def _exact_mapping_keys(name: str, mapping: Mapping[str, object], vessel_ids: tuple[str, ...]) -> None:
    expected = set(vessel_ids)
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PlanningHelperError(f"{name} keys must exactly match owner vessels; missing={missing}, extra={extra}")


def _planning_domains(
    context: AvoidancePlanningContext,
    geometry: PlanningSafetyGeometry,
) -> dict[str, ShipDomain] | None:
    if geometry is PlanningSafetyGeometry.CIRCULAR_SEPARATION:
        return None
    if geometry is not PlanningSafetyGeometry.PROFILE_DOMAINS_REQUIRED:
        raise AssertionError(f"unhandled planning safety geometry {geometry!r}")

    missing = tuple(sorted(vessel_id for vessel_id, domain in context.domains.items() if domain is None))
    if missing:
        raise MissingPlanningDomainEvidence(
            f"profile-domain planning requires domains for every owner vessel; missing={list(missing)}"
        )
    return {
        vessel_id: domain
        for vessel_id, domain in context.domains.items()
        if domain is not None
    }


def plan_pair_owner_avoidance(
    context: AvoidancePlanningContext,
    owner: FleetCommandOwner,
    *,
    candidate_configs: Mapping[str, AvoidanceCandidateConfig],
    coupled_config: CoupledAvoidanceConfig,
    arbitration_config: PairArbitrationConfig,
    stand_on_stage: StandOnStage = StandOnStage.NORMAL,
    safety_geometry: PlanningSafetyGeometry = PlanningSafetyGeometry.PROFILE_DOMAINS_REQUIRED,
) -> PairAvoidancePlanningResult:
    """Evaluate and explicitly arbitrate one isolated pair from owner-bound context."""

    _validate_owner_context(context, owner, backend=FleetSupervisorBackend.PAIR)
    if len(context.vessel_ids) != 2 or len(owner.component.decisions) != 1:
        raise PlanningOwnerMismatch("pair planning requires exactly one isolated pair")
    _exact_mapping_keys("candidate_configs", candidate_configs, context.vessel_ids)

    first_id, second_id = context.vessel_ids
    lifecycle_decision = owner.component.decisions[0].entry.decision
    pair_intents = select_pair_maneuver_intents(
        lifecycle_decision,
        stand_on_stage=stand_on_stage,
    )
    first_intent = pair_intents.for_vessel(first_id)
    second_intent = pair_intents.for_vessel(second_id)
    domains = _planning_domains(context, safety_geometry)

    grid = evaluate_coupled_avoidance_candidates(
        context.states[first_id],
        context.states[second_id],
        context.limits[first_id],
        context.limits[second_id],
        first_intent,
        second_intent,
        candidate_configs[first_id],
        candidate_configs[second_id],
        coupled_config,
        first_domain=None if domains is None else domains[first_id],
        second_domain=None if domains is None else domains[second_id],
    )
    arbitration = arbitrate_coupled_pairs(grid, config=arbitration_config)
    return PairAvoidancePlanningResult(
        context=context,
        intents=PairManeuverIntents(first_intent, second_intent),
        grid=grid,
        arbitration=arbitration,
        safety_geometry=safety_geometry,
    )


def plan_joint_owner_avoidance(
    context: AvoidancePlanningContext,
    owner: FleetCommandOwner,
    *,
    command_options: Mapping[str, tuple[ManeuverCommand, ...]],
    joint_config: JointAvoidanceConfig,
    arbitration_config: JointArbitrationConfig,
    stand_on_stages=None,
    safety_geometry: PlanningSafetyGeometry = PlanningSafetyGeometry.PROFILE_DOMAINS_REQUIRED,
    rule13_progress_coordination: Rule13ProgressCoordinationPolicy | None = None,
) -> JointAvoidancePlanningResult:
    """Evaluate and explicitly arbitrate one multi-vessel owner from bound context."""

    _validate_owner_context(context, owner, backend=FleetSupervisorBackend.JOINT)
    _exact_mapping_keys("command_options", command_options, context.vessel_ids)
    domains = _planning_domains(context, safety_geometry)

    aggregation = aggregate_component_intents(
        owner.component,
        stand_on_stages=stand_on_stages,
    )
    if aggregation.component_id != context.component_id:
        raise PlanningOwnerMismatch("intent aggregation does not match planning context component")
    progress_plan = (
        None
        if rule13_progress_coordination is None
        else build_rule13_progress_plan(
            aggregation, policy=rule13_progress_coordination
        )
    )

    grid = evaluate_joint_avoidance_candidates(
        aggregation,
        states=context.states,
        limits=context.limits,
        command_options=command_options,
        config=joint_config,
        domains=domains,
        rule13_progress_plan=progress_plan,
    )
    arbitration = arbitrate_joint_tuples(grid, config=arbitration_config)
    return JointAvoidancePlanningResult(
        context=context,
        aggregation=aggregation,
        grid=grid,
        arbitration=arbitration,
        safety_geometry=safety_geometry,
        rule13_progress_plan=progress_plan,
    )
