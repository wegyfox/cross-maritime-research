"""Explicit environmental JOINT→PAIR command-ownership transfer at one recapture boundary.

A115 consumes only the ``PAIR_BACKEND_REQUIRED`` disposition already produced by a114.
The canonical fleet observation was committed by a113 and classified by a114, so this
layer must not observe ``EncounterRegistry`` again at the delegation timestamp.

Before any PAIR command mutation it admits the caller's route targets for the exact
recaptured pair.  It then clones the committed pair lifecycle, rebuilds the established
a83/a87--a92 environmental pair planning chain, atomically applies the selected
water-relative pair commands at the same timestamp, starts a fresh a100 pair cycle and
wraps it in a102 using the already-committed fleet observation.

The transfer owns exactly one isolated PAIR.  It does not coordinate several touching
owners, perform JOINT recapture, synthesize route targets, or write nominal-route
commands directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .encounter_registry import EncounterRegistry
from .environmental_candidate_generation import (
    EnvironmentalPairCommandGrid,
    generate_environmental_pair_command_grid,
)
from .environmental_fleet_guarded_pair_control import (
    EnvironmentalFleetGuardedPairConfig,
    EnvironmentalFleetGuardedPairControlSession,
)
from .environmental_fleet_joint_recapture_dispatch import (
    EnvironmentalFleetJointRecaptureDispatchAction,
    EnvironmentalFleetJointRecaptureDispatchResult,
)
from .environmental_pair_arbiter import (
    EnvironmentalPairArbitrationResult,
    arbitrate_environmental_pair_rows,
)
from .environmental_pair_control_cycle import (
    EnvironmentalPairControlCycleConfig,
    EnvironmentalPairControlSession,
)
from .environmental_pair_execution import (
    AppliedEnvironmentalPairPlan,
    apply_environmental_pair_execution_plan,
)
from .environmental_pair_grid_evaluation import (
    EnvironmentalPairGridEvaluationBatch,
    evaluate_environmental_pair_command_grid,
)
from .environmental_pair_plan import (
    EnvironmentalPairExecutionPlan,
    build_environmental_pair_execution_plan,
)
from .environmental_planning_context import (
    EnvironmentalAvoidancePlanningContext,
    build_environmental_avoidance_planning_context,
)
from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
from .environmental_rule13_progress import (
    EnvironmentalRule13ProgressBatch,
    filter_environmental_rule13_progress,
)
from .environmental_world import EnvironmentalWorld2D
from .fleet_supervision import FleetCommandOwner, FleetSupervisorBackend
from .lifecycle import EncounterLifecycle, EncounterLifecycleDecision
from .maneuver_intent import PairManeuverIntents, select_pair_maneuver_intents
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalFleetJointPairTransferError(RuntimeError):
    """Base error for invalid a115 JOINT→PAIR transfers."""


class EnvironmentalFleetJointPairTransferAction(str, Enum):
    PAIR_CYCLE_STARTED = "pair_cycle_started"


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetJointPairTransferConfig:
    pair_cycle: EnvironmentalPairControlCycleConfig
    fleet_guard: EnvironmentalFleetGuardedPairConfig

    def __post_init__(self) -> None:
        if not isinstance(self.pair_cycle, EnvironmentalPairControlCycleConfig):
            raise TypeError("pair_cycle must be EnvironmentalPairControlCycleConfig")
        if not isinstance(self.fleet_guard, EnvironmentalFleetGuardedPairConfig):
            raise TypeError("fleet_guard must be EnvironmentalFleetGuardedPairConfig")


@dataclass(frozen=True, slots=True)
class EnvironmentalJointToPairBootstrap:
    owner: FleetCommandOwner
    route_targets: tuple[EnvironmentalRouteHandoffTarget, EnvironmentalRouteHandoffTarget]
    decision: EncounterLifecycleDecision
    lifecycle: EncounterLifecycle
    planning_context: EnvironmentalAvoidancePlanningContext
    intents: PairManeuverIntents
    command_grid: EnvironmentalPairCommandGrid
    evaluated: EnvironmentalPairGridEvaluationBatch
    progress: EnvironmentalRule13ProgressBatch
    arbitration: EnvironmentalPairArbitrationResult
    plan: EnvironmentalPairExecutionPlan
    applied: AppliedEnvironmentalPairPlan
    pair_cycle: EnvironmentalPairControlSession
    guarded_session: EnvironmentalFleetGuardedPairControlSession


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetJointPairTransferResult:
    source: EnvironmentalFleetJointRecaptureDispatchResult
    action: EnvironmentalFleetJointPairTransferAction
    owner: FleetCommandOwner
    bootstrap: EnvironmentalJointToPairBootstrap

    @property
    def pair_started(self) -> bool:
        return self.action is EnvironmentalFleetJointPairTransferAction.PAIR_CYCLE_STARTED


def _admit_route_targets(
    owner: FleetCommandOwner,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
) -> tuple[EnvironmentalRouteHandoffTarget, EnvironmentalRouteHandoffTarget]:
    """Validate route-control ownership before any a92 command write."""

    if not isinstance(route_targets, Mapping):
        raise TypeError("route_targets must be a mapping")
    ordered: list[EnvironmentalRouteHandoffTarget] = []
    for vessel_id in owner.vessel_ids:
        try:
            target = route_targets[vessel_id]
        except KeyError as exc:
            raise EnvironmentalFleetJointPairTransferError(
                f"missing route target for recaptured pair vessel {vessel_id!r}"
            ) from exc
        if not isinstance(target, EnvironmentalRouteHandoffTarget):
            raise EnvironmentalFleetJointPairTransferError(
                f"route target for {vessel_id!r} must be EnvironmentalRouteHandoffTarget"
            )
        if target.vessel_id != vessel_id:
            raise EnvironmentalFleetJointPairTransferError(
                f"route target mapping key {vessel_id!r} disagrees with target vessel_id {target.vessel_id!r}"
            )
        ordered.append(target)
    if len(ordered) != 2:
        raise EnvironmentalFleetJointPairTransferError(
            "JOINT→PAIR transfer requires exactly two admitted route targets"
        )
    return (ordered[0], ordered[1])


def _pair_decision(owner: FleetCommandOwner) -> EncounterLifecycleDecision:
    if owner.backend is not FleetSupervisorBackend.PAIR or len(owner.vessel_ids) != 2:
        raise EnvironmentalFleetJointPairTransferError(
            "a115 transfer requires the exact isolated PAIR owner selected by a114"
        )
    if len(owner.component.decisions) != 1:
        raise EnvironmentalFleetJointPairTransferError(
            "recaptured PAIR owner must contain exactly one admission decision"
        )
    decision = owner.component.decisions[0].entry.decision
    if not decision.responsibility_active:
        raise EnvironmentalFleetJointPairTransferError(
            "recaptured PAIR owner has no active responsibility"
        )
    return decision


def transfer_environmental_joint_to_pair(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    source: EnvironmentalFleetJointRecaptureDispatchResult,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    *,
    config: EnvironmentalFleetJointPairTransferConfig,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalFleetJointPairTransferResult:
    """Start one fresh a100+a102 PAIR cycle from an a114 JOINT→PAIR boundary."""

    if not isinstance(registry, EncounterRegistry):
        raise TypeError("registry must be EncounterRegistry")
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    if not isinstance(source, EnvironmentalFleetJointRecaptureDispatchResult):
        raise TypeError("source must be EnvironmentalFleetJointRecaptureDispatchResult")
    if not isinstance(config, EnvironmentalFleetJointPairTransferConfig):
        raise TypeError("config must be EnvironmentalFleetJointPairTransferConfig")
    if source.action is not EnvironmentalFleetJointRecaptureDispatchAction.PAIR_BACKEND_REQUIRED:
        raise EnvironmentalFleetJointPairTransferError(
            "a115 requires an a114 PAIR_BACKEND_REQUIRED disposition"
        )
    owner = source.selected_owner
    if owner is None:
        raise EnvironmentalFleetJointPairTransferError("a114 did not provide a selected PAIR owner")

    a113_source = source.source
    recapture = a113_source.recapture
    if recapture is None:
        raise EnvironmentalFleetJointPairTransferError("a114 source has no committed a113 recapture")
    if abs(a113_source.snapshot.time - world.time) > _EPS or a113_source.snapshot != world.snapshot():
        raise EnvironmentalFleetJointPairTransferError(
            "world no longer matches the a113/a114 JOINT→PAIR recapture boundary"
        )
    # A physical snapshot does not include command-channel state.  Preserve the exact
    # a113 transfer boundary by also requiring every command still owned by the old
    # JOINT cycle to match its last applied tuple before PAIR ownership is installed.
    old_applied = a113_source.joint_result.current_applied
    try:
        current_old_commands = tuple(
            (vessel_id, world.command_for(vessel_id))
            for vessel_id in old_applied.plan.vessel_ids
        )
    except KeyError as exc:
        raise EnvironmentalFleetJointPairTransferError(
            "old JOINT controlled vessel disappeared after the a113/a114 boundary"
        ) from exc
    if current_old_commands != old_applied.commands_after:
        raise EnvironmentalFleetJointPairTransferError(
            "world command state no longer matches the a113/a114 JOINT→PAIR recapture boundary"
        )
    observation = recapture.fleet_observation
    if abs(observation.time_min - world.time) > _EPS:
        raise EnvironmentalFleetJointPairTransferError(
            "committed fleet observation time does not match current environmental world time"
        )
    if source.touching_owners != (owner,) or owner not in observation.dispatch.owners:
        raise EnvironmentalFleetJointPairTransferError(
            "a114 selected PAIR owner is not the unique owner in the committed fleet dispatch"
        )

    # Route-target admission is deliberately first among all caller-controlled transfer
    # inputs.  Any missing/mismatched target must fail while the old JOINT commands are
    # still untouched.
    targets = _admit_route_targets(owner, route_targets)
    decision = _pair_decision(owner)

    lifecycle = registry.clone_lifecycle_for(*owner.vessel_ids)
    if not lifecycle.observations or lifecycle.observations[-1] != decision.observation:
        raise EnvironmentalFleetJointPairTransferError(
            "cloned registry lifecycle is not aligned with the committed PAIR boundary"
        )

    context = build_environmental_avoidance_planning_context(
        world,
        observation.dispatch,
        owner,
        profile_catalog=profile_catalog,
    )
    replanning = config.pair_cycle.replanning
    intents = select_pair_maneuver_intents(
        decision,
        stand_on_stage=replanning.stand_on_stage,
    )
    command_grid = generate_environmental_pair_command_grid(
        context,
        intents,
        policy=replanning.candidate_policy,
    )
    evaluated = evaluate_environmental_pair_command_grid(
        context,
        command_grid,
        config=replanning.candidate_config,
    )
    progress = filter_environmental_rule13_progress(
        context,
        evaluated,
        minimum_required_rate_nm_per_min=replanning.minimum_rule13_progress_rate_nm_per_min,
    )
    arbitration = arbitrate_environmental_pair_rows(
        progress,
        config=replanning.arbitration_config,
    )
    plan = build_environmental_pair_execution_plan(
        world,
        context,
        arbitration,
        profile_catalog=profile_catalog,
    )

    # Everything above this point is read-only. A92 is the first PAIR command mutation.
    applied = apply_environmental_pair_execution_plan(
        world,
        plan,
        profile_catalog=profile_catalog,
    )

    holder: dict[str, EnvironmentalFleetGuardedPairControlSession] = {}

    def planning_context_factory(current_world, _policy):
        guard = holder.get("guard")
        if guard is None:
            raise EnvironmentalFleetJointPairTransferError(
                "fresh PAIR planning factory used before a102 guard attachment"
            )
        if not guard.fleet_observations:
            raise EnvironmentalFleetJointPairTransferError(
                "a102 has no canonical fleet observation for PAIR replanning"
            )
        latest = guard.fleet_observations[-1]
        if abs(latest.time_min - current_world.time) > _EPS:
            raise EnvironmentalFleetJointPairTransferError(
                "PAIR replanning lacks a same-timestamp a102 fleet observation"
            )
        matches = tuple(
            item
            for item in latest.dispatch.owners
            if item.backend is FleetSupervisorBackend.PAIR
            and item.component_id == owner.component_id
            and item.vessel_ids == owner.vessel_ids
            and item.pair_keys == owner.pair_keys
        )
        if len(matches) != 1:
            raise EnvironmentalFleetJointPairTransferError(
                "a102 fleet observation no longer contains the exact delegated PAIR owner"
            )
        return build_environmental_avoidance_planning_context(
            current_world,
            latest.dispatch,
            matches[0],
            profile_catalog=profile_catalog,
        )

    pair_cycle = EnvironmentalPairControlSession(
        world,
        applied,
        lifecycle,
        decision,
        targets,
        planning_context_factory,
        config=config.pair_cycle,
        profile_catalog=profile_catalog,
    )
    guarded = EnvironmentalFleetGuardedPairControlSession(
        registry,
        pair_cycle,
        config=config.fleet_guard,
        initial_observation=observation,
    )
    holder["guard"] = guarded
    initial_ownership = guarded.start()
    if guarded.recapture is not None:
        # This should be impossible because a114 selected exactly this PAIR from the
        # identical committed dispatch. Failing here exposes a broken ownership contract.
        raise EnvironmentalFleetJointPairTransferError(
            f"fresh a102 guard rejected the delegated PAIR owner at attachment: {initial_ownership!r}"
        )

    bootstrap = EnvironmentalJointToPairBootstrap(
        owner=owner,
        route_targets=targets,
        decision=decision,
        lifecycle=lifecycle,
        planning_context=context,
        intents=intents,
        command_grid=command_grid,
        evaluated=evaluated,
        progress=progress,
        arbitration=arbitration,
        plan=plan,
        applied=applied,
        pair_cycle=pair_cycle,
        guarded_session=guarded,
    )
    return EnvironmentalFleetJointPairTransferResult(
        source=source,
        action=EnvironmentalFleetJointPairTransferAction.PAIR_CYCLE_STARTED,
        owner=owner,
        bootstrap=bootstrap,
    )
