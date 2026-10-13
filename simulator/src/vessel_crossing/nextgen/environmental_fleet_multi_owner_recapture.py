"""Atomic multi-owner bootstrap from one committed a113/a114 recapture boundary.

A116 closes the partial-write hole left intentionally explicit by a114.  When one old
JOINT owner is recaptured as several disjoint command owners, every replacement owner is
planned first against the *same* committed fleet observation and the same physical
``EnvironmentalWorld2D`` timestamp.  Only after every PAIR/JOINT owner has passed route
admission, lifecycle alignment, planning, arbitration and stale-plan validation is one
combined water-relative command batch published.

The module deliberately does **not** run the replacement owners concurrently after the
bootstrap.  It returns freshly attached a102/a113 guarded sessions which must be driven
by the subsequent shared-clock coordinator.  Running those guarded sessions
independently would reintroduce multiple registry clocks and is outside a116 scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .encounter_registry import EncounterRegistry, PairKey
from .environmental_candidate_generation import (
    EnvironmentalPairCommandGrid,
    generate_environmental_pair_command_grid,
)
from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_fleet_guarded_joint_control import (
    EnvironmentalFleetGuardedJointConfig,
    EnvironmentalFleetGuardedJointControlSession,
)
from .environmental_fleet_guarded_pair_control import (
    EnvironmentalFleetGuardedPairConfig,
    EnvironmentalFleetGuardedPairControlSession,
)
from .environmental_fleet_joint_recapture_dispatch import (
    EnvironmentalFleetJointRecaptureDispatchAction,
    EnvironmentalFleetJointRecaptureDispatchResult,
)
from .environmental_joint_arbiter import EnvironmentalJointArbitrationResult
from .environmental_joint_exact_search import EnvironmentalJointExactSearchResult
from .environmental_joint_control_cycle import (
    EnvironmentalJointControlCycleConfig,
    EnvironmentalJointControlSession,
)
from .environmental_joint_execution import AppliedEnvironmentalJointPlan
from .environmental_joint_plan import (
    EnvironmentalJointExecutionPlan,
    build_environmental_joint_execution_plan,
    validate_environmental_joint_execution_plan,
)
from .environmental_joint_semantics import EnvironmentalJointSemanticGrid
from .environmental_joint_selection import select_environmental_joint
from .environmental_pair_arbiter import (
    EnvironmentalPairArbitrationResult,
    arbitrate_environmental_pair_rows,
)
from .environmental_pair_control_cycle import (
    EnvironmentalPairControlCycleConfig,
    EnvironmentalPairControlSession,
)
from .environmental_pair_execution import AppliedEnvironmentalPairPlan
from .environmental_pair_grid_evaluation import (
    EnvironmentalPairGridEvaluationBatch,
    evaluate_environmental_pair_command_grid,
)
from .environmental_pair_plan import (
    EnvironmentalPairExecutionPlan,
    build_environmental_pair_execution_plan,
    validate_environmental_pair_execution_plan,
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
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .fleet_supervision import FleetCommandOwner, FleetSupervisorBackend
from .lifecycle import EncounterLifecycle, EncounterLifecycleDecision
from .maneuver_intent import PairManeuverIntents, select_pair_maneuver_intents
from .multi_intent import ComponentIntentAggregation, aggregate_component_intents, resolve_component_stand_on_stages
from .vessel_classes import VesselProfileCatalog
from .same_route_following import SameRouteFollowingSupervisor

_EPS = 1e-10


class EnvironmentalFleetMultiOwnerRecaptureError(RuntimeError):
    """Base error for invalid/non-atomic a116 replacement-owner bootstrap."""


class PendingEnvironmentalMultiOwnerCommandEvent(EnvironmentalFleetMultiOwnerRecaptureError):
    """Raised when current-time command-event state is ambiguous before the batch write."""


class EnvironmentalFleetMultiOwnerApplyInvariantViolation(EnvironmentalFleetMultiOwnerRecaptureError):
    """Raised if the single same-time replacement batch violates world invariants."""


class EnvironmentalFleetMultiOwnerRecaptureAction(str, Enum):
    MULTI_OWNER_CYCLES_STARTED = "multi_owner_cycles_started"


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetMultiOwnerRecaptureConfig:
    pair_cycle: EnvironmentalPairControlCycleConfig
    pair_fleet_guard: EnvironmentalFleetGuardedPairConfig
    joint_cycle: EnvironmentalJointControlCycleConfig
    joint_fleet_guard: EnvironmentalFleetGuardedJointConfig
    same_route_following: SameRouteFollowingSupervisor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pair_cycle, EnvironmentalPairControlCycleConfig):
            raise TypeError("pair_cycle must be EnvironmentalPairControlCycleConfig")
        if not isinstance(self.pair_fleet_guard, EnvironmentalFleetGuardedPairConfig):
            raise TypeError("pair_fleet_guard must be EnvironmentalFleetGuardedPairConfig")
        if not isinstance(self.joint_cycle, EnvironmentalJointControlCycleConfig):
            raise TypeError("joint_cycle must be EnvironmentalJointControlCycleConfig")
        if not isinstance(self.joint_fleet_guard, EnvironmentalFleetGuardedJointConfig):
            raise TypeError("joint_fleet_guard must be EnvironmentalFleetGuardedJointConfig")
        if self.same_route_following is not None and not isinstance(
            self.same_route_following, SameRouteFollowingSupervisor
        ):
            raise TypeError("same_route_following must be SameRouteFollowingSupervisor or None")


@dataclass(frozen=True, slots=True)
class EnvironmentalPreparedPairOwner:
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


@dataclass(frozen=True, slots=True)
class EnvironmentalPreparedJointOwner:
    owner: FleetCommandOwner
    decisions: dict[PairKey, EncounterLifecycleDecision]
    lifecycles: dict[PairKey, EncounterLifecycle]
    planning_context: EnvironmentalAvoidancePlanningContext
    aggregation: ComponentIntentAggregation
    semantic_grid: EnvironmentalJointSemanticGrid | None
    exact_search: EnvironmentalJointExactSearchResult | None
    arbitration: EnvironmentalJointArbitrationResult
    plan: EnvironmentalJointExecutionPlan


@dataclass(frozen=True, slots=True)
class EnvironmentalStartedPairOwner:
    prepared: EnvironmentalPreparedPairOwner
    applied: AppliedEnvironmentalPairPlan
    pair_cycle: EnvironmentalPairControlSession
    guarded_session: EnvironmentalFleetGuardedPairControlSession

    @property
    def owner(self) -> FleetCommandOwner:
        return self.prepared.owner


@dataclass(frozen=True, slots=True)
class EnvironmentalStartedJointOwner:
    prepared: EnvironmentalPreparedJointOwner
    applied: AppliedEnvironmentalJointPlan
    joint_cycle: EnvironmentalJointControlSession
    guarded_session: EnvironmentalFleetGuardedJointControlSession

    @property
    def owner(self) -> FleetCommandOwner:
        return self.prepared.owner


EnvironmentalStartedOwner = EnvironmentalStartedPairOwner | EnvironmentalStartedJointOwner


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetMultiOwnerRecaptureResult:
    source: EnvironmentalFleetJointRecaptureDispatchResult
    action: EnvironmentalFleetMultiOwnerRecaptureAction
    snapshot_before: EnvironmentalWorldSnapshot
    snapshot_after: EnvironmentalWorldSnapshot
    commands_before: tuple[tuple[str, WaterRelativeManeuverCommand], ...]
    commands_after: tuple[tuple[str, WaterRelativeManeuverCommand], ...]
    started_owners: tuple[EnvironmentalStartedOwner, ...]

    @property
    def owner_ids(self) -> tuple[str, ...]:
        return tuple(item.owner.component_id for item in self.started_owners)



def _admit_pair_targets(
    owner: FleetCommandOwner,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
) -> tuple[EnvironmentalRouteHandoffTarget, EnvironmentalRouteHandoffTarget]:
    if owner.backend is not FleetSupervisorBackend.PAIR or len(owner.vessel_ids) != 2:
        raise EnvironmentalFleetMultiOwnerRecaptureError("pair target admission requires a PAIR owner")
    ordered: list[EnvironmentalRouteHandoffTarget] = []
    for vessel_id in owner.vessel_ids:
        try:
            target = route_targets[vessel_id]
        except KeyError as exc:
            raise EnvironmentalFleetMultiOwnerRecaptureError(
                f"missing route target for replacement PAIR vessel {vessel_id!r}"
            ) from exc
        if not isinstance(target, EnvironmentalRouteHandoffTarget):
            raise EnvironmentalFleetMultiOwnerRecaptureError(
                f"route target for {vessel_id!r} must be EnvironmentalRouteHandoffTarget"
            )
        if target.vessel_id != vessel_id:
            raise EnvironmentalFleetMultiOwnerRecaptureError(
                f"route target mapping key {vessel_id!r} disagrees with target vessel_id {target.vessel_id!r}"
            )
        ordered.append(target)
    return (ordered[0], ordered[1])



def _pair_decision(owner: FleetCommandOwner) -> EncounterLifecycleDecision:
    if owner.backend is not FleetSupervisorBackend.PAIR or len(owner.vessel_ids) != 2:
        raise EnvironmentalFleetMultiOwnerRecaptureError("replacement PAIR owner is malformed")
    if len(owner.component.decisions) != 1:
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "replacement PAIR owner must contain exactly one admission decision"
        )
    decision = owner.component.decisions[0].entry.decision
    if not decision.responsibility_active:
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "replacement PAIR owner has no active responsibility"
        )
    return decision



def _joint_decisions(owner: FleetCommandOwner) -> dict[PairKey, EncounterLifecycleDecision]:
    if owner.backend is not FleetSupervisorBackend.JOINT or len(owner.vessel_ids) < 3:
        raise EnvironmentalFleetMultiOwnerRecaptureError("replacement JOINT owner is malformed")
    decisions = {item.key: item.entry.decision for item in owner.component.decisions}
    if tuple(decisions) != owner.pair_keys:
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "replacement JOINT admission decisions do not match owner pair order"
        )
    if not all(item.responsibility_active for item in decisions.values()):
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "replacement JOINT owner contains an inactive responsibility decision"
        )
    return decisions



def _prepare_pair_owner(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    owner: FleetCommandOwner,
    targets: tuple[EnvironmentalRouteHandoffTarget, EnvironmentalRouteHandoffTarget],
    all_route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    dispatch,
    *,
    config: EnvironmentalFleetMultiOwnerRecaptureConfig,
    profile_catalog: VesselProfileCatalog | None,
) -> EnvironmentalPreparedPairOwner:
    decision = _pair_decision(owner)
    lifecycle = registry.clone_lifecycle_for(*owner.vessel_ids)
    if not lifecycle.observations or lifecycle.observations[-1] != decision.observation:
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            f"cloned lifecycle for PAIR owner {owner.component_id!r} is not aligned with committed boundary"
        )
    context = build_environmental_avoidance_planning_context(
        world,
        dispatch,
        owner,
        profile_catalog=profile_catalog,
        route_targets=all_route_targets,
        same_route_following=config.same_route_following,
    )
    replanning = config.pair_cycle.replanning
    intents = select_pair_maneuver_intents(
        decision,
        stand_on_stage=replanning.stand_on_stage,
        rule13_liveness_policy=replanning.rule13_liveness_policy,
    )
    grid = generate_environmental_pair_command_grid(
        context, intents, policy=replanning.candidate_policy
    )
    evaluated = evaluate_environmental_pair_command_grid(
        context, grid, config=replanning.candidate_config
    )
    progress = filter_environmental_rule13_progress(
        context,
        evaluated,
        minimum_required_rate_nm_per_min=replanning.minimum_rule13_progress_rate_nm_per_min,
    )
    arbitration = arbitrate_environmental_pair_rows(
        progress, config=replanning.arbitration_config
    )
    plan = build_environmental_pair_execution_plan(
        world, context, arbitration, profile_catalog=profile_catalog
    )
    return EnvironmentalPreparedPairOwner(
        owner,
        targets,
        decision,
        lifecycle,
        context,
        intents,
        grid,
        evaluated,
        progress,
        arbitration,
        plan,
    )



def _prepare_joint_owner(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    owner: FleetCommandOwner,
    all_route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    dispatch,
    *,
    config: EnvironmentalFleetMultiOwnerRecaptureConfig,
    profile_catalog: VesselProfileCatalog | None,
) -> EnvironmentalPreparedJointOwner:
    decisions = _joint_decisions(owner)
    lifecycles: dict[PairKey, EncounterLifecycle] = {}
    for key in owner.pair_keys:
        lifecycle = registry.clone_lifecycle_for(*key)
        if not lifecycle.observations or lifecycle.observations[-1] != decisions[key].observation:
            raise EnvironmentalFleetMultiOwnerRecaptureError(
                f"cloned lifecycle for JOINT pair {key!r} is not aligned with committed boundary"
            )
        lifecycles[key] = lifecycle
    context = build_environmental_avoidance_planning_context(
        world,
        dispatch,
        owner,
        profile_catalog=profile_catalog,
        route_targets=all_route_targets,
        same_route_following=config.same_route_following,
    )
    replanning = config.joint_cycle.replanning
    effective_stages = resolve_component_stand_on_stages(
        owner.component,
        stand_on_stages=replanning.stand_on_stages,
        policy=replanning.multi_intent_resolution,
        rule13_liveness_policy=replanning.rule13_liveness_policy,
    )
    aggregation = aggregate_component_intents(
        owner.component,
        stand_on_stages=effective_stages,
        rule13_liveness_policy=replanning.rule13_liveness_policy,
    )
    selection = select_environmental_joint(
        context,
        aggregation,
        semantic_config=replanning.semantic_config,
        arbitration_config=replanning.arbitration_config,
        exact_search_config=replanning.exact_search_config,
    )
    semantic_grid = selection.semantic_grid
    exact_search = selection.exact_search
    arbitration = selection.arbitration
    plan = build_environmental_joint_execution_plan(
        world, context, arbitration, profile_catalog=profile_catalog
    )
    return EnvironmentalPreparedJointOwner(
        owner,
        decisions,
        lifecycles,
        context,
        aggregation,
        semantic_grid,
        exact_search,
        arbitration,
        plan,
    )



def _pair_commands(plan: EnvironmentalPairExecutionPlan) -> tuple[tuple[str, WaterRelativeManeuverCommand], ...]:
    return ((plan.first_id, plan.first_command), (plan.second_id, plan.second_command))



def _validate_boundary(
    world: EnvironmentalWorld2D,
    source: EnvironmentalFleetJointRecaptureDispatchResult,
) -> None:
    if source.action is not EnvironmentalFleetJointRecaptureDispatchAction.MULTI_OWNER_COORDINATION_REQUIRED:
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "a116 requires an a114 MULTI_OWNER_COORDINATION_REQUIRED disposition"
        )
    if len(source.touching_owners) < 2:
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "a116 requires at least two replacement command owners"
        )
    a113_source = source.source
    recapture = a113_source.recapture
    if recapture is None:
        raise EnvironmentalFleetMultiOwnerRecaptureError("a114 source has no committed a113 recapture")
    if abs(a113_source.snapshot.time - world.time) > _EPS or a113_source.snapshot != world.snapshot():
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "world no longer matches the a113/a114 multi-owner recapture boundary"
        )
    if abs(recapture.fleet_observation.time_min - world.time) > _EPS:
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "committed fleet observation time does not match current environmental world time"
        )

    old_applied = a113_source.joint_result.current_applied
    try:
        current_old_commands = tuple(
            (vessel_id, world.command_for(vessel_id)) for vessel_id in old_applied.plan.vessel_ids
        )
    except KeyError as exc:
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "old JOINT controlled vessel disappeared after the a113/a114 boundary"
        ) from exc
    if current_old_commands != old_applied.commands_after:
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "world command state no longer matches the a113/a114 multi-owner recapture boundary"
        )

    dispatch_owners = recapture.fleet_observation.dispatch.owners
    if any(owner not in dispatch_owners for owner in source.touching_owners):
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "a114 replacement owner is absent from the committed fleet dispatch"
        )
    old_ids = set(old_applied.plan.vessel_ids)
    if any(not old_ids.intersection(owner.vessel_ids) for owner in source.touching_owners):
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "a114 replacement owner does not touch the recaptured JOINT component"
        )
    claimed: set[str] = set()
    for owner in source.touching_owners:
        overlap = claimed.intersection(owner.vessel_ids)
        if overlap:
            raise EnvironmentalFleetMultiOwnerRecaptureError(
                f"replacement owners overlap command channels: {sorted(overlap)!r}"
            )
        claimed.update(owner.vessel_ids)



def bootstrap_environmental_multi_owner_recapture(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    source: EnvironmentalFleetJointRecaptureDispatchResult,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    *,
    config: EnvironmentalFleetMultiOwnerRecaptureConfig,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalFleetMultiOwnerRecaptureResult:
    """Preflight every replacement owner, then publish one atomic same-time batch."""

    if not isinstance(registry, EncounterRegistry):
        raise TypeError("registry must be EncounterRegistry")
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    if not isinstance(source, EnvironmentalFleetJointRecaptureDispatchResult):
        raise TypeError("source must be EnvironmentalFleetJointRecaptureDispatchResult")
    if not isinstance(route_targets, Mapping):
        raise TypeError("route_targets must be a mapping")
    if not isinstance(config, EnvironmentalFleetMultiOwnerRecaptureConfig):
        raise TypeError("config must be EnvironmentalFleetMultiOwnerRecaptureConfig")

    _validate_boundary(world, source)
    observation = source.source.recapture.fleet_observation  # type: ignore[union-attr]
    dispatch = observation.dispatch

    # Admit every PAIR route target before any planning/apply path.  A failure for the
    # final replacement owner must leave the first owner's old JOINT commands untouched.
    admitted_targets: dict[str, tuple[EnvironmentalRouteHandoffTarget, EnvironmentalRouteHandoffTarget]] = {}
    for owner in source.touching_owners:
        if owner.backend is FleetSupervisorBackend.PAIR:
            admitted_targets[owner.component_id] = _admit_pair_targets(owner, route_targets)

    prepared: list[EnvironmentalPreparedPairOwner | EnvironmentalPreparedJointOwner] = []
    for owner in source.touching_owners:
        if owner.backend is FleetSupervisorBackend.PAIR:
            prepared.append(
                _prepare_pair_owner(
                    registry,
                    world,
                    owner,
                    admitted_targets[owner.component_id],
                    route_targets,
                    dispatch,
                    config=config,
                    profile_catalog=profile_catalog,
                )
            )
        elif owner.backend is FleetSupervisorBackend.JOINT:
            prepared.append(
                _prepare_joint_owner(
                    registry,
                    world,
                    owner,
                    route_targets,
                    dispatch,
                    config=config,
                    profile_catalog=profile_catalog,
                )
            )
        else:  # pragma: no cover - FleetSupervisorBackend is closed
            raise AssertionError(owner.backend)

    # Revalidate every plan after the complete read-only preflight, then form one unique
    # fleet command transaction.  No owner is allowed to publish a prefix of the batch.
    command_batch: list[tuple[str, WaterRelativeManeuverCommand]] = []
    for item in prepared:
        if isinstance(item, EnvironmentalPreparedPairOwner):
            validate_environmental_pair_execution_plan(
                item.plan, world, profile_catalog=profile_catalog
            )
            commands = _pair_commands(item.plan)
        else:
            validate_environmental_joint_execution_plan(
                item.plan, world, profile_catalog=profile_catalog
            )
            commands = item.plan.selected_commands
        command_batch.extend(commands)

    ids = [vessel_id for vessel_id, _ in command_batch]
    if len(ids) != len(set(ids)):
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "replacement command batch contains duplicate vessel channels"
        )
    expected_ids = [vessel_id for owner in source.touching_owners for vessel_id in owner.vessel_ids]
    if set(ids) != set(expected_ids):
        raise EnvironmentalFleetMultiOwnerRecaptureError(
            "replacement command batch does not cover every delegated owner vessel exactly once"
        )

    pending = world.next_command_time()
    if pending is not None and pending <= world.time + _EPS:
        raise PendingEnvironmentalMultiOwnerCommandEvent(
            "cannot apply multi-owner replacement while a command event is pending at current time"
        )

    snapshot_before = world.snapshot()
    command_order = tuple(ids)
    commands_before = tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in command_order)
    applied_at = world.time
    world.schedule_commands_atomic(at=applied_at, commands=tuple(command_batch))
    world.run_until(applied_at)
    snapshot_after = world.snapshot()
    commands_after = tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in command_order)
    if snapshot_after != snapshot_before or world.time != applied_at:
        raise EnvironmentalFleetMultiOwnerApplyInvariantViolation(
            "same-timestamp multi-owner replacement changed physical state or world time"
        )
    if commands_after != tuple(command_batch):
        raise EnvironmentalFleetMultiOwnerApplyInvariantViolation(
            "same-timestamp multi-owner replacement did not install the complete selected batch"
        )

    before_map = dict(commands_before)
    after_map = dict(commands_after)
    started: list[EnvironmentalStartedOwner] = []

    for item in prepared:
        owner = item.owner
        if isinstance(item, EnvironmentalPreparedPairOwner):
            applied = AppliedEnvironmentalPairPlan(
                plan=item.plan,
                applied_at_time=applied_at,
                snapshot_before=snapshot_before,
                snapshot_after=snapshot_after,
                commands_before=(before_map[item.plan.first_id], before_map[item.plan.second_id]),
                commands_after=(after_map[item.plan.first_id], after_map[item.plan.second_id]),
            )
            holder: dict[str, EnvironmentalFleetGuardedPairControlSession] = {}

            def pair_factory(current_world, _policy, *, _owner=owner, _holder=holder):
                guard = _holder.get("guard")
                if guard is None or not guard.fleet_observations:
                    raise EnvironmentalFleetMultiOwnerRecaptureError(
                        "PAIR planning factory used without a current a102 fleet observation"
                    )
                latest = guard.fleet_observations[-1]
                if abs(latest.time_min - current_world.time) > _EPS:
                    raise EnvironmentalFleetMultiOwnerRecaptureError(
                        "PAIR replanning lacks same-timestamp fleet ownership evidence"
                    )
                matches = tuple(
                    candidate
                    for candidate in latest.dispatch.owners
                    if candidate.backend is FleetSupervisorBackend.PAIR
                    and candidate.component_id == _owner.component_id
                    and candidate.vessel_ids == _owner.vessel_ids
                    and candidate.pair_keys == _owner.pair_keys
                )
                if len(matches) != 1:
                    raise EnvironmentalFleetMultiOwnerRecaptureError(
                        "current fleet observation no longer contains the exact delegated PAIR owner"
                    )
                return build_environmental_avoidance_planning_context(
                    current_world,
                    latest.dispatch,
                    matches[0],
                    profile_catalog=profile_catalog,
                    route_targets=route_targets,
                    same_route_following=config.same_route_following,
                )

            pair_cycle = EnvironmentalPairControlSession(
                world,
                applied,
                item.lifecycle,
                item.decision,
                item.route_targets,
                pair_factory,
                config=config.pair_cycle,
                profile_catalog=profile_catalog,
            )
            guarded = EnvironmentalFleetGuardedPairControlSession(
                registry,
                pair_cycle,
                config=config.pair_fleet_guard,
                initial_observation=observation,
            )
            holder["guard"] = guarded
            initial = guarded.start()
            if guarded.recapture is not None:
                raise EnvironmentalFleetMultiOwnerRecaptureError(
                    f"fresh a102 guard rejected delegated PAIR owner {owner.component_id!r}: {initial!r}"
                )
            started.append(EnvironmentalStartedPairOwner(item, applied, pair_cycle, guarded))
            continue

        applied = AppliedEnvironmentalJointPlan(
            plan=item.plan,
            applied_at_time=applied_at,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
            commands_before=tuple((vessel_id, before_map[vessel_id]) for vessel_id in item.plan.vessel_ids),
            commands_after=tuple((vessel_id, after_map[vessel_id]) for vessel_id in item.plan.vessel_ids),
        )
        holder_j: dict[str, EnvironmentalFleetGuardedJointControlSession] = {}

        def joint_factory(current_world, _policy, *, _owner=owner, _holder=holder_j):
            guard = _holder.get("guard")
            if guard is None or not guard.fleet_observations:
                raise EnvironmentalFleetMultiOwnerRecaptureError(
                    "JOINT planning factory used without a current a113 fleet observation"
                )
            latest = guard.fleet_observations[-1]
            if abs(latest.time_min - current_world.time) > _EPS:
                raise EnvironmentalFleetMultiOwnerRecaptureError(
                    "JOINT replanning lacks same-timestamp fleet ownership evidence"
                )
            matches = tuple(
                candidate
                for candidate in latest.dispatch.owners
                if candidate.backend is FleetSupervisorBackend.JOINT
                and candidate.component_id == _owner.component_id
                and candidate.vessel_ids == _owner.vessel_ids
                and candidate.pair_keys == _owner.pair_keys
            )
            if len(matches) != 1:
                raise EnvironmentalFleetMultiOwnerRecaptureError(
                    "current fleet observation no longer contains the exact delegated JOINT owner"
                )
            return build_environmental_avoidance_planning_context(
                current_world,
                latest.dispatch,
                matches[0],
                profile_catalog=profile_catalog,
            )

        joint_cycle = EnvironmentalJointControlSession(
            world,
            applied,
            item.lifecycles,
            item.decisions,
            joint_factory,
            config=config.joint_cycle,
            profile_catalog=profile_catalog,
        )
        guarded_j = EnvironmentalFleetGuardedJointControlSession(
            registry,
            joint_cycle,
            config=config.joint_fleet_guard,
            initial_observation=observation,
        )
        holder_j["guard"] = guarded_j
        initial_j = guarded_j.start()
        if guarded_j.recapture is not None:
            raise EnvironmentalFleetMultiOwnerRecaptureError(
                f"fresh a113 guard rejected delegated JOINT owner {owner.component_id!r}: {initial_j!r}"
            )
        started.append(EnvironmentalStartedJointOwner(item, applied, joint_cycle, guarded_j))

    return EnvironmentalFleetMultiOwnerRecaptureResult(
        source=source,
        action=EnvironmentalFleetMultiOwnerRecaptureAction.MULTI_OWNER_CYCLES_STARTED,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
        commands_before=commands_before,
        commands_after=commands_after,
        started_owners=tuple(started),
    )
