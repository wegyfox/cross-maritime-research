"""Opt-in arm-local FIFO source integration for the persistent environmental runtime.

A151 is the first executable composition of the A148 source reservoir, A149 continuous
recheck proof and A150 source-event scheduler.  It deliberately does *not* modify the
historical A137/A141 scenario function.  A caller must choose this separately named
cold-start path and supply an immutable A148 reservoir.

The ordering invariant is the important part:

``physical advance / native mutation -> native membership -> FIFO source membership
-> one canonical fleet observation -> avoidance/bootstrap/clear-route ownership``.

A source-only wake that remains blocked produces no encounter-registry observation.  A
source event exactly on a native runtime boundary is suppressed by A150; after the native
mutation, the FIFO phase re-evaluates the head *before* that boundary's single canonical
observation.  No epsilon time shift and no fixed-step source polling are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol

from ..kinematics import Vector2
from .dynamics import ManeuverLimits
from .encounter_classification import EncounterClassificationConfig
from .encounter_registry import EncounterRegistry, PairKey
from .environmental_encounter_classification import EnvironmentalEncounterDirectionBasis
from .environmental_fleet_ingress import EnvironmentalFleetIngressAdmission, EnvironmentalFleetIngressRecord
from .environmental_fleet_multi_owner_recapture import EnvironmentalFleetMultiOwnerRecaptureConfig
from .environmental_fleet_runtime import (
    EnvironmentalFleetRuntimeConfig,
    EnvironmentalFleetRuntimeStartResult,
    start_environmental_fleet_runtime,
)
from .environmental_candidate_generation import EnvironmentalCandidateGenerationUnavailable
from .environmental_joint_arbiter import NoEnvironmentalJointTuple
from .environmental_joint_control_cycle import EnvironmentalJointControlReplanLimitExceeded
from .environmental_joint_physics import EnvironmentalJointPhysicsUnavailable
from .environmental_joint_exact_search import EnvironmentalJointExactSearchUnavailable
from .environmental_joint_replanning import EnvironmentalJointReplanningUnavailable
from .environmental_pair_arbiter import NoEnvironmentalPairCandidate
from .environmental_pair_control_cycle import EnvironmentalPairControlReplanLimitExceeded
from .environmental_pair_clearance import EnvironmentalPairTransitionUnavailable
from .environmental_pair_replanning import EnvironmentalPairReplanningUnavailable
from .environmental_idle_fleet_control import (
    EnvironmentalPersistentFleetControlResult,
    EnvironmentalPersistentFleetControlSession,
)
from .environmental_network_source_event_scheduler import (
    EnvironmentalNetworkSourceEventPlan,
    EnvironmentalNetworkSourceEventProposal,
    environmental_network_source_event_is_current,
    next_environmental_network_source_event_generation,
    plan_environmental_network_source_event,
)
from .environmental_network_source_recheck import (
    EnvironmentalNetworkSourceActiveMotion,
    EnvironmentalNetworkSourceRecheckConfig,
)
from .environmental_fleet_ownership import EnvironmentalFleetOwnershipObservation
from .environmental_registry import augment_environmental_registry_same_time_arrivals
from .fleet_supervision import build_fleet_supervision_dispatch
from .environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceAdmission,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalNetworkSourceResolution,
    EnvironmentalNetworkSourceFutureIngressConfig,
    resolve_environmental_network_source_reservoir,
)
from .environmental_route_handoff import EnvironmentalGroundVelocityUnavailable, EnvironmentalRouteHandoffTarget
from .environmental_world import EnvironmentalWorld2D
from .responsibility import ResponsibilityContext
from .vessel_classes import VesselClassProfile, VesselProfileCatalog
from .same_route_following import SameRouteFollowingSupervisor

_EPS = 1e-10


class EnvironmentalNetworkSourceFIFORuntimeError(RuntimeError):
    """Raised when A151 source/runtime evidence cannot be composed safely."""


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceControllerContext:
    """Read-only physical context offered to an opt-in source-management adapter.

    The adapter may change only source eligibility/order evidence.  Physics, vessel
    identity, commands, profiles, route targets and already committed admissions remain
    under the canonical runtime.
    """

    snapshot: object
    active_profiles: Mapping[str, VesselClassProfile]
    active_route_targets: Mapping[str, EnvironmentalRouteHandoffTarget]
    active_limits: Mapping[str, ManeuverLimits]
    active_environmental_drifts: Mapping[str, Vector2]
    active_commands: Mapping[str, object]
    active_motion_backend: object | None
    active_motion_states: Mapping[str, object | None]
    same_route_dynamic_ingress: SameRouteFollowingSupervisor | None
    future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None


class EnvironmentalNetworkSourceAdmissionController(Protocol):
    """Opt-in management adapter above the canonical physical source gate."""

    def prepare_reservoir(
        self,
        reservoir: EnvironmentalNetworkSourceReservoir,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> EnvironmentalNetworkSourceReservoir: ...

    def observe_resolution(
        self,
        resolution: EnvironmentalNetworkSourceResolution,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> None: ...


def _controller_context(
    *,
    world: EnvironmentalWorld2D,
    profile_catalog: VesselProfileCatalog | None,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    same_route_dynamic_ingress: SameRouteFollowingSupervisor | None,
    future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None,
) -> EnvironmentalNetworkSourceControllerContext:
    snapshot = world.snapshot()
    active_profiles: dict[str, VesselClassProfile] = {}
    for state in snapshot.vessels:
        if profile_catalog is None:
            raise EnvironmentalNetworkSourceFIFORuntimeError(
                "managed FIFO source requires profiles for every active vessel"
            )
        profile = profile_catalog.profile_for(state.vessel_id)
        if profile is None:
            raise EnvironmentalNetworkSourceFIFORuntimeError(
                f"managed FIFO source lacks active profile for {state.vessel_id!r}"
            )
        active_profiles[state.vessel_id] = profile
    return EnvironmentalNetworkSourceControllerContext(
        snapshot=snapshot,
        active_profiles=MappingProxyType(active_profiles),
        active_route_targets=MappingProxyType(
            {
                vessel_id: route_targets[vessel_id]
                for vessel_id in active_profiles
                if vessel_id in route_targets
            }
        ),
        active_limits=MappingProxyType(
            {state.vessel_id: world.limits_for(state.vessel_id) for state in snapshot.vessels}
        ),
        active_environmental_drifts=MappingProxyType(
            {
                state.vessel_id: tuple(world.drift_for(state.vessel_id))
                for state in snapshot.vessels
            }
        ),
        active_commands=MappingProxyType(
            {state.vessel_id: world.command_for(state.vessel_id) for state in snapshot.vessels}
        ),
        active_motion_backend=world.motion_backend,
        active_motion_states=MappingProxyType(
            {
                state.vessel_id: world.motion_state_for(state.vessel_id)
                for state in snapshot.vessels
            }
            if world.motion_backend is not None
            else {}
        ),
        same_route_dynamic_ingress=same_route_dynamic_ingress,
        future_ingress_clearance=future_ingress_clearance,
    )


def _apply_source_admission_controller(
    controller: EnvironmentalNetworkSourceAdmissionController | None,
    reservoir: EnvironmentalNetworkSourceReservoir,
    context: EnvironmentalNetworkSourceControllerContext,
) -> EnvironmentalNetworkSourceReservoir:
    if controller is None:
        return reservoir
    prepared = controller.prepare_reservoir(reservoir, context)
    if not isinstance(prepared, EnvironmentalNetworkSourceReservoir):
        raise EnvironmentalNetworkSourceFIFORuntimeError(
            "source admission controller must return EnvironmentalNetworkSourceReservoir"
        )
    if prepared.admissions != reservoir.admissions:
        raise EnvironmentalNetworkSourceFIFORuntimeError(
            "source admission controller cannot rewrite committed admission history"
        )
    if len(prepared.demands) != len(reservoir.demands):
        raise EnvironmentalNetworkSourceFIFORuntimeError(
            "source admission controller cannot add or remove exogenous demands"
        )
    for original, candidate in zip(reservoir.demands, prepared.demands, strict=True):
        if replace(candidate, release_not_before_time=original.release_not_before_time) != original:
            raise EnvironmentalNetworkSourceFIFORuntimeError(
                "source admission controller may change only not-before/service-order evidence"
            )
    return prepared


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceFIFOBoundary:
    """Audit evidence for one consumed source/native invalidation boundary."""

    time_min: float
    generation_before: int
    generation_after: int
    native_boundary_due: bool
    source_boundary_due: bool
    source_event: EnvironmentalNetworkSourceEventProposal | None
    resolution: EnvironmentalNetworkSourceResolution
    ingress_admissions: tuple[EnvironmentalFleetIngressAdmission, ...]

    @property
    def admitted_vessel_ids(self) -> tuple[str, ...]:
        return tuple(item.vessel_id for item in self.ingress_admissions)


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceFIFORuntimeResult:
    """One caller-horizon result plus the current immutable source state."""

    runtime: EnvironmentalPersistentFleetControlResult
    reservoir: EnvironmentalNetworkSourceReservoir
    source_generation: int
    source_boundaries: tuple[EnvironmentalNetworkSourceFIFOBoundary, ...]

    @property
    def snapshot(self):
        return self.runtime.snapshot


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceFIFORuntimeStartResult:
    """Cold-start evidence before the caller begins advancing the shared clock."""

    session: "EnvironmentalNetworkSourceFIFORuntimeSession"
    runtime_start: EnvironmentalFleetRuntimeStartResult
    initial_resolution: EnvironmentalNetworkSourceResolution
    initial_ingress_admissions: tuple[EnvironmentalFleetIngressAdmission, ...]
    initial_clear_route_owner_ids: tuple[str, ...]


class EnvironmentalNetworkSourceFIFOTerminalStatus(str, Enum):
    SUCCESS = "SUCCESS"
    MODELED_INFEASIBLE = "MODELED_INFEASIBLE"
    RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"


class EnvironmentalNetworkSourceFIFOResourceKind(str, Enum):
    JOINT_CARTESIAN_COMBINATIONS = "JOINT_CARTESIAN_COMBINATIONS"
    JOINT_EXACT_PAIR_EVIDENCE = "JOINT_EXACT_PAIR_EVIDENCE"
    JOINT_EXACT_BRANCH_NODES = "JOINT_EXACT_BRANCH_NODES"
    PAIR_REPLANS = "PAIR_REPLANS"
    JOINT_REPLANS = "JOINT_REPLANS"


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceFIFOTerminalOutcome:
    """Opt-in terminal capture with model and execution-resource outcomes separated.

    This does not change ``run_until`` semantics. A declared physical/model dead-end is
    ``MODELED_INFEASIBLE``; explicit Cartesian-budget exhaustion is
    ``RESOURCE_UNAVAILABLE`` and must not be interpreted as physical infeasibility.
    Unexpected programming/configuration failures remain exceptions.
    """

    status: EnvironmentalNetworkSourceFIFOTerminalStatus
    result: EnvironmentalNetworkSourceFIFORuntimeResult | None
    terminal_time_min: float
    snapshot: object
    reservoir: EnvironmentalNetworkSourceReservoir
    source_generation: int
    source_boundaries: tuple[EnvironmentalNetworkSourceFIFOBoundary, ...]
    ingress_admissions: tuple[EnvironmentalFleetIngressAdmission, ...]
    infeasibility_type: str | None = None
    infeasibility_message: str | None = None
    resource_unavailability_type: str | None = None
    resource_unavailability_message: str | None = None
    resource_kind: EnvironmentalNetworkSourceFIFOResourceKind | None = None
    resource_required_count: int | None = None
    resource_limit_count: int | None = None
    resource_option_counts: tuple[tuple[str, int], ...] = ()

    @property
    def modeled_infeasible(self) -> bool:
        return self.status is EnvironmentalNetworkSourceFIFOTerminalStatus.MODELED_INFEASIBLE

    @property
    def resource_unavailable(self) -> bool:
        return self.status is EnvironmentalNetworkSourceFIFOTerminalStatus.RESOURCE_UNAVAILABLE


_MODELED_FIFO_INFEASIBILITY_TYPES = (
    NoEnvironmentalPairCandidate,
    NoEnvironmentalJointTuple,
    EnvironmentalCandidateGenerationUnavailable,
    EnvironmentalPairTransitionUnavailable,
    EnvironmentalPairReplanningUnavailable,
    EnvironmentalJointReplanningUnavailable,
    EnvironmentalGroundVelocityUnavailable,
)


def _fifo_modeled_infeasibility(exc: BaseException) -> BaseException | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _MODELED_FIFO_INFEASIBILITY_TYPES):
            return current
        current = current.__cause__ or current.__context__
    return None


def _fifo_resource_unavailability(exc: BaseException) -> BaseException | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                EnvironmentalJointPhysicsUnavailable,
                EnvironmentalJointExactSearchUnavailable,
                EnvironmentalPairControlReplanLimitExceeded,
                EnvironmentalJointControlReplanLimitExceeded,
            ),
        ):
            return current
        current = current.__cause__ or current.__context__
    return None


class _SourceRuntimeExtension:
    """Package-private bridge consumed by the opt-in A119/A123 extension hooks."""

    def __init__(
        self,
        session: EnvironmentalPersistentFleetControlSession,
        reservoir: EnvironmentalNetworkSourceReservoir,
        *,
        recheck_config: EnvironmentalNetworkSourceRecheckConfig,
        same_route_dynamic_ingress: SameRouteFollowingSupervisor | None = None,
        future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None = None,
        source_admission_controller: EnvironmentalNetworkSourceAdmissionController | None = None,
    ) -> None:
        self.session = session
        self.world = session.world
        self.registry = session.registry
        self.reservoir = reservoir
        self.recheck_config = recheck_config
        if same_route_dynamic_ingress is not None and not isinstance(
            same_route_dynamic_ingress, SameRouteFollowingSupervisor
        ):
            raise TypeError("same_route_dynamic_ingress must be SameRouteFollowingSupervisor or None")
        self.same_route_dynamic_ingress = same_route_dynamic_ingress
        if future_ingress_clearance is not None and not isinstance(
            future_ingress_clearance, EnvironmentalNetworkSourceFutureIngressConfig
        ):
            raise TypeError(
                "future_ingress_clearance must be EnvironmentalNetworkSourceFutureIngressConfig or None"
            )
        self.future_ingress_clearance = future_ingress_clearance
        self.source_admission_controller = source_admission_controller
        self.generation = 0
        self._pending_plan: EnvironmentalNetworkSourceEventPlan | None = None
        self._plans: list[EnvironmentalNetworkSourceEventPlan] = []
        self._boundaries: list[EnvironmentalNetworkSourceFIFOBoundary] = []

    @property
    def plans(self) -> tuple[EnvironmentalNetworkSourceEventPlan, ...]:
        return tuple(self._plans)

    @property
    def boundaries(self) -> tuple[EnvironmentalNetworkSourceFIFOBoundary, ...]:
        return tuple(self._boundaries)

    def _active_profiles(self) -> dict[str, VesselClassProfile]:
        catalog = self.session.profile_catalog
        if catalog is None:
            return {}
        profiles: dict[str, VesselClassProfile] = {}
        for state in self.world.snapshot().vessels:
            profile = catalog.profile_for(state.vessel_id)
            if profile is not None:
                profiles[state.vessel_id] = profile
        return profiles

    def _active_motions(self) -> tuple[EnvironmentalNetworkSourceActiveMotion, ...]:
        snapshot = self.world.snapshot()
        catalog = self.session.profile_catalog
        if catalog is None and snapshot.vessels:
            raise EnvironmentalNetworkSourceFIFORuntimeError(
                "FIFO source recheck requires profiles for every active vessel"
            )
        motions: list[EnvironmentalNetworkSourceActiveMotion] = []
        for state in snapshot.vessels:
            assert catalog is not None
            profile = catalog.profile_for(state.vessel_id)
            if profile is None:
                raise EnvironmentalNetworkSourceFIFORuntimeError(
                    f"FIFO source recheck lacks active profile for {state.vessel_id!r}"
                )
            motions.append(
                EnvironmentalNetworkSourceActiveMotion(
                    state=state,
                    command=self.world.command_for(state.vessel_id),
                    limits=self.world.limits_for(state.vessel_id),
                    environmental_drift_velocity=tuple(self.world.drift_for(state.vessel_id)),
                    profile=profile,
                    route_target=self.session.route_targets.get(state.vessel_id),
                    motion_backend=self.world.motion_backend,
                    motion_state=self.world.motion_state_for(state.vessel_id),
                )
            )
        return tuple(motions)

    def _controller_context(self) -> EnvironmentalNetworkSourceControllerContext:
        return _controller_context(
            world=self.world,
            profile_catalog=self.session.profile_catalog,
            route_targets=self.session.route_targets,
            same_route_dynamic_ingress=self.same_route_dynamic_ingress,
            future_ingress_clearance=self.future_ingress_clearance,
        )

    def _prepare_managed_reservoir(self) -> EnvironmentalNetworkSourceControllerContext | None:
        if self.source_admission_controller is None:
            return None
        context = self._controller_context()
        self.reservoir = _apply_source_admission_controller(
            self.source_admission_controller,
            self.reservoir,
            context,
        )
        return context

    def next_boundary_time(self, native_frozen_until: float) -> float | None:
        if not isfinite(native_frozen_until) or native_frozen_until < self.world.time - _EPS:
            raise EnvironmentalNetworkSourceFIFORuntimeError(
                "native frozen horizon must be finite and not before the shared clock"
            )
        self._prepare_managed_reservoir()
        if not self.reservoir.pending_demands:
            self._pending_plan = None
            return None

        plan = plan_environmental_network_source_event(
            self.reservoir,
            self.world.snapshot(),
            self._active_motions(),
            frozen_until=float(native_frozen_until),
            generation=self.generation,
            recheck_config=self.recheck_config,
            same_route_dynamic_ingress=self.same_route_dynamic_ingress,
            future_ingress_clearance=self.future_ingress_clearance,
        )
        self._pending_plan = plan
        self._plans.append(plan)
        return None if plan.next_event is None else plan.next_event.at

    def _supersede_same_time_observation_after_ingress(
        self,
        vessel_ids: tuple[str, ...],
    ) -> None:
        """Replace the current canonical observation after exact same-time arrivals.

        Existing registry pair lifecycles are not advanced twice; the registry adapter
        only initializes pairs touching the newly added vessel ids.  The canonical fleet
        evidence object is then superseded in-place so downstream ownership logic sees
        final membership for this physical timestamp without duplicate history times.
        """
        coordinator = self.session.coordinator
        observations = coordinator._fleet_observations  # noqa: SLF001 - package transaction
        if not observations or abs(observations[-1].time_min - self.world.time) > _EPS:
            raise EnvironmentalNetworkSourceFIFORuntimeError(
                "same-time source augmentation requires an existing canonical observation"
            )
        prior = observations[-1]
        update = augment_environmental_registry_same_time_arrivals(
            self.registry,
            self.world,
            vessel_ids,
            uncertain_pairs=coordinator._uncertain_pairs,  # noqa: SLF001
            responsibility_context=coordinator._responsibility_context,  # noqa: SLF001
            classification_config=coordinator._classification_config,  # noqa: SLF001
            direction_basis=coordinator._direction_basis,  # noqa: SLF001
        )
        observation = EnvironmentalFleetOwnershipObservation(
            registry_update=update,
            dispatch=build_fleet_supervision_dispatch(update),
        )
        observations[-1] = observation

        # Guard histories are canonical-evidence mirrors, not independent registry
        # observations. Replace their same-time tail instead of appending a duplicate.
        for started in coordinator.started_owners:
            guard = started.guarded_session
            history = guard._fleet_observations  # noqa: SLF001
            if history and abs(history[-1].time_min - self.world.time) <= _EPS:
                history[-1] = observation

        mixed = self.session.mixed_session
        for owner in mixed._route_owners.values():  # noqa: SLF001
            if owner.pair_source is None:
                continue
            history = owner.pair_source.guarded_session._fleet_observations  # noqa: SLF001
            if history and abs(history[-1].time_min - self.world.time) <= _EPS:
                history[-1] = observation

        if abs(prior.time_min - observation.time_min) > _EPS:
            raise EnvironmentalNetworkSourceFIFORuntimeError(
                "same-time source augmentation changed canonical observation time"
            )

    def apply_pre_observation_phase(
        self,
        *,
        native_boundary_due: bool,
        source_boundary_due: bool,
    ) -> tuple[str, ...]:
        if not native_boundary_due and not source_boundary_due:
            return ()
        if not self.reservoir.pending_demands and self._pending_plan is None:
            return ()

        source_event: EnvironmentalNetworkSourceEventProposal | None = None
        if source_boundary_due:
            plan = self._pending_plan
            if plan is None or plan.next_event is None:
                raise EnvironmentalNetworkSourceFIFORuntimeError(
                    "source boundary fired without an A150 proposal"
                )
            source_event = plan.next_event
            if abs(source_event.at - self.world.time) > _EPS:
                raise EnvironmentalNetworkSourceFIFORuntimeError(
                    "source proposal time does not match the shared clock"
                )
            if not environmental_network_source_event_is_current(
                source_event,
                generation=self.generation,
            ):
                raise EnvironmentalNetworkSourceFIFORuntimeError(
                    "stale source proposal reached the runtime boundary"
                )

        controller_context = self._prepare_managed_reservoir()
        active_motions = self._active_motions()
        resolution = resolve_environmental_network_source_reservoir(
            self.reservoir,
            self.world.snapshot(),
            self._active_profiles(),
            same_route_dynamic_ingress=self.same_route_dynamic_ingress,
            active_route_targets={
                item.vessel_id: item.route_target
                for item in active_motions
                if item.route_target is not None
            },
            active_limits={item.vessel_id: item.limits for item in active_motions},
            active_environmental_drifts={
                item.vessel_id: item.environmental_drift_velocity
                for item in active_motions
            },
            active_commands={item.vessel_id: item.command for item in active_motions},
            active_motion_backend=self.world.motion_backend,
            active_motion_states=(
                {item.vessel_id: item.motion_state for item in active_motions}
                if self.world.motion_backend is not None else {}
            ),
            future_ingress_clearance=self.future_ingress_clearance,
        )
        admissions = resolution.admissions
        ingress_admissions: tuple[EnvironmentalFleetIngressAdmission, ...] = ()
        if admissions:
            last_registry_time = getattr(self.registry, "_last_update_time", None)
            same_time_prior_observation = (
                last_registry_time is not None
                and abs(last_registry_time - self.world.time) <= _EPS
            )
            ingress_admissions = self.session.apply_current_ingress_batch(
                tuple(item.record for item in admissions)
            )
            if tuple(item.vessel_id for item in ingress_admissions) != tuple(
                item.vessel_id for item in admissions
            ):
                raise EnvironmentalNetworkSourceFIFORuntimeError(
                    "runtime ingress audit disagrees with A148 source admission order"
                )
            if same_time_prior_observation:
                self._supersede_same_time_observation_after_ingress(
                    tuple(item.vessel_id for item in ingress_admissions)
                )

        self.reservoir = resolution.reservoir
        if self.source_admission_controller is not None:
            assert controller_context is not None
            self.source_admission_controller.observe_resolution(
                resolution,
                controller_context,
            )
        generation_before = self.generation
        self.generation = next_environmental_network_source_event_generation(self.generation)
        self._pending_plan = None
        if source_boundary_due or admissions:
            self._boundaries.append(
                EnvironmentalNetworkSourceFIFOBoundary(
                    time_min=self.world.time,
                    generation_before=generation_before,
                    generation_after=self.generation,
                    native_boundary_due=native_boundary_due,
                    source_boundary_due=source_boundary_due,
                    source_event=source_event,
                    resolution=resolution,
                    ingress_admissions=ingress_admissions,
                )
            )
        return tuple(item.vessel_id for item in ingress_admissions)

    def sync_after_caller_horizon(self) -> None:
        """Invalidate frozen evidence if an arbitrary caller horizon advanced the state."""

        plan = self._pending_plan
        if plan is None:
            return
        if abs(plan.snapshot_time - self.world.time) <= _EPS:
            return
        self.generation = next_environmental_network_source_event_generation(self.generation)
        self._pending_plan = None


class EnvironmentalNetworkSourceFIFORuntimeSession:
    """Caller-facing A151 wrapper; runtime authority remains in A119/A123."""

    def __init__(
        self,
        runtime_session: EnvironmentalPersistentFleetControlSession,
        extension: _SourceRuntimeExtension,
    ) -> None:
        self.runtime_session = runtime_session
        self._extension = extension

    @property
    def world(self) -> EnvironmentalWorld2D:
        return self.runtime_session.world

    @property
    def reservoir(self) -> EnvironmentalNetworkSourceReservoir:
        return self._extension.reservoir

    @property
    def source_generation(self) -> int:
        return self._extension.generation

    @property
    def source_plans(self) -> tuple[EnvironmentalNetworkSourceEventPlan, ...]:
        return self._extension.plans

    @property
    def source_boundaries(self) -> tuple[EnvironmentalNetworkSourceFIFOBoundary, ...]:
        return self._extension.boundaries

    @property
    def ingress_admissions(self) -> tuple[EnvironmentalFleetIngressAdmission, ...]:
        return self.runtime_session.ingress_admissions

    def run_until(self, end_time: float) -> EnvironmentalNetworkSourceFIFORuntimeResult:
        result = self.runtime_session.run_until(end_time)
        self._extension.sync_after_caller_horizon()
        return EnvironmentalNetworkSourceFIFORuntimeResult(
            runtime=result,
            reservoir=self.reservoir,
            source_generation=self.source_generation,
            source_boundaries=self.source_boundaries,
        )


    def run_until_capturing_terminal_outcome(
        self, end_time: float
    ) -> EnvironmentalNetworkSourceFIFOTerminalOutcome:
        """Run to ``end_time`` with modeled and resource terminals kept distinct.

        Unexpected exceptions remain exceptions. This is intentionally opt-in so the
        historical A151 caller contract and ordinary ``run_until`` behavior are unchanged.
        """

        try:
            result = self.run_until(end_time)
        except Exception as exc:
            resource = _fifo_resource_unavailability(exc)
            modeled = _fifo_modeled_infeasibility(exc)
            if resource is None and modeled is None:
                raise
            self._extension.sync_after_caller_horizon()
            if resource is not None:
                return EnvironmentalNetworkSourceFIFOTerminalOutcome(
                    status=EnvironmentalNetworkSourceFIFOTerminalStatus.RESOURCE_UNAVAILABLE,
                    result=None,
                    terminal_time_min=float(self.world.time),
                    snapshot=self.world.snapshot(),
                    reservoir=self.reservoir,
                    source_generation=self.source_generation,
                    source_boundaries=self.source_boundaries,
                    ingress_admissions=self.ingress_admissions,
                    resource_unavailability_type=type(resource).__name__,
                    resource_unavailability_message=str(resource),
                    resource_kind=(
                        EnvironmentalNetworkSourceFIFOResourceKind.JOINT_CARTESIAN_COMBINATIONS
                        if isinstance(resource, EnvironmentalJointPhysicsUnavailable)
                        else (
                            EnvironmentalNetworkSourceFIFOResourceKind.JOINT_EXACT_BRANCH_NODES
                            if resource.branch_nodes_visited is not None
                            else EnvironmentalNetworkSourceFIFOResourceKind.JOINT_EXACT_PAIR_EVIDENCE
                        )
                        if isinstance(resource, EnvironmentalJointExactSearchUnavailable)
                        else EnvironmentalNetworkSourceFIFOResourceKind.PAIR_REPLANS
                        if isinstance(resource, EnvironmentalPairControlReplanLimitExceeded)
                        else EnvironmentalNetworkSourceFIFOResourceKind.JOINT_REPLANS
                    ),
                    resource_required_count=(
                        resource.estimate.cartesian_tuple_count
                        if isinstance(resource, EnvironmentalJointPhysicsUnavailable)
                        and resource.estimate is not None
                        else resource.branch_nodes_visited
                        if isinstance(resource, EnvironmentalJointExactSearchUnavailable)
                        and resource.branch_nodes_visited is not None
                        else resource.resource_estimate.pair_command_evidence_count
                        if isinstance(resource, EnvironmentalJointExactSearchUnavailable)
                        and resource.resource_estimate is not None
                        else resource.attempted_replan_number
                        if isinstance(
                            resource,
                            (
                                EnvironmentalPairControlReplanLimitExceeded,
                                EnvironmentalJointControlReplanLimitExceeded,
                            ),
                        )
                        else None
                    ),
                    resource_limit_count=(
                        resource.max_combinations
                        if isinstance(resource, EnvironmentalJointPhysicsUnavailable)
                        else resource.max_branch_nodes
                        if isinstance(resource, EnvironmentalJointExactSearchUnavailable)
                        and resource.branch_nodes_visited is not None
                        else resource.max_pair_evidence
                        if isinstance(resource, EnvironmentalJointExactSearchUnavailable)
                        else resource.max_replans
                        if isinstance(
                            resource,
                            (
                                EnvironmentalPairControlReplanLimitExceeded,
                                EnvironmentalJointControlReplanLimitExceeded,
                            ),
                        )
                        else None
                    ),
                    resource_option_counts=(
                        resource.estimate.option_counts
                        if isinstance(resource, EnvironmentalJointPhysicsUnavailable)
                        and resource.estimate is not None
                        else resource.resource_estimate.option_counts
                        if isinstance(resource, EnvironmentalJointExactSearchUnavailable)
                        and resource.resource_estimate is not None
                        else ()
                    ),
                )
            assert modeled is not None
            return EnvironmentalNetworkSourceFIFOTerminalOutcome(
                status=EnvironmentalNetworkSourceFIFOTerminalStatus.MODELED_INFEASIBLE,
                result=None,
                terminal_time_min=float(self.world.time),
                snapshot=self.world.snapshot(),
                reservoir=self.reservoir,
                source_generation=self.source_generation,
                source_boundaries=self.source_boundaries,
                ingress_admissions=self.ingress_admissions,
                infeasibility_type=type(modeled).__name__,
                infeasibility_message=str(modeled),
            )
        return EnvironmentalNetworkSourceFIFOTerminalOutcome(
            status=EnvironmentalNetworkSourceFIFOTerminalStatus.SUCCESS,
            result=result,
            terminal_time_min=float(result.snapshot.time),
            snapshot=result.snapshot,
            reservoir=result.reservoir,
            source_generation=result.source_generation,
            source_boundaries=result.source_boundaries,
            ingress_admissions=self.ingress_admissions,
        )


def _validate_initial_ingress_world_constraints(
    world: EnvironmentalWorld2D,
    ingress: EnvironmentalFleetIngressRecord,
) -> tuple[ManeuverLimits | None, Vector2 | None]:
    effective_limits = ingress.effective_limits
    limits_store = world._limits  # noqa: SLF001 - cold-start preflight only
    if isinstance(limits_store, dict):
        if effective_limits is None:
            raise EnvironmentalNetworkSourceFIFORuntimeError(
                f"per-vessel world requires maneuver limits for source vessel {ingress.vessel_id!r}"
            )
    elif effective_limits is not None and effective_limits != limits_store:
        raise EnvironmentalNetworkSourceFIFORuntimeError(
            f"source limits for {ingress.vessel_id!r} disagree with shared world limits"
        )

    drift_store = world._drifts  # noqa: SLF001
    drift = ingress.environmental_drift_velocity
    if isinstance(drift_store, dict):
        if drift is None:
            raise EnvironmentalNetworkSourceFIFORuntimeError(
                f"per-vessel world requires environmental drift for source vessel {ingress.vessel_id!r}"
            )
    elif drift is not None and tuple(float(value) for value in drift) != drift_store:
        raise EnvironmentalNetworkSourceFIFORuntimeError(
            f"source drift for {ingress.vessel_id!r} disagrees with shared world drift"
        )
    return effective_limits, drift


def _commit_initial_source_admissions(
    world: EnvironmentalWorld2D,
    route_targets: dict[str, EnvironmentalRouteHandoffTarget],
    profile_catalog: VesselProfileCatalog,
    admissions: tuple[EnvironmentalNetworkSourceAdmission, ...],
) -> tuple[EnvironmentalFleetIngressAdmission, ...]:
    records = tuple(item.record for item in admissions)
    ids = tuple(item.vessel_id for item in records)
    if len(ids) != len(set(ids)):
        raise EnvironmentalNetworkSourceFIFORuntimeError(
            "initial source admission vessel ids must be unique"
        )
    active = {item.vessel_id for item in world.snapshot().vessels}
    overlap = active.intersection(ids)
    if overlap:
        raise EnvironmentalNetworkSourceFIFORuntimeError(
            f"initial source admissions overlap active vessels: {sorted(overlap)!r}"
        )

    validated = tuple(
        (record, *_validate_initial_ingress_world_constraints(world, record))
        for record in records
    )
    audit_flags = tuple(
        (
            record.vessel_id in route_targets
            and route_targets[record.vessel_id] != record.route_target,
            record.profile is not None,
        )
        for record in records
    )

    # Publish the complete mission metadata before adding physical members.  Every
    # fallible identity/world check above has already succeeded.
    for record, _, _ in validated:
        route_targets[record.vessel_id] = record.route_target
        existing_profile = profile_catalog.profile_for(record.vessel_id)
        if existing_profile is not None and existing_profile != record.profile:
            profile_catalog.remove(record.vessel_id)
        if record.profile is not None:
            profile_catalog.assign(record.vessel_id, record.profile)

    evidence: list[EnvironmentalFleetIngressAdmission] = []
    for (record, limits, drift), (route_replaced, profile_assigned) in zip(
        validated, audit_flags, strict=True
    ):
        world.add_vessel(
            record.vessel,
            limits=limits,
            environmental_drift_velocity=drift,
        )
        evidence.append(
            EnvironmentalFleetIngressAdmission(
                record,
                route_target_replaced=route_replaced,
                profile_assigned=profile_assigned,
            )
        )
    return tuple(evidence)


def start_environmental_network_source_fifo_runtime(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    reservoir: EnvironmentalNetworkSourceReservoir,
    *,
    recapture_config: EnvironmentalFleetMultiOwnerRecaptureConfig,
    runtime_config: EnvironmentalFleetRuntimeConfig = EnvironmentalFleetRuntimeConfig(),
    recheck_config: EnvironmentalNetworkSourceRecheckConfig = EnvironmentalNetworkSourceRecheckConfig(),
    uncertain_pairs: Iterable[PairKey] = (),
    responsibility_context: ResponsibilityContext | None = None,
    classification_config: EncounterClassificationConfig | None = None,
    direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
    profile_catalog: VesselProfileCatalog | None = None,
    same_route_dynamic_ingress: SameRouteFollowingSupervisor | None = None,
    future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None = None,
    persistent_same_route_following: SameRouteFollowingSupervisor | None = None,
    source_admission_controller: EnvironmentalNetworkSourceAdmissionController | None = None,
) -> EnvironmentalNetworkSourceFIFORuntimeStartResult:
    """Cold-start the existing persistent fleet runtime with arm-local FIFO demand.

    Due-and-clear source heads are committed *before* the one cold-start registry
    observation.  Future/blocked demand remains outside the world and is thereafter
    advanced only by the opt-in source boundary extension installed on A119/A123.
    """

    if not isinstance(registry, EncounterRegistry):
        raise TypeError("registry must be EncounterRegistry")
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    if not isinstance(route_targets, Mapping):
        raise TypeError("route_targets must be a mapping")
    if not isinstance(reservoir, EnvironmentalNetworkSourceReservoir):
        raise TypeError("reservoir must be EnvironmentalNetworkSourceReservoir")
    if not isinstance(recapture_config, EnvironmentalFleetMultiOwnerRecaptureConfig):
        raise TypeError("recapture_config must be EnvironmentalFleetMultiOwnerRecaptureConfig")
    if not isinstance(runtime_config, EnvironmentalFleetRuntimeConfig):
        raise TypeError("runtime_config must be EnvironmentalFleetRuntimeConfig")
    if not isinstance(recheck_config, EnvironmentalNetworkSourceRecheckConfig):
        raise TypeError("recheck_config must be EnvironmentalNetworkSourceRecheckConfig")
    if same_route_dynamic_ingress is not None and not isinstance(
        same_route_dynamic_ingress, SameRouteFollowingSupervisor
    ):
        raise TypeError("same_route_dynamic_ingress must be SameRouteFollowingSupervisor or None")
    if future_ingress_clearance is not None and not isinstance(
        future_ingress_clearance, EnvironmentalNetworkSourceFutureIngressConfig
    ):
        raise TypeError(
            "future_ingress_clearance must be EnvironmentalNetworkSourceFutureIngressConfig or None"
        )
    if persistent_same_route_following is not None and not isinstance(
        persistent_same_route_following, SameRouteFollowingSupervisor
    ):
        raise TypeError(
            "persistent_same_route_following must be SameRouteFollowingSupervisor or None"
        )
    if getattr(registry, "_last_update_time", None) is not None:
        raise EnvironmentalNetworkSourceFIFORuntimeError(
            "FIFO cold start requires a fresh EncounterRegistry"
        )

    runtime_targets = dict(route_targets)
    runtime_profiles = profile_catalog if profile_catalog is not None else VesselProfileCatalog()
    initial_controller_context: EnvironmentalNetworkSourceControllerContext | None = None
    if source_admission_controller is not None:
        initial_controller_context = _controller_context(
            world=world,
            profile_catalog=runtime_profiles,
            route_targets=runtime_targets,
            same_route_dynamic_ingress=same_route_dynamic_ingress,
            future_ingress_clearance=future_ingress_clearance,
        )
        reservoir = _apply_source_admission_controller(
            source_admission_controller,
            reservoir,
            initial_controller_context,
        )
    active_profiles = {
        state.vessel_id: profile
        for state in world.snapshot().vessels
        if (profile := runtime_profiles.profile_for(state.vessel_id)) is not None
    }
    initial_snapshot = world.snapshot()
    initial_resolution = resolve_environmental_network_source_reservoir(
        reservoir,
        initial_snapshot,
        active_profiles,
        same_route_dynamic_ingress=same_route_dynamic_ingress,
        active_route_targets={
            vessel_id: runtime_targets[vessel_id]
            for vessel_id in active_profiles
            if vessel_id in runtime_targets
        },
        active_limits={
            state.vessel_id: world.limits_for(state.vessel_id)
            for state in initial_snapshot.vessels
        },
        active_environmental_drifts={
            state.vessel_id: tuple(world.drift_for(state.vessel_id))
            for state in initial_snapshot.vessels
        },
        active_commands={
            state.vessel_id: world.command_for(state.vessel_id)
            for state in initial_snapshot.vessels
        },
        active_motion_backend=world.motion_backend,
        active_motion_states=(
            {state.vessel_id: world.motion_state_for(state.vessel_id) for state in initial_snapshot.vessels}
            if world.motion_backend is not None else {}
        ),
        future_ingress_clearance=future_ingress_clearance,
    )
    if source_admission_controller is not None:
        assert initial_controller_context is not None
        source_admission_controller.observe_resolution(
            initial_resolution,
            initial_controller_context,
        )
    initial_ingress = _commit_initial_source_admissions(
        world,
        runtime_targets,
        runtime_profiles,
        initial_resolution.admissions,
    )

    runtime_start = start_environmental_fleet_runtime(
        registry,
        world,
        runtime_targets,
        recapture_config=recapture_config,
        config=runtime_config,
        uncertain_pairs=uncertain_pairs,
        responsibility_context=responsibility_context,
        classification_config=classification_config,
        direction_basis=direction_basis,
        profile_catalog=runtime_profiles,
        persistent_same_route_following=persistent_same_route_following,
    )
    runtime_start.session._ingress_admissions.extend(initial_ingress)  # noqa: SLF001 - same package audit ledger
    initial_ids = tuple(item.vessel_id for item in initial_ingress)
    initial_clear_route_owner_ids = runtime_start.session.admit_current_clear_route_targets(initial_ids)

    extension = _SourceRuntimeExtension(
        runtime_start.session,
        initial_resolution.reservoir,
        recheck_config=recheck_config,
        same_route_dynamic_ingress=same_route_dynamic_ingress,
        future_ingress_clearance=future_ingress_clearance,
        source_admission_controller=source_admission_controller,
    )
    runtime_start.session.mixed_session._install_source_boundary_extension(extension)  # noqa: SLF001
    session = EnvironmentalNetworkSourceFIFORuntimeSession(runtime_start.session, extension)
    return EnvironmentalNetworkSourceFIFORuntimeStartResult(
        session=session,
        runtime_start=runtime_start,
        initial_resolution=initial_resolution,
        initial_ingress_admissions=initial_ingress,
        initial_clear_route_owner_ids=initial_clear_route_owner_ids,
    )
