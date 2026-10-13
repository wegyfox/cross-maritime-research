"""Opt-in Full CROSS runtime integration for the R1 P4C matched source-service gate.

DESIGN/QUALIFICATION ONLY — this module does not define or authorize a scientific P4C
campaign.  It composes the already qualified global M1-matched source-service gate with
existing A148/A149/A150 physical source clearance and the existing persistent CROSS
runtime without changing those base modules.

This runtime is a deliberately non-independent diagnostic CONTROL ARM.  It must never be
substituted for the ordinary R1 CROSS target, whose headways/admissions remain endogenous.
The imposed gate exists only to test matched mechanism transfer and incremental physical
source delay after that transfer.

Key invariant: only the current global P4C head is ever exposed to physical source
clearance.  A physically blocked head cannot be bypassed by another source stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping

from vessel_crossing.nextgen.encounter_classification import EncounterClassificationConfig
from vessel_crossing.nextgen.encounter_registry import EncounterRegistry, PairKey
from vessel_crossing.nextgen.environmental_encounter_classification import EnvironmentalEncounterDirectionBasis
from vessel_crossing.nextgen.environmental_fleet_ingress import EnvironmentalFleetIngressAdmission
from vessel_crossing.nextgen.environmental_fleet_multi_owner_recapture import EnvironmentalFleetMultiOwnerRecaptureConfig
from vessel_crossing.nextgen.environmental_fleet_runtime import EnvironmentalFleetRuntimeConfig, EnvironmentalFleetRuntimeStartResult
from vessel_crossing.nextgen.environmental_network_source_event_scheduler import (
    EnvironmentalNetworkSourceEventPlan,
    plan_environmental_network_source_event,
    next_environmental_network_source_event_generation,
)
from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import (
    EnvironmentalNetworkSourceFIFORuntimeSession,
    EnvironmentalNetworkSourceFIFORuntimeError,
    _SourceRuntimeExtension,
    start_environmental_network_source_fifo_runtime,
)
from vessel_crossing.nextgen.environmental_network_source_recheck import EnvironmentalNetworkSourceRecheckConfig
from vessel_crossing.nextgen.environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceFutureIngressConfig,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalNetworkSourceResolution,
    resolve_environmental_network_source_reservoir,
    source_queue_metrics,
)
from vessel_crossing.nextgen.environmental_route_handoff import EnvironmentalRouteHandoffTarget
from vessel_crossing.nextgen.environmental_world import EnvironmentalWorld2D
from vessel_crossing.nextgen.responsibility import ResponsibilityContext
from vessel_crossing.nextgen.same_route_following import SameRouteFollowingSupervisor
from vessel_crossing.nextgen.vessel_classes import VesselProfileCatalog

from .r1_p4b_publication_reference import GUO_2024_ILLUSTRATIVE, PairHeadwayMatrix, TypedDemandEvent
from .r1_p4c_matched_source_service import (
    MatchedSourceServiceCandidate,
    MatchedSourceServiceQueue,
    build_matched_source_service_queue,
    commit_matched_source_release,
)

_EPS = 1e-10


class P4CMatchedSourceRuntimeError(RuntimeError):
    """Raised when Full CROSS integration would violate the matched-mechanism contract."""


@dataclass(frozen=True, slots=True)
class P4CMatchedSourceRuntimePlan:
    generation: int
    snapshot_time: float
    native_frozen_until: float
    candidate: MatchedSourceServiceCandidate
    headway_ready_min: float
    next_boundary_min: float | None
    physical_plan: EnvironmentalNetworkSourceEventPlan | None


@dataclass(frozen=True, slots=True)
class P4CMatchedSourceRuntimeBoundary:
    time_min: float
    generation_before: int
    generation_after: int
    native_boundary_due: bool
    source_boundary_due: bool
    candidate: MatchedSourceServiceCandidate | None
    resolution: EnvironmentalNetworkSourceResolution | None
    ingress_admissions: tuple[EnvironmentalFleetIngressAdmission, ...]


@dataclass(frozen=True, slots=True)
class P4CMatchedSourceRuntimeStartResult:
    session: EnvironmentalNetworkSourceFIFORuntimeSession
    runtime_start: EnvironmentalFleetRuntimeStartResult
    initial_gate: MatchedSourceServiceQueue


class _P4CMatchedSourceRuntimeExtension(_SourceRuntimeExtension):
    """A151-compatible boundary extension exposing only the P4C global queue head."""

    def __init__(
        self,
        session,
        reservoir: EnvironmentalNetworkSourceReservoir,
        typed_events: Iterable[TypedDemandEvent],
        *,
        matrix: PairHeadwayMatrix,
        recheck_config: EnvironmentalNetworkSourceRecheckConfig,
        same_route_dynamic_ingress: SameRouteFollowingSupervisor | None = None,
        future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None = None,
    ) -> None:
        super().__init__(
            session,
            reservoir,
            recheck_config=recheck_config,
            same_route_dynamic_ingress=same_route_dynamic_ingress,
            future_ingress_clearance=future_ingress_clearance,
        )
        typed = tuple(typed_events)
        demand_by_id = {item.vessel_id: item for item in reservoir.demands}
        if len(demand_by_id) != len(reservoir.demands):
            raise P4CMatchedSourceRuntimeError("source demand vessel ids must be unique")
        if {item.event_id for item in typed} != set(demand_by_id):
            raise P4CMatchedSourceRuntimeError("typed events must cover source demand ids exactly")
        for event in typed:
            demand = demand_by_id[event.event_id]
            if abs(event.arrival_min - demand.demand_time) > _EPS or event.stream_id != demand.stream_id:
                raise P4CMatchedSourceRuntimeError(
                    f"typed demand identity/timing mismatch for {event.event_id!r}"
                )
        self.gate = build_matched_source_service_queue(typed, matrix=matrix)
        self._p4c_plans: list[P4CMatchedSourceRuntimePlan] = []
        self._p4c_boundaries: list[P4CMatchedSourceRuntimeBoundary] = []
        self._pending_p4c_plan: P4CMatchedSourceRuntimePlan | None = None

    @property
    def plans(self) -> tuple[P4CMatchedSourceRuntimePlan, ...]:
        return tuple(self._p4c_plans)

    @property
    def boundaries(self) -> tuple[P4CMatchedSourceRuntimeBoundary, ...]:
        return tuple(self._p4c_boundaries)

    def _demand_for_candidate(self, candidate: MatchedSourceServiceCandidate):
        pending = {item.vessel_id: item for item in self.reservoir.pending_demands}
        demand = pending.get(candidate.event_id)
        if demand is None:
            raise P4CMatchedSourceRuntimeError(
                f"matched global head {candidate.event_id!r} is not pending in the physical reservoir"
            )
        # Independent guard against accidental drift between gate order and reservoir contents.
        canonical_pending = min(
            self.reservoir.pending_demands,
            key=lambda item: (item.demand_time, item.vessel_id),
        )
        if canonical_pending.vessel_id != candidate.event_id:
            raise P4CMatchedSourceRuntimeError(
                "physical pending set disagrees with canonical global matched-head order"
            )
        return demand

    def _shadow_reservoir(self, candidate: MatchedSourceServiceCandidate) -> EnvironmentalNetworkSourceReservoir:
        return EnvironmentalNetworkSourceReservoir((self._demand_for_candidate(candidate),))

    def _resolve_current_head(self, candidate: MatchedSourceServiceCandidate) -> EnvironmentalNetworkSourceResolution:
        if self.world.time < candidate.headway_ready_min - _EPS:
            raise P4CMatchedSourceRuntimeError("physical resolver called before matched headway readiness")
        shadow = self._shadow_reservoir(candidate)
        active_motions = self._active_motions()
        shadow_resolution = resolve_environmental_network_source_reservoir(
            shadow,
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
                item.vessel_id: item.environmental_drift_velocity for item in active_motions
            },
            active_commands={item.vessel_id: item.command for item in active_motions},
            active_motion_backend=self.world.motion_backend,
            active_motion_states=(
                {item.vessel_id: item.motion_state for item in active_motions}
                if self.world.motion_backend is not None else {}
            ),
            future_ingress_clearance=self.future_ingress_clearance,
        )
        admissions = shadow_resolution.admissions
        if len(admissions) > 1:
            raise P4CMatchedSourceRuntimeError("one global P4C head produced multiple admissions")
        if admissions:
            admission = admissions[0]
            if admission.vessel_id != candidate.event_id:
                raise P4CMatchedSourceRuntimeError("physical admission disagrees with current P4C head")
            next_reservoir = EnvironmentalNetworkSourceReservoir(
                demands=self.reservoir.demands,
                admissions=self.reservoir.admissions + admissions,
            )
        else:
            next_reservoir = self.reservoir
        return EnvironmentalNetworkSourceResolution(
            snapshot_time=self.world.time,
            decisions=shadow_resolution.decisions,
            admissions=admissions,
            reservoir=next_reservoir,
            metrics=source_queue_metrics(next_reservoir, at=self.world.time),
        )

    def next_boundary_time(self, native_frozen_until: float) -> float | None:
        if not isfinite(native_frozen_until) or native_frozen_until < self.world.time - _EPS:
            raise P4CMatchedSourceRuntimeError("native frozen horizon must not precede shared clock")
        if not self.reservoir.pending_demands:
            if not self.gate.complete:
                raise P4CMatchedSourceRuntimeError("physical reservoir completed before matched gate")
            self._pending_p4c_plan = None
            return None
        candidate = self.gate.candidate
        if candidate is None:
            raise P4CMatchedSourceRuntimeError("matched gate completed while physical demand remains")
        self._demand_for_candidate(candidate)

        physical_plan: EnvironmentalNetworkSourceEventPlan | None = None
        next_boundary: float | None = None
        if candidate.headway_ready_min > self.world.time + _EPS:
            if candidate.headway_ready_min < native_frozen_until - _EPS:
                next_boundary = candidate.headway_ready_min
        else:
            physical_plan = plan_environmental_network_source_event(
                self._shadow_reservoir(candidate),
                self.world.snapshot(),
                self._active_motions(),
                frozen_until=float(native_frozen_until),
                generation=self.generation,
                recheck_config=self.recheck_config,
                same_route_dynamic_ingress=self.same_route_dynamic_ingress,
                future_ingress_clearance=self.future_ingress_clearance,
            )
            if physical_plan.next_event is not None:
                next_boundary = physical_plan.next_event.at

        plan = P4CMatchedSourceRuntimePlan(
            generation=self.generation,
            snapshot_time=self.world.time,
            native_frozen_until=float(native_frozen_until),
            candidate=candidate,
            headway_ready_min=candidate.headway_ready_min,
            next_boundary_min=next_boundary,
            physical_plan=physical_plan,
        )
        self._pending_p4c_plan = plan
        self._p4c_plans.append(plan)
        return next_boundary

    def apply_pre_observation_phase(
        self,
        *,
        native_boundary_due: bool,
        source_boundary_due: bool,
    ) -> tuple[str, ...]:
        if not native_boundary_due and not source_boundary_due:
            return ()
        if not self.reservoir.pending_demands:
            self._pending_p4c_plan = None
            return ()

        plan = self._pending_p4c_plan
        if source_boundary_due:
            if plan is None or plan.next_boundary_min is None:
                raise P4CMatchedSourceRuntimeError("P4C source boundary fired without a pending plan")
            if abs(plan.next_boundary_min - self.world.time) > _EPS:
                raise P4CMatchedSourceRuntimeError("P4C source boundary time disagrees with shared clock")
            if plan.generation != self.generation:
                raise P4CMatchedSourceRuntimeError("stale P4C source plan reached runtime boundary")

        candidate = self.gate.candidate
        if candidate is None:
            raise P4CMatchedSourceRuntimeError("physical demand remains after matched gate completion")
        resolution: EnvironmentalNetworkSourceResolution | None = None
        ingress_admissions: tuple[EnvironmentalFleetIngressAdmission, ...] = ()
        if self.world.time >= candidate.headway_ready_min - _EPS:
            resolution = self._resolve_current_head(candidate)
            if resolution.admissions:
                last_registry_time = getattr(self.registry, "_last_update_time", None)
                same_time_prior_observation = (
                    last_registry_time is not None and abs(last_registry_time - self.world.time) <= _EPS
                )
                ingress_admissions = self.session.apply_current_ingress_batch(
                    tuple(item.record for item in resolution.admissions)
                )
                if tuple(item.vessel_id for item in ingress_admissions) != tuple(
                    item.vessel_id for item in resolution.admissions
                ):
                    raise P4CMatchedSourceRuntimeError(
                        "runtime ingress audit disagrees with matched physical admission"
                    )
                if same_time_prior_observation:
                    self._supersede_same_time_observation_after_ingress(
                        tuple(item.vessel_id for item in ingress_admissions)
                    )
                self.reservoir = resolution.reservoir
                self.gate = commit_matched_source_release(
                    self.gate,
                    event_id=candidate.event_id,
                    release_min=self.world.time,
                )

        generation_before = self.generation
        self.generation = next_environmental_network_source_event_generation(self.generation)
        self._pending_p4c_plan = None
        self._p4c_boundaries.append(
            P4CMatchedSourceRuntimeBoundary(
                time_min=self.world.time,
                generation_before=generation_before,
                generation_after=self.generation,
                native_boundary_due=native_boundary_due,
                source_boundary_due=source_boundary_due,
                candidate=candidate,
                resolution=resolution,
                ingress_admissions=ingress_admissions,
            )
        )
        return tuple(item.vessel_id for item in ingress_admissions)

    def sync_after_caller_horizon(self) -> None:
        plan = self._pending_p4c_plan
        if plan is None:
            return
        if abs(plan.snapshot_time - self.world.time) <= _EPS:
            return
        self.generation = next_environmental_network_source_event_generation(self.generation)
        self._pending_p4c_plan = None


def start_r1_p4c_matched_source_runtime(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    reservoir: EnvironmentalNetworkSourceReservoir,
    typed_events: Iterable[TypedDemandEvent],
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
    matrix: PairHeadwayMatrix = GUO_2024_ILLUSTRATIVE,
) -> P4CMatchedSourceRuntimeStartResult:
    """Cold-start CROSS with a fresh opt-in P4C matched global source gate.

    The validated A151 starter is used only to initialize the persistent runtime with an
    empty source reservoir.  Its inert empty extension is then replaced by this P4C
    research extension.  No base nextgen module is modified.
    """

    base = start_environmental_network_source_fifo_runtime(
        registry,
        world,
        route_targets,
        EnvironmentalNetworkSourceReservoir(()),
        recapture_config=recapture_config,
        runtime_config=runtime_config,
        recheck_config=recheck_config,
        uncertain_pairs=uncertain_pairs,
        responsibility_context=responsibility_context,
        classification_config=classification_config,
        direction_basis=direction_basis,
        profile_catalog=profile_catalog,
        same_route_dynamic_ingress=same_route_dynamic_ingress,
        future_ingress_clearance=future_ingress_clearance,
        persistent_same_route_following=persistent_same_route_following,
    )
    core = base.runtime_start.session
    old_extension = core.mixed_session._source_boundary_extension  # noqa: SLF001 - opt-in research composition
    if old_extension is None or old_extension.reservoir.pending_demands:
        raise P4CMatchedSourceRuntimeError("expected one inert empty A151 source extension")
    extension = _P4CMatchedSourceRuntimeExtension(
        core,
        reservoir,
        typed_events,
        matrix=matrix,
        recheck_config=recheck_config,
        same_route_dynamic_ingress=same_route_dynamic_ingress,
        future_ingress_clearance=future_ingress_clearance,
    )
    core.mixed_session._source_boundary_extension = extension  # noqa: SLF001 - explicit opt-in replacement
    session = EnvironmentalNetworkSourceFIFORuntimeSession(core, extension)
    return P4CMatchedSourceRuntimeStartResult(
        session=session,
        runtime_start=base.runtime_start,
        initial_gate=extension.gate,
    )
