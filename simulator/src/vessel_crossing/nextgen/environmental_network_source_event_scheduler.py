"""Pure source-event planning above the A148 reservoir and A149 recheck proof.

A150 still does not mutate the environmental runtime.  It composes one exact source
snapshot resolution with frozen-segment recheck evidence and returns at most one next
source-event proposal.  Every proposal is bound to an explicit caller-owned generation
number; a runtime integration must increment that generation whenever source/reservoir,
world membership/state-at-boundary, active command, profile/domain or frozen-horizon
evidence is invalidated.

A proposal at the same timestamp as the supplied frozen boundary is deliberately
suppressed.  The external boundary must be processed first and the source plan rebuilt
from the resulting exact state, preventing ambiguous same-time ordering from leaking
into admission semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Iterable

from .environmental_network_source_recheck import (
    EnvironmentalNetworkSourceActiveMotion,
    EnvironmentalNetworkSourceRecheckConfig,
    EnvironmentalNetworkSourceRecheckResult,
    EnvironmentalNetworkSourceRecheckStatus,
    find_environmental_network_source_recheck,
)
from .environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceDemand,
    EnvironmentalNetworkSourceHeadStatus,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalNetworkSourceResolution,
    EnvironmentalNetworkSourceFutureIngressConfig,
    resolve_environmental_network_source_reservoir,
)
from .environmental_world import EnvironmentalWorldSnapshot
from .same_route_following import SameRouteFollowingSupervisor

_EPS = 1e-10


class EnvironmentalNetworkSourceEventPlanningError(RuntimeError):
    """Raised when source-event planning evidence is internally inconsistent."""


class EnvironmentalNetworkSourceEventCauseKind(str, Enum):
    CURRENT_ADMISSION = "CURRENT_ADMISSION"
    FUTURE_DEMAND = "FUTURE_DEMAND"
    CERTIFIED_RECHECK = "CERTIFIED_RECHECK"


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceEventCause:
    kind: EnvironmentalNetworkSourceEventCauseKind
    stream_id: str
    vessel_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EnvironmentalNetworkSourceEventCauseKind):
            raise TypeError("kind must be EnvironmentalNetworkSourceEventCauseKind")
        if not self.stream_id or not self.vessel_id:
            raise ValueError("source event cause requires non-empty stream/vessel ids")


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceHeadRecheckEvidence:
    stream_id: str
    vessel_id: str
    result: EnvironmentalNetworkSourceRecheckResult


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceEventProposal:
    at: float
    generation: int
    valid_until: float
    causes: tuple[EnvironmentalNetworkSourceEventCause, ...]
    proposal_id: str

    def __post_init__(self) -> None:
        if not isfinite(self.at) or not isfinite(self.valid_until):
            raise ValueError("proposal times must be finite")
        if self.at > self.valid_until + _EPS:
            raise ValueError("proposal cannot lie after its validity horizon")
        _validate_generation(self.generation)
        if not self.causes:
            raise ValueError("proposal must contain at least one cause")
        if tuple(sorted(self.causes, key=_cause_key)) != self.causes:
            raise ValueError("proposal causes must be in canonical order")
        if not self.proposal_id:
            raise ValueError("proposal_id must be non-empty")

    def is_current(self, generation: int) -> bool:
        _validate_generation(generation)
        return self.generation == generation


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceEventPlan:
    generation: int
    snapshot_time: float
    frozen_until: float
    current_resolution: EnvironmentalNetworkSourceResolution
    rechecks: tuple[EnvironmentalNetworkSourceHeadRecheckEvidence, ...]
    next_event: EnvironmentalNetworkSourceEventProposal | None
    awaiting_invalidation: bool

    @property
    def requires_current_admission_commit(self) -> bool:
        return bool(self.current_resolution.admissions)

    @property
    def complete(self) -> bool:
        return (
            not self.current_resolution.reservoir.pending_demands
            and not self.current_resolution.admissions
        )


def _validate_generation(generation: int) -> None:
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("source event generation must be a non-negative integer")


def next_environmental_network_source_event_generation(generation: int) -> int:
    """Return the next caller-owned invalidation generation."""

    _validate_generation(generation)
    return generation + 1


def environmental_network_source_event_is_current(
    proposal: EnvironmentalNetworkSourceEventProposal,
    *,
    generation: int,
) -> bool:
    if not isinstance(proposal, EnvironmentalNetworkSourceEventProposal):
        raise TypeError("proposal must be EnvironmentalNetworkSourceEventProposal")
    return proposal.is_current(generation)


def _cause_key(cause: EnvironmentalNetworkSourceEventCause) -> tuple[str, str, str]:
    return (cause.kind.value, cause.stream_id, cause.vessel_id)


def _proposal_id(
    *,
    at: float,
    generation: int,
    valid_until: float,
    causes: tuple[EnvironmentalNetworkSourceEventCause, ...],
) -> str:
    payload = {
        "at": at,
        "causes": [
            {"kind": cause.kind.value, "stream_id": cause.stream_id, "vessel_id": cause.vessel_id}
            for cause in causes
        ],
        "generation": generation,
        "valid_until": valid_until,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _build_proposal(
    *,
    at: float,
    generation: int,
    valid_until: float,
    causes: Iterable[EnvironmentalNetworkSourceEventCause],
) -> EnvironmentalNetworkSourceEventProposal:
    canonical_causes = tuple(sorted(set(causes), key=_cause_key))
    return EnvironmentalNetworkSourceEventProposal(
        at=float(at),
        generation=generation,
        valid_until=float(valid_until),
        causes=canonical_causes,
        proposal_id=_proposal_id(
            at=float(at),
            generation=generation,
            valid_until=float(valid_until),
            causes=canonical_causes,
        ),
    )


def _validate_active_motion_snapshot_alignment(
    snapshot: EnvironmentalWorldSnapshot,
    motions: tuple[EnvironmentalNetworkSourceActiveMotion, ...],
) -> None:
    ids = tuple(item.vessel_id for item in motions)
    if len(set(ids)) != len(ids):
        raise EnvironmentalNetworkSourceEventPlanningError("duplicate active motion vessel id")
    snapshot_by_id = {item.vessel_id: item for item in snapshot.vessels}
    if set(ids) != set(snapshot_by_id):
        raise EnvironmentalNetworkSourceEventPlanningError(
            "active motion ids must exactly match the environmental snapshot"
        )
    mismatched = sorted(
        item.vessel_id for item in motions if item.state != snapshot_by_id[item.vessel_id]
    )
    if mismatched:
        raise EnvironmentalNetworkSourceEventPlanningError(
            f"active motion state does not match snapshot for: {mismatched!r}"
        )


def _pending_stream_heads(
    reservoir: EnvironmentalNetworkSourceReservoir,
) -> tuple[EnvironmentalNetworkSourceDemand, ...]:
    heads: dict[str, EnvironmentalNetworkSourceDemand] = {}
    for demand in reservoir.pending_demands_in_service_order:
        if demand.stream_id not in heads:
            heads[demand.stream_id] = demand
    return tuple(
        sorted(
            heads.values(),
            key=lambda item: (
                item.physical_eligibility_time,
                item.demand_time,
                item.stream_id,
                item.vessel_id,
            ),
        )
    )


def plan_environmental_network_source_event(
    reservoir: EnvironmentalNetworkSourceReservoir,
    snapshot: EnvironmentalWorldSnapshot,
    active_motions: Iterable[EnvironmentalNetworkSourceActiveMotion],
    *,
    frozen_until: float,
    generation: int,
    recheck_config: EnvironmentalNetworkSourceRecheckConfig = EnvironmentalNetworkSourceRecheckConfig(),
    same_route_dynamic_ingress: SameRouteFollowingSupervisor | None = None,
    future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None = None,
) -> EnvironmentalNetworkSourceEventPlan:
    """Build one immutable source plan without mutating runtime state.

    ``frozen_until`` is the next external physical/runtime boundary after which the
    supplied active motions may no longer be reused.  Source events exactly on that
    boundary are suppressed so the external mutation wins same-time ordering and the
    source plan is recomputed afterwards.
    """

    if not isinstance(reservoir, EnvironmentalNetworkSourceReservoir):
        raise TypeError("reservoir must be EnvironmentalNetworkSourceReservoir")
    if not isinstance(snapshot, EnvironmentalWorldSnapshot):
        raise TypeError("snapshot must be EnvironmentalWorldSnapshot")
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
    _validate_generation(generation)
    if not isfinite(frozen_until) or frozen_until < snapshot.time - _EPS:
        raise ValueError("frozen_until must be finite and not before snapshot time")

    motions = tuple(active_motions)
    _validate_active_motion_snapshot_alignment(snapshot, motions)
    active_profiles = {item.vessel_id: item.profile for item in motions}
    active_route_targets = {
        item.vessel_id: item.route_target
        for item in motions
        if item.route_target is not None
    }
    active_limits = {item.vessel_id: item.limits for item in motions}
    active_drifts = {
        item.vessel_id: item.environmental_drift_velocity for item in motions
    }
    active_commands = {item.vessel_id: item.command for item in motions}
    backends = {id(item.motion_backend): item.motion_backend for item in motions if item.motion_backend is not None}
    if len(backends) > 1:
        raise EnvironmentalNetworkSourceEventPlanningError(
            "active motions must share one environmental motion backend"
        )
    active_motion_backend = next(iter(backends.values()), None)
    active_motion_states = {item.vessel_id: item.motion_state for item in motions} if active_motion_backend is not None else {}
    current = resolve_environmental_network_source_reservoir(
        reservoir,
        snapshot,
        active_profiles,
        same_route_dynamic_ingress=same_route_dynamic_ingress,
        active_route_targets=active_route_targets,
        active_limits=active_limits,
        active_environmental_drifts=active_drifts,
        active_commands=active_commands,
        active_motion_backend=active_motion_backend,
        active_motion_states=active_motion_states,
        future_ingress_clearance=future_ingress_clearance,
    )

    # Any current-time admission changes active membership immediately.  Future proof
    # evidence built from the pre-admission snapshot would therefore be stale by
    # construction; stop here and require the caller to commit/invalidate/replan.
    if current.admissions:
        causes = tuple(
            EnvironmentalNetworkSourceEventCause(
                EnvironmentalNetworkSourceEventCauseKind.CURRENT_ADMISSION,
                item.stream_id,
                item.vessel_id,
            )
            for item in current.admissions
        )
        proposal = _build_proposal(
            at=snapshot.time,
            generation=generation,
            valid_until=frozen_until,
            causes=causes,
        )
        return EnvironmentalNetworkSourceEventPlan(
            generation=generation,
            snapshot_time=snapshot.time,
            frozen_until=frozen_until,
            current_resolution=current,
            rechecks=(),
            next_event=proposal,
            awaiting_invalidation=False,
        )

    demand_by_id = {item.vessel_id: item for item in reservoir.pending_demands}
    duration = max(0.0, frozen_until - snapshot.time)
    rechecks: list[EnvironmentalNetworkSourceHeadRecheckEvidence] = []
    candidates: list[tuple[float, EnvironmentalNetworkSourceEventCause]] = []

    for decision in current.decisions:
        if decision.status is EnvironmentalNetworkSourceHeadStatus.ADMITTED:
            raise EnvironmentalNetworkSourceEventPlanningError(
                "admitted decision exists without current admission evidence"
            )
        demand = demand_by_id.get(decision.vessel_id)
        if demand is None:
            raise EnvironmentalNetworkSourceEventPlanningError(
                f"current source decision refers to unknown pending demand {decision.vessel_id!r}"
            )
        recheck = find_environmental_network_source_recheck(
            demand,
            motions,
            segment_start_time=snapshot.time,
            segment_duration_min=duration,
            config=recheck_config,
            same_route_dynamic_ingress=same_route_dynamic_ingress,
            future_ingress_clearance=future_ingress_clearance,
        )
        if recheck.status is EnvironmentalNetworkSourceRecheckStatus.ALREADY_CLEAR:
            raise EnvironmentalNetworkSourceEventPlanningError(
                "A148 blocked a source head that A149 reports already clear"
            )
        rechecks.append(
            EnvironmentalNetworkSourceHeadRecheckEvidence(
                stream_id=decision.stream_id,
                vessel_id=decision.vessel_id,
                result=recheck,
            )
        )
        if (
            recheck.status is EnvironmentalNetworkSourceRecheckStatus.CERTIFIED_CLEARANCE_WINDOW
            and recheck.certified_clear_time is not None
            and recheck.certified_clear_time < frozen_until - _EPS
        ):
            candidates.append(
                (
                    recheck.certified_clear_time,
                    EnvironmentalNetworkSourceEventCause(
                        EnvironmentalNetworkSourceEventCauseKind.CERTIFIED_RECHECK,
                        decision.stream_id,
                        decision.vessel_id,
                    ),
                )
            )

    # A stream that already has a due head cannot be changed by later same-stream demand;
    # FIFO keeps that later demand behind the current head.  For streams whose head is
    # not due yet, the exact demand timestamp is a genuine source eligibility boundary.
    due_streams = {item.stream_id for item in current.decisions}
    for head in _pending_stream_heads(reservoir):
        if head.stream_id in due_streams:
            continue
        if head.physical_eligibility_time <= snapshot.time + _EPS:
            raise EnvironmentalNetworkSourceEventPlanningError(
                "due source head missing from instantaneous A148 resolution"
            )
        if head.physical_eligibility_time < frozen_until - _EPS:
            candidates.append(
                (
                    head.physical_eligibility_time,
                    EnvironmentalNetworkSourceEventCause(
                        EnvironmentalNetworkSourceEventCauseKind.FUTURE_DEMAND,
                        head.stream_id,
                        head.vessel_id,
                    ),
                )
            )

    next_event: EnvironmentalNetworkSourceEventProposal | None = None
    if candidates:
        earliest = min(item[0] for item in candidates)
        causes = tuple(cause for at, cause in candidates if abs(at - earliest) <= _EPS)
        next_event = _build_proposal(
            at=earliest,
            generation=generation,
            valid_until=frozen_until,
            causes=causes,
        )

    pending_after_current = current.reservoir.pending_demands
    awaiting_invalidation = bool(pending_after_current) and next_event is None
    return EnvironmentalNetworkSourceEventPlan(
        generation=generation,
        snapshot_time=snapshot.time,
        frozen_until=frozen_until,
        current_resolution=current,
        rechecks=tuple(sorted(rechecks, key=lambda item: (item.stream_id, item.vessel_id))),
        next_event=next_event,
        awaiting_invalidation=awaiting_invalidation,
    )
