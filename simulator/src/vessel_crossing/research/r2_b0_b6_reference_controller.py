"""Frozen B4/B5 source-management controller for the R2 B0--B6 reference ladder.

This module is deliberately a research-layer adapter above the qualified FIFO source
runtime.  It may delay source eligibility and (for B5 only) apply one bounded adjacent
same-stream staging swap.  It never changes vessel identity, route, plant, maneuver
limits, safety domains, the pair-context mechanism, or the canonical physical source
clearance gate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, isfinite, pi, sin
from typing import Callable, Iterable, Mapping, Sequence

from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import (
    EnvironmentalNetworkSourceControllerContext,
)
from vessel_crossing.nextgen.environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceDemand,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalNetworkSourceResolution,
    resolve_environmental_network_source_reservoir,
)
from vessel_crossing.nextgen.ship_domain import ShipDomain, evaluate_pair_ship_domain
from vessel_crossing.research.r1_p4d_conflict_observer import P4DConflictObservation

_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class R2ManagedReferenceConfig:
    """Frozen current-R2 management semantics used by B4 and B5."""

    level: str
    delta_arm: str
    room_nm: float = 1.375
    cruise_speed_nm_per_min: float = 0.10
    slot_time_tolerance_min: float = 0.01
    max_slot_search_min: float = 27.50
    staging_depth: int = 2
    max_bypass_per_demand: int = 1

    def __post_init__(self) -> None:
        if self.level not in {"B4", "B5"}:
            raise ValueError("level must be B4 or B5")
        if self.delta_arm not in {"NEG050", "ZERO", "POS050"}:
            raise ValueError("unsupported R2 delta arm")
        for name in (
            "room_nm",
            "cruise_speed_nm_per_min",
            "slot_time_tolerance_min",
            "max_slot_search_min",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.staging_depth != 2:
            raise ValueError("current frozen B5 staging depth must equal 2")
        if self.max_bypass_per_demand != 1:
            raise ValueError("current frozen B5 bypass cap must equal 1")

    @property
    def source_to_center_free_flow_min(self) -> float:
        return self.room_nm / self.cruise_speed_nm_per_min


@dataclass(frozen=True, slots=True)
class R2ManagedReferenceTrace:
    time_min: float
    event: str
    vessel_id: str | None = None
    stream_id: str | None = None
    other_vessel_id: str | None = None
    center_target_min: float | None = None
    release_target_min: float | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class _SlotAnchor:
    vessel_id: str
    stream_id: str
    center_time_min: float
    heading_deg: float
    domain: ShipDomain


@dataclass(frozen=True, slots=True)
class _Reservation:
    anchor: _SlotAnchor
    release_time_min: float


CenterObservationProvider = Callable[[], Sequence[P4DConflictObservation]]


def _velocity(*, heading_deg: float, speed_nm_per_min: float) -> tuple[float, float]:
    theta = heading_deg * pi / 180.0
    return (speed_nm_per_min * sin(theta), speed_nm_per_min * cos(theta))


def _is_circular(domain: ShipDomain) -> bool:
    values = (domain.ahead_nm, domain.astern_nm, domain.port_nm, domain.starboard_nm)
    return max(values) - min(values) <= 1e-12


def _closest_free_flow_relative_position(
    *,
    predecessor_heading_deg: float,
    successor_heading_deg: float,
    separation_min: float,
    speed_nm_per_min: float,
) -> tuple[float, float]:
    """Relative position p_successor - p_predecessor at closest approach.

    The predecessor crosses the conflict center at local time 0 and the successor at
    ``separation_min``.  Current R2 uses equal constant free-flow speed and circular
    ship domains for this slot-oracle derivation.  The closest point is solved exactly
    from the two linear free-flow trajectories; domain clearance itself is then checked
    by the canonical ship-domain evaluator.
    """

    v1 = _velocity(heading_deg=predecessor_heading_deg, speed_nm_per_min=speed_nm_per_min)
    v2 = _velocity(heading_deg=successor_heading_deg, speed_nm_per_min=speed_nm_per_min)
    # r(t) = v2*(t-separation) - v1*t = a*t + b.
    a = (v2[0] - v1[0], v2[1] - v1[1])
    b = (-v2[0] * separation_min, -v2[1] * separation_min)
    denom = a[0] * a[0] + a[1] * a[1]
    if denom <= _EPS:
        t_closest = 0.0
    else:
        unconstrained = -(a[0] * b[0] + a[1] * b[1]) / denom
        t_closest = min(max(unconstrained, 0.0), separation_min)
    return (a[0] * t_closest + b[0], a[1] * t_closest + b[1])


def derive_r2_free_flow_slot_separation_min(
    *,
    predecessor_heading_deg: float,
    successor_heading_deg: float,
    predecessor_domain: ShipDomain,
    successor_domain: ShipDomain,
    speed_nm_per_min: float = 0.10,
    time_tolerance_min: float = 0.01,
    max_search_min: float = 27.50,
) -> float:
    """Return first 0.01-min grid separation with strictly positive domain margin.

    This is not an empirical/reference headway.  It is an internal deterministic oracle
    derived from the frozen R2 free-flow geometry/speed and the same canonical
    ship-domain semantics used by source admission.  Current R2 domains are circular;
    fail closed if a future configuration tries to reuse this oracle with directional
    domains without a separately frozen continuous minimizer.
    """

    if not _is_circular(predecessor_domain) or not _is_circular(successor_domain):
        raise ValueError("R2 B4 slot oracle is frozen for the current circular domains only")
    if not isfinite(speed_nm_per_min) or speed_nm_per_min <= 0.0:
        raise ValueError("speed_nm_per_min must be finite and positive")
    if not isfinite(time_tolerance_min) or time_tolerance_min <= 0.0:
        raise ValueError("time_tolerance_min must be finite and positive")
    if not isfinite(max_search_min) or max_search_min <= 0.0:
        raise ValueError("max_search_min must be finite and positive")

    steps = int(max_search_min / time_tolerance_min + _EPS)
    for step in range(1, steps + 1):
        separation = step * time_tolerance_min
        relative = _closest_free_flow_relative_position(
            predecessor_heading_deg=predecessor_heading_deg,
            successor_heading_deg=successor_heading_deg,
            separation_min=separation,
            speed_nm_per_min=speed_nm_per_min,
        )
        clearance = evaluate_pair_ship_domain(
            first_heading_deg=predecessor_heading_deg,
            second_heading_deg=successor_heading_deg,
            relative_position=relative,
            first_domain=predecessor_domain,
            second_domain=successor_domain,
        )
        if not clearance.intrudes and clearance.actual_distance_nm > clearance.required_distance_nm:
            return separation
    raise RuntimeError("no physically clear R2 free-flow center-slot separation found")


class R2B4B5SourceManagementController:
    """Online B4/B5 controller implementing the formally frozen reference ladder."""

    def __init__(
        self,
        *,
        config: R2ManagedReferenceConfig,
        type_by_vessel: Mapping[str, str],
        center_observation_provider: CenterObservationProvider | None = None,
    ) -> None:
        self.config = config
        self._type_by_vessel = dict(type_by_vessel)
        self._center_observation_provider = center_observation_provider
        self._service_order_by_stream: dict[str, list[str]] = {}
        self._bypass_count: dict[str, int] = {}
        self._last_admitted_type_by_stream: dict[str, str] = {}
        self._inflight_slot_anchors: dict[str, _SlotAnchor] = {}
        self._last_actual_anchor: _SlotAnchor | None = None
        self._reservation: _Reservation | None = None
        self._seen_center_passes: set[tuple[str, float]] = set()
        self._trace: list[R2ManagedReferenceTrace] = []

    @property
    def trace(self) -> tuple[R2ManagedReferenceTrace, ...]:
        return tuple(self._trace)

    @property
    def bypass_count(self) -> Mapping[str, int]:
        return dict(self._bypass_count)

    def _record(self, at: float, event: str, **kwargs: object) -> None:
        self._trace.append(R2ManagedReferenceTrace(float(at), event, **kwargs))

    def _sync_service_order(self, reservoir: EnvironmentalNetworkSourceReservoir) -> None:
        explicit = dict(reservoir.service_order_by_stream)
        by_stream: dict[str, list[str]] = {}
        for demand in reservoir.demands:
            by_stream.setdefault(demand.stream_id, []).append(demand.vessel_id)
        for stream_id, canonical in by_stream.items():
            incoming = list(explicit.get(stream_id, tuple(canonical)))
            existing = self._service_order_by_stream.get(stream_id)
            if existing is None:
                self._service_order_by_stream[stream_id] = incoming
            elif set(existing) != set(incoming):
                # Demands are exogenous and immutable for a campaign point.  Fail closed
                # rather than silently rebuilding a policy state around drifted demand.
                raise RuntimeError("managed source demand identity changed after controller start")

    def _consume_center_pass_observations(self, reservoir: EnvironmentalNetworkSourceReservoir) -> None:
        if self._center_observation_provider is None:
            return
        demand_by_id = {d.vessel_id: d for d in reservoir.demands}
        rows = sorted(
            self._center_observation_provider(),
            key=lambda row: (row.observation_time_min, row.vessel_id, row.role),
        )
        for row in rows:
            if row.role != "CONFLICT_CENTER_PASS":
                continue
            key = (row.vessel_id, float(row.observation_time_min))
            if key in self._seen_center_passes:
                continue
            self._seen_center_passes.add(key)
            demand = demand_by_id.get(row.vessel_id)
            if demand is None or demand.profile is None or demand.profile.safety_domain is None:
                continue
            anchor = _SlotAnchor(
                vessel_id=row.vessel_id,
                stream_id=row.stream_id,
                center_time_min=float(row.observation_time_min),
                heading_deg=float(row.through_water_heading_deg),
                domain=demand.profile.safety_domain,
            )
            self._inflight_slot_anchors.pop(row.vessel_id, None)
            if (
                self._last_actual_anchor is None
                or anchor.center_time_min >= self._last_actual_anchor.center_time_min - _EPS
            ):
                self._last_actual_anchor = anchor
            self._record(
                anchor.center_time_min,
                "ACTUAL_CENTER_PASS_REANCHOR",
                vessel_id=anchor.vessel_id,
                stream_id=anchor.stream_id,
                center_target_min=anchor.center_time_min,
            )

    def _effective_anchor(self) -> _SlotAnchor | None:
        candidates = list(self._inflight_slot_anchors.values())
        if self._last_actual_anchor is not None:
            candidates.append(self._last_actual_anchor)
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.center_time_min, item.vessel_id))

    def _pending_by_stream(
        self, reservoir: EnvironmentalNetworkSourceReservoir
    ) -> dict[str, list[EnvironmentalNetworkSourceDemand]]:
        pending_by_id = {item.vessel_id: item for item in reservoir.pending_demands}
        out: dict[str, list[EnvironmentalNetworkSourceDemand]] = {}
        for stream_id, order in sorted(self._service_order_by_stream.items()):
            rows = [pending_by_id[vessel_id] for vessel_id in order if vessel_id in pending_by_id]
            if rows:
                out[stream_id] = rows
        return out

    @staticmethod
    def _canonical_physical_admissible(
        demand: EnvironmentalNetworkSourceDemand,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> bool:
        candidate = replace(demand, release_not_before_time=None)
        temporary = EnvironmentalNetworkSourceReservoir((candidate,))
        resolution = resolve_environmental_network_source_reservoir(
            temporary,
            context.snapshot,
            context.active_profiles,
            same_route_dynamic_ingress=context.same_route_dynamic_ingress,
            future_ingress_clearance=context.future_ingress_clearance,
            active_route_targets=context.active_route_targets,
            active_limits=context.active_limits,
            active_environmental_drifts=context.active_environmental_drifts,
            active_commands=context.active_commands,
            active_motion_backend=context.active_motion_backend,
            active_motion_states=context.active_motion_states,
        )
        return any(item.vessel_id == demand.vessel_id for item in resolution.admissions)

    def _maybe_stage_b5(
        self,
        reservoir: EnvironmentalNetworkSourceReservoir,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> None:
        if self.config.level != "B5" or self.config.delta_arm == "ZERO":
            return
        now = float(getattr(context.snapshot, "time"))
        pending = self._pending_by_stream(reservoir)
        for stream_id in sorted(pending):
            rows = pending[stream_id]
            if len(rows) < self.config.staging_depth:
                continue
            first, second = rows[0], rows[1]
            if first.demand_time > now + _EPS or second.demand_time > now + _EPS:
                continue
            if self._bypass_count.get(first.vessel_id, 0) >= self.config.max_bypass_per_demand:
                continue
            anchor_type = self._last_admitted_type_by_stream.get(stream_id)
            if anchor_type is None:
                continue
            first_type = self._type_by_vessel.get(first.vessel_id)
            second_type = self._type_by_vessel.get(second.vessel_id)
            if first_type is None or second_type is None:
                raise RuntimeError("B5 requires a type label for every staged demand")
            if self.config.delta_arm == "NEG050":
                first_preferred = first_type != anchor_type
                second_preferred = second_type != anchor_type
            else:  # POS050
                first_preferred = first_type == anchor_type
                second_preferred = second_type == anchor_type
            if first_preferred or not second_preferred:
                continue
            if not self._canonical_physical_admissible(first, context):
                continue
            if not self._canonical_physical_admissible(second, context):
                continue

            order = self._service_order_by_stream[stream_id]
            first_index = order.index(first.vessel_id)
            second_index = order.index(second.vessel_id)
            if second_index != first_index + 1:
                raise RuntimeError("B5 bounded staging candidates ceased to be adjacent")
            order[first_index], order[second_index] = order[second_index], order[first_index]
            self._bypass_count[first.vessel_id] = self._bypass_count.get(first.vessel_id, 0) + 1
            self._record(
                now,
                "B5_ADJACENT_SWAP",
                vessel_id=second.vessel_id,
                other_vessel_id=first.vessel_id,
                stream_id=stream_id,
                detail=f"{anchor_type}->{second_type}; bypass_count={self._bypass_count[first.vessel_id]}",
            )

    def _slot_separation(self, anchor: _SlotAnchor, demand: EnvironmentalNetworkSourceDemand) -> float:
        domain = None if demand.profile is None else demand.profile.safety_domain
        if domain is None:
            raise RuntimeError("B4/B5 requires explicit incoming safety domain")
        return derive_r2_free_flow_slot_separation_min(
            predecessor_heading_deg=anchor.heading_deg,
            successor_heading_deg=float(demand.vessel.through_water_heading_deg),
            predecessor_domain=anchor.domain,
            successor_domain=domain,
            speed_nm_per_min=self.config.cruise_speed_nm_per_min,
            time_tolerance_min=self.config.slot_time_tolerance_min,
            max_search_min=self.config.max_slot_search_min,
        )

    def _new_reservation(
        self,
        reservoir: EnvironmentalNetworkSourceReservoir,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> _Reservation | None:
        now = float(getattr(context.snapshot, "time"))
        pending = self._pending_by_stream(reservoir)
        heads = [rows[0] for _, rows in sorted(pending.items()) if rows and rows[0].demand_time <= now + _EPS]
        if not heads:
            return None
        anchor = self._effective_anchor()
        candidates: list[tuple[float, float, str, str, EnvironmentalNetworkSourceDemand]] = []
        for demand in heads:
            earliest = now + self.config.source_to_center_free_flow_min
            if anchor is not None:
                earliest = max(earliest, anchor.center_time_min + self._slot_separation(anchor, demand))
            candidates.append((earliest, demand.demand_time, demand.stream_id, demand.vessel_id, demand))
        center_target, _, _, _, chosen = min(candidates, key=lambda row: row[:4])
        domain = None if chosen.profile is None else chosen.profile.safety_domain
        if domain is None:
            raise RuntimeError("B4/B5 requires explicit safety domain")
        release_target = center_target - self.config.source_to_center_free_flow_min
        reservation = _Reservation(
            anchor=_SlotAnchor(
                vessel_id=chosen.vessel_id,
                stream_id=chosen.stream_id,
                center_time_min=center_target,
                heading_deg=float(chosen.vessel.through_water_heading_deg),
                domain=domain,
            ),
            release_time_min=max(now, release_target),
        )
        self._record(
            now,
            "B4_SLOT_RESERVED",
            vessel_id=chosen.vessel_id,
            stream_id=chosen.stream_id,
            center_target_min=center_target,
            release_target_min=reservation.release_time_min,
        )
        return reservation

    def _apply_reservation_gates(
        self,
        reservoir: EnvironmentalNetworkSourceReservoir,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> EnvironmentalNetworkSourceReservoir:
        reservation = self._reservation
        if reservation is None:
            return reservoir
        now = float(getattr(context.snapshot, "time"))
        pending_ids = {item.vessel_id for item in reservoir.pending_demands}
        if reservation.anchor.vessel_id not in pending_ids:
            self._reservation = None
            return reservoir

        pending = self._pending_by_stream(reservoir)
        competing_heads = {rows[0].vessel_id for rows in pending.values() if rows}
        changed: list[EnvironmentalNetworkSourceDemand] = []
        for demand in reservoir.demands:
            gate = demand.release_not_before_time
            if demand.vessel_id == reservation.anchor.vessel_id:
                gate = max(demand.demand_time, reservation.release_time_min)
            elif demand.vessel_id in competing_heads and demand.demand_time <= now + _EPS:
                # Keep the non-selected head out of this exact resolution without
                # introducing an artificial short polling boundary.  A +0.01-min gate
                # here would pre-empt the canonical certified physical recheck after a
                # source denial and can create dense wake-up churn under saturation.
                # Holding to the selected center target is only temporary arbitration
                # evidence: immediately after admission/denial the controller replans
                # and rewrites this gate before the next physical source resolution.
                gate = max(demand.demand_time, reservation.anchor.center_time_min)
            elif demand.vessel_id in pending_ids:
                # Clear stale controller evidence on demands that are not current heads.
                gate = None
            changed.append(replace(demand, release_not_before_time=gate))
        return EnvironmentalNetworkSourceReservoir(
            demands=tuple(changed),
            admissions=reservoir.admissions,
            service_order_by_stream=tuple(
                (stream_id, tuple(order))
                for stream_id, order in sorted(self._service_order_by_stream.items())
            ),
        )

    def prepare_reservoir(
        self,
        reservoir: EnvironmentalNetworkSourceReservoir,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> EnvironmentalNetworkSourceReservoir:
        self._sync_service_order(reservoir)
        self._consume_center_pass_observations(reservoir)
        pending_ids = {item.vessel_id for item in reservoir.pending_demands}
        if self._reservation is not None and self._reservation.anchor.vessel_id not in pending_ids:
            self._reservation = None
        if self._reservation is None:
            self._maybe_stage_b5(reservoir, context)
            self._reservation = self._new_reservation(reservoir, context)
        return self._apply_reservation_gates(reservoir, context)

    def observe_resolution(
        self,
        resolution: EnvironmentalNetworkSourceResolution,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> None:
        now = float(resolution.snapshot_time)
        admitted = {item.vessel_id: item for item in resolution.admissions}
        for item in resolution.admissions:
            vessel_type = self._type_by_vessel.get(item.vessel_id)
            if vessel_type is not None:
                self._last_admitted_type_by_stream[item.stream_id] = vessel_type

        reservation = self._reservation
        if reservation is None:
            return
        vessel_id = reservation.anchor.vessel_id
        if vessel_id in admitted:
            self._inflight_slot_anchors[vessel_id] = reservation.anchor
            self._record(
                now,
                "B4_SLOT_ADMITTED_BY_CANONICAL_GATE",
                vessel_id=vessel_id,
                stream_id=reservation.anchor.stream_id,
                center_target_min=reservation.anchor.center_time_min,
                release_target_min=reservation.release_time_min,
            )
            self._reservation = None
            return

        if now + _EPS >= reservation.release_time_min:
            # The canonical gate had the final say.  Clear only the management
            # reservation; the normal source recheck scheduler owns the next physical
            # wake and prevents zero-time busy polling.
            decision = next((row for row in resolution.decisions if row.vessel_id == vessel_id), None)
            detail = None if decision is None else str(decision.status.value)
            self._record(
                now,
                "B4_SOURCE_DENIAL_REPLAN",
                vessel_id=vessel_id,
                stream_id=reservation.anchor.stream_id,
                center_target_min=reservation.anchor.center_time_min,
                release_target_min=reservation.release_time_min,
                detail=detail,
            )
            self._reservation = None


def controller_trace_as_dicts(
    trace: Iterable[R2ManagedReferenceTrace],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "time_min": row.time_min,
            "event": row.event,
            "vessel_id": row.vessel_id,
            "stream_id": row.stream_id,
            "other_vessel_id": row.other_vessel_id,
            "center_target_min": row.center_target_min,
            "release_target_min": row.release_target_min,
            "detail": row.detail,
        }
        for row in trace
    )
