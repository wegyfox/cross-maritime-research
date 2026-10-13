"""Qualified B4 earliest-safe conflict-slot coordination for PSP two-stream v1.

B4 is deliberately an eligibility controller above the frozen CROSS D0 source resolver.
It may delay a source head, but never admits a vessel, changes physical state, mutates
service order, or bypasses the canonical source/same-route safety machinery.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Callable, Iterable

from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import (
    EnvironmentalNetworkSourceControllerContext,
)
from vessel_crossing.nextgen.environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceDemand,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalNetworkSourceResolution,
)

from .contract import TwoStreamProfileContract
from .solver import (
    CERTIFIED_GRID_SLOT,
    PlannedCrossingMotion,
    SafeConflictSlotCertificate,
    SafeConflictSlotSolver,
    SafeSlotTable,
)

_EPS = 1e-10
B4_POLICY_ALGORITHM_ID = "cross-b4-earliest-safe-slot-v1"


class B4PolicyError(RuntimeError):
    """Raised when B4 would leave its qualified two-stream v1 domain."""


@dataclass(frozen=True, slots=True)
class B4CenterPassObservation:
    vessel_id: str
    stream_id: str
    observation_time_min: float
    heading_deg: float
    speed_nm_per_min: float

    def __post_init__(self) -> None:
        if not self.vessel_id or not self.stream_id:
            raise ValueError("observation vessel_id and stream_id must be non-empty")
        if not isfinite(self.observation_time_min):
            raise ValueError("observation_time_min must be finite")
        if not isfinite(self.heading_deg):
            raise ValueError("heading_deg must be finite")
        if not isfinite(self.speed_nm_per_min) or self.speed_nm_per_min <= 0.0:
            raise ValueError("speed_nm_per_min must be finite and positive")
        object.__setattr__(self, "heading_deg", float(self.heading_deg) % 360.0)
        object.__setattr__(self, "observation_time_min", float(self.observation_time_min))
        object.__setattr__(self, "speed_nm_per_min", float(self.speed_nm_per_min))


@dataclass(frozen=True, slots=True)
class B4PolicyTrace:
    time_min: float
    event: str
    vessel_id: str | None = None
    stream_id: str | None = None
    other_vessel_id: str | None = None
    center_target_min: float | None = None
    release_target_min: float | None = None
    solver_evidence_sha256: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class _Anchor:
    vessel_id: str
    stream_id: str
    center_time_min: float
    motion: PlannedCrossingMotion
    actual: bool


@dataclass(frozen=True, slots=True)
class _Reservation:
    vessel_id: str
    stream_id: str
    center_target_min: float
    release_target_min: float
    motion: PlannedCrossingMotion
    solver_certificate: SafeConflictSlotCertificate | None
    anchor_vessel_id: str | None


CenterObservationProvider = Callable[[], Iterable[B4CenterPassObservation]]


class B4EarliestSafeSlotController:
    """Profile-owned B4 controller; canonical D0 source resolution remains final authority."""

    def __init__(
        self,
        profile: TwoStreamProfileContract,
        *,
        center_observation_provider: CenterObservationProvider,
        slot_table: SafeSlotTable | None = None,
    ) -> None:
        if not isinstance(profile, TwoStreamProfileContract):
            raise TypeError("profile must be TwoStreamProfileContract")
        if not callable(center_observation_provider):
            raise TypeError("center_observation_provider must be callable")
        self.profile = profile
        self.center_observation_provider = center_observation_provider
        self.slot_table = slot_table if slot_table is not None else SafeSlotTable()
        self.solver = SafeConflictSlotSolver(profile.slot_solver, table=self.slot_table)
        self.trace: list[B4PolicyTrace] = []
        self._baseline_gate: dict[str, float | None] = {}
        self._demand_identity: dict[str, EnvironmentalNetworkSourceDemand] = {}
        self._service_order: tuple[tuple[str, tuple[str, ...]], ...] | None = None
        self._motion_by_vessel: dict[str, PlannedCrossingMotion] = {}
        self._reservation: _Reservation | None = None
        self._inflight: dict[str, _Anchor] = {}
        self._last_actual: _Anchor | None = None
        self._seen_observations: set[tuple[str, float]] = set()

    @property
    def reservation(self) -> _Reservation | None:
        return self._reservation

    def _record(self, time_min: float, event: str, **kwargs: object) -> None:
        self.trace.append(B4PolicyTrace(time_min=float(time_min), event=event, **kwargs))

    @staticmethod
    def _time(context: EnvironmentalNetworkSourceControllerContext) -> float:
        value = float(getattr(context.snapshot, "time"))
        if not isfinite(value):
            raise B4PolicyError("controller snapshot time must be finite")
        return value

    @staticmethod
    def _heading_angle(a: float, b: float) -> float:
        raw = abs((a - b) % 360.0)
        return min(raw, 360.0 - raw)

    def _stream_geometry(self, stream_id: str):
        for row in self.profile.streams:
            if row.stream_id == stream_id:
                return row
        raise B4PolicyError(f"demand stream {stream_id!r} is outside profile contract")

    def _motion(self, demand: EnvironmentalNetworkSourceDemand) -> PlannedCrossingMotion:
        cached = self._motion_by_vessel.get(demand.vessel_id)
        if cached is not None:
            return cached
        dx, dy = demand.environmental_drift_velocity
        if abs(dx) > _EPS or abs(dy) > _EPS:
            raise B4PolicyError("B4 planned-slot v1 requires zero environmental drift")
        if demand.profile is None or demand.profile.safety_domain is None:
            raise B4PolicyError("B4 requires an explicit incoming VesselClassProfile safety domain")
        speed = float(demand.vessel.through_water_speed)
        if not isfinite(speed) or speed <= 0.0:
            raise B4PolicyError("B4 requires positive constant nominal source speed")
        geometry = self._stream_geometry(demand.stream_id)
        motion = PlannedCrossingMotion(
            stream_id=demand.stream_id,
            heading_deg=float(demand.vessel.through_water_heading_deg),
            speed_nm_per_min=speed,
            source_to_center_distance_nm=geometry.source_to_center_distance_nm,
            center_to_exit_distance_nm=geometry.center_to_exit_distance_nm,
            ship_domain=demand.profile.safety_domain,
            profile_physics_identity=f"{demand.stream_id}:{demand.profile.class_id}",
        )
        self._motion_by_vessel[demand.vessel_id] = motion
        return motion

    def _validate_demand_identity_and_domain(self, reservoir: EnvironmentalNetworkSourceReservoir) -> None:
        current_ids = {d.vessel_id for d in reservoir.demands}
        if self._demand_identity and current_ids != set(self._demand_identity):
            raise B4PolicyError("B4 demand set changed after initialization")
        for demand in reservoir.demands:
            identity = replace(demand, release_not_before_time=None)
            prior = self._demand_identity.get(demand.vessel_id)
            if prior is None:
                self._demand_identity[demand.vessel_id] = identity
                self._baseline_gate[demand.vessel_id] = demand.release_not_before_time
            elif identity != prior:
                raise B4PolicyError("B4 demand physical/stochastic identity changed")
            self._motion(demand)

        stream_headings: dict[str, float] = {}
        for demand in reservoir.demands:
            heading = self._motion(demand).heading_deg
            prior = stream_headings.setdefault(demand.stream_id, heading)
            if self._heading_angle(prior, heading) > 1e-9:
                raise B4PolicyError("B4 v1 requires a fixed straight heading within each stream")
        if set(stream_headings) != {row.stream_id for row in self.profile.streams}:
            raise B4PolicyError("B4 reservoir streams must exactly match the two profile streams")
        h = tuple(stream_headings[s.stream_id] for s in self.profile.streams)
        if abs(self._heading_angle(h[0], h[1]) - 90.0) > 1e-9:
            raise B4PolicyError("B4 v1 requires orthogonal stream headings")

    def _sync_and_validate(self, reservoir: EnvironmentalNetworkSourceReservoir) -> None:
        if self._service_order is None:
            self._service_order = reservoir.service_order_by_stream
        elif reservoir.service_order_by_stream != self._service_order:
            raise B4PolicyError("B4 may not mutate or accept mutated source service order")
        self._validate_demand_identity_and_domain(reservoir)

    def _slot_cost_for_demand(
        self,
        demand: EnvironmentalNetworkSourceDemand,
        *,
        now: float,
        anchor: _Anchor | None,
    ) -> tuple[float, SafeConflictSlotCertificate | None]:
        motion = self._motion(demand)
        baseline = self._baseline_gate[demand.vessel_id]
        free_release = max(
            now,
            demand.demand_time,
            baseline if baseline is not None else demand.demand_time,
        )
        travel = motion.source_to_center_distance_nm / motion.speed_nm_per_min
        earliest = free_release + travel
        certificate: SafeConflictSlotCertificate | None = None
        if anchor is not None:
            if anchor.stream_id == demand.stream_id:
                # Same-route following remains canonical source/FIFO responsibility.
                earliest = max(earliest, anchor.center_time_min)
            else:
                certificate = self.solver.solve(anchor.motion, motion)
                if certificate.status != CERTIFIED_GRID_SLOT or certificate.separation_min is None:
                    raise B4PolicyError("no certified cross-stream B4 slot within declared search")
                earliest = max(earliest, anchor.center_time_min + certificate.separation_min)
        return earliest, certificate

    def _consume_center_observations(self, now: float, reservoir: EnvironmentalNetworkSourceReservoir) -> None:
        admitted_ids = set(reservoir.admitted_vessel_ids)
        for obs in self.center_observation_provider():
            if not isinstance(obs, B4CenterPassObservation):
                raise B4PolicyError("center observation provider returned invalid evidence")
            if obs.observation_time_min > now + _EPS:
                continue
            if obs.vessel_id not in admitted_ids:
                raise B4PolicyError("center observation requires a prior canonical source admission")
            key = (obs.vessel_id, obs.observation_time_min)
            if key in self._seen_observations:
                continue
            demand = self._demand_identity.get(obs.vessel_id)
            if demand is None:
                raise B4PolicyError(f"center observation references unknown demand {obs.vessel_id!r}")
            motion = self._motion(demand)
            if obs.stream_id != motion.stream_id:
                raise B4PolicyError("center observation stream disagrees with immutable demand")
            if self._heading_angle(obs.heading_deg, motion.heading_deg) > 1e-9:
                raise B4PolicyError("B4 v1 actual center heading left the planned straight-motion domain")
            if abs(obs.speed_nm_per_min - motion.speed_nm_per_min) > 1e-9:
                raise B4PolicyError("B4 v1 actual center speed left the planned constant-speed domain")
            self._seen_observations.add(key)
            self._inflight.pop(obs.vessel_id, None)
            anchor = _Anchor(
                vessel_id=obs.vessel_id,
                stream_id=obs.stream_id,
                center_time_min=obs.observation_time_min,
                motion=motion,
                actual=True,
            )
            if self._last_actual is None or (anchor.center_time_min, anchor.vessel_id) > (
                self._last_actual.center_time_min,
                self._last_actual.vessel_id,
            ):
                self._last_actual = anchor
            self._record(
                now,
                "B4_ACTUAL_CENTER_REANCHOR",
                vessel_id=obs.vessel_id,
                stream_id=obs.stream_id,
                center_target_min=obs.observation_time_min,
            )

    def _effective_anchor(self) -> _Anchor | None:
        candidates = list(self._inflight.values())
        if self._last_actual is not None:
            candidates.append(self._last_actual)
        if not candidates:
            return None
        return max(candidates, key=lambda x: (x.center_time_min, x.vessel_id))

    @staticmethod
    def _pending_heads(reservoir: EnvironmentalNetworkSourceReservoir):
        by_stream: dict[str, list[EnvironmentalNetworkSourceDemand]] = {}
        for demand in reservoir.pending_demands_in_service_order:
            by_stream.setdefault(demand.stream_id, []).append(demand)
        return {stream: rows[0] for stream, rows in by_stream.items() if rows}

    def _new_reservation(
        self,
        reservoir: EnvironmentalNetworkSourceReservoir,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> _Reservation | None:
        now = self._time(context)
        heads = self._pending_heads(reservoir)
        due_heads = [d for d in heads.values() if d.demand_time <= now + _EPS]
        if not due_heads:
            return None
        anchor = self._effective_anchor()
        candidates: list[tuple[float, float, str, str, EnvironmentalNetworkSourceDemand, SafeConflictSlotCertificate | None]] = []
        for demand in due_heads:
            earliest, certificate = self._slot_cost_for_demand(demand, now=now, anchor=anchor)
            candidates.append((earliest, demand.demand_time, demand.stream_id, demand.vessel_id, demand, certificate))
        center, _, _, _, demand, certificate = min(candidates, key=lambda row: row[:4])
        motion = self._motion(demand)
        travel = motion.source_to_center_distance_nm / motion.speed_nm_per_min
        baseline = self._baseline_gate[demand.vessel_id]
        release = max(now, demand.demand_time, center - travel, baseline if baseline is not None else demand.demand_time)
        reservation = _Reservation(
            vessel_id=demand.vessel_id,
            stream_id=demand.stream_id,
            center_target_min=center,
            release_target_min=release,
            motion=motion,
            solver_certificate=certificate,
            anchor_vessel_id=None if anchor is None else anchor.vessel_id,
        )
        self._record(
            now,
            "B4_SLOT_RESERVED",
            vessel_id=demand.vessel_id,
            stream_id=demand.stream_id,
            other_vessel_id=reservation.anchor_vessel_id,
            center_target_min=center,
            release_target_min=release,
            solver_evidence_sha256=None if certificate is None else certificate.evidence_sha256,
        )
        return reservation

    def _apply_gates(
        self,
        reservoir: EnvironmentalNetworkSourceReservoir,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> EnvironmentalNetworkSourceReservoir:
        reservation = self._reservation
        if reservation is None:
            return reservoir
        now = self._time(context)
        pending_ids = {d.vessel_id for d in reservoir.pending_demands}
        if reservation.vessel_id not in pending_ids:
            self._reservation = None
            return reservoir
        competing_heads = set(d.vessel_id for d in self._pending_heads(reservoir).values())
        changed: list[EnvironmentalNetworkSourceDemand] = []
        for demand in reservoir.demands:
            baseline = self._baseline_gate[demand.vessel_id]
            gate = baseline
            if demand.vessel_id == reservation.vessel_id:
                desired = reservation.release_target_min
                if baseline is None and desired <= demand.demand_time + _EPS:
                    gate = None
                else:
                    gate = max(demand.demand_time, desired, baseline if baseline is not None else demand.demand_time)
            elif demand.vessel_id in competing_heads and demand.demand_time <= now + _EPS:
                hold = reservation.center_target_min
                gate = max(demand.demand_time, hold, baseline if baseline is not None else demand.demand_time)
            changed.append(replace(demand, release_not_before_time=gate))
        candidate = EnvironmentalNetworkSourceReservoir(
            demands=tuple(changed),
            admissions=reservoir.admissions,
            service_order_by_stream=reservoir.service_order_by_stream,
        )
        return candidate

    def prepare_reservoir(
        self,
        reservoir: EnvironmentalNetworkSourceReservoir,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> EnvironmentalNetworkSourceReservoir:
        self._sync_and_validate(reservoir)
        now = self._time(context)
        self._consume_center_observations(now, reservoir)
        pending_ids = {d.vessel_id for d in reservoir.pending_demands}
        if self._reservation is not None and self._reservation.vessel_id not in pending_ids:
            self._reservation = None
        if self._reservation is None:
            self._reservation = self._new_reservation(reservoir, context)
        return self._apply_gates(reservoir, context)

    def observe_resolution(
        self,
        resolution: EnvironmentalNetworkSourceResolution,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> None:
        reservation = self._reservation
        if reservation is None:
            return
        now = float(resolution.snapshot_time)
        admitted = next((a for a in resolution.admissions if a.vessel_id == reservation.vessel_id), None)
        if admitted is not None:
            travel = reservation.motion.source_to_center_distance_nm / reservation.motion.speed_nm_per_min
            predicted_center = admitted.admission_time + travel
            self._inflight[reservation.vessel_id] = _Anchor(
                vessel_id=reservation.vessel_id,
                stream_id=reservation.stream_id,
                center_time_min=predicted_center,
                motion=reservation.motion,
                actual=False,
            )
            self._record(
                now,
                "B4_CANONICAL_ADMISSION_COMMITTED",
                vessel_id=reservation.vessel_id,
                stream_id=reservation.stream_id,
                center_target_min=predicted_center,
                release_target_min=reservation.release_target_min,
                solver_evidence_sha256=None if reservation.solver_certificate is None else reservation.solver_certificate.evidence_sha256,
                detail="center re-anchored from actual canonical admission time",
            )
            self._reservation = None
            return
        if now + _EPS >= reservation.release_target_min:
            decision = next((d for d in resolution.decisions if d.vessel_id == reservation.vessel_id), None)
            self._record(
                now,
                "B4_SOURCE_DENIAL_REPLAN",
                vessel_id=reservation.vessel_id,
                stream_id=reservation.stream_id,
                center_target_min=reservation.center_target_min,
                release_target_min=reservation.release_target_min,
                solver_evidence_sha256=None if reservation.solver_certificate is None else reservation.solver_certificate.evidence_sha256,
                detail=None if decision is None else decision.status.value,
            )
            self._reservation = None


def b4_trace_as_dicts(trace: Iterable[B4PolicyTrace]) -> tuple[dict[str, object], ...]:
    return tuple({
        "time_min": row.time_min,
        "event": row.event,
        "vessel_id": row.vessel_id,
        "stream_id": row.stream_id,
        "other_vessel_id": row.other_vessel_id,
        "center_target_min": row.center_target_min,
        "release_target_min": row.release_target_min,
        "solver_evidence_sha256": row.solver_evidence_sha256,
        "detail": row.detail,
    } for row in trace)
