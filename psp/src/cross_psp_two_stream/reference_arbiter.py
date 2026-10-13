"""Reference-only fair earliest-slot arbitration for the A69 B6 physical reference.

This module is deliberately separate from deployment B4/B5. It reuses A64 planned-slot
physics, but canonical frozen-D0 source admission remains the final physical authority.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import (
    EnvironmentalNetworkSourceControllerContext,
)
from vessel_crossing.nextgen.environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceDemand,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalNetworkSourceResolution,
)
from vessel_crossing.nextgen.vessel_classes import VesselClassProfile

from .contract import REFERENCE_ARBITER_CONTRACT_ID, TwoStreamProfileContract
from .solver import (
    CERTIFIED_GRID_SLOT,
    PlannedCrossingMotion,
    SafeConflictSlotCertificate,
    SafeConflictSlotSolver,
    SafeSlotTable,
)

REFERENCE_ARBITER_ALGORITHM_ID = "cross-b6-reference-fair-slot-arbiter-v1"
_EPS = 1e-10


class ReferenceArbiterError(RuntimeError):
    """Fail-closed reference-arbiter failure with an evidence-visible status code."""

    def __init__(self, status: str, detail: str | None = None) -> None:
        super().__init__(status if detail is None else f"{status}: {detail}")
        self.status = status
        self.detail = detail


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _domain_dict(domain) -> dict[str, float] | None:
    if domain is None:
        return None
    return {
        "ahead_nm": float(domain.ahead_nm),
        "astern_nm": float(domain.astern_nm),
        "port_nm": float(domain.port_nm),
        "starboard_nm": float(domain.starboard_nm),
    }


def _profile_physics(profile: VesselClassProfile) -> dict[str, object]:
    return {
        "class_id": profile.class_id,
        "dimensions": {
            "length_nm": float(profile.dimensions.length_nm),
            "beam_nm": float(profile.dimensions.beam_nm),
        },
        "maneuver_limits": {
            "max_acceleration": float(profile.maneuver_limits.max_acceleration),
            "max_deceleration": float(profile.maneuver_limits.max_deceleration),
            "max_turn_rate_deg_per_min": float(profile.maneuver_limits.max_turn_rate_deg_per_min),
        },
        "safety_domain": _domain_dict(profile.safety_domain),
        "speed_envelope": None if profile.speed_envelope is None else {
            "maximum_speed": float(profile.speed_envelope.maximum_speed),
        },
    }


def build_reference_stream_keys(
    profile: TwoStreamProfileContract,
    profile_classes: Mapping[str, VesselClassProfile],
) -> Mapping[str, str]:
    """Derive rename-stable semantic stream keys from A69 identity material."""
    shares = {x.stream_id: float(x.share) for x in profile.b6_reference.stream_shares}
    templates = {x.stream_id: x for x in profile.b6_reference.supply_templates}
    geometry = {x.stream_id: x for x in profile.streams}
    out: dict[str, str] = {}
    for stream_id in geometry:
        template = templates.get(stream_id)
        if template is None or not template.cycle:
            raise ReferenceArbiterError("FAIL_MISSING_REFERENCE_SUPPLY_TEMPLATE")
        used_profiles: dict[str, object] = {}
        cycle: list[dict[str, object]] = []
        for item in template.cycle:
            vessel_profile = profile_classes.get(item.profile_class_id)
            if vessel_profile is None or vessel_profile.safety_domain is None:
                raise ReferenceArbiterError("FAIL_MISSING_REFERENCE_SUPPLY_TEMPLATE")
            used_profiles[item.profile_class_id] = _profile_physics(vessel_profile)
            cycle.append({
                "profile_class_id": item.profile_class_id,
                "through_water_heading_deg": float(item.through_water_heading_deg),
                "through_water_speed_nm_per_min": float(item.through_water_speed_nm_per_min),
            })
        geo = geometry[stream_id]
        payload = {
            "route_model_id": profile.route_model_id,
            "planned_slot_prediction_scope_id": profile.planned_slot_prediction_scope_id,
            "environment_model_id": profile.environment_model_id,
            "environment_scope": "zero-drift-reference-v1",
            "source_to_center_distance_nm": float(geo.source_to_center_distance_nm),
            "center_to_exit_distance_nm": float(geo.center_to_exit_distance_nm),
            "target_share": shares[stream_id],
            "supply_template_cycle": cycle,
            "profile_physics": used_profiles,
        }
        out[stream_id] = _sha(payload)
    if len(set(out.values())) != len(out):
        raise ReferenceArbiterError("FAIL_NON_UNIQUE_REFERENCE_STREAM_KEY")
    return MappingProxyType(out)


@dataclass(frozen=True, slots=True)
class ReferenceCenterPassObservation:
    vessel_id: str
    stream_id: str
    observation_time_min: float
    heading_deg: float
    speed_nm_per_min: float

    def __post_init__(self) -> None:
        if not self.vessel_id or not self.stream_id:
            raise ValueError("center observation identities must be non-empty")
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
class _Anchor:
    vessel_id: str
    stream_id: str
    stream_key: str
    center_time_min: float
    motion: PlannedCrossingMotion
    actual: bool


@dataclass(frozen=True, slots=True)
class _Reservation:
    vessel_id: str
    stream_id: str
    stream_key: str
    sequence_index: int
    anchor_stream_key: str | None
    center_target_min: float
    release_target_min: float
    motion: PlannedCrossingMotion
    solver_certificate: SafeConflictSlotCertificate | None
    anchor_vessel_id: str | None


CenterObservationProvider = Callable[[], Iterable[ReferenceCenterPassObservation]]


class ReferenceFairSlotArbiter:
    """Greedy earliest-safe reference scheduler under the frozen A69 fair-share band."""

    def __init__(
        self,
        profile: TwoStreamProfileContract,
        *,
        center_observation_provider: CenterObservationProvider,
        reference_stream_keys_by_stream: Mapping[str, str],
        slot_table: SafeSlotTable | None = None,
    ) -> None:
        if not isinstance(profile, TwoStreamProfileContract):
            raise TypeError("profile must be TwoStreamProfileContract")
        if not callable(center_observation_provider):
            raise TypeError("center_observation_provider must be callable")
        keys = dict(reference_stream_keys_by_stream)
        expected = {x.stream_id for x in profile.streams}
        if set(keys) != expected or len(set(keys.values())) != 2 or any(len(x) != 64 for x in keys.values()):
            raise ReferenceArbiterError("FAIL_NON_UNIQUE_REFERENCE_STREAM_KEY")
        self.profile = profile
        self.center_observation_provider = center_observation_provider
        self.stream_keys = MappingProxyType(keys)
        self.stream_by_key = MappingProxyType({value: key for key, value in keys.items()})
        self.share_by_key = MappingProxyType({
            keys[row.stream_id]: float(row.share) for row in profile.b6_reference.stream_shares
        })
        self.solver = SafeConflictSlotSolver(profile.slot_solver, table=slot_table if slot_table is not None else SafeSlotTable())
        self.trace: list[dict[str, object]] = []
        self.commitments_by_key: dict[str, int] = {key: 0 for key in self.stream_by_key}
        self.max_positive_deficit_by_key: dict[str, float] = {key: 0.0 for key in self.stream_by_key}
        self._baseline_gate: dict[str, float | None] = {}
        self._demand_identity: dict[str, EnvironmentalNetworkSourceDemand] = {}
        self._sequence_index_by_vessel: dict[str, int] = {}
        self._service_order: tuple[tuple[str, tuple[str, ...]], ...] | None = None
        self._motion_by_vessel: dict[str, PlannedCrossingMotion] = {}
        self._reservation: _Reservation | None = None
        self._inflight: dict[str, _Anchor] = {}
        self._last_actual: _Anchor | None = None
        self._seen_observations: set[tuple[str, float]] = set()
        self.fairness_idle_episode_count = 0
        self.fairness_idle_duration_min = 0.0

    @property
    def algorithm_id(self) -> str:
        return REFERENCE_ARBITER_ALGORITHM_ID

    @property
    def contract_id(self) -> str:
        return REFERENCE_ARBITER_CONTRACT_ID

    @property
    def reservation(self) -> _Reservation | None:
        return self._reservation

    def cancel_reservation_for_cutoff(self) -> str | None:
        vessel_id = None if self._reservation is None else self._reservation.vessel_id
        self._reservation = None
        return vessel_id

    @staticmethod
    def _heading_angle(a: float, b: float) -> float:
        raw = abs((float(a) - float(b)) % 360.0)
        return min(raw, 360.0 - raw)

    @staticmethod
    def _time(context: EnvironmentalNetworkSourceControllerContext) -> float:
        value = float(context.snapshot.time)
        if not isfinite(value):
            raise ReferenceArbiterError("FAIL_CANONICAL_ADMISSION_STATE_MISMATCH", "non-finite runtime time")
        return value

    def _record(self, event: str, *, now: float, **fields: object) -> None:
        row = {"time_min": float(now), "event": event}
        row.update(fields)
        self.trace.append(row)

    def _geometry(self, stream_id: str):
        for row in self.profile.streams:
            if row.stream_id == stream_id:
                return row
        raise ReferenceArbiterError("FAIL_UNSUPPORTED_DOMAIN", f"unknown stream {stream_id!r}")

    def _motion(self, demand: EnvironmentalNetworkSourceDemand) -> PlannedCrossingMotion:
        cached = self._motion_by_vessel.get(demand.vessel_id)
        if cached is not None:
            return cached
        dx, dy = demand.environmental_drift_velocity
        if abs(dx) > _EPS or abs(dy) > _EPS:
            raise ReferenceArbiterError("FAIL_UNSUPPORTED_DOMAIN", "reference planned-slot v1 requires zero drift")
        if demand.profile is None or demand.profile.safety_domain is None:
            raise ReferenceArbiterError("FAIL_UNSUPPORTED_DOMAIN", "explicit safety domain required")
        speed = float(demand.vessel.through_water_speed)
        if not isfinite(speed) or speed <= 0.0:
            raise ReferenceArbiterError("FAIL_UNSUPPORTED_DOMAIN", "positive nominal speed required")
        geo = self._geometry(demand.stream_id)
        key = self.stream_keys[demand.stream_id]
        motion = PlannedCrossingMotion(
            stream_id=demand.stream_id,
            heading_deg=float(demand.vessel.through_water_heading_deg),
            speed_nm_per_min=speed,
            source_to_center_distance_nm=geo.source_to_center_distance_nm,
            center_to_exit_distance_nm=geo.center_to_exit_distance_nm,
            ship_domain=demand.profile.safety_domain,
            profile_physics_identity=f"reference-stream:{key}:profile:{demand.profile.class_id}",
        )
        self._motion_by_vessel[demand.vessel_id] = motion
        return motion

    def _sync_and_validate(self, reservoir: EnvironmentalNetworkSourceReservoir) -> None:
        if self._service_order is None:
            self._service_order = reservoir.service_order_by_stream
            explicit = dict(reservoir.service_order_by_stream)
            by_stream: dict[str, list[str]] = {}
            for demand in reservoir.demands:
                by_stream.setdefault(demand.stream_id, []).append(demand.vessel_id)
            for stream_id, canonical in by_stream.items():
                order = explicit.get(stream_id, tuple(canonical))
                for index, vessel_id in enumerate(order):
                    self._sequence_index_by_vessel[vessel_id] = index
        elif reservoir.service_order_by_stream != self._service_order:
            raise ReferenceArbiterError("FAIL_CANONICAL_ADMISSION_STATE_MISMATCH", "reference FIFO order mutated")

        stream_headings: dict[str, float] = {}
        for demand in reservoir.demands:
            if demand.stream_id not in self.stream_keys:
                raise ReferenceArbiterError("FAIL_UNSUPPORTED_DOMAIN", "reservoir stream outside profile")
            identity = replace(demand, release_not_before_time=None)
            prior = self._demand_identity.get(demand.vessel_id)
            if prior is None:
                self._demand_identity[demand.vessel_id] = identity
                self._baseline_gate[demand.vessel_id] = demand.release_not_before_time
            elif identity != prior:
                raise ReferenceArbiterError("FAIL_CANONICAL_ADMISSION_STATE_MISMATCH", "demand identity changed")
            heading = self._motion(demand).heading_deg
            old = stream_headings.setdefault(demand.stream_id, heading)
            if self._heading_angle(old, heading) > 1e-9:
                raise ReferenceArbiterError("FAIL_UNSUPPORTED_DOMAIN", "fixed stream heading required")
        if set(stream_headings) != {x.stream_id for x in self.profile.streams}:
            raise ReferenceArbiterError("FAIL_UNSUPPORTED_DOMAIN", "exactly two profile streams required")
        hs = [stream_headings[x.stream_id] for x in self.profile.streams]
        if abs(self._heading_angle(hs[0], hs[1]) - 90.0) > 1e-9:
            raise ReferenceArbiterError("FAIL_UNSUPPORTED_DOMAIN", "orthogonal headings required")

    @staticmethod
    def _pending_heads(reservoir: EnvironmentalNetworkSourceReservoir) -> dict[str, EnvironmentalNetworkSourceDemand]:
        by_stream: dict[str, list[EnvironmentalNetworkSourceDemand]] = {}
        for demand in reservoir.pending_demands_in_service_order:
            by_stream.setdefault(demand.stream_id, []).append(demand)
        return {stream: rows[0] for stream, rows in by_stream.items() if rows}

    def deficits(self) -> dict[str, float]:
        total = sum(self.commitments_by_key.values())
        out = {
            key: self.share_by_key[key] * total - self.commitments_by_key[key]
            for key in self.commitments_by_key
        }
        for key, value in out.items():
            self.max_positive_deficit_by_key[key] = max(self.max_positive_deficit_by_key[key], value)
        return out

    def projected_deficits(self, selected_key: str) -> dict[str, float]:
        total = sum(self.commitments_by_key.values()) + 1
        return {
            key: self.share_by_key[key] * total - (self.commitments_by_key[key] + (1 if key == selected_key else 0))
            for key in self.commitments_by_key
        }

    def fairness_feasible(self, selected_key: str) -> tuple[bool, dict[str, float]]:
        projected = self.projected_deficits(selected_key)
        bound = self.profile.b6_reference.max_share_deficit_passages + self.profile.b6_reference.fairness_comparison_tolerance
        return max(projected.values()) <= bound + _EPS, projected

    def _consume_center_observations(self, now: float, reservoir: EnvironmentalNetworkSourceReservoir) -> None:
        admitted_ids = set(reservoir.admitted_vessel_ids)
        for obs in self.center_observation_provider():
            if not isinstance(obs, ReferenceCenterPassObservation):
                raise ReferenceArbiterError("FAIL_CANONICAL_ADMISSION_STATE_MISMATCH", "invalid center observation type")
            if obs.observation_time_min > now + _EPS:
                continue
            if obs.vessel_id not in admitted_ids:
                raise ReferenceArbiterError("FAIL_CANONICAL_ADMISSION_STATE_MISMATCH", "center passage without prior admission")
            marker = (obs.vessel_id, obs.observation_time_min)
            if marker in self._seen_observations:
                continue
            demand = self._demand_identity.get(obs.vessel_id)
            if demand is None:
                raise ReferenceArbiterError("FAIL_CANONICAL_ADMISSION_STATE_MISMATCH", "unknown center passage vessel")
            motion = self._motion(demand)
            if obs.stream_id != demand.stream_id or self._heading_angle(obs.heading_deg, motion.heading_deg) > 1e-9 or abs(obs.speed_nm_per_min - motion.speed_nm_per_min) > 1e-9:
                raise ReferenceArbiterError("FAIL_UNSUPPORTED_DOMAIN", "actual center passage left planned-motion contract")
            self._seen_observations.add(marker)
            self._inflight.pop(obs.vessel_id, None)
            anchor = _Anchor(
                vessel_id=obs.vessel_id,
                stream_id=obs.stream_id,
                stream_key=self.stream_keys[obs.stream_id],
                center_time_min=obs.observation_time_min,
                motion=motion,
                actual=True,
            )
            if self._last_actual is None or (anchor.center_time_min, anchor.stream_key, anchor.vessel_id) > (
                self._last_actual.center_time_min, self._last_actual.stream_key, self._last_actual.vessel_id
            ):
                self._last_actual = anchor
            self._record("REFERENCE_ACTUAL_CENTER_REANCHOR", now=now, vessel_id=obs.vessel_id, stream_key=anchor.stream_key, center_time_min=anchor.center_time_min)

    def _effective_anchor(self) -> _Anchor | None:
        rows = list(self._inflight.values())
        if self._last_actual is not None:
            rows.append(self._last_actual)
        if not rows:
            return None
        return max(rows, key=lambda x: (x.center_time_min, x.stream_key, x.vessel_id))

    def _slot_cost(self, demand: EnvironmentalNetworkSourceDemand, *, now: float, anchor: _Anchor | None) -> tuple[float, SafeConflictSlotCertificate | None]:
        motion = self._motion(demand)
        baseline = self._baseline_gate[demand.vessel_id]
        free_release = max(now, demand.demand_time, baseline if baseline is not None else demand.demand_time)
        travel = motion.source_to_center_distance_nm / motion.speed_nm_per_min
        earliest = free_release + travel
        certificate = None
        if anchor is not None:
            if anchor.stream_id == demand.stream_id:
                earliest = max(earliest, anchor.center_time_min)
            else:
                certificate = self.solver.solve(anchor.motion, motion)
                if certificate.status != CERTIFIED_GRID_SLOT or certificate.separation_min is None:
                    raise ReferenceArbiterError("FAIL_NO_CERTIFIED_FAIR_SLOT_WITHIN_SEARCH")
                earliest = max(earliest, anchor.center_time_min + certificate.separation_min)
        return earliest, certificate

    def _new_reservation(self, reservoir: EnvironmentalNetworkSourceReservoir, context: EnvironmentalNetworkSourceControllerContext) -> _Reservation | None:
        now = self._time(context)
        heads = self._pending_heads(reservoir)
        due = [x for x in heads.values() if x.demand_time <= now + _EPS]
        if not due:
            return None
        current_deficits = self.deficits()
        feasible: list[tuple[EnvironmentalNetworkSourceDemand, str, dict[str, float]]] = []
        for demand in due:
            key = self.stream_keys[demand.stream_id]
            ok, projected = self.fairness_feasible(key)
            self._record(
                "REFERENCE_FAIRNESS_CANDIDATE",
                now=now,
                vessel_id=demand.vessel_id,
                stream_key=key,
                current_deficits=current_deficits,
                projected_deficits=projected,
                fairness_feasible=ok,
            )
            if ok:
                feasible.append((demand, key, projected))
        if not feasible:
            raise ReferenceArbiterError("FAIL_REFERENCE_FAIRNESS_INFEASIBLE")

        anchor = self._effective_anchor()
        candidates: list[tuple[float, float, str, int, EnvironmentalNetworkSourceDemand, SafeConflictSlotCertificate | None]] = []
        failures: list[str] = []
        for demand, key, _ in feasible:
            try:
                earliest, certificate = self._slot_cost(demand, now=now, anchor=anchor)
            except ReferenceArbiterError as exc:
                if exc.status != "FAIL_NO_CERTIFIED_FAIR_SLOT_WITHIN_SEARCH":
                    raise
                failures.append(key)
                self._record("REFERENCE_FAIR_SLOT_UNAVAILABLE", now=now, vessel_id=demand.vessel_id, stream_key=key)
                continue
            sequence_index = self._sequence_index_by_vessel[demand.vessel_id]
            candidates.append((earliest, -current_deficits[key], key, sequence_index, demand, certificate))
            self._record(
                "REFERENCE_FAIR_SLOT_EVALUATED",
                now=now,
                vessel_id=demand.vessel_id,
                stream_key=key,
                earliest_safe_center_time=earliest,
                current_deficit=current_deficits[key],
                solver_evidence_sha256=None if certificate is None else certificate.evidence_sha256,
            )
        if not candidates:
            raise ReferenceArbiterError("FAIL_NO_CERTIFIED_FAIR_SLOT_WITHIN_SEARCH", ",".join(sorted(failures)))

        center, _, key, sequence_index, demand, certificate = min(candidates, key=lambda row: row[:4])
        motion = self._motion(demand)
        travel = motion.source_to_center_distance_nm / motion.speed_nm_per_min
        baseline = self._baseline_gate[demand.vessel_id]
        release = max(now, demand.demand_time, center - travel, baseline if baseline is not None else demand.demand_time)
        reservation = _Reservation(
            vessel_id=demand.vessel_id,
            stream_id=demand.stream_id,
            stream_key=key,
            sequence_index=sequence_index,
            anchor_stream_key=None if anchor is None else anchor.stream_key,
            center_target_min=center,
            release_target_min=release,
            motion=motion,
            solver_certificate=certificate,
            anchor_vessel_id=None if anchor is None else anchor.vessel_id,
        )
        self._record(
            "REFERENCE_SLOT_RESERVED",
            now=now,
            vessel_id=demand.vessel_id,
            stream_key=key,
            sequence_index=sequence_index,
            anchor_stream_key=None if anchor is None else anchor.stream_key,
            center_target_min=center,
            release_target_min=release,
            current_deficits=current_deficits,
            selection_tuple=[center, -current_deficits[key], key, sequence_index],
            solver_evidence_sha256=None if certificate is None else certificate.evidence_sha256,
        )
        return reservation

    def _apply_gates(self, reservoir: EnvironmentalNetworkSourceReservoir, context: EnvironmentalNetworkSourceControllerContext) -> EnvironmentalNetworkSourceReservoir:
        reservation = self._reservation
        if reservation is None:
            return reservoir
        now = self._time(context)
        pending_ids = {d.vessel_id for d in reservoir.pending_demands}
        if reservation.vessel_id not in pending_ids:
            self._reservation = None
            return reservoir
        competing_heads = {d.vessel_id for d in self._pending_heads(reservoir).values()}
        rows: list[EnvironmentalNetworkSourceDemand] = []
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
                gate = max(demand.demand_time, reservation.center_target_min, baseline if baseline is not None else demand.demand_time)
            rows.append(replace(demand, release_not_before_time=gate))
        return EnvironmentalNetworkSourceReservoir(
            demands=tuple(rows), admissions=reservoir.admissions, service_order_by_stream=reservoir.service_order_by_stream
        )

    def prepare_reservoir(self, reservoir: EnvironmentalNetworkSourceReservoir, context: EnvironmentalNetworkSourceControllerContext) -> EnvironmentalNetworkSourceReservoir:
        self._sync_and_validate(reservoir)
        now = self._time(context)
        self._consume_center_observations(now, reservoir)
        pending_ids = {d.vessel_id for d in reservoir.pending_demands}
        if self._reservation is not None and self._reservation.vessel_id not in pending_ids:
            raise ReferenceArbiterError("FAIL_CANONICAL_ADMISSION_STATE_MISMATCH", "reservation disappeared without observation")
        if self._reservation is None:
            self._reservation = self._new_reservation(reservoir, context)
        return self._apply_gates(reservoir, context)

    def observe_resolution(self, resolution: EnvironmentalNetworkSourceResolution, context: EnvironmentalNetworkSourceControllerContext) -> None:
        reservation = self._reservation
        if reservation is None:
            if resolution.admissions:
                raise ReferenceArbiterError("FAIL_CANONICAL_ADMISSION_STATE_MISMATCH", "admission without reference reservation")
            return
        now = float(resolution.snapshot_time)
        unexpected = [x for x in resolution.admissions if x.vessel_id != reservation.vessel_id]
        if unexpected:
            raise ReferenceArbiterError("FAIL_CANONICAL_ADMISSION_STATE_MISMATCH", "canonical source admitted non-selected reference head")
        admitted = next((x for x in resolution.admissions if x.vessel_id == reservation.vessel_id), None)
        if admitted is not None:
            key = reservation.stream_key
            self.commitments_by_key[key] += 1
            travel = reservation.motion.source_to_center_distance_nm / reservation.motion.speed_nm_per_min
            predicted_center = admitted.admission_time + travel
            self._inflight[reservation.vessel_id] = _Anchor(
                vessel_id=reservation.vessel_id,
                stream_id=reservation.stream_id,
                stream_key=key,
                center_time_min=predicted_center,
                motion=reservation.motion,
                actual=False,
            )
            deficits = self.deficits()
            self._record(
                "REFERENCE_CANONICAL_ADMISSION_COMMITTED",
                now=now,
                vessel_id=reservation.vessel_id,
                stream_key=key,
                center_target_min=predicted_center,
                commitments=dict(self.commitments_by_key),
                deficits=deficits,
            )
            self._reservation = None
            return
        if now + _EPS >= reservation.release_target_min:
            decision = next((x for x in resolution.decisions if x.vessel_id == reservation.vessel_id), None)
            self._record(
                "REFERENCE_SOURCE_DENIAL_REPLAN",
                now=now,
                vessel_id=reservation.vessel_id,
                stream_key=reservation.stream_key,
                detail=None if decision is None else decision.status.value,
            )
            self._reservation = None

    def semantic_trace(self) -> tuple[dict[str, object], ...]:
        """Trace stripped of display names; suitable for rename-invariance hashing."""
        rows: list[dict[str, object]] = []
        for item in self.trace:
            row = {
                key: value
                for key, value in item.items()
                if key not in {"vessel_id", "solver_evidence_sha256"}
            }
            rows.append(row)
        rows.sort(
            key=lambda row: (
                float(row.get("time_min", 0.0)),
                str(row.get("event", "")),
                str(row.get("stream_key", "")),
                int(row.get("sequence_index", -1)),
                _canonical_bytes(row),
            )
        )
        return tuple(rows)
