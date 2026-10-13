"""A70 B6 v2: balanced saturated physical reference with the A69 reference arbiter."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import cos, pi, sin
from statistics import mean, median
from types import MappingProxyType
from typing import Mapping, Sequence

from vessel_crossing.nextgen import (
    CrossTrackRecovery,
    EnvironmentalNetworkSourceDemand,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalRouteHandoffTarget,
    Route2D,
    RouteEndAction,
    RouteFollower,
    RouteNavigator,
    VesselClassProfile,
    VesselProfileCatalog,
    WaterRelativeVesselState,
)
from vessel_crossing.nextgen.encounter_registry import EncounterRegistry
from vessel_crossing.nextgen.environmental_candidate_transition import build_environmental_candidate_transition
from vessel_crossing.nextgen.environmental_network_paired_experiment import _MetricEnvironmentalWorld
from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import (
    EnvironmentalNetworkSourceControllerContext,
    EnvironmentalNetworkSourceFIFOTerminalStatus,
    start_environmental_network_source_fifo_runtime,
)

from .contract import TwoStreamProfileContract
from .reference_arbiter import (
    REFERENCE_ARBITER_ALGORITHM_ID,
    ReferenceArbiterError,
    ReferenceCenterPassObservation,
    ReferenceFairSlotArbiter,
    build_reference_stream_keys,
)
from .solver import SOLVER_ALGORITHM_ID

B6_REFERENCE_ALGORITHM_ID = "cross-b6-balanced-saturated-physical-reference-v2"
B6_EVIDENCE_SCHEMA = "cross-b6-balanced-saturated-reference-evidence-v2"
B6_COORDINATE_CONVENTION_ID = "conflict-center-origin-marine-heading-vector-v1"
B6_WINDOW_CONVENTION_ID = "warmup-measure-cutoff-drain-half-open-v1"
B6_CONVERGENCE_CONVENTION_ID = "nominal-double-measurement-double-warmup-v1"
B6_HEADWAY_QUANTILE_CONVENTION_ID = "linear-order-statistic-interpolation-v1"
B6_SEMANTIC_NORMALIZATION_ID = "reference-stream-key-display-name-normalization-v1"
_EPS = 1e-9


class B6ReferenceError(RuntimeError):
    """Raised when the reference cannot be constructed inside the frozen A69 contract."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _heading_unit(heading_deg: float) -> tuple[float, float]:
    rad = heading_deg * pi / 180.0
    return (sin(rad), cos(rad))


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    rows = sorted(float(x) for x in values)
    if len(rows) == 1:
        return rows[0]
    x = (len(rows) - 1) * q
    lo = int(x)
    hi = min(lo + 1, len(rows) - 1)
    frac = x - lo
    return rows[lo] * (1.0 - frac) + rows[hi] * frac


def _headway_summary(times: Sequence[float]) -> dict[str, object]:
    rows = sorted(float(t) for t in times)
    gaps = [b - a for a, b in zip(rows, rows[1:])]
    if not gaps:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p05": None, "p95": None}
    return {
        "count": len(gaps),
        "mean": mean(gaps),
        "median": median(gaps),
        "min": min(gaps),
        "max": max(gaps),
        "p05": _quantile(gaps, 0.05),
        "p95": _quantile(gaps, 0.95),
    }


@dataclass(frozen=True, slots=True)
class B6RuntimeBinding:
    """Explicit frozen-D0 runtime binding injected into the profile-owned reference."""

    binding_id: str
    identity_material: Mapping[str, object]
    discovery: object
    lifecycle: object
    recapture_config: object
    runtime_config: object
    recheck_config: object
    safety_config: object
    same_route_dynamic_ingress: object | None = None
    future_ingress_clearance: object | None = None
    persistent_same_route_following: object | None = None
    classification_config: object | None = None
    motion_backend: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.binding_id, str) or not self.binding_id.strip():
            raise B6ReferenceError("runtime binding_id must be non-empty")
        material = dict(self.identity_material)
        _canonical_bytes(material)
        object.__setattr__(self, "identity_material", MappingProxyType(material))

    @property
    def identity_sha256(self) -> str:
        return _sha({"binding_id": self.binding_id, "identity_material": dict(self.identity_material)})


@dataclass(frozen=True, slots=True)
class B6ReferenceResult:
    evidence: Mapping[str, object]

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(dict(self.evidence))

    @property
    def evidence_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()

    @property
    def promotable_reference(self) -> bool:
        return bool(self.evidence.get("promotable_reference", False))


class _ObservedReferenceWorld(_MetricEnvironmentalWorld):
    """Passive exact-trajectory conflict-center observer plus frozen D0 safety telemetry."""

    def __init__(self, *args, stream_by_vessel: Mapping[str, str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream_by_vessel = dict(stream_by_vessel)
        self._center_seen: set[str] = set()
        self._center_observations: list[ReferenceCenterPassObservation] = []

    @property
    def center_observations(self) -> tuple[ReferenceCenterPassObservation, ...]:
        return tuple(sorted(self._center_observations, key=lambda x: (x.observation_time_min, x.vessel_id)))

    @staticmethod
    def _progress(state) -> float:
        ux, uy = _heading_unit(state.through_water_heading_deg)
        return state.position[0] * ux + state.position[1] * uy

    def _crossing_time(self, trajectory, duration: float) -> float | None:
        p0 = self._progress(trajectory.state_at_or_continue(0.0))
        p1 = self._progress(trajectory.state_at_or_continue(duration))
        if p0 > _EPS or p1 < -_EPS:
            return None
        if abs(p0) <= _EPS:
            return 0.0
        lo, hi = 0.0, duration
        for _ in range(64):
            if hi - lo <= 1e-10:
                break
            mid = 0.5 * (lo + hi)
            if self._progress(trajectory.state_at_or_continue(mid)) >= 0.0:
                hi = mid
            else:
                lo = mid
        return hi

    def _advance_all(self, duration: float):  # noqa: SLF001 - telemetry-only subclass
        if duration > _EPS:
            snapshot = self.snapshot()
            for state in snapshot.vessels:
                if state.vessel_id in self._center_seen or state.vessel_id not in self._stream_by_vessel:
                    continue
                trajectory = build_environmental_candidate_transition(
                    state,
                    self.drift_for(state.vessel_id),
                    self.command_for(state.vessel_id),
                    self.limits_for(state.vessel_id),
                    motion_backend=self.motion_backend,
                    motion_state=self.motion_state_for(state.vessel_id),
                    evaluation_horizon_min=duration,
                )
                crossing = self._crossing_time(trajectory, duration)
                if crossing is None:
                    continue
                crossed = trajectory.state_at_or_continue(crossing)
                self._center_seen.add(state.vessel_id)
                self._center_observations.append(
                    ReferenceCenterPassObservation(
                        vessel_id=state.vessel_id,
                        stream_id=self._stream_by_vessel[state.vessel_id],
                        observation_time_min=snapshot.time + crossing,
                        heading_deg=crossed.through_water_heading_deg,
                        speed_nm_per_min=crossed.through_water_speed,
                    )
                )
        return super()._advance_all(duration)


class _CutoffSaturationController:
    """Audit finite saturated supply and close unadmitted reference supply at cutoff."""

    def __init__(
        self,
        inner: ReferenceFairSlotArbiter,
        *,
        cutoff_min: float,
        block_until_min: float,
        stream_ids: Sequence[str],
        required_pending_depth: int,
    ) -> None:
        self.inner = inner
        self.cutoff_min = float(cutoff_min)
        self.block_until_min = float(block_until_min)
        self.stream_ids = tuple(stream_ids)
        self.required_pending_depth = int(required_pending_depth)
        self.closed = False
        self.supply_exhausted = False
        self.min_pending_by_stream = {stream: 10**18 for stream in self.stream_ids}
        self.cutoff_invalidated_reservation: str | None = None

    def _audit(self, reservoir: EnvironmentalNetworkSourceReservoir, now: float) -> None:
        if now >= self.cutoff_min - _EPS:
            return
        for stream in self.stream_ids:
            count = sum(1 for d in reservoir.pending_demands if d.stream_id == stream and d.demand_time <= now + _EPS)
            self.min_pending_by_stream[stream] = min(self.min_pending_by_stream[stream], count)
            if count < self.required_pending_depth:
                self.supply_exhausted = True

    def close(self) -> None:
        self.closed = True
        self.cutoff_invalidated_reservation = self.inner.cancel_reservation_for_cutoff()

    def prepare_reservoir(self, reservoir, context: EnvironmentalNetworkSourceControllerContext):
        now = float(context.snapshot.time)
        self._audit(reservoir, now)
        if not self.closed and now < self.cutoff_min - _EPS:
            return self.inner.prepare_reservoir(reservoir, context)
        rows = []
        admitted = set(reservoir.admitted_vessel_ids)
        for demand in reservoir.demands:
            if demand.vessel_id in admitted:
                rows.append(demand)
            else:
                rows.append(replace(demand, release_not_before_time=max(self.block_until_min, demand.physical_eligibility_time)))
        return EnvironmentalNetworkSourceReservoir(
            demands=tuple(rows), admissions=reservoir.admissions, service_order_by_stream=reservoir.service_order_by_stream
        )

    def observe_resolution(self, resolution, context: EnvironmentalNetworkSourceControllerContext) -> None:
        if not self.closed and float(context.snapshot.time) < self.cutoff_min - _EPS:
            self.inner.observe_resolution(resolution, context)


def _semantic_supply_cycle(template) -> list[dict[str, object]]:
    return [
        {
            "profile_class_id": item.profile_class_id,
            "through_water_heading_deg": float(item.through_water_heading_deg),
            "through_water_speed_nm_per_min": float(item.through_water_speed_nm_per_min),
        }
        for item in template.cycle
    ]


def _supply(
    profile: TwoStreamProfileContract,
    profile_classes: Mapping[str, VesselClassProfile],
    stream_keys: Mapping[str, str],
):
    demands: list[EnvironmentalNetworkSourceDemand] = []
    service_order: list[tuple[str, tuple[str, ...]]] = []
    route_targets: dict[str, EnvironmentalRouteHandoffTarget] = {}
    stream_by_vessel: dict[str, str] = {}
    profile_assignments: dict[str, VesselClassProfile] = {}
    raw_manifest: list[dict[str, object]] = []
    semantic_manifest: list[dict[str, object]] = []
    geometry = {x.stream_id: x for x in profile.streams}
    templates = {x.stream_id: x for x in profile.b6_reference.supply_templates}
    n = profile.b6_reference.saturated_demands_per_stream
    for stream_id, key in sorted(stream_keys.items(), key=lambda item: item[1]):
        template = templates.get(stream_id)
        if template is None or not template.cycle:
            raise B6ReferenceError("FAIL_MISSING_REFERENCE_SUPPLY_TEMPLATE")
        geo = geometry[stream_id]
        ids: list[str] = []
        for index in range(n):
            cycle_index = index % len(template.cycle)
            item = template.cycle[cycle_index]
            vessel_profile = profile_classes.get(item.profile_class_id)
            if vessel_profile is None or vessel_profile.safety_domain is None:
                raise B6ReferenceError("FAIL_MISSING_REFERENCE_SUPPLY_TEMPLATE")
            semantic_identity = {
                "reference_stream_key": key,
                "sequence_index": index,
                "template_cycle_index": cycle_index,
                "profile_class_id": item.profile_class_id,
                "heading_deg": float(item.through_water_heading_deg),
                "speed_nm_per_min": float(item.through_water_speed_nm_per_min),
            }
            uid = _sha(semantic_identity)[:24]
            vessel_id = f"b6r-{key[:12]}-{index:05d}-{uid}"
            ux, uy = _heading_unit(item.through_water_heading_deg)
            start = (-geo.source_to_center_distance_nm * ux, -geo.source_to_center_distance_nm * uy)
            end = (geo.center_to_exit_distance_nm * ux, geo.center_to_exit_distance_nm * uy)
            route = Route2D((start, end))
            target = EnvironmentalRouteHandoffTarget(
                vessel_id=vessel_id,
                navigator=RouteNavigator(RouteFollower(route, item.through_water_speed_nm_per_min), CrossTrackRecovery(route)),
                end_action=RouteEndAction.REMOVE,
            )
            demand = EnvironmentalNetworkSourceDemand(
                stream_id=stream_id,
                nominal_vessel_id=vessel_id,
                demand_time=0.0,
                vessel=WaterRelativeVesselState(vessel_id, start, item.through_water_heading_deg, item.through_water_speed_nm_per_min),
                route_target=target,
                limits=vessel_profile.maneuver_limits,
                environmental_drift_velocity=(0.0, 0.0),
                profile=vessel_profile,
            )
            demands.append(demand)
            ids.append(vessel_id)
            route_targets[vessel_id] = target
            stream_by_vessel[vessel_id] = stream_id
            profile_assignments[vessel_id] = vessel_profile
            semantic_manifest.append(semantic_identity)
            raw_manifest.append({"stream_id": stream_id, "vessel_id": vessel_id, **semantic_identity})
        service_order.append((stream_id, tuple(ids)))
    return (
        EnvironmentalNetworkSourceReservoir(tuple(demands), service_order_by_stream=tuple(service_order)),
        route_targets,
        stream_by_vessel,
        VesselProfileCatalog(profile_assignments),
        tuple(raw_manifest),
        tuple(semantic_manifest),
    )


def _semantic_measurement(
    measured: Sequence[ReferenceCenterPassObservation],
    stream_keys: Mapping[str, str],
    measurement_min: float,
    targets_by_stream: Mapping[str, float],
) -> dict[str, object]:
    by_key: dict[str, list[ReferenceCenterPassObservation]] = {key: [] for key in stream_keys.values()}
    for row in measured:
        by_key[stream_keys[row.stream_id]].append(row)
    counts = {key: len(rows) for key, rows in sorted(by_key.items())}
    total = sum(counts.values())
    shares = {key: (counts[key] / total if total else None) for key in counts}
    target_by_key = {stream_keys[s]: float(q) for s, q in targets_by_stream.items()}
    times = [x.observation_time_min for x in measured]
    return {
        "total_center_passages": total,
        "per_stream_key_center_passages": counts,
        "service_rate_per_min": total / measurement_min,
        "per_stream_key_service_rate_per_min": {key: counts[key] / measurement_min for key in counts},
        "achieved_stream_key_shares": shares,
        "target_stream_key_shares": dict(sorted(target_by_key.items())),
        "center_passages": [
            {"vessel_id": x.vessel_id, "reference_stream_key": stream_keys[x.stream_id], "time_min": x.observation_time_min}
            for x in sorted(measured, key=lambda x: (x.observation_time_min, x.vessel_id))
        ],
        "headways_total": _headway_summary(times),
        "headways_by_stream_key": {key: _headway_summary([x.observation_time_min for x in rows]) for key, rows in sorted(by_key.items())},
    }


def _normalized_run_view(run: Mapping[str, object]) -> dict[str, object]:
    """Display-name-free Q4 rename comparison surface."""
    return {
        "status": run.get("status"),
        "warmup_min": run.get("warmup_min"),
        "measurement_duration_min": run.get("measurement_duration_min"),
        "drain_min": run.get("drain_min"),
        "semantic_supply_manifest_sha256": run.get("semantic_supply_manifest_sha256"),
        "semantic_measurement": run.get("semantic_measurement"),
        "safety": run.get("safety"),
        "reference_arbiter_semantic_trace_sha256": run.get("reference_arbiter_semantic_trace_sha256"),
        "reference_arbiter": run.get("reference_arbiter"),
        "semantic_admissions": run.get("semantic_admissions"),
        "drain_complete": (run.get("drain") or {}).get("complete") if isinstance(run.get("drain"), Mapping) else None,
    }


def _run_once(
    profile: TwoStreamProfileContract,
    profile_classes: Mapping[str, VesselClassProfile],
    runtime: B6RuntimeBinding,
    *,
    warmup_min: float,
    measurement_min: float,
    drain_min: float,
) -> dict[str, object]:
    cutoff = warmup_min + measurement_min
    end = cutoff + drain_min
    try:
        stream_keys = build_reference_stream_keys(profile, profile_classes)
        reservoir, route_targets, stream_by_vessel, profiles, raw_manifest, semantic_manifest = _supply(profile, profile_classes, stream_keys)
        world = _ObservedReferenceWorld(
            (), limits={}, environmental_drift_velocity={}, start_time=0.0,
            profile_catalog=profiles, safety_config=runtime.safety_config,
            motion_backend=runtime.motion_backend, retain_safety_intervals=True,
            stream_by_vessel=stream_by_vessel,
        )
        registry = EncounterRegistry(discovery=runtime.discovery, lifecycle=runtime.lifecycle, profile_catalog=profiles)
        inner = ReferenceFairSlotArbiter(
            profile,
            center_observation_provider=lambda: world.center_observations,
            reference_stream_keys_by_stream=stream_keys,
        )
        controller = _CutoffSaturationController(
            inner,
            cutoff_min=cutoff,
            block_until_min=end + 1.0,
            stream_ids=tuple(stream_keys),
            # A69 excludes B5 execution, but A67's conservative finite-supply saturation
            # depth is retained as a censoring gate; it does not affect arbitration.
            required_pending_depth=max(1, profile.b5_fairness.lookahead_depth),
        )
        started = start_environmental_network_source_fifo_runtime(
            registry, world, route_targets, reservoir,
            recapture_config=runtime.recapture_config,
            runtime_config=runtime.runtime_config,
            recheck_config=runtime.recheck_config,
            classification_config=runtime.classification_config,
            profile_catalog=profiles,
            same_route_dynamic_ingress=runtime.same_route_dynamic_ingress,
            future_ingress_clearance=runtime.future_ingress_clearance,
            persistent_same_route_following=runtime.persistent_same_route_following,
            source_admission_controller=controller,
        )
        pre = started.session.run_until_capturing_terminal_outcome(cutoff)
        if pre.status is not EnvironmentalNetworkSourceFIFOTerminalStatus.SUCCESS:
            return {"status": "FAIL_CONTROLLER_OR_SOLVER", "terminal": pre.status.value, "terminal_time_min": pre.terminal_time_min}
        controller.close()
        post = started.session.run_until_capturing_terminal_outcome(end)
        if post.status is not EnvironmentalNetworkSourceFIFOTerminalStatus.SUCCESS:
            return {"status": "FAIL_CONTROLLER_OR_SOLVER", "terminal": post.status.value, "terminal_time_min": post.terminal_time_min}
    except ReferenceArbiterError as exc:
        return {"status": exc.status, "detail": exc.detail}
    except B6ReferenceError as exc:
        return {"status": str(exc)}

    measured = [x for x in world.center_observations if warmup_min - _EPS <= x.observation_time_min < cutoff - _EPS]
    by_stream = {stream: [] for stream in sorted(stream_keys)}
    for row in measured:
        by_stream[row.stream_id].append(row)
    counts = {stream: len(rows) for stream, rows in by_stream.items()}
    total = sum(counts.values())
    targets = {x.stream_id: float(x.share) for x in profile.b6_reference.stream_shares}
    shares = {stream: (counts[stream] / total if total else None) for stream in by_stream}
    semantic_measurement = _semantic_measurement(measured, stream_keys, measurement_min, targets)

    safety = world.finalize_safety_experiment_summary()
    safety_outcomes: dict[str, int] = {}
    for interval in world.safety_intervals:
        key = interval.metric.outcome.value
        safety_outcomes[key] = safety_outcomes.get(key, 0) + 1
    unresolved_clearance = sum(safety_outcomes.get(key, 0) for key in ("hull_clearance_unresolved", "safety_domain_clearance_unresolved"))
    safety_pass = safety.modeled_hull_overlap_episode_count == 0 and safety.safety_domain_intrusion_episode_count == 0 and unresolved_clearance == 0
    active_ids = {x.vessel_id for x in world.snapshot().vessels}
    admitted_ids = set(post.reservoir.admitted_vessel_ids)
    drain_incomplete = bool(active_ids & admitted_ids)

    if controller.supply_exhausted:
        status = "CENSORED_SUPPLY_EXHAUSTED_BEFORE_CUTOFF"
    elif drain_incomplete:
        status = "CENSORED_DRAIN_INCOMPLETE"
    elif not safety_pass:
        status = "FAIL_SAFETY"
    elif total == 0 or any(counts[s] < 2 for s in counts):
        status = "FAIL_INSUFFICIENT_MEASUREMENT_PASSAGES"
    elif any(abs(float(shares[s]) - targets[s]) > profile.b6_reference.balance_tolerance_fraction + _EPS for s in shares):
        status = "FAIL_BALANCE"
    else:
        status = "PASS_REFERENCE_RUN"

    times = [x.observation_time_min for x in measured]
    semantic_trace = inner.semantic_trace()
    semantic_admissions = [
        {
            "vessel_id": x.vessel_id,
            "reference_stream_key": stream_keys[x.stream_id],
            "admission_time": x.admission_time,
        }
        for x in post.reservoir.admissions
    ]
    result = {
        "status": status,
        "warmup_min": warmup_min,
        "measurement_duration_min": measurement_min,
        "drain_min": drain_min,
        "cutoff_min": cutoff,
        "end_min": end,
        "supply_manifest_sha256": _sha(raw_manifest),
        "semantic_supply_manifest_sha256": _sha(semantic_manifest),
        "reference_stream_keys_by_display_stream": dict(stream_keys),
        "measurement": {
            "total_center_passages": total,
            "per_stream_center_passages": counts,
            "service_rate_per_min": total / measurement_min,
            "per_stream_service_rate_per_min": {s: counts[s] / measurement_min for s in counts},
            "achieved_stream_shares": shares,
            "target_stream_shares": targets,
            "center_passages": [
                {"vessel_id": x.vessel_id, "stream_id": x.stream_id, "time_min": x.observation_time_min}
                for x in sorted(measured, key=lambda x: (x.observation_time_min, x.vessel_id))
            ],
            "headways_total": _headway_summary(times),
            "headways_by_stream": {s: _headway_summary([x.observation_time_min for x in rows]) for s, rows in by_stream.items()},
        },
        "semantic_measurement": semantic_measurement,
        "safety": {
            "pass": safety_pass,
            "interval_count": safety.interval_summary.interval_count,
            "modeled_hull_overlap_episode_count": safety.modeled_hull_overlap_episode_count,
            "safety_domain_intrusion_episode_count": safety.safety_domain_intrusion_episode_count,
            "near_miss_episode_count": safety.near_miss_episode_count,
            "unresolved_episode_count": safety.unresolved_episode_count,
            "interval_outcome_counts": safety_outcomes,
            "unresolved_clearance_interval_count": unresolved_clearance,
            "sampled_min_hull_margin_nm": safety.interval_summary.sampled_min_hull_margin_nm,
            "sampled_min_safety_domain_margin_nm": safety.interval_summary.sampled_min_safety_domain_margin_nm,
            "worst_observed_evidence": safety.worst_observed_evidence.value,
        },
        "reference_arbiter": {
            "algorithm_id": REFERENCE_ARBITER_ALGORITHM_ID,
            "deployment_b4_executed": False,
            "deployment_b5_executed": False,
            "commitments_by_stream_key": dict(sorted(inner.commitments_by_key.items())),
            "max_positive_service_deficit_by_stream_key": dict(sorted(inner.max_positive_deficit_by_key.items())),
            "max_share_deficit_passages": profile.b6_reference.max_share_deficit_passages,
            "fairness_comparison_tolerance": profile.b6_reference.fairness_comparison_tolerance,
            "fairness_idle_episode_count": inner.fairness_idle_episode_count,
            "fairness_idle_duration_min": inner.fairness_idle_duration_min,
            "source_denial_replan_count": sum(x["event"] == "REFERENCE_SOURCE_DENIAL_REPLAN" for x in inner.trace),
        },
        "reference_arbiter_trace": tuple(inner.trace),
        "reference_arbiter_semantic_trace_sha256": _sha(semantic_trace),
        "source_denial_count": sum(not x.admitted for b in post.source_boundaries for x in b.resolution.decisions),
        "saturation": {
            "exhausted_before_cutoff": controller.supply_exhausted,
            "min_pending_by_stream": controller.min_pending_by_stream,
            "required_pending_depth": controller.required_pending_depth,
        },
        "cutoff": {"invalidated_reservation_vessel_id": controller.cutoff_invalidated_reservation},
        "drain": {"complete": not drain_incomplete, "active_admitted_reference_vessel_ids": sorted(active_ids & admitted_ids)},
        "admissions": [
            {"vessel_id": x.vessel_id, "stream_id": x.stream_id, "admission_time": x.admission_time}
            for x in post.reservoir.admissions
        ],
        "semantic_admissions": semantic_admissions,
    }
    result["normalized_semantic_evidence_sha256"] = _sha(_normalized_run_view(result))
    return result


def run_balanced_saturated_reference(
    profile: TwoStreamProfileContract,
    profile_classes: Mapping[str, VesselClassProfile],
    runtime: B6RuntimeBinding,
    *,
    b6_spec_sha256: str,
    implementation_manifest_sha256: str,
) -> B6ReferenceResult:
    """Execute A69/A70 R0/R1/R2 and exact replay gates using frozen D0 physical runtime."""
    if not isinstance(profile, TwoStreamProfileContract):
        raise TypeError("profile must be TwoStreamProfileContract")
    if not isinstance(runtime, B6RuntimeBinding):
        raise TypeError("runtime must be B6RuntimeBinding")
    if len(b6_spec_sha256) != 64 or len(implementation_manifest_sha256) != 64:
        raise B6ReferenceError("B6 spec and implementation manifest identities must be SHA-256 hex strings")

    b6 = profile.b6_reference
    variants = {
        "R0": (b6.warmup_min, b6.measurement_duration_min, b6.drain_min),
        "R1": (b6.warmup_min, 2.0 * b6.measurement_duration_min, b6.drain_min),
        "R2": (2.0 * b6.warmup_min, b6.measurement_duration_min, b6.drain_min),
    }
    runs: dict[str, object] = {}
    for name, (w, m, d) in variants.items():
        first = _run_once(profile, profile_classes, runtime, warmup_min=w, measurement_min=m, drain_min=d)
        second = _run_once(profile, profile_classes, runtime, warmup_min=w, measurement_min=m, drain_min=d)
        replay_equal = _canonical_bytes(first) == _canonical_bytes(second)
        if not replay_equal:
            first = dict(first)
            first["status"] = "FAIL_NONDETERMINISTIC_REPLAY"
        runs[name] = {
            "run": first,
            "replay_equal": replay_equal,
            "run_sha256": _sha(first),
            "replay_sha256": _sha(second),
        }

    pass_runs = all(runs[name]["run"]["status"] == "PASS_REFERENCE_RUN" and runs[name]["replay_equal"] for name in variants)  # type: ignore[index]
    r0 = runs["R0"]["run"].get("measurement", {}).get("service_rate_per_min")  # type: ignore[index]
    r1 = runs["R1"]["run"].get("measurement", {}).get("service_rate_per_min")  # type: ignore[index]
    r2 = runs["R2"]["run"].get("measurement", {}).get("service_rate_per_min")  # type: ignore[index]
    threshold = 1.0 / b6.measurement_duration_min
    stable = bool(pass_runs and r0 is not None and r1 is not None and r2 is not None and abs(r0 - r1) <= threshold + _EPS and abs(r0 - r2) <= threshold + _EPS)

    stream_keys = build_reference_stream_keys(profile, profile_classes)
    identity = {
        "algorithm_id": B6_REFERENCE_ALGORITHM_ID,
        "b6_spec_sha256": b6_spec_sha256,
        "profile_id": profile.profile_id,
        "profile_schema": profile.schema,
        "profile_version": profile.profile_version,
        "profile_contract_sha256": profile.contract_sha256,
        "base_framework_version": profile.base_framework_version,
        "runtime_adapter_id": profile.runtime_adapter_id,
        "runtime_binding_id": runtime.binding_id,
        "runtime_binding_sha256": runtime.identity_sha256,
        "a64_solver_algorithm_id": SOLVER_ALGORITHM_ID,
        "reference_arbiter_algorithm_id": REFERENCE_ARBITER_ALGORITHM_ID,
        "deployment_b4_executed": False,
        "deployment_b5_executed": False,
        "reference_stream_keys": sorted(stream_keys.values()),
        "b6_reference_contract": profile.canonical_dict()["b6_reference"],
        "coordinate_convention_id": B6_COORDINATE_CONVENTION_ID,
        "window_convention_id": B6_WINDOW_CONVENTION_ID,
        "convergence_convention_id": B6_CONVERGENCE_CONVENTION_ID,
        "headway_quantile_convention_id": B6_HEADWAY_QUANTILE_CONVENTION_ID,
        "semantic_normalization_id": B6_SEMANTIC_NORMALIZATION_ID,
        "implementation_manifest_sha256": implementation_manifest_sha256,
    }
    promotable = pass_runs and stable
    evidence: dict[str, object] = {
        "schema": B6_EVIDENCE_SCHEMA,
        "algorithm_id": B6_REFERENCE_ALGORITHM_ID,
        "reference_arbiter_algorithm_id": REFERENCE_ARBITER_ALGORITHM_ID,
        "deployment_b4_executed": False,
        "deployment_b5_executed": False,
        "causal": False,
        "field_calibrated": False,
        "deployment_recommendation": False,
        "global_optimum_claim": False,
        "promotable_reference": promotable,
        "aggregate_status": "PASS_BALANCED_SATURATED_REFERENCE" if promotable else "FAIL_NON_PROMOTABLE_REFERENCE",
        "reference_execution_identity": identity,
        "reference_execution_identity_sha256": _sha(identity),
        "profile_contract_sha256": profile.contract_sha256,
        "b6_spec_sha256": b6_spec_sha256,
        "semantic_supply_manifest_sha256": runs["R0"]["run"].get("semantic_supply_manifest_sha256"),  # type: ignore[index]
        "implementation_manifest_sha256": implementation_manifest_sha256,
        "runs": runs,
        "rate_stability": {
            "threshold_per_min": threshold,
            "r0_minus_r1_abs": None if r0 is None or r1 is None else abs(r0 - r1),
            "r0_minus_r2_abs": None if r0 is None or r2 is None else abs(r0 - r2),
            "pass": stable,
        },
        "claim_boundary": {
            "quantity_name": "balanced_saturated_service_reference",
            "field_capacity": False,
            "deployment_safety": False,
            "causal_effect": False,
            "global_policy_optimum": False,
            "reference_policy": "greedy-earliest-safe-under-bounded-fair-share",
        },
    }
    if promotable:
        evidence["balanced_saturated_service_reference"] = runs["R0"]["run"]["semantic_measurement"]  # type: ignore[index]
    return B6ReferenceResult(MappingProxyType(evidence))
