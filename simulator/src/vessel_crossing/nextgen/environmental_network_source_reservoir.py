"""Pure arm-local FIFO source-reservoir and entry-clearance evidence.

A148 deliberately stops before runtime integration.  The shared stochastic object is the
Poisson demand realization; each experiment arm may own an immutable reservoir state that
maps those demands to physical admissions from that arm's *current* physical entry
clearance.  No demand is redrawn, thinned, reordered within a stream, or moved earlier.

The resolver is a pure function of:

* immutable source demand records;
* prior immutable admission evidence;
* one exact environmental world snapshot; and
* an immutable mapping of active vessel profiles.

It never advances a world, writes a command, updates an encounter registry, or restores
runtime authority.  Admission eligibility is instantaneous and fail-closed: an incoming
ship domain must be available, every active vessel must have a ship domain, and the
incoming domain must not intrude any active domain.  Heads from different source streams
are considered in deterministic demand/stream/vessel order; already accepted same-time
heads become virtual occupants so mutually incompatible entries cannot both be admitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping

from ..kinematics import Vector2
from .dynamics import ManeuverLimits
from .environmental_dynamics import WaterRelativeManeuverCommand, WaterRelativeVesselState
from .environmental_fleet_ingress import EnvironmentalFleetIngressRecord
from .environmental_candidate_transition import (
    EnvironmentalMotionBackend,
    EnvironmentalMotionState,
    build_environmental_candidate_transition,
    validate_environmental_motion_state,
)
from .environmental_pair_clearance import EnvironmentalPairTransitionConfig, EnvironmentalPairTransitionUnavailable
from .environmental_pair_command_clearance import evaluate_environmental_pair_command_clearance
from .environmental_network_scenario import EnvironmentalNetworkScenarioIngress
from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
from .environmental_world import EnvironmentalWorldSnapshot
from .ship_domain import PairShipDomainClearance, evaluate_pair_ship_domain
from .vessel_classes import VesselClassProfile
from .same_route_following import (
    SameRouteFollowingSupervisor,
    SameRouteFollowingUnavailable,
    compute_same_route_following_constraint,
    same_route_longitudinal_pair_applicable,
    same_route_safety_projection,
)

_EPS = 1e-10


class EnvironmentalNetworkSourceReservoirError(RuntimeError):
    """Raised when FIFO source-reservoir evidence is internally inconsistent."""


class EnvironmentalNetworkSourceHeadStatus(str, Enum):
    """Fail-closed result for one due per-stream FIFO head."""

    ADMITTED = "ADMITTED"
    BLOCKED_MISSING_INCOMING_DOMAIN = "BLOCKED_MISSING_INCOMING_DOMAIN"
    BLOCKED_MISSING_ACTIVE_DOMAIN = "BLOCKED_MISSING_ACTIVE_DOMAIN"
    BLOCKED_ACTIVE_DOMAIN_INTRUSION = "BLOCKED_ACTIVE_DOMAIN_INTRUSION"
    BLOCKED_PRECEDENCE_DOMAIN_INTRUSION = "BLOCKED_PRECEDENCE_DOMAIN_INTRUSION"
    BLOCKED_DYNAMIC_LAUNCH_INFEASIBLE = "BLOCKED_DYNAMIC_LAUNCH_INFEASIBLE"
    BLOCKED_FUTURE_CLEARANCE_INFEASIBLE = "BLOCKED_FUTURE_CLEARANCE_INFEASIBLE"


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceDemand:
    """One immutable exogenous demand waiting outside the modeled waterway."""

    stream_id: str
    nominal_vessel_id: str
    demand_time: float
    vessel: WaterRelativeVesselState
    route_target: EnvironmentalRouteHandoffTarget
    limits: ManeuverLimits | None
    environmental_drift_velocity: Vector2
    profile: VesselClassProfile | None
    release_not_before_time: float | None = None

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ValueError("stream_id must be non-empty")
        if not self.nominal_vessel_id:
            raise ValueError("nominal_vessel_id must be non-empty")
        if not isfinite(self.demand_time):
            raise ValueError("demand_time must be finite")
        if not isinstance(self.vessel, WaterRelativeVesselState):
            raise TypeError("vessel must be WaterRelativeVesselState")
        if not isinstance(self.route_target, EnvironmentalRouteHandoffTarget):
            raise TypeError("route_target must be EnvironmentalRouteHandoffTarget")
        if self.nominal_vessel_id != self.vessel.vessel_id:
            raise EnvironmentalNetworkSourceReservoirError(
                "nominal_vessel_id must equal the physical vessel id"
            )
        if self.route_target.vessel_id != self.vessel.vessel_id:
            raise EnvironmentalNetworkSourceReservoirError(
                "route target vessel id must equal the physical vessel id"
            )
        if self.limits is not None and not isinstance(self.limits, ManeuverLimits):
            raise TypeError("limits must be ManeuverLimits or None")
        if self.profile is not None and not isinstance(self.profile, VesselClassProfile):
            raise TypeError("profile must be VesselClassProfile or None")
        if len(self.environmental_drift_velocity) != 2 or any(
            not isfinite(component) for component in self.environmental_drift_velocity
        ):
            raise ValueError("environmental_drift_velocity must contain two finite components")
        if self.release_not_before_time is not None:
            if not isfinite(self.release_not_before_time):
                raise ValueError("release_not_before_time must be finite when supplied")
            if self.release_not_before_time < self.demand_time - _EPS:
                raise EnvironmentalNetworkSourceReservoirError(
                    "release_not_before_time cannot precede stochastic demand"
                )

    @property
    def vessel_id(self) -> str:
        return self.vessel.vessel_id

    @property
    def physical_eligibility_time(self) -> float:
        """Earliest controller-permitted physical admission time.

        ``demand_time`` remains the immutable exogenous-arrival timestamp.  The optional
        not-before gate is a separate management outcome and therefore never rewrites
        stochastic demand identity or queue-wait accounting.
        """

        return (
            self.demand_time
            if self.release_not_before_time is None
            else self.release_not_before_time
        )

    @classmethod
    def from_scenario_ingress(
        cls,
        ingress: EnvironmentalNetworkScenarioIngress,
    ) -> "EnvironmentalNetworkSourceDemand":
        if not isinstance(ingress, EnvironmentalNetworkScenarioIngress):
            raise TypeError("ingress must be EnvironmentalNetworkScenarioIngress")
        record = ingress.record
        return cls(
            stream_id=ingress.stream_id,
            nominal_vessel_id=ingress.nominal_vessel_id,
            demand_time=ingress.demand_time,
            vessel=record.vessel,
            route_target=record.route_target,
            limits=record.limits,
            environmental_drift_velocity=record.environmental_drift_velocity
            if record.environmental_drift_velocity is not None
            else ingress.environmental_drift_velocity,
            profile=record.profile,
        )

    def admission_record(self, admission_time: float) -> EnvironmentalFleetIngressRecord:
        """Bind this demand to one later/equal physical membership time."""
        if not isfinite(admission_time):
            raise ValueError("admission_time must be finite")
        if admission_time < self.demand_time - _EPS:
            raise EnvironmentalNetworkSourceReservoirError(
                "physical admission cannot precede stochastic demand"
            )
        return EnvironmentalFleetIngressRecord(
            at=float(admission_time),
            vessel=self.vessel,
            route_target=self.route_target,
            limits=self.limits,
            environmental_drift_velocity=self.environmental_drift_velocity,
            profile=self.profile,
        )


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceClearanceEvidence:
    """One pairwise ship-domain check used by a source-head decision."""

    other_vessel_id: str
    other_is_same_time_precedence_admission: bool
    clearance: PairShipDomainClearance


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceFutureIngressConfig:
    """Opt-in full-horizon launch viability using the same pair physics as A104.

    No independent CPA threshold is introduced. ``pair_clearance`` is the existing
    A85/A104 physical horizon/tolerance contract.  The incoming vessel is evaluated on
    an explicit launch-hold command while active vessels retain their currently issued
    commands.
    """

    pair_clearance: EnvironmentalPairTransitionConfig

    def __post_init__(self) -> None:
        if not isinstance(self.pair_clearance, EnvironmentalPairTransitionConfig):
            raise TypeError("pair_clearance must be EnvironmentalPairTransitionConfig")


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceFutureClearanceEvidence:
    other_vessel_id: str
    other_is_same_time_precedence_admission: bool
    certified_clear: bool
    transition_certified_clear: bool
    post_transition_certified_clear: bool
    predicted_min_separation_nm: float | None
    minimum_certified_domain_margin_nm: float | None


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceHeadDecision:
    """Decision/evidence for the due head of one source stream."""

    stream_id: str
    vessel_id: str
    demand_time: float
    evaluated_at: float
    due_queue_depth: int
    status: EnvironmentalNetworkSourceHeadStatus
    clearances: tuple[EnvironmentalNetworkSourceClearanceEvidence, ...] = ()
    missing_domain_vessel_ids: tuple[str, ...] = ()
    dynamic_launch_speed_ceiling_nm_per_min: float | None = None
    dynamic_limiting_leader_id: str | None = None
    future_clearances: tuple[EnvironmentalNetworkSourceFutureClearanceEvidence, ...] = ()

    @property
    def admitted(self) -> bool:
        return self.status is EnvironmentalNetworkSourceHeadStatus.ADMITTED

    @property
    def waiting_time_min(self) -> float:
        return self.evaluated_at - self.demand_time


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceAdmission:
    """Immutable arm-local physical admission derived from one exogenous demand."""

    stream_id: str
    vessel_id: str
    demand_time: float
    admission_time: float
    record: EnvironmentalFleetIngressRecord

    def __post_init__(self) -> None:
        if self.record.vessel_id != self.vessel_id:
            raise EnvironmentalNetworkSourceReservoirError(
                "admission record vessel id disagrees with admission evidence"
            )
        if abs(self.record.at - self.admission_time) > _EPS:
            raise EnvironmentalNetworkSourceReservoirError(
                "admission record time disagrees with admission evidence"
            )
        if self.admission_time < self.demand_time - _EPS:
            raise EnvironmentalNetworkSourceReservoirError(
                "physical admission cannot precede stochastic demand"
            )

    @property
    def waiting_time_min(self) -> float:
        return self.admission_time - self.demand_time


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceReservoir:
    """Immutable demand set plus arm-local admission history."""

    demands: tuple[EnvironmentalNetworkSourceDemand, ...]
    admissions: tuple[EnvironmentalNetworkSourceAdmission, ...] = ()
    service_order_by_stream: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        vessel_ids = [item.vessel_id for item in self.demands]
        if len(set(vessel_ids)) != len(vessel_ids):
            raise EnvironmentalNetworkSourceReservoirError("duplicate demand vessel id")

        # Preserve the external stochastic realization.  Sorting is not used to repair a
        # malformed stream because doing so would hide demand-order corruption.
        last_by_stream: dict[str, tuple[float, str]] = {}
        demand_ids_by_stream: dict[str, list[str]] = {}
        for demand in self.demands:
            prior = last_by_stream.get(demand.stream_id)
            current = (demand.demand_time, demand.vessel_id)
            if prior is not None and current < prior:
                raise EnvironmentalNetworkSourceReservoirError(
                    f"demand order for stream {demand.stream_id!r} is not FIFO-canonical"
                )
            last_by_stream[demand.stream_id] = current
            demand_ids_by_stream.setdefault(demand.stream_id, []).append(demand.vessel_id)

        explicit_order: dict[str, tuple[str, ...]] = {}
        for stream_id, vessel_order in self.service_order_by_stream:
            if not isinstance(stream_id, str) or not stream_id:
                raise EnvironmentalNetworkSourceReservoirError(
                    "controlled service-order stream id must be non-empty"
                )
            if stream_id in explicit_order:
                raise EnvironmentalNetworkSourceReservoirError(
                    f"duplicate controlled service-order stream {stream_id!r}"
                )
            order = tuple(vessel_order)
            canonical_ids = tuple(demand_ids_by_stream.get(stream_id, ()))
            if not canonical_ids:
                raise EnvironmentalNetworkSourceReservoirError(
                    f"controlled service order refers to unknown stream {stream_id!r}"
                )
            if len(order) != len(set(order)) or set(order) != set(canonical_ids):
                raise EnvironmentalNetworkSourceReservoirError(
                    f"controlled service order for stream {stream_id!r} must contain each demand exactly once"
                )
            explicit_order[stream_id] = order
        normalized_service_order = tuple(
            (stream_id, explicit_order[stream_id]) for stream_id in sorted(explicit_order)
        )
        object.__setattr__(self, "service_order_by_stream", normalized_service_order)

        service_order = {
            stream_id: explicit_order.get(stream_id, tuple(ids))
            for stream_id, ids in demand_ids_by_stream.items()
        }

        admitted_ids: set[str] = set()
        admitted_by_stream: dict[str, list[str]] = {}
        demand_by_id = {item.vessel_id: item for item in self.demands}
        previous_time = float("-inf")
        for admission in self.admissions:
            if admission.vessel_id in admitted_ids:
                raise EnvironmentalNetworkSourceReservoirError("duplicate physical admission")
            demand = demand_by_id.get(admission.vessel_id)
            if demand is None:
                raise EnvironmentalNetworkSourceReservoirError(
                    f"admission for unknown demand vessel {admission.vessel_id!r}"
                )
            if admission.stream_id != demand.stream_id or abs(
                admission.demand_time - demand.demand_time
            ) > _EPS:
                raise EnvironmentalNetworkSourceReservoirError(
                    "admission does not preserve source demand identity"
                )
            if admission.admission_time < previous_time - _EPS:
                raise EnvironmentalNetworkSourceReservoirError(
                    "admission history must be non-decreasing in physical time"
                )
            prior_stream_admissions = admitted_by_stream.setdefault(demand.stream_id, [])
            order = service_order[demand.stream_id]
            expected_index = len(prior_stream_admissions)
            if expected_index >= len(order) or order[expected_index] != demand.vessel_id:
                if demand.stream_id in explicit_order:
                    raise EnvironmentalNetworkSourceReservoirError(
                        f"admission for {demand.vessel_id!r} violates controlled source service order"
                    )
                raise EnvironmentalNetworkSourceReservoirError(
                    f"admission for {demand.vessel_id!r} overtakes an earlier source demand"
                )
            prior_stream_admissions.append(demand.vessel_id)
            admitted_ids.add(admission.vessel_id)
            previous_time = admission.admission_time

    @classmethod
    def from_scenario_ingress(
        cls,
        ingress: Iterable[EnvironmentalNetworkScenarioIngress],
    ) -> "EnvironmentalNetworkSourceReservoir":
        demands = tuple(EnvironmentalNetworkSourceDemand.from_scenario_ingress(item) for item in ingress)
        return cls(demands=demands)

    @property
    def admitted_vessel_ids(self) -> tuple[str, ...]:
        return tuple(item.vessel_id for item in self.admissions)

    @property
    def pending_demands(self) -> tuple[EnvironmentalNetworkSourceDemand, ...]:
        admitted = set(self.admitted_vessel_ids)
        return tuple(item for item in self.demands if item.vessel_id not in admitted)

    @property
    def pending_demands_in_service_order(self) -> tuple[EnvironmentalNetworkSourceDemand, ...]:
        """Pending demand ordered by explicit per-stream service order when supplied."""

        pending_by_id = {item.vessel_id: item for item in self.pending_demands}
        explicit = dict(self.service_order_by_stream)
        rows: list[EnvironmentalNetworkSourceDemand] = []
        by_stream: dict[str, list[EnvironmentalNetworkSourceDemand]] = {}
        for demand in self.demands:
            by_stream.setdefault(demand.stream_id, []).append(demand)
        for stream_id in sorted(by_stream):
            order = explicit.get(stream_id, tuple(item.vessel_id for item in by_stream[stream_id]))
            rows.extend(pending_by_id[vessel_id] for vessel_id in order if vessel_id in pending_by_id)
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceQueueMetrics:
    evaluated_at: float
    demand_keys: tuple[tuple[str, str, float], ...]
    demand_count: int
    admission_count: int
    pending_count: int
    due_pending_count: int
    mean_admitted_waiting_time_min: float | None
    max_admitted_waiting_time_min: float | None
    mean_due_waiting_time_min: float | None
    max_due_waiting_time_min: float | None


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceQueueMetricDifference:
    """Control-minus-baseline queue outcome deltas over one shared demand set."""

    admission_count_delta: int
    pending_count_delta: int
    due_pending_count_delta: int
    mean_admitted_waiting_time_min_delta: float | None
    max_admitted_waiting_time_min_delta: float | None
    mean_due_waiting_time_min_delta: float | None
    max_due_waiting_time_min_delta: float | None


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceResolution:
    """One pure FIFO/clearance evaluation at an exact physical snapshot."""

    snapshot_time: float
    decisions: tuple[EnvironmentalNetworkSourceHeadDecision, ...]
    admissions: tuple[EnvironmentalNetworkSourceAdmission, ...]
    reservoir: EnvironmentalNetworkSourceReservoir
    metrics: EnvironmentalNetworkSourceQueueMetrics


def _due_heads(
    reservoir: EnvironmentalNetworkSourceReservoir,
    at: float,
) -> tuple[tuple[EnvironmentalNetworkSourceDemand, int], ...]:
    pending = reservoir.pending_demands_in_service_order
    by_stream: dict[str, list[EnvironmentalNetworkSourceDemand]] = {}
    for demand in pending:
        by_stream.setdefault(demand.stream_id, []).append(demand)

    heads: list[tuple[EnvironmentalNetworkSourceDemand, int]] = []
    for stream_id, items in by_stream.items():
        if not items:
            continue
        head = items[0]
        if head.physical_eligibility_time <= at + _EPS:
            due_depth = sum(item.demand_time <= at + _EPS for item in items)
            heads.append((head, due_depth))
    return tuple(
        sorted(
            heads,
            key=lambda item: (
                item[0].physical_eligibility_time,
                item[0].demand_time,
                item[0].stream_id,
                item[0].vessel_id,
            ),
        )
    )


def _clearance(
    first: WaterRelativeVesselState,
    first_profile: VesselClassProfile,
    second: WaterRelativeVesselState,
    second_profile: VesselClassProfile,
) -> PairShipDomainClearance:
    assert first_profile.safety_domain is not None
    assert second_profile.safety_domain is not None
    return evaluate_pair_ship_domain(
        first_heading_deg=first.through_water_heading_deg,
        second_heading_deg=second.through_water_heading_deg,
        relative_position=(
            second.position[0] - first.position[0],
            second.position[1] - first.position[1],
        ),
        first_domain=first_profile.safety_domain,
        second_domain=second_profile.safety_domain,
    )


def source_queue_metrics(
    reservoir: EnvironmentalNetworkSourceReservoir,
    *,
    at: float,
) -> EnvironmentalNetworkSourceQueueMetrics:
    if not isfinite(at):
        raise ValueError("metric time must be finite")
    if reservoir.admissions and at < reservoir.admissions[-1].admission_time - _EPS:
        raise EnvironmentalNetworkSourceReservoirError(
            "queue metrics cannot be evaluated before recorded admission history"
        )
    waits = tuple(item.waiting_time_min for item in reservoir.admissions)
    pending = reservoir.pending_demands
    due_waits = tuple(at - item.demand_time for item in pending if item.demand_time <= at + _EPS)
    return EnvironmentalNetworkSourceQueueMetrics(
        evaluated_at=float(at),
        demand_keys=tuple(
            (item.stream_id, item.vessel_id, item.demand_time) for item in reservoir.demands
        ),
        demand_count=len(reservoir.demands),
        admission_count=len(reservoir.admissions),
        pending_count=len(pending),
        due_pending_count=len(due_waits),
        mean_admitted_waiting_time_min=(sum(waits) / len(waits)) if waits else None,
        max_admitted_waiting_time_min=max(waits) if waits else None,
        mean_due_waiting_time_min=(sum(due_waits) / len(due_waits)) if due_waits else None,
        max_due_waiting_time_min=max(due_waits) if due_waits else None,
    )


def _optional_delta(control: float | None, baseline: float | None) -> float | None:
    if control is None or baseline is None:
        return None
    return control - baseline


def compare_source_queue_metrics(
    baseline: EnvironmentalNetworkSourceQueueMetrics,
    control: EnvironmentalNetworkSourceQueueMetrics,
) -> EnvironmentalNetworkSourceQueueMetricDifference:
    if baseline.demand_keys != control.demand_keys:
        raise EnvironmentalNetworkSourceReservoirError(
            "paired queue metrics require the exact same exogenous demand keys"
        )
    if abs(baseline.evaluated_at - control.evaluated_at) > _EPS:
        raise EnvironmentalNetworkSourceReservoirError(
            "paired queue metrics require the same physical evaluation horizon"
        )
    return EnvironmentalNetworkSourceQueueMetricDifference(
        admission_count_delta=control.admission_count - baseline.admission_count,
        pending_count_delta=control.pending_count - baseline.pending_count,
        due_pending_count_delta=control.due_pending_count - baseline.due_pending_count,
        mean_admitted_waiting_time_min_delta=_optional_delta(
            control.mean_admitted_waiting_time_min,
            baseline.mean_admitted_waiting_time_min,
        ),
        max_admitted_waiting_time_min_delta=_optional_delta(
            control.max_admitted_waiting_time_min,
            baseline.max_admitted_waiting_time_min,
        ),
        mean_due_waiting_time_min_delta=_optional_delta(
            control.mean_due_waiting_time_min,
            baseline.mean_due_waiting_time_min,
        ),
        max_due_waiting_time_min_delta=_optional_delta(
            control.max_due_waiting_time_min,
            baseline.max_due_waiting_time_min,
        ),
    )



@dataclass(frozen=True, slots=True)
class _SourceReservoirResolutionContext:
    """Normalized immutable view used by the staged source-head resolver."""

    snapshot: EnvironmentalWorldSnapshot
    profile_map: Mapping[str, VesselClassProfile]
    active_by_id: Mapping[str, WaterRelativeVesselState]
    route_target_map: Mapping[str, EnvironmentalRouteHandoffTarget]
    limits_map: Mapping[str, ManeuverLimits]
    drift_map: Mapping[str, Vector2]
    command_map: Mapping[str, WaterRelativeManeuverCommand]
    motion_state_map: Mapping[str, EnvironmentalMotionState | None]
    same_route_dynamic_ingress: SameRouteFollowingSupervisor | None
    future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None
    active_motion_backend: EnvironmentalMotionBackend | None


def _prepare_source_reservoir_resolution_context(
    reservoir: EnvironmentalNetworkSourceReservoir,
    snapshot: EnvironmentalWorldSnapshot,
    active_profiles: Mapping[str, VesselClassProfile],
    *,
    same_route_dynamic_ingress: SameRouteFollowingSupervisor | None,
    future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None,
    active_route_targets: Mapping[str, EnvironmentalRouteHandoffTarget] | None,
    active_limits: Mapping[str, ManeuverLimits] | None,
    active_environmental_drifts: Mapping[str, Vector2] | None,
    active_commands: Mapping[str, WaterRelativeManeuverCommand] | None,
    active_motion_backend: EnvironmentalMotionBackend | None,
    active_motion_states: Mapping[str, EnvironmentalMotionState | None] | None,
) -> _SourceReservoirResolutionContext:
    """Normalize/validate one resolver call without changing historical check order."""

    if not isinstance(reservoir, EnvironmentalNetworkSourceReservoir):
        raise TypeError("reservoir must be EnvironmentalNetworkSourceReservoir")
    if not isinstance(snapshot, EnvironmentalWorldSnapshot):
        raise TypeError("snapshot must be EnvironmentalWorldSnapshot")
    profile_map = MappingProxyType(dict(active_profiles))
    if any(not isinstance(profile, VesselClassProfile) for profile in profile_map.values()):
        raise TypeError("active_profiles values must be VesselClassProfile")
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

    active_by_id = {vessel.vessel_id: vessel for vessel in snapshot.vessels}
    route_target_map = {} if active_route_targets is None else dict(active_route_targets)
    limits_map = {} if active_limits is None else dict(active_limits)
    drift_map = {} if active_environmental_drifts is None else dict(active_environmental_drifts)
    command_map = {} if active_commands is None else dict(active_commands)
    motion_state_map = {} if active_motion_states is None else dict(active_motion_states)
    if active_motion_backend is not None and not isinstance(active_motion_backend, EnvironmentalMotionBackend):
        raise TypeError("active_motion_backend must implement EnvironmentalMotionBackend")
    if active_motion_backend is None and motion_state_map:
        raise EnvironmentalNetworkSourceReservoirError("active motion states require active_motion_backend")
    if active_motion_backend is not None:
        for vessel_id, motion_state in motion_state_map.items():
            validate_environmental_motion_state(
                active_motion_backend,
                motion_state,
                boundary=f"source reservoir active motion {vessel_id!r}",
            )
    if same_route_dynamic_ingress is not None:
        missing_routes = sorted(set(active_by_id) - set(route_target_map))
        missing_limits = sorted(set(active_by_id) - set(limits_map))
        missing_drifts = sorted(set(active_by_id) - set(drift_map))
        if missing_routes or missing_limits or missing_drifts:
            raise EnvironmentalNetworkSourceReservoirError(
                "dynamic ingress requires route target, maneuver limits and drift for every active vessel; "
                f"missing_routes={missing_routes!r}, missing_limits={missing_limits!r}, "
                f"missing_drifts={missing_drifts!r}"
            )
        if any(not isinstance(item, EnvironmentalRouteHandoffTarget) for item in route_target_map.values()):
            raise TypeError("active_route_targets values must be EnvironmentalRouteHandoffTarget")
        if any(not isinstance(item, ManeuverLimits) for item in limits_map.values()):
            raise TypeError("active_limits values must be ManeuverLimits")

    if future_ingress_clearance is not None:
        missing_limits = sorted(set(active_by_id) - set(limits_map))
        missing_drifts = sorted(set(active_by_id) - set(drift_map))
        missing_commands = sorted(set(active_by_id) - set(command_map))
        missing_motion_states = (
            sorted(set(active_by_id) - set(motion_state_map))
            if active_motion_backend is not None else []
        )
        if missing_limits or missing_drifts or missing_commands or missing_motion_states:
            raise EnvironmentalNetworkSourceReservoirError(
                "future ingress clearance requires maneuver limits, drift, current command, "
                "and backend state when a richer backend is active; "
                f"missing_limits={missing_limits!r}, missing_drifts={missing_drifts!r}, "
                f"missing_commands={missing_commands!r}, missing_motion_states={missing_motion_states!r}"
            )
        if any(not isinstance(item, ManeuverLimits) for item in limits_map.values()):
            raise TypeError("active_limits values must be ManeuverLimits")
        if any(not isinstance(item, WaterRelativeManeuverCommand) for item in command_map.values()):
            raise TypeError("active_commands values must be WaterRelativeManeuverCommand")

    pending_ids = {item.vessel_id for item in reservoir.pending_demands}
    already_active_pending = sorted(pending_ids & set(active_by_id))
    if already_active_pending:
        raise EnvironmentalNetworkSourceReservoirError(
            f"pending source demand already active in world: {already_active_pending!r}"
        )

    return _SourceReservoirResolutionContext(
        snapshot=snapshot,
        profile_map=profile_map,
        active_by_id=active_by_id,
        route_target_map=route_target_map,
        limits_map=limits_map,
        drift_map=drift_map,
        command_map=command_map,
        motion_state_map=motion_state_map,
        same_route_dynamic_ingress=same_route_dynamic_ingress,
        future_ingress_clearance=future_ingress_clearance,
        active_motion_backend=active_motion_backend,
    )


def _instantaneous_source_head_clearance(
    context: _SourceReservoirResolutionContext,
    demand: EnvironmentalNetworkSourceDemand,
    due_depth: int,
    accepted_virtual: list[tuple[EnvironmentalNetworkSourceDemand, VesselClassProfile]],
) -> tuple[
    VesselClassProfile | None,
    list[EnvironmentalNetworkSourceClearanceEvidence],
    EnvironmentalNetworkSourceHeadDecision | None,
]:
    """Stage 1: exact current-time domain evidence and deterministic precedence."""

    incoming_profile = demand.profile
    if incoming_profile is None or incoming_profile.safety_domain is None:
        return None, [], EnvironmentalNetworkSourceHeadDecision(
            stream_id=demand.stream_id,
            vessel_id=demand.vessel_id,
            demand_time=demand.demand_time,
            evaluated_at=context.snapshot.time,
            due_queue_depth=due_depth,
            status=EnvironmentalNetworkSourceHeadStatus.BLOCKED_MISSING_INCOMING_DOMAIN,
            missing_domain_vessel_ids=(demand.vessel_id,),
        )

    clearances: list[EnvironmentalNetworkSourceClearanceEvidence] = []
    missing_active = tuple(
        sorted(
            vessel_id
            for vessel_id in context.active_by_id
            if context.profile_map.get(vessel_id) is None
            or context.profile_map[vessel_id].safety_domain is None
        )
    )
    if missing_active:
        return incoming_profile, clearances, EnvironmentalNetworkSourceHeadDecision(
            stream_id=demand.stream_id,
            vessel_id=demand.vessel_id,
            demand_time=demand.demand_time,
            evaluated_at=context.snapshot.time,
            due_queue_depth=due_depth,
            status=EnvironmentalNetworkSourceHeadStatus.BLOCKED_MISSING_ACTIVE_DOMAIN,
            missing_domain_vessel_ids=missing_active,
        )

    active_intrusion = False
    for vessel_id in sorted(context.active_by_id):
        state = context.active_by_id[vessel_id]
        profile = context.profile_map[vessel_id]
        pair = _clearance(state, profile, demand.vessel, incoming_profile)
        clearances.append(
            EnvironmentalNetworkSourceClearanceEvidence(
                other_vessel_id=vessel_id,
                other_is_same_time_precedence_admission=False,
                clearance=pair,
            )
        )
        active_intrusion = active_intrusion or pair.intrudes
    if active_intrusion:
        return incoming_profile, clearances, EnvironmentalNetworkSourceHeadDecision(
            stream_id=demand.stream_id,
            vessel_id=demand.vessel_id,
            demand_time=demand.demand_time,
            evaluated_at=context.snapshot.time,
            due_queue_depth=due_depth,
            status=EnvironmentalNetworkSourceHeadStatus.BLOCKED_ACTIVE_DOMAIN_INTRUSION,
            clearances=tuple(clearances),
        )

    precedence_intrusion = False
    for prior, prior_profile in accepted_virtual:
        pair = _clearance(prior.vessel, prior_profile, demand.vessel, incoming_profile)
        clearances.append(
            EnvironmentalNetworkSourceClearanceEvidence(
                other_vessel_id=prior.vessel_id,
                other_is_same_time_precedence_admission=True,
                clearance=pair,
            )
        )
        precedence_intrusion = precedence_intrusion or pair.intrudes
    if precedence_intrusion:
        return incoming_profile, clearances, EnvironmentalNetworkSourceHeadDecision(
            stream_id=demand.stream_id,
            vessel_id=demand.vessel_id,
            demand_time=demand.demand_time,
            evaluated_at=context.snapshot.time,
            due_queue_depth=due_depth,
            status=EnvironmentalNetworkSourceHeadStatus.BLOCKED_PRECEDENCE_DOMAIN_INTRUSION,
            clearances=tuple(clearances),
        )

    return incoming_profile, clearances, None


def _dynamic_source_head_block(
    context: _SourceReservoirResolutionContext,
    demand: EnvironmentalNetworkSourceDemand,
    due_depth: int,
    incoming_profile: VesselClassProfile,
    clearances: list[EnvironmentalNetworkSourceClearanceEvidence],
    accepted_virtual: list[tuple[EnvironmentalNetworkSourceDemand, VesselClassProfile]],
) -> EnvironmentalNetworkSourceHeadDecision | None:
    """Stage 2: optional same-route longitudinal launch feasibility."""

    supervisor = context.same_route_dynamic_ingress
    if supervisor is None:
        return None

    incoming_limits = demand.limits
    incoming_following_config = supervisor.config_for_follower(demand.vessel_id)
    if incoming_limits is None:
        raise EnvironmentalNetworkSourceReservoirError(
            "dynamic ingress requires incoming maneuver limits and profile"
        )
    if incoming_following_config.require_zero_environmental_drift and (
        abs(demand.environmental_drift_velocity[0]) > _EPS
        or abs(demand.environmental_drift_velocity[1]) > _EPS
    ):
        raise SameRouteFollowingUnavailable(
            "dynamic ingress v1 is restricted to zero environmental drift"
        )

    incoming_route = demand.route_target.navigator.follower.route
    incoming_projection = same_route_safety_projection(incoming_route, demand.vessel.position)
    constraints = []

    for leader_id in sorted(context.active_by_id):
        target = context.route_target_map[leader_id]
        if target.navigator.follower.route != incoming_route:
            continue
        pair_following_config = supervisor.config_for_pair(demand.vessel_id, leader_id)
        drift = context.drift_map[leader_id]
        if pair_following_config.require_zero_environmental_drift and (
            abs(drift[0]) > _EPS or abs(drift[1]) > _EPS
        ):
            raise SameRouteFollowingUnavailable(
                "dynamic ingress v1 is restricted to zero environmental drift"
            )
        leader_state = context.active_by_id[leader_id]
        leader_projection = same_route_safety_projection(incoming_route, leader_state.position)
        if leader_projection.along_track_nm <= incoming_projection.along_track_nm + _EPS:
            continue
        if not same_route_longitudinal_pair_applicable(
            follower_state=demand.vessel,
            leader_state=leader_state,
            follower_domain=incoming_profile.safety_domain,
            leader_domain=context.profile_map[leader_id].safety_domain,
            follower_cross_track_nm=float(incoming_projection.cross_track_nm),
            leader_cross_track_nm=float(leader_projection.cross_track_nm),
            follower_route_heading_deg=float(incoming_projection.route_heading_deg),
            leader_route_heading_deg=float(leader_projection.route_heading_deg),
            config=pair_following_config,
        ):
            continue
        constraints.append(
            compute_same_route_following_constraint(
                follower_id=demand.vessel_id,
                leader_id=leader_id,
                follower_state=demand.vessel,
                leader_state=leader_state,
                follower_along_track_nm=float(incoming_projection.along_track_nm),
                leader_along_track_nm=float(leader_projection.along_track_nm),
                follower_cross_track_nm=float(incoming_projection.cross_track_nm),
                leader_cross_track_nm=float(leader_projection.cross_track_nm),
                follower_profile=incoming_profile,
                leader_profile=context.profile_map[leader_id],
                follower_limits=incoming_limits,
                leader_limits=context.limits_map[leader_id],
                route_heading_deg=float(incoming_projection.route_heading_deg),
                config=pair_following_config,
            )
        )

    for prior, prior_profile in accepted_virtual:
        if prior.route_target.navigator.follower.route != incoming_route:
            continue
        pair_following_config = supervisor.config_for_pair(demand.vessel_id, prior.vessel_id)
        if prior.limits is None:
            raise EnvironmentalNetworkSourceReservoirError(
                "dynamic ingress requires maneuver limits for same-time precedence admission"
            )
        if pair_following_config.require_zero_environmental_drift and (
            abs(prior.environmental_drift_velocity[0]) > _EPS
            or abs(prior.environmental_drift_velocity[1]) > _EPS
        ):
            raise SameRouteFollowingUnavailable(
                "dynamic ingress v1 is restricted to zero environmental drift"
            )
        leader_projection = same_route_safety_projection(incoming_route, prior.vessel.position)
        if leader_projection.along_track_nm <= incoming_projection.along_track_nm + _EPS:
            continue
        if not same_route_longitudinal_pair_applicable(
            follower_state=demand.vessel,
            leader_state=prior.vessel,
            follower_domain=incoming_profile.safety_domain,
            leader_domain=prior_profile.safety_domain,
            follower_cross_track_nm=float(incoming_projection.cross_track_nm),
            leader_cross_track_nm=float(leader_projection.cross_track_nm),
            follower_route_heading_deg=float(incoming_projection.route_heading_deg),
            leader_route_heading_deg=float(leader_projection.route_heading_deg),
            config=pair_following_config,
        ):
            continue
        constraints.append(
            compute_same_route_following_constraint(
                follower_id=demand.vessel_id,
                leader_id=prior.vessel_id,
                follower_state=demand.vessel,
                leader_state=prior.vessel,
                follower_along_track_nm=float(incoming_projection.along_track_nm),
                leader_along_track_nm=float(leader_projection.along_track_nm),
                follower_cross_track_nm=float(incoming_projection.cross_track_nm),
                leader_cross_track_nm=float(leader_projection.cross_track_nm),
                follower_profile=incoming_profile,
                leader_profile=prior_profile,
                follower_limits=incoming_limits,
                leader_limits=prior.limits,
                route_heading_deg=float(incoming_projection.route_heading_deg),
                config=pair_following_config,
            )
        )

    if not constraints:
        return None
    limiting = min(
        constraints,
        key=lambda item: (
            item.max_follower_speed_nm_per_min,
            item.projected_longitudinal_gap_nm,
            item.leader_id,
        ),
    )
    if demand.vessel.through_water_speed <= limiting.max_follower_speed_nm_per_min + _EPS:
        return None
    return EnvironmentalNetworkSourceHeadDecision(
        stream_id=demand.stream_id,
        vessel_id=demand.vessel_id,
        demand_time=demand.demand_time,
        evaluated_at=context.snapshot.time,
        due_queue_depth=due_depth,
        status=EnvironmentalNetworkSourceHeadStatus.BLOCKED_DYNAMIC_LAUNCH_INFEASIBLE,
        clearances=tuple(clearances),
        dynamic_launch_speed_ceiling_nm_per_min=limiting.max_follower_speed_nm_per_min,
        dynamic_limiting_leader_id=limiting.leader_id,
    )


def _future_source_head_block(
    context: _SourceReservoirResolutionContext,
    demand: EnvironmentalNetworkSourceDemand,
    due_depth: int,
    incoming_profile: VesselClassProfile,
    clearances: list[EnvironmentalNetworkSourceClearanceEvidence],
    accepted_virtual: list[tuple[EnvironmentalNetworkSourceDemand, VesselClassProfile]],
) -> EnvironmentalNetworkSourceHeadDecision | None:
    """Stage 3: optional full-horizon pair-command launch certificate."""

    future_config = context.future_ingress_clearance
    if future_config is None:
        return None
    if demand.limits is None:
        raise EnvironmentalNetworkSourceReservoirError(
            "future ingress clearance requires incoming maneuver limits"
        )
    launch_command = WaterRelativeManeuverCommand(
        target_through_water_heading_deg=demand.vessel.through_water_heading_deg,
        target_through_water_speed=demand.vessel.through_water_speed,
    )
    incoming_motion_state = (
        None if context.active_motion_backend is None else context.active_motion_backend.initialize_motion_state(
            demand.vessel, demand.limits, demand.environmental_drift_velocity
        )
    )
    incoming_transition = build_environmental_candidate_transition(
        demand.vessel,
        demand.environmental_drift_velocity,
        launch_command,
        demand.limits,
        motion_backend=context.active_motion_backend,
        motion_state=incoming_motion_state,
        evaluation_horizon_min=future_config.pair_clearance.evaluation_horizon_min,
    )
    future_evidence: list[EnvironmentalNetworkSourceFutureClearanceEvidence] = []
    future_blocked = False

    for vessel_id in sorted(context.active_by_id):
        profile = context.profile_map[vessel_id]
        assert profile.safety_domain is not None
        active_transition = build_environmental_candidate_transition(
            context.active_by_id[vessel_id],
            context.drift_map[vessel_id],
            context.command_map[vessel_id],
            context.limits_map[vessel_id],
            motion_backend=context.active_motion_backend,
            motion_state=context.motion_state_map.get(vessel_id),
            evaluation_horizon_min=future_config.pair_clearance.evaluation_horizon_min,
        )
        try:
            pair = evaluate_environmental_pair_command_clearance(
                active_transition,
                incoming_transition,
                future_config.pair_clearance,
                first_domain=profile.safety_domain,
                second_domain=incoming_profile.safety_domain,
            )
            clear = pair.certified_clear
            future_evidence.append(
                EnvironmentalNetworkSourceFutureClearanceEvidence(
                    other_vessel_id=vessel_id,
                    other_is_same_time_precedence_admission=False,
                    certified_clear=clear,
                    transition_certified_clear=pair.transition_certified_clear,
                    post_transition_certified_clear=pair.post_transition_certified_clear,
                    predicted_min_separation_nm=pair.predicted_min_separation_nm,
                    minimum_certified_domain_margin_nm=pair.minimum_certified_domain_margin_nm,
                )
            )
        except EnvironmentalPairTransitionUnavailable:
            clear = False
            future_evidence.append(
                EnvironmentalNetworkSourceFutureClearanceEvidence(
                    other_vessel_id=vessel_id,
                    other_is_same_time_precedence_admission=False,
                    certified_clear=False,
                    transition_certified_clear=False,
                    post_transition_certified_clear=False,
                    predicted_min_separation_nm=None,
                    minimum_certified_domain_margin_nm=None,
                )
            )
        future_blocked = future_blocked or not clear

    for prior, prior_profile in accepted_virtual:
        if prior.limits is None:
            raise EnvironmentalNetworkSourceReservoirError(
                "future ingress clearance requires same-time precedence maneuver limits"
            )
        assert prior_profile.safety_domain is not None
        prior_command = WaterRelativeManeuverCommand(
            target_through_water_heading_deg=prior.vessel.through_water_heading_deg,
            target_through_water_speed=prior.vessel.through_water_speed,
        )
        prior_motion_state = (
            None if context.active_motion_backend is None else context.active_motion_backend.initialize_motion_state(
                prior.vessel, prior.limits, prior.environmental_drift_velocity
            )
        )
        prior_transition = build_environmental_candidate_transition(
            prior.vessel,
            prior.environmental_drift_velocity,
            prior_command,
            prior.limits,
            motion_backend=context.active_motion_backend,
            motion_state=prior_motion_state,
            evaluation_horizon_min=future_config.pair_clearance.evaluation_horizon_min,
        )
        try:
            pair = evaluate_environmental_pair_command_clearance(
                prior_transition,
                incoming_transition,
                future_config.pair_clearance,
                first_domain=prior_profile.safety_domain,
                second_domain=incoming_profile.safety_domain,
            )
            clear = pair.certified_clear
            future_evidence.append(
                EnvironmentalNetworkSourceFutureClearanceEvidence(
                    other_vessel_id=prior.vessel_id,
                    other_is_same_time_precedence_admission=True,
                    certified_clear=clear,
                    transition_certified_clear=pair.transition_certified_clear,
                    post_transition_certified_clear=pair.post_transition_certified_clear,
                    predicted_min_separation_nm=pair.predicted_min_separation_nm,
                    minimum_certified_domain_margin_nm=pair.minimum_certified_domain_margin_nm,
                )
            )
        except EnvironmentalPairTransitionUnavailable:
            clear = False
            future_evidence.append(
                EnvironmentalNetworkSourceFutureClearanceEvidence(
                    other_vessel_id=prior.vessel_id,
                    other_is_same_time_precedence_admission=True,
                    certified_clear=False,
                    transition_certified_clear=False,
                    post_transition_certified_clear=False,
                    predicted_min_separation_nm=None,
                    minimum_certified_domain_margin_nm=None,
                )
            )
        future_blocked = future_blocked or not clear

    if not future_blocked:
        return None
    return EnvironmentalNetworkSourceHeadDecision(
        stream_id=demand.stream_id,
        vessel_id=demand.vessel_id,
        demand_time=demand.demand_time,
        evaluated_at=context.snapshot.time,
        due_queue_depth=due_depth,
        status=EnvironmentalNetworkSourceHeadStatus.BLOCKED_FUTURE_CLEARANCE_INFEASIBLE,
        clearances=tuple(clearances),
        future_clearances=tuple(future_evidence),
    )


def _resolve_due_source_head(
    context: _SourceReservoirResolutionContext,
    demand: EnvironmentalNetworkSourceDemand,
    due_depth: int,
    accepted_virtual: list[tuple[EnvironmentalNetworkSourceDemand, VesselClassProfile]],
) -> tuple[EnvironmentalNetworkSourceHeadDecision, EnvironmentalNetworkSourceAdmission | None]:
    """Run the invariant stages for one deterministic FIFO head."""

    incoming_profile, clearances, blocked = _instantaneous_source_head_clearance(
        context, demand, due_depth, accepted_virtual
    )
    if blocked is not None:
        return blocked, None
    assert incoming_profile is not None and incoming_profile.safety_domain is not None

    blocked = _dynamic_source_head_block(
        context,
        demand,
        due_depth,
        incoming_profile,
        clearances,
        accepted_virtual,
    )
    if blocked is not None:
        return blocked, None

    blocked = _future_source_head_block(
        context,
        demand,
        due_depth,
        incoming_profile,
        clearances,
        accepted_virtual,
    )
    if blocked is not None:
        return blocked, None

    record = demand.admission_record(context.snapshot.time)
    admission = EnvironmentalNetworkSourceAdmission(
        stream_id=demand.stream_id,
        vessel_id=demand.vessel_id,
        demand_time=demand.demand_time,
        admission_time=context.snapshot.time,
        record=record,
    )
    accepted_virtual.append((demand, incoming_profile))
    return EnvironmentalNetworkSourceHeadDecision(
        stream_id=demand.stream_id,
        vessel_id=demand.vessel_id,
        demand_time=demand.demand_time,
        evaluated_at=context.snapshot.time,
        due_queue_depth=due_depth,
        status=EnvironmentalNetworkSourceHeadStatus.ADMITTED,
        clearances=tuple(clearances),
    ), admission


def resolve_environmental_network_source_reservoir(
    reservoir: EnvironmentalNetworkSourceReservoir,
    snapshot: EnvironmentalWorldSnapshot,
    active_profiles: Mapping[str, VesselClassProfile],
    *,
    same_route_dynamic_ingress: SameRouteFollowingSupervisor | None = None,
    future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None = None,
    active_route_targets: Mapping[str, EnvironmentalRouteHandoffTarget] | None = None,
    active_limits: Mapping[str, ManeuverLimits] | None = None,
    active_environmental_drifts: Mapping[str, Vector2] | None = None,
    active_commands: Mapping[str, WaterRelativeManeuverCommand] | None = None,
    active_motion_backend: EnvironmentalMotionBackend | None = None,
    active_motion_states: Mapping[str, EnvironmentalMotionState | None] | None = None,
) -> EnvironmentalNetworkSourceResolution:
    """Resolve all currently due per-stream FIFO heads against exact entry evidence.

    The function is intentionally arm-local.  It may be called independently for a
    baseline and a controlled arm that share the same ``demands`` but have different
    physical snapshots.  This is the corrected A147 CRN boundary: stochastic demand is
    shared, while waiting/admission is an outcome of each arm.
    """

    context = _prepare_source_reservoir_resolution_context(
        reservoir,
        snapshot,
        active_profiles,
        same_route_dynamic_ingress=same_route_dynamic_ingress,
        future_ingress_clearance=future_ingress_clearance,
        active_route_targets=active_route_targets,
        active_limits=active_limits,
        active_environmental_drifts=active_environmental_drifts,
        active_commands=active_commands,
        active_motion_backend=active_motion_backend,
        active_motion_states=active_motion_states,
    )

    accepted_virtual: list[tuple[EnvironmentalNetworkSourceDemand, VesselClassProfile]] = []
    decisions: list[EnvironmentalNetworkSourceHeadDecision] = []
    new_admissions: list[EnvironmentalNetworkSourceAdmission] = []

    for demand, due_depth in _due_heads(reservoir, snapshot.time):
        decision, admission = _resolve_due_source_head(
            context, demand, due_depth, accepted_virtual
        )
        decisions.append(decision)
        if admission is not None:
            new_admissions.append(admission)

    next_reservoir = EnvironmentalNetworkSourceReservoir(
        demands=reservoir.demands,
        admissions=reservoir.admissions + tuple(new_admissions),
        service_order_by_stream=reservoir.service_order_by_stream,
    )
    return EnvironmentalNetworkSourceResolution(
        snapshot_time=snapshot.time,
        decisions=tuple(decisions),
        admissions=tuple(new_admissions),
        reservoir=next_reservoir,
        metrics=source_queue_metrics(next_reservoir, at=snapshot.time),
    )
