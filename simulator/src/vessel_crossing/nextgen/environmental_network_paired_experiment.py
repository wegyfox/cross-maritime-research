"""Common-random-number paired nominal-baseline vs avoidance-control experiments.

A139 is the first comparative research protocol on top of the A137/A138 product path.
One Poisson network realization is generated exactly once and is then reused as the
exogenous traffic input for both arms:

* ``baseline`` follows the same environmental routes under the same drift/limits/profile
  evidence but deliberately performs no encounter avoidance;
* ``control`` uses the full A137 persistent PAIR/JOINT avoidance runtime.

Safety evidence is recorded read-only from every constant-command physical interval of
an ``EnvironmentalWorld2D`` subclass.  The recorder does not schedule commands or create
registry observations.  Continuous oriented-domain certificates reuse the existing
adaptive domain-clearance machinery with exact environmental state queries.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import combinations
from math import hypot, isfinite

from ..archive import environment_manifest
from ..version import __version__

from .domain_clearance import DomainClearanceCertification, certify_pair_domain_clearance
from .environmental_candidate_transition import EnvironmentalCommandTrajectory, build_environmental_candidate_transition
from .dynamics import ManeuverLimits
from .encounter_registry import EncounterRegistry
from .environmental_dynamics import (
    WaterRelativeManeuverCommand,
    WaterRelativeVesselState,
    advance_with_uniform_drift,
    water_relative_command_transition_duration,
)
from .environmental_fleet_runtime import start_environmental_fleet_runtime
from .environmental_network_experiment import (
    EnvironmentalNetworkExperimentConfig,
    EnvironmentalNetworkExperimentResult,
    _canonical_json,
    _json_value,
    _summary,
)
from .environmental_network_scenario import (
    EnvironmentalNetworkScenarioIngress,
    EnvironmentalNetworkScenarioResult,
    _environmental_ingress_records,
)
from .environmental_nominal_route_session import EnvironmentalNominalRouteSessionConfig
from .environmental_route_handoff import (
    EnvironmentalGroundVelocityUnavailable,
    EnvironmentalRouteHandoffTarget,
    command_for_desired_ground_velocity,
)
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .follower import RouteFollowerState
from .mission_safety import (
    MissionSafetyExperimentSummary,
    MissionSafetySummaryAccumulator,
    TimedPairSafetyIntervalMetric,
    summarize_mission_safety,
)
from .network_traffic import PoissonNetworkTrafficRealization, generate_poisson_network_traffic
from .route import marine_bearing_between
from .route_exit import RouteEndAction
from .safety_metrics import (
    PairSafetyIntervalMetric,
    PairSafetyMetricConfig,
    PairSafetyOutcome,
    SafetyMetricUnavailable,
    modeled_hull_domain,
)
from .ship_domain import ShipDomain
from .vessel_classes import VesselClassProfile, VesselProfileCatalog

_EPS = 1e-10
_PAIR_CONFIG_FORMAT = "vessel-crossing-nextgen-environmental-network-paired-config"
_PAIR_CONFIG_SCHEMA_VERSION = 1
_PAIR_RESULT_FORMAT = "vessel-crossing-nextgen-environmental-network-paired-result"
_PAIR_RESULT_SCHEMA_VERSION = 1
_PAIR_DOMAIN = b"vessel-crossing-nextgen-paired-experiment\x00"
class EnvironmentalNetworkPairedExperimentError(RuntimeError):
    """Raised when the paired protocol cannot be preflighted or executed."""


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkPairedExperimentConfig:
    """One CRN comparison: fixed nominal baseline versus the A138 avoidance control."""

    control: EnvironmentalNetworkExperimentConfig
    safety: PairSafetyMetricConfig

    def __post_init__(self) -> None:
        if not isinstance(self.control, EnvironmentalNetworkExperimentConfig):
            raise TypeError("control must be EnvironmentalNetworkExperimentConfig")
        if not isinstance(self.safety, PairSafetyMetricConfig):
            raise TypeError("safety must be PairSafetyMetricConfig")
        for stream in self.control.streams:
            profile = stream.vessel_class
            if profile is None or profile.safety_domain is None:
                raise EnvironmentalNetworkPairedExperimentError(
                    f"paired safety protocol requires vessel_class with safety_domain for stream {stream.stream_id!r}"
                )

    def to_json_object(self) -> dict[str, object]:
        return {
            "format": _PAIR_CONFIG_FORMAT,
            "schema_version": _PAIR_CONFIG_SCHEMA_VERSION,
            "software_version": __version__,
            "control": self.control.to_json_object(),
            "control_config_sha256": self.control.config_sha256,
            "control_run_id": self.control.run_id,
            "safety": _json_value(self.safety),
            "baseline_policy": "nominal_route_no_encounter_avoidance",
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def config_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()

    @property
    def pair_id(self) -> str:
        payload = _canonical_json(
            {
                "control_config_sha256": self.control.config_sha256,
                "control_run_id": self.control.run_id,
                "safety": _json_value(self.safety),
                "baseline_policy": "nominal_route_no_encounter_avoidance",
            }
        )
        return sha256(_PAIR_DOMAIN + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkRouteCompletion:
    vessel_id: str
    demand_time_min: float
    admission_time_min: float
    completion_time_min: float
    end_action: RouteEndAction

    @property
    def arrival_time_min(self) -> float:
        """Compatibility alias for the physical admission timestamp."""
        return self.admission_time_min

    @property
    def ingress_waiting_time_min(self) -> float:
        return self.admission_time_min - self.demand_time_min

    @property
    def route_duration_min(self) -> float:
        return self.completion_time_min - self.admission_time_min

    @property
    def system_duration_min(self) -> float:
        return self.completion_time_min - self.demand_time_min


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkArmMetrics:
    realized_vessel_count: int
    route_completed_count: int
    route_completion_fraction: float | None
    route_completion_rate_per_hour: float | None
    mean_route_duration_min: float | None
    final_active_vessel_count: int
    safety: MissionSafetyExperimentSummary


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkPairedDifference:
    """Control-minus-baseline differences; negative safety counts favor control."""

    route_completed_count: int
    route_completion_rate_per_hour: float | None
    mean_route_duration_min: float | None
    modeled_hull_overlap_episode_count: int
    safety_domain_intrusion_episode_count: int
    near_miss_episode_count: int
    unresolved_episode_count: int
    nonclear_pair_exposure_min: float


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkNominalBaselineResult:
    traffic: PoissonNetworkTrafficRealization
    ingress: tuple[EnvironmentalNetworkScenarioIngress, ...]
    final_snapshot: EnvironmentalWorldSnapshot
    final_commands: tuple[tuple[str, WaterRelativeManeuverCommand], ...]
    completions: tuple[EnvironmentalNetworkRouteCompletion, ...]
    safety_intervals: tuple[TimedPairSafetyIntervalMetric, ...]
    metrics: EnvironmentalNetworkArmMetrics


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkPairedExperimentResult:
    config: EnvironmentalNetworkPairedExperimentConfig
    pair_id: str
    traffic: PoissonNetworkTrafficRealization
    baseline: EnvironmentalNetworkNominalBaselineResult
    control: EnvironmentalNetworkExperimentResult
    control_completions: tuple[EnvironmentalNetworkRouteCompletion, ...]
    control_safety_intervals: tuple[TimedPairSafetyIntervalMetric, ...]
    control_metrics: EnvironmentalNetworkArmMetrics
    difference: EnvironmentalNetworkPairedDifference

    def to_json_object(self) -> dict[str, object]:
        return {
            "format": _PAIR_RESULT_FORMAT,
            "schema_version": _PAIR_RESULT_SCHEMA_VERSION,
            "pair_id": self.pair_id,
            "config_sha256": self.config.config_sha256,
            "config": self.config.to_json_object(),
            "execution_environment": environment_manifest(),
            "traffic": _json_value(self.traffic),
            "baseline": {
                "ingress": _json_value(self.baseline.ingress),
                "final_snapshot": _json_value(self.baseline.final_snapshot),
                "final_commands": _json_value(self.baseline.final_commands),
                "completions": _json_value(self.baseline.completions),
                "safety_intervals": _json_value(self.baseline.safety_intervals),
                "metrics": _json_value(self.baseline.metrics),
            },
            "control": self.control.to_json_object(),
            "control_completions": _json_value(self.control_completions),
            "control_safety_intervals": _json_value(self.control_safety_intervals),
            "control_metrics": _json_value(self.control_metrics),
            "difference": _json_value(self.difference),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def result_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class _DomainState:
    position: tuple[float, float]
    heading_deg: float


@dataclass(slots=True)
class _BaselineCursor:
    target: EnvironmentalRouteHandoffTarget
    follower_state: RouteFollowerState
    demand_time: float
    admission_time: float
    next_review_at: float | None = None

    @property
    def vessel_id(self) -> str:
        return self.target.vessel_id


def _domain_state_at(
    state: WaterRelativeVesselState,
    command: WaterRelativeManeuverCommand,
    limits: ManeuverLimits,
    drift: tuple[float, float],
    at: float,
) -> _DomainState:
    advanced = advance_with_uniform_drift(
        state,
        command,
        at,
        limits,
        environmental_drift_velocity=drift,
    ).state
    return _DomainState(advanced.position, advanced.through_water_heading_deg)


def _environmental_certificate(
    *,
    first_state: WaterRelativeVesselState,
    second_state: WaterRelativeVesselState,
    first_command: WaterRelativeManeuverCommand,
    second_command: WaterRelativeManeuverCommand,
    first_limits: ManeuverLimits,
    second_limits: ManeuverLimits,
    first_drift: tuple[float, float],
    second_drift: tuple[float, float],
    first_domain: ShipDomain,
    second_domain: ShipDomain,
    duration_min: float,
    config: PairSafetyMetricConfig,
) -> DomainClearanceCertification:
    first_speed_bound = max(first_state.through_water_speed, first_command.target_through_water_speed) + hypot(*first_drift)
    second_speed_bound = max(second_state.through_water_speed, second_command.target_through_water_speed) + hypot(*second_drift)
    return certify_pair_domain_clearance(
        first_state_at=lambda t: _domain_state_at(first_state, first_command, first_limits, first_drift, t),
        second_state_at=lambda t: _domain_state_at(second_state, second_command, second_limits, second_drift, t),
        first_domain=first_domain,
        second_domain=second_domain,
        duration_min=duration_min,
        first_speed_bound_nm_per_min=first_speed_bound,
        second_speed_bound_nm_per_min=second_speed_bound,
        first_turn_rate_bound_deg_per_min=first_limits.max_turn_rate_deg_per_min,
        second_turn_rate_bound_deg_per_min=second_limits.max_turn_rate_deg_per_min,
        time_tolerance_min=config.time_tolerance_min,
        margin_tolerance_nm=config.margin_tolerance_nm,
        max_subdivisions=config.max_subdivisions,
    )


def evaluate_environmental_pair_safety_interval(
    *,
    first_state: WaterRelativeVesselState,
    second_state: WaterRelativeVesselState,
    first_command: WaterRelativeManeuverCommand,
    second_command: WaterRelativeManeuverCommand,
    first_limits: ManeuverLimits,
    second_limits: ManeuverLimits,
    first_drift: tuple[float, float],
    second_drift: tuple[float, float],
    first_profile: VesselClassProfile,
    second_profile: VesselClassProfile,
    duration_min: float,
    config: PairSafetyMetricConfig,
) -> PairSafetyIntervalMetric:
    """Continuous environmental safety metric over one constant-command interval."""

    if first_profile.safety_domain is None or second_profile.safety_domain is None:
        raise SafetyMetricUnavailable("environmental safety metric requires both safety domains")
    if first_state.vessel_id == second_state.vessel_id:
        raise ValueError("pair safety metric requires two distinct vessel ids")
    if not isfinite(duration_min) or duration_min < 0.0:
        raise ValueError("duration_min must be finite and non-negative")

    hull = _environmental_certificate(
        first_state=first_state,
        second_state=second_state,
        first_command=first_command,
        second_command=second_command,
        first_limits=first_limits,
        second_limits=second_limits,
        first_drift=first_drift,
        second_drift=second_drift,
        first_domain=modeled_hull_domain(first_profile.dimensions),
        second_domain=modeled_hull_domain(second_profile.dimensions),
        duration_min=duration_min,
        config=config,
    )
    safety = _environmental_certificate(
        first_state=first_state,
        second_state=second_state,
        first_command=first_command,
        second_command=second_command,
        first_limits=first_limits,
        second_limits=second_limits,
        first_drift=first_drift,
        second_drift=second_drift,
        first_domain=first_profile.safety_domain,
        second_domain=second_profile.safety_domain,
        duration_min=duration_min,
        config=config,
    )

    if hull.sampled_min_margin_nm <= _EPS:
        outcome = PairSafetyOutcome.MODELED_HULL_OVERLAP
    elif hull.conservative_lower_bound_nm <= _EPS:
        outcome = PairSafetyOutcome.HULL_CLEARANCE_UNRESOLVED
    elif safety.sampled_min_margin_nm <= _EPS:
        outcome = PairSafetyOutcome.SAFETY_DOMAIN_INTRUSION
    elif safety.conservative_lower_bound_nm <= _EPS:
        outcome = PairSafetyOutcome.SAFETY_DOMAIN_CLEARANCE_UNRESOLVED
    elif safety.sampled_min_margin_nm <= config.near_miss_margin_nm + _EPS:
        outcome = PairSafetyOutcome.NEAR_MISS
    elif safety.conservative_lower_bound_nm <= config.near_miss_margin_nm + _EPS:
        outcome = PairSafetyOutcome.NEAR_MISS_UNRESOLVED
    else:
        outcome = PairSafetyOutcome.CLEAR

    return PairSafetyIntervalMetric(
        first_vessel_id=first_state.vessel_id,
        second_vessel_id=second_state.vessel_id,
        duration_min=duration_min,
        outcome=outcome,
        hull_clearance=hull,
        safety_domain_clearance=safety,
        near_miss_margin_nm=config.near_miss_margin_nm,
    )


def _trajectory_domain_state_at(
    trajectory: EnvironmentalCommandTrajectory,
    at: float,
) -> _DomainState:
    state = trajectory.state_at_or_continue(at)
    return _DomainState(state.position, state.through_water_heading_deg)


def _trajectory_certificate(
    *,
    first_trajectory: EnvironmentalCommandTrajectory,
    second_trajectory: EnvironmentalCommandTrajectory,
    first_domain: ShipDomain,
    second_domain: ShipDomain,
    duration_min: float,
    config: PairSafetyMetricConfig,
) -> DomainClearanceCertification:
    """Certify one interval using the exact selected environmental backend trajectories."""

    return certify_pair_domain_clearance(
        first_state_at=lambda t: _trajectory_domain_state_at(first_trajectory, t),
        second_state_at=lambda t: _trajectory_domain_state_at(second_trajectory, t),
        first_domain=first_domain,
        second_domain=second_domain,
        duration_min=duration_min,
        first_speed_bound_nm_per_min=first_trajectory.ground_speed_bound,
        second_speed_bound_nm_per_min=second_trajectory.ground_speed_bound,
        first_turn_rate_bound_deg_per_min=first_trajectory.turn_rate_bound_deg_per_min,
        second_turn_rate_bound_deg_per_min=second_trajectory.turn_rate_bound_deg_per_min,
        time_tolerance_min=config.time_tolerance_min,
        margin_tolerance_nm=config.margin_tolerance_nm,
        max_subdivisions=config.max_subdivisions,
    )


def evaluate_environmental_pair_safety_trajectories(
    *,
    first_trajectory: EnvironmentalCommandTrajectory,
    second_trajectory: EnvironmentalCommandTrajectory,
    first_profile: VesselClassProfile,
    second_profile: VesselClassProfile,
    duration_min: float,
    config: PairSafetyMetricConfig,
) -> PairSafetyIntervalMetric:
    """Measure safety on backend-native trajectories instead of assuming A162 dynamics.

    This is telemetry only.  It never changes arbitration, commands, or backend state.
    Historical A162 runs continue to use :func:`evaluate_environmental_pair_safety_interval`
    so their numerical evidence remains unchanged.
    """

    if first_profile.safety_domain is None or second_profile.safety_domain is None:
        raise SafetyMetricUnavailable("environmental safety metric requires both safety domains")
    first_id = first_trajectory.initial_state.vessel_id
    second_id = second_trajectory.initial_state.vessel_id
    if first_id == second_id:
        raise ValueError("pair safety metric requires two distinct vessel ids")
    if not isfinite(duration_min) or duration_min < 0.0:
        raise ValueError("duration_min must be finite and non-negative")
    if not first_trajectory.reachable or not second_trajectory.reachable:
        raise SafetyMetricUnavailable("backend trajectory is not physically reachable")

    hull = _trajectory_certificate(
        first_trajectory=first_trajectory,
        second_trajectory=second_trajectory,
        first_domain=modeled_hull_domain(first_profile.dimensions),
        second_domain=modeled_hull_domain(second_profile.dimensions),
        duration_min=duration_min,
        config=config,
    )
    safety = _trajectory_certificate(
        first_trajectory=first_trajectory,
        second_trajectory=second_trajectory,
        first_domain=first_profile.safety_domain,
        second_domain=second_profile.safety_domain,
        duration_min=duration_min,
        config=config,
    )

    if hull.sampled_min_margin_nm <= _EPS:
        outcome = PairSafetyOutcome.MODELED_HULL_OVERLAP
    elif hull.conservative_lower_bound_nm <= _EPS:
        outcome = PairSafetyOutcome.HULL_CLEARANCE_UNRESOLVED
    elif safety.sampled_min_margin_nm <= _EPS:
        outcome = PairSafetyOutcome.SAFETY_DOMAIN_INTRUSION
    elif safety.conservative_lower_bound_nm <= _EPS:
        outcome = PairSafetyOutcome.SAFETY_DOMAIN_CLEARANCE_UNRESOLVED
    elif safety.sampled_min_margin_nm <= config.near_miss_margin_nm + _EPS:
        outcome = PairSafetyOutcome.NEAR_MISS
    elif safety.conservative_lower_bound_nm <= config.near_miss_margin_nm + _EPS:
        outcome = PairSafetyOutcome.NEAR_MISS_UNRESOLVED
    else:
        outcome = PairSafetyOutcome.CLEAR

    return PairSafetyIntervalMetric(
        first_vessel_id=first_id,
        second_vessel_id=second_id,
        duration_min=duration_min,
        outcome=outcome,
        hull_clearance=hull,
        safety_domain_clearance=safety,
        near_miss_margin_nm=config.near_miss_margin_nm,
    )


class _MetricEnvironmentalWorld(EnvironmentalWorld2D):
    """A139-only read-only physical-interval recorder."""

    def __init__(
        self,
        *args,
        profile_catalog: VesselProfileCatalog,
        safety_config: PairSafetyMetricConfig,
        retain_safety_intervals: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._metric_profile_catalog = profile_catalog
        self._metric_config = safety_config
        self._retain_safety_intervals = bool(retain_safety_intervals)
        self._metric_intervals: list[TimedPairSafetyIntervalMetric] = []
        self._metric_summary = MissionSafetySummaryAccumulator()

    @property
    def safety_intervals(self) -> tuple[TimedPairSafetyIntervalMetric, ...]:
        return tuple(self._metric_intervals)

    @property
    def safety_interval_count(self) -> int:
        return self._metric_summary.interval_count

    def finalize_safety_experiment_summary(self) -> MissionSafetyExperimentSummary:
        return self._metric_summary.finalize()

    def _advance_all(self, duration: float):  # noqa: SLF001 - deliberate telemetry subclass
        if duration > _EPS:
            snapshot = self.snapshot()
            by_id = {item.vessel_id: item for item in snapshot.vessels}
            for first_id, second_id in combinations(sorted(by_id), 2):
                first_profile = self._metric_profile_catalog.profile_for(first_id)
                second_profile = self._metric_profile_catalog.profile_for(second_id)
                if first_profile is None or second_profile is None:
                    missing = first_id if first_profile is None else second_id
                    raise SafetyMetricUnavailable(f"missing environmental safety profile for {missing!r}")
                if self.motion_backend is None:
                    metric = evaluate_environmental_pair_safety_interval(
                        first_state=by_id[first_id],
                        second_state=by_id[second_id],
                        first_command=self.command_for(first_id),
                        second_command=self.command_for(second_id),
                        first_limits=self.limits_for(first_id),
                        second_limits=self.limits_for(second_id),
                        first_drift=self.drift_for(first_id),
                        second_drift=self.drift_for(second_id),
                        first_profile=first_profile,
                        second_profile=second_profile,
                        duration_min=duration,
                        config=self._metric_config,
                    )
                else:
                    first_trajectory = build_environmental_candidate_transition(
                        by_id[first_id],
                        self.drift_for(first_id),
                        self.command_for(first_id),
                        self.limits_for(first_id),
                        motion_backend=self.motion_backend,
                        motion_state=self.motion_state_for(first_id),
                        evaluation_horizon_min=duration,
                    )
                    second_trajectory = build_environmental_candidate_transition(
                        by_id[second_id],
                        self.drift_for(second_id),
                        self.command_for(second_id),
                        self.limits_for(second_id),
                        motion_backend=self.motion_backend,
                        motion_state=self.motion_state_for(second_id),
                        evaluation_horizon_min=duration,
                    )
                    metric = evaluate_environmental_pair_safety_trajectories(
                        first_trajectory=first_trajectory,
                        second_trajectory=second_trajectory,
                        first_profile=first_profile,
                        second_profile=second_profile,
                        duration_min=duration,
                        config=self._metric_config,
                    )
                timed = TimedPairSafetyIntervalMetric(
                    snapshot.time,
                    snapshot.time + duration,
                    metric,
                )
                self._metric_summary.observe(timed)
                if self._retain_safety_intervals:
                    self._metric_intervals.append(timed)
        return super()._advance_all(duration)


def _run_control_from_shared_traffic(
    config: EnvironmentalNetworkExperimentConfig,
    traffic: PoissonNetworkTrafficRealization,
    safety: PairSafetyMetricConfig,
) -> tuple[EnvironmentalNetworkExperimentResult, tuple[TimedPairSafetyIntervalMetric, ...]]:
    scenario_config = config.scenario
    ingress = _environmental_ingress_records(traffic, scenario_config)
    profiles = VesselProfileCatalog()
    world = _MetricEnvironmentalWorld(
        (),
        limits={},
        environmental_drift_velocity={},
        start_time=scenario_config.start_time,
        profile_catalog=profiles,
        safety_config=safety,
    )
    registry = EncounterRegistry(
        discovery=scenario_config.discovery,
        lifecycle=scenario_config.lifecycle,
        profile_catalog=profiles,
    )
    started = start_environmental_fleet_runtime(
        registry,
        world,
        {},
        recapture_config=scenario_config.recapture,
        config=scenario_config.runtime,
        profile_catalog=profiles,
    )
    admissions = tuple(started.session.schedule_ingress(item.record) for item in ingress)
    nominal_owner_ids: list[str] = []
    index = 0
    while index < len(ingress):
        boundary = ingress[index].at
        batch: list[str] = []
        while index < len(ingress) and abs(ingress[index].at - boundary) <= _EPS:
            batch.append(ingress[index].vessel_id)
            index += 1
        started.session.run_until(boundary)
        nominal_owner_ids.extend(started.session.admit_current_clear_route_targets(tuple(batch)))
    run = started.session.run_until(scenario_config.end_time)
    scenario = EnvironmentalNetworkScenarioResult(
        traffic=traffic,
        ingress=ingress,
        admissions=admissions,
        runtime_start=started,
        run=run,
        world=world,
        registry=registry,
        profile_catalog=profiles,
        nominal_route_owner_ids=tuple(nominal_owner_ids),
    )
    result = EnvironmentalNetworkExperimentResult(config, _summary(config, scenario), scenario)
    return result, world.safety_intervals


def _baseline_projection(world: EnvironmentalWorld2D, cursor: _BaselineCursor, config: EnvironmentalNominalRouteSessionConfig):
    follower = cursor.target.navigator.follower
    route = follower.route
    state = world.snapshot().by_id(cursor.vessel_id)
    min_s = min(cursor.follower_state.along_track, route.length)
    projection = route.project(state.position, min_along_track=min_s)
    progress = max(min_s, projection.along_track)
    completed = (
        progress >= route.length - follower.position_tolerance
        and abs(projection.cross_track) <= max(follower.position_tolerance, config.cross_track_tolerance_nm)
    )
    cursor.follower_state = replace(
        cursor.follower_state,
        along_track=min(progress, route.length),
        active_turn_waypoint=None,
        completed=completed,
    )
    return projection


def _refresh_baseline(
    world: EnvironmentalWorld2D,
    profiles: VesselProfileCatalog,
    cursors: dict[str, _BaselineCursor],
    due_ids: set[str],
    config: EnvironmentalNominalRouteSessionConfig,
    completions: list[EnvironmentalNetworkRouteCompletion],
) -> None:
    commands: list[tuple[str, WaterRelativeManeuverCommand]] = []
    completed: list[str] = []
    for vessel_id in sorted(due_ids):
        cursor = cursors[vessel_id]
        follower = cursor.target.navigator.follower
        route = follower.route
        state = world.snapshot().by_id(vessel_id)
        projection = _baseline_projection(world, cursor, config)
        if cursor.follower_state.completed:
            completed.append(vessel_id)
            continue
        base_s = max(cursor.follower_state.along_track, projection.along_track)
        target_s = min(route.length, base_s + config.route_lookahead_nm)
        target_point = route.point_at(target_s)
        if hypot(target_point[0] - state.position[0], target_point[1] - state.position[1]) <= _EPS:
            desired_heading = route.heading_at(min(base_s, route.length))
        else:
            desired_heading = marine_bearing_between(state.position, target_point)
        profile = profiles.profile_for(vessel_id)
        try:
            target = command_for_desired_ground_velocity(
                state,
                world.drift_for(vessel_id),
                desired_ground_heading_deg=desired_heading,
                desired_ground_speed=follower.cruise_speed,
                profile=profile,
            )
        except EnvironmentalGroundVelocityUnavailable as exc:
            raise EnvironmentalNetworkPairedExperimentError(
                f"nominal baseline route command unavailable for {vessel_id!r}"
            ) from exc
        duration = water_relative_command_transition_duration(state, target.command, world.limits_for(vessel_id))
        if not isfinite(duration):
            raise EnvironmentalNetworkPairedExperimentError(
                f"nominal baseline route command unreachable for {vessel_id!r}"
            )
        commands.append((vessel_id, target.command))
        cursor.next_review_at = world.time + config.review_interval_min

    if commands:
        world.schedule_commands_atomic(at=world.time, commands=commands)
        world.run_until(world.time)

    for vessel_id in completed:
        cursor = cursors.pop(vessel_id)
        action = cursor.target.end_action
        completions.append(
            EnvironmentalNetworkRouteCompletion(
                vessel_id, cursor.demand_time, cursor.admission_time, world.time, action
            )
        )
        if action is RouteEndAction.STOP:
            current = world.command_for(vessel_id)
            world.schedule_commands_atomic(
                at=world.time,
                commands=((vessel_id, WaterRelativeManeuverCommand(current.target_through_water_heading_deg, 0.0)),),
            )
            world.run_until(world.time)
        elif action is RouteEndAction.REMOVE:
            world.remove_vessel(vessel_id)
            profiles.remove(vessel_id)
        elif action is not RouteEndAction.CONTINUE:
            raise AssertionError(action)


def _run_nominal_baseline(
    traffic: PoissonNetworkTrafficRealization,
    ingress: tuple[EnvironmentalNetworkScenarioIngress, ...],
    config: EnvironmentalNetworkExperimentConfig,
    safety: PairSafetyMetricConfig,
) -> EnvironmentalNetworkNominalBaselineResult:
    profiles = VesselProfileCatalog()
    for item in ingress:
        assert item.record.profile is not None
        profiles.assign(item.vessel_id, item.record.profile)
    scenario = config.scenario
    world = _MetricEnvironmentalWorld(
        (),
        limits={},
        environmental_drift_velocity={},
        start_time=scenario.start_time,
        profile_catalog=profiles,
        safety_config=safety,
    )
    route_config = scenario.runtime.mixed.nominal_route
    cursors: dict[str, _BaselineCursor] = {}
    completions: list[EnvironmentalNetworkRouteCompletion] = []
    index = 0
    max_boundaries = 100000
    for _ in range(max_boundaries):
        if world.time >= scenario.end_time - _EPS:
            break
        candidates = [float(scenario.end_time)]
        if index < len(ingress):
            candidates.append(ingress[index].at)
        candidates.extend(
            cursor.next_review_at for cursor in cursors.values() if cursor.next_review_at is not None
        )
        boundary = min(value for value in candidates if value >= world.time - _EPS)
        world.run_until(boundary)

        due: set[str] = {
            vessel_id
            for vessel_id, cursor in cursors.items()
            if cursor.next_review_at is not None and cursor.next_review_at <= world.time + _EPS
        }
        while index < len(ingress) and abs(ingress[index].at - world.time) <= _EPS:
            record = ingress[index].record
            world.add_vessel(
                record.vessel,
                limits=record.effective_limits,
                environmental_drift_velocity=record.environmental_drift_velocity,
            )
            cursors[record.vessel_id] = _BaselineCursor(
                target=record.route_target,
                follower_state=record.route_target.follower_state,
                demand_time=ingress[index].demand_time,
                admission_time=record.at,
                next_review_at=world.time,
            )
            due.add(record.vessel_id)
            index += 1
        if due:
            _refresh_baseline(world, profiles, cursors, due, route_config, completions)
    else:
        raise EnvironmentalNetworkPairedExperimentError("nominal baseline exceeded boundary limit")

    if world.time < scenario.end_time - _EPS:
        world.run_until(scenario.end_time)
    metrics = _arm_metrics(
        traffic=traffic,
        start_time=scenario.start_time,
        end_time=float(scenario.end_time),
        final_snapshot=world.snapshot(),
        completions=tuple(completions),
        intervals=world.safety_intervals,
    )
    final_snapshot = world.snapshot()
    final_commands = tuple(
        (vessel.vessel_id, world.command_for(vessel.vessel_id))
        for vessel in final_snapshot.vessels
    )
    return EnvironmentalNetworkNominalBaselineResult(
        traffic=traffic,
        ingress=ingress,
        final_snapshot=final_snapshot,
        final_commands=final_commands,
        completions=tuple(completions),
        safety_intervals=world.safety_intervals,
        metrics=metrics,
    )


def _control_completions(result: EnvironmentalNetworkExperimentResult) -> tuple[EnvironmentalNetworkRouteCompletion, ...]:
    ingress_by_vessel = {item.vessel_id: item for item in result.scenario.ingress}
    items: list[EnvironmentalNetworkRouteCompletion] = []
    for disposition in result.scenario.runtime_start.session.persistent.disposition_results:
        for vessel in disposition.dispositions:
            items.append(
                EnvironmentalNetworkRouteCompletion(
                    vessel.vessel_id,
                    ingress_by_vessel[vessel.vessel_id].demand_time,
                    ingress_by_vessel[vessel.vessel_id].admission_time,
                    disposition.committed_observation_time,
                    vessel.action,
                )
            )
    return tuple(sorted(items, key=lambda item: (item.completion_time_min, item.vessel_id)))


def _arm_metrics(
    *,
    traffic: PoissonNetworkTrafficRealization,
    start_time: float,
    end_time: float,
    final_snapshot: EnvironmentalWorldSnapshot,
    completions: tuple[EnvironmentalNetworkRouteCompletion, ...],
    intervals: tuple[TimedPairSafetyIntervalMetric, ...],
) -> EnvironmentalNetworkArmMetrics:
    count = len(completions)
    realized = traffic.count
    duration = end_time - start_time
    rate = None if duration <= _EPS else count / duration * 60.0
    mean_duration = None if not completions else sum(item.route_duration_min for item in completions) / count
    fraction = None if realized == 0 else count / realized
    return EnvironmentalNetworkArmMetrics(
        realized_vessel_count=realized,
        route_completed_count=count,
        route_completion_fraction=fraction,
        route_completion_rate_per_hour=rate,
        mean_route_duration_min=mean_duration,
        final_active_vessel_count=len(final_snapshot.vessels),
        safety=summarize_mission_safety(intervals),
    )


def _difference(
    control: EnvironmentalNetworkArmMetrics,
    baseline: EnvironmentalNetworkArmMetrics,
) -> EnvironmentalNetworkPairedDifference:
    def optional_delta(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else a - b

    cs = control.safety
    bs = baseline.safety
    return EnvironmentalNetworkPairedDifference(
        route_completed_count=control.route_completed_count - baseline.route_completed_count,
        route_completion_rate_per_hour=optional_delta(control.route_completion_rate_per_hour, baseline.route_completion_rate_per_hour),
        mean_route_duration_min=optional_delta(control.mean_route_duration_min, baseline.mean_route_duration_min),
        modeled_hull_overlap_episode_count=cs.modeled_hull_overlap_episode_count - bs.modeled_hull_overlap_episode_count,
        safety_domain_intrusion_episode_count=cs.safety_domain_intrusion_episode_count - bs.safety_domain_intrusion_episode_count,
        near_miss_episode_count=cs.near_miss_episode_count - bs.near_miss_episode_count,
        unresolved_episode_count=cs.unresolved_episode_count - bs.unresolved_episode_count,
        nonclear_pair_exposure_min=cs.nonclear_pair_exposure_min - bs.nonclear_pair_exposure_min,
    )


def run_environmental_network_paired_experiment(
    config: EnvironmentalNetworkPairedExperimentConfig,
) -> EnvironmentalNetworkPairedExperimentResult:
    """Run one exact-CRN nominal-baseline/A137-control comparison."""

    if not isinstance(config, EnvironmentalNetworkPairedExperimentConfig):
        raise TypeError("config must be EnvironmentalNetworkPairedExperimentConfig")
    control_config = config.control
    scenario = control_config.scenario

    # Generate exogenous stochastic input exactly once.  Both arms below receive this
    # same immutable realization object; neither arm is allowed to draw traffic RNG.
    traffic = generate_poisson_network_traffic(
        control_config.network,
        control_config.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    ingress = _environmental_ingress_records(traffic, scenario)

    baseline = _run_nominal_baseline(traffic, ingress, control_config, config.safety)
    control, control_intervals = _run_control_from_shared_traffic(control_config, traffic, config.safety)
    completions = _control_completions(control)
    control_metrics = _arm_metrics(
        traffic=traffic,
        start_time=scenario.start_time,
        end_time=float(scenario.end_time),
        final_snapshot=control.scenario.run.snapshot,
        completions=completions,
        intervals=control_intervals,
    )
    return EnvironmentalNetworkPairedExperimentResult(
        config=config,
        pair_id=config.pair_id,
        traffic=traffic,
        baseline=baseline,
        control=control,
        control_completions=completions,
        control_safety_intervals=control_intervals,
        control_metrics=control_metrics,
        difference=_difference(control_metrics, baseline.metrics),
    )
