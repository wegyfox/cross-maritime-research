"""Event-driven first-clearance isolation for an A148 FIFO source head.

A149 remains a pure physical proof layer.  Given one due source demand and a *frozen
motion segment* for the currently active vessels (no membership, command, ownership or
other runtime mutation inside the supplied horizon), it searches left-to-right for the
first source-entry instant that can be certified clear.

This is not fixed-step polling.  State queries use the exact environmental dynamics and
adaptive interval subdivision.  A conservative Lipschitz upper bound is used to prove
that an earlier interval cannot contain a clear instant.  If that proof cannot be made
within the configured time/subdivision tolerance, the result is fail-closed
``UNCERTIFIED_WITHIN_SEGMENT``; a later clear sample is never used to skip unresolved
earlier time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot, isfinite, pi, sqrt
from typing import Iterable

from .dynamics import ManeuverLimits
from .environmental_dynamics import (
    WaterRelativeManeuverCommand,
    WaterRelativeVesselState,
    advance_with_uniform_drift,
)
from .environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceDemand,
    EnvironmentalNetworkSourceFutureIngressConfig,
)
from .environmental_candidate_transition import (
    EnvironmentalMotionBackend,
    EnvironmentalMotionState,
    StatefulEnvironmentalCommandTrajectory,
    build_environmental_candidate_transition,
    validate_environmental_motion_state,
)
from .environmental_pair_clearance import EnvironmentalPairTransitionUnavailable
from .environmental_pair_command_clearance import evaluate_environmental_pair_command_clearance
from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
from .same_route_following import (
    SameRouteFollowingSupervisor,
    SameRouteFollowingUnavailable,
    compute_same_route_following_constraint,
    same_route_longitudinal_pair_applicable,
    same_route_safety_projection,
)
from .ship_domain import ShipDomain, evaluate_pair_ship_domain
from .vessel_classes import VesselClassProfile

_EPS = 1e-12


class EnvironmentalNetworkSourceRecheckError(RuntimeError):
    """Raised when a frozen source-recheck segment is internally inconsistent."""


class EnvironmentalNetworkSourceRecheckStatus(str, Enum):
    ALREADY_CLEAR = "ALREADY_CLEAR"
    CERTIFIED_CLEARANCE_WINDOW = "CERTIFIED_CLEARANCE_WINDOW"
    NO_CLEARANCE_WITHIN_SEGMENT = "NO_CLEARANCE_WITHIN_SEGMENT"
    UNCERTIFIED_WITHIN_SEGMENT = "UNCERTIFIED_WITHIN_SEGMENT"
    BLOCKED_MISSING_DOMAIN = "BLOCKED_MISSING_DOMAIN"


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceRecheckConfig:
    time_tolerance_min: float = 0.01
    max_subdivisions: int = 24

    def __post_init__(self) -> None:
        if not isfinite(self.time_tolerance_min) or self.time_tolerance_min <= 0.0:
            raise ValueError("time_tolerance_min must be finite and positive")
        if self.max_subdivisions < 1:
            raise ValueError("max_subdivisions must be at least 1")


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceActiveMotion:
    """One active vessel's exact state/control evidence at the segment start."""

    state: WaterRelativeVesselState
    command: WaterRelativeManeuverCommand
    limits: ManeuverLimits
    environmental_drift_velocity: tuple[float, float]
    profile: VesselClassProfile
    route_target: EnvironmentalRouteHandoffTarget | None = None
    motion_backend: EnvironmentalMotionBackend | None = None
    motion_state: EnvironmentalMotionState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, WaterRelativeVesselState):
            raise TypeError("state must be WaterRelativeVesselState")
        if not isinstance(self.command, WaterRelativeManeuverCommand):
            raise TypeError("command must be WaterRelativeManeuverCommand")
        if not isinstance(self.limits, ManeuverLimits):
            raise TypeError("limits must be ManeuverLimits")
        if not isinstance(self.profile, VesselClassProfile):
            raise TypeError("profile must be VesselClassProfile")
        if self.route_target is not None and not isinstance(
            self.route_target, EnvironmentalRouteHandoffTarget
        ):
            raise TypeError("route_target must be EnvironmentalRouteHandoffTarget or None")
        if self.motion_backend is not None and not isinstance(self.motion_backend, EnvironmentalMotionBackend):
            raise TypeError("motion_backend must implement EnvironmentalMotionBackend")
        if self.motion_backend is None:
            if self.motion_state is not None:
                raise ValueError("motion_state requires motion_backend")
        else:
            validate_environmental_motion_state(
                self.motion_backend,
                self.motion_state,
                boundary=f"source recheck active motion {self.state.vessel_id!r}",
            )
        if len(self.environmental_drift_velocity) != 2 or any(
            not isfinite(value) for value in self.environmental_drift_velocity
        ):
            raise ValueError("environmental_drift_velocity must contain two finite components")

    @property
    def vessel_id(self) -> str:
        return self.state.vessel_id

    @property
    def ground_speed_bound_nm_per_min(self) -> float:
        return max(
            self.state.through_water_speed,
            self.command.target_through_water_speed,
        ) + hypot(*self.environmental_drift_velocity)


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkSourceRecheckResult:
    status: EnvironmentalNetworkSourceRecheckStatus
    segment_start_time: float
    segment_end_time: float
    earliest_possible_clear_time: float | None
    certified_clear_time: float | None
    clearance_window_width_min: float | None
    state_evaluations: int
    missing_domain_vessel_ids: tuple[str, ...] = ()

    @property
    def has_certified_clear_time(self) -> bool:
        return self.certified_clear_time is not None


def _max_extent(domain: ShipDomain) -> float:
    return max(domain.ahead_nm, domain.astern_nm, domain.port_nm, domain.starboard_nm)


def _min_extent(domain: ShipDomain) -> float:
    return min(domain.ahead_nm, domain.astern_nm, domain.port_nm, domain.starboard_nm)


def _angular_sensitivity_per_rad(domain: ShipDomain) -> float:
    """Same global radial-function derivative bound used by domain_clearance.py."""
    best = 0.0
    for longitudinal in (domain.ahead_nm, domain.astern_nm):
        for transverse in (domain.port_nm, domain.starboard_nm):
            radius = max(longitudinal, transverse)
            sensitivity = 0.5 * radius**3 * abs(
                1.0 / longitudinal**2 - 1.0 / transverse**2
            )
            best = max(best, sensitivity)
    return best


@dataclass(frozen=True, slots=True)
class _PairSample:
    margin_nm: float
    distance_nm: float


@dataclass(frozen=True, slots=True)
class _Sample:
    pair_samples: tuple[_PairSample, ...]
    dynamic_launch_feasible: bool = True
    future_launch_feasible: bool = True

    @property
    def clear(self) -> bool:
        return (
            all(item.margin_nm > _EPS for item in self.pair_samples)
            and self.dynamic_launch_feasible
            and self.future_launch_feasible
        )


class _SearchKind(str, Enum):
    NO_CLEAR = "NO_CLEAR"
    FOUND = "FOUND"
    UNCERTIFIED = "UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class _SearchResult:
    kind: _SearchKind
    lower: float | None = None
    certified: float | None = None




def _prepare_source_recheck_inputs(
    demand: EnvironmentalNetworkSourceDemand,
    active_motions: Iterable[EnvironmentalNetworkSourceActiveMotion],
    *,
    segment_start_time: float,
    segment_duration_min: float,
    config: EnvironmentalNetworkSourceRecheckConfig,
    same_route_dynamic_ingress: SameRouteFollowingSupervisor | None,
    future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None,
) -> tuple[
    tuple[EnvironmentalNetworkSourceActiveMotion, ...],
    VesselClassProfile | None,
    EnvironmentalNetworkSourceRecheckResult | None,
]:
    """Stage 1: normalize one frozen segment and preserve historical fail-fast order."""

    if not isinstance(demand, EnvironmentalNetworkSourceDemand):
        raise TypeError("demand must be EnvironmentalNetworkSourceDemand")
    if not isinstance(config, EnvironmentalNetworkSourceRecheckConfig):
        raise TypeError("config must be EnvironmentalNetworkSourceRecheckConfig")
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
    if not isfinite(segment_start_time):
        raise ValueError("segment_start_time must be finite")
    if not isfinite(segment_duration_min) or segment_duration_min < 0.0:
        raise ValueError("segment_duration_min must be finite and non-negative")
    if segment_start_time < demand.demand_time - _EPS:
        raise EnvironmentalNetworkSourceRecheckError(
            "source recheck segment cannot begin before stochastic demand"
        )

    motions = tuple(active_motions)
    ids = [item.vessel_id for item in motions]
    if len(set(ids)) != len(ids):
        raise EnvironmentalNetworkSourceRecheckError("duplicate active motion vessel id")
    if demand.vessel_id in set(ids):
        raise EnvironmentalNetworkSourceRecheckError(
            "pending source demand cannot already be an active motion"
        )
    if future_ingress_clearance is not None and demand.limits is None:
        raise EnvironmentalNetworkSourceRecheckError(
            "future ingress clearance recheck requires incoming maneuver limits"
        )
    if same_route_dynamic_ingress is not None:
        incoming_following_config = same_route_dynamic_ingress.config_for_follower(demand.vessel_id)
        missing_route_targets = tuple(
            sorted(item.vessel_id for item in motions if item.route_target is None)
        )
        if missing_route_targets:
            raise EnvironmentalNetworkSourceRecheckError(
                "dynamic ingress recheck requires route metadata for every active motion: "
                f"{missing_route_targets!r}"
            )
        if demand.limits is None:
            raise EnvironmentalNetworkSourceRecheckError(
                "dynamic ingress recheck requires incoming maneuver limits"
            )
        if incoming_following_config.require_zero_environmental_drift:
            if abs(demand.environmental_drift_velocity[0]) > _EPS or abs(demand.environmental_drift_velocity[1]) > _EPS:
                raise SameRouteFollowingUnavailable(
                    "dynamic ingress v1 is restricted to zero environmental drift"
                )
            for item in motions:
                if abs(item.environmental_drift_velocity[0]) > _EPS or abs(item.environmental_drift_velocity[1]) > _EPS:
                    raise SameRouteFollowingUnavailable(
                        "dynamic ingress v1 is restricted to zero environmental drift"
                    )

    incoming_profile = demand.profile
    missing = []
    if incoming_profile is None or incoming_profile.safety_domain is None:
        missing.append(demand.vessel_id)
    missing.extend(
        item.vessel_id for item in motions if item.profile.safety_domain is None
    )
    if missing:
        return motions, incoming_profile, EnvironmentalNetworkSourceRecheckResult(
            status=EnvironmentalNetworkSourceRecheckStatus.BLOCKED_MISSING_DOMAIN,
            segment_start_time=segment_start_time,
            segment_end_time=segment_start_time + segment_duration_min,
            earliest_possible_clear_time=None,
            certified_clear_time=None,
            clearance_window_width_min=None,
            state_evaluations=0,
            missing_domain_vessel_ids=tuple(sorted(set(missing))),
        )

    return motions, incoming_profile, None


class _SourceRecheckSearchContext:
    """Stages 2-4: cached physical sampling, launch guards and adaptive proof search."""

    __slots__ = (
        "demand",
        "motions",
        "segment_duration_min",
        "config",
        "same_route_dynamic_ingress",
        "future_ingress_clearance",
        "incoming_profile",
        "incoming_domain",
        "incoming_state",
        "caches",
        "state_caches",
        "trajectory_caches",
        "sample_cache",
    )

    def __init__(
        self,
        demand: EnvironmentalNetworkSourceDemand,
        motions: tuple[EnvironmentalNetworkSourceActiveMotion, ...],
        *,
        segment_duration_min: float,
        config: EnvironmentalNetworkSourceRecheckConfig,
        same_route_dynamic_ingress: SameRouteFollowingSupervisor | None,
        future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None,
        incoming_profile: VesselClassProfile,
    ) -> None:
        assert incoming_profile.safety_domain is not None
        self.demand = demand
        self.motions = motions
        self.segment_duration_min = segment_duration_min
        self.config = config
        self.same_route_dynamic_ingress = same_route_dynamic_ingress
        self.future_ingress_clearance = future_ingress_clearance
        self.incoming_profile = incoming_profile
        self.incoming_domain = incoming_profile.safety_domain
        self.incoming_state = demand.vessel
        self.caches: list[dict[float, _PairSample]] = [dict() for _ in motions]
        self.state_caches: list[dict[float, WaterRelativeVesselState]] = [dict() for _ in motions]
        self.trajectory_caches: list[object | None] = [None for _ in motions]
        self.sample_cache: dict[float, _Sample] = {}

    @property
    def evaluations(self) -> int:
        return sum(len(cache) for cache in self.caches)

    def active_trajectory(self, index: int):
        cached = self.trajectory_caches[index]
        if cached is not None:
            return cached
        motion = self.motions[index]
        trajectory = build_environmental_candidate_transition(
            motion.state,
            motion.environmental_drift_velocity,
            motion.command,
            motion.limits,
            motion_backend=motion.motion_backend,
            motion_state=motion.motion_state,
            evaluation_horizon_min=max(self.segment_duration_min, self.config.time_tolerance_min),
        )
        self.trajectory_caches[index] = trajectory
        return trajectory

    def active_state(self, index: int, elapsed: float) -> WaterRelativeVesselState:
        key = round(elapsed, 14)
        cache = self.state_caches[index]
        if key not in cache:
            cache[key] = self.active_trajectory(index).state_at_or_continue(elapsed)
        return cache[key]

    def pair_sample(self, index: int, elapsed: float) -> _PairSample:
        key = round(elapsed, 14)
        cache = self.caches[index]
        if key in cache:
            return cache[key]
        motion = self.motions[index]
        assert motion.profile.safety_domain is not None
        state = self.active_state(index, elapsed)
        relative = (
            self.incoming_state.position[0] - state.position[0],
            self.incoming_state.position[1] - state.position[1],
        )
        distance = hypot(*relative)
        if distance <= _EPS:
            margin = -(
                _max_extent(motion.profile.safety_domain) + _max_extent(self.incoming_domain)
            )
        else:
            clearance = evaluate_pair_ship_domain(
                first_heading_deg=state.through_water_heading_deg,
                second_heading_deg=self.incoming_state.through_water_heading_deg,
                relative_position=relative,
                first_domain=motion.profile.safety_domain,
                second_domain=self.incoming_domain,
            )
            margin = clearance.actual_distance_nm - clearance.required_distance_nm
        value = _PairSample(margin, distance)
        cache[key] = value
        return value

    def dynamic_launch_feasible(self, elapsed: float) -> bool:
        supervisor = self.same_route_dynamic_ingress
        if supervisor is None:
            return True
        assert self.demand.limits is not None
        incoming_route = self.demand.route_target.navigator.follower.route
        incoming_projection = same_route_safety_projection(incoming_route, self.incoming_state.position)
        constraints = []
        for index, motion in enumerate(self.motions):
            assert motion.route_target is not None
            if motion.route_target.navigator.follower.route != incoming_route:
                continue
            pair_following_config = supervisor.config_for_pair(
                self.demand.vessel_id, motion.vessel_id
            )
            state = self.active_state(index, elapsed)
            projection = same_route_safety_projection(incoming_route, state.position)
            if projection.along_track_nm <= incoming_projection.along_track_nm + _EPS:
                continue
            if not same_route_longitudinal_pair_applicable(
                follower_state=self.incoming_state,
                leader_state=state,
                follower_domain=self.incoming_profile.safety_domain,
                leader_domain=motion.profile.safety_domain,
                follower_cross_track_nm=float(incoming_projection.cross_track_nm),
                leader_cross_track_nm=float(projection.cross_track_nm),
                follower_route_heading_deg=float(incoming_projection.route_heading_deg),
                leader_route_heading_deg=float(projection.route_heading_deg),
                config=pair_following_config,
            ):
                continue
            constraints.append(
                compute_same_route_following_constraint(
                    follower_id=self.demand.vessel_id,
                    leader_id=motion.vessel_id,
                    follower_state=self.incoming_state,
                    leader_state=state,
                    follower_along_track_nm=float(incoming_projection.along_track_nm),
                    leader_along_track_nm=float(projection.along_track_nm),
                    follower_cross_track_nm=float(incoming_projection.cross_track_nm),
                    leader_cross_track_nm=float(projection.cross_track_nm),
                    follower_profile=self.incoming_profile,
                    leader_profile=motion.profile,
                    follower_limits=self.demand.limits,
                    leader_limits=motion.limits,
                    route_heading_deg=float(incoming_projection.route_heading_deg),
                    config=pair_following_config,
                )
            )
        if not constraints:
            return True
        limiting = min(
            constraints,
            key=lambda item: (
                item.max_follower_speed_nm_per_min,
                item.projected_longitudinal_gap_nm,
                item.leader_id,
            ),
        )
        return self.incoming_state.through_water_speed <= limiting.max_follower_speed_nm_per_min + _EPS

    def future_launch_feasible(self, elapsed: float) -> bool:
        future_config = self.future_ingress_clearance
        if future_config is None:
            return True
        assert self.demand.limits is not None
        launch_command = WaterRelativeManeuverCommand(
            target_through_water_heading_deg=self.incoming_state.through_water_heading_deg,
            target_through_water_speed=self.incoming_state.through_water_speed,
        )
        backends = {id(item.motion_backend): item.motion_backend for item in self.motions if item.motion_backend is not None}
        if len(backends) > 1:
            raise EnvironmentalNetworkSourceRecheckError("future ingress requires one shared environmental motion backend")
        backend = next(iter(backends.values()), None)
        incoming_motion_state = None
        if backend is not None:
            incoming_motion_state = validate_environmental_motion_state(
                backend,
                backend.initialize_motion_state(
                    self.incoming_state, self.demand.limits, self.demand.environmental_drift_velocity
                ),
                boundary=f"source recheck incoming initialize {self.demand.vessel_id!r}",
            )
        incoming_transition = build_environmental_candidate_transition(
            self.incoming_state,
            self.demand.environmental_drift_velocity,
            launch_command,
            self.demand.limits,
            motion_backend=backend,
            motion_state=incoming_motion_state,
            evaluation_horizon_min=future_config.pair_clearance.evaluation_horizon_min,
        )
        for index, motion in enumerate(self.motions):
            domain = motion.profile.safety_domain
            assert domain is not None
            trajectory = self.active_trajectory(index)
            if motion.motion_backend is not None and isinstance(
                trajectory, StatefulEnvironmentalCommandTrajectory
            ):
                # Rich cached plant trajectories must supply public state and opaque
                # plant sidecar from the SAME continuation path. Recomputing the sidecar
                # via an independent one-shot backend.advance() can follow a different
                # numerical partition and make the two states internally inconsistent.
                active_start, active_motion_state = trajectory.state_and_motion_at_or_continue(elapsed)
                active_motion_state = validate_environmental_motion_state(
                    motion.motion_backend,
                    active_motion_state,
                    boundary=f"source recheck trajectory continuation {motion.vessel_id!r}",
                )
                self.state_caches[index][round(elapsed, 14)] = active_start
            else:
                active_start = self.active_state(index, elapsed)
                if motion.motion_backend is None:
                    active_motion_state = None
                else:
                    # Compatibility fallback for richer third-party backends whose
                    # trajectories do not yet expose a paired continuation accessor.
                    advanced = motion.motion_backend.advance(
                        motion.state, motion.command, elapsed, motion.limits,
                        motion.environmental_drift_velocity, motion.motion_state
                    )
                    active_motion_state = validate_environmental_motion_state(
                        motion.motion_backend,
                        advanced.motion_state,
                        boundary=f"source recheck fallback continuation {motion.vessel_id!r}",
                    )
            active_transition = build_environmental_candidate_transition(
                active_start,
                motion.environmental_drift_velocity,
                motion.command,
                motion.limits,
                motion_backend=motion.motion_backend,
                motion_state=active_motion_state,
                evaluation_horizon_min=future_config.pair_clearance.evaluation_horizon_min,
            )
            try:
                evidence = evaluate_environmental_pair_command_clearance(
                    active_transition,
                    incoming_transition,
                    future_config.pair_clearance,
                    first_domain=domain,
                    second_domain=self.incoming_profile.safety_domain,
                )
            except EnvironmentalPairTransitionUnavailable:
                return False
            if not evidence.certified_clear:
                return False
        return True

    def sample(self, elapsed: float) -> _Sample:
        # Exact-float memoization only: recursive siblings share identical binary
        # endpoints/midpoints. Do not round here; nearby-but-distinct times retain
        # their historical independent future-ingress verdicts.
        key = float(elapsed)
        cached = self.sample_cache.get(key)
        if cached is not None:
            return cached
        pair_samples = tuple(self.pair_sample(i, elapsed) for i in range(len(self.motions)))
        # Admission is a conjunction. Preserve the exact verdict while avoiding the
        # expensive nested future-ingress certificate when an earlier necessary
        # condition already fails.
        if not all(item.margin_nm > _EPS for item in pair_samples):
            result = _Sample(pair_samples)
            self.sample_cache[key] = result
            return result
        dynamic_clear = self.dynamic_launch_feasible(elapsed)
        if not dynamic_clear:
            result = _Sample(pair_samples, dynamic_launch_feasible=False)
            self.sample_cache[key] = result
            return result
        result = _Sample(
            pair_samples,
            dynamic_launch_feasible=True,
            future_launch_feasible=self.future_launch_feasible(elapsed),
        )
        self.sample_cache[key] = result
        return result

    def pair_interval_upper(self, index: int, a: float, mid: float, b: float) -> float | None:
        """Conservative upper bound on one pair's margin throughout [a,b]."""
        motion = self.motions[index]
        domain = motion.profile.safety_domain
        assert domain is not None
        sa = self.pair_sample(index, a)
        sm = self.pair_sample(index, mid)
        sb = self.pair_sample(index, b)
        width = b - a
        half = 0.5 * width
        speed_bound = self.active_trajectory(index).ground_speed_bound

        distance_upper = min(
            sa.distance_nm + speed_bound * width,
            sb.distance_nm + speed_bound * width,
            sm.distance_nm + speed_bound * half,
        )
        minimum_required = _min_extent(domain) + _min_extent(self.incoming_domain)
        if distance_upper <= minimum_required + _EPS:
            return distance_upper - minimum_required

        if speed_bound <= _EPS:
            margin_rate = (
                _angular_sensitivity_per_rad(domain)
                * motion.limits.max_turn_rate_deg_per_min
                * pi
                / 180.0
            )
        else:
            distance_floor = max(0.0, sm.distance_nm - speed_bound * half)
            if distance_floor <= _EPS:
                return None
            los_rate = speed_bound / distance_floor
            margin_rate = speed_bound + (
                _angular_sensitivity_per_rad(domain)
                * (los_rate + motion.limits.max_turn_rate_deg_per_min * pi / 180.0)
                + _angular_sensitivity_per_rad(self.incoming_domain) * los_rate
            )
        return min(
            sa.margin_nm + margin_rate * width,
            sb.margin_nm + margin_rate * width,
            sm.margin_nm + margin_rate * half,
        )

    def interval_proven_blocked(self, a: float, b: float) -> bool:
        mid = 0.5 * (a + b)
        for index in range(len(self.motions)):
            upper = self.pair_interval_upper(index, a, mid, b)
            if upper is not None and upper <= _EPS:
                return True
        return False

    def search(self, a: float, b: float, depth: int) -> _SearchResult:
        if self.interval_proven_blocked(a, b):
            return _SearchResult(_SearchKind.NO_CLEAR)

        mid = 0.5 * (a + b)
        width = b - a
        if width <= self.config.time_tolerance_min or depth >= self.config.max_subdivisions:
            for certified in (a, mid, b):
                if self.sample(certified).clear:
                    return _SearchResult(_SearchKind.FOUND, lower=a, certified=certified)
            return _SearchResult(_SearchKind.UNCERTIFIED)

        left = self.search(a, mid, depth + 1)
        if left.kind is _SearchKind.FOUND:
            return left
        if left.kind is _SearchKind.UNCERTIFIED:
            return left
        return self.search(mid, b, depth + 1)


def _source_recheck_result_from_search(
    found: _SearchResult,
    *,
    segment_start_time: float,
    segment_duration_min: float,
    evaluations: int,
) -> EnvironmentalNetworkSourceRecheckResult:
    """Stage 5: map one proof-search outcome to the public immutable result."""

    segment_end = segment_start_time + segment_duration_min
    if found.kind is _SearchKind.NO_CLEAR:
        return EnvironmentalNetworkSourceRecheckResult(
            status=EnvironmentalNetworkSourceRecheckStatus.NO_CLEARANCE_WITHIN_SEGMENT,
            segment_start_time=segment_start_time,
            segment_end_time=segment_end,
            earliest_possible_clear_time=None,
            certified_clear_time=None,
            clearance_window_width_min=None,
            state_evaluations=evaluations,
        )
    if found.kind is _SearchKind.UNCERTIFIED:
        return EnvironmentalNetworkSourceRecheckResult(
            status=EnvironmentalNetworkSourceRecheckStatus.UNCERTIFIED_WITHIN_SEGMENT,
            segment_start_time=segment_start_time,
            segment_end_time=segment_end,
            earliest_possible_clear_time=None,
            certified_clear_time=None,
            clearance_window_width_min=None,
            state_evaluations=evaluations,
        )

    assert found.lower is not None and found.certified is not None
    lower_abs = segment_start_time + found.lower
    certified_abs = segment_start_time + found.certified
    return EnvironmentalNetworkSourceRecheckResult(
        status=EnvironmentalNetworkSourceRecheckStatus.CERTIFIED_CLEARANCE_WINDOW,
        segment_start_time=segment_start_time,
        segment_end_time=segment_end,
        earliest_possible_clear_time=lower_abs,
        certified_clear_time=certified_abs,
        clearance_window_width_min=certified_abs - lower_abs,
        state_evaluations=evaluations,
    )


def find_environmental_network_source_recheck(
    demand: EnvironmentalNetworkSourceDemand,
    active_motions: Iterable[EnvironmentalNetworkSourceActiveMotion],
    *,
    segment_start_time: float,
    segment_duration_min: float,
    config: EnvironmentalNetworkSourceRecheckConfig = EnvironmentalNetworkSourceRecheckConfig(),
    same_route_dynamic_ingress: SameRouteFollowingSupervisor | None = None,
    future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None = None,
) -> EnvironmentalNetworkSourceRecheckResult:
    """Isolate the first certifiably clear source-entry instant in one frozen segment.

    The caller owns the event clock. ``segment_duration_min`` must end no later than the
    next command/membership/ownership/source-admission mutation that would invalidate any
    supplied active motion. A future runtime adapter must therefore split on such a
    boundary before reusing this result.
    """

    motions, incoming_profile, immediate = _prepare_source_recheck_inputs(
        demand,
        active_motions,
        segment_start_time=segment_start_time,
        segment_duration_min=segment_duration_min,
        config=config,
        same_route_dynamic_ingress=same_route_dynamic_ingress,
        future_ingress_clearance=future_ingress_clearance,
    )
    if immediate is not None:
        return immediate
    assert incoming_profile is not None and incoming_profile.safety_domain is not None

    if not motions:
        return EnvironmentalNetworkSourceRecheckResult(
            status=EnvironmentalNetworkSourceRecheckStatus.ALREADY_CLEAR,
            segment_start_time=segment_start_time,
            segment_end_time=segment_start_time + segment_duration_min,
            earliest_possible_clear_time=segment_start_time,
            certified_clear_time=segment_start_time,
            clearance_window_width_min=0.0,
            state_evaluations=0,
        )

    context = _SourceRecheckSearchContext(
        demand,
        motions,
        segment_duration_min=segment_duration_min,
        config=config,
        same_route_dynamic_ingress=same_route_dynamic_ingress,
        future_ingress_clearance=future_ingress_clearance,
        incoming_profile=incoming_profile,
    )

    start_sample = context.sample(0.0)
    if start_sample.clear:
        return EnvironmentalNetworkSourceRecheckResult(
            status=EnvironmentalNetworkSourceRecheckStatus.ALREADY_CLEAR,
            segment_start_time=segment_start_time,
            segment_end_time=segment_start_time + segment_duration_min,
            earliest_possible_clear_time=segment_start_time,
            certified_clear_time=segment_start_time,
            clearance_window_width_min=0.0,
            state_evaluations=context.evaluations,
        )

    if segment_duration_min <= _EPS:
        return EnvironmentalNetworkSourceRecheckResult(
            status=EnvironmentalNetworkSourceRecheckStatus.NO_CLEARANCE_WITHIN_SEGMENT,
            segment_start_time=segment_start_time,
            segment_end_time=segment_start_time,
            earliest_possible_clear_time=None,
            certified_clear_time=None,
            clearance_window_width_min=None,
            state_evaluations=context.evaluations,
        )

    found = context.search(0.0, segment_duration_min, 0)
    return _source_recheck_result_from_search(
        found,
        segment_start_time=segment_start_time,
        segment_duration_min=segment_duration_min,
        evaluations=context.evaluations,
    )
