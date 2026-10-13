"""Optional same-route longitudinal following constraints for research traffic flows.

The layer is deliberately independent of any macroscopic/headway reference model.  It
uses route geometry, current physical state, ship domains and braking envelopes to
compute a conservative upper bound on a follower's commanded through-water speed.
Cross-stream encounter safety remains the responsibility of the ordinary PAIR/JOINT
physical evaluators; same-route pairs remain available to independent safety telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, isfinite, radians, sin, sqrt
from typing import Mapping, TYPE_CHECKING
from types import MappingProxyType

from .encounter_registry import PairKey, canonical_pair_key
from .ship_domain import ShipDomain, evaluate_pair_ship_domain
from .dynamics import ManeuverLimits, signed_heading_delta
from .environmental_dynamics import WaterRelativeVesselState
from .vessel_classes import VesselClassProfile

if TYPE_CHECKING:
    from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
    from .environmental_world import EnvironmentalWorld2D
    from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class SameRouteFollowingUnavailable(RuntimeError):
    """The v1 following abstraction cannot safely evaluate the supplied state."""




@dataclass(frozen=True, slots=True)
class SameRouteSafetyProjection:
    """Route-frame evidence for longitudinal safety, including past-finish progress.

    Navigation intentionally uses :meth:`Route2D.project`, whose arc-length is clamped
    to the finite mission route.  Same-route following needs a different final-leg
    contract: an active vessel that has physically crossed the finish line must remain
    ahead by its true forward displacement instead of collapsing back onto the endpoint.
    """

    along_track_nm: float
    cross_track_nm: float
    route_heading_deg: float


def same_route_safety_projection(route, point) -> SameRouteSafetyProjection:
    """Project ``point`` for following safety without final-endpoint longitudinal clamp.

    The ordinary bounded route projection is preserved everywhere except beyond the
    finish line on the final segment.  There, the final segment is extended as a ray for
    safety geometry only.  This preserves the true physical leader/follower gap for
    active vessels past route end while leaving navigation completion semantics unchanged.
    """

    bounded = route.project(point)
    final_index = route.segment_count - 1
    if bounded.segment_index != final_index or bounded.segment_fraction < 1.0 - _EPS:
        return SameRouteSafetyProjection(
            float(bounded.along_track),
            float(bounded.cross_track),
            float(bounded.heading_deg),
        )

    a = route.waypoints[-2]
    b = route.waypoints[-1]
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = route.segment_lengths[-1]
    rel_x = float(point[0]) - a[0]
    rel_y = float(point[1]) - a[1]
    raw_t = (rel_x * dx + rel_y * dy) / (length * length)
    if raw_t <= 1.0 + _EPS:
        return SameRouteSafetyProjection(
            float(bounded.along_track),
            float(bounded.cross_track),
            float(bounded.heading_deg),
        )

    along = route.cumulative_lengths[-2] + raw_t * length
    projected_x = a[0] + raw_t * dx
    projected_y = a[1] + raw_t * dy
    offset_x = float(point[0]) - projected_x
    offset_y = float(point[1]) - projected_y
    cross_track = (dx * offset_y - dy * offset_x) / length
    return SameRouteSafetyProjection(
        float(along),
        float(cross_track),
        float(route.segment_heading(final_index)),
    )

@dataclass(frozen=True, slots=True)
class SameRouteFollowingConfig:
    reaction_interval_min: float = 0.25
    near_route_cross_track_tolerance_nm: float = 0.03
    heading_tolerance_deg: float = 10.0
    require_zero_environmental_drift: bool = True

    def __post_init__(self) -> None:
        for name in ("reaction_interval_min", "near_route_cross_track_tolerance_nm", "heading_tolerance_deg"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.require_zero_environmental_drift, bool):
            raise TypeError("require_zero_environmental_drift must be bool")


@dataclass(frozen=True, slots=True)
class SameRouteFollowingConstraint:
    pair_key: PairKey
    follower_id: str
    leader_id: str
    follower_along_track_nm: float
    leader_along_track_nm: float
    follower_cross_track_nm: float
    leader_cross_track_nm: float
    lateral_separation_nm: float
    required_planar_domain_separation_nm: float
    required_longitudinal_domain_gap_nm: float
    projected_longitudinal_gap_nm: float
    available_braking_gap_nm: float
    leader_stop_credit_nm: float
    max_follower_speed_nm_per_min: float

    def __post_init__(self) -> None:
        if self.pair_key != canonical_pair_key(self.follower_id, self.leader_id):
            raise ValueError("pair_key does not match follower/leader ids")
        if self.leader_along_track_nm + _EPS < self.follower_along_track_nm:
            raise ValueError("leader must not project behind follower")
        for name in (
            "lateral_separation_nm",
            "required_planar_domain_separation_nm",
            "required_longitudinal_domain_gap_nm",
            "projected_longitudinal_gap_nm",
            "leader_stop_credit_nm",
            "max_follower_speed_nm_per_min",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < -_EPS:
                raise ValueError(f"{name} must be finite and non-negative")



def _circular_radius_or_none(domain: ShipDomain | None) -> float | None:
    if domain is None:
        raise SameRouteFollowingUnavailable("same-route following requires ship-domain evidence")
    values = (domain.ahead_nm, domain.astern_nm, domain.port_nm, domain.starboard_nm)
    if max(values) - min(values) > 1e-12:
        return None
    return sum(values) / 4.0


def _route_relative_vector(
    *,
    route_heading_deg: float,
    longitudinal_nm: float,
    signed_lateral_nm: float,
) -> tuple[float, float]:
    """Reconstruct a route-frame relative vector in global east/north coordinates.

    ``signed_lateral_nm`` follows ``RouteProjection.cross_track``: positive is port.
    """

    theta = radians(route_heading_deg)
    forward = (sin(theta), cos(theta))
    port = (-cos(theta), sin(theta))
    return (
        longitudinal_nm * forward[0] + signed_lateral_nm * port[0],
        longitudinal_nm * forward[1] + signed_lateral_nm * port[1],
    )


def _directional_required_longitudinal_gap(
    *,
    follower_state: WaterRelativeVesselState,
    leader_state: WaterRelativeVesselState,
    follower_domain: ShipDomain,
    leader_domain: ShipDomain,
    route_heading_deg: float,
    signed_lateral_nm: float,
) -> tuple[float, float]:
    """Return (current-ray required distance, minimum route-longitudinal gap).

    This extends the original circular-only following kernel to the simulator's
    four-quadrant elliptical ship domains without inventing a scalar CPA.  For a fixed
    signed cross-track separation and current vessel headings, it solves the same radial
    pair-domain model used elsewhere for the minimum along-route separation that lies
    outside the two domains.  Circular domains stay on the old closed-form path, so
    frozen legacy behaviour remains byte/behaviour compatible.
    """

    def margin(longitudinal_nm: float) -> tuple[float, float]:
        relative = _route_relative_vector(
            route_heading_deg=route_heading_deg,
            longitudinal_nm=longitudinal_nm,
            signed_lateral_nm=signed_lateral_nm,
        )
        clearance = evaluate_pair_ship_domain(
            first_heading_deg=follower_state.through_water_heading_deg,
            second_heading_deg=leader_state.through_water_heading_deg,
            relative_position=relative,
            first_domain=follower_domain,
            second_domain=leader_domain,
        )
        return (
            clearance.actual_distance_nm - clearance.required_distance_nm,
            clearance.required_distance_nm,
        )

    zero_margin, zero_required = margin(0.0)
    if zero_margin > 1e-12:
        return zero_required, 0.0

    max_extent = max(
        follower_domain.ahead_nm, follower_domain.astern_nm,
        follower_domain.port_nm, follower_domain.starboard_nm,
        leader_domain.ahead_nm, leader_domain.astern_nm,
        leader_domain.port_nm, leader_domain.starboard_nm,
    )
    high = max(2.0 * max_extent + abs(signed_lateral_nm), 1e-6)
    high_margin, high_required = margin(high)
    for _ in range(40):
        if high_margin > 1e-12:
            break
        high *= 2.0
        high_margin, high_required = margin(high)
    else:
        raise SameRouteFollowingUnavailable(
            "directional same-route domain gap could not be bounded"
        )

    low = 0.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        mid_margin, _ = margin(mid)
        if mid_margin > 0.0:
            high = mid
        else:
            low = mid
    return high_required, high


def _braking_distance(speed: float, deceleration: float) -> float:
    if speed <= _EPS:
        return 0.0
    if deceleration <= _EPS:
        return float("inf")
    return speed * speed / (2.0 * deceleration)


def _speed_ceiling(*, budget_nm: float, deceleration: float, reaction_min: float) -> float:
    if budget_nm <= _EPS:
        return 0.0
    if deceleration <= _EPS:
        return budget_nm / reaction_min if reaction_min > _EPS else 0.0
    # v*r + v^2/(2a) <= budget; return the non-negative quadratic root.
    return max(0.0, -deceleration * reaction_min + sqrt(
        (deceleration * reaction_min) ** 2 + 2.0 * deceleration * budget_nm
    ))


def same_route_longitudinal_state_applicable(
    *,
    state: WaterRelativeVesselState,
    cross_track_nm: float,
    route_heading_deg: float,
    config: SameRouteFollowingConfig,
) -> bool:
    """Whether one vessel is inside the validity envelope of longitudinal following.

    The same-route supervisor is a one-dimensional longitudinal abstraction.  Merely
    sharing a mission route is not enough to make that abstraction valid: the vessel
    must still be close to the route corridor and its hull heading must remain aligned
    with the local route tangent.  Outside this envelope the pair is left to the
    ordinary physical swept/domain evaluator instead of being delegated to following.

    These thresholds already existed in :class:`SameRouteFollowingConfig`; this helper
    makes them an applicability contract rather than using cross-track proximity only
    for optimistic leader braking credit.
    """

    if not isfinite(cross_track_nm) or not isfinite(route_heading_deg):
        raise SameRouteFollowingUnavailable(
            "longitudinal-following applicability requires finite route projection evidence"
        )
    if abs(float(cross_track_nm)) > config.near_route_cross_track_tolerance_nm + _EPS:
        return False
    return (
        abs(signed_heading_delta(state.through_water_heading_deg, route_heading_deg))
        <= config.heading_tolerance_deg + _EPS
    )


def same_route_longitudinal_pair_applicable(
    *,
    follower_state: WaterRelativeVesselState,
    leader_state: WaterRelativeVesselState,
    follower_domain: ShipDomain,
    leader_domain: ShipDomain,
    follower_cross_track_nm: float,
    leader_cross_track_nm: float,
    follower_route_heading_deg: float,
    leader_route_heading_deg: float,
    config: SameRouteFollowingConfig,
) -> bool:
    """Whether a pair shares one local longitudinal route frame.

    Legacy circular domains are isotropic and the original following kernel already
    handles lateral route-recovery separation explicitly. Preserve that frozen
    behaviour. Directional/non-circular domains are orientation-sensitive: for those,
    require each vessel to be route-aligned and require the two local route tangents to
    agree. This prevents an off-route turning pair from being collapsed into a
    one-dimensional leader/follower model while keeping circular controls compatible.
    """

    if (
        _circular_radius_or_none(follower_domain) is not None
        and _circular_radius_or_none(leader_domain) is not None
    ):
        return True

    if not same_route_longitudinal_state_applicable(
        state=follower_state,
        cross_track_nm=follower_cross_track_nm,
        route_heading_deg=follower_route_heading_deg,
        config=config,
    ):
        return False
    if not same_route_longitudinal_state_applicable(
        state=leader_state,
        cross_track_nm=leader_cross_track_nm,
        route_heading_deg=leader_route_heading_deg,
        config=config,
    ):
        return False
    return (
        abs(signed_heading_delta(follower_route_heading_deg, leader_route_heading_deg))
        <= config.heading_tolerance_deg + _EPS
    )




def compute_same_route_following_constraint(
    *,
    follower_id: str,
    leader_id: str,
    follower_state: WaterRelativeVesselState,
    leader_state: WaterRelativeVesselState,
    follower_along_track_nm: float,
    leader_along_track_nm: float,
    follower_cross_track_nm: float,
    leader_cross_track_nm: float,
    follower_profile: VesselClassProfile,
    leader_profile: VesselClassProfile,
    follower_limits: ManeuverLimits,
    leader_limits: ManeuverLimits,
    route_heading_deg: float | None = None,
    config: SameRouteFollowingConfig = SameRouteFollowingConfig(),
) -> SameRouteFollowingConstraint:
    """Pure physical following constraint shared by runtime and source ingress.

    No macroscopic headway is used.  The result depends only on current route
    projection, ship domains, bounded braking and the configured reaction interval.
    """

    if leader_along_track_nm <= follower_along_track_nm + _EPS:
        raise SameRouteFollowingUnavailable("leader must project strictly ahead of follower")
    follower_domain = follower_profile.safety_domain
    leader_domain = leader_profile.safety_domain
    follower_radius = _circular_radius_or_none(follower_domain)
    leader_radius = _circular_radius_or_none(leader_domain)
    signed_lateral = float(leader_cross_track_nm) - float(follower_cross_track_nm)
    lateral = abs(signed_lateral)
    longitudinal_gap = max(0.0, float(leader_along_track_nm - follower_along_track_nm))
    if follower_radius is not None and leader_radius is not None:
        required_planar = follower_radius + leader_radius
        required_longitudinal = sqrt(
            max(0.0, required_planar * required_planar - lateral * lateral)
        )
    else:
        if follower_domain is None or leader_domain is None:
            raise SameRouteFollowingUnavailable(
                "same-route following requires ship-domain evidence"
            )
        if route_heading_deg is None or not isfinite(route_heading_deg):
            raise SameRouteFollowingUnavailable(
                "directional same-route following requires route heading evidence"
            )
        relative_now = _route_relative_vector(
            route_heading_deg=route_heading_deg,
            longitudinal_nm=longitudinal_gap,
            signed_lateral_nm=signed_lateral,
        )
        current_clearance = evaluate_pair_ship_domain(
            first_heading_deg=follower_state.through_water_heading_deg,
            second_heading_deg=leader_state.through_water_heading_deg,
            relative_position=relative_now,
            first_domain=follower_domain,
            second_domain=leader_domain,
        )
        required_planar = current_clearance.required_distance_nm
        _root_required, required_longitudinal = _directional_required_longitudinal_gap(
            follower_state=follower_state,
            leader_state=leader_state,
            follower_domain=follower_domain,
            leader_domain=leader_domain,
            route_heading_deg=route_heading_deg,
            signed_lateral_nm=signed_lateral,
        )
    available = max(0.0, longitudinal_gap - required_longitudinal)

    leader_near_route = abs(float(leader_cross_track_nm)) <= config.near_route_cross_track_tolerance_nm + _EPS
    follower_near_route = abs(float(follower_cross_track_nm)) <= config.near_route_cross_track_tolerance_nm + _EPS
    leader_credit = 0.0
    if leader_near_route and follower_near_route:
        leader_credit = _braking_distance(
            leader_state.through_water_speed,
            leader_limits.max_deceleration,
        )
        if not isfinite(leader_credit):
            leader_credit = 0.0

    ceiling = _speed_ceiling(
        budget_nm=available + leader_credit,
        deceleration=follower_limits.max_deceleration,
        reaction_min=config.reaction_interval_min,
    )
    return SameRouteFollowingConstraint(
        pair_key=canonical_pair_key(follower_id, leader_id),
        follower_id=follower_id,
        leader_id=leader_id,
        follower_along_track_nm=float(follower_along_track_nm),
        leader_along_track_nm=float(leader_along_track_nm),
        follower_cross_track_nm=float(follower_cross_track_nm),
        leader_cross_track_nm=float(leader_cross_track_nm),
        lateral_separation_nm=lateral,
        required_planar_domain_separation_nm=required_planar,
        required_longitudinal_domain_gap_nm=required_longitudinal,
        projected_longitudinal_gap_nm=longitudinal_gap,
        available_braking_gap_nm=available,
        leader_stop_credit_nm=leader_credit,
        max_follower_speed_nm_per_min=ceiling,
    )


class SameRouteFollowingSupervisor:
    """Build deterministic nearest-leader constraints from one physical snapshot.

    ``additional_reaction_interval_min_by_vessel`` is an optional microscopic response
    allowance above the generic following-controller reaction interval.  The mapping is
    follower-specific and contains no macroscopic/reference headway.

    ``additional_leader_observation_interval_min_by_vessel`` is an optional allowance
    for stale/uncertain leader-state information.  It is leader-specific and is composed
    with the follower response allowance only when that vessel is actually evaluated as
    the leader of a same-route pair.  This creates a mechanistic leader axis without
    introducing a pair-headway target.  Zero/omitted maps preserve historical behavior.
    """

    def __init__(
        self,
        config: SameRouteFollowingConfig = SameRouteFollowingConfig(),
        *,
        additional_reaction_interval_min_by_vessel: Mapping[str, float] | None = None,
        additional_leader_observation_interval_min_by_vessel: Mapping[str, float] | None = None,
    ) -> None:
        if not isinstance(config, SameRouteFollowingConfig):
            raise TypeError("config must be SameRouteFollowingConfig")

        def _validated_interval_map(
            value: Mapping[str, float] | None,
            *,
            id_error: str,
            value_error: str,
        ) -> MappingProxyType:
            raw = {} if value is None else dict(value)
            for vessel_id, interval in raw.items():
                if not isinstance(vessel_id, str) or not vessel_id:
                    raise ValueError(id_error)
                if not isfinite(interval) or interval < 0.0:
                    raise ValueError(value_error)
            return MappingProxyType({str(k): float(v) for k, v in raw.items()})

        self.config = config
        self.additional_reaction_interval_min_by_vessel = _validated_interval_map(
            additional_reaction_interval_min_by_vessel,
            id_error="reaction-interval override vessel ids must be non-empty strings",
            value_error="additional reaction intervals must be finite and non-negative",
        )
        self.additional_leader_observation_interval_min_by_vessel = _validated_interval_map(
            additional_leader_observation_interval_min_by_vessel,
            id_error="leader-observation override vessel ids must be non-empty strings",
            value_error="additional leader-observation intervals must be finite and non-negative",
        )

    def config_for_follower(self, follower_id: str) -> SameRouteFollowingConfig:
        if not isinstance(follower_id, str) or not follower_id:
            raise ValueError("follower_id must be a non-empty string")
        extra = self.additional_reaction_interval_min_by_vessel.get(follower_id, 0.0)
        if extra <= _EPS:
            return self.config
        return replace(
            self.config,
            reaction_interval_min=self.config.reaction_interval_min + extra,
        )

    def config_for_pair(self, follower_id: str, leader_id: str) -> SameRouteFollowingConfig:
        """Return the physical reaction budget for one concrete follower/leader pair."""
        if not isinstance(leader_id, str) or not leader_id:
            raise ValueError("leader_id must be a non-empty string")
        follower_config = self.config_for_follower(follower_id)
        leader_extra = self.additional_leader_observation_interval_min_by_vessel.get(
            leader_id, 0.0
        )
        if leader_extra <= _EPS:
            return follower_config
        return replace(
            follower_config,
            reaction_interval_min=follower_config.reaction_interval_min + leader_extra,
        )

    @staticmethod
    def same_mission_route(
        first_id: str,
        second_id: str,
        route_targets: Mapping[str, "EnvironmentalRouteHandoffTarget"],
    ) -> bool:
        try:
            first = route_targets[first_id]
            second = route_targets[second_id]
        except KeyError:
            return False
        return first.navigator.follower.route == second.navigator.follower.route

    def same_route_pair_keys_for(
        self,
        vessel_ids: tuple[str, ...],
        route_targets: Mapping[str, "EnvironmentalRouteHandoffTarget"],
    ) -> tuple[PairKey, ...]:
        result: list[PairKey] = []
        ordered = tuple(sorted(set(vessel_ids)))
        for index, first_id in enumerate(ordered):
            for second_id in ordered[index + 1 :]:
                if self.same_mission_route(first_id, second_id, route_targets):
                    result.append(canonical_pair_key(first_id, second_id))
        return tuple(result)

    def delegation_resolver(
        self, route_targets: Mapping[str, "EnvironmentalRouteHandoffTarget"]
    ):
        def resolve(key: PairKey) -> bool:
            return self.same_mission_route(key[0], key[1], route_targets)
        return resolve

    def constraints_for(
        self,
        world: "EnvironmentalWorld2D",
        vessel_ids: tuple[str, ...],
        route_targets: Mapping[str, "EnvironmentalRouteHandoffTarget"],
        profile_catalog: "VesselProfileCatalog | None",
    ) -> tuple[SameRouteFollowingConstraint, ...]:
        snapshot = world.snapshot()
        by_id = {state.vessel_id: state for state in snapshot.vessels}
        active_ids = set(by_id)
        result: list[SameRouteFollowingConstraint] = []

        if profile_catalog is None and vessel_ids:
            raise SameRouteFollowingUnavailable("same-route following requires a profile catalog")
        if self.config.require_zero_environmental_drift:
            for vessel_id in active_ids:
                drift = world.drift_for(vessel_id)
                if abs(drift[0]) > _EPS or abs(drift[1]) > _EPS:
                    raise SameRouteFollowingUnavailable(
                        "same-route following v1 is restricted to zero environmental drift"
                    )

        for follower_id in vessel_ids:
            if follower_id not in active_ids or follower_id not in route_targets:
                continue
            route = route_targets[follower_id].navigator.follower.route
            follower_state = by_id[follower_id]
            follower_projection = same_route_safety_projection(route, follower_state.position)
            assert profile_catalog is not None
            follower_profile = profile_catalog.profile_for(follower_id)
            if follower_profile is None:
                raise SameRouteFollowingUnavailable("same-route following lacks a vessel profile")
            follower_domain = follower_profile.safety_domain
            if follower_domain is None:
                raise SameRouteFollowingUnavailable("same-route following requires ship-domain evidence")
            candidate_constraints: list[SameRouteFollowingConstraint] = []
            for leader_id in sorted(active_ids):
                if leader_id == follower_id or leader_id not in route_targets:
                    continue
                if route_targets[leader_id].navigator.follower.route != route:
                    continue
                leader_state = by_id[leader_id]
                leader_projection = same_route_safety_projection(route, leader_state.position)
                if leader_projection.along_track_nm <= follower_projection.along_track_nm + _EPS:
                    continue
                leader_profile = profile_catalog.profile_for(leader_id)
                if leader_profile is None:
                    raise SameRouteFollowingUnavailable("same-route following lacks a vessel profile")
                leader_domain = leader_profile.safety_domain
                if leader_domain is None:
                    raise SameRouteFollowingUnavailable("same-route following requires ship-domain evidence")
                pair_config = self.config_for_pair(follower_id, leader_id)
                if not same_route_longitudinal_pair_applicable(
                    follower_state=follower_state,
                    leader_state=leader_state,
                    follower_domain=follower_domain,
                    leader_domain=leader_domain,
                    follower_cross_track_nm=float(follower_projection.cross_track_nm),
                    leader_cross_track_nm=float(leader_projection.cross_track_nm),
                    follower_route_heading_deg=float(follower_projection.route_heading_deg),
                    leader_route_heading_deg=float(leader_projection.route_heading_deg),
                    config=pair_config,
                ):
                    continue
                candidate_constraints.append(
                    compute_same_route_following_constraint(
                        follower_id=follower_id,
                        leader_id=leader_id,
                        follower_state=follower_state,
                        leader_state=leader_state,
                        follower_along_track_nm=float(follower_projection.along_track_nm),
                        leader_along_track_nm=float(leader_projection.along_track_nm),
                        follower_cross_track_nm=float(follower_projection.cross_track_nm),
                        leader_cross_track_nm=float(leader_projection.cross_track_nm),
                        follower_profile=follower_profile,
                        leader_profile=leader_profile,
                        follower_limits=world.limits_for(follower_id),
                        leader_limits=world.limits_for(leader_id),
                        route_heading_deg=float(follower_projection.route_heading_deg),
                        config=pair_config,
                    )
                )
            if not candidate_constraints:
                continue
            # The safety-critical leader is the one imposing the lowest physically
            # admissible follower speed, not necessarily the nearest along-track point.
            # Tie-break deterministically by gap and vessel id.
            result.append(
                min(
                    candidate_constraints,
                    key=lambda item: (
                        item.max_follower_speed_nm_per_min,
                        item.projected_longitudinal_gap_nm,
                        item.leader_id,
                    ),
                )
            )
        return tuple(sorted(result, key=lambda item: (item.follower_id, item.leader_id)))
