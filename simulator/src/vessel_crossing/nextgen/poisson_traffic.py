"""Immutable seeded homogeneous-Poisson route traffic realizations for nextgen."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import isfinite
from typing import Iterable

import numpy as np

from ..traffic import generate_poisson_arrivals
from .dynamics import ManeuverLimits, Vessel2DState
from .dynamic_fleet_scheduler import NominalVesselArrival
from .follower import RouteFollower, RouteFollowerState
from .navigator import RouteNavigator
from .recovery import CrossTrackRecovery
from .route import Route2D
from .route_exit import RouteEndAction
from .vessel_classes import VesselClassProfile
from .nominal_fleet_scheduler import NominalFleetTarget


@dataclass(frozen=True, slots=True)
class PoissonRouteStreamSpec:
    stream_id: str
    route: Route2D
    rate_per_minute: float
    cruise_speed: float
    end_action: RouteEndAction = RouteEndAction.REMOVE
    recovery_max_intercept_angle_deg: float = 35.0
    limits: ManeuverLimits | None = None
    vessel_class: VesselClassProfile | None = None

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ValueError("stream_id must be non-empty")
        if not isfinite(self.rate_per_minute) or self.rate_per_minute < 0:
            raise ValueError("rate_per_minute must be finite and non-negative")
        if not isfinite(self.cruise_speed) or self.cruise_speed <= 0:
            raise ValueError("cruise_speed must be finite and positive")
        if (
            self.vessel_class is not None
            and self.limits is not None
            and self.limits != self.vessel_class.maneuver_limits
        ):
            raise ValueError("stream maneuver limits conflict with vessel_class profile")
        if (
            self.vessel_class is not None
            and self.vessel_class.speed_envelope is not None
            and not self.vessel_class.speed_envelope.permits(self.cruise_speed)
        ):
            raise ValueError("stream cruise_speed exceeds vessel_class speed envelope")


@dataclass(frozen=True, slots=True)
class PoissonStreamRealization:
    stream_id: str
    arrival_times: tuple[float, ...]
    vessel_ids: tuple[str, ...]
    vessel_class_id: str | None = None


@dataclass(frozen=True, slots=True)
class PoissonTrafficRealization:
    seed: int
    start_time: float
    horizon: float
    streams: tuple[PoissonStreamRealization, ...]
    arrivals: tuple[NominalVesselArrival, ...]

    @property
    def count(self) -> int:
        return len(self.arrivals)

    def stream(self, stream_id: str) -> PoissonStreamRealization:
        for item in self.streams:
            if item.stream_id == stream_id:
                return item
        raise KeyError(stream_id)


def _subseed(seed: int, stream_id: str) -> int:
    """Stable order-independent pseudorandom substream seed."""
    digest = hashlib.sha256(f"vessel-crossing-nextgen\0{seed}\0{stream_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big", signed=False)


def _arrival_from_spec(spec: PoissonRouteStreamSpec, *, at: float, index: int) -> NominalVesselArrival:
    vessel_id = f"{spec.stream_id}-{index:06d}"
    position = spec.route.waypoints[0]
    heading = spec.route.segment_heading(0)
    vessel = Vessel2DState(vessel_id, position, heading, spec.cruise_speed)
    navigator = RouteNavigator(
        RouteFollower(spec.route, cruise_speed=spec.cruise_speed),
        CrossTrackRecovery(
            spec.route,
            max_intercept_angle_deg=spec.recovery_max_intercept_angle_deg,
        ),
    )
    target = NominalFleetTarget(
        vessel_id,
        navigator,
        RouteFollowerState(),
        spec.end_action,
    )
    return NominalVesselArrival(at, vessel, target, spec.limits, spec.vessel_class)


def generate_poisson_route_traffic(
    specs: Iterable[PoissonRouteStreamSpec],
    *,
    horizon: float,
    seed: int,
    start_time: float = 0.0,
) -> PoissonTrafficRealization:
    """Generate one replayable independent-Poisson realization for multiple route streams.

    Each ``stream_id`` receives a deterministic substream derived from ``seed`` and the
    identifier. Therefore reordering specs, or adding an unrelated stream, does not alter
    the arrival times of an existing stream with the same ID/rate/horizon.
    """
    specs_tuple = tuple(specs)
    if not specs_tuple:
        raise ValueError("at least one Poisson route stream is required")
    ids = [spec.stream_id for spec in specs_tuple]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate Poisson stream_id")
    if not isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be finite and non-negative")
    if not isfinite(start_time):
        raise ValueError("start_time must be finite")

    stream_results: list[PoissonStreamRealization] = []
    arrivals: list[NominalVesselArrival] = []
    for spec in sorted(specs_tuple, key=lambda item: item.stream_id):
        rng = np.random.default_rng(_subseed(seed, spec.stream_id))
        times = generate_poisson_arrivals(
            spec.rate_per_minute,
            horizon,
            rng,
            start_time=start_time,
        )
        stream_arrivals = [
            _arrival_from_spec(spec, at=float(at), index=index)
            for index, at in enumerate(times, start=1)
        ]
        arrivals.extend(stream_arrivals)
        stream_results.append(
            PoissonStreamRealization(
                stream_id=spec.stream_id,
                arrival_times=tuple(float(t) for t in times),
                vessel_ids=tuple(item.vessel.vessel_id for item in stream_arrivals),
                vessel_class_id=(None if spec.vessel_class is None else spec.vessel_class.class_id),
            )
        )

    arrivals.sort(key=lambda item: (item.at, item.vessel.vessel_id))
    return PoissonTrafficRealization(
        seed=int(seed),
        start_time=float(start_time),
        horizon=float(horizon),
        streams=tuple(stream_results),
        arrivals=tuple(arrivals),
    )
