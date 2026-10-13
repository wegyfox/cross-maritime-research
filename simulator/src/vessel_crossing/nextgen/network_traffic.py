"""Seeded Poisson traffic whose nominal routes are resolved from ``RouteNetwork``.

This layer composes two already-established contracts instead of inventing another
traffic process:

* a63 resolves an immutable ``RouteRequest`` to deterministic network geometry;
* a21/a57 ``generate_poisson_route_traffic`` owns the homogeneous-Poisson substreams,
  vessel IDs, class/speed-envelope validation and nominal arrival construction.

The resolved path is archived in the realization, so a stochastic scenario records both
*when* vessels arrived and *which exact network path* they were assigned.  No congestion,
VTS priority, random route choice or feedback from encounters is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from .dynamics import ManeuverLimits
from .dynamic_fleet_scheduler import NominalVesselArrival
from .poisson_traffic import PoissonRouteStreamSpec, generate_poisson_route_traffic
from .route_exit import RouteEndAction
from .route_network import RouteNetwork, RouteNetworkPath, RouteRequest
from .vessel_classes import VesselClassProfile


@dataclass(frozen=True, slots=True)
class PoissonNetworkStreamSpec:
    stream_id: str
    request: RouteRequest
    rate_per_minute: float
    cruise_speed: float
    end_action: RouteEndAction = RouteEndAction.REMOVE
    recovery_max_intercept_angle_deg: float = 35.0
    limits: ManeuverLimits | None = None
    vessel_class: VesselClassProfile | None = None

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ValueError("stream_id must be non-empty")
        if not isinstance(self.request, RouteRequest):
            raise TypeError("request must be a RouteRequest")
        if not isfinite(self.rate_per_minute) or self.rate_per_minute < 0.0:
            raise ValueError("rate_per_minute must be finite and non-negative")
        if not isfinite(self.cruise_speed) or self.cruise_speed <= 0.0:
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
class PoissonNetworkStreamRealization:
    stream_id: str
    request: RouteRequest
    resolved_path: RouteNetworkPath
    arrival_times: tuple[float, ...]
    vessel_ids: tuple[str, ...]
    vessel_class_id: str | None = None


@dataclass(frozen=True, slots=True)
class PoissonNetworkTrafficRealization:
    seed: int
    start_time: float
    horizon: float
    streams: tuple[PoissonNetworkStreamRealization, ...]
    arrivals: tuple[NominalVesselArrival, ...]

    @property
    def count(self) -> int:
        return len(self.arrivals)

    def stream(self, stream_id: str) -> PoissonNetworkStreamRealization:
        for item in self.streams:
            if item.stream_id == stream_id:
                return item
        raise KeyError(stream_id)


def _resolved_route_spec(
    spec: PoissonNetworkStreamSpec,
    path: RouteNetworkPath,
) -> PoissonRouteStreamSpec:
    return PoissonRouteStreamSpec(
        stream_id=spec.stream_id,
        route=path.route,
        rate_per_minute=spec.rate_per_minute,
        cruise_speed=spec.cruise_speed,
        end_action=spec.end_action,
        recovery_max_intercept_angle_deg=spec.recovery_max_intercept_angle_deg,
        limits=spec.limits,
        vessel_class=spec.vessel_class,
    )


def generate_poisson_network_traffic(
    network: RouteNetwork,
    specs: Iterable[PoissonNetworkStreamSpec],
    *,
    horizon: float,
    seed: int,
    start_time: float = 0.0,
) -> PoissonNetworkTrafficRealization:
    """Resolve every route request, then reuse the existing seeded Poisson generator.

    Resolution is completed for the full stream set before stochastic arrivals are
    generated.  A disconnected/invalid request therefore fails the scenario before any
    partial traffic realization exists.  Existing per-``stream_id`` RNG independence is
    inherited unchanged from ``generate_poisson_route_traffic``.
    """
    if not isinstance(network, RouteNetwork):
        raise TypeError("network must be a RouteNetwork")
    items = tuple(specs)
    if not items:
        raise ValueError("at least one Poisson network stream is required")
    ids = [item.stream_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate Poisson stream_id")

    ordered = tuple(sorted(items, key=lambda item: item.stream_id))
    resolved = tuple((item, network.resolve(item.request)) for item in ordered)
    route_specs = tuple(_resolved_route_spec(item, path) for item, path in resolved)
    base = generate_poisson_route_traffic(
        route_specs,
        horizon=horizon,
        seed=seed,
        start_time=start_time,
    )

    realized_streams: list[PoissonNetworkStreamRealization] = []
    for item, path in resolved:
        stream = base.stream(item.stream_id)
        realized_streams.append(
            PoissonNetworkStreamRealization(
                stream_id=item.stream_id,
                request=item.request,
                resolved_path=path,
                arrival_times=stream.arrival_times,
                vessel_ids=stream.vessel_ids,
                vessel_class_id=stream.vessel_class_id,
            )
        )

    return PoissonNetworkTrafficRealization(
        seed=base.seed,
        start_time=base.start_time,
        horizon=base.horizon,
        streams=tuple(realized_streams),
        arrivals=base.arrivals,
    )
