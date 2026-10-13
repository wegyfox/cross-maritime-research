"""Reproducible stochastic traffic-scenario construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CrossingGeometry
from .traffic import generate_poisson_arrivals
from .vessel import Vessel


@dataclass(frozen=True, slots=True)
class RouteDemand:
    """Homogeneous Poisson demand for one named route."""

    route_name: str
    rate_per_minute: float

    def __post_init__(self) -> None:
        if not self.route_name:
            raise ValueError("route_name must be non-empty")
        if self.rate_per_minute < 0:
            raise ValueError("rate_per_minute must be non-negative")


@dataclass(frozen=True, slots=True)
class PoissonScenarioConfig:
    """Traffic inputs for one fixed-horizon realization."""

    horizon: float
    desired_speed: float
    route_demands: tuple[RouteDemand, ...]

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.desired_speed < 0:
            raise ValueError("desired_speed must be non-negative")
        names = [demand.route_name for demand in self.route_demands]
        if len(names) != len(set(names)):
            raise ValueError("route demands must have unique route names")


def generate_poisson_scenario(
    geometry: CrossingGeometry,
    config: PoissonScenarioConfig,
    *,
    seed: int,
) -> tuple[Vessel, ...]:
    """Generate one immutable multi-route traffic realization.

    Route streams use child random-number generators spawned from one root seed.
    Demands are sorted by route name before spawning, so merely reordering the
    configuration tuple does not change a route's random realization.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")

    demands = tuple(sorted(config.route_demands, key=lambda demand: demand.route_name))
    known_routes = {route.name for route in geometry.routes}
    unknown = [demand.route_name for demand in demands if demand.route_name not in known_routes]
    if unknown:
        raise ValueError(f"unknown route demand(s): {', '.join(unknown)}")

    root = np.random.SeedSequence(seed)
    child_sequences = root.spawn(len(demands))
    vessels: list[Vessel] = []

    for demand, child in zip(demands, child_sequences):
        route = geometry.by_name(demand.route_name)
        rng = np.random.default_rng(child)
        arrivals = generate_poisson_arrivals(
            demand.rate_per_minute,
            config.horizon,
            rng,
        )
        for index, arrival in enumerate(arrivals, start=1):
            vessels.append(
                Vessel(
                    vessel_id=f"{route.name}-{index:05d}",
                    route=route,
                    arrival_time=float(arrival),
                    desired_speed=config.desired_speed,
                    # Vessels are traffic already moving toward/through the
                    # route entrance; activation is not a cold start from rest.
                    current_speed=config.desired_speed,
                )
            )

    return tuple(sorted(vessels, key=lambda vessel: (vessel.arrival_time, vessel.vessel_id)))
