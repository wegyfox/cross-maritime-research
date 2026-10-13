"""Vertical stochastic route-network adapter for the persistent environmental runtime.

A137 closes a product-level composition gap rather than adding another control law.  The
existing route-network/Poisson layer produces ground-referenced nominal arrivals, while
A124-A128 own environmental fleet membership, route metadata and persistent encounter
control.  This module binds those contracts explicitly:

* resolve and realize seeded Poisson network traffic first;
* translate each nominal initial **ground** velocity through the selected per-stream
  environmental drift into an exact water-relative ingress state;
* bind the existing navigator/end-action/profile metadata to A125 ingress records;
* cold-start one empty per-vessel ``EnvironmentalWorld2D``/``EncounterRegistry``;
* admit the whole realized arrival set deterministically and run the persistent runtime
  on that same physical clock.

No new traffic process, encounter observer, command authority or physical clock is
introduced.  ``cruise_speed`` keeps its historical route/ground-speed meaning.  A vessel
profile speed envelope is checked again after drift translation because environmental
control constrains **through-water** speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping

from ..kinematics import Vector2
from .encounter_registry import EncounterDiscoveryConfig, EncounterRegistry
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_fleet_ingress import (
    EnvironmentalFleetIngressAdmission,
    EnvironmentalFleetIngressRecord,
)
from .environmental_fleet_multi_owner_recapture import EnvironmentalFleetMultiOwnerRecaptureConfig
from .environmental_fleet_runtime import (
    EnvironmentalFleetRuntimeConfig,
    EnvironmentalFleetRuntimeStartResult,
    start_environmental_fleet_runtime,
)
from .environmental_idle_fleet_control import EnvironmentalPersistentFleetControlResult
from .environmental_route_handoff import (
    EnvironmentalGroundVelocityUnavailable,
    EnvironmentalRouteHandoffTarget,
    command_for_desired_ground_velocity,
)
from .environmental_world import EnvironmentalWorld2D
from .lifecycle import EncounterLifecycleConfig
from .network_traffic import (
    PoissonNetworkStreamSpec,
    PoissonNetworkTrafficRealization,
    generate_poisson_network_traffic,
)
from .route_network import RouteNetwork
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalNetworkScenarioError(RuntimeError):
    """Raised when a stochastic network scenario cannot be bound to environmental runtime."""


class EnvironmentalNetworkIngressTimingPolicy(str, Enum):
    """Public A141 contract between stochastic demand and physical fleet admission.

    A141 deliberately freezes the v1 behavior instead of silently altering the Poisson
    process: every realized demand timestamp is also the physical ingress timestamp.
    Dense demand that is physically inadmissible therefore remains explicit modeled
    infeasibility in the higher research layers.  A future FIFO policy must be a new,
    explicit admission model with separately archived demand/admission timestamps.
    """

    DEMAND_TIME_IS_PHYSICAL_INGRESS = "DEMAND_TIME_IS_PHYSICAL_INGRESS"


def _finite_drift(value: Vector2) -> Vector2:
    if len(value) != 2:
        raise ValueError("environmental drift must contain exactly two components")
    x, y = float(value[0]), float(value[1])
    if not isfinite(x) or not isfinite(y):
        raise ValueError("environmental drift components must be finite")
    return (x, y)


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkScenarioConfig:
    """Execution policy for one seeded environmental network realization.

    ``traffic_horizon_min`` is the Poisson generation duration measured from
    ``start_time``.  ``end_time`` is the absolute physical runtime horizon and defaults
    to ``start_time + traffic_horizon_min``.
    """

    seed: int
    traffic_horizon_min: float
    discovery: EncounterDiscoveryConfig
    lifecycle: EncounterLifecycleConfig
    recapture: EnvironmentalFleetMultiOwnerRecaptureConfig
    start_time: float = 0.0
    end_time: float | None = None
    default_environmental_drift_velocity: Vector2 = (0.0, 0.0)
    stream_environmental_drift_velocity: Mapping[str, Vector2] = field(default_factory=dict)
    runtime: EnvironmentalFleetRuntimeConfig = EnvironmentalFleetRuntimeConfig()
    ingress_timing_policy: EnvironmentalNetworkIngressTimingPolicy = (
        EnvironmentalNetworkIngressTimingPolicy.DEMAND_TIME_IS_PHYSICAL_INGRESS
    )

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if not isfinite(self.traffic_horizon_min) or self.traffic_horizon_min < 0.0:
            raise ValueError("traffic_horizon_min must be finite and non-negative")
        if not isfinite(self.start_time) or self.start_time < 0.0:
            raise ValueError("start_time must be finite and non-negative")
        end_time = self.start_time + self.traffic_horizon_min if self.end_time is None else self.end_time
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.start_time + self.traffic_horizon_min - _EPS:
            raise ValueError("end_time must cover the complete Poisson traffic generation window")
        if not isinstance(self.discovery, EncounterDiscoveryConfig):
            raise TypeError("discovery must be EncounterDiscoveryConfig")
        if not isinstance(self.lifecycle, EncounterLifecycleConfig):
            raise TypeError("lifecycle must be EncounterLifecycleConfig")
        if not isinstance(self.recapture, EnvironmentalFleetMultiOwnerRecaptureConfig):
            raise TypeError("recapture must be EnvironmentalFleetMultiOwnerRecaptureConfig")
        if not isinstance(self.runtime, EnvironmentalFleetRuntimeConfig):
            raise TypeError("runtime must be EnvironmentalFleetRuntimeConfig")
        if not isinstance(self.ingress_timing_policy, EnvironmentalNetworkIngressTimingPolicy):
            raise TypeError("ingress_timing_policy must be EnvironmentalNetworkIngressTimingPolicy")
        default_drift = _finite_drift(self.default_environmental_drift_velocity)
        stream_drifts = {
            str(stream_id): _finite_drift(drift)
            for stream_id, drift in self.stream_environmental_drift_velocity.items()
        }
        if any(not stream_id for stream_id in stream_drifts):
            raise ValueError("stream environmental drift keys must be non-empty")
        object.__setattr__(self, "end_time", float(end_time))
        object.__setattr__(self, "default_environmental_drift_velocity", default_drift)
        object.__setattr__(self, "stream_environmental_drift_velocity", MappingProxyType(stream_drifts))

    def drift_for_stream(self, stream_id: str) -> Vector2:
        return self.stream_environmental_drift_velocity.get(
            stream_id,
            self.default_environmental_drift_velocity,
        )


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkScenarioIngress:
    """One demand realization bound to one physical environmental admission.

    ``demand_time`` is the unchanged Poisson timestamp. ``admission_time`` is the
    physical fleet-membership timestamp. Under the frozen A141 v1 contract they are
    equal; keeping both names in evidence prevents the two concepts from being
    conflated if an explicit queueing policy is added later.
    """

    stream_id: str
    nominal_vessel_id: str
    demand_time: float
    environmental_drift_velocity: Vector2
    record: EnvironmentalFleetIngressRecord

    @property
    def admission_time(self) -> float:
        return self.record.at

    @property
    def waiting_time_min(self) -> float:
        return self.admission_time - self.demand_time

    @property
    def at(self) -> float:
        """Compatibility alias for the physical admission timestamp."""
        return self.admission_time

    @property
    def vessel_id(self) -> str:
        return self.record.vessel_id


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkScenarioResult:
    """One complete stochastic-network-to-persistent-runtime execution result."""

    traffic: PoissonNetworkTrafficRealization
    ingress: tuple[EnvironmentalNetworkScenarioIngress, ...]
    admissions: tuple[EnvironmentalFleetIngressAdmission, ...]
    runtime_start: EnvironmentalFleetRuntimeStartResult
    run: EnvironmentalPersistentFleetControlResult
    world: EnvironmentalWorld2D
    registry: EncounterRegistry
    profile_catalog: VesselProfileCatalog
    nominal_route_owner_ids: tuple[str, ...] = ()

    @property
    def final_time(self) -> float:
        return self.run.snapshot.time

    @property
    def realized_vessel_ids(self) -> tuple[str, ...]:
        return tuple(item.vessel_id for item in self.ingress)


def _stream_id_by_vessel(
    traffic: PoissonNetworkTrafficRealization,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for stream in traffic.streams:
        for vessel_id in stream.vessel_ids:
            if vessel_id in result:
                raise EnvironmentalNetworkScenarioError(
                    f"realized vessel id {vessel_id!r} appears in multiple Poisson streams"
                )
            result[vessel_id] = stream.stream_id
    if set(result) != {item.vessel.vessel_id for item in traffic.arrivals}:
        raise EnvironmentalNetworkScenarioError(
            "Poisson stream realization vessel ids do not cover the arrival realization exactly"
        )
    return result


def _environmental_ingress_records(
    traffic: PoissonNetworkTrafficRealization,
    config: EnvironmentalNetworkScenarioConfig,
) -> tuple[EnvironmentalNetworkScenarioIngress, ...]:
    stream_by_vessel = _stream_id_by_vessel(traffic)
    result: list[EnvironmentalNetworkScenarioIngress] = []

    for arrival in traffic.arrivals:
        vessel_id = arrival.vessel.vessel_id
        stream_id = stream_by_vessel[vessel_id]
        drift = config.drift_for_stream(stream_id)
        if arrival.limits is None:
            raise EnvironmentalNetworkScenarioError(
                f"environmental stochastic stream {stream_id!r} has no maneuver limits "
                f"for realized vessel {vessel_id!r}"
            )

        # The Poisson/route layer's heading+speed are Earth-fixed nominal ground motion.
        # Environmental dynamics instead control through-water motion, so derive the
        # exact initial water-relative state that composes with drift to the same ground
        # velocity.  A profile envelope is checked in the controlled frame here.
        placeholder = WaterRelativeVesselState(
            vessel_id=vessel_id,
            position=arrival.vessel.position,
            through_water_heading_deg=arrival.vessel.heading_deg,
            through_water_speed=0.0,
        )
        try:
            translated = command_for_desired_ground_velocity(
                placeholder,
                drift,
                desired_ground_heading_deg=arrival.vessel.heading_deg,
                desired_ground_speed=arrival.vessel.speed,
                profile=arrival.vessel_class,
            )
        except EnvironmentalGroundVelocityUnavailable as exc:
            raise EnvironmentalNetworkScenarioError(
                f"environmental drift makes nominal ingress velocity unavailable for "
                f"vessel {vessel_id!r} in stream {stream_id!r}"
            ) from exc

        water_state = WaterRelativeVesselState(
            vessel_id=vessel_id,
            position=arrival.vessel.position,
            through_water_heading_deg=translated.command.target_through_water_heading_deg,
            through_water_speed=translated.command.target_through_water_speed,
        )
        route_target = EnvironmentalRouteHandoffTarget(
            vessel_id=vessel_id,
            navigator=arrival.target.navigator,
            follower_state=arrival.target.follower_state,
            end_action=arrival.target.end_action,
        )
        record = EnvironmentalFleetIngressRecord(
            at=arrival.at,
            vessel=water_state,
            route_target=route_target,
            limits=arrival.limits,
            environmental_drift_velocity=drift,
            profile=arrival.vessel_class,
        )
        result.append(
            EnvironmentalNetworkScenarioIngress(
                stream_id=stream_id,
                nominal_vessel_id=vessel_id,
                demand_time=arrival.at,
                environmental_drift_velocity=drift,
                record=record,
            )
        )

    return tuple(sorted(result, key=lambda item: (item.at, item.vessel_id)))


def build_environmental_network_scenario_ingress(
    traffic: PoissonNetworkTrafficRealization,
    config: EnvironmentalNetworkScenarioConfig,
) -> tuple[EnvironmentalNetworkScenarioIngress, ...]:
    """Public deterministic adapter from realized network demand to ingress records.

    This is the read-only materialization seam used when a caller must preserve an
    already-realized CRN traffic input while executing a specialized world/runtime.
    It is exactly the same conversion used by ``run_environmental_poisson_network_scenario``.
    """
    return _environmental_ingress_records(traffic, config)


def run_environmental_poisson_network_scenario(
    network: RouteNetwork,
    streams: Iterable[PoissonNetworkStreamSpec],
    *,
    config: EnvironmentalNetworkScenarioConfig,
) -> EnvironmentalNetworkScenarioResult:
    """Generate and execute one complete seeded environmental network scenario.

    All route resolution, Poisson realization and environmental frame translation are
    completed before the world/registry runtime is created.  A malformed stream or an
    unavailable drift-adjusted speed therefore cannot leave a partially admitted fleet.
    """

    if not isinstance(network, RouteNetwork):
        raise TypeError("network must be RouteNetwork")
    if not isinstance(config, EnvironmentalNetworkScenarioConfig):
        raise TypeError("config must be EnvironmentalNetworkScenarioConfig")
    stream_items = tuple(streams)
    stream_ids = {item.stream_id for item in stream_items}
    unknown_drift_streams = set(config.stream_environmental_drift_velocity) - stream_ids
    if unknown_drift_streams:
        raise EnvironmentalNetworkScenarioError(
            f"environmental drift configured for unknown streams: {sorted(unknown_drift_streams)!r}"
        )

    traffic = generate_poisson_network_traffic(
        network,
        stream_items,
        horizon=config.traffic_horizon_min,
        seed=config.seed,
        start_time=config.start_time,
    )
    ingress = _environmental_ingress_records(traffic, config)

    # Per-vessel dictionaries are intentional even when every stream currently shares
    # one value: future Poisson arrivals may carry heterogeneous limits/drifts and A125
    # is the single authority that publishes those values at membership time.
    world = EnvironmentalWorld2D(
        (),
        limits={},
        environmental_drift_velocity={},
        start_time=config.start_time,
    )
    profiles = VesselProfileCatalog()
    registry = EncounterRegistry(
        discovery=config.discovery,
        lifecycle=config.lifecycle,
        profile_catalog=profiles,
    )
    started = start_environmental_fleet_runtime(
        registry,
        world,
        {},
        recapture_config=config.recapture,
        config=config.runtime,
        profile_catalog=profiles,
    )

    admissions = tuple(started.session.schedule_ingress(item.record) for item in ingress)

    # Stop exactly at each realized membership time.  The runtime itself applies the
    # arrival and commits the canonical fleet observation.  Only then may a vessel that
    # is not claimed by the resulting avoidance dispatch receive direct nominal route
    # authority.  These are caller horizons on the one existing world clock, not new
    # observations or synthetic scheduler events.
    nominal_route_owner_ids: list[str] = []
    index = 0
    while index < len(ingress):
        boundary = ingress[index].at
        batch: list[str] = []
        while index < len(ingress) and abs(ingress[index].at - boundary) <= _EPS:
            batch.append(ingress[index].vessel_id)
            index += 1
        started.session.run_until(boundary)
        nominal_route_owner_ids.extend(
            started.session.admit_current_clear_route_targets(tuple(batch))
        )

    run = started.session.run_until(config.end_time)
    return EnvironmentalNetworkScenarioResult(
        traffic=traffic,
        ingress=ingress,
        admissions=admissions,
        runtime_start=started,
        run=run,
        world=world,
        registry=registry,
        profile_catalog=profiles,
        nominal_route_owner_ids=tuple(nominal_route_owner_ids),
    )
