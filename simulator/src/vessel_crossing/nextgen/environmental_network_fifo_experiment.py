"""Portable single-run experiment surface for the opt-in A151 FIFO source runtime.

A152 is deliberately separate from the frozen A141 experiment/result contract.  It
reuses the same route-network and seeded Poisson demand machinery, but physical fleet
membership is owned by the A148-A151 arm-local FIFO policy rather than by
``DEMAND_TIME_IS_PHYSICAL_INGRESS``.

The public JSON document is evidence only: it contains immutable demand, arm-local
admission/waiting, source planning/generation/boundary evidence, final queue metrics and
terminal state.  It has no restore or JSON-to-runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from hashlib import sha256
from typing import Any, Callable

from ..archive import environment_manifest
from ..version import __version__
from .encounter_registry import EncounterRegistry
from .environmental_network_experiment import _canonical_json, _json_value
from .environmental_network_scenario import (
    EnvironmentalNetworkScenarioConfig,
    EnvironmentalNetworkScenarioIngress,
    _environmental_ingress_records,
)
from .environmental_network_source_fifo_runtime import (
    EnvironmentalNetworkSourceFIFORuntimeResult,
    EnvironmentalNetworkSourceFIFORuntimeStartResult,
    start_environmental_network_source_fifo_runtime,
)
from .environmental_network_source_recheck import EnvironmentalNetworkSourceRecheckConfig
from .environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceQueueMetrics,
    EnvironmentalNetworkSourceReservoir,
    source_queue_metrics,
)
from .environmental_world import EnvironmentalWorld2D
from .environmental_world_builder import (
    EnvironmentalWorldBuildContext,
    EnvironmentalWorldBuilder,
    build_environmental_world,
)
from .network_traffic import (
    PoissonNetworkStreamSpec,
    PoissonNetworkTrafficRealization,
    generate_poisson_network_traffic,
)
from .route_network import RouteNetwork
from .vessel_classes import VesselProfileCatalog
from .same_route_following import SameRouteFollowingSupervisor

_FIFO_CONFIG_FORMAT = "vessel-crossing-nextgen-environmental-network-fifo-experiment"
_FIFO_CONFIG_SCHEMA_VERSION = 1
_FIFO_RESULT_FORMAT = "vessel-crossing-nextgen-environmental-network-fifo-result"
_FIFO_RESULT_SCHEMA_VERSION = 1
_FIFO_RUN_DOMAIN = b"vessel-crossing-nextgen-environmental-network-fifo-run\x00"


class EnvironmentalNetworkFIFOExperimentError(RuntimeError):
    """Raised when the separately named FIFO experiment cannot be executed safely."""


class EnvironmentalNetworkFIFOAdmissionPolicy(str, Enum):
    """A152 physical source-admission policy identity."""

    ARM_LOCAL_FIFO_SHIP_DOMAIN = "ARM_LOCAL_FIFO_SHIP_DOMAIN"
    ARM_LOCAL_FIFO_SHIP_DOMAIN_WITH_DYNAMIC_SAME_ROUTE_BRAKING = (
        "ARM_LOCAL_FIFO_SHIP_DOMAIN_WITH_DYNAMIC_SAME_ROUTE_BRAKING"
    )


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOExperimentConfig:
    """Immutable A152 definition, separate from the A141 experiment schema.

    ``scenario`` supplies stochastic/environment/runtime parameters only.  Its historical
    A141 ingress-policy field is deliberately not serialized into this config because
    A152 has its own explicit ``admission_policy`` identity.
    """

    network: RouteNetwork
    streams: tuple[PoissonNetworkStreamSpec, ...]
    scenario: EnvironmentalNetworkScenarioConfig
    recheck: EnvironmentalNetworkSourceRecheckConfig = EnvironmentalNetworkSourceRecheckConfig()
    admission_policy: EnvironmentalNetworkFIFOAdmissionPolicy = (
        EnvironmentalNetworkFIFOAdmissionPolicy.ARM_LOCAL_FIFO_SHIP_DOMAIN
    )
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.network, RouteNetwork):
            raise TypeError("network must be RouteNetwork")
        streams = tuple(self.streams)
        if not streams:
            raise ValueError("at least one Poisson network stream is required")
        if any(not isinstance(item, PoissonNetworkStreamSpec) for item in streams):
            raise TypeError("streams must contain only PoissonNetworkStreamSpec values")
        ids = [item.stream_id for item in streams]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate Poisson stream_id")
        if not isinstance(self.scenario, EnvironmentalNetworkScenarioConfig):
            raise TypeError("scenario must be EnvironmentalNetworkScenarioConfig")
        if not isinstance(self.recheck, EnvironmentalNetworkSourceRecheckConfig):
            raise TypeError("recheck must be EnvironmentalNetworkSourceRecheckConfig")
        if not isinstance(self.admission_policy, EnvironmentalNetworkFIFOAdmissionPolicy):
            raise TypeError("admission_policy must be EnvironmentalNetworkFIFOAdmissionPolicy")
        for stream in streams:
            if stream.limits is None:
                raise EnvironmentalNetworkFIFOExperimentError(
                    f"FIFO stream {stream.stream_id!r} requires explicit maneuver limits"
                )
            if stream.vessel_class is None or stream.vessel_class.safety_domain is None:
                raise EnvironmentalNetworkFIFOExperimentError(
                    f"FIFO stream {stream.stream_id!r} requires an explicit vessel class with safety domain"
                )
        if self.label is not None:
            label = self.label.strip()
            if not label:
                raise ValueError("label must be non-empty when supplied")
            object.__setattr__(self, "label", label)
        object.__setattr__(self, "streams", tuple(sorted(streams, key=lambda item: item.stream_id)))

    def _scenario_parameters(self) -> dict[str, Any]:
        data = _json_value(self.scenario)
        assert isinstance(data, dict)
        # A141 owns this historical field; A152 must not imply that demand time is
        # physical ingress when the FIFO layer explicitly separates the two.
        data.pop("ingress_timing_policy", None)
        return data

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _FIFO_CONFIG_FORMAT,
            "schema_version": _FIFO_CONFIG_SCHEMA_VERSION,
            "label": self.label,
            "admission_policy": self.admission_policy.value,
            "network": _json_value(self.network),
            "streams": _json_value(self.streams),
            "scenario_parameters": self._scenario_parameters(),
            "source_recheck": _json_value(self.recheck),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def config_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()

    @property
    def run_id(self) -> str:
        material = (
            _FIFO_RUN_DOMAIN
            + __version__.encode("utf-8")
            + b"\x00"
            + self.config_sha256.encode("ascii")
        )
        return sha256(material).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOExperimentSummary:
    run_id: str
    config_sha256: str
    software_version: str
    seed: int
    start_time: float
    traffic_horizon_min: float
    end_time: float
    realized_demand_count: int
    physical_admission_count: int
    pending_source_count: int
    due_pending_source_count: int
    mean_admitted_waiting_time_min: float | None
    max_admitted_waiting_time_min: float | None
    final_source_generation: int
    source_plan_count: int
    source_boundary_count: int
    final_time: float
    final_vessel_count: int
    final_vessel_ids: tuple[str, ...]
    final_snapshot_sha256: str
    fleet_observation_count: int
    stop_reason: str


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOExperimentResult:
    """A152 result with live diagnostic handles plus frozen canonical evidence.

    ``runtime_start`` and ``run`` remain available for in-process diagnostics, but the
    portable result document is captured exactly once at experiment completion.  Later
    continuation or mutation of the live runtime therefore cannot rewrite historical
    evidence or change ``result_sha256``.
    """

    config: EnvironmentalNetworkFIFOExperimentConfig
    summary: EnvironmentalNetworkFIFOExperimentSummary
    traffic: PoissonNetworkTrafficRealization
    demand_bindings: tuple[EnvironmentalNetworkScenarioIngress, ...]
    runtime_start: EnvironmentalNetworkSourceFIFORuntimeStartResult
    run: EnvironmentalNetworkSourceFIFORuntimeResult
    final_queue_metrics: EnvironmentalNetworkSourceQueueMetrics
    canonical_bytes: bytes = field(repr=False)

    @property
    def run_id(self) -> str:
        return self.summary.run_id

    @property
    def config_sha256(self) -> str:
        return self.summary.config_sha256

    def to_json_object(self) -> dict[str, Any]:
        document = json.loads(self.canonical_bytes.decode("utf-8"))
        if not isinstance(document, dict):  # pragma: no cover - construction invariant
            raise EnvironmentalNetworkFIFOExperimentError("frozen result document is not a JSON object")
        return document

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes

    @property
    def result_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()


def _frozen_result_document(
    *,
    config: EnvironmentalNetworkFIFOExperimentConfig,
    summary: EnvironmentalNetworkFIFOExperimentSummary,
    traffic: PoissonNetworkTrafficRealization,
    initial_resolution: Any,
    admissions: tuple[Any, ...],
    ingress_audits: tuple[Any, ...],
    plans: tuple[Any, ...],
    boundaries: tuple[Any, ...],
    final_generation: int,
    final_queue_metrics: EnvironmentalNetworkSourceQueueMetrics,
    final_state: dict[str, Any],
    execution_environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": _FIFO_RESULT_FORMAT,
        "schema_version": _FIFO_RESULT_SCHEMA_VERSION,
        "run_id": summary.run_id,
        "config_sha256": summary.config_sha256,
        "config": config.to_json_object(),
        "summary": _json_value(summary),
        "execution_environment": execution_environment,
        "traffic": _json_value(traffic),
        "source": {
            "demands": _json_value(initial_resolution.reservoir.demands),
            "initial_resolution": _json_value(initial_resolution),
            "admissions": _json_value(admissions),
            "ingress_audits": _json_value(ingress_audits),
            "plans": _json_value(plans),
            "boundaries": _json_value(boundaries),
            "final_generation": final_generation,
            "final_queue_metrics": _json_value(final_queue_metrics),
        },
        "final_state": final_state,
    }


def _validate_stream_drifts(config: EnvironmentalNetworkFIFOExperimentConfig) -> None:
    stream_ids = {item.stream_id for item in config.streams}
    unknown = set(config.scenario.stream_environmental_drift_velocity) - stream_ids
    if unknown:
        raise EnvironmentalNetworkFIFOExperimentError(
            f"environmental drift configured for unknown streams: {sorted(unknown)!r}"
        )


def _validate_pre_realized_fifo_traffic(
    config: EnvironmentalNetworkFIFOExperimentConfig,
    traffic: PoissonNetworkTrafficRealization,
) -> None:
    if not isinstance(traffic, PoissonNetworkTrafficRealization):
        raise TypeError("traffic must be PoissonNetworkTrafficRealization")
    scenario = config.scenario
    if traffic.seed != scenario.seed:
        raise EnvironmentalNetworkFIFOExperimentError("pre-realized traffic seed disagrees with config")
    if abs(traffic.start_time - scenario.start_time) > 1e-10:
        raise EnvironmentalNetworkFIFOExperimentError(
            "pre-realized traffic start_time disagrees with config"
        )
    if abs(traffic.horizon - scenario.traffic_horizon_min) > 1e-10:
        raise EnvironmentalNetworkFIFOExperimentError(
            "pre-realized traffic horizon disagrees with config"
        )
    expected_ids = tuple(item.stream_id for item in config.streams)
    actual_ids = tuple(item.stream_id for item in traffic.streams)
    if actual_ids != expected_ids:
        raise EnvironmentalNetworkFIFOExperimentError(
            "pre-realized traffic stream set/order disagrees with config"
        )
    for spec in config.streams:
        realized = traffic.stream(spec.stream_id)
        expected_path = config.network.resolve(spec.request)
        expected_class_id = None if spec.vessel_class is None else spec.vessel_class.class_id
        if realized.request != spec.request or realized.resolved_path != expected_path:
            raise EnvironmentalNetworkFIFOExperimentError(
                f"pre-realized traffic route disagrees for stream {spec.stream_id!r}"
            )
        if realized.vessel_class_id != expected_class_id:
            raise EnvironmentalNetworkFIFOExperimentError(
                f"pre-realized traffic vessel class disagrees for stream {spec.stream_id!r}"
            )


def _run_environmental_network_fifo_experiment_with_traffic(
    config: EnvironmentalNetworkFIFOExperimentConfig,
    traffic: PoissonNetworkTrafficRealization,
    *,
    world_builder: EnvironmentalWorldBuilder | None = None,
    _world_factory: Callable[[VesselProfileCatalog], EnvironmentalWorld2D] | None = None,
    _same_route_following_override: SameRouteFollowingSupervisor | None = None,
) -> EnvironmentalNetworkFIFOExperimentResult:
    """Run A152 semantics on one already realized CRN input.

    ``world_builder`` is the M0-05 public deterministic extension seam.  The historical
    ``_world_factory`` callback remains package-private for compatibility while internal
    telemetry users migrate; both extension mechanisms are mutually exclusive.

    ``_same_route_following_override`` is a package-private post-freeze research seam for
    identity-bearing wrappers.  It never changes the historical public A152 config/result
    contract by itself; callers that use it must freeze their own mechanism provenance.
    """

    if not isinstance(config, EnvironmentalNetworkFIFOExperimentConfig):
        raise TypeError("config must be EnvironmentalNetworkFIFOExperimentConfig")
    if _same_route_following_override is not None and not isinstance(
        _same_route_following_override, SameRouteFollowingSupervisor
    ):
        raise TypeError(
            "_same_route_following_override must be SameRouteFollowingSupervisor or None"
        )
    _validate_stream_drifts(config)
    _validate_pre_realized_fifo_traffic(config, traffic)

    scenario = config.scenario
    demand_bindings = _environmental_ingress_records(traffic, scenario)
    reservoir = EnvironmentalNetworkSourceReservoir.from_scenario_ingress(demand_bindings)

    if world_builder is not None and _world_factory is not None:
        raise EnvironmentalNetworkFIFOExperimentError(
            "world_builder and legacy _world_factory are mutually exclusive"
        )

    profiles = VesselProfileCatalog()
    world_builder_provenance: dict[str, object] | None = None
    if world_builder is not None:
        built = build_environmental_world(
            world_builder,
            EnvironmentalWorldBuildContext(
                start_time=scenario.start_time,
                profile_catalog=profiles,
            ),
        )
        world = built.world
        world_builder_provenance = built.identity.to_json_object()
    elif _world_factory is not None:
        world = _world_factory(profiles)
        if not isinstance(world, EnvironmentalWorld2D):
            raise TypeError("_world_factory must return EnvironmentalWorld2D")
        if abs(world.time - scenario.start_time) > 1e-10 or world.snapshot().vessels:
            raise EnvironmentalNetworkFIFOExperimentError(
                "telemetry world must be empty and aligned to scenario start_time"
            )
    else:
        world = EnvironmentalWorld2D(
            (),
            limits={},
            environmental_drift_velocity={},
            start_time=scenario.start_time,
        )
    registry = EncounterRegistry(
        discovery=scenario.discovery,
        lifecycle=scenario.lifecycle,
        profile_catalog=profiles,
    )
    started = start_environmental_network_source_fifo_runtime(
        registry,
        world,
        {},
        reservoir,
        recapture_config=scenario.recapture,
        runtime_config=scenario.runtime,
        recheck_config=config.recheck,
        profile_catalog=profiles,
        same_route_dynamic_ingress=(
            (
                _same_route_following_override
                if _same_route_following_override is not None
                else SameRouteFollowingSupervisor()
            )
            if config.admission_policy
            is EnvironmentalNetworkFIFOAdmissionPolicy.ARM_LOCAL_FIFO_SHIP_DOMAIN_WITH_DYNAMIC_SAME_ROUTE_BRAKING
            else None
        ),
        persistent_same_route_following=_same_route_following_override,
    )
    run = started.session.run_until(float(scenario.end_time))
    final_metrics = source_queue_metrics(run.reservoir, at=run.snapshot.time)

    source_ids = tuple(item.vessel_id for item in run.reservoir.admissions)
    audit_ids = tuple(item.vessel_id for item in started.session.ingress_admissions)
    if source_ids != audit_ids:
        raise EnvironmentalNetworkFIFOExperimentError(
            "source admission history disagrees with runtime ingress audit ledger"
        )

    final_ids = tuple(vessel.vessel_id for vessel in run.snapshot.vessels)
    observations = tuple(started.runtime_start.session.coordinator.fleet_observations)
    final_state = {
        "snapshot": _json_value(run.snapshot),
        "commands": _json_value(
            tuple(
                (vessel.vessel_id, started.session.world.command_for(vessel.vessel_id))
                for vessel in run.snapshot.vessels
            )
        ),
    }
    final_state_sha256 = sha256(_canonical_json(final_state)).hexdigest()
    summary = EnvironmentalNetworkFIFOExperimentSummary(
        run_id=config.run_id,
        config_sha256=config.config_sha256,
        software_version=__version__,
        seed=scenario.seed,
        start_time=scenario.start_time,
        traffic_horizon_min=scenario.traffic_horizon_min,
        end_time=float(scenario.end_time),
        realized_demand_count=traffic.count,
        physical_admission_count=final_metrics.admission_count,
        pending_source_count=final_metrics.pending_count,
        due_pending_source_count=final_metrics.due_pending_count,
        mean_admitted_waiting_time_min=final_metrics.mean_admitted_waiting_time_min,
        max_admitted_waiting_time_min=final_metrics.max_admitted_waiting_time_min,
        final_source_generation=run.source_generation,
        source_plan_count=len(started.session.source_plans),
        source_boundary_count=len(run.source_boundaries),
        final_time=run.snapshot.time,
        final_vessel_count=len(final_ids),
        final_vessel_ids=final_ids,
        final_snapshot_sha256=final_state_sha256,
        fleet_observation_count=len(observations),
        stop_reason=run.runtime.stop_reason.value,
    )
    document = _frozen_result_document(
        config=config,
        summary=summary,
        traffic=traffic,
        initial_resolution=started.initial_resolution,
        admissions=tuple(run.reservoir.admissions),
        ingress_audits=tuple(started.session.ingress_admissions),
        plans=tuple(started.session.source_plans),
        boundaries=tuple(run.source_boundaries),
        final_generation=run.source_generation,
        final_queue_metrics=final_metrics,
        final_state=final_state,
        execution_environment=environment_manifest(),
    )
    if world_builder_provenance is not None:
        # Result SHA/archive bytes therefore bind the exact public builder identity,
        # while the historical default/private paths remain byte-for-byte unchanged.
        document["world_builder"] = world_builder_provenance
    canonical_bytes = _canonical_json(document)
    return EnvironmentalNetworkFIFOExperimentResult(
        config=config,
        summary=summary,
        traffic=traffic,
        demand_bindings=demand_bindings,
        runtime_start=started,
        run=run,
        final_queue_metrics=final_metrics,
        canonical_bytes=canonical_bytes,
    )

def run_environmental_network_fifo_experiment_with_traffic(
    config: EnvironmentalNetworkFIFOExperimentConfig,
    traffic: PoissonNetworkTrafficRealization,
    *,
    world_builder: EnvironmentalWorldBuilder | None = None,
) -> EnvironmentalNetworkFIFOExperimentResult:
    """Execute one already-realized CRN input through the public FIFO experiment seam."""

    return _run_environmental_network_fifo_experiment_with_traffic(
        config,
        traffic,
        world_builder=world_builder,
    )


def run_environmental_network_fifo_experiment(
    config: EnvironmentalNetworkFIFOExperimentConfig,
    *,
    world_builder: EnvironmentalWorldBuilder | None = None,
) -> EnvironmentalNetworkFIFOExperimentResult:
    """Execute one seeded Poisson realization through the separately named A151 path."""

    if not isinstance(config, EnvironmentalNetworkFIFOExperimentConfig):
        raise TypeError("config must be EnvironmentalNetworkFIFOExperimentConfig")
    _validate_stream_drifts(config)
    scenario = config.scenario
    traffic = generate_poisson_network_traffic(
        config.network,
        config.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    return _run_environmental_network_fifo_experiment_with_traffic(
        config, traffic, world_builder=world_builder
    )

