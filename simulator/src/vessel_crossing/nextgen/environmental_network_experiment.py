"""Reproducible top-level experiment facade for the A137 environmental network runtime.

A138 is intentionally a product-layer composition surface, not a new simulator clock,
observer, traffic generator, controller or metrics engine.  It freezes one complete
A137 scenario definition into a canonical configuration document, derives stable
configuration/run identities, executes the existing vertical scenario once, and returns
both a compact deterministic summary and the full A137 result for deeper inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..archive import environment_manifest
from ..version import __version__
from .environmental_network_scenario import (
    EnvironmentalNetworkIngressTimingPolicy,
    EnvironmentalNetworkScenarioConfig,
    EnvironmentalNetworkScenarioResult,
    run_environmental_poisson_network_scenario,
)
from .network_traffic import PoissonNetworkStreamSpec
from .route_network import RouteNetwork
from .candidate_generation import IntentDrivenCandidatePolicy
from .environmental_joint_replanning import EnvironmentalJointReplanningConfig
from .same_route_following import SameRouteFollowingSupervisor

_EXPERIMENT_FORMAT = "vessel-crossing-nextgen-environmental-network-experiment"
_EXPERIMENT_SCHEMA_VERSION = 2
_RESULT_FORMAT = "vessel-crossing-nextgen-environmental-network-result"
_RESULT_SCHEMA_VERSION = 2
_RUN_DOMAIN = b"vessel-crossing-nextgen-environmental-network-run\x00"


class EnvironmentalNetworkExperimentError(RuntimeError):
    """Raised when a top-level experiment cannot be frozen or executed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    """Convert immutable model/config values into explicit canonical JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise EnvironmentalNetworkExperimentError("experiment config contains non-finite float")
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, RouteNetwork):
        return {
            "nodes": [_json_value(item) for item in sorted(value.nodes, key=lambda item: item.node_id)],
            "edges": [_json_value(item) for item in sorted(value.edges, key=lambda item: item.edge_id)],
        }
    if isinstance(value, SameRouteFollowingSupervisor):
        # The supervisor is immutable policy plumbing around one dataclass config.
        # Canonical experiment identity must describe the policy, not the Python
        # object identity/repr.  Keep the wrapper explicit so future mutable runtime
        # state cannot accidentally enter a frozen protocol hash.
        return {"config": _json_value(value.config)}
    if isinstance(value, EnvironmentalJointReplanningConfig):
        # Backward-compatible optional exact-search backend.  Historical experiment
        # identities predate this default-off field, so omit it entirely when disabled.
        data = {
            "semantic_config": _json_value(value.semantic_config),
            "arbitration_config": _json_value(value.arbitration_config),
            "stand_on_stages": _json_value(value.stand_on_stages),
            "multi_intent_resolution": _json_value(value.multi_intent_resolution),
            "rule13_liveness_policy": _json_value(value.rule13_liveness_policy),
        }
        if value.exact_search_config is not None:
            data["exact_search_config"] = _json_value(value.exact_search_config)
        return data
    if isinstance(value, IntentDrivenCandidatePolicy):
        # Backward-compatible schema extension: default-off Rule-13 envelope-ceiling
        # support must not perturb hashes of already-frozen configurations.  Emit the
        # new key only when explicitly enabled; old/default policy JSON stays byte-for-byte
        # identical while opt-in experiments receive a distinct canonical identity.
        data = {
            "template": _json_value(value.template),
            "rule13_progress_speed_fractions": _json_value(
                value.rule13_progress_speed_fractions
            ),
            "rule13_progress_course_offsets_from_counterpart_deg": _json_value(
                value.rule13_progress_course_offsets_from_counterpart_deg
            ),
        }
        if value.rule13_progress_include_speed_envelope_ceiling:
            data["rule13_progress_include_speed_envelope_ceiling"] = True
        return data
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise EnvironmentalNetworkExperimentError(
                    "experiment config mappings must use string keys"
                )
            result[key] = _json_value(value[key])
        return result
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise EnvironmentalNetworkExperimentError(
        f"experiment config contains unsupported value type {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkExperimentConfig:
    """One immutable, reproducibly identifiable A137 experiment definition."""

    network: RouteNetwork
    streams: tuple[PoissonNetworkStreamSpec, ...]
    scenario: EnvironmentalNetworkScenarioConfig
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
        if self.label is not None:
            label = self.label.strip()
            if not label:
                raise ValueError("label must be non-empty when supplied")
            object.__setattr__(self, "label", label)
        # Stream order is not semantically meaningful to the underlying generator, which
        # already canonicalizes by stream_id.  Freeze the facade the same way so callers
        # cannot accidentally create two run identities for the same experiment.
        object.__setattr__(self, "streams", tuple(sorted(streams, key=lambda item: item.stream_id)))

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _EXPERIMENT_FORMAT,
            "schema_version": _EXPERIMENT_SCHEMA_VERSION,
            "label": self.label,
            "network": _json_value(self.network),
            "streams": _json_value(self.streams),
            "scenario": _json_value(self.scenario),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def config_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()

    @property
    def run_id(self) -> str:
        material = _RUN_DOMAIN + __version__.encode("utf-8") + b"\x00" + self.config_sha256.encode("ascii")
        return sha256(material).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkExperimentSummary:
    """Compact deterministic outcome suitable for manifests and experiment tables."""

    run_id: str
    config_sha256: str
    software_version: str
    seed: int
    start_time: float
    traffic_horizon_min: float
    end_time: float
    realized_vessel_count: int
    realized_vessel_ids: tuple[str, ...]
    ingress_timing_policy: EnvironmentalNetworkIngressTimingPolicy
    physical_admission_count: int
    mean_ingress_waiting_time_min: float | None
    max_ingress_waiting_time_min: float | None
    final_time: float
    final_vessel_count: int
    final_vessel_ids: tuple[str, ...]
    final_snapshot_sha256: str
    fleet_observation_count: int
    pair_owner_observation_count: int
    joint_owner_observation_count: int
    nominal_route_owner_count: int
    rebootstrap_count: int
    route_disposition_count: int
    stop_reason: str


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkExperimentResult:
    """Product-level A138 result: compact summary plus complete A137 evidence."""

    config: EnvironmentalNetworkExperimentConfig
    summary: EnvironmentalNetworkExperimentSummary
    scenario: EnvironmentalNetworkScenarioResult

    @property
    def run_id(self) -> str:
        return self.summary.run_id

    @property
    def config_sha256(self) -> str:
        return self.summary.config_sha256

    def to_json_object(self) -> dict[str, Any]:
        """Return the frozen user-facing A141 result document.

        Internal runtime/session/controller objects deliberately stay outside this
        document.  The public result binds the canonical experiment definition, compact
        summary and explicit stochastic-demand/physical-admission timeline.
        """

        return {
            "format": _RESULT_FORMAT,
            "schema_version": _RESULT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "config": self.config.to_json_object(),
            "summary": _json_value(self.summary),
            "execution_environment": environment_manifest(),
            "traffic": _json_value(self.scenario.traffic),
            "ingress": [
                {
                    "stream_id": item.stream_id,
                    "vessel_id": item.vessel_id,
                    "demand_time": item.demand_time,
                    "admission_time": item.admission_time,
                    "waiting_time_min": item.waiting_time_min,
                }
                for item in self.scenario.ingress
            ],
            "final_state": _snapshot_payload(self.scenario),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def result_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()


def _snapshot_payload(result: EnvironmentalNetworkScenarioResult) -> dict[str, Any]:
    snapshot = result.run.snapshot
    commands = tuple(
        (vessel.vessel_id, result.world.command_for(vessel.vessel_id))
        for vessel in snapshot.vessels
    )
    return {
        "snapshot": _json_value(snapshot),
        "commands": _json_value(commands),
    }


def _snapshot_sha256(result: EnvironmentalNetworkScenarioResult) -> str:
    return sha256(_canonical_json(_snapshot_payload(result))).hexdigest()


def _summary(
    config: EnvironmentalNetworkExperimentConfig,
    result: EnvironmentalNetworkScenarioResult,
) -> EnvironmentalNetworkExperimentSummary:
    observations = result.runtime_start.session.coordinator.fleet_observations
    pair_observations = sum(bool(item.dispatch.pair_owners) for item in observations)
    joint_observations = sum(bool(item.dispatch.joint_owners) for item in observations)
    final_ids = tuple(vessel.vessel_id for vessel in result.run.snapshot.vessels)
    persistent = result.runtime_start.session.persistent
    waiting_times = tuple(item.waiting_time_min for item in result.ingress)
    return EnvironmentalNetworkExperimentSummary(
        run_id=config.run_id,
        config_sha256=config.config_sha256,
        software_version=__version__,
        seed=config.scenario.seed,
        start_time=config.scenario.start_time,
        traffic_horizon_min=config.scenario.traffic_horizon_min,
        end_time=float(config.scenario.end_time),
        realized_vessel_count=result.traffic.count,
        realized_vessel_ids=result.realized_vessel_ids,
        ingress_timing_policy=config.scenario.ingress_timing_policy,
        physical_admission_count=len(result.admissions),
        mean_ingress_waiting_time_min=(
            None if not waiting_times else sum(waiting_times) / len(waiting_times)
        ),
        max_ingress_waiting_time_min=None if not waiting_times else max(waiting_times),
        final_time=result.final_time,
        final_vessel_count=len(final_ids),
        final_vessel_ids=final_ids,
        final_snapshot_sha256=_snapshot_sha256(result),
        fleet_observation_count=len(observations),
        pair_owner_observation_count=pair_observations,
        joint_owner_observation_count=joint_observations,
        nominal_route_owner_count=len(result.nominal_route_owner_ids),
        rebootstrap_count=len(persistent.rebootstrap_results),
        route_disposition_count=len(persistent.disposition_results),
        stop_reason=result.run.stop_reason.value,
    )


def run_environmental_network_experiment(
    config: EnvironmentalNetworkExperimentConfig,
) -> EnvironmentalNetworkExperimentResult:
    """Execute exactly one immutable A138 experiment through the existing A137 path."""

    if not isinstance(config, EnvironmentalNetworkExperimentConfig):
        raise TypeError("config must be EnvironmentalNetworkExperimentConfig")
    scenario = run_environmental_poisson_network_scenario(
        config.network,
        config.streams,
        config=config.scenario,
    )
    return EnvironmentalNetworkExperimentResult(
        config=config,
        summary=_summary(config, scenario),
        scenario=scenario,
    )
