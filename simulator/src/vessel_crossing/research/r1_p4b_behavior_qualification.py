"""Non-scientific R1 P4B behavior-adapter qualification harness.

This module exists to prove the B1 adapter wiring before any P4B publication campaign
is frozen. M0-05 uses the public identity-bearing pre-realized FIFO/world-builder seam,
so the traffic realization is fixed before behavior bindings are installed without a
research dependency on the historical private telemetry hook.

Nothing here authorizes a numeric human-response delay. Qualification cases may use a
diagnostic delay solely to prove that the adapter path is behaviorally active; those
values are not publication parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Mapping

from vessel_crossing.nextgen.environmental_candidate_transition import EnvironmentalMotionBackend
from vessel_crossing.nextgen.environmental_network_fifo_experiment import (
    EnvironmentalNetworkFIFOExperimentConfig,
    EnvironmentalNetworkFIFOExperimentResult,
    run_environmental_network_fifo_experiment_with_traffic,
)
from vessel_crossing.nextgen.environmental_world_builder import (
    EnvironmentalMotionBackendWorldBuilder,
    environmental_world_builder_configuration_sha256,
)
from vessel_crossing.nextgen.network_traffic import PoissonNetworkTrafficRealization
from vessel_crossing.testbed.behavior import BehaviorProfile

from .r1_p4b_publication_reference import (
    AS,
    HS,
    ExogenousDemandEvent,
    TypedDemandEvent,
    assign_nested_types,
)
from .r1_p4b_behavior_adapter import (
    P4BBehaviorBinding,
    P4BBehaviorEnvironmentalMotionBackend,
)

_FORMAT = "cross-r1-p4b-behavior-adapter-qualification"
_SCHEMA_VERSION = 1
_EVIDENCE_ROLE = "NON_SCIENTIFIC_ADAPTER_QUALIFICATION_DO_NOT_INTERPRET_AS_P4B_RESULT"


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def traffic_input_identity(traffic: PoissonNetworkTrafficRealization) -> dict[str, object]:
    if not isinstance(traffic, PoissonNetworkTrafficRealization):
        raise TypeError("traffic must be PoissonNetworkTrafficRealization")
    return {
        "seed": traffic.seed,
        "start_time": traffic.start_time,
        "horizon": traffic.horizon,
        "streams": [
            {
                "stream_id": stream.stream_id,
                "arrival_times": list(stream.arrival_times),
                "vessel_ids": list(stream.vessel_ids),
                "vessel_class_id": stream.vessel_class_id,
            }
            for stream in traffic.streams
        ],
    }


def traffic_input_sha256(traffic: PoissonNetworkTrafficRealization) -> str:
    return sha256(_canonical_json(traffic_input_identity(traffic))).hexdigest()


def endpoint_bindings(
    traffic: PoissonNetworkTrafficRealization,
    *,
    profile: BehaviorProfile,
    adapter_id: str,
    command_delay_min: float,
) -> dict[str, P4BBehaviorBinding]:
    template = P4BBehaviorBinding(profile, adapter_id, command_delay_min)
    return {arrival.vessel.vessel_id: template for arrival in traffic.arrivals}


def bindings_for_type_map(
    traffic: PoissonNetworkTrafficRealization,
    type_by_vessel: Mapping[str, str],
    *,
    as_binding: P4BBehaviorBinding,
    hs_binding: P4BBehaviorBinding,
) -> dict[str, P4BBehaviorBinding]:
    expected = {arrival.vessel.vessel_id for arrival in traffic.arrivals}
    supplied = set(type_by_vessel)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"type map must match traffic vessel ids exactly; missing={missing}, extra={extra}")
    out: dict[str, P4BBehaviorBinding] = {}
    for vessel_id in sorted(expected):
        label = type_by_vessel[vessel_id]
        if label == AS:
            out[vessel_id] = as_binding
        elif label == HS:
            out[vessel_id] = hs_binding
        else:
            raise ValueError(f"unsupported P4B vessel type {label!r} for {vessel_id!r}")
    return out


def typed_demand_for_penetration(
    traffic: PoissonNetworkTrafficRealization,
    *,
    penetration: float,
    campaign_id: str,
    replication: int,
) -> tuple[TypedDemandEvent, ...]:
    """Bind pre-realized demand events to nested AS/HS types without changing timestamps."""
    stream_by_vessel: dict[str, str] = {}
    for stream in traffic.streams:
        for vessel_id in stream.vessel_ids:
            if vessel_id in stream_by_vessel:
                raise ValueError(f"duplicate traffic vessel id {vessel_id!r}")
            stream_by_vessel[vessel_id] = stream.stream_id
    events = tuple(
        ExogenousDemandEvent(
            event_id=arrival.vessel.vessel_id,
            arrival_min=arrival.at,
            stream_id=stream_by_vessel[arrival.vessel.vessel_id],
        )
        for arrival in traffic.arrivals
    )
    typed = assign_nested_types(
        events, penetration=penetration, campaign_id=campaign_id, replication=replication
    )
    original = {event.event_id: (event.arrival_min, event.stream_id) for event in events}
    for event in typed:
        if (event.arrival_min, event.stream_id) != original[event.event_id]:
            raise RuntimeError("type assignment changed exogenous demand timing or stream identity")
    return typed


def bindings_for_penetration(
    traffic: PoissonNetworkTrafficRealization,
    *,
    penetration: float,
    campaign_id: str,
    replication: int,
    as_binding: P4BBehaviorBinding,
    hs_binding: P4BBehaviorBinding,
) -> tuple[tuple[TypedDemandEvent, ...], dict[str, P4BBehaviorBinding]]:
    typed = typed_demand_for_penetration(
        traffic, penetration=penetration, campaign_id=campaign_id, replication=replication
    )
    type_map = {event.event_id: event.vessel_type for event in typed}
    bindings = bindings_for_type_map(
        traffic, type_map, as_binding=as_binding, hs_binding=hs_binding
    )
    return typed, bindings


@dataclass(frozen=True, slots=True)
class P4BBehaviorQualificationResult:
    inner_result: EnvironmentalNetworkFIFOExperimentResult
    behavior_backend_id: str
    traffic_sha256: str
    behavior_manifest: tuple[dict[str, object], ...]
    canonical_bytes: bytes

    @property
    def qualification_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes

    def to_json_object(self) -> dict[str, object]:
        value = json.loads(self.canonical_bytes.decode("utf-8"))
        assert isinstance(value, dict)
        return value


def _validate_binding_coverage(
    traffic: PoissonNetworkTrafficRealization,
    bindings: Mapping[str, P4BBehaviorBinding],
) -> None:
    expected = {arrival.vessel.vessel_id for arrival in traffic.arrivals}
    supplied = set(bindings)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"behavior bindings must match traffic vessel ids exactly; missing={missing}, extra={extra}")


def run_behavior_adapter_qualification_case(
    config: EnvironmentalNetworkFIFOExperimentConfig,
    traffic: PoissonNetworkTrafficRealization,
    *,
    base_backend_factory: Callable[[], EnvironmentalMotionBackend],
    bindings: Mapping[str, P4BBehaviorBinding],
) -> P4BBehaviorQualificationResult:
    if not isinstance(config, EnvironmentalNetworkFIFOExperimentConfig):
        raise TypeError("config must be EnvironmentalNetworkFIFOExperimentConfig")
    if not callable(base_backend_factory):
        raise TypeError("base_backend_factory must be callable")
    frozen_bindings = dict(bindings)
    _validate_binding_coverage(traffic, frozen_bindings)
    traffic_before = traffic_input_sha256(traffic)
    wrapped = P4BBehaviorEnvironmentalMotionBackend(
        base_backend_factory(), frozen_bindings
    )
    builder_configuration_sha256 = environmental_world_builder_configuration_sha256(
        {
            "role": "R1_P4B_BEHAVIOR_QUALIFICATION",
            "behavior_backend_id": wrapped.backend_id,
            "traffic_input_sha256": traffic_before,
        }
    )
    world_builder = EnvironmentalMotionBackendWorldBuilder(
        builder_id="R1_P4B_BEHAVIOR_QUALIFICATION_WORLD/V1",
        configuration_sha256=builder_configuration_sha256,
        motion_backend=wrapped,
    )

    inner = run_environmental_network_fifo_experiment_with_traffic(
        config, traffic, world_builder=world_builder
    )
    traffic_after = traffic_input_sha256(traffic)
    if traffic_after != traffic_before:
        raise RuntimeError("behavior execution mutated the pre-realized traffic input")
    backend = wrapped
    manifest = tuple(
        {
            "vessel_id": vessel_id,
            "profile": binding.profile.value,
            "adapter_id": binding.adapter_id,
            "command_delay_min": binding.command_delay_min,
            "binding_sha256": binding.identity_sha256,
        }
        for vessel_id, binding in sorted(frozen_bindings.items())
    )
    document = {
        "format": _FORMAT,
        "schema_version": _SCHEMA_VERSION,
        "evidence_role": _EVIDENCE_ROLE,
        "scientific_interpretation_authorized": False,
        "config_sha256": config.config_sha256,
        "traffic_input_sha256": traffic_before,
        "inner_result_sha256": inner.result_sha256,
        "inner_final_snapshot_sha256": inner.summary.final_snapshot_sha256,
        "behavior_backend_id": backend.backend_id,
        "behavior_manifest": list(manifest),
    }
    return P4BBehaviorQualificationResult(
        inner_result=inner,
        behavior_backend_id=backend.backend_id,
        traffic_sha256=traffic_before,
        behavior_manifest=manifest,
        canonical_bytes=_canonical_json(document),
    )
