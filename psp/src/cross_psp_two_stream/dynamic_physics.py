"""Normalized custom-physics contract for RC2 dynamic references.

This module owns domain validation and deterministic identity construction only.  It does
not execute B4/B5/B6 and does not reinterpret canonical D0 safety semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import hypot, isfinite
from types import MappingProxyType
from typing import Any, Mapping

from vessel_crossing.nextgen import VesselClassProfile

from .b4 import B4_POLICY_ALGORITHM_ID
from .b5 import B5_POLICY_ALGORITHM_ID
from .b6 import B6_REFERENCE_ALGORITHM_ID, B6RuntimeBinding
from .contract import TwoStreamProfileContract, profile_contract_from_mapping
from .reference_arbiter import REFERENCE_ARBITER_ALGORITHM_ID
from .solver import SOLVER_ALGORITHM_ID

PHYSICS_IDENTITY_SCHEMA = "cross-psp-dynamic-physics-identity-v1"
QUALIFIED_PROFILE = "QUALIFIED_PROFILE"
SUPPORTED_EXPLORATORY = "SUPPORTED_EXPLORATORY"
UNSUPPORTED = "UNSUPPORTED"
_EPS = 1e-9


class DynamicPhysicsError(ValueError):
    """Raised when a requested physical scenario leaves the RC2 v1 domain."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _heading_delta(a: float, b: float) -> float:
    raw = abs((float(a) - float(b)) % 360.0)
    return min(raw, 360.0 - raw)


def _domain_dict(profile: VesselClassProfile) -> dict[str, float]:
    domain = profile.safety_domain
    if domain is None:
        raise DynamicPhysicsError("explicit safety domain required for every custom-physics stream")
    return {
        "ahead_nm": float(domain.ahead_nm),
        "astern_nm": float(domain.astern_nm),
        "port_nm": float(domain.port_nm),
        "starboard_nm": float(domain.starboard_nm),
    }


def _profile_physics(profile: VesselClassProfile) -> dict[str, object]:
    maximum_speed = None
    if profile.speed_envelope is not None:
        maximum_speed = float(profile.speed_envelope.maximum_speed)
    return {
        "dimensions": {
            "length_nm": float(profile.dimensions.length_nm),
            "beam_nm": float(profile.dimensions.beam_nm),
        },
        "maneuver_limits": {
            "max_acceleration": float(profile.maneuver_limits.max_acceleration),
            "max_deceleration": float(profile.maneuver_limits.max_deceleration),
            "max_turn_rate_deg_per_min": float(profile.maneuver_limits.max_turn_rate_deg_per_min),
        },
        "safety_domain": _domain_dict(profile),
        "maximum_speed_nm_per_min": maximum_speed,
    }


def _route_geometry(config: object, stream: object) -> tuple[float, float, float]:
    route = config.network.resolve(stream.request).route
    if route.segment_count != 1:
        raise DynamicPhysicsError("RC2 v1 requires exactly one straight route segment per stream")
    start, end = route.waypoints
    source_distance = hypot(float(start[0]), float(start[1]))
    exit_distance = hypot(float(end[0]), float(end[1]))
    if source_distance <= _EPS or exit_distance <= _EPS:
        raise DynamicPhysicsError("route source and exit must both be away from the conflict center")
    cross = float(start[0]) * float(end[1]) - float(start[1]) * float(end[0])
    if abs(cross) > _EPS * max(1.0, source_distance * exit_distance):
        raise DynamicPhysicsError("RC2 v1 straight route must pass through the conflict center at local origin")
    dot = float(start[0]) * float(end[0]) + float(start[1]) * float(end[1])
    if dot >= -_EPS:
        raise DynamicPhysicsError("route source and exit must lie on opposite sides of the conflict center")
    return source_distance, exit_distance, float(route.segment_heading(0))


def target_shares_from_metadata(
    metadata: Mapping[str, Any],
    stream_ids: tuple[str, str],
    defaults: Mapping[str, float],
) -> dict[str, float]:
    raw = metadata.get("b6_target_stream_shares")
    if raw is None:
        return {stream_id: float(defaults[stream_id]) for stream_id in stream_ids}
    if not isinstance(raw, Mapping) or set(raw) != set(stream_ids):
        raise DynamicPhysicsError(
            "metadata.b6_target_stream_shares must map exactly the two stream IDs"
        )
    shares = {stream_id: float(raw[stream_id]) for stream_id in stream_ids}
    _validate_target_shares(shares)
    return shares


def _validate_target_shares(shares: Mapping[str, float]) -> None:
    if any(not isfinite(value) or value <= 0.0 for value in shares.values()):
        raise DynamicPhysicsError("B6 target stream shares must be finite and positive")
    if abs(sum(shares.values()) - 1.0) > 1e-12:
        raise DynamicPhysicsError("B6 target stream shares must sum to 1")


def _validate_zero_drift(config: object) -> None:
    default_drift = tuple(float(v) for v in config.scenario.default_environmental_drift_velocity)
    stream_drifts = tuple(
        tuple(float(v) for v in value)
        for value in dict(config.scenario.stream_environmental_drift_velocity).values()
    )
    if any(abs(v) > _EPS for v in default_drift) or any(
        any(abs(v) > _EPS for v in drift) for drift in stream_drifts
    ):
        raise DynamicPhysicsError("RC2 v1 dynamic B6 requires exactly zero environmental drift")


@dataclass(frozen=True, slots=True)
class _NormalizedStream:
    stream_id: str
    class_id: str
    vessel_profile: VesselClassProfile
    heading_deg: float
    geometry: Mapping[str, object]
    supply_template: Mapping[str, object]
    identity: Mapping[str, object]


def _normalize_stream(config: object, stream: object) -> _NormalizedStream:
    source_distance, exit_distance, heading = _route_geometry(config, stream)
    vessel_profile = stream.vessel_class
    if vessel_profile is None:
        raise DynamicPhysicsError(f"stream {stream.stream_id!r} requires a vessel class")
    physics = _profile_physics(vessel_profile)
    speed = float(stream.cruise_speed)
    if not isfinite(speed) or speed <= 0.0:
        raise DynamicPhysicsError(f"stream {stream.stream_id!r} requires a finite positive cruise speed")
    maximum_speed = physics["maximum_speed_nm_per_min"]
    if maximum_speed is not None and speed > float(maximum_speed) + _EPS:
        raise DynamicPhysicsError(
            f"stream {stream.stream_id!r} cruise speed exceeds its maximum-speed envelope"
        )
    class_id = str(vessel_profile.class_id)
    geometry = {
        "stream_id": stream.stream_id,
        "source_to_center_distance_nm": source_distance,
        "center_to_exit_distance_nm": exit_distance,
    }
    supply_template = {
        "stream_id": stream.stream_id,
        "cycle": [
            {
                "profile_class_id": class_id,
                "through_water_heading_deg": heading,
                "through_water_speed_nm_per_min": speed,
            }
        ],
    }
    identity = {
        **geometry,
        "heading_deg": heading,
        "nominal_speed_nm_per_min": speed,
        "vessel_class_id": class_id,
        "vessel_class_physics": physics,
        "environmental_drift_nm_per_min": [0.0, 0.0],
    }
    return _NormalizedStream(
        stream_id=str(stream.stream_id),
        class_id=class_id,
        vessel_profile=vessel_profile,
        heading_deg=heading,
        geometry=geometry,
        supply_template=supply_template,
        identity=identity,
    )


def _collect_streams(config: object, base_profile: TwoStreamProfileContract) -> tuple[_NormalizedStream, _NormalizedStream]:
    streams = tuple(config.streams)
    expected_ids = tuple(item.stream_id for item in base_profile.streams)
    if len(streams) != 2 or tuple(stream.stream_id for stream in streams) != expected_ids:
        raise DynamicPhysicsError(
            "RC2 v1 requires exactly the canonical two stream IDs in canonical order"
        )
    rows = tuple(_normalize_stream(config, stream) for stream in streams)
    if abs(_heading_delta(rows[0].heading_deg, rows[1].heading_deg) - 90.0) > _EPS:
        raise DynamicPhysicsError("RC2 v1 requires an exactly orthogonal 90-degree crossing")
    return rows  # type: ignore[return-value]


def _profile_class_catalog(rows: tuple[_NormalizedStream, _NormalizedStream]) -> dict[str, VesselClassProfile]:
    catalog: dict[str, VesselClassProfile] = {}
    for row in rows:
        previous = catalog.get(row.class_id)
        if previous is not None and _profile_physics(previous) != _profile_physics(row.vessel_profile):
            raise DynamicPhysicsError(f"vessel class ID {row.class_id!r} maps to inconsistent physics")
        catalog[row.class_id] = row.vessel_profile
    return catalog


def _dynamic_profile(
    base_profile: TwoStreamProfileContract,
    rows: tuple[_NormalizedStream, _NormalizedStream],
    shares: Mapping[str, float],
) -> TwoStreamProfileContract:
    payload = base_profile.canonical_dict()
    payload.pop("component_contract_ids", None)
    payload["streams"] = [dict(row.geometry) for row in rows]
    payload["b6_reference"]["stream_shares"] = [
        {"stream_id": row.stream_id, "share": float(shares[row.stream_id])} for row in rows
    ]
    payload["b6_reference"]["supply_templates"] = [dict(row.supply_template) for row in rows]
    return profile_contract_from_mapping(payload)


@dataclass(frozen=True, slots=True)
class NormalizedPhysicalScenario:
    profile: TwoStreamProfileContract
    profile_classes: Mapping[str, VesselClassProfile]
    identity_material: Mapping[str, object]
    physics_identity_sha256: str
    scope_status: str
    frozen_a70_reference_reuse_eligible: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_classes", MappingProxyType(dict(self.profile_classes)))
        object.__setattr__(self, "identity_material", MappingProxyType(dict(self.identity_material)))


def build_normalized_physical_scenario(
    config: object,
    base_profile: TwoStreamProfileContract,
    reference_runtime: B6RuntimeBinding,
    *,
    target_shares: Mapping[str, float],
    qualified_physics_identity_sha256: str | None = None,
    frozen_a70_physics_identity_sha256: str | None = None,
) -> NormalizedPhysicalScenario:
    _validate_zero_drift(config)
    rows = _collect_streams(config, base_profile)
    shares = {row.stream_id: float(target_shares[row.stream_id]) for row in rows}
    _validate_target_shares(shares)
    profile = _dynamic_profile(base_profile, rows, shares)
    profile_classes = _profile_class_catalog(rows)
    identity_material = {
        "schema": PHYSICS_IDENTITY_SCHEMA,
        "fixed_domain": {
            "route_model_id": base_profile.route_model_id,
            "environment_model_id": base_profile.environment_model_id,
            "planned_slot_prediction_scope_id": base_profile.planned_slot_prediction_scope_id,
            "crossing_angle_deg": 90.0,
            "stream_count": 2,
            "conflict_center_count": 1,
        },
        "streams": [dict(row.identity) for row in rows],
        "b6_target_stream_shares": shares,
        "reference_runtime": {
            "binding_id": reference_runtime.binding_id,
            "binding_sha256": reference_runtime.identity_sha256,
        },
        "mechanisms": {
            "solver": SOLVER_ALGORITHM_ID,
            "b4": B4_POLICY_ALGORITHM_ID,
            "b5": B5_POLICY_ALGORITHM_ID,
            "reference_arbiter": REFERENCE_ARBITER_ALGORITHM_ID,
            "b6": B6_REFERENCE_ALGORITHM_ID,
        },
        "profile_contract_sha256": profile.contract_sha256,
    }
    physics_sha = _sha(identity_material)
    scope_status = (
        QUALIFIED_PROFILE
        if qualified_physics_identity_sha256 is not None
        and physics_sha == qualified_physics_identity_sha256
        else SUPPORTED_EXPLORATORY
    )
    reuse = (
        frozen_a70_physics_identity_sha256 is not None
        and physics_sha == frozen_a70_physics_identity_sha256
    )
    return NormalizedPhysicalScenario(
        profile=profile,
        profile_classes=profile_classes,
        identity_material=identity_material,
        physics_identity_sha256=physics_sha,
        scope_status=scope_status,
        frozen_a70_reference_reuse_eligible=reuse,
    )
