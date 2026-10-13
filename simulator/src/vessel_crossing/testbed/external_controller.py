"""C15 external-controller adapter boundary.

The core bridge serializes controller-visible evidence only, normalizes a controller
response, records simulated response latency/status and returns a requested maneuver.
It does not expose GroundTruth objects and it does not apply commands to the plant.
Actual IPC/network transports remain external adapters implementing the transport
protocol below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Any, Mapping, Protocol

from .behavior import RequestedManeuver
from .environment_belief import EnvironmentalBeliefSnapshot
from .observation import ObservedWorldSnapshot
from .scope import TestbedFeature
from .world_objects import CircleGeometry, PointGeometry, PolygonGeometry, PolylineGeometry


class ExternalControllerStatus(str, Enum):
    OK = "OK"
    TIMEOUT = "TIMEOUT"
    NO_RESPONSE = "NO_RESPONSE"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    CONTROLLER_ERROR = "CONTROLLER_ERROR"


@dataclass(frozen=True, slots=True)
class ExternalControllerSpec:
    controller_id: str
    controller_version: str
    adapter_id: str
    request_schema_version: str = "cross-c15-request/v1"
    response_schema_version: str = "cross-c15-response/v1"
    max_response_latency_min: float = 0.1

    def __post_init__(self) -> None:
        for name in ("controller_id", "controller_version", "adapter_id", "request_schema_version", "response_schema_version"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if not isfinite(self.max_response_latency_min) or self.max_response_latency_min < 0.0:
            raise ValueError("max_response_latency_min must be finite and non-negative")

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "controller_id": self.controller_id,
            "controller_version": self.controller_version,
            "adapter_id": self.adapter_id,
            "request_schema_version": self.request_schema_version,
            "response_schema_version": self.response_schema_version,
            "max_response_latency_min": self.max_response_latency_min,
        })


@dataclass(frozen=True, slots=True)
class ExternalControllerRequest:
    request_id: str
    controller_spec_sha256: str
    request_schema_version: str
    observation_time_min: float
    ego_vessel_id: str
    payload_json: str
    component_id: TestbedFeature = TestbedFeature.C15_EXTERNAL_CONTROLLER

    @property
    def payload(self) -> Mapping[str, Any]:
        return json.loads(self.payload_json)

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "request_id": self.request_id,
            "controller_spec_sha256": self.controller_spec_sha256,
            "request_schema_version": self.request_schema_version,
            "observation_time_min": self.observation_time_min,
            "ego_vessel_id": self.ego_vessel_id,
            "payload": json.loads(self.payload_json),
        })


@dataclass(frozen=True, slots=True)
class ExternalTransportOutcome:
    """Normalized transport result supplied by an IPC/network/callback adapter.

    `response_payload` is the controller's JSON-like response body.  `latency_min` is
    simulation/protocol latency evidence, not wall-clock timing measured by this core.
    """

    response_payload: Mapping[str, Any] | None
    latency_min: float
    error: str = ""

    def __post_init__(self) -> None:
        if not isfinite(self.latency_min) or self.latency_min < 0.0:
            raise ValueError("latency_min must be finite and non-negative")


class ExternalControllerTransport(Protocol):
    transport_id: str

    def exchange(self, request: ExternalControllerRequest) -> ExternalTransportOutcome:
        ...


@dataclass(frozen=True, slots=True)
class ExternalControllerResult:
    request_sha256: str
    controller_spec_sha256: str
    transport_id: str
    status: ExternalControllerStatus
    response_latency_min: float
    maneuver: RequestedManeuver | None
    response_payload_sha256: str | None
    reason: str = ""
    component_id: TestbedFeature = TestbedFeature.C15_EXTERNAL_CONTROLLER

    def __post_init__(self) -> None:
        if not self.request_sha256 or not self.controller_spec_sha256 or not self.transport_id:
            raise ValueError("external controller result identity fields must be non-empty")
        if not isfinite(self.response_latency_min) or self.response_latency_min < 0.0:
            raise ValueError("response_latency_min must be finite and non-negative")
        if self.status is ExternalControllerStatus.OK and self.maneuver is None:
            raise ValueError("OK external controller result requires a maneuver")
        if self.status is not ExternalControllerStatus.OK and self.maneuver is not None:
            raise ValueError("non-OK external controller result must not carry a maneuver")

    @property
    def usable(self) -> bool:
        return self.status is ExternalControllerStatus.OK and self.maneuver is not None

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "request_sha256": self.request_sha256,
            "controller_spec_sha256": self.controller_spec_sha256,
            "transport_id": self.transport_id,
            "status": self.status.value,
            "response_latency_min": self.response_latency_min,
            "maneuver": None if self.maneuver is None else {
                "requested_at_min": self.maneuver.requested_at_min,
                "target_heading_deg": self.maneuver.target_heading_deg,
                "target_speed_knots": self.maneuver.target_speed_knots,
                "source": self.maneuver.source,
            },
            "response_payload_sha256": self.response_payload_sha256,
            "reason": self.reason,
        })


class ExternalControllerBridge:
    component_id = TestbedFeature.C15_EXTERNAL_CONTROLLER

    def __init__(self, *, spec: ExternalControllerSpec, transport: ExternalControllerTransport) -> None:
        if not getattr(transport, "transport_id", ""):
            raise ValueError("transport_id must be non-empty")
        self.spec = spec
        self.transport = transport

    def build_request(
        self,
        *,
        observed_world: ObservedWorldSnapshot,
        ego_vessel_id: str,
        environment_belief: EnvironmentalBeliefSnapshot | None = None,
        mission_context: Mapping[str, Any] | None = None,
    ) -> ExternalControllerRequest:
        # Fail before serialization if the ego is not actually controller-visible.
        observed_world.by_vessel_id(ego_vessel_id)
        payload = {
            "schema_version": self.spec.request_schema_version,
            "observation_time_min": observed_world.observation_time_min,
            "ego_vessel_id": ego_vessel_id,
            "observed_world": _observed_world_payload(observed_world),
            "environment_belief": _environment_belief_payload(environment_belief),
            "mission_context": _canonical_json_value(dict(mission_context or {})),
        }
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        request_id = _sha256_payload({
            "controller_spec_sha256": self.spec.identity_sha256(),
            "payload": json.loads(payload_json),
        })
        return ExternalControllerRequest(
            request_id=request_id,
            controller_spec_sha256=self.spec.identity_sha256(),
            request_schema_version=self.spec.request_schema_version,
            observation_time_min=observed_world.observation_time_min,
            ego_vessel_id=ego_vessel_id,
            payload_json=payload_json,
        )

    def invoke(self, request: ExternalControllerRequest) -> ExternalControllerResult:
        if request.controller_spec_sha256 != self.spec.identity_sha256():
            raise ValueError("request does not bind this external controller spec")
        try:
            outcome = self.transport.exchange(request)
        except Exception as exc:  # transport boundary: normalize but do not hide the class/message
            return _result(
                request=request,
                spec=self.spec,
                transport_id=self.transport.transport_id,
                status=ExternalControllerStatus.CONTROLLER_ERROR,
                latency=0.0,
                reason=f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(outcome, ExternalTransportOutcome):
            return _result(
                request=request, spec=self.spec, transport_id=self.transport.transport_id,
                status=ExternalControllerStatus.INVALID_RESPONSE, latency=0.0,
                reason="transport did not return ExternalTransportOutcome",
            )
        if outcome.latency_min > self.spec.max_response_latency_min:
            return _result(
                request=request, spec=self.spec, transport_id=self.transport.transport_id,
                status=ExternalControllerStatus.TIMEOUT, latency=outcome.latency_min,
                response_payload=outcome.response_payload,
                reason="response latency exceeded configured limit",
            )
        if outcome.response_payload is None:
            status = ExternalControllerStatus.CONTROLLER_ERROR if outcome.error else ExternalControllerStatus.NO_RESPONSE
            return _result(
                request=request, spec=self.spec, transport_id=self.transport.transport_id,
                status=status, latency=outcome.latency_min, reason=outcome.error or "no response payload",
            )
        try:
            maneuver = _parse_response(self.spec, outcome.response_payload, requested_at_min=request.observation_time_min + outcome.latency_min)
        except (TypeError, ValueError, KeyError) as exc:
            return _result(
                request=request, spec=self.spec, transport_id=self.transport.transport_id,
                status=ExternalControllerStatus.INVALID_RESPONSE, latency=outcome.latency_min,
                response_payload=outcome.response_payload,
                reason=f"{type(exc).__name__}: {exc}",
            )
        return _result(
            request=request, spec=self.spec, transport_id=self.transport.transport_id,
            status=ExternalControllerStatus.OK, latency=outcome.latency_min,
            response_payload=outcome.response_payload, maneuver=maneuver,
        )


class NoOpExternalControllerBridge:
    """Compatibility surface: constructs no request and produces no external action."""

    component_id = TestbedFeature.C15_EXTERNAL_CONTROLLER

    def build_request(self, **_: Any) -> None:
        return None

    def invoke(self, request: ExternalControllerRequest) -> None:
        return None


def _parse_response(spec: ExternalControllerSpec, payload: Mapping[str, Any], *, requested_at_min: float) -> RequestedManeuver:
    canonical = _canonical_json_value(dict(payload))
    if canonical.get("schema_version") != spec.response_schema_version:
        raise ValueError("response schema_version mismatch")
    if canonical.get("controller_id") != spec.controller_id:
        raise ValueError("response controller_id mismatch")
    heading = canonical["target_heading_deg"]
    speed = canonical["target_speed_knots"]
    if isinstance(heading, bool) or not isinstance(heading, (int, float)) or not isfinite(float(heading)):
        raise TypeError("target_heading_deg must be a finite number")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not isfinite(float(speed)):
        raise TypeError("target_speed_knots must be a finite number")
    heading_f = float(heading)
    speed_f = float(speed)
    if not 0.0 <= heading_f < 360.0:
        raise ValueError("target_heading_deg must be in [0, 360)")
    if speed_f < 0.0:
        raise ValueError("target_speed_knots must be non-negative")
    return RequestedManeuver(requested_at_min=requested_at_min, target_heading_deg=heading_f, target_speed_knots=speed_f, source=f"external:{spec.controller_id}")


def _observed_world_payload(snapshot: ObservedWorldSnapshot) -> Mapping[str, Any]:
    # Deliberately exclude source_truth_sha256: the external controller receives only
    # controller-visible state, not a truth identity side channel.
    return {
        "provider_id": snapshot.provider_id,
        "observation_time_min": snapshot.observation_time_min,
        "vessels": [
            {
                "vessel_id": v.vessel_id,
                "x_east_nm": v.x_east_nm,
                "y_north_nm": v.y_north_nm,
                "heading_deg": v.heading_deg,
                "course_over_ground_deg": v.course_over_ground_deg,
                "speed_over_ground_knots": v.speed_over_ground_knots,
                "source_time_min": v.source_time_min,
                "track_status": v.track_status.value,
                "observation_source": v.observation_source.value,
            }
            for v in snapshot.vessels
        ],
        "objects": [
            {
                "object_id": obj.object_id,
                "kind": obj.kind.value,
                "geometry": _geometry_payload(obj.geometry),
                "source_time_min": obj.source_time_min,
                "track_status": obj.track_status.value,
                "observation_source": obj.observation_source.value,
            }
            for obj in snapshot.objects
        ],
    }


def _environment_belief_payload(snapshot: EnvironmentalBeliefSnapshot | None) -> Mapping[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "provider_id": snapshot.provider_id,
        "observation_time_min": snapshot.observation_time_min,
        "source_time_min": snapshot.source_time_min,
        "x_east_nm": snapshot.x_east_nm,
        "y_north_nm": snapshot.y_north_nm,
        "estimated_current_velocity_nm_per_min": list(snapshot.estimated_current_velocity_nm_per_min),
        "estimated_wind_velocity_nm_per_min": list(snapshot.estimated_wind_velocity_nm_per_min),
        "uncertainty": {
            "current": {
                "east_abs_nm_per_min": snapshot.uncertainty.current.east_abs_nm_per_min,
                "north_abs_nm_per_min": snapshot.uncertainty.current.north_abs_nm_per_min,
            },
            "wind": {
                "east_abs_nm_per_min": snapshot.uncertainty.wind.east_abs_nm_per_min,
                "north_abs_nm_per_min": snapshot.uncertainty.wind.north_abs_nm_per_min,
            },
        },
    }


def _geometry_payload(geometry: Any) -> Mapping[str, Any]:
    if isinstance(geometry, PointGeometry):
        return {"type": "POINT", "point": [geometry.point.x_east_nm, geometry.point.y_north_nm]}
    if isinstance(geometry, CircleGeometry):
        return {
            "type": "CIRCLE", "center": [geometry.center.x_east_nm, geometry.center.y_north_nm],
            "radius_nm": geometry.radius_nm,
        }
    if isinstance(geometry, PolylineGeometry):
        return {"type": "POLYLINE", "points": [[p.x_east_nm, p.y_north_nm] for p in geometry.points]}
    if isinstance(geometry, PolygonGeometry):
        return {"type": "POLYGON", "vertices": [[p.x_east_nm, p.y_north_nm] for p in geometry.vertices]}
    raise TypeError(f"unsupported observed geometry type: {type(geometry).__name__}")


def _result(
    *,
    request: ExternalControllerRequest,
    spec: ExternalControllerSpec,
    transport_id: str,
    status: ExternalControllerStatus,
    latency: float,
    response_payload: Mapping[str, Any] | None = None,
    maneuver: RequestedManeuver | None = None,
    reason: str = "",
) -> ExternalControllerResult:
    response_sha = None if response_payload is None else _sha256_payload(_canonical_json_value(dict(response_payload)))
    return ExternalControllerResult(
        request_sha256=request.identity_sha256(),
        controller_spec_sha256=spec.identity_sha256(),
        transport_id=transport_id,
        status=status,
        response_latency_min=latency,
        maneuver=maneuver,
        response_payload_sha256=response_sha,
        reason=reason,
    )


def _canonical_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("external-controller payload cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("external-controller payload mapping keys must be strings")
            result[key] = _canonical_json_value(item)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError(f"unsupported external-controller payload type: {type(value).__name__}")


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(_canonical_json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
