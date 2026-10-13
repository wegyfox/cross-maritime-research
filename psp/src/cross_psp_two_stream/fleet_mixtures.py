"""Coherent deterministic fleet-mixture sampling for Custom Physics v2.

A sampled member always resolves to one complete named class profile.  This prevents
independent randomization from combining dimensions, speeds, maneuver limits, and
safety-domain values that were never declared as one coherent class.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from hashlib import sha256
import json
from math import isfinite
from random import Random
from typing import Any, Mapping

FLEET_MIXTURE_SCHEMA = "cross-psp-fleet-mixture-v1"


class FleetMixtureError(ValueError):
    """Raised when a fleet-mixture declaration is incomplete or inconsistent."""


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


def _non_empty(name: str, value: object) -> str:
    out = str(value).strip()
    if not out:
        raise FleetMixtureError(f"{name} must be non-empty")
    return out


def _positive(name: str, value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise FleetMixtureError(f"{name} must be finite and positive") from exc
    if not isfinite(out) or out <= 0.0:
        raise FleetMixtureError(f"{name} must be finite and positive")
    return out


@dataclass(frozen=True, slots=True)
class VesselClassSpec:
    class_id: str
    length_m: float
    beam_m: float
    cruise_speed_knots: float
    maximum_speed_knots: float
    max_acceleration_knots_per_min: float
    max_deceleration_knots_per_min: float
    max_turn_rate_deg_per_min: float
    safety_ahead_nm: float
    safety_astern_nm: float
    safety_port_nm: float
    safety_starboard_nm: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_id", _non_empty("class_id", self.class_id))
        names = (
            "length_m", "beam_m", "cruise_speed_knots", "maximum_speed_knots",
            "max_acceleration_knots_per_min", "max_deceleration_knots_per_min",
            "max_turn_rate_deg_per_min", "safety_ahead_nm", "safety_astern_nm",
            "safety_port_nm", "safety_starboard_nm",
        )
        for name in names:
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        ratio = self.length_m / self.beam_m
        if ratio <= 1.0 or ratio > 20.0:
            raise FleetMixtureError("length/beam ratio must be > 1 and <= 20")
        if self.cruise_speed_knots > self.maximum_speed_knots:
            raise FleetMixtureError("cruise_speed_knots must not exceed maximum_speed_knots")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "dimensions_m": {"length_m": self.length_m, "beam_m": self.beam_m},
            "speed": {
                "cruise_speed_knots": self.cruise_speed_knots,
                "maximum_speed_knots": self.maximum_speed_knots,
            },
            "maneuver_limits": {
                "max_acceleration_knots_per_min": self.max_acceleration_knots_per_min,
                "max_deceleration_knots_per_min": self.max_deceleration_knots_per_min,
                "max_turn_rate_deg_per_min": self.max_turn_rate_deg_per_min,
            },
            "safety_domain": {
                "type": "four_quadrant_nm",
                "ahead_nm": self.safety_ahead_nm,
                "astern_nm": self.safety_astern_nm,
                "port_nm": self.safety_port_nm,
                "starboard_nm": self.safety_starboard_nm,
            },
        }


@dataclass(frozen=True, slots=True)
class WeightedVesselClass:
    profile: VesselClassSpec
    normalized_weight: str

    def canonical_dict(self) -> dict[str, object]:
        return {"weight": self.normalized_weight, "profile": self.profile.canonical_dict()}


@dataclass(frozen=True, slots=True)
class StreamFleetMixture:
    stream_id: str
    classes: tuple[WeightedVesselClass, ...]

    def canonical_dict(self) -> dict[str, object]:
        return {"stream_id": self.stream_id, "classes": [item.canonical_dict() for item in self.classes]}


@dataclass(frozen=True, slots=True)
class FleetMixturePlan:
    schema: str
    master_seed: int
    streams: tuple[StreamFleetMixture, ...]
    plan_identity_sha256: str

    def canonical_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "master_seed": self.master_seed,
            "streams": [stream.canonical_dict() for stream in self.streams],
        }
        if include_identity:
            payload["plan_identity_sha256"] = self.plan_identity_sha256
        return payload


@dataclass(frozen=True, slots=True)
class FleetMixtureSample:
    stream_id: str
    sample_index: int
    derived_seed: int
    profile: VesselClassSpec
    plan_identity_sha256: str
    sample_identity_sha256: str

    def canonical_dict(self) -> dict[str, object]:
        return {
            "stream_id": self.stream_id,
            "sample_index": self.sample_index,
            "derived_seed": self.derived_seed,
            "profile": self.profile.canonical_dict(),
            "plan_identity_sha256": self.plan_identity_sha256,
            "sample_identity_sha256": self.sample_identity_sha256,
        }


def _profile_from_mapping(raw: Mapping[str, Any]) -> VesselClassSpec:
    required = {"class_id", "dimensions_m", "speed", "maneuver_limits", "safety_domain"}
    if set(raw) != required:
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required)
        raise FleetMixtureError(f"vessel class fields mismatch; missing={missing}, extra={extra}")
    dimensions = raw["dimensions_m"]
    speed = raw["speed"]
    maneuver = raw["maneuver_limits"]
    domain = raw["safety_domain"]
    if not isinstance(dimensions, Mapping) or set(dimensions) != {"length_m", "beam_m"}:
        raise FleetMixtureError("dimensions_m must contain exactly length_m and beam_m")
    if not isinstance(speed, Mapping) or set(speed) != {"cruise_speed_knots", "maximum_speed_knots"}:
        raise FleetMixtureError("speed must contain exactly cruise_speed_knots and maximum_speed_knots")
    maneuver_keys = {
        "max_acceleration_knots_per_min",
        "max_deceleration_knots_per_min",
        "max_turn_rate_deg_per_min",
    }
    if not isinstance(maneuver, Mapping) or set(maneuver) != maneuver_keys:
        raise FleetMixtureError("maneuver_limits must contain the complete declared limit set")
    domain_keys = {"type", "ahead_nm", "astern_nm", "port_nm", "starboard_nm"}
    if not isinstance(domain, Mapping) or set(domain) != domain_keys:
        raise FleetMixtureError("safety_domain must contain a complete four-quadrant domain")
    if domain["type"] != "four_quadrant_nm":
        raise FleetMixtureError("safety_domain.type must be 'four_quadrant_nm'")
    return VesselClassSpec(
        class_id=raw["class_id"],
        length_m=dimensions["length_m"],
        beam_m=dimensions["beam_m"],
        cruise_speed_knots=speed["cruise_speed_knots"],
        maximum_speed_knots=speed["maximum_speed_knots"],
        max_acceleration_knots_per_min=maneuver["max_acceleration_knots_per_min"],
        max_deceleration_knots_per_min=maneuver["max_deceleration_knots_per_min"],
        max_turn_rate_deg_per_min=maneuver["max_turn_rate_deg_per_min"],
        safety_ahead_nm=domain["ahead_nm"],
        safety_astern_nm=domain["astern_nm"],
        safety_port_nm=domain["port_nm"],
        safety_starboard_nm=domain["starboard_nm"],
    )


def _decimal_weight(value: object) -> Decimal:
    if isinstance(value, bool):
        raise FleetMixtureError("weight must be finite and non-negative")
    try:
        out = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise FleetMixtureError("weight must be finite and non-negative") from exc
    if not out.is_finite() or out < 0:
        raise FleetMixtureError("weight must be finite and non-negative")
    return out


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def fleet_mixture_plan_from_mapping(mapping: Mapping[str, Any]) -> FleetMixturePlan:
    """Validate complete class profiles and canonically normalize weights."""
    if not isinstance(mapping, Mapping) or set(mapping) != {"schema", "master_seed", "streams"}:
        raise FleetMixtureError("fleet mixture plan must contain exactly schema, master_seed, streams")
    if mapping["schema"] != FLEET_MIXTURE_SCHEMA:
        raise FleetMixtureError(f"schema must be {FLEET_MIXTURE_SCHEMA!r}")
    seed = mapping["master_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FleetMixtureError("master_seed must be a non-negative integer")
    raw_streams = mapping["streams"]
    if not isinstance(raw_streams, (list, tuple)) or not raw_streams:
        raise FleetMixtureError("streams must be a non-empty sequence")

    streams: list[StreamFleetMixture] = []
    seen_streams: set[str] = set()
    for raw_stream in raw_streams:
        if not isinstance(raw_stream, Mapping) or set(raw_stream) != {"stream_id", "classes"}:
            raise FleetMixtureError("each stream must contain exactly stream_id and classes")
        stream_id = _non_empty("stream_id", raw_stream["stream_id"])
        if stream_id in seen_streams:
            raise FleetMixtureError(f"duplicate stream mixture {stream_id!r}")
        seen_streams.add(stream_id)
        raw_classes = raw_stream["classes"]
        if not isinstance(raw_classes, (list, tuple)) or not raw_classes:
            raise FleetMixtureError("classes must be a non-empty sequence")

        parsed: list[tuple[VesselClassSpec, Decimal]] = []
        seen_classes: set[str] = set()
        for raw_member in raw_classes:
            if not isinstance(raw_member, Mapping) or set(raw_member) != {"weight", "profile"}:
                raise FleetMixtureError("each mixture member must contain exactly weight and profile")
            if not isinstance(raw_member["profile"], Mapping):
                raise FleetMixtureError("profile must be a mapping")
            profile = _profile_from_mapping(raw_member["profile"])
            if profile.class_id in seen_classes:
                raise FleetMixtureError(f"duplicate class {profile.class_id!r} in stream {stream_id!r}")
            seen_classes.add(profile.class_id)
            parsed.append((profile, _decimal_weight(raw_member["weight"])))

        total = sum((weight for _, weight in parsed), Decimal(0))
        if total <= 0:
            raise FleetMixtureError("at least one mixture weight must be positive")
        with localcontext() as ctx:
            ctx.prec = 50
            members = tuple(
                WeightedVesselClass(profile, _canonical_decimal(weight / total))
                for profile, weight in sorted(parsed, key=lambda row: row[0].class_id)
            )
        streams.append(StreamFleetMixture(stream_id, members))

    streams.sort(key=lambda item: item.stream_id)
    material = {
        "schema": FLEET_MIXTURE_SCHEMA,
        "master_seed": int(seed),
        "streams": [stream.canonical_dict() for stream in streams],
    }
    return FleetMixturePlan(FLEET_MIXTURE_SCHEMA, int(seed), tuple(streams), _sha(material))


def _derived_seed(plan: FleetMixturePlan, stream_id: str, sample_index: int) -> int:
    material = {
        "schema": FLEET_MIXTURE_SCHEMA,
        "plan_identity_sha256": plan.plan_identity_sha256,
        "master_seed": plan.master_seed,
        "stream_id": stream_id,
        "sample_index": sample_index,
    }
    return int.from_bytes(sha256(_canonical_bytes(material)).digest()[:8], "big", signed=False)


def _select(stream: StreamFleetMixture, rng: Random) -> VesselClassSpec:
    point = Decimal(str(rng.random()))
    cumulative = Decimal(0)
    for member in stream.classes:
        cumulative += Decimal(member.normalized_weight)
        if point < cumulative:
            return member.profile
    # Decimal normalization/printing can leave a final representational remainder;
    # the last positive member is the deterministic closed-interval fallback.
    for member in reversed(stream.classes):
        if Decimal(member.normalized_weight) > 0:
            return member.profile
    raise FleetMixtureError("fleet mixture has no positive-weight class")


def sample_fleet_mixture(plan: FleetMixturePlan, sample_index: int) -> tuple[FleetMixtureSample, ...]:
    """Select one complete coherent class per stream for a deterministic vessel index."""
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise FleetMixtureError("sample_index must be a non-negative integer")
    output: list[FleetMixtureSample] = []
    for stream in plan.streams:
        seed = _derived_seed(plan, stream.stream_id, sample_index)
        profile = _select(stream, Random(seed))
        material = {
            "schema": FLEET_MIXTURE_SCHEMA,
            "plan_identity_sha256": plan.plan_identity_sha256,
            "stream_id": stream.stream_id,
            "sample_index": sample_index,
            "derived_seed": seed,
            "profile": profile.canonical_dict(),
        }
        output.append(
            FleetMixtureSample(
                stream_id=stream.stream_id,
                sample_index=sample_index,
                derived_seed=seed,
                profile=profile,
                plan_identity_sha256=plan.plan_identity_sha256,
                sample_identity_sha256=_sha(material),
            )
        )
    return tuple(output)
