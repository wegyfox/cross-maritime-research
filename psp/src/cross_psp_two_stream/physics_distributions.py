"""Deterministic bounded scalar distributions for Custom Physics v2.

This layer normalizes and samples declared per-vessel scalar distributions only.  It
is intentionally separate from the qualified simulator binding: sampled values do not
inherit any frozen physics/evidence identity until a higher layer explicitly binds them.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from random import Random
from types import MappingProxyType
from typing import Any, Mapping

PHYSICS_DISTRIBUTION_SCHEMA = "cross-psp-physics-distribution-v1"
SAMPLING_SCOPE_PER_VESSEL = "per_vessel"
_SUPPORTED_FAMILIES = frozenset({"uniform", "normal", "triangular"})
_SAMPLE_FLOAT_SIGNIFICANT_DIGITS = 15


class PhysicsDistributionError(ValueError):
    """Raised when a public distribution plan is invalid or cannot be sampled safely."""


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


def _finite(name: str, value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise PhysicsDistributionError(f"{name} must be a finite number") from exc
    if not isfinite(out):
        raise PhysicsDistributionError(f"{name} must be a finite number")
    return out


def _non_empty(name: str, value: object) -> str:
    out = str(value).strip()
    if not out:
        raise PhysicsDistributionError(f"{name} must be non-empty")
    return out


@dataclass(frozen=True, slots=True)
class ScalarDistributionSpec:
    """One independently varied scalar for one stream."""

    parameter_id: str
    unit: str
    family: str
    lower: float
    upper: float
    parameters: Mapping[str, float]
    sampling_scope: str = SAMPLING_SCOPE_PER_VESSEL

    def __post_init__(self) -> None:
        parameter_id = _non_empty("parameter", self.parameter_id)
        unit = _non_empty("unit", self.unit)
        family = str(self.family).strip().lower()
        if family not in _SUPPORTED_FAMILIES:
            raise PhysicsDistributionError(
                f"unsupported distribution family {family!r}; expected one of {sorted(_SUPPORTED_FAMILIES)}"
            )
        if self.sampling_scope != SAMPLING_SCOPE_PER_VESSEL:
            raise PhysicsDistributionError("sampling_scope must be 'per_vessel'")
        lower = _finite("bounds.lower", self.lower)
        upper = _finite("bounds.upper", self.upper)
        if not lower < upper:
            raise PhysicsDistributionError("distribution bounds require lower < upper")

        raw = dict(self.parameters)
        if family == "uniform":
            if raw:
                raise PhysicsDistributionError("uniform parameters must be empty; use bounds")
            normalized: dict[str, float] = {}
        elif family == "normal":
            if set(raw) != {"mean", "stddev"}:
                raise PhysicsDistributionError("normal parameters must contain exactly mean and stddev")
            mean = _finite("parameters.mean", raw["mean"])
            stddev = _finite("parameters.stddev", raw["stddev"])
            if stddev <= 0.0:
                raise PhysicsDistributionError("normal stddev must be positive")
            if not lower <= mean <= upper:
                raise PhysicsDistributionError("normal mean must lie within declared bounds")
            normalized = {"mean": mean, "stddev": stddev}
        else:
            if set(raw) != {"mode"}:
                raise PhysicsDistributionError("triangular parameters must contain exactly mode")
            mode = _finite("parameters.mode", raw["mode"])
            if not lower <= mode <= upper:
                raise PhysicsDistributionError("triangular mode must lie within declared bounds")
            normalized = {"mode": mode}

        object.__setattr__(self, "parameter_id", parameter_id)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "parameters", MappingProxyType(normalized))

    def canonical_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter_id,
            "unit": self.unit,
            "family": self.family,
            "parameters": dict(self.parameters),
            "bounds": {"lower": self.lower, "upper": self.upper},
            "sampling_scope": self.sampling_scope,
        }


@dataclass(frozen=True, slots=True)
class StreamDistributionSpec:
    stream_id: str
    distribution: ScalarDistributionSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", _non_empty("stream_id", self.stream_id))

    def canonical_dict(self) -> dict[str, object]:
        return {"stream_id": self.stream_id, **self.distribution.canonical_dict()}


@dataclass(frozen=True, slots=True)
class PhysicsDistributionPlan:
    schema: str
    master_seed: int
    streams: tuple[StreamDistributionSpec, ...]
    plan_identity_sha256: str

    def canonical_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "master_seed": self.master_seed,
            "streams": [item.canonical_dict() for item in self.streams],
        }
        if include_identity:
            payload["plan_identity_sha256"] = self.plan_identity_sha256
        return payload


@dataclass(frozen=True, slots=True)
class DistributionSample:
    stream_id: str
    parameter_id: str
    unit: str
    sample_index: int
    derived_seed: int
    value: float
    plan_identity_sha256: str
    sample_identity_sha256: str

    def canonical_dict(self) -> dict[str, object]:
        return {
            "stream_id": self.stream_id,
            "parameter": self.parameter_id,
            "unit": self.unit,
            "sample_index": self.sample_index,
            "derived_seed": self.derived_seed,
            "value": self.value,
            "plan_identity_sha256": self.plan_identity_sha256,
            "sample_identity_sha256": self.sample_identity_sha256,
        }


def _distribution_from_mapping(raw: Mapping[str, Any]) -> ScalarDistributionSpec:
    required = {"parameter", "unit", "family", "parameters", "bounds", "sampling_scope"}
    if set(raw) != required:
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required)
        raise PhysicsDistributionError(f"distribution fields mismatch; missing={missing}, extra={extra}")
    bounds = raw["bounds"]
    if not isinstance(bounds, Mapping) or set(bounds) != {"lower", "upper"}:
        raise PhysicsDistributionError("bounds must contain exactly lower and upper")
    parameters = raw["parameters"]
    if not isinstance(parameters, Mapping):
        raise PhysicsDistributionError("parameters must be a mapping")
    return ScalarDistributionSpec(
        parameter_id=raw["parameter"],
        unit=raw["unit"],
        family=raw["family"],
        lower=bounds["lower"],
        upper=bounds["upper"],
        parameters=parameters,
        sampling_scope=raw["sampling_scope"],
    )


def physics_distribution_plan_from_mapping(mapping: Mapping[str, Any]) -> PhysicsDistributionPlan:
    """Validate and normalize the versioned public distribution schema."""
    if not isinstance(mapping, Mapping):
        raise PhysicsDistributionError("distribution plan must be a mapping")
    if set(mapping) != {"schema", "master_seed", "streams"}:
        raise PhysicsDistributionError("distribution plan must contain exactly schema, master_seed, streams")
    if mapping["schema"] != PHYSICS_DISTRIBUTION_SCHEMA:
        raise PhysicsDistributionError(f"schema must be {PHYSICS_DISTRIBUTION_SCHEMA!r}")
    seed = mapping["master_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PhysicsDistributionError("master_seed must be a non-negative integer")
    raw_streams = mapping["streams"]
    if not isinstance(raw_streams, (list, tuple)) or not raw_streams:
        raise PhysicsDistributionError("streams must be a non-empty sequence")

    streams: list[StreamDistributionSpec] = []
    seen: set[str] = set()
    for raw in raw_streams:
        if not isinstance(raw, Mapping) or set(raw) != {"stream_id", "distribution"}:
            raise PhysicsDistributionError("each stream must contain exactly stream_id and distribution")
        stream_id = _non_empty("stream_id", raw["stream_id"])
        if stream_id in seen:
            raise PhysicsDistributionError(f"duplicate stream distribution {stream_id!r}")
        seen.add(stream_id)
        distribution = raw["distribution"]
        if not isinstance(distribution, Mapping):
            raise PhysicsDistributionError("distribution must be a mapping")
        streams.append(StreamDistributionSpec(stream_id, _distribution_from_mapping(distribution)))

    streams.sort(key=lambda item: item.stream_id)
    material = {
        "schema": PHYSICS_DISTRIBUTION_SCHEMA,
        "master_seed": int(seed),
        "streams": [item.canonical_dict() for item in streams],
    }
    identity = _sha(material)
    return PhysicsDistributionPlan(PHYSICS_DISTRIBUTION_SCHEMA, int(seed), tuple(streams), identity)


def _derived_seed(plan: PhysicsDistributionPlan, stream_id: str, sample_index: int) -> int:
    material = {
        "schema": PHYSICS_DISTRIBUTION_SCHEMA,
        "plan_identity_sha256": plan.plan_identity_sha256,
        "master_seed": plan.master_seed,
        "stream_id": stream_id,
        "sample_index": sample_index,
    }
    return int.from_bytes(sha256(_canonical_bytes(material)).digest()[:8], "big", signed=False)


def _canonicalize_sample_float(value: float) -> float:
    """Normalize harmless libm/RNG last-bit drift before identity materialization.

    CPython's seeded RNG state is deterministic, but distribution helpers such as
    ``Random.gauss`` and ``Random.triangular`` call platform math functions.  Their
    final IEEE-754 result can therefore differ by a few ULPs across supported
    Python/OS combinations even when the seed and algorithmic path are identical.
    Fifteen significant decimal digits retain far more precision than the public
    physical inputs require while giving the identity layer a stable canonical
    float representation.
    """
    if not isfinite(value):
        raise PhysicsDistributionError("distribution produced a non-finite value")
    normalized = float(format(float(value), f".{_SAMPLE_FLOAT_SIGNIFICANT_DIGITS}g"))
    # Collapse signed zero so JSON identity material cannot distinguish -0.0/0.0.
    return 0.0 if normalized == 0.0 else normalized


def _sample_value(spec: ScalarDistributionSpec, rng: Random) -> float:
    if spec.family == "uniform":
        return rng.uniform(spec.lower, spec.upper)
    if spec.family == "triangular":
        return rng.triangular(spec.lower, spec.upper, spec.parameters["mode"])

    mean = spec.parameters["mean"]
    stddev = spec.parameters["stddev"]
    # Deterministic truncated-normal rejection.  A pathological public request that
    # cannot produce an in-bounds draw fails closed instead of clipping silently.
    for _ in range(10000):
        value = rng.gauss(mean, stddev)
        if spec.lower <= value <= spec.upper and isfinite(value):
            return value
    raise PhysicsDistributionError("bounded normal sampling failed after 10000 deterministic draws")


def sample_distribution_plan(plan: PhysicsDistributionPlan, sample_index: int) -> tuple[DistributionSample, ...]:
    """Resolve one deterministic per-vessel sample for every declared stream."""
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise PhysicsDistributionError("sample_index must be a non-negative integer")
    output: list[DistributionSample] = []
    for stream in plan.streams:
        seed = _derived_seed(plan, stream.stream_id, sample_index)
        value = _canonicalize_sample_float(
            _sample_value(stream.distribution, Random(seed))
        )
        if not stream.distribution.lower <= value <= stream.distribution.upper:
            raise PhysicsDistributionError(
                "canonicalized distribution value is out of declared bounds"
            )
        sample_material = {
            "schema": PHYSICS_DISTRIBUTION_SCHEMA,
            "plan_identity_sha256": plan.plan_identity_sha256,
            "stream_id": stream.stream_id,
            "parameter": stream.distribution.parameter_id,
            "unit": stream.distribution.unit,
            "sample_index": sample_index,
            "derived_seed": seed,
            "value": value,
        }
        output.append(
            DistributionSample(
                stream_id=stream.stream_id,
                parameter_id=stream.distribution.parameter_id,
                unit=stream.distribution.unit,
                sample_index=sample_index,
                derived_seed=seed,
                value=value,
                plan_identity_sha256=plan.plan_identity_sha256,
                sample_identity_sha256=_sha(sample_material),
            )
        )
    return tuple(output)
