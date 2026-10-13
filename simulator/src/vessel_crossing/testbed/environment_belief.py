"""C6 environmental truth versus environmental belief.

This layer extends the post-A162 research testbed without replacing the validated
A162 environmental kernel.  It defines explicit spatial/temporal environmental truth
fields and separate controller-visible environmental beliefs.  The existing
``EnvironmentalWorld2D`` uniform drift can be represented exactly through an adapter.

Spatial/temporal fields are infrastructure and are not automatically injected into the
A162 plant by this module.  Later fault/recoverability protocols can sample these truth
fields through explicit adapters while preserving the A162 compatibility path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import cos, isfinite, log, pi, sin, sqrt
from typing import Protocol

from vessel_crossing.kinematics import Vector2
from vessel_crossing.nextgen.environmental_world import EnvironmentalWorld2D

from .scope import TestbedFeature

_EPS = 1e-12


def _vector2(value: Vector2, *, name: str) -> Vector2:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two components")
    x, y = float(value[0]), float(value[1])
    if not isfinite(x) or not isfinite(y):
        raise ValueError(f"{name} components must be finite")
    return (x, y)


def _nonnegative(value: float, *, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _sha256_payload(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalFieldSample:
    """Physical environmental truth at one location/time.

    Current and wind are local east/north velocity vectors in nautical miles/minute.
    Wind is represented as environmental truth but is not silently coupled into the
    inherited A162 plant, whose validated environmental dynamics use additive drift.
    """

    time_min: float
    x_east_nm: float
    y_north_nm: float
    current_velocity_nm_per_min: Vector2 = (0.0, 0.0)
    wind_velocity_nm_per_min: Vector2 = (0.0, 0.0)
    field_id: str = "cross-c6/field/unspecified"
    active_anomaly_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.time_min) or self.time_min < 0.0:
            raise ValueError("time_min must be finite and non-negative")
        if not isfinite(self.x_east_nm) or not isfinite(self.y_north_nm):
            raise ValueError("sample position must be finite")
        if not self.field_id:
            raise ValueError("field_id must be non-empty")
        object.__setattr__(
            self,
            "current_velocity_nm_per_min",
            _vector2(self.current_velocity_nm_per_min, name="current_velocity_nm_per_min"),
        )
        object.__setattr__(
            self,
            "wind_velocity_nm_per_min",
            _vector2(self.wind_velocity_nm_per_min, name="wind_velocity_nm_per_min"),
        )
        ids = tuple(self.active_anomaly_ids)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("active_anomaly_ids must be unique and sorted")

    def identity_sha256(self) -> str:
        return _sha256_payload(
            {
                "time_min": self.time_min,
                "x_east_nm": self.x_east_nm,
                "y_north_nm": self.y_north_nm,
                "current": list(self.current_velocity_nm_per_min),
                "wind": list(self.wind_velocity_nm_per_min),
                "field_id": self.field_id,
                "active_anomaly_ids": list(self.active_anomaly_ids),
            }
        )


class EnvironmentalTruthField(Protocol):
    field_id: str

    def sample(self, *, time_min: float, x_east_nm: float, y_north_nm: float) -> EnvironmentalFieldSample:
        ...

    def configuration_sha256(self) -> str:
        ...


@dataclass(frozen=True, slots=True)
class UniformEnvironmentalField:
    current_velocity_nm_per_min: Vector2 = (0.0, 0.0)
    wind_velocity_nm_per_min: Vector2 = (0.0, 0.0)
    field_id: str = "cross-c6/uniform-field/v1"

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("field_id must be non-empty")
        object.__setattr__(
            self,
            "current_velocity_nm_per_min",
            _vector2(self.current_velocity_nm_per_min, name="current_velocity_nm_per_min"),
        )
        object.__setattr__(
            self,
            "wind_velocity_nm_per_min",
            _vector2(self.wind_velocity_nm_per_min, name="wind_velocity_nm_per_min"),
        )

    def sample(self, *, time_min: float, x_east_nm: float, y_north_nm: float) -> EnvironmentalFieldSample:
        return EnvironmentalFieldSample(
            time_min=time_min,
            x_east_nm=x_east_nm,
            y_north_nm=y_north_nm,
            current_velocity_nm_per_min=self.current_velocity_nm_per_min,
            wind_velocity_nm_per_min=self.wind_velocity_nm_per_min,
            field_id=self.field_id,
        )

    def configuration_sha256(self) -> str:
        return _sha256_payload(
            {
                "kind": "uniform",
                "field_id": self.field_id,
                "current": list(self.current_velocity_nm_per_min),
                "wind": list(self.wind_velocity_nm_per_min),
            }
        )


@dataclass(frozen=True, slots=True)
class CircularFieldAnomaly:
    anomaly_id: str
    start_min: float
    end_min: float
    center_east_nm: float
    center_north_nm: float
    radius_nm: float
    current_delta_nm_per_min: Vector2 = (0.0, 0.0)
    wind_delta_nm_per_min: Vector2 = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.anomaly_id:
            raise ValueError("anomaly_id must be non-empty")
        if not isfinite(self.start_min) or self.start_min < 0.0:
            raise ValueError("start_min must be finite and non-negative")
        if not isfinite(self.end_min) or self.end_min <= self.start_min:
            raise ValueError("end_min must be finite and greater than start_min")
        if not isfinite(self.center_east_nm) or not isfinite(self.center_north_nm):
            raise ValueError("anomaly center must be finite")
        if not isfinite(self.radius_nm) or self.radius_nm <= 0.0:
            raise ValueError("radius_nm must be finite and positive")
        object.__setattr__(
            self,
            "current_delta_nm_per_min",
            _vector2(self.current_delta_nm_per_min, name="current_delta_nm_per_min"),
        )
        object.__setattr__(
            self,
            "wind_delta_nm_per_min",
            _vector2(self.wind_delta_nm_per_min, name="wind_delta_nm_per_min"),
        )

    def active_at(self, *, time_min: float, x_east_nm: float, y_north_nm: float) -> bool:
        if not self.start_min <= time_min < self.end_min:
            return False
        dx = x_east_nm - self.center_east_nm
        dy = y_north_nm - self.center_north_nm
        return dx * dx + dy * dy <= self.radius_nm * self.radius_nm + _EPS

    def payload(self) -> dict[str, object]:
        return {
            "anomaly_id": self.anomaly_id,
            "start_min": self.start_min,
            "end_min": self.end_min,
            "center": [self.center_east_nm, self.center_north_nm],
            "radius_nm": self.radius_nm,
            "current_delta": list(self.current_delta_nm_per_min),
            "wind_delta": list(self.wind_delta_nm_per_min),
        }


@dataclass(frozen=True, slots=True)
class AnomalousEnvironmentalField:
    base: EnvironmentalTruthField
    anomalies: tuple[CircularFieldAnomaly, ...]
    field_id: str = "cross-c6/anomalous-field/v1"

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("field_id must be non-empty")
        ids = [a.anomaly_id for a in self.anomalies]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("anomalies must be unique and sorted by anomaly_id")

    @classmethod
    def from_unsorted(
        cls,
        *,
        base: EnvironmentalTruthField,
        anomalies: tuple[CircularFieldAnomaly, ...],
        field_id: str = "cross-c6/anomalous-field/v1",
    ) -> "AnomalousEnvironmentalField":
        return cls(base=base, anomalies=tuple(sorted(anomalies, key=lambda a: a.anomaly_id)), field_id=field_id)

    def sample(self, *, time_min: float, x_east_nm: float, y_north_nm: float) -> EnvironmentalFieldSample:
        base = self.base.sample(time_min=time_min, x_east_nm=x_east_nm, y_north_nm=y_north_nm)
        current = list(base.current_velocity_nm_per_min)
        wind = list(base.wind_velocity_nm_per_min)
        active: list[str] = []
        for anomaly in self.anomalies:
            if anomaly.active_at(time_min=time_min, x_east_nm=x_east_nm, y_north_nm=y_north_nm):
                active.append(anomaly.anomaly_id)
                current[0] += anomaly.current_delta_nm_per_min[0]
                current[1] += anomaly.current_delta_nm_per_min[1]
                wind[0] += anomaly.wind_delta_nm_per_min[0]
                wind[1] += anomaly.wind_delta_nm_per_min[1]
        return EnvironmentalFieldSample(
            time_min=time_min,
            x_east_nm=x_east_nm,
            y_north_nm=y_north_nm,
            current_velocity_nm_per_min=(current[0], current[1]),
            wind_velocity_nm_per_min=(wind[0], wind[1]),
            field_id=self.field_id,
            active_anomaly_ids=tuple(active),
        )

    def configuration_sha256(self) -> str:
        return _sha256_payload(
            {
                "kind": "anomalous",
                "field_id": self.field_id,
                "base": self.base.configuration_sha256(),
                "anomalies": [a.payload() for a in self.anomalies],
            }
        )


@dataclass(frozen=True, slots=True)
class SeededStochasticEnvironmentalField:
    """Frozen-seed, bucketed environmental perturbation field.

    The pseudo-random value is a pure function of seed + time bucket + spatial cell,
    so evaluation order cannot alter results and common-random-number schedules remain
    possible in later protocols.
    """

    base: EnvironmentalTruthField
    seed: int
    time_bucket_min: float = 5.0
    spatial_cell_nm: float = 1.0
    current_std_nm_per_min: float = 0.0
    wind_std_nm_per_min: float = 0.0
    field_id: str = "cross-c6/seeded-stochastic-field/v1"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.field_id:
            raise ValueError("field_id must be non-empty")
        if not isfinite(self.time_bucket_min) or self.time_bucket_min <= 0.0:
            raise ValueError("time_bucket_min must be finite and positive")
        if not isfinite(self.spatial_cell_nm) or self.spatial_cell_nm <= 0.0:
            raise ValueError("spatial_cell_nm must be finite and positive")
        _nonnegative(self.current_std_nm_per_min, name="current_std_nm_per_min")
        _nonnegative(self.wind_std_nm_per_min, name="wind_std_nm_per_min")

    def sample(self, *, time_min: float, x_east_nm: float, y_north_nm: float) -> EnvironmentalFieldSample:
        base = self.base.sample(time_min=time_min, x_east_nm=x_east_nm, y_north_nm=y_north_nm)
        tb = int(time_min // self.time_bucket_min)
        xb = int(x_east_nm // self.spatial_cell_nm)
        yb = int(y_north_nm // self.spatial_cell_nm)
        key = f"{self.seed}|{tb}|{xb}|{yb}"
        c = (
            base.current_velocity_nm_per_min[0]
            + self.current_std_nm_per_min * _normal01(key, "current-east"),
            base.current_velocity_nm_per_min[1]
            + self.current_std_nm_per_min * _normal01(key, "current-north"),
        )
        w = (
            base.wind_velocity_nm_per_min[0]
            + self.wind_std_nm_per_min * _normal01(key, "wind-east"),
            base.wind_velocity_nm_per_min[1]
            + self.wind_std_nm_per_min * _normal01(key, "wind-north"),
        )
        return EnvironmentalFieldSample(
            time_min=time_min,
            x_east_nm=x_east_nm,
            y_north_nm=y_north_nm,
            current_velocity_nm_per_min=c,
            wind_velocity_nm_per_min=w,
            field_id=self.field_id,
            active_anomaly_ids=base.active_anomaly_ids,
        )

    def configuration_sha256(self) -> str:
        return _sha256_payload(
            {
                "kind": "seeded-stochastic",
                "field_id": self.field_id,
                "base": self.base.configuration_sha256(),
                "seed": self.seed,
                "time_bucket_min": self.time_bucket_min,
                "spatial_cell_nm": self.spatial_cell_nm,
                "current_std_nm_per_min": self.current_std_nm_per_min,
                "wind_std_nm_per_min": self.wind_std_nm_per_min,
            }
        )


@dataclass(frozen=True, slots=True)
class VectorUncertaintyBound:
    east_abs_nm_per_min: float = 0.0
    north_abs_nm_per_min: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative(self.east_abs_nm_per_min, name="east_abs_nm_per_min")
        _nonnegative(self.north_abs_nm_per_min, name="north_abs_nm_per_min")


@dataclass(frozen=True, slots=True)
class EnvironmentalUncertaintyBounds:
    current: VectorUncertaintyBound = VectorUncertaintyBound()
    wind: VectorUncertaintyBound = VectorUncertaintyBound()


@dataclass(frozen=True, slots=True)
class EnvironmentalBeliefSnapshot:
    """Controller-visible environment estimate with explicit age/bounds."""

    observation_time_min: float
    source_time_min: float
    x_east_nm: float
    y_north_nm: float
    estimated_current_velocity_nm_per_min: Vector2
    estimated_wind_velocity_nm_per_min: Vector2
    uncertainty: EnvironmentalUncertaintyBounds
    provider_id: str
    source_field_id: str
    source_field_configuration_sha256: str

    def __post_init__(self) -> None:
        if not isfinite(self.observation_time_min) or self.observation_time_min < 0.0:
            raise ValueError("observation_time_min must be finite and non-negative")
        if not isfinite(self.source_time_min) or self.source_time_min < 0.0:
            raise ValueError("source_time_min must be finite and non-negative")
        if self.source_time_min > self.observation_time_min + _EPS:
            raise ValueError("environment belief source time cannot be in the future")
        if not isfinite(self.x_east_nm) or not isfinite(self.y_north_nm):
            raise ValueError("belief position must be finite")
        if not self.provider_id or not self.source_field_id:
            raise ValueError("provider_id and source_field_id must be non-empty")
        object.__setattr__(
            self,
            "estimated_current_velocity_nm_per_min",
            _vector2(self.estimated_current_velocity_nm_per_min, name="estimated_current_velocity_nm_per_min"),
        )
        object.__setattr__(
            self,
            "estimated_wind_velocity_nm_per_min",
            _vector2(self.estimated_wind_velocity_nm_per_min, name="estimated_wind_velocity_nm_per_min"),
        )
        if len(self.source_field_configuration_sha256) != 64:
            raise ValueError("source_field_configuration_sha256 must be a SHA-256 digest")
        try:
            int(self.source_field_configuration_sha256, 16)
        except ValueError as exc:
            raise ValueError("source_field_configuration_sha256 must be a SHA-256 digest") from exc

    @property
    def age_min(self) -> float:
        return self.observation_time_min - self.source_time_min


class EnvironmentalBeliefProvider(Protocol):
    component_id: TestbedFeature
    provider_id: str

    def observe(
        self,
        field: EnvironmentalTruthField,
        *,
        time_min: float,
        x_east_nm: float,
        y_north_nm: float,
    ) -> EnvironmentalBeliefSnapshot:
        ...


@dataclass(frozen=True, slots=True)
class PerfectEnvironmentalBeliefProvider:
    component_id = TestbedFeature.C6_ENVIRONMENT_BELIEF
    provider_id: str = "cross-c6/perfect-environment-belief/v1"

    def observe(
        self,
        field: EnvironmentalTruthField,
        *,
        time_min: float,
        x_east_nm: float,
        y_north_nm: float,
    ) -> EnvironmentalBeliefSnapshot:
        sample = field.sample(time_min=time_min, x_east_nm=x_east_nm, y_north_nm=y_north_nm)
        return EnvironmentalBeliefSnapshot(
            observation_time_min=time_min,
            source_time_min=time_min,
            x_east_nm=x_east_nm,
            y_north_nm=y_north_nm,
            estimated_current_velocity_nm_per_min=sample.current_velocity_nm_per_min,
            estimated_wind_velocity_nm_per_min=sample.wind_velocity_nm_per_min,
            uncertainty=EnvironmentalUncertaintyBounds(),
            provider_id=self.provider_id,
            source_field_id=field.field_id,
            source_field_configuration_sha256=field.configuration_sha256(),
        )


NoOpEnvironmentalBeliefProvider = PerfectEnvironmentalBeliefProvider


@dataclass(frozen=True, slots=True)
class DelayedBiasedEnvironmentalBeliefProvider:
    delay_min: float = 0.0
    current_bias_nm_per_min: Vector2 = (0.0, 0.0)
    wind_bias_nm_per_min: Vector2 = (0.0, 0.0)
    uncertainty: EnvironmentalUncertaintyBounds = EnvironmentalUncertaintyBounds()
    provider_id: str = "cross-c6/delayed-biased-environment-belief/v1"
    component_id = TestbedFeature.C6_ENVIRONMENT_BELIEF

    def __post_init__(self) -> None:
        _nonnegative(self.delay_min, name="delay_min")
        if not self.provider_id:
            raise ValueError("provider_id must be non-empty")
        object.__setattr__(
            self,
            "current_bias_nm_per_min",
            _vector2(self.current_bias_nm_per_min, name="current_bias_nm_per_min"),
        )
        object.__setattr__(
            self,
            "wind_bias_nm_per_min",
            _vector2(self.wind_bias_nm_per_min, name="wind_bias_nm_per_min"),
        )

    def observe(
        self,
        field: EnvironmentalTruthField,
        *,
        time_min: float,
        x_east_nm: float,
        y_north_nm: float,
    ) -> EnvironmentalBeliefSnapshot:
        if not isfinite(time_min) or time_min < 0.0:
            raise ValueError("time_min must be finite and non-negative")
        source_time = max(0.0, time_min - self.delay_min)
        sample = field.sample(time_min=source_time, x_east_nm=x_east_nm, y_north_nm=y_north_nm)
        current = (
            sample.current_velocity_nm_per_min[0] + self.current_bias_nm_per_min[0],
            sample.current_velocity_nm_per_min[1] + self.current_bias_nm_per_min[1],
        )
        wind = (
            sample.wind_velocity_nm_per_min[0] + self.wind_bias_nm_per_min[0],
            sample.wind_velocity_nm_per_min[1] + self.wind_bias_nm_per_min[1],
        )
        return EnvironmentalBeliefSnapshot(
            observation_time_min=time_min,
            source_time_min=source_time,
            x_east_nm=x_east_nm,
            y_north_nm=y_north_nm,
            estimated_current_velocity_nm_per_min=current,
            estimated_wind_velocity_nm_per_min=wind,
            uncertainty=self.uncertainty,
            provider_id=self.provider_id,
            source_field_id=field.field_id,
            source_field_configuration_sha256=field.configuration_sha256(),
        )


def truth_sample_from_environmental_world2d(
    world: EnvironmentalWorld2D,
    *,
    vessel_id: str,
    wind_velocity_nm_per_min: Vector2 = (0.0, 0.0),
    field_id: str = "cross-c6/a162-environmental-world-adapter/v1",
) -> EnvironmentalFieldSample:
    """Represent the current A162 uniform drift exactly as C6 truth evidence."""

    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    state = world.snapshot().by_id(vessel_id)
    return EnvironmentalFieldSample(
        time_min=world.time,
        x_east_nm=state.position[0],
        y_north_nm=state.position[1],
        current_velocity_nm_per_min=world.drift_for(vessel_id),
        wind_velocity_nm_per_min=_vector2(wind_velocity_nm_per_min, name="wind_velocity_nm_per_min"),
        field_id=field_id,
    )


def _normal01(key: str, component: str) -> float:
    # Box-Muller with two domain-separated deterministic uniforms.
    u1 = _uniform01(key, component, "u1")
    u2 = _uniform01(key, component, "u2")
    u1 = max(u1, 1e-15)
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


def _uniform01(*parts: object) -> float:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 0.5) / 2**64
