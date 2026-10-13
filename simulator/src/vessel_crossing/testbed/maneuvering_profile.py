"""Vessel-specific manoeuvring-table provenance and deterministic interpolation.

This module is data infrastructure for MANEUVERING_REALISM MR-3.  It deliberately
stores bridge/booklet-style turning and stopping evidence without pretending that
one full-speed table defines low-speed or astern behaviour.  Interpolation is only
performed inside explicitly supplied one-dimensional operating families.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from typing import Any

_EPS = 1e-12


class ManeuveringProfileError(ValueError):
    """Invalid or unsupported manoeuvring-profile request."""


class MissingManeuveringDataError(ManeuveringProfileError):
    """Requested operating region has no declared data support."""


def _finite_nonnegative(name: str, value: float | None) -> None:
    if value is not None and (not isfinite(value) or value < 0.0):
        raise ManeuveringProfileError(f"{name} must be finite and non-negative when supplied")


@dataclass(frozen=True, slots=True)
class TurningTablePoint:
    """One turning-circle/booklet operating point.

    ``rudder_angle_deg`` is signed in the profile's declared rudder convention.
    Distances are nautical miles so the values can be consumed directly by CROSS.
    Missing quantities remain ``None`` rather than being inferred.
    """

    speed_knots: float
    rudder_angle_deg: float
    advance_nm: float | None = None
    transfer_nm: float | None = None
    tactical_diameter_nm: float | None = None
    time_to_90_min: float | None = None
    steady_turn_speed_knots: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.speed_knots) or self.speed_knots < 0.0:
            raise ManeuveringProfileError("speed_knots must be finite and non-negative")
        if not isfinite(self.rudder_angle_deg) or abs(self.rudder_angle_deg) <= _EPS:
            raise ManeuveringProfileError("rudder_angle_deg must be finite and non-zero")
        for name in ("advance_nm", "transfer_nm", "tactical_diameter_nm", "time_to_90_min", "steady_turn_speed_knots"):
            _finite_nonnegative(name, getattr(self, name))
        if all(getattr(self, name) is None for name in (
            "advance_nm", "transfer_nm", "tactical_diameter_nm", "time_to_90_min", "steady_turn_speed_knots"
        )):
            raise ManeuveringProfileError("turning point must contain at least one measured quantity")


@dataclass(frozen=True, slots=True)
class StoppingTablePoint:
    """One stopping/reversing operating point for a named engine order family."""

    initial_speed_knots: float
    engine_order_id: str
    stopping_distance_nm: float | None = None
    stopping_time_min: float | None = None
    residual_speed_after_delay_knots: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.initial_speed_knots) or self.initial_speed_knots < 0.0:
            raise ManeuveringProfileError("initial_speed_knots must be finite and non-negative")
        if not self.engine_order_id:
            raise ManeuveringProfileError("engine_order_id must be non-empty")
        for name in ("stopping_distance_nm", "stopping_time_min", "residual_speed_after_delay_knots"):
            _finite_nonnegative(name, getattr(self, name))
        if self.stopping_distance_nm is None and self.stopping_time_min is None:
            raise ManeuveringProfileError("stopping point requires stopping distance and/or stopping time")


@dataclass(frozen=True, slots=True)
class InterpolatedTurningData:
    speed_knots: float
    rudder_angle_deg: float
    advance_nm: float | None
    transfer_nm: float | None
    tactical_diameter_nm: float | None
    time_to_90_min: float | None
    steady_turn_speed_knots: float | None
    lower_source_speed_knots: float
    upper_source_speed_knots: float


@dataclass(frozen=True, slots=True)
class InterpolatedStoppingData:
    initial_speed_knots: float
    engine_order_id: str
    stopping_distance_nm: float | None
    stopping_time_min: float | None
    residual_speed_after_delay_knots: float | None
    lower_source_speed_knots: float
    upper_source_speed_knots: float


def _lerp_optional(a: float | None, b: float | None, weight: float) -> float | None:
    # Missing empirical data stay missing; CROSS must not fabricate a quantity from
    # one endpoint simply because an adjacent operating point contains it.
    if a is None or b is None:
        return None
    return a + weight * (b - a)


def _canonical_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ManeuveringProfileError("canonical profile cannot contain NaN/inf")
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical_payload(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical_payload(v) for k, v in sorted(value.items())}
    raise TypeError(type(value).__name__)


@dataclass(frozen=True, slots=True)
class VesselManeuveringProfile:
    """Provenance-bound bridge/manoeuvring-booklet data for one vessel/class.

    Interpolation is allowed only between points sharing the exact same declared
    rudder angle or engine-order identifier.  Extrapolation is forbidden.  This is
    intentional: full/half-ahead tables do not silently define dead-slow behaviour.
    """

    profile_id: str
    profile_version: str
    vessel_or_class_id: str
    source_reference: str
    rudder_sign_convention: str
    turning_points: tuple[TurningTablePoint, ...] = ()
    stopping_points: tuple[StoppingTablePoint, ...] = ()

    def __post_init__(self) -> None:
        for name in ("profile_id", "profile_version", "vessel_or_class_id", "source_reference", "rudder_sign_convention"):
            if not getattr(self, name):
                raise ManeuveringProfileError(f"{name} must be non-empty")
        turning_keys = [(p.rudder_angle_deg, p.speed_knots) for p in self.turning_points]
        if len(turning_keys) != len(set(turning_keys)):
            raise ManeuveringProfileError("duplicate turning point for rudder/speed")
        stopping_keys = [(p.engine_order_id, p.initial_speed_knots) for p in self.stopping_points]
        if len(stopping_keys) != len(set(stopping_keys)):
            raise ManeuveringProfileError("duplicate stopping point for order/speed")

    def identity_sha256(self) -> str:
        payload = {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "vessel_or_class_id": self.vessel_or_class_id,
            "source_reference": self.source_reference,
            "rudder_sign_convention": self.rudder_sign_convention,
            "turning_points": [
                {
                    "speed_knots": p.speed_knots,
                    "rudder_angle_deg": p.rudder_angle_deg,
                    "advance_nm": p.advance_nm,
                    "transfer_nm": p.transfer_nm,
                    "tactical_diameter_nm": p.tactical_diameter_nm,
                    "time_to_90_min": p.time_to_90_min,
                    "steady_turn_speed_knots": p.steady_turn_speed_knots,
                }
                for p in sorted(self.turning_points, key=lambda item: (item.rudder_angle_deg, item.speed_knots))
            ],
            "stopping_points": [
                {
                    "initial_speed_knots": p.initial_speed_knots,
                    "engine_order_id": p.engine_order_id,
                    "stopping_distance_nm": p.stopping_distance_nm,
                    "stopping_time_min": p.stopping_time_min,
                    "residual_speed_after_delay_knots": p.residual_speed_after_delay_knots,
                }
                for p in sorted(self.stopping_points, key=lambda item: (item.engine_order_id, item.initial_speed_knots))
            ],
        }
        raw = json.dumps(_canonical_payload(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _bracket(points, requested: float, key):
        ordered = tuple(sorted(points, key=key))
        if not ordered:
            raise MissingManeuveringDataError("no declared points for requested operating family")
        values = tuple(key(point) for point in ordered)
        if requested < values[0] - _EPS or requested > values[-1] + _EPS:
            raise MissingManeuveringDataError("requested operating point would require extrapolation")
        for point in ordered:
            if abs(key(point) - requested) <= _EPS:
                return point, point, 0.0
        for lower, upper in zip(ordered, ordered[1:]):
            lo = key(lower)
            hi = key(upper)
            if lo < requested < hi:
                return lower, upper, (requested - lo) / (hi - lo)
        raise AssertionError("bracketing fell through")

    def turning_at(self, *, speed_knots: float, rudder_angle_deg: float) -> InterpolatedTurningData:
        if not isfinite(speed_knots) or speed_knots < 0.0:
            raise ManeuveringProfileError("speed_knots must be finite and non-negative")
        family = tuple(p for p in self.turning_points if abs(p.rudder_angle_deg - rudder_angle_deg) <= _EPS)
        if not family:
            raise MissingManeuveringDataError("no turning data for exact requested rudder angle")
        lo, hi, w = self._bracket(family, speed_knots, lambda p: p.speed_knots)
        return InterpolatedTurningData(
            speed_knots=speed_knots,
            rudder_angle_deg=rudder_angle_deg,
            advance_nm=_lerp_optional(lo.advance_nm, hi.advance_nm, w),
            transfer_nm=_lerp_optional(lo.transfer_nm, hi.transfer_nm, w),
            tactical_diameter_nm=_lerp_optional(lo.tactical_diameter_nm, hi.tactical_diameter_nm, w),
            time_to_90_min=_lerp_optional(lo.time_to_90_min, hi.time_to_90_min, w),
            steady_turn_speed_knots=_lerp_optional(lo.steady_turn_speed_knots, hi.steady_turn_speed_knots, w),
            lower_source_speed_knots=lo.speed_knots,
            upper_source_speed_knots=hi.speed_knots,
        )

    def stopping_at(self, *, initial_speed_knots: float, engine_order_id: str) -> InterpolatedStoppingData:
        if not isfinite(initial_speed_knots) or initial_speed_knots < 0.0:
            raise ManeuveringProfileError("initial_speed_knots must be finite and non-negative")
        if not engine_order_id:
            raise ManeuveringProfileError("engine_order_id must be non-empty")
        family = tuple(p for p in self.stopping_points if p.engine_order_id == engine_order_id)
        if not family:
            raise MissingManeuveringDataError("no stopping data for requested engine order")
        lo, hi, w = self._bracket(family, initial_speed_knots, lambda p: p.initial_speed_knots)
        return InterpolatedStoppingData(
            initial_speed_knots=initial_speed_knots,
            engine_order_id=engine_order_id,
            stopping_distance_nm=_lerp_optional(lo.stopping_distance_nm, hi.stopping_distance_nm, w),
            stopping_time_min=_lerp_optional(lo.stopping_time_min, hi.stopping_time_min, w),
            residual_speed_after_delay_knots=_lerp_optional(
                lo.residual_speed_after_delay_knots, hi.residual_speed_after_delay_knots, w
            ),
            lower_source_speed_knots=lo.initial_speed_knots,
            upper_source_speed_knots=hi.initial_speed_knots,
        )
