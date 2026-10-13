"""Timestamped pair observations and compass-bearing trend evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot
from typing import Sequence

from .dynamics import Vessel2DState, signed_heading_delta
from .encounter import absolute_bearing_deg

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class PairObservation:
    time_min: float
    first_id: str
    second_id: str
    range_nm: float
    first_to_second_bearing_deg: float
    second_to_first_bearing_deg: float

    def __post_init__(self) -> None:
        if self.time_min < 0.0:
            raise ValueError("time_min must be non-negative")
        if not self.first_id or not self.second_id or self.first_id == self.second_id:
            raise ValueError("observation requires two distinct non-empty vessel ids")
        if self.range_nm < 0.0:
            raise ValueError("range_nm must be non-negative")


@dataclass(frozen=True, slots=True)
class BearingTrendConfig:
    """Explicit observation thresholds; these are model parameters, not COLREG numbers."""

    min_duration_min: float
    appreciable_change_deg: float
    min_range_closure_nm: float = 0.0

    def __post_init__(self) -> None:
        if self.min_duration_min <= 0.0:
            raise ValueError("min_duration_min must be positive")
        if self.appreciable_change_deg <= 0.0 or self.appreciable_change_deg > 180.0:
            raise ValueError("appreciable_change_deg must be in (0, 180]")
        if self.min_range_closure_nm < 0.0:
            raise ValueError("min_range_closure_nm must be non-negative")


class BearingTrendEvidence(str, Enum):
    INSUFFICIENT_HISTORY = "insufficient_history"
    NOT_CLOSING = "not_closing"
    STEADY_CLOSING_BEARING = "steady_closing_bearing"
    APPRECIABLY_CHANGING_BEARING = "appreciably_changing_bearing"


@dataclass(frozen=True, slots=True)
class BearingTrendAssessment:
    evidence: BearingTrendEvidence
    duration_min: float
    range_change_nm: float
    bearing_net_change_deg: float
    bearing_excursion_deg: float
    observation_count: int

    @property
    def indicates_collision_risk(self) -> bool:
        return self.evidence is BearingTrendEvidence.STEADY_CLOSING_BEARING


def observe_pair(first: Vessel2DState, second: Vessel2DState, *, time_min: float) -> PairObservation:
    """Create a compass-bearing/range observation from a stable ordered vessel pair."""
    dx = second.position[0] - first.position[0]
    dy = second.position[1] - first.position[1]
    return PairObservation(
        time_min=time_min,
        first_id=first.vessel_id,
        second_id=second.vessel_id,
        range_nm=hypot(dx, dy),
        first_to_second_bearing_deg=absolute_bearing_deg(first.position, second.position),
        second_to_first_bearing_deg=absolute_bearing_deg(second.position, first.position),
    )


def _validate_history(observations: Sequence[PairObservation]) -> None:
    if not observations:
        return
    pair = (observations[0].first_id, observations[0].second_id)
    previous_time = -1.0
    for observation in observations:
        if (observation.first_id, observation.second_id) != pair:
            raise ValueError("all bearing observations must use the same ordered vessel pair")
        if observation.time_min <= previous_time + _EPS:
            raise ValueError("observation times must be strictly increasing")
        previous_time = observation.time_min


def _unwrap_bearings(observations: Sequence[PairObservation]) -> list[float]:
    if not observations:
        return []
    unwrapped = [observations[0].first_to_second_bearing_deg]
    previous_raw = observations[0].first_to_second_bearing_deg
    for observation in observations[1:]:
        current_raw = observation.first_to_second_bearing_deg
        unwrapped.append(unwrapped[-1] + signed_heading_delta(previous_raw, current_raw))
        previous_raw = current_raw
    return unwrapped


def assess_bearing_trend(
    observations: Sequence[PairObservation],
    *,
    config: BearingTrendConfig,
) -> BearingTrendAssessment:
    """Assess compass-bearing change over a timestamped observation window.

    Bearings are unwrapped through 0/360 before excursion is measured, avoiding the
    classic false 358° change when a line of sight moves from 359° to 1°.
    """
    _validate_history(observations)
    if len(observations) < 2:
        return BearingTrendAssessment(
            BearingTrendEvidence.INSUFFICIENT_HISTORY,
            0.0,
            0.0,
            0.0,
            0.0,
            len(observations),
        )

    duration = observations[-1].time_min - observations[0].time_min
    unwrapped = _unwrap_bearings(observations)
    net_change = abs(unwrapped[-1] - unwrapped[0])
    excursion = max(unwrapped) - min(unwrapped)
    range_change = observations[-1].range_nm - observations[0].range_nm

    if duration + _EPS < config.min_duration_min:
        evidence = BearingTrendEvidence.INSUFFICIENT_HISTORY
    else:
        closing_amount = -range_change
        if closing_amount <= config.min_range_closure_nm + _EPS:
            evidence = BearingTrendEvidence.NOT_CLOSING
        elif excursion <= config.appreciable_change_deg + _EPS:
            evidence = BearingTrendEvidence.STEADY_CLOSING_BEARING
        else:
            evidence = BearingTrendEvidence.APPRECIABLY_CHANGING_BEARING

    return BearingTrendAssessment(
        evidence=evidence,
        duration_min=duration,
        range_change_nm=range_change,
        bearing_net_change_deg=net_change,
        bearing_excursion_deg=excursion,
        observation_count=len(observations),
    )
