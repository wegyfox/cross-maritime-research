"""Neutral geometric encounter classification, separate from COLREG responsibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dynamics import Vessel2DState, heading_unit_vector, signed_heading_delta
from .encounter import EncounterPrediction, predict_encounter

_EPS = 1e-12


class EncounterType(str, Enum):
    UNDEFINED = "undefined"
    NO_RELATIVE_MOTION = "no_relative_motion"
    DIVERGING = "diverging"
    PARALLEL = "parallel"
    OVERTAKING = "overtaking"
    HEAD_ON = "head_on"
    CROSSING = "crossing"


@dataclass(frozen=True, slots=True)
class EncounterClassificationConfig:
    """Numerical interpretation parameters for qualitative COLREG geometry.

    COLREG Rule 13 supplies the exact 22.5°-abaft-the-beam overtaking boundary.
    Rule 14 instead says reciprocal or *nearly* reciprocal / ahead or *nearly* ahead,
    so the numerical head-on tolerances below are explicit model choices rather than
    pretending the rule specifies an exact degree tolerance.
    """

    overtaking_abaft_beam_deg: float = 22.5
    head_on_course_tolerance_deg: float = 10.0
    head_on_bearing_tolerance_deg: float = 10.0
    parallel_course_tolerance_deg: float = 10.0
    require_overtaking_longitudinal_gain: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.overtaking_abaft_beam_deg < 90.0:
            raise ValueError("overtaking_abaft_beam_deg must be in [0, 90)")
        if not isinstance(self.require_overtaking_longitudinal_gain, bool):
            raise TypeError("require_overtaking_longitudinal_gain must be bool")
        for name in (
            "head_on_course_tolerance_deg",
            "head_on_bearing_tolerance_deg",
            "parallel_course_tolerance_deg",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 90.0:
                raise ValueError(f"{name} must be in [0, 90]")


@dataclass(frozen=True, slots=True)
class EncounterClassification:
    encounter_type: EncounterType
    prediction: EncounterPrediction
    course_difference_deg: float
    closing: bool
    overtaking_vessel_id: str | None = None
    overtaken_vessel_id: str | None = None


def _near_zero_bearing(relative_bearing: float, tolerance: float) -> bool:
    return relative_bearing <= tolerance + _EPS or relative_bearing >= 360.0 - tolerance - _EPS


def _in_overtaken_stern_sector(relative_bearing: float, abaft_beam: float) -> bool:
    # More than 22.5° abaft either beam -> strictly inside 112.5°..247.5° at default.
    start = 90.0 + abaft_beam
    end = 270.0 - abaft_beam
    return relative_bearing > start + _EPS and relative_bearing < end - _EPS


def _longitudinal_gain(overtaker: Vessel2DState, overtaken: Vessel2DState) -> float:
    """Relative velocity projected on the overtaken vessel's forward axis."""
    axis = heading_unit_vector(overtaken.heading_deg)
    rel = (overtaker.velocity[0] - overtaken.velocity[0], overtaker.velocity[1] - overtaken.velocity[1])
    return rel[0] * axis[0] + rel[1] * axis[1]


def classify_encounter(
    first: Vessel2DState,
    second: Vessel2DState,
    *,
    horizon: float | None = None,
    config: EncounterClassificationConfig | None = None,
) -> EncounterClassification:
    """Classify relative geometry without assigning give-way/stand-on responsibility."""
    cfg = config or EncounterClassificationConfig()
    prediction = predict_encounter(first, second, horizon=horizon)
    course_difference = abs(signed_heading_delta(first.heading_deg, second.heading_deg))

    if prediction.cpa.current_distance <= _EPS:
        return EncounterClassification(
            EncounterType.UNDEFINED,
            prediction,
            course_difference,
            closing=False,
        )

    if prediction.cpa.relative_speed <= _EPS:
        return EncounterClassification(
            EncounterType.NO_RELATIVE_MOTION,
            prediction,
            course_difference,
            closing=False,
        )

    closing = prediction.cpa.raw_tcpa > _EPS
    if not closing:
        return EncounterClassification(
            EncounterType.DIVERGING,
            prediction,
            course_difference,
            closing=False,
        )

    first_overtakes_second = _in_overtaken_stern_sector(
        prediction.second_relative_bearing_deg,
        cfg.overtaking_abaft_beam_deg,
    )
    second_overtakes_first = _in_overtaken_stern_sector(
        prediction.first_relative_bearing_deg,
        cfg.overtaking_abaft_beam_deg,
    )
    if first_overtakes_second and not second_overtakes_first:
        if cfg.require_overtaking_longitudinal_gain and _longitudinal_gain(first, second) <= _EPS:
            first_overtakes_second = False
    if second_overtakes_first and not first_overtakes_second:
        if cfg.require_overtaking_longitudinal_gain and _longitudinal_gain(second, first) <= _EPS:
            second_overtakes_first = False
    if first_overtakes_second and not second_overtakes_first:
        return EncounterClassification(
            EncounterType.OVERTAKING,
            prediction,
            course_difference,
            closing=True,
            overtaking_vessel_id=first.vessel_id,
            overtaken_vessel_id=second.vessel_id,
        )
    if second_overtakes_first and not first_overtakes_second:
        return EncounterClassification(
            EncounterType.OVERTAKING,
            prediction,
            course_difference,
            closing=True,
            overtaking_vessel_id=second.vessel_id,
            overtaken_vessel_id=first.vessel_id,
        )

    reciprocal = abs(180.0 - course_difference) <= cfg.head_on_course_tolerance_deg + _EPS
    both_ahead = _near_zero_bearing(
        prediction.first_relative_bearing_deg, cfg.head_on_bearing_tolerance_deg
    ) and _near_zero_bearing(
        prediction.second_relative_bearing_deg, cfg.head_on_bearing_tolerance_deg
    )
    if reciprocal and both_ahead:
        return EncounterClassification(
            EncounterType.HEAD_ON,
            prediction,
            course_difference,
            closing=True,
        )

    if course_difference <= cfg.parallel_course_tolerance_deg + _EPS:
        return EncounterClassification(
            EncounterType.PARALLEL,
            prediction,
            course_difference,
            closing=True,
        )

    return EncounterClassification(
        EncounterType.CROSSING,
        prediction,
        course_difference,
        closing=True,
    )
