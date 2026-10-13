"""Neutral encounter classification for environment-aware in-sight geometry.

The a76 baseline names its directional model choice explicitly: ``HULL_ASPECT``.
Relative bearings, Rule-13 stern sector, head-on aspect and qualitative parallel/course
comparison use hull headings; closing/diverging and CPA use ground motion from a74.
Course over ground remains visible diagnostic evidence but does not replace visual
aspect in this baseline.

This remains a simplified geometric scaffold, not a claim of complete COLREG legal
classification under every environmental circumstance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dynamics import normalize_heading, signed_heading_delta
from .encounter import EncounterPrediction, absolute_bearing_deg
from .encounter_classification import (
    EncounterClassification,
    EncounterClassificationConfig,
    EncounterType,
)
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_evidence import (
    EnvironmentalPairMotionEvidence,
    predict_environmental_pair_motion,
)

_EPS = 1e-12


class EnvironmentalEncounterDirectionBasis(str, Enum):
    HULL_ASPECT = "hull_aspect"


@dataclass(frozen=True, slots=True)
class EnvironmentalEncounterClassification:
    classification: EncounterClassification
    motion_evidence: EnvironmentalPairMotionEvidence
    direction_basis: EnvironmentalEncounterDirectionBasis
    hull_heading_difference_deg: float
    ground_course_difference_deg: float | None

    @property
    def encounter_type(self) -> EncounterType:
        return self.classification.encounter_type


def _near_zero_bearing(relative_bearing: float, tolerance: float) -> bool:
    return relative_bearing <= tolerance + _EPS or relative_bearing >= 360.0 - tolerance - _EPS


def _in_overtaken_stern_sector(relative_bearing: float, abaft_beam: float) -> bool:
    start = 90.0 + abaft_beam
    end = 270.0 - abaft_beam
    return relative_bearing > start + _EPS and relative_bearing < end - _EPS


def _environmental_longitudinal_gain(motion, overtaker_id: str, overtaken_id: str) -> float:
    by_id = {motion.first.vessel_id: motion.first, motion.second.vessel_id: motion.second}
    overtaker = by_id[overtaker_id]
    overtaken = by_id[overtaken_id]
    from .dynamics import heading_unit_vector
    axis = heading_unit_vector(overtaken.hull_heading_deg)
    rel = (
        overtaker.ground_velocity[0] - overtaken.ground_velocity[0],
        overtaker.ground_velocity[1] - overtaken.ground_velocity[1],
    )
    return rel[0] * axis[0] + rel[1] * axis[1]


def _relative_bearing(
    observer: WaterRelativeVesselState,
    target: WaterRelativeVesselState,
) -> float:
    return normalize_heading(
        absolute_bearing_deg(observer.position, target.position)
        - observer.through_water_heading_deg
    )


def _ground_course_difference(motion: EnvironmentalPairMotionEvidence) -> float | None:
    first = motion.first.course_over_ground_deg
    second = motion.second.course_over_ground_deg
    if first is None or second is None:
        return None
    return abs(signed_heading_delta(first, second))


def classify_environmental_encounter(
    first: WaterRelativeVesselState,
    first_drift_velocity: tuple[float, float],
    second: WaterRelativeVesselState,
    second_drift_velocity: tuple[float, float],
    *,
    horizon: float | None = None,
    config: EncounterClassificationConfig | None = None,
    direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
) -> EnvironmentalEncounterClassification:
    """Classify neutral encounter geometry with explicit environmental frame semantics."""
    if direction_basis is not EnvironmentalEncounterDirectionBasis.HULL_ASPECT:
        raise ValueError("a76 supports only the explicit HULL_ASPECT direction basis")
    cfg = config or EncounterClassificationConfig()
    motion = predict_environmental_pair_motion(
        first,
        first_drift_velocity,
        second,
        second_drift_velocity,
        horizon=horizon,
    )
    prediction = EncounterPrediction(
        first_id=first.vessel_id,
        second_id=second.vessel_id,
        first_relative_bearing_deg=_relative_bearing(first, second),
        second_relative_bearing_deg=_relative_bearing(second, first),
        cpa=motion.cpa,
    )
    hull_difference = abs(
        signed_heading_delta(
            first.through_water_heading_deg,
            second.through_water_heading_deg,
        )
    )
    ground_difference = _ground_course_difference(motion)

    def wrap(
        encounter_type: EncounterType,
        *,
        closing: bool,
        overtaking_vessel_id: str | None = None,
        overtaken_vessel_id: str | None = None,
    ) -> EnvironmentalEncounterClassification:
        return EnvironmentalEncounterClassification(
            classification=EncounterClassification(
                encounter_type=encounter_type,
                prediction=prediction,
                course_difference_deg=hull_difference,
                closing=closing,
                overtaking_vessel_id=overtaking_vessel_id,
                overtaken_vessel_id=overtaken_vessel_id,
            ),
            motion_evidence=motion,
            direction_basis=direction_basis,
            hull_heading_difference_deg=hull_difference,
            ground_course_difference_deg=ground_difference,
        )

    if prediction.cpa.current_distance <= _EPS:
        return wrap(EncounterType.UNDEFINED, closing=False)
    if prediction.cpa.relative_speed <= _EPS:
        return wrap(EncounterType.NO_RELATIVE_MOTION, closing=False)

    closing = prediction.cpa.raw_tcpa > _EPS
    if not closing:
        return wrap(EncounterType.DIVERGING, closing=False)

    first_overtakes_second = _in_overtaken_stern_sector(
        prediction.second_relative_bearing_deg,
        cfg.overtaking_abaft_beam_deg,
    )
    second_overtakes_first = _in_overtaken_stern_sector(
        prediction.first_relative_bearing_deg,
        cfg.overtaking_abaft_beam_deg,
    )
    if first_overtakes_second and not second_overtakes_first:
        if (
            cfg.require_overtaking_longitudinal_gain
            and _environmental_longitudinal_gain(motion, first.vessel_id, second.vessel_id) <= _EPS
        ):
            first_overtakes_second = False
    if second_overtakes_first and not first_overtakes_second:
        if (
            cfg.require_overtaking_longitudinal_gain
            and _environmental_longitudinal_gain(motion, second.vessel_id, first.vessel_id) <= _EPS
        ):
            second_overtakes_first = False
    if first_overtakes_second and not second_overtakes_first:
        return wrap(
            EncounterType.OVERTAKING,
            closing=True,
            overtaking_vessel_id=first.vessel_id,
            overtaken_vessel_id=second.vessel_id,
        )
    if second_overtakes_first and not first_overtakes_second:
        return wrap(
            EncounterType.OVERTAKING,
            closing=True,
            overtaking_vessel_id=second.vessel_id,
            overtaken_vessel_id=first.vessel_id,
        )

    reciprocal = abs(180.0 - hull_difference) <= cfg.head_on_course_tolerance_deg + _EPS
    both_ahead = _near_zero_bearing(
        prediction.first_relative_bearing_deg,
        cfg.head_on_bearing_tolerance_deg,
    ) and _near_zero_bearing(
        prediction.second_relative_bearing_deg,
        cfg.head_on_bearing_tolerance_deg,
    )
    if reciprocal and both_ahead:
        return wrap(EncounterType.HEAD_ON, closing=True)

    if hull_difference <= cfg.parallel_course_tolerance_deg + _EPS:
        return wrap(EncounterType.PARALLEL, closing=True)

    return wrap(EncounterType.CROSSING, closing=True)
