"""Neutral encounter evidence built from read-only recorded vessel tracks.

This adapter intentionally stops before encounter classification or COLREG responsibility.
Range/compass bearing come from recorded positions; local CPA/TCPA comes from the a69
explicit one-sided interval ground-motion evidence.  The two are bundled for later
analysis without constructing a commandable ``Vessel2DState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from .encounter import absolute_bearing_deg
from .observation import PairObservation
from .recorded_track import RecordedTrackError, RecordedVesselTrack
from .recorded_track_kinematics import (
    RecordedPairCpaEvidence,
    RecordedTrackTimeSide,
    predict_recorded_pair_cpa,
)


@dataclass(frozen=True, slots=True)
class RecordedPairEncounterEvidence:
    time: float
    observation: PairObservation
    cpa_evidence: RecordedPairCpaEvidence

    def __post_init__(self) -> None:
        if not isfinite(self.time):
            raise RecordedTrackError("recorded encounter evidence time must be finite")
        if abs(self.observation.time_min - self.time) > 1e-12:
            raise RecordedTrackError("observation timestamp does not match evidence time")
        if abs(self.cpa_evidence.time - self.time) > 1e-12:
            raise RecordedTrackError("CPA timestamp does not match evidence time")
        if self.observation.first_id != self.cpa_evidence.first.vessel_id:
            raise RecordedTrackError("first vessel identity mismatch in evidence bundle")
        if self.observation.second_id != self.cpa_evidence.second.vessel_id:
            raise RecordedTrackError("second vessel identity mismatch in evidence bundle")


def observe_recorded_pair(
    first_track: RecordedVesselTrack,
    second_track: RecordedVesselTrack,
    *,
    time: float,
) -> PairObservation:
    """Return neutral range/compass-bearing evidence from recorded positions only."""
    if first_track.vessel_id == second_track.vessel_id:
        raise RecordedTrackError("recorded pair requires two distinct vessel ids")
    first = first_track.sample(time)
    second = second_track.sample(time)
    dx = second.position[0] - first.position[0]
    dy = second.position[1] - first.position[1]
    return PairObservation(
        time_min=time,
        first_id=first_track.vessel_id,
        second_id=second_track.vessel_id,
        range_nm=hypot(dx, dy),
        first_to_second_bearing_deg=absolute_bearing_deg(first.position, second.position),
        second_to_first_bearing_deg=absolute_bearing_deg(second.position, first.position),
    )


def build_recorded_pair_encounter_evidence(
    first_track: RecordedVesselTrack,
    second_track: RecordedVesselTrack,
    *,
    time: float,
    first_side: RecordedTrackTimeSide,
    second_side: RecordedTrackTimeSide,
    horizon: float | None = None,
) -> RecordedPairEncounterEvidence:
    """Bundle neutral recorded observation and explicit local CPA evidence.

    This function does not classify the encounter, assign responsibilities, update an
    ``EncounterRegistry`` or feed a controller.  Those are separate future assimilation
    decisions and must not happen merely because replay evidence exists.
    """
    observation = observe_recorded_pair(first_track, second_track, time=time)
    cpa_evidence = predict_recorded_pair_cpa(
        first_track,
        second_track,
        time=time,
        first_side=first_side,
        second_side=second_side,
        horizon=horizon,
    )
    return RecordedPairEncounterEvidence(
        time=time,
        observation=observation,
        cpa_evidence=cpa_evidence,
    )
