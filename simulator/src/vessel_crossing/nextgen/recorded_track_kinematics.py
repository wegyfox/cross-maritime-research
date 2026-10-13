"""Explicit interval-derived kinematics for read-only recorded tracks.

A recorded position history is piecewise linear under the a68 replay contract.  That
makes ground velocity constant *inside each report interval*, but not uniquely defined
at an interior report timestamp where the incoming and outgoing segment slopes may
differ.  This module requires callers to name ``LEFT`` or ``RIGHT`` at every query, so
there is no hidden corner convention.

The resulting motion evidence is read-only and uses ground velocity derived from
positions.  Optional reported COG/SOG metadata does not override it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, degrees, hypot

from ..cpa import ClosestApproach, closest_approach
from ..kinematics import MotionState, Vector2
from .dynamics import normalize_heading
from .recorded_track import (
    RecordedTrackError,
    RecordedTrackTimeOutOfRange,
    RecordedVesselTrack,
)

_EPS = 1e-12


class RecordedTrackTimeSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class RecordedKinematicsUnavailable(RecordedTrackError):
    """Requested one-sided kinematics do not exist at the track boundary."""


@dataclass(frozen=True, slots=True)
class RecordedKinematicInterval:
    vessel_id: str
    interval_index: int
    start_time: float
    end_time: float
    start_position: Vector2
    end_position: Vector2
    ground_velocity: Vector2
    speed_over_ground: float
    course_over_ground_deg: float | None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_motion_state(self, *, position: Vector2) -> MotionState:
        return MotionState(position=position, velocity=self.ground_velocity)


@dataclass(frozen=True, slots=True)
class RecordedGroundMotionSample:
    vessel_id: str
    time: float
    position: Vector2
    side: RecordedTrackTimeSide
    interval: RecordedKinematicInterval

    @property
    def ground_velocity(self) -> Vector2:
        return self.interval.ground_velocity

    @property
    def speed_over_ground(self) -> float:
        return self.interval.speed_over_ground

    @property
    def course_over_ground_deg(self) -> float | None:
        return self.interval.course_over_ground_deg

    def to_motion_state(self) -> MotionState:
        return self.interval.to_motion_state(position=self.position)


@dataclass(frozen=True, slots=True)
class RecordedPairCpaEvidence:
    time: float
    first: RecordedGroundMotionSample
    second: RecordedGroundMotionSample
    cpa: ClosestApproach


def recorded_kinematic_interval(
    track: RecordedVesselTrack,
    interval_index: int,
) -> RecordedKinematicInterval:
    if not isinstance(track, RecordedVesselTrack):
        raise TypeError("track must be a RecordedVesselTrack")
    if interval_index < 0 or interval_index >= len(track.points) - 1:
        raise RecordedKinematicsUnavailable("recorded interval index out of range")
    start = track.points[interval_index]
    end = track.points[interval_index + 1]
    duration = end.time - start.time
    vx = (end.position[0] - start.position[0]) / duration
    vy = (end.position[1] - start.position[1]) / duration
    speed = hypot(vx, vy)
    course = None if speed <= _EPS else normalize_heading(degrees(atan2(vx, vy)))
    return RecordedKinematicInterval(
        vessel_id=track.vessel_id,
        interval_index=interval_index,
        start_time=start.time,
        end_time=end.time,
        start_position=start.position,
        end_position=end.position,
        ground_velocity=(vx, vy),
        speed_over_ground=speed,
        course_over_ground_deg=course,
    )


def recorded_ground_motion_at(
    track: RecordedVesselTrack,
    time: float,
    *,
    side: RecordedTrackTimeSide,
) -> RecordedGroundMotionSample:
    """Return piecewise-linear ground-motion evidence at ``time``.

    ``side`` is mandatory even away from a report corner.  At an exact interior report
    timestamp it selects the incoming (LEFT) or outgoing (RIGHT) interval.  The
    unavailable outward side at the first/last report fails closed.
    """
    if not isinstance(side, RecordedTrackTimeSide):
        raise TypeError("side must be a RecordedTrackTimeSide")
    sample = track.sample(time)  # owns finite/range validation and interpolation

    if not sample.is_reported_point:
        interval_index = sample.left_index
    else:
        point_index = sample.left_index
        if side is RecordedTrackTimeSide.LEFT:
            if point_index == 0:
                raise RecordedKinematicsUnavailable(
                    "LEFT kinematics unavailable at first recorded timestamp"
                )
            interval_index = point_index - 1
        else:
            if point_index == len(track.points) - 1:
                raise RecordedKinematicsUnavailable(
                    "RIGHT kinematics unavailable at last recorded timestamp"
                )
            interval_index = point_index

    interval = recorded_kinematic_interval(track, interval_index)
    return RecordedGroundMotionSample(
        vessel_id=track.vessel_id,
        time=sample.time,
        position=sample.position,
        side=side,
        interval=interval,
    )


def predict_recorded_pair_cpa(
    first_track: RecordedVesselTrack,
    second_track: RecordedVesselTrack,
    *,
    time: float,
    first_side: RecordedTrackTimeSide,
    second_side: RecordedTrackTimeSide,
    horizon: float | None = None,
) -> RecordedPairCpaEvidence:
    """Apply the validated constant-velocity CPA kernel to explicit replay evidence.

    The CPA is a local projection from the selected recorded intervals.  It does not
    assert that the recorded vessels continue those velocities beyond their evidence,
    and it does not create COLREG responsibility or a controlled ``World2D`` state.
    """
    first = recorded_ground_motion_at(first_track, time, side=first_side)
    second = recorded_ground_motion_at(second_track, time, side=second_side)
    cpa = closest_approach(
        first.to_motion_state(),
        second.to_motion_state(),
        horizon=horizon,
    )
    return RecordedPairCpaEvidence(time=time, first=first, second=second, cpa=cpa)
