"""Read-only recorded vessel-track replay primitives.

The replay layer is deliberately outside ``World2D`` control semantics.  It stores
observed positions on the simulator's canonical east/north nautical-mile plane and
provides deterministic, piecewise-linear position interpolation between reports.

No extrapolation is performed, no maneuver command can be applied to a recorded
track, and optional reported course/speed fields are preserved as report metadata
rather than silently being treated as dynamically consistent truth.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from ..kinematics import Vector2, knots_to_nm_per_minute
from .dynamics import normalize_heading

_EPS = 1e-12


class RecordedTrackError(ValueError):
    """Base class for invalid recorded-track data or replay requests."""


class RecordedTrackTimeOutOfRange(RecordedTrackError):
    """Raised when replay is requested outside a track's recorded time span."""


@dataclass(frozen=True, slots=True)
class RecordedTrackPoint:
    """One immutable reported position in canonical next-gen units.

    ``reported_course_over_ground_deg`` and ``reported_speed_over_ground`` are
    optional source metadata.  They are not used to alter/interpolate ``position``.
    Speed, when present, is NM/min internally.
    """

    time: float
    position: Vector2
    reported_course_over_ground_deg: float | None = None
    reported_speed_over_ground: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.time):
            raise RecordedTrackError("recorded point time must be finite")
        if len(self.position) != 2 or not all(isfinite(value) for value in self.position):
            raise RecordedTrackError("recorded point position must contain two finite coordinates")
        if self.reported_course_over_ground_deg is not None:
            if not isfinite(self.reported_course_over_ground_deg):
                raise RecordedTrackError("reported course over ground must be finite")
            object.__setattr__(
                self,
                "reported_course_over_ground_deg",
                normalize_heading(self.reported_course_over_ground_deg),
            )
        if self.reported_speed_over_ground is not None:
            if (
                not isfinite(self.reported_speed_over_ground)
                or self.reported_speed_over_ground < 0.0
            ):
                raise RecordedTrackError(
                    "reported speed over ground must be finite and non-negative"
                )

    @classmethod
    def from_knots(
        cls,
        *,
        time: float,
        position: Vector2,
        reported_course_over_ground_deg: float | None = None,
        reported_speed_over_ground_knots: float | None = None,
    ) -> "RecordedTrackPoint":
        speed = (
            None
            if reported_speed_over_ground_knots is None
            else knots_to_nm_per_minute(reported_speed_over_ground_knots)
        )
        return cls(
            time=time,
            position=position,
            reported_course_over_ground_deg=reported_course_over_ground_deg,
            reported_speed_over_ground=speed,
        )


@dataclass(frozen=True, slots=True)
class RecordedTrackSample:
    """A deterministic replay sample.

    For an exact recorded timestamp, ``reported_point`` contains that original
    report.  For an interpolated timestamp it is ``None``; optional source COG/SOG
    are intentionally not fabricated by interpolation.
    """

    vessel_id: str
    time: float
    position: Vector2
    left_index: int
    right_index: int
    interpolation_fraction: float
    reported_point: RecordedTrackPoint | None

    @property
    def is_reported_point(self) -> bool:
        return self.reported_point is not None


@dataclass(frozen=True, slots=True)
class RecordedVesselTrack:
    """One vessel's strictly time-ordered recorded position history."""

    vessel_id: str
    points: tuple[RecordedTrackPoint, ...]

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise RecordedTrackError("vessel_id must be non-empty")
        if len(self.points) < 2:
            raise RecordedTrackError("recorded track requires at least two points")
        if not all(isinstance(point, RecordedTrackPoint) for point in self.points):
            raise TypeError("points must contain RecordedTrackPoint values")
        previous = self.points[0].time
        for point in self.points[1:]:
            if point.time <= previous:
                raise RecordedTrackError("recorded track times must be strictly increasing")
            previous = point.time

    @classmethod
    def from_points(
        cls,
        vessel_id: str,
        points: Iterable[RecordedTrackPoint],
    ) -> "RecordedVesselTrack":
        return cls(vessel_id=vessel_id, points=tuple(points))

    @property
    def start_time(self) -> float:
        return self.points[0].time

    @property
    def end_time(self) -> float:
        return self.points[-1].time

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def contains_time(self, time: float) -> bool:
        return self.start_time - _EPS <= time <= self.end_time + _EPS

    def sample(self, time: float) -> RecordedTrackSample:
        if not isfinite(time):
            raise RecordedTrackError("sample time must be finite")
        if time < self.start_time - _EPS or time > self.end_time + _EPS:
            raise RecordedTrackTimeOutOfRange(
                f"time {time!r} outside recorded span [{self.start_time!r}, {self.end_time!r}]"
            )

        # Clamp only floating-point epsilon overshoot at the closed endpoints.
        if time <= self.start_time:
            time = self.start_time
        elif time >= self.end_time:
            time = self.end_time

        times = tuple(point.time for point in self.points)
        index = bisect_left(times, time)
        if index < len(times) and abs(times[index] - time) <= _EPS:
            point = self.points[index]
            return RecordedTrackSample(
                vessel_id=self.vessel_id,
                time=point.time,
                position=point.position,
                left_index=index,
                right_index=index,
                interpolation_fraction=0.0,
                reported_point=point,
            )

        right_index = index
        left_index = right_index - 1
        left = self.points[left_index]
        right = self.points[right_index]
        fraction = (time - left.time) / (right.time - left.time)
        x = left.position[0] + fraction * (right.position[0] - left.position[0])
        y = left.position[1] + fraction * (right.position[1] - left.position[1])
        return RecordedTrackSample(
            vessel_id=self.vessel_id,
            time=time,
            position=(x, y),
            left_index=left_index,
            right_index=right_index,
            interpolation_fraction=fraction,
            reported_point=None,
        )


@dataclass(frozen=True, slots=True)
class RecordedTrafficReplay:
    """Immutable collection of independent recorded tracks.

    ``sample_active`` returns only vessels whose recorded time span contains the
    requested time.  Absence outside a track span is therefore explicit rather than
    extrapolated.
    """

    tracks: tuple[RecordedVesselTrack, ...]

    def __post_init__(self) -> None:
        if not self.tracks:
            raise RecordedTrackError("recorded traffic replay requires at least one track")
        if not all(isinstance(track, RecordedVesselTrack) for track in self.tracks):
            raise TypeError("tracks must contain RecordedVesselTrack values")
        ids = [track.vessel_id for track in self.tracks]
        if len(ids) != len(set(ids)):
            raise RecordedTrackError("duplicate recorded vessel_id")

    @classmethod
    def from_tracks(
        cls,
        tracks: Iterable[RecordedVesselTrack],
    ) -> "RecordedTrafficReplay":
        return cls(tuple(sorted(tuple(tracks), key=lambda track: track.vessel_id)))

    def track(self, vessel_id: str) -> RecordedVesselTrack:
        for track in self.tracks:
            if track.vessel_id == vessel_id:
                return track
        raise KeyError(vessel_id)

    def sample_active(self, time: float) -> tuple[RecordedTrackSample, ...]:
        if not isfinite(time):
            raise RecordedTrackError("sample time must be finite")
        return tuple(track.sample(time) for track in self.tracks if track.contains_time(time))
