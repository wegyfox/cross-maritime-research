"""Generation-safe future membership episodes for reusable vessel identifiers.

A127 extends A125/A126 from one pending event per vessel ID to an explicit incarnation
sequence.  One immutable episode binds a future ingress record to its optional physical
departure time.  The persistent fleet runtime assigns a monotonically increasing
incarnation number and queues generation-tagged membership events on the shared A117
calendar.  A stale removal from an older incarnation can therefore never remove a newer
vessel that reuses the same external ``vessel_id``.

Mission metadata remains keyed by external vessel ID for compatibility with the route
and profile layers.  Only the earliest pending incarnation owns that live catalogue
entry; later episodes remain staged until the earlier incarnation physically retires.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .environmental_fleet_ingress import EnvironmentalFleetIngressRecord


class EnvironmentalFleetIncarnationError(RuntimeError):
    """Raised when a reusable-vessel incarnation episode is invalid or overlaps."""


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetIncarnationEpisode:
    """One future incarnation: ingress plus an optional later physical departure."""

    ingress: EnvironmentalFleetIngressRecord
    egress_at: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ingress, EnvironmentalFleetIngressRecord):
            raise TypeError("ingress must be EnvironmentalFleetIngressRecord")
        if self.egress_at is not None:
            if not isfinite(self.egress_at):
                raise ValueError("incarnation egress time must be finite")
            if self.egress_at <= self.ingress.at:
                raise EnvironmentalFleetIncarnationError(
                    "incarnation egress time must be strictly after ingress time"
                )

    @property
    def vessel_id(self) -> str:
        return self.ingress.vessel_id

    @property
    def ingress_at(self) -> float:
        return self.ingress.at


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetIncarnationAdmission:
    """Audit evidence for one generation assigned and queued by the persistent runtime."""

    episode: EnvironmentalFleetIncarnationEpisode
    incarnation: int
    metadata_activated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.episode, EnvironmentalFleetIncarnationEpisode):
            raise TypeError("episode must be EnvironmentalFleetIncarnationEpisode")
        if not isinstance(self.incarnation, int) or isinstance(self.incarnation, bool) or self.incarnation < 1:
            raise ValueError("incarnation must be a positive integer")

    @property
    def vessel_id(self) -> str:
        return self.episode.vessel_id

    @property
    def ingress_at(self) -> float:
        return self.episode.ingress_at

    @property
    def egress_at(self) -> float | None:
        return self.episode.egress_at


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetIncarnationCancellation:
    """Audit evidence for one not-yet-arrived incarnation removed from the future timeline."""

    vessel_id: str
    incarnation: int
    episode: EnvironmentalFleetIncarnationEpisode
    removed_event_count: int
    metadata_reassigned: bool

    def __post_init__(self) -> None:
        if not self.vessel_id or not self.vessel_id.strip():
            raise ValueError("vessel_id must be non-empty")
        if not isinstance(self.incarnation, int) or isinstance(self.incarnation, bool) or self.incarnation < 1:
            raise ValueError("incarnation must be a positive integer")
        if not isinstance(self.episode, EnvironmentalFleetIncarnationEpisode):
            raise TypeError("episode must be EnvironmentalFleetIncarnationEpisode")
        if self.episode.vessel_id != self.vessel_id:
            raise EnvironmentalFleetIncarnationError("cancellation vessel id disagrees with episode")
        if self.removed_event_count < 1:
            raise ValueError("removed_event_count must be positive")


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetIncarnationAmendment:
    """Audit evidence for one not-yet-arrived incarnation edited in place."""

    vessel_id: str
    incarnation: int
    previous_episode: EnvironmentalFleetIncarnationEpisode
    amended_episode: EnvironmentalFleetIncarnationEpisode
    metadata_republished: bool

    def __post_init__(self) -> None:
        if not self.vessel_id or not self.vessel_id.strip():
            raise ValueError("vessel_id must be non-empty")
        if not isinstance(self.incarnation, int) or isinstance(self.incarnation, bool) or self.incarnation < 1:
            raise ValueError("incarnation must be a positive integer")
        if not isinstance(self.previous_episode, EnvironmentalFleetIncarnationEpisode):
            raise TypeError("previous_episode must be EnvironmentalFleetIncarnationEpisode")
        if not isinstance(self.amended_episode, EnvironmentalFleetIncarnationEpisode):
            raise TypeError("amended_episode must be EnvironmentalFleetIncarnationEpisode")
        if self.previous_episode.vessel_id != self.vessel_id or self.amended_episode.vessel_id != self.vessel_id:
            raise EnvironmentalFleetIncarnationError("amendment vessel ids must agree")
