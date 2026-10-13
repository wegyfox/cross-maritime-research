"""Self-contained future fleet egress and mission-metadata retirement.

A126 makes physical vessel removal symmetric with A125 ingress.  A vessel may leave the
shared environmental world through a scheduled membership removal or through A121 route-end
REMOVE.  In either case route/profile mission metadata must stop describing the removed vessel
at the same committed physical timestamp; otherwise a later reuse of the same vessel id can
inherit stale mission authority.

The module contains immutable admission/completion evidence only.  Calendar ownership remains
with A117/A123 and route-end physical removal remains with A121.  The top-level persistent fleet
session observes those already-committed removals and retires its runtime-owned mission metadata
without creating another registry observation or physical writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
from .vessel_classes import VesselClassProfile


class EnvironmentalFleetEgressError(RuntimeError):
    """Raised when a future fleet egress admission is inconsistent or stale."""


class EnvironmentalFleetEgressSource(str, Enum):
    """Physical writer that committed removal before mission metadata was retired."""

    MEMBERSHIP_REMOVAL = "membership_removal"
    ROUTE_END_REMOVE = "route_end_remove"


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetEgressRecord:
    """One future physical removal request owned by the persistent fleet runtime."""

    at: float
    vessel_id: str

    def __post_init__(self) -> None:
        if not isfinite(self.at):
            raise ValueError("egress time must be finite")
        if not self.vessel_id or not self.vessel_id.strip():
            raise ValueError("vessel_id must be non-empty")
        object.__setattr__(self, "vessel_id", self.vessel_id.strip())


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetEgressAdmission:
    """Audit evidence captured before a future membership removal is queued."""

    record: EnvironmentalFleetEgressRecord
    route_target_before: EnvironmentalRouteHandoffTarget | None
    profile_before: VesselClassProfile | None

    @property
    def vessel_id(self) -> str:
        return self.record.vessel_id

    @property
    def at(self) -> float:
        return self.record.at


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetEgressCompletion:
    """Audit evidence for mission metadata retired after physical removal committed."""

    vessel_id: str
    at: float
    source: EnvironmentalFleetEgressSource
    admission: EnvironmentalFleetEgressAdmission | None
    retired_route_target: EnvironmentalRouteHandoffTarget | None
    retired_profile: VesselClassProfile | None
    incarnation: int | None = None

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")
        if not isfinite(self.at):
            raise ValueError("completion time must be finite")
        if not isinstance(self.source, EnvironmentalFleetEgressSource):
            raise TypeError("source must be EnvironmentalFleetEgressSource")
        if self.incarnation is not None and (
            not isinstance(self.incarnation, int)
            or isinstance(self.incarnation, bool)
            or self.incarnation < 1
        ):
            raise ValueError("incarnation must be a positive integer or None")
