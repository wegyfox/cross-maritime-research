"""Self-contained future vessel/mission admission for the persistent fleet runtime.

A125 binds a future environmental vessel arrival to the mission metadata that later
avoidance/route ownership requires.  Earlier runtime layers accepted a raw membership
arrival while the caller separately mutated the shared route-target catalogue.  That
split could make the vessel visible first and only fail later when a fresh PAIR/JOINT
owner attempted to plan or when the vessel was released to nominal route control.

The ingress record is deliberately immutable.  High-level runtime admission validates
its vessel/target/profile identity and world metadata before queuing the membership
boundary, then publishes the already-validated mission metadata into the runtime-owned
catalogues.  The existing A117 membership calendar remains the only physical membership
writer and A123 remains the only idle/bootstrap owner writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..kinematics import Vector2
from .dynamics import ManeuverLimits
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
from .vessel_classes import VesselClassProfile


class EnvironmentalFleetIngressError(RuntimeError):
    """Raised when a future vessel/mission admission is inconsistent or stale."""


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetIngressRecord:
    """One future vessel arrival plus the mission metadata required after admission."""

    at: float
    vessel: WaterRelativeVesselState
    route_target: EnvironmentalRouteHandoffTarget
    limits: ManeuverLimits | None = None
    environmental_drift_velocity: Vector2 | None = None
    profile: VesselClassProfile | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.at):
            raise ValueError("ingress time must be finite")
        if not isinstance(self.vessel, WaterRelativeVesselState):
            raise TypeError("vessel must be WaterRelativeVesselState")
        if not isinstance(self.route_target, EnvironmentalRouteHandoffTarget):
            raise TypeError("route_target must be EnvironmentalRouteHandoffTarget")
        if self.route_target.vessel_id != self.vessel.vessel_id:
            raise EnvironmentalFleetIngressError(
                f"route target vessel id {self.route_target.vessel_id!r} disagrees with "
                f"ingress vessel id {self.vessel.vessel_id!r}"
            )
        if self.limits is not None and not isinstance(self.limits, ManeuverLimits):
            raise TypeError("limits must be ManeuverLimits or None")
        if self.profile is not None and not isinstance(self.profile, VesselClassProfile):
            raise TypeError("profile must be VesselClassProfile or None")
        if self.profile is not None and self.limits is not None and self.profile.maneuver_limits != self.limits:
            raise EnvironmentalFleetIngressError(
                "explicit ingress limits disagree with the supplied vessel profile"
            )
        if self.environmental_drift_velocity is not None:
            if len(self.environmental_drift_velocity) != 2:
                raise ValueError("environmental_drift_velocity must contain exactly two components")
            if not all(isfinite(float(value)) for value in self.environmental_drift_velocity):
                raise ValueError("environmental_drift_velocity components must be finite")

    @property
    def vessel_id(self) -> str:
        return self.vessel.vessel_id

    @property
    def effective_limits(self) -> ManeuverLimits | None:
        if self.limits is not None:
            return self.limits
        if self.profile is not None:
            return self.profile.maneuver_limits
        return None


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetIngressAdmission:
    """Audit evidence that one complete future ingress record was accepted."""

    record: EnvironmentalFleetIngressRecord
    route_target_replaced: bool
    profile_assigned: bool

    @property
    def vessel_id(self) -> str:
        return self.record.vessel_id

    @property
    def at(self) -> float:
        return self.record.at
