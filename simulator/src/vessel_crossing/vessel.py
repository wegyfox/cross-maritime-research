"""Immutable vessel state for the modern simulation core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .geometry import Route
from .kinematics import MotionState, SpeedLimits, integrate_speed_target


class VesselStatus(str, Enum):
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class Vessel:
    """Controlled state of one vessel on one fixed route.

    ``desired_speed`` is the vessel's unconstrained cruise preference.
    ``target_speed`` is the command requested by its own control policy.  The
    simulation may apply a lower *effective* target when a same-route leader
    constrains the vessel.  Keeping these concepts separate lets a temporary
    queue constraint disappear without losing the vessel's own command.
    """

    vessel_id: str
    route: Route
    arrival_time: float
    desired_speed: float
    current_speed: float = 0.0
    progress: float = 0.0
    status: VesselStatus = VesselStatus.SCHEDULED
    target_speed: float | None = None
    activated_at: float | None = None
    completed_at: float | None = None

    _EPSILON = 1e-12

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")
        if self.desired_speed < 0 or self.current_speed < 0:
            raise ValueError("speeds must be non-negative")

        # ``None`` means "start with the normal cruise command".  Normalizing
        # inside the frozen dataclass preserves backwards-compatible
        # construction while ensuring runtime state always carries a command.
        if self.target_speed is None:
            object.__setattr__(self, "target_speed", self.desired_speed)
        elif self.target_speed < 0:
            raise ValueError("target_speed must be non-negative")
        elif self.target_speed - self.desired_speed > self._EPSILON:
            raise ValueError("target_speed cannot exceed desired_speed")

        if self.progress < 0 or self.progress > self.route.length:
            raise ValueError("progress must lie on the route")
        if self.activated_at is not None:
            if self.activated_at < self.arrival_time - self._EPSILON:
                raise ValueError("activated_at cannot precede arrival_time")
        if self.completed_at is not None:
            if self.activated_at is None:
                raise ValueError("completed_at requires activated_at")
            if self.completed_at < self.activated_at - self._EPSILON:
                raise ValueError("completed_at cannot precede activated_at")

        if self.status is VesselStatus.SCHEDULED:
            if self.activated_at is not None or self.completed_at is not None:
                raise ValueError("scheduled vessel cannot have lifecycle timestamps")
        elif self.status is VesselStatus.WAITING:
            if self.activated_at is not None or self.completed_at is not None:
                raise ValueError("waiting vessel cannot have lifecycle timestamps")
        elif self.status is VesselStatus.ACTIVE:
            if self.completed_at is not None:
                raise ValueError("active vessel cannot have completed_at")
        if self.status is VesselStatus.COMPLETED and self.progress != self.route.length:
            raise ValueError("completed vessel must be at route end")

    @property
    def position(self) -> tuple[float, float]:
        ux, uy = self.route.unit_direction
        return (
            self.route.start[0] + ux * self.progress,
            self.route.start[1] + uy * self.progress,
        )

    @property
    def motion_state(self) -> MotionState:
        return MotionState(self.position, self.route.velocity(self.current_speed))

    def activate(self, *, at: float) -> "Vessel":
        if self.status is VesselStatus.COMPLETED:
            raise ValueError("completed vessel cannot be activated")
        if at < self.arrival_time - self._EPSILON:
            raise ValueError("activation cannot precede arrival")
        activated_at = self.activated_at if self.activated_at is not None else at
        return replace(
            self,
            status=VesselStatus.ACTIVE,
            activated_at=activated_at,
            completed_at=None,
        )

    def with_target_speed(self, target_speed: float) -> "Vessel":
        """Return a state with a new own-policy speed command."""
        if target_speed < 0:
            raise ValueError("target_speed must be non-negative")
        if target_speed - self.desired_speed > self._EPSILON:
            raise ValueError("target_speed cannot exceed desired_speed")
        return replace(self, target_speed=target_speed)


def advance_vessel(
    vessel: Vessel,
    *,
    target_speed: float,
    duration: float,
    limits: SpeedLimits,
) -> Vessel:
    """Advance an active vessel under a bounded longitudinal speed command."""

    if vessel.status is not VesselStatus.ACTIVE:
        raise ValueError("only active vessels can advance")

    transition = integrate_speed_target(
        vessel.current_speed,
        target_speed,
        duration,
        limits,
    )
    new_progress = vessel.progress + transition.distance

    if new_progress >= vessel.route.length:
        return replace(
            vessel,
            progress=vessel.route.length,
            current_speed=0.0,
            status=VesselStatus.COMPLETED,
            completed_at=None,
        )

    return replace(
        vessel,
        progress=new_progress,
        current_speed=transition.final_speed,
    )
