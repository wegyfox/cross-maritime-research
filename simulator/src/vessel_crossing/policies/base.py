"""Policy-facing immutable observations and actions.

Policies deliberately receive no reference to ``Simulation``.  They can see a
read-only snapshot and return declarative speed commands.  This prevents policy
logic from mutating the event queue, vessel dictionaries or clock directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from ..safety import EncounterAssessment
from ..vessel import Vessel, VesselStatus


@dataclass(frozen=True, slots=True)
class EncounterObservation:
    """One unordered active-vessel pair and its safety assessment."""

    first_vessel_id: str
    second_vessel_id: str
    assessment: EncounterAssessment

    def __post_init__(self) -> None:
        if not self.first_vessel_id or not self.second_vessel_id:
            raise ValueError("encounter vessel IDs must be non-empty")
        if self.first_vessel_id == self.second_vessel_id:
            raise ValueError("encounter requires two distinct vessels")

    @property
    def vessel_ids(self) -> frozenset[str]:
        return frozenset((self.first_vessel_id, self.second_vessel_id))


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """Read-only state visible to one policy evaluation."""

    time: float
    vessels: tuple[Vessel, ...]
    encounters: tuple[EncounterObservation, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.time) or self.time < 0:
            raise ValueError("snapshot time must be finite and non-negative")

        ids = [v.vessel_id for v in self.vessels]
        if len(ids) != len(set(ids)):
            raise ValueError("snapshot vessel IDs must be unique")
        if any(v.status is not VesselStatus.ACTIVE for v in self.vessels):
            raise ValueError("policy snapshots may contain only active vessels")

        known = set(ids)
        seen_pairs: set[frozenset[str]] = set()
        for encounter in self.encounters:
            if not encounter.vessel_ids <= known:
                raise ValueError("encounter references vessel absent from snapshot")
            if encounter.vessel_ids in seen_pairs:
                raise ValueError("duplicate encounter pair in snapshot")
            seen_pairs.add(encounter.vessel_ids)

    def vessel(self, vessel_id: str) -> Vessel:
        for vessel in self.vessels:
            if vessel.vessel_id == vessel_id:
                return vessel
        raise KeyError(vessel_id)

    @property
    def dangerous_encounters(self) -> tuple[EncounterObservation, ...]:
        return tuple(e for e in self.encounters if e.assessment.dangerous)

    def encounters_for(self, vessel_id: str) -> tuple[EncounterObservation, ...]:
        self.vessel(vessel_id)  # explicit unknown-ID validation
        return tuple(e for e in self.encounters if vessel_id in e.vessel_ids)


@dataclass(frozen=True, slots=True)
class SpeedCommand:
    """Declarative request returned by a policy.

    Validation against current vessel existence and desired speed belongs to
    ``validate_speed_commands`` because those facts live in the snapshot.
    """

    vessel_id: str
    target_speed: float
    reason: str

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("command vessel_id must be non-empty")
        if not isfinite(self.target_speed) or self.target_speed < 0:
            raise ValueError("command target_speed must be finite and non-negative")
        if not self.reason.strip():
            raise ValueError("command reason must be non-empty")


@runtime_checkable
class ControlPolicy(Protocol):
    """Interface implemented by every crossing-control policy."""

    name: str

    def decide(self, snapshot: PolicySnapshot) -> tuple[SpeedCommand, ...]:
        """Return declarative commands for the supplied immutable snapshot."""
        ...


def validate_speed_commands(
    snapshot: PolicySnapshot,
    commands: tuple[SpeedCommand, ...],
) -> tuple[SpeedCommand, ...]:
    """Validate one policy decision without mutating simulation state."""

    seen: set[str] = set()
    for command in commands:
        if command.vessel_id in seen:
            raise ValueError("policy returned multiple commands for one vessel")
        seen.add(command.vessel_id)

        try:
            vessel = snapshot.vessel(command.vessel_id)
        except KeyError as exc:
            raise ValueError("policy command references unknown active vessel") from exc

        if command.target_speed - vessel.desired_speed > 1e-12:
            raise ValueError("policy command exceeds vessel desired speed")

    return commands


@dataclass(frozen=True, slots=True)
class NoControlPolicy:
    """Reference policy that never changes vessel speed."""

    name: str = "no_control"

    def decide(self, snapshot: PolicySnapshot) -> tuple[SpeedCommand, ...]:
        del snapshot
        return ()
