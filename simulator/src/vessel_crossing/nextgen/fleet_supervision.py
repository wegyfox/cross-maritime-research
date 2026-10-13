"""Fleet-level command ownership dispatch for active encounter components.

This layer deliberately does **not** advance ``World2D``.  Running independent pair or
joint supervisors sequentially would let the first controller advance the shared world
clock past review events belonging to another independent component.  A31 therefore
establishes one unambiguous command owner for every vessel at a single registry event
boundary and records which execution backend the component requires.  A later shared
review-clock layer can execute all owners concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .encounter_components import (
    ActiveEncounterComponent,
    EncounterComponentBatch,
    EncounterComponentKind,
    build_active_encounter_components,
)
from .encounter_registry import EncounterRegistryUpdate, PairKey
from .supervision_admission import SupervisionAdmissionDecision, admit_supervision


class FleetSupervisionError(RuntimeError):
    """Raised when a fleet dispatch cannot establish unique command ownership."""


class FleetSupervisorBackend(str, Enum):
    """Execution backend required by one connected active component."""

    PAIR = "pair"
    JOINT = "joint"


@dataclass(frozen=True, slots=True)
class FleetCommandOwner:
    """One component-level owner of every command channel in ``vessel_ids``."""

    component_id: str
    backend: FleetSupervisorBackend
    vessel_ids: tuple[str, ...]
    pair_keys: tuple[PairKey, ...]
    component: ActiveEncounterComponent

    def __post_init__(self) -> None:
        if self.component_id != self.component.component_id:
            raise ValueError("component_id does not match component")
        if self.vessel_ids != self.component.vessel_ids:
            raise ValueError("vessel_ids do not match component")
        if self.pair_keys != self.component.pair_keys:
            raise ValueError("pair_keys do not match component")
        expected = (
            FleetSupervisorBackend.PAIR
            if self.component.kind is EncounterComponentKind.ISOLATED_PAIR
            else FleetSupervisorBackend.JOINT
        )
        if self.backend is not expected:
            raise ValueError("backend does not match component kind")
        if self.backend is FleetSupervisorBackend.PAIR and not self.component.ready_for_pair_supervisor:
            raise ValueError("pair backend requires a component admitted for pair supervision")


@dataclass(frozen=True, slots=True)
class FleetSupervisionDispatch:
    """Immutable command-ownership decision for one encounter registry timestamp."""

    time_min: float
    owners: tuple[FleetCommandOwner, ...]
    observation_only: tuple[SupervisionAdmissionDecision, ...]
    components: EncounterComponentBatch

    @property
    def pair_owners(self) -> tuple[FleetCommandOwner, ...]:
        return tuple(owner for owner in self.owners if owner.backend is FleetSupervisorBackend.PAIR)

    @property
    def joint_owners(self) -> tuple[FleetCommandOwner, ...]:
        return tuple(owner for owner in self.owners if owner.backend is FleetSupervisorBackend.JOINT)

    @property
    def owned_vessel_ids(self) -> tuple[str, ...]:
        return tuple(sorted(vessel_id for owner in self.owners for vessel_id in owner.vessel_ids))

    def owner_for(self, vessel_id: str) -> FleetCommandOwner:
        matches = [owner for owner in self.owners if vessel_id in owner.vessel_ids]
        if not matches:
            raise KeyError(vessel_id)
        if len(matches) != 1:
            raise FleetSupervisionError(f"vessel {vessel_id!r} has multiple command owners")
        return matches[0]


def build_fleet_supervision_dispatch(update: EncounterRegistryUpdate) -> FleetSupervisionDispatch:
    """Route active connected components to pair vs joint command ownership.

    The returned dispatch is valid only for ``update.time_min``.  It is a topology and
    ownership decision, not an execution loop.  Connected components are disjoint by
    construction, and this function independently verifies that no vessel is assigned
    to two owners before returning.
    """

    components = build_active_encounter_components(admit_supervision(update))
    owners: list[FleetCommandOwner] = []
    claimed: dict[str, str] = {}

    for component in components.components:
        backend = (
            FleetSupervisorBackend.PAIR
            if component.kind is EncounterComponentKind.ISOLATED_PAIR
            else FleetSupervisorBackend.JOINT
        )
        owner = FleetCommandOwner(
            component_id=component.component_id,
            backend=backend,
            vessel_ids=component.vessel_ids,
            pair_keys=component.pair_keys,
            component=component,
        )
        for vessel_id in owner.vessel_ids:
            previous = claimed.get(vessel_id)
            if previous is not None:
                raise FleetSupervisionError(
                    f"vessel {vessel_id!r} would be owned by both {previous!r} and {owner.component_id!r}"
                )
            claimed[vessel_id] = owner.component_id
        owners.append(owner)

    return FleetSupervisionDispatch(
        time_min=update.time_min,
        owners=tuple(owners),
        observation_only=components.observation_only,
        components=components,
    )
