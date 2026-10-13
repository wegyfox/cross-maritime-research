"""Connected components of simultaneously active encounter responsibilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .encounter_registry import PairKey
from .supervision_admission import (
    SupervisionAdmissionBatch,
    SupervisionAdmissionDecision,
    SupervisionAdmissionStatus,
)


class EncounterComponentKind(str, Enum):
    ISOLATED_PAIR = "isolated_pair"
    MULTI_VESSEL = "multi_vessel"


@dataclass(frozen=True, slots=True)
class ActiveEncounterComponent:
    vessel_ids: tuple[str, ...]
    pair_keys: tuple[PairKey, ...]
    decisions: tuple[SupervisionAdmissionDecision, ...]
    kind: EncounterComponentKind

    @property
    def component_id(self) -> str:
        return "|".join(self.vessel_ids)

    @property
    def ready_for_pair_supervisor(self) -> bool:
        return (
            self.kind is EncounterComponentKind.ISOLATED_PAIR
            and len(self.decisions) == 1
            and self.decisions[0].status is SupervisionAdmissionStatus.READY_SINGLE_PAIR
        )


@dataclass(frozen=True, slots=True)
class EncounterComponentBatch:
    time_min: float
    components: tuple[ActiveEncounterComponent, ...]
    observation_only: tuple[SupervisionAdmissionDecision, ...]

    @property
    def isolated_pairs(self) -> tuple[ActiveEncounterComponent, ...]:
        return tuple(c for c in self.components if c.kind is EncounterComponentKind.ISOLATED_PAIR)

    @property
    def multi_vessel(self) -> tuple[ActiveEncounterComponent, ...]:
        return tuple(c for c in self.components if c.kind is EncounterComponentKind.MULTI_VESSEL)


def build_active_encounter_components(
    admission: SupervisionAdmissionBatch,
) -> EncounterComponentBatch:
    """Partition active responsibility pairs into deterministic connected subfleets."""

    active = [d for d in admission.decisions if d.status is not SupervisionAdmissionStatus.OBSERVE_ONLY]
    if not active:
        return EncounterComponentBatch(admission.time_min, (), admission.observe_only)

    adjacency: dict[str, set[str]] = {}
    by_pair: dict[PairKey, SupervisionAdmissionDecision] = {}
    for decision in active:
        a, b = decision.key
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
        by_pair[decision.key] = decision

    visited: set[str] = set()
    components: list[ActiveEncounterComponent] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack = [start]
        vessels: set[str] = set()
        while stack:
            vessel = stack.pop()
            if vessel in visited:
                continue
            visited.add(vessel)
            vessels.add(vessel)
            for neighbor in sorted(adjacency[vessel], reverse=True):
                if neighbor not in visited:
                    stack.append(neighbor)
        vessel_ids = tuple(sorted(vessels))
        pair_keys = tuple(
            sorted(key for key in by_pair if key[0] in vessels and key[1] in vessels)
        )
        decisions = tuple(by_pair[key] for key in pair_keys)
        kind = (
            EncounterComponentKind.ISOLATED_PAIR
            if len(vessel_ids) == 2 and len(pair_keys) == 1
            else EncounterComponentKind.MULTI_VESSEL
        )
        components.append(ActiveEncounterComponent(vessel_ids, pair_keys, decisions, kind))

    components.sort(key=lambda component: component.vessel_ids)
    return EncounterComponentBatch(admission.time_min, tuple(components), admission.observe_only)
