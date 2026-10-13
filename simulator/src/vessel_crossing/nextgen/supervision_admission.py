"""Admission layer between encounter registry decisions and pair avoidance supervision.

The module intentionally does not select maneuvers.  It answers whether a tracked pair
is ready to be handed to a pair supervisor *without* creating two independent command
owners for the same vessel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .encounter_registry import EncounterRegistryEntry, EncounterRegistryUpdate, PairKey


class SupervisionAdmissionStatus(str, Enum):
    OBSERVE_ONLY = "observe_only"
    READY_SINGLE_PAIR = "ready_single_pair"
    BLOCKED_SHARED_VESSEL = "blocked_shared_vessel"


class SupervisionAdmissionReason(str, Enum):
    NO_ACTIVE_RESPONSIBILITY = "no_active_responsibility"
    ACTIVE_ISOLATED_RESPONSIBILITY = "active_isolated_responsibility"
    MULTI_ENCOUNTER_COMMAND_OWNERSHIP_CONFLICT = "multi_encounter_command_ownership_conflict"


@dataclass(frozen=True, slots=True)
class SupervisionAdmissionDecision:
    key: PairKey
    status: SupervisionAdmissionStatus
    reason: SupervisionAdmissionReason
    entry: EncounterRegistryEntry
    shared_vessel_ids: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status is SupervisionAdmissionStatus.READY_SINGLE_PAIR


@dataclass(frozen=True, slots=True)
class SupervisionAdmissionBatch:
    time_min: float
    decisions: tuple[SupervisionAdmissionDecision, ...]
    conflicted_vessel_ids: tuple[str, ...]

    @property
    def ready(self) -> tuple[SupervisionAdmissionDecision, ...]:
        return tuple(item for item in self.decisions if item.ready)

    @property
    def blocked(self) -> tuple[SupervisionAdmissionDecision, ...]:
        return tuple(
            item
            for item in self.decisions
            if item.status is SupervisionAdmissionStatus.BLOCKED_SHARED_VESSEL
        )

    @property
    def observe_only(self) -> tuple[SupervisionAdmissionDecision, ...]:
        return tuple(
            item
            for item in self.decisions
            if item.status is SupervisionAdmissionStatus.OBSERVE_ONLY
        )


def admit_supervision(update: EncounterRegistryUpdate) -> SupervisionAdmissionBatch:
    """Build deterministic pair-supervision admission from one registry update.

    Every pair with an effective responsibility scaffold is an *active* supervision
    candidate.  If a vessel belongs to more than one such active pair, every affected
    pair is blocked here.  Resolving a multi-encounter command conflict requires a
    future fleet/multi-encounter arbiter; pairwise supervisors must not race to own the
    same vessel command channel.
    """

    active_entries = [entry for entry in update.entries if entry.responsibility_active]
    active_degree: dict[str, int] = {}
    for entry in active_entries:
        for vessel_id in entry.key:
            active_degree[vessel_id] = active_degree.get(vessel_id, 0) + 1

    conflicted = tuple(sorted(v for v, degree in active_degree.items() if degree > 1))
    conflicted_set = set(conflicted)

    decisions: list[SupervisionAdmissionDecision] = []
    for entry in sorted(update.entries, key=lambda item: item.key):
        if not entry.responsibility_active:
            decisions.append(
                SupervisionAdmissionDecision(
                    entry.key,
                    SupervisionAdmissionStatus.OBSERVE_ONLY,
                    SupervisionAdmissionReason.NO_ACTIVE_RESPONSIBILITY,
                    entry,
                )
            )
            continue
        shared = tuple(v for v in entry.key if v in conflicted_set)
        if shared:
            decisions.append(
                SupervisionAdmissionDecision(
                    entry.key,
                    SupervisionAdmissionStatus.BLOCKED_SHARED_VESSEL,
                    SupervisionAdmissionReason.MULTI_ENCOUNTER_COMMAND_OWNERSHIP_CONFLICT,
                    entry,
                    shared,
                )
            )
        else:
            decisions.append(
                SupervisionAdmissionDecision(
                    entry.key,
                    SupervisionAdmissionStatus.READY_SINGLE_PAIR,
                    SupervisionAdmissionReason.ACTIVE_ISOLATED_RESPONSIBILITY,
                    entry,
                )
            )

    return SupervisionAdmissionBatch(update.time_min, tuple(decisions), conflicted)
