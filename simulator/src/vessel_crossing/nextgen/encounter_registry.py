"""Event-boundary encounter discovery and stateful pair lifecycle registry.

Discovery is deliberately broader than the Rule-7 risk gate.  It answers only whether
an active vessel pair is worth tracking at this event boundary; the associated
``EncounterLifecycle`` remains responsible for classification, risk evidence and
responsibility persistence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from math import hypot, isfinite
from typing import Callable, Iterable

from .dynamics import Vessel2DState
from .encounter import EncounterPrediction, predict_encounter
from .lifecycle import (
    EncounterLifecycle,
    EncounterLifecycleConfig,
    EncounterLifecycleDecision,
    EncounterLifecycleEvidence,
    build_state_lifecycle_evidence,
)
from .responsibility import ResponsibilityContext
from .world import World2D
from .ship_domain import ShipDomain
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


PairKey = tuple[str, str]


def canonical_pair_key(first_id: str, second_id: str) -> PairKey:
    if not first_id or not second_id or first_id == second_id:
        raise ValueError("pair requires two distinct non-empty vessel ids")
    return tuple(sorted((first_id, second_id)))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class EncounterDiscoveryConfig:
    """Broad deterministic candidate-pair screen; values are model parameters."""

    current_distance_nm: float
    dcpa_distance_nm: float
    tcpa_horizon_min: float

    def __post_init__(self) -> None:
        if self.current_distance_nm <= 0.0:
            raise ValueError("current_distance_nm must be positive")
        if self.dcpa_distance_nm <= 0.0:
            raise ValueError("dcpa_distance_nm must be positive")
        if self.tcpa_horizon_min <= 0.0:
            raise ValueError("tcpa_horizon_min must be positive")


@dataclass(frozen=True, slots=True)
class PairDiscovery:
    key: PairKey
    prediction: EncounterPrediction
    current_range_nm: float
    inside_distance_window: bool
    future_cpa_in_window: bool
    effective_current_distance_nm: float | None = None
    effective_dcpa_distance_nm: float | None = None

    @property
    def discovered(self) -> bool:
        return self.inside_distance_window or self.future_cpa_in_window


@dataclass(frozen=True, slots=True)
class EncounterRegistryEntry:
    key: PairKey
    discovery: PairDiscovery
    decision: EncounterLifecycleDecision
    first_seen_min: float
    last_seen_min: float
    update_count: int

    @property
    def responsibility_active(self) -> bool:
        return self.decision.responsibility_active


@dataclass(frozen=True, slots=True)
class EncounterRegistryUpdate:
    time_min: float
    entries: tuple[EncounterRegistryEntry, ...]
    discovered_keys: tuple[PairKey, ...]
    retained_keys: tuple[PairKey, ...]
    retired_keys: tuple[PairKey, ...]

    def by_key(self, first_id: str, second_id: str) -> EncounterRegistryEntry:
        key = canonical_pair_key(first_id, second_id)
        for entry in self.entries:
            if entry.key == key:
                return entry
        raise KeyError(key)


@dataclass(slots=True)
class _TrackedPair:
    lifecycle: EncounterLifecycle
    first_seen_min: float
    last_seen_min: float
    update_count: int
    responsibility_delegated: bool = False
    entry: EncounterRegistryEntry | None = None


class EncounterRegistry:
    """Discover/track all relevant active-vessel pairs at event boundaries.

    A pair is updated while it is inside the broad discovery window.  If a pair leaves
    that window but still carries persisted Rule-13/14/15 responsibility, it remains
    tracked so the lifecycle can observe the clear/past-and-clear release.  A pair with
    neither discovery relevance nor persisted responsibility is retired.  Removing a
    vessel from ``World2D`` retires every pair containing that vessel immediately.

    ``update`` must be called at most once per timestamp.  Scheduler phase boundaries at
    the same physical time are not independent navigation observations.
    """

    def __init__(
        self,
        *,
        discovery: EncounterDiscoveryConfig,
        lifecycle: EncounterLifecycleConfig,
        profile_catalog: VesselProfileCatalog | None = None,
        responsibility_delegation_resolver: Callable[[PairKey], bool] | None = None,
    ) -> None:
        self.discovery_config = discovery
        self.lifecycle_config = lifecycle
        self.profile_catalog = profile_catalog
        self._responsibility_delegation_resolver = responsibility_delegation_resolver
        self._tracked: dict[PairKey, _TrackedPair] = {}
        self._last_update_time: float | None = None

    @property
    def tracked_keys(self) -> tuple[PairKey, ...]:
        return tuple(sorted(self._tracked))

    def bind_responsibility_delegation_resolver(
        self, resolver: Callable[[PairKey], bool] | None
    ) -> None:
        """Bind a stable pair-responsibility delegation policy before first observation."""
        if self._last_update_time is not None or self._tracked:
            raise RuntimeError("responsibility delegation resolver is cold-start-only")
        if resolver is not None and not callable(resolver):
            raise TypeError("resolver must be callable or None")
        self._responsibility_delegation_resolver = resolver

    def clone_lifecycle_for(self, first_id: str, second_id: str) -> EncounterLifecycle:
        """Return an independent copy of one currently tracked pair lifecycle.

        Fleet-level registry observations and a local pair-control loop intentionally use
        separate lifecycle instances once command ownership is delegated.  The clone
        preserves all persistence/hysteresis state at the delegation boundary while
        preventing the registry and local controller from double-writing one history.
        """

        key = canonical_pair_key(first_id, second_id)
        tracked = self._tracked.get(key)
        if tracked is None or tracked.entry is None:
            raise KeyError(key)
        return deepcopy(tracked.lifecycle)

    def pair_domains(
        self, first_id: str, second_id: str
    ) -> tuple[ShipDomain | None, ShipDomain | None]:
        """Return ordered pair domains when both are available from the profile catalog."""
        key = canonical_pair_key(first_id, second_id)
        if self.profile_catalog is None:
            return None, None
        first = self.profile_catalog.domain_for(key[0])
        second = self.profile_catalog.domain_for(key[1])
        if first is None or second is None:
            return None, None
        return first, second

    def evaluate_pair_discovery(
        self,
        first_id: str,
        second_id: str,
        *,
        prediction: EncounterPrediction,
        current_range_nm: float,
    ) -> PairDiscovery:
        """Evaluate broad discovery from neutral pair prediction evidence.

        The registry needs only current range and constant-velocity CPA/TCPA for broad
        tracking. The physical state class that produced that evidence is deliberately
        outside this method.
        """
        key = canonical_pair_key(first_id, second_id)
        if not isfinite(current_range_nm) or current_range_nm < 0.0:
            raise ValueError("current_range_nm must be finite and non-negative")
        if {prediction.first_id, prediction.second_id} != set(key):
            raise ValueError("prediction vessel ids do not match discovery pair")
        first_domain, second_domain = self.pair_domains(*key)
        if first_domain is not None and second_domain is not None:
            domain_bound = max(
                first_domain.ahead_nm,
                first_domain.astern_nm,
                first_domain.port_nm,
                first_domain.starboard_nm,
            ) + max(
                second_domain.ahead_nm,
                second_domain.astern_nm,
                second_domain.port_nm,
                second_domain.starboard_nm,
            )
        else:
            domain_bound = 0.0
        effective_current = max(self.discovery_config.current_distance_nm, domain_bound)
        effective_dcpa = max(self.discovery_config.dcpa_distance_nm, domain_bound)
        inside = current_range_nm <= effective_current + _EPS
        raw_tcpa = prediction.cpa.raw_tcpa
        future = (
            isfinite(raw_tcpa)
            and raw_tcpa > _EPS
            and raw_tcpa <= self.discovery_config.tcpa_horizon_min + _EPS
            and prediction.cpa.dcpa <= effective_dcpa + _EPS
        )
        return PairDiscovery(
            key,
            prediction,
            current_range_nm,
            inside,
            future,
            effective_current,
            effective_dcpa,
        )

    def _discover_pair(self, first: Vessel2DState, second: Vessel2DState) -> PairDiscovery:
        key = canonical_pair_key(first.vessel_id, second.vessel_id)
        if first.vessel_id != key[0]:
            first, second = second, first
        prediction = predict_encounter(first, second, horizon=None)
        dx = second.position[0] - first.position[0]
        dy = second.position[1] - first.position[1]
        return self.evaluate_pair_discovery(
            key[0],
            key[1],
            prediction=prediction,
            current_range_nm=hypot(dx, dy),
        )

    def scan_all(self, vessels: Iterable[Vessel2DState]) -> tuple[PairDiscovery, ...]:
        """Evaluate every active pair in canonical order, including non-discovered pairs."""
        vessel_list = tuple(vessels)
        by_id = {v.vessel_id: v for v in vessel_list}
        if len(by_id) != len(vessel_list):
            raise ValueError("duplicate vessel_id")
        ordered = [by_id[vessel_id] for vessel_id in sorted(by_id)]
        result: list[PairDiscovery] = []
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                result.append(self._discover_pair(first, second))
        return tuple(result)

    def scan(self, vessels: Iterable[Vessel2DState]) -> tuple[PairDiscovery, ...]:
        return tuple(item for item in self.scan_all(vessels) if item.discovered)

    def update_from_evidence(
        self,
        *,
        time_min: float,
        active_vessel_ids: Iterable[str],
        pair_discoveries: Iterable[PairDiscovery],
        evidence_for: Callable[[PairKey], EncounterLifecycleEvidence],
    ) -> EncounterRegistryUpdate:
        """Advance registry state from state-class-independent pair evidence.

        ``pair_discoveries`` must contain exactly one evaluation for every active pair,
        including pairs outside the broad window. The registry decides which new pairs
        to create and which persisted pairs to retain/retire; ``evidence_for`` is called
        only for those candidate keys whose lifecycle actually needs an observation.
        """
        if not isfinite(time_min) or time_min < 0.0:
            raise ValueError("time_min must be finite and non-negative")
        if self._last_update_time is not None and time_min <= self._last_update_time + _EPS:
            raise ValueError("encounter registry updates require strictly increasing world time")

        active_ids_tuple = tuple(active_vessel_ids)
        active_ids = set(active_ids_tuple)
        if len(active_ids) != len(active_ids_tuple) or any(not item for item in active_ids):
            raise ValueError("active_vessel_ids must be unique and non-empty")

        all_discovery: dict[PairKey, PairDiscovery] = {}
        for item in pair_discoveries:
            if not isinstance(item, PairDiscovery):
                raise TypeError("pair_discoveries must contain PairDiscovery values")
            if item.key in all_discovery:
                raise ValueError(f"duplicate pair discovery: {item.key}")
            if item.key != canonical_pair_key(*item.key):
                raise ValueError("pair discovery key must be canonical")
            if item.key[0] not in active_ids or item.key[1] not in active_ids:
                raise ValueError("pair discovery contains inactive vessel id")
            all_discovery[item.key] = item

        expected_pairs = {
            (first, second)
            for i, first in enumerate(sorted(active_ids))
            for second in sorted(active_ids)[i + 1 :]
        }
        if set(all_discovery) != expected_pairs:
            raise ValueError("pair_discoveries must cover every active vessel pair exactly once")

        discovered = {key: item for key, item in all_discovery.items() if item.discovered}
        retired: list[PairKey] = []
        retained: list[PairKey] = []
        entries: list[EncounterRegistryEntry] = []

        # Pair observations happen only after the whole batch validates, so malformed
        # discovery coverage cannot partially mutate lifecycle history.
        self._last_update_time = time_min

        for key in sorted(tuple(self._tracked)):
            if key[0] not in active_ids or key[1] not in active_ids:
                retired.append(key)
                del self._tracked[key]

        candidate_keys = sorted(set(discovered) | set(self._tracked))
        for key in candidate_keys:
            discovery = all_discovery[key]
            tracked = self._tracked.get(key)
            if tracked is None:
                delegated = bool(
                    self._responsibility_delegation_resolver(key)
                    if self._responsibility_delegation_resolver is not None
                    else False
                )
                tracked = _TrackedPair(
                    lifecycle=EncounterLifecycle(key[0], key[1], config=self.lifecycle_config),
                    first_seen_min=time_min,
                    last_seen_min=time_min,
                    update_count=0,
                    responsibility_delegated=delegated,
                )
                self._tracked[key] = tracked

            delegated_now = bool(
                self._responsibility_delegation_resolver(key)
                if self._responsibility_delegation_resolver is not None
                else False
            )
            if delegated_now != tracked.responsibility_delegated:
                raise RuntimeError("responsibility delegation mode changed during pair lifecycle")
            evidence = evidence_for(key)
            if evidence.responsibility_delegated != tracked.responsibility_delegated:
                evidence = replace(
                    evidence,
                    responsibility_delegated=tracked.responsibility_delegated,
                )
            decision = tracked.lifecycle.update_evidence(evidence)
            tracked.last_seen_min = time_min
            tracked.update_count += 1
            entry = EncounterRegistryEntry(
                key=key,
                discovery=discovery,
                decision=decision,
                first_seen_min=tracked.first_seen_min,
                last_seen_min=tracked.last_seen_min,
                update_count=tracked.update_count,
            )
            tracked.entry = entry

            if not discovery.discovered and not decision.responsibility_active:
                retired.append(key)
                del self._tracked[key]
                continue
            if not discovery.discovered:
                retained.append(key)
            entries.append(entry)

        return EncounterRegistryUpdate(
            time_min=time_min,
            entries=tuple(sorted(entries, key=lambda e: e.key)),
            discovered_keys=tuple(sorted(discovered)),
            retained_keys=tuple(sorted(retained)),
            retired_keys=tuple(sorted(set(retired))),
        )

    def augment_same_time_arrivals_from_evidence(
        self,
        *,
        time_min: float,
        active_vessel_ids: Iterable[str],
        added_vessel_ids: Iterable[str],
        pair_discoveries: Iterable[PairDiscovery],
        evidence_for: Callable[[PairKey], EncounterLifecycleEvidence],
    ) -> EncounterRegistryUpdate:
        """Supersede one timestamp after zero-time membership additions.

        Existing tracked pair lifecycles are deliberately *not* advanced again.  Their
        physical state has not advanced in time; only pairs touching newly added vessel
        ids receive a first lifecycle observation.  This supports exact same-time source
        ingress after a post-observation control/ownership transition without epsilon
        time shifts or double-counting existing encounter histories.
        """
        if not isfinite(time_min) or time_min < 0.0:
            raise ValueError("time_min must be finite and non-negative")
        if self._last_update_time is None or abs(time_min - self._last_update_time) > _EPS:
            raise ValueError("same-time arrival augmentation requires the current registry timestamp")

        active_ids_tuple = tuple(active_vessel_ids)
        active_ids = set(active_ids_tuple)
        if len(active_ids) != len(active_ids_tuple) or any(not item for item in active_ids):
            raise ValueError("active_vessel_ids must be unique and non-empty")
        added_tuple = tuple(added_vessel_ids)
        added = set(added_tuple)
        if not added or len(added) != len(added_tuple) or any(not item for item in added):
            raise ValueError("added_vessel_ids must be unique and non-empty")
        if not added.issubset(active_ids):
            raise ValueError("added_vessel_ids must be active")
        if any(added.intersection(key) for key in self._tracked):
            raise ValueError("same-time arrival augmentation ids are already present in tracked pairs")

        all_discovery: dict[PairKey, PairDiscovery] = {}
        for item in pair_discoveries:
            if not isinstance(item, PairDiscovery):
                raise TypeError("pair_discoveries must contain PairDiscovery values")
            if item.key in all_discovery:
                raise ValueError(f"duplicate pair discovery: {item.key}")
            if item.key != canonical_pair_key(*item.key):
                raise ValueError("pair discovery key must be canonical")
            if item.key[0] not in active_ids or item.key[1] not in active_ids:
                raise ValueError("pair discovery contains inactive vessel id")
            all_discovery[item.key] = item

        expected_pairs = {
            (first, second)
            for i, first in enumerate(sorted(active_ids))
            for second in sorted(active_ids)[i + 1 :]
        }
        if set(all_discovery) != expected_pairs:
            raise ValueError("pair_discoveries must cover every active vessel pair exactly once")

        # Existing-pair physical discovery evidence must be unchanged at the same
        # physical timestamp.  If it changed, this is not a pure membership augmentation
        # and the caller must use a later ordinary registry observation instead.
        for key, tracked in self._tracked.items():
            if tracked.entry is None:
                raise RuntimeError("tracked pair lacks committed registry entry")
            discovery = all_discovery.get(key)
            if discovery is None:
                raise ValueError("same-time arrival augmentation cannot remove an existing tracked pair")
            if discovery != tracked.entry.discovery:
                raise ValueError("same-time arrival augmentation changed existing pair discovery evidence")

        new_pair_keys = sorted(
            key for key in all_discovery
            if added.intersection(key)
        )
        for key in new_pair_keys:
            discovery = all_discovery[key]
            if not discovery.discovered:
                continue
            delegated = bool(
                self._responsibility_delegation_resolver(key)
                if self._responsibility_delegation_resolver is not None
                else False
            )
            tracked = _TrackedPair(
                lifecycle=EncounterLifecycle(key[0], key[1], config=self.lifecycle_config),
                first_seen_min=time_min,
                last_seen_min=time_min,
                update_count=0,
                responsibility_delegated=delegated,
            )
            evidence = evidence_for(key)
            if evidence.responsibility_delegated != delegated:
                evidence = replace(evidence, responsibility_delegated=delegated)
            decision = tracked.lifecycle.update_evidence(evidence)
            tracked.update_count = 1
            entry = EncounterRegistryEntry(
                key=key,
                discovery=discovery,
                decision=decision,
                first_seen_min=time_min,
                last_seen_min=time_min,
                update_count=1,
            )
            tracked.entry = entry
            self._tracked[key] = tracked

        entries = tuple(
            sorted(
                (tracked.entry for tracked in self._tracked.values() if tracked.entry is not None),
                key=lambda item: item.key,
            )
        )
        discovered_keys = tuple(sorted(entry.key for entry in entries if entry.discovery.discovered))
        retained_keys = tuple(
            sorted(
                entry.key
                for entry in entries
                if not entry.discovery.discovered and entry.decision.responsibility_active
            )
        )
        return EncounterRegistryUpdate(
            time_min=time_min,
            entries=entries,
            discovered_keys=discovered_keys,
            retained_keys=retained_keys,
            retired_keys=(),
        )

    def update(
        self,
        world: World2D,
        *,
        uncertain_pairs: Iterable[PairKey] = (),
        responsibility_context: ResponsibilityContext | None = None,
    ) -> EncounterRegistryUpdate:
        """Historical World2D wrapper around ``update_from_evidence``."""
        time_min = world.time
        snapshot = world.snapshot()
        by_id = {v.vessel_id: v for v in snapshot.vessels}
        uncertain = {canonical_pair_key(*key) for key in uncertain_pairs}
        all_discoveries = self.scan_all(snapshot.vessels)

        def evidence_for(key: PairKey) -> EncounterLifecycleEvidence:
            first = by_id[key[0]]
            second = by_id[key[1]]
            first_domain, second_domain = self.pair_domains(*key)
            return build_state_lifecycle_evidence(
                first,
                second,
                time_min=time_min,
                config=self.lifecycle_config,
                uncertain=key in uncertain,
                responsibility_context=responsibility_context,
                first_domain=first_domain,
                second_domain=second_domain,
            )

        return self.update_from_evidence(
            time_min=time_min,
            active_vessel_ids=by_id,
            pair_discoveries=all_discoveries,
            evidence_for=evidence_for,
        )

