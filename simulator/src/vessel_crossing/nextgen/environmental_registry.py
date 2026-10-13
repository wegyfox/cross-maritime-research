"""Read-only encounter-registry assimilation for ``EnvironmentalWorld2D``.

A80 reuses the ordinary ``EncounterRegistry`` state machine, admission layer and active
component builder. This adapter supplies two physical facts that differ under current:

- broad pair discovery uses ground-motion CPA/TCPA;
- persistent lifecycle evidence uses the a79 environmental evidence contract.

No maneuver command is generated and no water-relative controller migration is implied.
"""

from __future__ import annotations

from math import hypot
from typing import Iterable

from .encounter_classification import EncounterClassificationConfig
from .encounter_registry import (
    EncounterRegistry,
    EncounterRegistryUpdate,
    PairDiscovery,
    PairKey,
    canonical_pair_key,
)
from .environmental_encounter_classification import (
    EnvironmentalEncounterDirectionBasis,
    classify_environmental_encounter,
)
from .environmental_lifecycle import build_environmental_lifecycle_evidence
from .environmental_world import EnvironmentalWorld2D
from .responsibility import ResponsibilityContext


def scan_environmental_registry_pairs(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    *,
    classification_config: EncounterClassificationConfig | None = None,
    direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
) -> tuple[PairDiscovery, ...]:
    """Evaluate broad discovery for every active environmental-world pair."""
    if not isinstance(registry, EncounterRegistry):
        raise TypeError("registry must be EncounterRegistry")
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    snapshot = world.snapshot()
    by_id = {state.vessel_id: state for state in snapshot.vessels}
    ordered_ids = sorted(by_id)
    result: list[PairDiscovery] = []
    for index, first_id in enumerate(ordered_ids):
        first = by_id[first_id]
        for second_id in ordered_ids[index + 1 :]:
            second = by_id[second_id]
            classification = classify_environmental_encounter(
                first,
                world.drift_for(first_id),
                second,
                world.drift_for(second_id),
                config=classification_config,
                direction_basis=direction_basis,
            )
            dx = second.position[0] - first.position[0]
            dy = second.position[1] - first.position[1]
            result.append(
                registry.evaluate_pair_discovery(
                    first_id,
                    second_id,
                    prediction=classification.classification.prediction,
                    current_range_nm=hypot(dx, dy),
                )
            )
    return tuple(result)


def update_environmental_registry(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    *,
    uncertain_pairs: Iterable[PairKey] = (),
    responsibility_context: ResponsibilityContext | None = None,
    classification_config: EncounterClassificationConfig | None = None,
    direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
) -> EncounterRegistryUpdate:
    """Update the ordinary registry from one environmental-world event boundary."""
    if not isinstance(registry, EncounterRegistry):
        raise TypeError("registry must be EncounterRegistry")
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    snapshot = world.snapshot()
    by_id = {state.vessel_id: state for state in snapshot.vessels}
    uncertain = {canonical_pair_key(*key) for key in uncertain_pairs}
    all_discoveries = scan_environmental_registry_pairs(
        registry,
        world,
        classification_config=classification_config,
        direction_basis=direction_basis,
    )

    def evidence_for(key: PairKey):
        first = by_id[key[0]]
        second = by_id[key[1]]
        first_domain, second_domain = registry.pair_domains(*key)
        return build_environmental_lifecycle_evidence(
            first,
            world.drift_for(key[0]),
            second,
            world.drift_for(key[1]),
            time_min=snapshot.time,
            config=registry.lifecycle_config,
            uncertain=key in uncertain,
            responsibility_context=responsibility_context,
            classification_config=classification_config,
            direction_basis=direction_basis,
            first_domain=first_domain,
            second_domain=second_domain,
        )

    return registry.update_from_evidence(
        time_min=snapshot.time,
        active_vessel_ids=by_id,
        pair_discoveries=all_discoveries,
        evidence_for=evidence_for,
    )


def augment_environmental_registry_same_time_arrivals(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    added_vessel_ids: Iterable[str],
    *,
    uncertain_pairs: Iterable[PairKey] = (),
    responsibility_context: ResponsibilityContext | None = None,
    classification_config: EncounterClassificationConfig | None = None,
    direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
) -> EncounterRegistryUpdate:
    """Supersede one environmental observation after exact same-time arrivals."""
    if not isinstance(registry, EncounterRegistry):
        raise TypeError("registry must be EncounterRegistry")
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    added = tuple(added_vessel_ids)
    snapshot = world.snapshot()
    by_id = {state.vessel_id: state for state in snapshot.vessels}
    uncertain = {canonical_pair_key(*key) for key in uncertain_pairs}
    all_discoveries = scan_environmental_registry_pairs(
        registry,
        world,
        classification_config=classification_config,
        direction_basis=direction_basis,
    )

    def evidence_for(key: PairKey):
        first = by_id[key[0]]
        second = by_id[key[1]]
        first_domain, second_domain = registry.pair_domains(*key)
        return build_environmental_lifecycle_evidence(
            first,
            world.drift_for(key[0]),
            second,
            world.drift_for(key[1]),
            time_min=snapshot.time,
            config=registry.lifecycle_config,
            uncertain=key in uncertain,
            responsibility_context=responsibility_context,
            classification_config=classification_config,
            direction_basis=direction_basis,
            first_domain=first_domain,
            second_domain=second_domain,
        )

    return registry.augment_same_time_arrivals_from_evidence(
        time_min=snapshot.time,
        active_vessel_ids=by_id,
        added_vessel_ids=added,
        pair_discoveries=all_discoveries,
        evidence_for=evidence_for,
    )
