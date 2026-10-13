"""Rule-13/14/15 responsibility scaffold for a76 environmental classification.

This module assigns only the conditional responsibility structure. It does not activate
snapshot risk, mutate encounter lifecycle or issue maneuver commands.
"""

from __future__ import annotations

from .encounter_classification import EncounterClassificationConfig
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_encounter_classification import (
    EnvironmentalEncounterClassification,
    EnvironmentalEncounterDirectionBasis,
    classify_environmental_encounter,
)
from .responsibility import (
    ResponsibilityContext,
    ResponsibilityScaffold,
    assign_responsibility_from_classification,
)


def assign_environmental_responsibility_scaffold(
    first: WaterRelativeVesselState,
    first_drift_velocity: tuple[float, float],
    second: WaterRelativeVesselState,
    second_drift_velocity: tuple[float, float],
    *,
    classification: EnvironmentalEncounterClassification | None = None,
    classification_config: EncounterClassificationConfig | None = None,
    direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
    context: ResponsibilityContext | None = None,
) -> ResponsibilityScaffold:
    """Map explicit environmental neutral geometry to the existing responsibility scaffold."""
    result = classification or classify_environmental_encounter(
        first,
        first_drift_velocity,
        second,
        second_drift_velocity,
        config=classification_config,
        direction_basis=direction_basis,
    )
    if {result.classification.prediction.first_id, result.classification.prediction.second_id} != {
        first.vessel_id,
        second.vessel_id,
    }:
        raise ValueError("environmental classification vessel ids do not match supplied states")
    return assign_responsibility_from_classification(
        first.vessel_id,
        second.vessel_id,
        classification=result.classification,
        context=context,
    )
