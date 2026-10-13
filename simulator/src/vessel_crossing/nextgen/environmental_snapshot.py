"""Read-only environmental encounter/responsibility/risk snapshot composition.

A78 composes already-validated layers without adding new maneuver semantics:

    environmental classification
      -> Rule-13/14/15 responsibility scaffold
      -> environmental snapshot risk
      -> existing responsibility activation

The result is instantaneous evidence only. No lifecycle history, maneuver intent,
arbitration or command ownership is mutated here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .encounter_classification import EncounterClassificationConfig
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_encounter_classification import (
    EnvironmentalEncounterClassification,
    EnvironmentalEncounterDirectionBasis,
    classify_environmental_encounter,
)
from .environmental_risk import assess_environmental_cpa_tcpa_risk
from .responsibility import (
    ResponsibilityContext,
    ResponsibilityScaffold,
    assign_responsibility_from_classification,
)
from .risk import ActivatedResponsibility, RiskAssessment, RiskGateConfig, activate_responsibility
from .ship_domain import ShipDomain


@dataclass(frozen=True, slots=True)
class EnvironmentalResponsibilitySnapshot:
    classification: EnvironmentalEncounterClassification
    scaffold: ResponsibilityScaffold
    risk: RiskAssessment
    activated: ActivatedResponsibility

    @property
    def active(self) -> bool:
        return self.activated.active


def assess_environmental_responsibility_snapshot(
    first: WaterRelativeVesselState,
    first_drift_velocity: tuple[float, float],
    second: WaterRelativeVesselState,
    second_drift_velocity: tuple[float, float],
    *,
    risk_config: RiskGateConfig,
    classification_config: EncounterClassificationConfig | None = None,
    direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
    context: ResponsibilityContext | None = None,
    uncertain: bool = False,
    first_domain: ShipDomain | None = None,
    second_domain: ShipDomain | None = None,
) -> EnvironmentalResponsibilitySnapshot:
    """Evaluate one environment-aware Section-II snapshot without persistent state."""
    classification = classify_environmental_encounter(
        first,
        first_drift_velocity,
        second,
        second_drift_velocity,
        config=classification_config,
        direction_basis=direction_basis,
    )
    scaffold = assign_responsibility_from_classification(
        first.vessel_id,
        second.vessel_id,
        classification=classification.classification,
        context=context,
    )
    risk = assess_environmental_cpa_tcpa_risk(
        first,
        first_drift_velocity,
        second,
        second_drift_velocity,
        config=risk_config,
        uncertain=uncertain,
        first_domain=first_domain,
        second_domain=second_domain,
    )
    activated = activate_responsibility(scaffold, risk)
    return EnvironmentalResponsibilitySnapshot(
        classification=classification,
        scaffold=scaffold,
        risk=risk,
        activated=activated,
    )
