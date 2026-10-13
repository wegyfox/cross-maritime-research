"""Encounter lifecycle assimilation for explicit environmental velocity frames.

A79 deliberately reuses the single ``EncounterLifecycle`` state machine.  This module
only builds the state-class-independent evidence contract from water-relative hull state
plus Earth-fixed environmental drift and then delegates to ``update_evidence``.

- compass-bearing history uses actual Earth-fixed positions;
- neutral classification uses the explicit a76 hull-aspect direction basis plus ground
  closing/CPA evidence;
- snapshot risk uses ground velocity;
- Rule-13 past-and-clear geometry uses the overtaken vessel's hull heading;
- no fake ``Vessel2DState`` is created.
"""

from __future__ import annotations

from math import hypot

from .encounter import absolute_bearing_deg
from .encounter_classification import EncounterClassificationConfig
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_encounter_classification import (
    EnvironmentalEncounterDirectionBasis,
    classify_environmental_encounter,
)
from .environmental_risk import assess_environmental_cpa_tcpa_risk
from .lifecycle import (
    EncounterLifecycle,
    EncounterLifecycleConfig,
    EncounterLifecycleDecision,
    EncounterLifecycleEvidence,
    LifecycleVesselGeometryEvidence,
)
from .observation import PairObservation
from .responsibility import ResponsibilityContext
from .ship_domain import ShipDomain


def _pair_observation(
    first: WaterRelativeVesselState,
    second: WaterRelativeVesselState,
    *,
    time_min: float,
) -> PairObservation:
    dx = second.position[0] - first.position[0]
    dy = second.position[1] - first.position[1]
    return PairObservation(
        time_min=time_min,
        first_id=first.vessel_id,
        second_id=second.vessel_id,
        range_nm=hypot(dx, dy),
        first_to_second_bearing_deg=absolute_bearing_deg(first.position, second.position),
        second_to_first_bearing_deg=absolute_bearing_deg(second.position, first.position),
    )


def build_environmental_lifecycle_evidence(
    first: WaterRelativeVesselState,
    first_drift_velocity: tuple[float, float],
    second: WaterRelativeVesselState,
    second_drift_velocity: tuple[float, float],
    *,
    time_min: float,
    config: EncounterLifecycleConfig,
    uncertain: bool = False,
    responsibility_context: ResponsibilityContext | None = None,
    classification_config: EncounterClassificationConfig | None = None,
    direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
    first_domain: ShipDomain | None = None,
    second_domain: ShipDomain | None = None,
) -> EncounterLifecycleEvidence:
    """Build one ordered environment-aware evidence snapshot for lifecycle assimilation."""
    if not isinstance(first, WaterRelativeVesselState) or not isinstance(
        second, WaterRelativeVesselState
    ):
        raise TypeError("first and second must be WaterRelativeVesselState")
    if not isinstance(config, EncounterLifecycleConfig):
        raise TypeError("config must be EncounterLifecycleConfig")
    if first.vessel_id == second.vessel_id:
        raise ValueError("environmental lifecycle evidence requires distinct vessel ids")

    classification = classify_environmental_encounter(
        first,
        first_drift_velocity,
        second,
        second_drift_velocity,
        config=classification_config,
        direction_basis=direction_basis,
    ).classification
    snapshot_risk = assess_environmental_cpa_tcpa_risk(
        first,
        first_drift_velocity,
        second,
        second_drift_velocity,
        config=config.risk_gate,
        uncertain=uncertain,
        first_domain=first_domain,
        second_domain=second_domain,
    )
    return EncounterLifecycleEvidence(
        observation=_pair_observation(first, second, time_min=time_min),
        classification=classification,
        snapshot_risk=snapshot_risk,
        first_geometry=LifecycleVesselGeometryEvidence(
            vessel_id=first.vessel_id,
            position=first.position,
            hull_heading_deg=first.through_water_heading_deg,
        ),
        second_geometry=LifecycleVesselGeometryEvidence(
            vessel_id=second.vessel_id,
            position=second.position,
            hull_heading_deg=second.through_water_heading_deg,
        ),
        responsibility_context=responsibility_context,
    )


def update_environmental_lifecycle(
    lifecycle: EncounterLifecycle,
    first: WaterRelativeVesselState,
    first_drift_velocity: tuple[float, float],
    second: WaterRelativeVesselState,
    second_drift_velocity: tuple[float, float],
    *,
    time_min: float,
    uncertain: bool = False,
    responsibility_context: ResponsibilityContext | None = None,
    classification_config: EncounterClassificationConfig | None = None,
    direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
    first_domain: ShipDomain | None = None,
    second_domain: ShipDomain | None = None,
) -> EncounterLifecycleDecision:
    """Assimilate environmental evidence into an existing ``EncounterLifecycle``.

    Call order may be reversed exactly like the historical state-based lifecycle API;
    evidence is canonicalized to the lifecycle's stable ``first_id``/``second_id``.
    """
    if not isinstance(lifecycle, EncounterLifecycle):
        raise TypeError("lifecycle must be EncounterLifecycle")
    supplied = {first.vessel_id, second.vessel_id}
    expected = {lifecycle.first_id, lifecycle.second_id}
    if supplied != expected:
        raise ValueError("supplied vessel ids do not match lifecycle pair")
    if first.vessel_id != lifecycle.first_id:
        first, second = second, first
        first_drift_velocity, second_drift_velocity = (
            second_drift_velocity,
            first_drift_velocity,
        )
        first_domain, second_domain = second_domain, first_domain

    evidence = build_environmental_lifecycle_evidence(
        first,
        first_drift_velocity,
        second,
        second_drift_velocity,
        time_min=time_min,
        config=lifecycle.config,
        uncertain=uncertain,
        responsibility_context=responsibility_context,
        classification_config=classification_config,
        direction_basis=direction_basis,
        first_domain=first_domain,
        second_domain=second_domain,
    )
    return lifecycle.update_evidence(evidence)
