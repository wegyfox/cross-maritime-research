"""Snapshot CPA/TCPA risk gate for explicit environmental velocity frames.

This adapter preserves the existing ``RiskGateConfig`` / ``RiskEvidence`` /
``RiskAssessment`` contract while sourcing kinematics from a74 evidence:

- relative bearings are measured from hull headings;
- CPA/TCPA uses ground velocities;
- current/future shaped-domain orientation uses hull headings under the same
  constant-heading snapshot approximation as the historical gate.

No responsibility is activated automatically. Callers may explicitly pass the returned
``RiskAssessment`` to the existing responsibility activation layer if their scenario
semantics support that step.
"""

from __future__ import annotations

from math import isfinite

from .dynamics import normalize_heading
from .encounter import EncounterPrediction, absolute_bearing_deg
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_evidence import predict_environmental_pair_motion
from .risk import RiskAssessment, RiskEvidence, RiskGateConfig
from .ship_domain import ShipDomain, evaluate_pair_ship_domain

_EPS = 1e-12


def _relative_bearing_from_hull(
    observer_position: tuple[float, float],
    observer_hull_heading_deg: float,
    target_position: tuple[float, float],
) -> float:
    return normalize_heading(
        absolute_bearing_deg(observer_position, target_position) - observer_hull_heading_deg
    )


def assess_environmental_cpa_tcpa_risk(
    first: WaterRelativeVesselState,
    first_drift_velocity: tuple[float, float],
    second: WaterRelativeVesselState,
    second_drift_velocity: tuple[float, float],
    *,
    config: RiskGateConfig,
    uncertain: bool = False,
    first_domain: ShipDomain | None = None,
    second_domain: ShipDomain | None = None,
) -> RiskAssessment:
    """Apply the existing transparent snapshot risk semantics to environmental motion."""
    if (first_domain is None) != (second_domain is None):
        raise ValueError("first_domain and second_domain must be supplied together")

    motion = predict_environmental_pair_motion(
        first,
        first_drift_velocity,
        second,
        second_drift_velocity,
        horizon=None,
        first_domain=first_domain,
        second_domain=second_domain,
    )
    cpa = motion.cpa
    prediction = EncounterPrediction(
        first_id=first.vessel_id,
        second_id=second.vessel_id,
        first_relative_bearing_deg=_relative_bearing_from_hull(
            first.position,
            motion.first.hull_heading_deg,
            second.position,
        ),
        second_relative_bearing_deg=_relative_bearing_from_hull(
            second.position,
            motion.second.hull_heading_deg,
            first.position,
        ),
        cpa=cpa,
    )

    relative_position = (
        second.position[0] - first.position[0],
        second.position[1] - first.position[1],
    )
    immediate_domain_clearance = motion.current_domain_clearance
    future_domain_clearance = None
    if immediate_domain_clearance is not None:
        inside_immediate = immediate_domain_clearance.intrudes
    else:
        inside_immediate = cpa.current_distance <= config.effective_immediate_distance_nm + _EPS

    raw_tcpa = cpa.raw_tcpa
    future_in_horizon = (
        isfinite(raw_tcpa)
        and raw_tcpa > _EPS
        and raw_tcpa <= config.tcpa_horizon_min + _EPS
    )

    if future_in_horizon and first_domain is not None and second_domain is not None:
        relative_velocity = (
            motion.second.ground_velocity[0] - motion.first.ground_velocity[0],
            motion.second.ground_velocity[1] - motion.first.ground_velocity[1],
        )
        future_relative = (
            relative_position[0] + relative_velocity[0] * raw_tcpa,
            relative_position[1] + relative_velocity[1] * raw_tcpa,
        )
        future_domain_clearance = evaluate_pair_ship_domain(
            first_heading_deg=motion.first.hull_heading_deg,
            second_heading_deg=motion.second.hull_heading_deg,
            relative_position=future_relative,
            first_domain=first_domain,
            second_domain=second_domain,
        )
        future_intrusion = future_domain_clearance.intrudes
    else:
        future_intrusion = future_in_horizon and cpa.dcpa <= config.dcpa_threshold_nm + _EPS

    if inside_immediate:
        risk_exists = True
        evidence = RiskEvidence.IMMEDIATE_DOMAIN_INTRUSION
    elif uncertain:
        risk_exists = True
        evidence = RiskEvidence.UNCERTAINTY_DEEMED_RISK
    elif cpa.relative_speed <= _EPS:
        risk_exists = False
        evidence = RiskEvidence.NO_RELATIVE_MOTION
    elif raw_tcpa <= _EPS:
        risk_exists = False
        evidence = RiskEvidence.DIVERGING_OR_PAST_CPA
    elif not future_in_horizon:
        risk_exists = False
        evidence = RiskEvidence.TCPA_BEYOND_HORIZON
    elif future_intrusion:
        risk_exists = True
        evidence = RiskEvidence.FUTURE_CPA_DOMAIN_INTRUSION
    else:
        risk_exists = False
        evidence = RiskEvidence.DCPA_CLEAR

    return RiskAssessment(
        first_id=first.vessel_id,
        second_id=second.vessel_id,
        risk_exists=risk_exists,
        evidence=evidence,
        prediction=prediction,
        config=config,
        uncertain=uncertain,
        inside_immediate_domain=inside_immediate,
        future_cpa_in_horizon=future_in_horizon,
        future_dcpa_intrudes_domain=future_intrusion,
        immediate_domain_clearance=immediate_domain_clearance,
        future_domain_clearance=future_domain_clearance,
    )
