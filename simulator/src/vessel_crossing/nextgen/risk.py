"""Snapshot CPA/TCPA collision-risk activation for the next-generation simulator.

This is deliberately narrower than a claim of full COLREG Rule-7 appraisal.  A single
kinematic snapshot cannot reproduce proper lookout, radar plotting history, systematic
bearing observation, target-size effects, visibility, or all other information available
to a navigator.  The module therefore exposes a transparent CPA/TCPA activation gate
with explicit model thresholds and an explicit uncertainty override.

The gate is kept separate from encounter classification, responsibility assignment, and
maneuver selection.  In particular, Rule-13 overtaking responsibility remains active
independently of this Rule-14/15 activation gate, as encoded by the a8 responsibility
scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .dynamics import Vessel2DState
from .encounter import EncounterPrediction, predict_encounter
from .responsibility import ResponsibilityBasis, ResponsibilityRole, ResponsibilityScaffold
from .ship_domain import PairShipDomainClearance, ShipDomain, evaluate_pair_ship_domain

_EPS = 1e-12


class RiskEvidence(str, Enum):
    """Dominant reason for the snapshot gate decision."""

    IMMEDIATE_DOMAIN_INTRUSION = "immediate_domain_intrusion"
    FUTURE_CPA_DOMAIN_INTRUSION = "future_cpa_domain_intrusion"
    UNCERTAINTY_DEEMED_RISK = "uncertainty_deemed_risk"
    DIVERGING_OR_PAST_CPA = "diverging_or_past_cpa"
    NO_RELATIVE_MOTION = "no_relative_motion"
    TCPA_BEYOND_HORIZON = "tcpa_beyond_horizon"
    DCPA_CLEAR = "dcpa_clear"


@dataclass(frozen=True, slots=True)
class RiskGateConfig:
    """Explicit numerical model thresholds for snapshot CPA/TCPA activation.

    These are simulation parameters, not asserted COLREG numerical limits.

    Parameters
    ----------
    dcpa_threshold_nm:
        Future closest-approach distance at or below which the modeled safety domain is
        treated as intruded.
    tcpa_horizon_min:
        Maximum positive TCPA considered actionable by this snapshot gate.
    immediate_distance_nm:
        Current separation at or below which risk is activated immediately, even when
        mathematical TCPA is non-positive.  When omitted it equals ``dcpa_threshold_nm``.
    """

    dcpa_threshold_nm: float
    tcpa_horizon_min: float
    immediate_distance_nm: float | None = None

    def __post_init__(self) -> None:
        if self.dcpa_threshold_nm <= 0.0:
            raise ValueError("dcpa_threshold_nm must be positive")
        if self.tcpa_horizon_min <= 0.0:
            raise ValueError("tcpa_horizon_min must be positive")
        if self.immediate_distance_nm is not None and self.immediate_distance_nm <= 0.0:
            raise ValueError("immediate_distance_nm must be positive when supplied")

    @property
    def effective_immediate_distance_nm(self) -> float:
        return self.dcpa_threshold_nm if self.immediate_distance_nm is None else self.immediate_distance_nm


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    first_id: str
    second_id: str
    risk_exists: bool
    evidence: RiskEvidence
    prediction: EncounterPrediction
    config: RiskGateConfig
    uncertain: bool
    inside_immediate_domain: bool
    future_cpa_in_horizon: bool
    future_dcpa_intrudes_domain: bool
    immediate_domain_clearance: PairShipDomainClearance | None = None
    future_domain_clearance: PairShipDomainClearance | None = None

    def involves(self, first_id: str, second_id: str) -> bool:
        return {self.first_id, self.second_id} == {first_id, second_id}


@dataclass(frozen=True, slots=True)
class ActivatedResponsibility:
    """Operational activation state layered over the immutable a8 scaffold."""

    scaffold: ResponsibilityScaffold
    risk: RiskAssessment
    active: bool
    activation_reason: str

    def role_for(self, vessel_id: str) -> ResponsibilityRole:
        if vessel_id not in {self.scaffold.first_id, self.scaffold.second_id}:
            raise KeyError(vessel_id)
        if not self.active:
            return ResponsibilityRole.NONE
        return self.scaffold.role_for(vessel_id)


def assess_cpa_tcpa_risk(
    first: Vessel2DState,
    second: Vessel2DState,
    *,
    config: RiskGateConfig,
    uncertain: bool = False,
    first_domain: ShipDomain | None = None,
    second_domain: ShipDomain | None = None,
) -> RiskAssessment:
    """Evaluate the deliberately narrow snapshot CPA/TCPA activation gate.

    The CPA is recomputed without a horizon clamp so the gate cannot accidentally use a
    DCPA measured at a prior caller's truncated prediction horizon.  ``raw_tcpa`` is then
    compared with this gate's own explicit horizon.

    ``uncertain=True`` represents external doubt/insufficient confidence supplied by a
    future observation layer.  It conservatively activates risk instead of pretending
    that a single snapshot can itself determine whether the available information is
    sufficient.
    """

    if (first_domain is None) != (second_domain is None):
        raise ValueError("first_domain and second_domain must be supplied together")

    prediction = predict_encounter(first, second, horizon=None)
    cpa = prediction.cpa
    relative_position = (
        second.position[0] - first.position[0],
        second.position[1] - first.position[1],
    )
    immediate_domain_clearance = None
    future_domain_clearance = None
    if first_domain is not None and second_domain is not None:
        immediate_domain_clearance = evaluate_pair_ship_domain(
            first_heading_deg=first.heading_deg,
            second_heading_deg=second.heading_deg,
            relative_position=relative_position,
            first_domain=first_domain,
            second_domain=second_domain,
        )
        inside_immediate = immediate_domain_clearance.intrudes
    else:
        immediate_limit = config.effective_immediate_distance_nm
        inside_immediate = cpa.current_distance <= immediate_limit + _EPS

    raw_tcpa = cpa.raw_tcpa
    future_in_horizon = (
        isfinite(raw_tcpa)
        and raw_tcpa > _EPS
        and raw_tcpa <= config.tcpa_horizon_min + _EPS
    )
    if future_in_horizon and first_domain is not None and second_domain is not None:
        relative_velocity = (
            second.velocity[0] - first.velocity[0],
            second.velocity[1] - first.velocity[1],
        )
        future_relative = (
            relative_position[0] + relative_velocity[0] * raw_tcpa,
            relative_position[1] + relative_velocity[1] * raw_tcpa,
        )
        future_domain_clearance = evaluate_pair_ship_domain(
            first_heading_deg=first.heading_deg,
            second_heading_deg=second.heading_deg,
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


def activate_responsibility(
    scaffold: ResponsibilityScaffold,
    risk: RiskAssessment,
) -> ActivatedResponsibility:
    """Activate conditional responsibility without choosing a maneuver command.

    Rule 13 is intentionally not weakened by the snapshot risk gate.  Rule 14/15 roles
    activate only when the gate reports risk.  Encounters with no pair responsibility
    remain inactive regardless of a risk assessment.
    """

    if not risk.involves(scaffold.first_id, scaffold.second_id):
        raise ValueError("risk assessment vessel ids do not match responsibility scaffold")

    if scaffold.basis is ResponsibilityBasis.NONE:
        return ActivatedResponsibility(scaffold, risk, False, "no_pair_responsibility")

    if not scaffold.requires_collision_risk_gate:
        return ActivatedResponsibility(scaffold, risk, True, "responsibility_not_gated_by_cpa_tcpa")

    if risk.risk_exists:
        return ActivatedResponsibility(scaffold, risk, True, f"risk_gate:{risk.evidence.value}")

    return ActivatedResponsibility(scaffold, risk, False, f"risk_gate_inactive:{risk.evidence.value}")
