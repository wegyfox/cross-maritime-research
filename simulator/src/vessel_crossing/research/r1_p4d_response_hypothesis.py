"""Literature-informed P4D command-response hypothesis candidate.

This is a *candidate* scientific construct, not an empirical calibration.  Human
ship-handling simulator evidence reports emergency response-to-first-action times on the
order of one minute.  P4D therefore uses 1.0 min as a rounded, literature-informed HS
extra-action-latency hypothesis and 0.0 min as an idealized zero-*extra*-latency AS
mechanism-isolation baseline.  Both roles still use the same physical plant dynamics.

The value must pass P4D interaction-positive qualification before any confirmatory freeze.
"""
from __future__ import annotations

from vessel_crossing.research.r1_p4d_behavior_contract import (
    P4DBehaviorContract,
    P4DMicroscopicRoleSpec,
    P4DParameterProvenance,
    P4DProvenanceKind,
    control_only_provenance,
    validate_confirmatory_independent_target_contract,
)
from vessel_crossing.testbed.behavior import BehaviorProfile

P4D_RESPONSE_HYPOTHESIS_ID = "P4D_RESP60_LITERATURE_INFORMED_V1"
KONISHI_2025_DOI = "10.18949/jintransnavi.10.2_69"
HS_EXTRA_LATENCY_MIN = 1.0
AS_EXTRA_LATENCY_MIN = 0.0


def candidate_response_contract() -> P4DBehaviorContract:
    as_p = P4DParameterProvenance(
        P4DProvenanceKind.EXPERIMENTAL_HYPOTHESIS,
        "P4D_ZERO_EXTRA_AUTOMATION_LATENCY_BASELINE_V1",
        "Idealized zero extra behavior/action latency above the common plant. This is a mechanism-isolation baseline, not a claim of zero real autonomous-system latency.",
    )
    hs_p = P4DParameterProvenance(
        P4DProvenanceKind.EXPERIMENTAL_HYPOTHESIS,
        f"KONISHI_ET_AL_2025_DOI_{KONISHI_2025_DOI}",
        "Literature-informed rounded hypothesis: Konishi et al. report 72.9 s mean emergency response time for non-collision officers (SD 29.07 s); P4D uses 60 s extra action latency as a nearby round mechanism contrast, not as a direct empirical calibration.",
    )
    domain_p = control_only_provenance("Unit operational-domain scale retained in P4D confirmatory candidate to avoid source-coupled domain premetering.")
    contract = P4DBehaviorContract(
        contract_id=P4D_RESPONSE_HYPOTHESIS_ID,
        as_spec=P4DMicroscopicRoleSpec(
            "AS", BehaviorProfile.AUTONOMOUS_POLICY, AS_EXTRA_LATENCY_MIN, 1.0, as_p, domain_p,
        ),
        hs_spec=P4DMicroscopicRoleSpec(
            "HS", BehaviorProfile.COMPLIANT_MERCHANT, HS_EXTRA_LATENCY_MIN, 1.0, hs_p, domain_p,
        ),
        reference_headway_forcing=False,
        scientific_parameter_status="CANDIDATE_REQUIRES_INTERACTION_POSITIVE_QUALIFICATION_BEFORE_FREEZE",
    )
    validate_confirmatory_independent_target_contract(contract)
    return contract


def provenance_payload() -> dict[str, object]:
    c = candidate_response_contract()
    return {
        "format": "cross-r1-p4d-response-hypothesis-candidate-v1",
        "hypothesis_id": P4D_RESPONSE_HYPOTHESIS_ID,
        "contract_sha256": c.identity_sha256,
        "scientific_status": c.scientific_parameter_status,
        "construct": "extra bounded behavior/action latency above a common physical plant",
        "not_constructs": [
            "macroscopic pair headway",
            "whole-encounter collision-avoidance timing",
            "network/communication latency alone",
            "plant maneuvering lag",
        ],
        "AS_extra_latency_min": AS_EXTRA_LATENCY_MIN,
        "HS_extra_latency_min": HS_EXTRA_LATENCY_MIN,
        "reference_headway_forcing": False,
        "operational_domain_scale_AS": 1.0,
        "operational_domain_scale_HS": 1.0,
        "literature_anchor": {
            "citation": "Konishi T, Fuchi M, Fujimoto S, Hirono K, Adhita I.G.M.S. (2025), Features of the time required by navigator to respond to emergency collision avoidance, Transactions of Navigation 10(2):69-75",
            "doi": KONISHI_2025_DOI,
            "reported_non_collision_officer_mean_response_s": 72.9,
            "reported_non_collision_officer_sd_s": 29.07,
            "mapping_status": "LITERATURE_INFORMED_EXPERIMENTAL_HYPOTHESIS_NOT_DIRECT_CALIBRATION",
        },
        "freeze_gate": "must remain candidate until low/balanced/stress interaction-positive qualification and repository regression pass",
    }
