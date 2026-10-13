"""P4D bounded-action-latency sensitivity axis.

This module deliberately replaces the failed attempt to interpret a 60-s repeated
command/action latency as a literature-calibrated HS model.  The current P4D backend
implements a bounded latency on each new maneuver target; that construct is useful as a
transparent model-sensitivity axis, but it is not the same construct as navigator
"time to first emergency action".

The axis is therefore experimental and pre-confirmatory: AS retains zero *extra* action
latency above the common plant and HS is evaluated at 15, 30 and 60 s.  No level is
claimed to be a calibrated human parameter.
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

P4D_ACTION_LATENCY_SENSITIVITY_ID = "R1_P4D_BOUNDED_ACTION_LATENCY_SENSITIVITY_V1"
LATENCY_LEVELS_MIN = (0.0, 0.25, 0.5, 1.0)
ARM_TO_LATENCY_MIN = {
    "NEUTRAL": 0.0,
    "LAT15": 0.25,
    "LAT30": 0.5,
    "LAT60": 1.0,
}


def action_latency_contract(hs_extra_latency_min: float) -> P4DBehaviorContract:
    value=float(hs_extra_latency_min)
    if value not in LATENCY_LEVELS_MIN:
        raise ValueError(f"unsupported P4D action-latency sensitivity level {value!r}")
    neutral = value == 0.0
    as_prov=P4DParameterProvenance(
        P4DProvenanceKind.EXPERIMENTAL_HYPOTHESIS,
        "P4D_ZERO_EXTRA_AUTOMATION_ACTION_LATENCY_BASELINE_V1",
        "Idealized zero extra action latency above the common plant; mechanism-isolation baseline, not a real-system zero-latency claim.",
    )
    if neutral:
        hs_prov=control_only_provenance(
            "Neutral P4D sensitivity control: HS and AS share zero extra bounded action latency."
        )
    else:
        hs_prov=P4DParameterProvenance(
            P4DProvenanceKind.EXPERIMENTAL_HYPOTHESIS,
            P4D_ACTION_LATENCY_SENSITIVITY_ID,
            f"Transparent bounded-action-latency sensitivity level {value*60:.0f} s applied to HS maneuver-target updates; not a calibrated navigator first-action time.",
        )
    domain=control_only_provenance(
        "Unit operational-domain scale retained to avoid source-coupled domain pre-metering."
    )
    c=P4DBehaviorContract(
        contract_id=f"{P4D_ACTION_LATENCY_SENSITIVITY_ID}/HS{int(round(value*60)):02d}S",
        as_spec=P4DMicroscopicRoleSpec(
            "AS",BehaviorProfile.AUTONOMOUS_POLICY,0.0,1.0,as_prov,domain,
        ),
        hs_spec=P4DMicroscopicRoleSpec(
            "HS",BehaviorProfile.COMPLIANT_MERCHANT,value,1.0,hs_prov,domain,
        ),
        reference_headway_forcing=False,
        scientific_parameter_status="PRE_CONFIRMATORY_SENSITIVITY_AXIS_NOT_HUMAN_CALIBRATION",
    )
    validate_confirmatory_independent_target_contract(c)
    return c


def arm_contract(arm: str) -> P4DBehaviorContract:
    try:
        value=ARM_TO_LATENCY_MIN[arm]
    except KeyError as exc:
        raise KeyError(f"unsupported sensitivity arm {arm!r}") from exc
    return action_latency_contract(value)


def sensitivity_axis_payload() -> dict[str, object]:
    return {
        "format":"cross-r1-p4d-action-latency-sensitivity-axis-v1",
        "axis_id":P4D_ACTION_LATENCY_SENSITIVITY_ID,
        "evidence_role":"PRE_CONFIRMATORY_MODEL_SENSITIVITY_ONLY",
        "construct":"bounded extra action latency on each new maneuver target above a common plant",
        "not_constructs":[
            "navigator time to first emergency action",
            "direct empirical HS calibration",
            "macroscopic pair headway",
            "plant maneuvering lag",
        ],
        "arms":[
            {"arm":arm,"HS_extra_latency_min":value,"HS_extra_latency_s":value*60.0,
             "contract_sha256":arm_contract(arm).identity_sha256}
            for arm,value in ARM_TO_LATENCY_MIN.items()
        ],
        "selection_rule":"levels fixed a priori at 0/15/30/60 s; no level chosen because it produces a preferred throughput/safety result",
        "v2_60s_status":"prior single-candidate V2 failed its frozen stress-success gate; V3 does not relabel that result",
    }
