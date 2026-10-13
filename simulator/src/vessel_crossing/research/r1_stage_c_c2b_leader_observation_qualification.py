"""Stage-C C2b qualification for leader-sensitive same-route observation allowance.

The mechanism is intentionally microscopic.  It composes follower command-response timing and
leader-state observation uncertainty inside the physical following reaction budget.  It does not
accept or enforce an external pair-headway matrix.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
from math import isclose

from .r1_p4d_longitudinal_qualification import _run_longitudinal_response_arm

FORMAT = "cross-r1-stage-c-c2b-leader-observation-qualification"
SCHEMA_VERSION = 1
BASE_REACTION_MIN = 0.25
FROZEN_ARMS = (
    ("NEUTRAL", 0, 0),
    ("FOLLOWER_15S", 15, 0),
    ("LEADER_15S", 0, 15),
    ("COMPOSED_15S_15S", 15, 15),
    ("LEADER_60S", 0, 60),
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _arm(name: str, follower_seconds: int, leader_seconds: int) -> dict[str, object]:
    raw = _run_longitudinal_response_arm(
        follower_delay_min=follower_seconds / 60.0,
        leader_observation_delay_min=leader_seconds / 60.0,
    )
    records = raw["center_service"]["records"]
    if len(records) != 2:
        raise RuntimeError("C2b fixture must produce exactly two center passages")
    return {
        "arm_id": name,
        "follower_response_allowance_seconds": follower_seconds,
        "leader_observation_allowance_seconds": leader_seconds,
        "pair_reaction_budget_min": raw["pair_reaction_budget_min"],
        "terminal_status": raw["terminal_status"],
        "service_order_preserved": [row["vessel_id"] for row in records]
        == ["RESP-LEADER", "RESP-FOLLOWER"],
        "arrival_order_inversion_count": raw["center_service"]["arrival_order_inversion_count"],
        "realized_headway_min": records[1]["observed_headway_min"],
        "follower_admission_wait_min": raw["admissions"][1]["waiting_time_min"],
        "follower_request_speed_nm_per_min": raw["follower_request_speed_nm_per_min"],
        "follower_center_speed_nm_per_min": raw["follower_center_speed_nm_per_min"],
        "modeled_hull_overlap_episode_count": (
            None
            if raw["safety"] is None
            else raw["safety"]["modeled_hull_overlap_episode_count"]
        ),
        "runtime_reference_headway_input": None,
    }


@lru_cache(maxsize=1)
def run_stage_c_c2b_leader_observation_qualification() -> dict[str, object]:
    """Execute the frozen C2b mechanism-qualification arms."""
    arms = [_arm(*spec) for spec in FROZEN_ARMS]
    by_id = {str(arm["arm_id"]): arm for arm in arms}

    exact_budget_composition = True
    for arm in arms:
        expected = (
            BASE_REACTION_MIN
            + int(arm["follower_response_allowance_seconds"]) / 60.0
            + int(arm["leader_observation_allowance_seconds"]) / 60.0
        )
        exact_budget_composition = exact_budget_composition and isclose(
            float(arm["pair_reaction_budget_min"]), expected, abs_tol=1e-12
        )

    all_success = all(arm["terminal_status"] == "SUCCESS" for arm in arms)
    order_preserved = all(
        bool(arm["service_order_preserved"])
        and int(arm["arrival_order_inversion_count"]) == 0
        for arm in arms
    )
    safety_evidence_present = all(
        arm["modeled_hull_overlap_episode_count"] is not None for arm in arms
    )
    no_hull_overlap = safety_evidence_present and all(
        int(arm["modeled_hull_overlap_episode_count"]) == 0 for arm in arms
    )
    reference_absent = all(arm["runtime_reference_headway_input"] is None for arm in arms)

    neutral = by_id["NEUTRAL"]
    follower_only = by_id["FOLLOWER_15S"]
    leader_only = by_id["LEADER_15S"]
    composed = by_id["COMPOSED_15S_15S"]
    leader_60 = by_id["LEADER_60S"]

    document = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "evidence_role": "ENGINEERING_MECHANISM_QUALIFICATION_ONLY",
        "scientific_validation_authorized": False,
        "runtime_reference_headway_used": False,
        "mechanism": {
            "reaction_budget_contract": (
                "base_reaction + follower_response_allowance + leader_observation_allowance"
            ),
            "base_reaction_min": BASE_REACTION_MIN,
            "pair_headway_target_present": False,
            "arbitrary_2x2_headway_matrix_supported": False,
            "structural_family": "ADDITIVE_FOLLOWER_AND_LEADER_TIME_ALLOWANCES",
        },
        "arms": arms,
        "qualification": {
            "all_terminal_success": all_success,
            "service_order_preserved": order_preserved,
            "safety_evidence_present": safety_evidence_present,
            "no_modeled_hull_overlap": no_hull_overlap,
            "exact_pair_reaction_budget_composition": exact_budget_composition,
            "reference_values_absent_from_runtime_control": reference_absent,
            "pass": bool(
                all_success
                and order_preserved
                and no_hull_overlap
                and exact_budget_composition
                and reference_absent
            ),
        },
        "observed_response": {
            "neutral_headway_min": neutral["realized_headway_min"],
            "follower_15s_headway_min": follower_only["realized_headway_min"],
            "leader_15s_headway_min": leader_only["realized_headway_min"],
            "composed_15s_15s_headway_min": composed["realized_headway_min"],
            "leader_60s_headway_min": leader_60["realized_headway_min"],
            "neutral_wait_min": neutral["follower_admission_wait_min"],
            "leader_15s_wait_min": leader_only["follower_admission_wait_min"],
            "leader_60s_wait_min": leader_60["follower_admission_wait_min"],
            "interpretation": (
                "Leader-state observation uncertainty is now a real controller/safety axis. "
                "Passage headway remains endogenous; this qualification does not map AS/HS labels "
                "to observation delays or fit any reference matrix."
            ),
        },
    }
    raw = _canonical_json(document)
    document["result_sha256"] = sha256(raw).hexdigest()
    return document
