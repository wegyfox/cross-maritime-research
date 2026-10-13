"""Independent Stage-C C2 qualification of CROSS longitudinal AS/HS behavior.

C2 measures endogenous same-route passage headway. External reference headways are not
inputs to the controller and are deliberately omitted from this artifact.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
from math import isclose

from .r1_p4d_longitudinal_qualification import _run_longitudinal_response_arm

FORMAT = "cross-r1-stage-c-c2-target-behavior-qualification"
SCHEMA_VERSION = 1
LATENCY_SECONDS = (0, 15, 30, 60)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _observed_arm(*, leader_type: str, follower_type: str, latency_seconds: int) -> dict[str, object]:
    if leader_type not in {"AS", "HS"} or follower_type not in {"AS", "HS"}:
        raise ValueError("leader_type and follower_type must be AS or HS")
    if latency_seconds not in LATENCY_SECONDS:
        raise ValueError("latency_seconds must be one of the frozen C2 latency points")
    raw = _run_longitudinal_response_arm(follower_delay_min=latency_seconds / 60.0)
    records = raw["center_service"]["records"]
    if len(records) != 2:
        raise RuntimeError("C2 two-vessel fixture must produce exactly two center passages")
    follower_record = records[1]
    admissions = raw["admissions"]
    return {
        "leader_type": leader_type,
        "follower_type": follower_type,
        "follower_latency_seconds": latency_seconds,
        "terminal_status": raw["terminal_status"],
        "service_order_preserved": [row["vessel_id"] for row in records] == ["RESP-LEADER", "RESP-FOLLOWER"],
        "arrival_order_inversion_count": raw["center_service"]["arrival_order_inversion_count"],
        "realized_headway_min": follower_record["observed_headway_min"],
        "follower_admission_wait_min": admissions[1]["waiting_time_min"],
        "follower_request_speed_nm_per_min": raw["follower_request_speed_nm_per_min"],
        "follower_center_speed_nm_per_min": raw["follower_center_speed_nm_per_min"],
        "modeled_hull_overlap_episode_count": None if raw["safety"] is None else raw["safety"]["modeled_hull_overlap_episode_count"],
        "runtime_reference_headway_input": None,
    }


@lru_cache(maxsize=1)
def run_stage_c_c2_behavior_qualification() -> dict[str, object]:
    """Execute the frozen C2 matrix without fitting to an external headway matrix."""
    arms: list[dict[str, object]] = []
    # Zero-latency label-only 2x2 control.
    for leader in ("AS", "HS"):
        for follower in ("AS", "HS"):
            arms.append(_observed_arm(leader_type=leader, follower_type=follower, latency_seconds=0))
    # Frozen follower-latency sweep for both leader labels. Zero is already above.
    for leader in ("AS", "HS"):
        for seconds in LATENCY_SECONDS[1:]:
            arms.append(_observed_arm(leader_type=leader, follower_type="HS", latency_seconds=seconds))

    zero = [a for a in arms if a["follower_latency_seconds"] == 0]
    zero_headways = [float(a["realized_headway_min"]) for a in zero]
    zero_label_invariant = max(zero_headways) - min(zero_headways) <= 1e-9

    leader_label_control = True
    for seconds in LATENCY_SECONDS:
        matching = [a for a in arms if a["follower_type"] == "HS" and a["follower_latency_seconds"] == seconds]
        if len(matching) != 2 or not isclose(float(matching[0]["realized_headway_min"]), float(matching[1]["realized_headway_min"]), abs_tol=1e-9):
            leader_label_control = False

    hs_sweep = [a for a in arms if a["leader_type"] == "AS" and a["follower_type"] == "HS"]
    hs_sweep.sort(key=lambda a: int(a["follower_latency_seconds"]))
    reaction_allowance_monotone = all(
        int(b["follower_latency_seconds"]) >= int(a["follower_latency_seconds"])
        for a, b in zip(hs_sweep, hs_sweep[1:])
    )
    all_success = all(a["terminal_status"] == "SUCCESS" for a in arms)
    order_preserved = all(bool(a["service_order_preserved"]) and int(a["arrival_order_inversion_count"]) == 0 for a in arms)
    no_reference_runtime_input = all(a["runtime_reference_headway_input"] is None for a in arms)
    safety_evidence_present = all(a["modeled_hull_overlap_episode_count"] is not None for a in arms)
    no_modeled_hull_overlap = safety_evidence_present and all(int(a["modeled_hull_overlap_episode_count"]) == 0 for a in arms)

    headway_span = max(float(a["realized_headway_min"]) for a in hs_sweep) - min(float(a["realized_headway_min"]) for a in hs_sweep)
    document = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "evidence_role": "ENGINEERING_QUALIFICATION_ONLY_NOT_REFERENCE_VALIDATION",
        "measurement": "LONGITUDINAL_REFERENCE_LINE_PASS_EQUIVALENT_CENTER_GATE_IN_CONTROLLED_SAME_ROUTE_FIXTURE",
        "reference_values_used_by_runtime": False,
        "arms": arms,
        "qualification": {
            "all_terminal_success": all_success,
            "service_order_preserved": order_preserved,
            "zero_latency_type_labels_physically_inert": zero_label_invariant,
            "leader_label_physically_inert_when_microscopic_parameters_equal": leader_label_control,
            "follower_reaction_allowance_sweep_ordered": reaction_allowance_monotone,
            "reference_values_absent_from_runtime_control": no_reference_runtime_input,
            "safety_evidence_present": safety_evidence_present,
            "no_modeled_hull_overlap": no_modeled_hull_overlap,
            "pass": all_success and order_preserved and zero_label_invariant and leader_label_control and reaction_allowance_monotone and no_reference_runtime_input and no_modeled_hull_overlap,
        },
        "observed_mechanism": {
            "hs_follower_latency_headway_span_min": headway_span,
            "arbitrary_2x2_pair_matrix_supported_by_current_latency_only_mechanism": False,
            "interpretation": "Type labels alone are inert; the qualified differentiator is follower-specific latency/reaction allowance, so leader-specific pair values are not independently controllable.",
        },
    }
    raw = _canonical_json(document)
    document["result_sha256"] = sha256(raw).hexdigest()
    return document
