"""Deterministic predeclared analysis for CROSS R1 P4B publication results.

Protocol V3 preserves the Protocol V2 primary measurement boundary (90 min) from the physical
execution/safety horizon (150 min).  The primary microscopic quantity is explicitly
*source-admission throughput*, i.e. upstream physically admissible inflow under the
frozen downstream interaction/backpressure model.  It is not silently renamed direct
crossing throughput or real-world channel capacity.

Primary estimand (P1 only), per replication r:

    G_T(r,d,p,e,k) = [T_CROSS_90(p)-T_CROSS_90(P00)]
                     - [T_M1_90(p)-T_M1_90(P00)]

Continuous G_T exists only when both microscopic rows are full-horizon SUCCESS.
MODELED_INFEASIBLE and execution/resource failures remain a separate terminal endpoint
and are never converted to numeric zero performance.
"""
from __future__ import annotations

from collections import Counter
import json
from math import isfinite, sqrt
from statistics import mean, stdev
from typing import Iterable, Mapping, Sequence

from scipy.stats import t as student_t

from vessel_crossing.research.r1_p4b_behavior_grounding import T0, T60
from vessel_crossing.research.r1_p4b_publication_execution import (
    P4B_EVALUATION_END_MIN,
    P4B_P2_SENTINEL_ROOM_DEMAND,
    P4B_PENETRATIONS,
    P4B_POINT_FORMAT,
    P4B_PRIMARY_MEASUREMENT_END_MIN,
    P4BPublicationExecutionPlan,
    P4BConceptualRow,
)
from vessel_crossing.research.r1_robustness_stage2 import (
    first_order_profile,
    tabulated_steerage_profile,
)

P4B_ANALYSIS_FORMAT = "cross-r1-p4b-publication-analysis"
P4B_ANALYSIS_SCHEMA_VERSION = 2
P4B_PRIMARY_FAMILY_SIZE = 72
P4B_ALPHA = 0.05
P4B_PRIMARY_METRIC = "primary_source_admission_throughput_per_hour_90"
P4B_PRIMARY_MEASUREMENT = "physical source admissions in [0,90] min / 1.5 h"
P4B_MACRO_MEASUREMENT = "M1 departures in [0,90] min / 1.5 h"
P4B_PRIMARY_NONZERO_PENETRATIONS = tuple(p for p in P4B_PENETRATIONS if p > 0.0)
_EPS = 1e-12

_COUNT_WORSE_HIGHER = (
    "modeled_hull_overlap_episode_count",
    "safety_domain_intrusion_episode_count",
    "near_miss_episode_count",
    "unresolved_episode_count",
)
_MARGIN_WORSE_LOWER = (
    "sampled_min_center_separation_nm",
    "sampled_min_hull_margin_nm",
    "sampled_min_safety_domain_margin_nm",
)


def _sign(value: float | None) -> int | None:
    if value is None:
        return None
    if abs(value) <= _EPS:
        return 0
    return 1 if value > 0 else -1


def _normalize_terminal(document: Mapping[str, object] | None) -> str:
    if document is None:
        return "MISSING_RESULT"
    terminal = document.get("terminal")
    if not isinstance(terminal, Mapping):
        return "MALFORMED_TERMINAL"
    status = str(terminal.get("status"))
    if status == "SUCCESS":
        try:
            terminal_time = float(terminal.get("terminal_time_min"))
        except (TypeError, ValueError):
            return "SUCCESS_MALFORMED_TIME"
        if abs(terminal_time - P4B_EVALUATION_END_MIN) <= _EPS:
            return "SUCCESS"
        return "SUCCESS_NOT_FULL_HORIZON"
    return status


def _performance_value(document: Mapping[str, object] | None, key: str) -> float | None:
    if _normalize_terminal(document) != "SUCCESS":
        return None
    performance = document.get("performance") if document is not None else None
    if not isinstance(performance, Mapping):
        return None
    value = performance.get(key)
    if not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None
    return float(value)


def _response_value(document: Mapping[str, object] | None, key: str) -> float | None:
    if _normalize_terminal(document) != "SUCCESS":
        return None
    response = document.get("response_diagnostics") if document is not None else None
    if not isinstance(response, Mapping):
        return None
    value = response.get(key)
    if not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None
    return float(value)


def _micro_throughput(document: Mapping[str, object] | None) -> float | None:
    return _performance_value(document, P4B_PRIMARY_METRIC)


def _safety_relation(
    candidate: Mapping[str, object] | None,
    reference: Mapping[str, object] | None,
) -> str:
    """Fieldwise deterministic relation of candidate safety to reference safety."""
    c = None if candidate is None else candidate.get("safety")
    r = None if reference is None else reference.get("safety")
    if c is None and r is None:
        return "IDENTICAL_NO_SUMMARY"
    if not isinstance(c, Mapping) or not isinstance(r, Mapping):
        return "UNAVAILABLE_OR_ASYMMETRIC_SUMMARY"
    worse = False
    better = False
    for field in _COUNT_WORSE_HIGHER:
        cv, rv = c.get(field), r.get(field)
        if not isinstance(cv, (int, float)) or not isinstance(rv, (int, float)):
            continue
        if float(cv) > float(rv) + _EPS:
            worse = True
        elif float(cv) < float(rv) - _EPS:
            better = True
    for field in _MARGIN_WORSE_LOWER:
        cv, rv = c.get(field), r.get(field)
        if not isinstance(cv, (int, float)) or not isinstance(rv, (int, float)):
            continue
        if float(cv) < float(rv) - _EPS:
            worse = True
        elif float(cv) > float(rv) + _EPS:
            better = True
    if worse and better:
        return "MIXED"
    if worse:
        return "CANDIDATE_WORSE_ONLY"
    if better:
        return "CANDIDATE_BETTER_ONLY"
    if c.get("worst_observed_evidence") != r.get("worst_observed_evidence"):
        return "EVIDENCE_LABEL_DIFF_ONLY"
    return "IDENTICAL"


def _index_results(
    plan: P4BPublicationExecutionPlan,
    documents: Iterable[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    points = {p.point_id: p for p in plan.physical_points}
    plan_sha256 = plan.identity_sha256
    indexed: dict[str, Mapping[str, object]] = {}
    for document in documents:
        if document.get("format") != P4B_POINT_FORMAT:
            raise ValueError("unexpected P4B result format")
        if document.get("campaign_id") != plan.campaign_id:
            raise ValueError("result campaign mismatch")
        if document.get("plan_sha256") != plan_sha256:
            raise ValueError("result plan mismatch")
        if document.get("source_archive_sha256") != plan.source_archive_sha256:
            raise ValueError("result source mismatch")
        point_obj = document.get("point")
        if not isinstance(point_obj, Mapping):
            raise ValueError("result point metadata missing")
        point_id = point_obj.get("point_id")
        if not isinstance(point_id, str) or point_id not in points:
            raise ValueError("result references unknown physical point")
        if point_id in indexed:
            raise ValueError("duplicate physical result document")
        if point_obj.get("identity_sha256") != points[point_id].identity_sha256:
            raise ValueError("result physical point identity mismatch")
        indexed[point_id] = document
    return indexed


def _row_lookup(plan: P4BPublicationExecutionPlan) -> dict[tuple[object, ...], P4BConceptualRow]:
    return {
        (r.replication, r.plant_id, r.room_nm, r.demand_per_hour, r.timing_endpoint_id, r.penetration): r
        for r in plan.conceptual_rows
    }


def _macro_lookup(plan: P4BPublicationExecutionPlan) -> dict[tuple[int, float, float], Mapping[str, object]]:
    lookup: dict[tuple[int, float, float], Mapping[str, object]] = {}
    for basis in plan.input_bases:
        for row in basis.macro_by_penetration:
            lookup[(basis.replication, basis.demand_per_hour, float(row["penetration"]))] = row
    return lookup


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    n = len(values)
    if n == 0:
        return {
            "n_available": 0, "mean": None, "sample_sd": None, "standard_error": None,
            "ci95_low": None, "ci95_high": None, "raw_p_value": None,
        }
    mu = mean(values)
    if n < 2:
        return {
            "n_available": n, "mean": mu, "sample_sd": None, "standard_error": None,
            "ci95_low": None, "ci95_high": None, "raw_p_value": None,
        }
    if all(v == values[0] for v in values):
        raw_p = 1.0 if values[0] == 0.0 else 0.0
        return {
            "n_available": n, "mean": mu, "sample_sd": 0.0, "standard_error": 0.0,
            "ci95_low": mu, "ci95_high": mu, "raw_p_value": raw_p,
        }
    sd = stdev(values)
    se = sd / sqrt(n)
    critical = float(student_t.ppf(0.975, df=n - 1))
    half = critical * se
    raw_p = float(2.0 * student_t.sf(abs(mu / se), df=n - 1))
    return {
        "n_available": n, "mean": mu, "sample_sd": sd, "standard_error": se,
        "ci95_low": mu - half, "ci95_high": mu + half, "raw_p_value": raw_p,
    }


def _descriptive_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    row = _summary(values)
    row.pop("raw_p_value", None)
    return row


def _holm_adjust(raw_p_values: Sequence[float | None]) -> list[float]:
    if len(raw_p_values) != P4B_PRIMARY_FAMILY_SIZE:
        raise ValueError("Holm family must contain all 72 primary contrasts")
    family = [1.0 if p is None else float(p) for p in raw_p_values]
    order = sorted(range(len(family)), key=lambda i: (family[i], i))
    adjusted = [1.0] * len(family)
    running = 0.0
    m = len(family)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * family[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def _primary_replication_value(
    *,
    plan: P4BPublicationExecutionPlan,
    result_map: Mapping[str, Mapping[str, object]],
    rows: Mapping[tuple[object, ...], P4BConceptualRow],
    macros: Mapping[tuple[int, float, float], Mapping[str, object]],
    replication: int,
    room: float,
    demand: float,
    endpoint: str,
    penetration: float,
    plant_id: str,
) -> dict[str, object]:
    p_row = rows[(replication, plant_id, room, demand, endpoint, penetration)]
    b_row = rows[(replication, plant_id, room, demand, endpoint, 0.0)]
    p_doc = result_map.get(p_row.physical_point_id)
    b_doc = result_map.get(b_row.physical_point_id)
    p_status = _normalize_terminal(p_doc)
    b_status = _normalize_terminal(b_doc)
    p_micro = _micro_throughput(p_doc)
    b_micro = _micro_throughput(b_doc)
    macro_p = float(macros[(replication, demand, penetration)]["primary_throughput_per_hour"])
    macro_b = float(macros[(replication, demand, 0.0)]["primary_throughput_per_hour"])
    value = None
    micro_gain = None
    macro_gain = macro_p - macro_b
    if p_micro is not None and b_micro is not None:
        micro_gain = p_micro - b_micro
        value = micro_gain - macro_gain
    return {
        "replication": replication,
        "penetration_status": p_status,
        "baseline_status": b_status,
        "micro_gain_per_hour": micro_gain,
        "macro_gain_per_hour": macro_gain,
        "g_t_per_hour": value,
        "autonomy_safety_relation": _safety_relation(p_doc, b_doc),
    }


def _gain(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _secondary_replication_value(
    *,
    result_map: Mapping[str, Mapping[str, object]],
    rows: Mapping[tuple[object, ...], P4BConceptualRow],
    macros: Mapping[tuple[int, float, float], Mapping[str, object]],
    replication: int,
    room: float,
    demand: float,
    endpoint: str,
    penetration: float,
    plant_id: str,
) -> dict[str, object]:
    p_row = rows[(replication, plant_id, room, demand, endpoint, penetration)]
    b_row = rows[(replication, plant_id, room, demand, endpoint, 0.0)]
    p_doc = result_map.get(p_row.physical_point_id)
    b_doc = result_map.get(b_row.physical_point_id)

    p_backlog = _performance_value(p_doc, "primary_source_backlog_count_90")
    b_backlog = _performance_value(b_doc, "primary_source_backlog_count_90")
    macro_p = macros[(replication, demand, penetration)]
    macro_b = macros[(replication, demand, 0.0)]
    macro_backlog_gain = float(macro_p["primary_backlog_count"]) - float(macro_b["primary_backlog_count"])
    micro_backlog_gain = _gain(p_backlog, b_backlog)
    backlog_discrepancy = None if micro_backlog_gain is None else micro_backlog_gain - macro_backlog_gain

    p_wait = _performance_value(p_doc, "cohort_mean_source_wait_min_if_complete_by_150")
    b_wait = _performance_value(b_doc, "cohort_mean_source_wait_min_if_complete_by_150")
    macro_wait_gain = float(macro_p["primary_mean_waiting_min"]) - float(macro_b["primary_mean_waiting_min"])
    micro_wait_gain = _gain(p_wait, b_wait)
    wait_discrepancy = None if micro_wait_gain is None else micro_wait_gain - macro_wait_gain

    return {
        "replication": replication,
        "backlog_gain_discrepancy_count": backlog_discrepancy,
        "source_wait_gain_discrepancy_min": wait_discrepancy,
        "busy_headway_gain_min_90": _gain(
            _performance_value(p_doc, "primary_busy_admission_headway_mean_min_90"),
            _performance_value(b_doc, "primary_busy_admission_headway_mean_min_90"),
        ),
        "route_completion_count_gain_150": _gain(
            _performance_value(p_doc, "evaluation_route_completed_count_150"),
            _performance_value(b_doc, "evaluation_route_completed_count_150"),
        ),
        "mean_route_duration_gain_min_150": _gain(
            _performance_value(p_doc, "evaluation_mean_route_duration_min_150"),
            _performance_value(b_doc, "evaluation_mean_route_duration_min_150"),
        ),
        "mean_system_duration_gain_min_150": _gain(
            _performance_value(p_doc, "evaluation_mean_system_duration_min_150"),
            _performance_value(b_doc, "evaluation_mean_system_duration_min_150"),
        ),
        "rebootstrap_count_gain": _gain(
            _response_value(p_doc, "rebootstrap_count"),
            _response_value(b_doc, "rebootstrap_count"),
        ),
        "fleet_observation_count_gain": _gain(
            _response_value(p_doc, "fleet_observation_count"),
            _response_value(b_doc, "fleet_observation_count"),
        ),
    }


def _metric_summary(records: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    values = [float(r[key]) for r in records if isinstance(r.get(key), (int, float))]
    return {"metric": key, **_descriptive_summary(values)}


def analyze_publication_results(
    plan: P4BPublicationExecutionPlan,
    result_documents: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Analyze P4B protocol-V2 results deterministically under frozen conventions."""
    result_map = _index_results(plan, result_documents)
    rows = _row_lookup(plan)
    macros = _macro_lookup(plan)
    p1 = first_order_profile().plant_id
    p2 = tabulated_steerage_profile().plant_id

    primary: list[dict[str, object]] = []
    secondary: list[dict[str, object]] = []
    for endpoint in (T0.endpoint_id, T60.endpoint_id):
        for room in (1.300, 1.375, 1.600):
            for demand in (4.0, 6.0, 8.0):
                for penetration in P4B_PRIMARY_NONZERO_PENETRATIONS:
                    reps = [
                        _primary_replication_value(
                            plan=plan, result_map=result_map, rows=rows, macros=macros,
                            replication=rep, room=room, demand=demand, endpoint=endpoint,
                            penetration=penetration, plant_id=p1,
                        )
                        for rep in range(1, plan.replication_count + 1)
                    ]
                    values = [float(x["g_t_per_hour"]) for x in reps if x["g_t_per_hour"] is not None]
                    summary = _summary(values)
                    terminal_counts = Counter(f'{x["penetration_status"]}|{x["baseline_status"]}' for x in reps)
                    contrast_id = (
                        f"P1-{endpoint}-R{int(round(room*1000)):04d}-D{int(demand):02d}-"
                        f"P{int(penetration*100):03d}"
                    )
                    primary.append({
                        "contrast_id": contrast_id,
                        "timing_endpoint_id": endpoint,
                        "room_nm": room,
                        "demand_per_hour": demand,
                        "penetration": penetration,
                        "n_planned": plan.replication_count,
                        **summary,
                        "terminal_pair_counts": dict(sorted(terminal_counts.items())),
                        "replication_values": reps,
                    })
                    sec_records = [
                        _secondary_replication_value(
                            result_map=result_map, rows=rows, macros=macros,
                            replication=rep, room=room, demand=demand, endpoint=endpoint,
                            penetration=penetration, plant_id=p1,
                        )
                        for rep in range(1, plan.replication_count + 1)
                    ]
                    keys = (
                        "backlog_gain_discrepancy_count",
                        "source_wait_gain_discrepancy_min",
                        "busy_headway_gain_min_90",
                        "route_completion_count_gain_150",
                        "mean_route_duration_gain_min_150",
                        "mean_system_duration_gain_min_150",
                        "rebootstrap_count_gain",
                        "fleet_observation_count_gain",
                    )
                    secondary.append({
                        "contrast_id": contrast_id,
                        "timing_endpoint_id": endpoint,
                        "room_nm": room,
                        "demand_per_hour": demand,
                        "penetration": penetration,
                        "inference_role": "SUPPORTING_DESCRIPTIVE_NOT_IN_PRIMARY_HOLM_FAMILY",
                        "summaries": {key: _metric_summary(sec_records, key) for key in keys},
                        "replication_values": sec_records,
                    })

    if len(primary) != P4B_PRIMARY_FAMILY_SIZE:
        raise AssertionError("primary family construction drift")
    if len(secondary) != P4B_PRIMARY_FAMILY_SIZE:
        raise AssertionError("secondary family construction drift")
    adjusted = _holm_adjust([row["raw_p_value"] for row in primary])
    for row, adj in zip(primary, adjusted):
        row["holm_adjusted_p_value"] = adj
        row["formal_status"] = (
            "RESOLVED_NONZERO"
            if row["raw_p_value"] is not None and adj <= P4B_ALPHA
            else "NOT_RESOLVED_AT_CURRENT_PRECISION"
        )
        row["equivalence_conclusion_permitted"] = False

    primary_by_key = {
        (r["timing_endpoint_id"], r["room_nm"], r["demand_per_hour"], r["penetration"]): r
        for r in primary
    }
    timing_flags: list[dict[str, object]] = []
    for room in (1.300, 1.375, 1.600):
        for demand in (4.0, 6.0, 8.0):
            for penetration in P4B_PRIMARY_NONZERO_PENETRATIONS:
                a = primary_by_key[(T0.endpoint_id, room, demand, penetration)]
                b = primary_by_key[(T60.endpoint_id, room, demand, penetration)]
                a_rep = {x["replication"]: x for x in a["replication_values"]}
                b_rep = {x["replication"]: x for x in b["replication_values"]}
                terminal_disagreement = sum(
                    (a_rep[r]["penetration_status"], a_rep[r]["baseline_status"])
                    != (b_rep[r]["penetration_status"], b_rep[r]["baseline_status"])
                    for r in a_rep
                )
                safety_disagreement = sum(
                    a_rep[r]["autonomy_safety_relation"] != b_rep[r]["autonomy_safety_relation"]
                    for r in a_rep
                )
                opposite = (
                    _sign(a["mean"]) is not None and _sign(b["mean"]) is not None
                    and _sign(a["mean"]) * _sign(b["mean"]) == -1
                )
                resolved_disagreement = a["formal_status"] != b["formal_status"]
                timing_flags.append({
                    "room_nm": room,
                    "demand_per_hour": demand,
                    "penetration": penetration,
                    "t0_contrast_id": a["contrast_id"],
                    "t60_contrast_id": b["contrast_id"],
                    "opposite_primary_mean_signs": opposite,
                    "resolution_status_disagreement": resolved_disagreement,
                    "terminal_disagreement_replications": terminal_disagreement,
                    "safety_relation_disagreement_replications": safety_disagreement,
                    "behavior_timing_sensitive": bool(
                        opposite or resolved_disagreement or terminal_disagreement or safety_disagreement
                    ),
                })

    # P2 sentinel diagnostics include P00 explicitly.  Autonomy-gain sign comparison is
    # algebraically non-applicable at P00, but direct terminal/safety comparison remains important.
    sentinel: list[dict[str, object]] = []
    for endpoint in (T0.endpoint_id, T60.endpoint_id):
        for room, demand in P4B_P2_SENTINEL_ROOM_DEMAND:
            for penetration in (0.0, 0.5, 1.0):
                records = []
                for rep in range(1, plan.replication_count + 1):
                    p1row = rows[(rep, p1, room, demand, endpoint, penetration)]
                    p2row = rows[(rep, p2, room, demand, endpoint, penetration)]
                    p1doc = result_map.get(p1row.physical_point_id)
                    p2doc = result_map.get(p2row.physical_point_id)
                    p1g = p2g = None
                    sign_change = None
                    if penetration > 0.0:
                        p1v = _primary_replication_value(
                            plan=plan, result_map=result_map, rows=rows, macros=macros,
                            replication=rep, room=room, demand=demand, endpoint=endpoint,
                            penetration=penetration, plant_id=p1,
                        )
                        p2v = _primary_replication_value(
                            plan=plan, result_map=result_map, rows=rows, macros=macros,
                            replication=rep, room=room, demand=demand, endpoint=endpoint,
                            penetration=penetration, plant_id=p2,
                        )
                        p1g, p2g = p1v["g_t_per_hour"], p2v["g_t_per_hour"]
                        s1, s2 = _sign(p1g), _sign(p2g)
                        sign_change = s1 is not None and s2 is not None and s1 * s2 == -1
                    records.append({
                        "replication": rep,
                        "p1_terminal": _normalize_terminal(p1doc),
                        "p2_terminal": _normalize_terminal(p2doc),
                        "plant_safety_relation_p2_vs_p1": _safety_relation(p2doc, p1doc),
                        "p1_g_t_per_hour": p1g,
                        "p2_g_t_per_hour": p2g,
                        "g_t_sign_change": sign_change,
                    })
                applicable = penetration > 0.0
                sentinel.append({
                    "timing_endpoint_id": endpoint,
                    "room_nm": room,
                    "demand_per_hour": demand,
                    "penetration": penetration,
                    "terminal_class_difference_replications": sum(
                        r["p1_terminal"] != r["p2_terminal"] for r in records
                    ),
                    "safety_relation_counts": dict(sorted(Counter(
                        r["plant_safety_relation_p2_vs_p1"] for r in records
                    ).items())),
                    "autonomy_gain_sign_comparison_applicable": applicable,
                    "autonomy_gain_sign_change_replications": (
                        sum(r["g_t_sign_change"] is True for r in records) if applicable else None
                    ),
                    "replication_records": records,
                    "global_plant_equivalence_permitted": False,
                })

    return {
        "format": P4B_ANALYSIS_FORMAT,
        "schema_version": P4B_ANALYSIS_SCHEMA_VERSION,
        "campaign_id": plan.campaign_id,
        "protocol_id": plan.protocol_id,
        "plan_sha256": plan.identity_sha256,
        "source_archive_sha256": plan.source_archive_sha256,
        "measurement": {
            "cross_primary": P4B_PRIMARY_MEASUREMENT,
            "macro_primary": P4B_MACRO_MEASUREMENT,
            "metric_key": P4B_PRIMARY_METRIC,
            "warmup_min": 0.0,
            "primary_measurement_end_min": P4B_PRIMARY_MEASUREMENT_END_MIN,
            "evaluation_end_min": P4B_EVALUATION_END_MIN,
            "cross_quantity_scope": "UPSTREAM_PHYSICAL_SOURCE_ADMISSIBLE_INFLOW_NOT_LITERAL_CROSSING_CAPACITY",
        },
        "result_documents_present": len(result_map),
        "result_documents_expected": len(plan.physical_points),
        "primary_family_size": P4B_PRIMARY_FAMILY_SIZE,
        "primary_contrasts": primary,
        "secondary_supporting_contrasts": secondary,
        "timing_sensitivity_flags": timing_flags,
        "p2_sentinel_diagnostics": sentinel,
        "interpretation_guards": {
            "modeled_infeasible_is_zero_throughput": False,
            "t0_t60_pooling_permitted": False,
            "p2_is_co_primary": False,
            "secondary_metrics_are_co_primary": False,
            "non_significance_implies_equivalence": False,
            "source_admission_equals_literal_crossing_throughput": False,
            "t60_is_calibrated_human_reaction_time": False,
        },
    }


def analysis_to_json_bytes(analysis: Mapping[str, object]) -> bytes:
    return (json.dumps(analysis, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
