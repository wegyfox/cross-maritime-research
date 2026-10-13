"""Reproduce the frozen R2 ideal-gap decomposition from authoritative result archives.

This module is R2-specific analysis glue, not a report renderer and not simulator runtime.
It reads immutable FS2/B4/B5/B6 result archives, computes the qualified reference-envelope
summary, and returns the generic ``cross-ideal-gap-decomposition-v1`` object consumed by
``vessel_crossing.reporting.reference_envelope``.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping
from zipfile import BadZipFile, ZipFile

from scipy.stats import t as student_t

_R2_ARMS = ("NEG050", "ZERO", "POS050")
_EXPECTED_REPLICATIONS = 48
_FS2_RHO_ZERO_RE = re.compile(r"^results/FS2-R(?P<rep>\d{3})-RZERO-(?P<arm>NEG050|ZERO|POS050)\.json$")
_MANAGED_RE = re.compile(r"^results/(?P<level>B4|B5)-FS2-R(?P<rep>\d{3})-RZERO-(?P<arm>NEG050|ZERO|POS050)\.json$")
_B6V1_RE = re.compile(r"^results/B6-(?P<arm>NEG050|ZERO|POS050)\.json$")
_B6V2_RE = re.compile(r"^results/B6V2-(?P<arm>NEG050|ZERO|POS050)\.json$")


class R2ReferenceEnvelopeAnalysisError(ValueError):
    """Raised when authoritative R2 archives cannot support the frozen analysis."""


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_archive(
    path: str | Path,
    *,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    path = Path(path)
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise R2ReferenceEnvelopeAnalysisError(
            f"archive SHA-256 mismatch for {path.name}: {actual_sha} != {expected_sha256}"
        )
    try:
        with ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise R2ReferenceEnvelopeAnalysisError(
                    f"ZIP CRC failure in {path.name}: {bad}"
                )
            documents: dict[str, Any] = {}
            for name in archive.namelist():
                if not name.endswith(".json"):
                    continue
                try:
                    documents[name] = json.loads(archive.read(name))
                except json.JSONDecodeError as exc:
                    raise R2ReferenceEnvelopeAnalysisError(
                        f"invalid JSON in {path.name}:{name}: {exc}"
                    ) from exc
    except (OSError, BadZipFile) as exc:
        raise R2ReferenceEnvelopeAnalysisError(f"cannot read {path}: {exc}") from exc
    return documents, actual_sha


def _percentile(values: Iterable[float], q: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise R2ReferenceEnvelopeAnalysisError("cannot compute percentile of empty data")
    position = (len(rows) - 1) * q
    low = int(position)
    high = min(low + 1, len(rows) - 1)
    fraction = position - low
    return rows[low] * (1.0 - fraction) + rows[high] * fraction


def _fixed_window_center_yield(document: Mapping[str, Any]) -> float:
    center = document.get("center_service")
    if not isinstance(center, Mapping):
        raise R2ReferenceEnvelopeAnalysisError("FS2 center_service evidence is missing")
    records = center.get("records")
    if not isinstance(records, list):
        raise R2ReferenceEnvelopeAnalysisError("FS2 center_service.records is missing")
    count = sum(
        1
        for row in records
        if isinstance(row, Mapping)
        and isinstance(row.get("passage_time_min"), (int, float))
        and float(row["passage_time_min"]) <= 90.0 + 1e-12
    )
    return 60.0 * count / 90.0


def _nominal_route_duration(document: Mapping[str, Any]) -> float:
    conflict = document.get("conflict")
    if not isinstance(conflict, Mapping):
        raise R2ReferenceEnvelopeAnalysisError("conflict evidence is missing")
    value = conflict.get("nominal_route_duration_min")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise R2ReferenceEnvelopeAnalysisError("nominal route duration is missing")
    return float(value)


def _route_extra_rows(document: Mapping[str, Any]) -> list[float]:
    route = document.get("route")
    if not isinstance(route, Mapping) or not isinstance(route.get("records"), list):
        raise R2ReferenceEnvelopeAnalysisError("route records are missing")
    nominal = _nominal_route_duration(document)
    return [float(row["route_duration_min"]) - nominal for row in route["records"]]


def _admitted_wait_rows(document: Mapping[str, Any]) -> list[float]:
    source = document.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("admissions"), list):
        raise R2ReferenceEnvelopeAnalysisError("source admissions are missing")
    return [float(row["waiting_time_min"]) for row in source["admissions"]]


def _aggregate_runs(
    documents: list[Mapping[str, Any]],
    *,
    managed: bool,
) -> dict[str, Any]:
    waits: list[float] = []
    route_extra: list[float] = []
    events: Counter[str] = Counter()
    bypass_total = 0
    rows: list[dict[str, Any]] = []
    for document in documents:
        source = document["source"]
        route = document["route"]
        safety = document["safety"]
        terminal = document["terminal"]
        waits.extend(_admitted_wait_rows(document))
        route_extra.extend(_route_extra_rows(document))
        if managed:
            primary = document["primary_kpi"]["fixed_window_service_yield_per_hour"]
            controller = document["controller"]
            events.update(row["event"] for row in controller["trace"])
            bypass_total += sum(int(value) for value in controller["bypass_count"].values())
            trace_count = int(controller["trace_count"])
        else:
            primary = _fixed_window_center_yield(document)
            trace_count = 0
        rows.append(
            {
                "kpi": float(primary),
                "status": terminal["status"],
                "demand": int(source["demand_count"]),
                "admissions": int(source["admission_count"]),
                "completed": int(route["completion_count"]),
                "pending": int(source["pending_count"]),
                "due_pending": int(source["due_pending_count"]),
                "near": int(safety["near_miss_episode_count"]),
                "intrusion": int(safety["safety_domain_intrusion_episode_count"]),
                "hull": int(safety["modeled_hull_overlap_episode_count"]),
                "min_domain": float(safety["sampled_min_safety_domain_margin_nm"]),
                "trace": trace_count,
            }
        )

    if len(rows) != _EXPECTED_REPLICATIONS:
        raise R2ReferenceEnvelopeAnalysisError(
            f"expected {_EXPECTED_REPLICATIONS} replications, got {len(rows)}"
        )
    total_demand = sum(row["demand"] for row in rows)
    total_admissions = sum(row["admissions"] for row in rows)
    total_completed = sum(row["completed"] for row in rows)
    result: dict[str, Any] = {
        "replications": len(rows),
        "mean_service_yield_per_hour": statistics.mean(row["kpi"] for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "total_demand": total_demand,
        "total_admissions": total_admissions,
        "admission_fraction": total_admissions / total_demand,
        "total_completions": total_completed,
        "completion_fraction": total_completed / total_demand,
        "total_pending": sum(row["pending"] for row in rows),
        "total_due_pending": sum(row["due_pending"] for row in rows),
        "mean_admitted_wait_min": statistics.mean(waits),
        "p95_admitted_wait_min": _percentile(waits, 0.95),
        "mean_route_extra_min_completed": statistics.mean(route_extra),
        "near_miss_episode_count": sum(row["near"] for row in rows),
        "safety_domain_intrusion_episode_count": sum(row["intrusion"] for row in rows),
        "modeled_hull_overlap_episode_count": sum(row["hull"] for row in rows),
        "median_run_min_safety_domain_margin_nm": statistics.median(
            row["min_domain"] for row in rows
        ),
    }
    if managed:
        result.update(
            {
                "controller_trace_total": sum(row["trace"] for row in rows),
                "controller_trace_per_admission": sum(row["trace"] for row in rows)
                / total_admissions,
                "controller_event_counts": dict(events),
                "bypass_total": bypass_total,
            }
        )
    return result


def _paired_effect(
    baseline_by_replication: Mapping[int, Mapping[str, Any]],
    managed_by_replication: Mapping[int, Mapping[str, Any]],
) -> tuple[float, list[float], list[int]]:
    replications = sorted(baseline_by_replication)
    if replications != sorted(managed_by_replication) or len(replications) != _EXPECTED_REPLICATIONS:
        raise R2ReferenceEnvelopeAnalysisError("paired replication identities do not match")
    differences = [
        float(managed_by_replication[rep]["primary_kpi"]["fixed_window_service_yield_per_hour"])
        - _fixed_window_center_yield(baseline_by_replication[rep])
        for rep in replications
    ]
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    half = float(student_t.ppf(0.975, len(differences) - 1)) * sd / (len(differences) ** 0.5)
    positive = sum(value > 1e-12 for value in differences)
    negative = sum(value < -1e-12 for value in differences)
    zero = len(differences) - positive - negative
    return mean, [mean - half, mean + half], [positive, zero, negative]


def _b6v2_metrics(document: Mapping[str, Any]) -> dict[str, Any]:
    headways = [float(value) for value in document["conflict"]["center_passage_headway_min"]]
    if not headways:
        raise R2ReferenceEnvelopeAnalysisError("B6v2 headway evidence is empty")
    mean_headway = statistics.mean(headways)
    return {
        "deterministic": True,
        "terminal_status": document["terminal"]["status"],
        "fixed_window_service_yield_per_hour": float(
            document["primary_kpi"]["fixed_window_service_yield_per_hour"]
        ),
        "measurement_passage_count": int(document["primary_kpi"]["passage_count"]),
        "measurement_window_min": list(document["primary_kpi"]["window_min"]),
        "admission_count": int(document["source"]["admission_count"]),
        "admission_count_by_stream": dict(document["fairness"]["admission_count_by_stream"]),
        "measurement_passage_count_by_stream": dict(
            document["fairness"]["measurement_passage_count_by_stream"]
        ),
        "max_prefix_admission_count_imbalance": int(
            document["fairness"]["max_prefix_admission_count_imbalance"]
        ),
        "strict_balance_gate_pass": bool(document["fairness"]["strict_balance_gate_pass"]),
        "pending_count": int(document["source"]["pending_count"]),
        "mean_admitted_wait_min": float(document["source"]["mean_admitted_waiting_time_min"]),
        "completed_count": int(document["lifecycle"]["completed_count"]),
        "near_miss_episode_count": int(document["safety"]["near_miss_episode_count"]),
        "safety_domain_intrusion_episode_count": int(
            document["safety"]["safety_domain_intrusion_episode_count"]
        ),
        "modeled_hull_overlap_episode_count": int(
            document["safety"]["modeled_hull_overlap_episode_count"]
        ),
        "sampled_min_safety_domain_margin_nm": float(
            document["safety"]["sampled_min_safety_domain_margin_nm"]
        ),
        "mean_center_headway_min": mean_headway,
        "headway_implied_asymptotic_rate_per_hour": 60.0 / mean_headway,
        "bypass_total": sum(int(value) for value in document["controller"]["bypass_count"].values()),
        "stochastic_confidence_interval": False,
        "field_calibrated_capacity": False,
        "deployment_safety": False,
    }


def _stream_counts(document: Mapping[str, Any]) -> Counter[str]:
    return Counter(row["stream_id"] for row in document["source"]["admissions"])


def build_r2_ideal_gap_decomposition(
    fs2_archive: str | Path,
    managed_archive: str | Path,
    b6v2_archive: str | Path,
    *,
    expected_fs2_sha256: str | None = None,
    expected_managed_sha256: str | None = None,
    expected_b6v2_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the frozen R2 decomposition and a separate archive-provenance record."""

    fs2_docs, fs2_sha = _load_archive(fs2_archive, expected_sha256=expected_fs2_sha256)
    managed_docs, managed_sha = _load_archive(
        managed_archive, expected_sha256=expected_managed_sha256
    )
    b6v2_docs, b6v2_sha = _load_archive(b6v2_archive, expected_sha256=expected_b6v2_sha256)

    fs2: dict[str, dict[int, Mapping[str, Any]]] = {arm: {} for arm in _R2_ARMS}
    for name, document in fs2_docs.items():
        match = _FS2_RHO_ZERO_RE.match(name)
        if not match:
            continue
        arm = match.group("arm")
        rep = int(match.group("rep"))
        if float(document["point"]["rho"]) != 0.0 or document["point"]["delta_arm"] != arm:
            raise R2ReferenceEnvelopeAnalysisError(f"FS2 identity mismatch: {name}")
        fs2[arm][rep] = document

    managed: dict[str, dict[str, dict[int, Mapping[str, Any]]]] = {
        arm: {"B4": {}, "B5": {}} for arm in _R2_ARMS
    }
    b6v1: dict[str, Mapping[str, Any]] = {}
    for name, document in managed_docs.items():
        match = _MANAGED_RE.match(name)
        if match:
            arm = match.group("arm")
            level = match.group("level")
            rep = int(match.group("rep"))
            if document["delta_arm"] != arm or document["level"] != level:
                raise R2ReferenceEnvelopeAnalysisError(f"managed identity mismatch: {name}")
            parent = document.get("parent_fs2")
            if not isinstance(parent, Mapping) or int(parent["replication"]) != rep or float(parent["rho"]) != 0.0:
                raise R2ReferenceEnvelopeAnalysisError(f"managed parent mismatch: {name}")
            managed[arm][level][rep] = document
            continue
        match = _B6V1_RE.match(name)
        if match:
            b6v1[match.group("arm")] = document

    b6v2: dict[str, Mapping[str, Any]] = {}
    for name, document in b6v2_docs.items():
        match = _B6V2_RE.match(name)
        if match:
            arm = match.group("arm")
            if document["delta_arm"] != arm:
                raise R2ReferenceEnvelopeAnalysisError(f"B6v2 identity mismatch: {name}")
            b6v2[arm] = document

    for arm in _R2_ARMS:
        if len(fs2[arm]) != _EXPECTED_REPLICATIONS:
            raise R2ReferenceEnvelopeAnalysisError(f"FS2 {arm}: expected 48 rho=0 results")
        for level in ("B4", "B5"):
            if len(managed[arm][level]) != _EXPECTED_REPLICATIONS:
                raise R2ReferenceEnvelopeAnalysisError(f"{level} {arm}: expected 48 results")
        if arm not in b6v1 or arm not in b6v2:
            raise R2ReferenceEnvelopeAnalysisError(f"missing B6/B6v2 result for {arm}")

    # Preserve the scientifically awkward B6v1 outcome instead of discarding it.
    b6v1_starves = all(len(_stream_counts(b6v1[arm])) == 1 for arm in _R2_ARMS)
    if not b6v1_starves:
        raise R2ReferenceEnvelopeAnalysisError(
            "frozen R2 B6v1 disposition no longer matches the authoritative starvation evidence"
        )

    actual: dict[str, Any] = {}
    reference_ladder: dict[str, Any] = {}
    efficiency: dict[str, Any] = {}
    losses: dict[str, Any] = {}
    paired_effects: dict[str, float] = {}
    b6_metrics_by_arm: dict[str, Any] = {}
    b5_swap_total = 0

    for arm in _R2_ARMS:
        fs2_rows = [fs2[arm][rep] for rep in sorted(fs2[arm])]
        b4_rows = [managed[arm]["B4"][rep] for rep in sorted(managed[arm]["B4"])]
        b5_rows = [managed[arm]["B5"][rep] for rep in sorted(managed[arm]["B5"])]
        b0_metrics = _aggregate_runs(fs2_rows, managed=False)
        b4_metrics = _aggregate_runs(b4_rows, managed=True)
        b5_metrics = _aggregate_runs(b5_rows, managed=True)
        b6_metrics = _b6v2_metrics(b6v2[arm])
        b6_metrics_by_arm[arm] = b6_metrics
        b5_swap_total += int(b5_metrics["controller_event_counts"].get("B5_ADJACENT_SWAP", 0))

        c0 = float(b0_metrics["mean_service_yield_per_hour"])
        c4 = float(b4_metrics["mean_service_yield_per_hour"])
        c5 = float(b5_metrics["mean_service_yield_per_hour"])
        c6 = float(b6_metrics["fixed_window_service_yield_per_hour"])
        paired_mean, ci95, signs = _paired_effect(fs2[arm], managed[arm]["B4"])
        # Frozen R2 summary reports the effect as the difference of the two
        # arm-level means.  The paired mean is used only to center the paired CI.
        effect = c4 - c0
        if abs(effect - paired_mean) > 1e-12:
            raise R2ReferenceEnvelopeAnalysisError("paired and aggregate B4 effects disagree")
        paired_effects[arm] = paired_mean

        actual[arm] = b0_metrics
        reference_ladder[arm] = {
            "B1_dedicated_corridor": {
                "identity_of": "B0",
                "reason": "CORRIDOR_EXTERNAL_INTRUSION_FACTOR_NOT_MODELED_IN_R2",
                "service_yield_per_hour": c0,
            },
            "B2_controlled_entry": {
                "identity_of": "B1",
                "reason": "EARLIEST_FEASIBLE_SOURCE_METERING_ALREADY_CANONICAL_IN_R2",
                "service_yield_per_hour": c0,
            },
            "B3_speed_harmonization": {
                "identity_of": "B2",
                "reason": "SPEED_PLANT_HETEROGENEITY_NOT_MODELED_IN_R2",
                "service_yield_per_hour": c0,
            },
            "B4_coordination": b4_metrics,
            "B5_platoon_structure": b5_metrics,
            "B6_balanced_saturated_reference": b6_metrics,
        }
        efficiency[arm] = {
            "C_actual": c0,
            "C_managed_current_demand": c5,
            "C_balanced_saturated_reference": c6,
            "eta_managed": c0 / c5,
            "eta_reference": c0 / c6,
            "managed_recoverable_gap_abs": c5 - c0,
            "managed_recoverable_gain_pct_of_actual": 100.0 * (c5 - c0) / c0,
            "actual_to_reference_gap_abs": c6 - c0,
            "actual_to_reference_gap_pct_of_reference": 100.0 * (c6 - c0) / c6,
            "managed_level_fraction_of_reference": c5 / c6,
        }
        losses[arm] = {
            "B0_to_B1_external_corridor_exclusion": 0.0,
            "B1_to_B2_admission_organization": 0.0,
            "B2_to_B3_speed_plant_harmonization": 0.0,
            "B3_to_B4_global_slot_coordination": {
                "mean_paired_effect_service_yield_per_hour": effect,
                "paired_ci95": ci95,
                "positive_zero_negative_blocks": signs,
            },
            "B4_to_B5_type_order_organization": {
                "mean_effect_service_yield_per_hour": c5 - c4,
                "activation_note": (
                    "near-zero activation at tested stochastic demand; one swap total "
                    "across all arms, in POS050"
                ),
            },
            "B5_to_B6_residual_to_balanced_saturation": {
                "difference_service_yield_per_hour": c6 - c5,
                "interpretation": (
                    "bundled residual to deterministic balanced saturation; not a paired "
                    "causal effect and not attributable to one mechanism"
                ),
            },
        }

    if b5_swap_total != 1:
        raise R2ReferenceEnvelopeAnalysisError(
            f"frozen B5 activation evidence expected exactly one swap, got {b5_swap_total}"
        )

    b6_values = [
        float(b6_metrics_by_arm[arm]["fixed_window_service_yield_per_hour"]) for arm in _R2_ARMS
    ]
    if max(b6_values) - min(b6_values) > 1e-12:
        raise R2ReferenceEnvelopeAnalysisError("B6v2 service yield is not common across Delta arms")
    headways = [float(b6_metrics_by_arm[arm]["mean_center_headway_min"]) for arm in _R2_ARMS]
    if max(headways) - min(headways) > 1e-12:
        raise R2ReferenceEnvelopeAnalysisError("B6v2 mean headway is not common across Delta arms")
    if not all(bool(b6_metrics_by_arm[arm]["strict_balance_gate_pass"]) for arm in _R2_ARMS):
        raise R2ReferenceEnvelopeAnalysisError("B6v2 strict balance gate did not pass")

    b6_rate = b6_values[0]
    headway = headways[0]
    convergence_headway = round(headway, 2)
    decomposition = {
        "format": "cross-ideal-gap-decomposition-v1",
        "date": "2026-09-30",
        "scope": "R2 full-network reference-envelope synthesis",
        "actual": actual,
        "reference_ladder": reference_ladder,
        "efficiency": efficiency,
        "incremental_loss_decomposition": losses,
        "dominant_bottleneck": {
            "classification": "GLOBAL_SLOT_COORDINATION_CONFLICT",
            "basis": (
                "B3->B4 is the only positive qualified implementable stochastic step; "
                "paired CI excludes zero in all three Delta arms"
            ),
            "caveat": (
                "classification is conditional on the frozen ordered ladder and currently modeled factors"
            ),
        },
        "recommended_next_counterfactual": {
            "for_B0_like_operation": "B4 rolling physical center-slot coordination",
            "expected_paired_effect_service_yield_per_hour_by_arm": paired_effects,
            "safety_feasibility_tradeoff": (
                "modeled infeasibility falls strongly, but minimum safety-domain margin "
                "compresses and near-miss burden increases"
            ),
            "after_B4": (
                "NO_ADDITIONAL_POSITIVE_B1_B5_RECOVERY_IDENTIFIED_AT_TESTED_6_VESSELS_PER_HOUR"
            ),
            "calibration": "MODEL_DERIVED_NOT_FIELD_CALIBRATED",
            "deployment_recommendation": False,
        },
        "ordering_theory": {
            "penetration_alone_sufficient": False,
            "B5_stochastic_result": (
                "NULL_PRIMARY_GAIN_AT_TESTED_LOAD_DUE_TO_NEAR_ZERO_STAGING_ACTIVATION"
            ),
            "B6v2_result": (
                "NEG050 exercises bounded ordering actions but all Delta arms share the same "
                "balanced saturated service yield"
            ),
            "interpretation": (
                "under balanced saturation the cross-stream physical/safety separation is "
                "binding before pair-order differences can change throughput"
            ),
        },
        "B6v1_disposition": (
            "RETAIN_AS_UNCONSTRAINED_STREAM_STARVATION_DIAGNOSTIC_NOT_BALANCED_FULL_NETWORK_ENVELOPE"
        ),
        "claim_scope": {
            "field_calibrated": False,
            "deployment_recommendation": False,
            "exact_Guo_section3_replication": False,
            "empirical_AS_HS_calibration": False,
            "ordered_ladder_is_order_independent_causal_decomposition": False,
            "B6v2_is_asymptotic_capacity_estimate": False,
        },
        "convergence_limit": {
            "B6v2_fixed_window_yield_per_hour": b6_rate,
            "observed_center_headway_min": convergence_headway,
            "headway_implied_long_run_rate_per_hour": 60.0 / convergence_headway,
            "note": (
                "B6v2 is a deterministic finite-window reference point; no horizon-convergence "
                "campaign was run."
            ),
        },
    }
    provenance = {
        "format": "cross-r2-reference-envelope-analysis-provenance-v1",
        "date": "2026-09-30",
        "input_archives": {
            "fs2": {"sha256": fs2_sha, "rho_zero_points": sum(len(fs2[arm]) for arm in _R2_ARMS)},
            "managed": {
                "sha256": managed_sha,
                "B4_points": sum(len(managed[arm]["B4"]) for arm in _R2_ARMS),
                "B5_points": sum(len(managed[arm]["B5"]) for arm in _R2_ARMS),
                "B6v1_points": len(b6v1),
            },
            "b6v2": {"sha256": b6v2_sha, "points": len(b6v2)},
        },
        "decomposition_canonical_sha256": sha256(
            (json.dumps(decomposition, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "scientific_model_change": False,
    }
    return decomposition, provenance
