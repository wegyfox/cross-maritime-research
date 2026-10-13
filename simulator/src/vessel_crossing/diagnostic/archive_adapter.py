"""Migration adapter from immutable network-FIFO result archives to D0 evidence.

The adapter is intentionally configuration driven.  Archive identities, hashes, member
patterns, scenario ids, replication count and the measurement window are inputs.  The
module derives measurements only; it contains no bottleneck or recommendation answer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
import statistics
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from zipfile import BadZipFile, ZipFile

from ..nextgen.reference_envelope import ReferenceEnvelopeLevel
from .evidence import (
    DiagnosticEvidenceError,
    DiagnosticLevelEvidence,
    DiagnosticReplicationEvidence,
    DiagnosticStudyEvidence,
)
from .interventions import DEFAULT_INTERVENTION_REGISTRY


@dataclass(frozen=True, slots=True)
class LegacyNetworkFifoArchiveSpec:
    baseline_archive: str | Path
    managed_archive: str | Path
    reference_archive: str | Path
    scenario_ids: tuple[str, ...]
    baseline_member_pattern: str
    managed_member_pattern: str
    reference_member_pattern: str
    b6v1_member_pattern: str | None = None
    expected_replications: int = 1
    measurement_window_end_min: float = 90.0
    expected_baseline_sha256: str | None = None
    expected_managed_sha256: str | None = None
    expected_reference_sha256: str | None = None
    identity_reasons: Mapping[str, str] = field(default_factory=dict)
    claim_scope: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_ids or len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("scenario_ids must be unique and non-empty")
        if not isinstance(self.expected_replications, int) or self.expected_replications <= 0:
            raise ValueError("expected_replications must be positive")
        if self.measurement_window_end_min <= 0:
            raise ValueError("measurement_window_end_min must be positive")
        for pattern in (
            self.baseline_member_pattern,
            self.managed_member_pattern,
            self.reference_member_pattern,
        ):
            re.compile(pattern)
        if self.b6v1_member_pattern is not None:
            re.compile(self.b6v1_member_pattern)
        object.__setattr__(
            self, "identity_reasons", MappingProxyType(dict(self.identity_reasons))
        )
        object.__setattr__(self, "claim_scope", MappingProxyType(dict(self.claim_scope)))


@dataclass(frozen=True, slots=True)
class LegacyNetworkFifoEvidenceBundle:
    studies: tuple[DiagnosticStudyEvidence, ...]
    archive_sha256: Mapping[str, str]
    b6v1_stream_counts: Mapping[str, Mapping[str, int]]

    def study(self, scenario_id: str) -> DiagnosticStudyEvidence:
        for item in self.studies:
            if item.scenario_id == scenario_id:
                return item
        raise KeyError(scenario_id)


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
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise DiagnosticEvidenceError(
            f"archive SHA-256 mismatch for {path.name}: {actual} != {expected_sha256}"
        )
    try:
        with ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise DiagnosticEvidenceError(f"ZIP CRC failure in {path.name}: {bad}")
            docs: dict[str, Any] = {}
            for name in archive.namelist():
                if name.endswith(".json"):
                    docs[name] = json.loads(archive.read(name))
    except (OSError, BadZipFile, json.JSONDecodeError) as exc:
        raise DiagnosticEvidenceError(f"cannot read result archive {path}: {exc}") from exc
    return docs, actual


def _percentile(values: Iterable[float], q: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise DiagnosticEvidenceError("cannot compute percentile of empty evidence")
    position = (len(rows) - 1) * q
    low = int(position)
    high = min(low + 1, len(rows) - 1)
    fraction = position - low
    return rows[low] * (1.0 - fraction) + rows[high] * fraction


def _fixed_window_center_yield(document: Mapping[str, Any], end_min: float) -> float:
    center = document.get("center_service")
    if not isinstance(center, Mapping) or not isinstance(center.get("records"), list):
        raise DiagnosticEvidenceError("center_service.records is missing")
    count = sum(
        1
        for row in center["records"]
        if isinstance(row, Mapping)
        and isinstance(row.get("passage_time_min"), (int, float))
        and not isinstance(row.get("passage_time_min"), bool)
        and float(row["passage_time_min"]) <= end_min + 1e-12
    )
    return 60.0 * count / end_min


def _nominal_route_duration(document: Mapping[str, Any]) -> float:
    conflict = document.get("conflict")
    if not isinstance(conflict, Mapping):
        raise DiagnosticEvidenceError("conflict evidence is missing")
    value = conflict.get("nominal_route_duration_min")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticEvidenceError("nominal route duration is missing")
    return float(value)


def _route_extra_rows(document: Mapping[str, Any]) -> list[float]:
    route = document.get("route")
    if not isinstance(route, Mapping) or not isinstance(route.get("records"), list):
        raise DiagnosticEvidenceError("route records are missing")
    nominal = _nominal_route_duration(document)
    return [float(row["route_duration_min"]) - nominal for row in route["records"]]


def _wait_rows(document: Mapping[str, Any]) -> list[float]:
    source = document.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("admissions"), list):
        raise DiagnosticEvidenceError("source admissions are missing")
    return [float(row["waiting_time_min"]) for row in source["admissions"]]


def _primary(document: Mapping[str, Any], *, managed: bool, end_min: float) -> float:
    if managed:
        return float(document["primary_kpi"]["fixed_window_service_yield_per_hour"])
    return _fixed_window_center_yield(document, end_min)


def _replication_evidence(
    replication: int,
    document: Mapping[str, Any],
    *,
    managed: bool,
    end_min: float,
) -> DiagnosticReplicationEvidence:
    safety = document["safety"]
    terminal = document["terminal"]
    metrics = {
        "terminal_status": terminal["status"],
        "near_miss_episode_count": int(safety["near_miss_episode_count"]),
        "safety_domain_intrusion_episode_count": int(
            safety["safety_domain_intrusion_episode_count"]
        ),
        "modeled_hull_overlap_episode_count": int(
            safety["modeled_hull_overlap_episode_count"]
        ),
        "sampled_min_safety_domain_margin_nm": float(
            safety["sampled_min_safety_domain_margin_nm"]
        ),
    }
    activation_counts: Counter[str] = Counter()
    if managed:
        controller = document.get("controller", {})
        trace = controller.get("trace", []) if isinstance(controller, Mapping) else []
        activation_counts.update(
            row["event"]
            for row in trace
            if isinstance(row, Mapping) and isinstance(row.get("event"), str)
        )
    return DiagnosticReplicationEvidence(
        replication_id=f"R{replication:03d}",
        primary_kpi=_primary(document, managed=managed, end_min=end_min),
        metrics=metrics,
        activation_counts=dict(activation_counts),
    )


def _aggregate_runs(
    by_replication: Mapping[int, Mapping[str, Any]],
    *,
    managed: bool,
    expected_replications: int,
    end_min: float,
) -> dict[str, Any]:
    documents = [by_replication[index] for index in sorted(by_replication)]
    if len(documents) != expected_replications:
        raise DiagnosticEvidenceError(
            f"expected {expected_replications} replications, got {len(documents)}"
        )
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
        waits.extend(_wait_rows(document))
        route_extra.extend(_route_extra_rows(document))
        if managed:
            controller = document["controller"]
            events.update(row["event"] for row in controller["trace"])
            bypass_total += sum(int(value) for value in controller["bypass_count"].values())
            trace_count = int(controller["trace_count"])
        else:
            trace_count = 0
        rows.append(
            {
                "kpi": _primary(document, managed=managed, end_min=end_min),
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


def _reference_metrics(document: Mapping[str, Any]) -> dict[str, Any]:
    headways = [float(value) for value in document["conflict"]["center_passage_headway_min"]]
    if not headways:
        raise DiagnosticEvidenceError("reference center headway evidence is empty")
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


def _definition(level: ReferenceEnvelopeLevel):
    for item in DEFAULT_INTERVENTION_REGISTRY.definitions:
        if item.level is level:
            return item
    raise AssertionError(level)


def build_legacy_network_fifo_evidence(
    spec: LegacyNetworkFifoArchiveSpec,
) -> LegacyNetworkFifoEvidenceBundle:
    baseline_docs, baseline_sha = _load_archive(
        spec.baseline_archive, expected_sha256=spec.expected_baseline_sha256
    )
    managed_docs, managed_sha = _load_archive(
        spec.managed_archive, expected_sha256=spec.expected_managed_sha256
    )
    reference_docs, reference_sha = _load_archive(
        spec.reference_archive, expected_sha256=spec.expected_reference_sha256
    )
    base_re = re.compile(spec.baseline_member_pattern)
    managed_re = re.compile(spec.managed_member_pattern)
    reference_re = re.compile(spec.reference_member_pattern)
    b6v1_re = re.compile(spec.b6v1_member_pattern) if spec.b6v1_member_pattern else None

    baseline: dict[str, dict[int, Mapping[str, Any]]] = {key: {} for key in spec.scenario_ids}
    managed: dict[str, dict[str, dict[int, Mapping[str, Any]]]] = {
        key: {"B4": {}, "B5": {}} for key in spec.scenario_ids
    }
    references: dict[str, Mapping[str, Any]] = {}
    b6v1: dict[str, Mapping[str, Any]] = {}

    for name, document in baseline_docs.items():
        match = base_re.match(name)
        if match:
            scenario = match.group("scenario")
            if scenario in baseline:
                baseline[scenario][int(match.group("rep"))] = document
    for name, document in managed_docs.items():
        match = managed_re.match(name)
        if match:
            scenario = match.group("scenario")
            level = match.group("level")
            if scenario in managed and level in {"B4", "B5"}:
                managed[scenario][level][int(match.group("rep"))] = document
            continue
        if b6v1_re is not None:
            match = b6v1_re.match(name)
            if match and match.group("scenario") in spec.scenario_ids:
                b6v1[match.group("scenario")] = document
    for name, document in reference_docs.items():
        match = reference_re.match(name)
        if match and match.group("scenario") in spec.scenario_ids:
            references[match.group("scenario")] = document

    b6v1_counts: dict[str, Mapping[str, int]] = {}
    studies: list[DiagnosticStudyEvidence] = []
    for scenario in spec.scenario_ids:
        for group_name, rows in (
            ("baseline", baseline[scenario]),
            ("B4", managed[scenario]["B4"]),
            ("B5", managed[scenario]["B5"]),
        ):
            if len(rows) != spec.expected_replications:
                raise DiagnosticEvidenceError(
                    f"{scenario} {group_name}: expected {spec.expected_replications} results, got {len(rows)}"
                )
        if scenario not in references:
            raise DiagnosticEvidenceError(f"missing reference result for {scenario}")
        if b6v1_re is not None:
            if scenario not in b6v1:
                raise DiagnosticEvidenceError(f"missing B6v1 result for {scenario}")
            b6v1_counts[scenario] = dict(
                Counter(row["stream_id"] for row in b6v1[scenario]["source"]["admissions"])
            )

        b0_metrics = _aggregate_runs(
            baseline[scenario],
            managed=False,
            expected_replications=spec.expected_replications,
            end_min=spec.measurement_window_end_min,
        )
        b4_metrics = _aggregate_runs(
            managed[scenario]["B4"],
            managed=True,
            expected_replications=spec.expected_replications,
            end_min=spec.measurement_window_end_min,
        )
        b5_metrics = _aggregate_runs(
            managed[scenario]["B5"],
            managed=True,
            expected_replications=spec.expected_replications,
            end_min=spec.measurement_window_end_min,
        )
        b6_metrics = _reference_metrics(references[scenario])

        b0_reps = tuple(
            _replication_evidence(
                rep,
                baseline[scenario][rep],
                managed=False,
                end_min=spec.measurement_window_end_min,
            )
            for rep in sorted(baseline[scenario])
        )
        b4_reps = tuple(
            _replication_evidence(
                rep,
                managed[scenario]["B4"][rep],
                managed=True,
                end_min=spec.measurement_window_end_min,
            )
            for rep in sorted(managed[scenario]["B4"])
        )
        b5_reps = tuple(
            _replication_evidence(
                rep,
                managed[scenario]["B5"][rep],
                managed=True,
                end_min=spec.measurement_window_end_min,
            )
            for rep in sorted(managed[scenario]["B5"])
        )

        levels: list[DiagnosticLevelEvidence] = [
            DiagnosticLevelEvidence(
                ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK,
                b0_metrics,
                b0_reps,
            )
        ]
        previous = ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK
        for level in (
            ReferenceEnvelopeLevel.B1_DEDICATED_AUTONOMOUS_CORRIDOR,
            ReferenceEnvelopeLevel.B2_CONTROLLED_ENTRY,
            ReferenceEnvelopeLevel.B3_SPEED_TRAJECTORY_HARMONIZATION,
        ):
            definition = _definition(level)
            reason = spec.identity_reasons.get(level.value, "IDENTITY_DERIVED")
            levels.append(
                DiagnosticLevelEvidence(
                    level,
                    {"service_yield_per_hour": b0_metrics["mean_service_yield_per_hour"]},
                    b0_reps,
                    intervention_id=definition.intervention_id,
                    target_factor=definition.target_factor,
                    causal=False,
                    identity_of=previous,
                    reason=reason,
                )
            )
            previous = level
        for level, metrics, reps in (
            (ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION, b4_metrics, b4_reps),
            (ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE, b5_metrics, b5_reps),
        ):
            definition = _definition(level)
            levels.append(
                DiagnosticLevelEvidence(
                    level,
                    metrics,
                    reps,
                    intervention_id=definition.intervention_id,
                    target_factor=definition.target_factor,
                    causal=True,
                )
            )
        definition = _definition(ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE)
        levels.append(
            DiagnosticLevelEvidence(
                ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE,
                b6_metrics,
                intervention_id=definition.intervention_id,
                target_factor=definition.target_factor,
                causal=False,
                reason="DETERMINISTIC_REFERENCE",
            )
        )
        studies.append(
            DiagnosticStudyEvidence(
                scenario,
                tuple(levels),
                claim_scope=spec.claim_scope,
                provenance={
                    "baseline_archive_sha256": baseline_sha,
                    "managed_archive_sha256": managed_sha,
                    "reference_archive_sha256": reference_sha,
                    "measurement_window_end_min": spec.measurement_window_end_min,
                },
            )
        )

    return LegacyNetworkFifoEvidenceBundle(
        tuple(studies),
        MappingProxyType(
            {
                "baseline": baseline_sha,
                "managed": managed_sha,
                "reference": reference_sha,
            }
        ),
        MappingProxyType(
            {key: MappingProxyType(dict(value)) for key, value in b6v1_counts.items()}
        ),
    )
