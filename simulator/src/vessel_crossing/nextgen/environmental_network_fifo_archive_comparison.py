"""Read-only comparison reports over verified A154 FIFO result archives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .environmental_network_fifo_result_archive import (
    EnvironmentalNetworkFIFOResultArchive,
    EnvironmentalNetworkFIFOResultArchiveKind,
    read_environmental_network_fifo_result_archive,
)

_FIFO_COMPARISON_FORMAT = "vessel-crossing-nextgen-environmental-network-fifo-archive-comparison"
_FIFO_COMPARISON_SCHEMA_VERSION = 1
_DEFAULT_MAX_DIFFERENCE_PATHS = 128


class EnvironmentalNetworkFIFOArchiveComparisonError(RuntimeError):
    """Raised when two verified FIFO archives cannot be meaningfully compared."""


class EnvironmentalNetworkFIFOArchiveComparisonRelation(str, Enum):
    IDENTICAL = "IDENTICAL"
    SAME_CONFIG_DIFFERENT_RESULT = "SAME_CONFIG_DIFFERENT_RESULT"
    DIFFERENT_CONFIG = "DIFFERENT_CONFIG"


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOArchiveEvidenceFingerprint:
    result_sha256: str
    config_sha256: str
    execution_environment_sha256: str
    traffic_evidence_sha256: str
    source_evidence_sha256: str
    terminal_evidence_sha256: str
    metric_evidence_sha256: str


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOArchiveComparison:
    kind: EnvironmentalNetworkFIFOResultArchiveKind
    relation: EnvironmentalNetworkFIFOArchiveComparisonRelation
    left: EnvironmentalNetworkFIFOArchiveEvidenceFingerprint
    right: EnvironmentalNetworkFIFOArchiveEvidenceFingerprint
    same_config: bool
    same_execution_environment: bool
    same_traffic_evidence: bool
    same_source_evidence: bool
    same_terminal_evidence: bool
    same_metric_evidence: bool
    difference_count: int
    difference_paths: tuple[str, ...]
    difference_paths_truncated: bool

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _FIFO_COMPARISON_FORMAT,
            "schema_version": _FIFO_COMPARISON_SCHEMA_VERSION,
            "kind": self.kind.value,
            "relation": self.relation.value,
            "left": asdict(self.left),
            "right": asdict(self.right),
            "same_config": self.same_config,
            "same_execution_environment": self.same_execution_environment,
            "same_traffic_evidence": self.same_traffic_evidence,
            "same_source_evidence": self.same_source_evidence,
            "same_terminal_evidence": self.same_terminal_evidence,
            "same_metric_evidence": self.same_metric_evidence,
            "difference_count": self.difference_count,
            "difference_paths": list(self.difference_paths),
            "difference_paths_truncated": self.difference_paths_truncated,
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def report_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()

    def to_text(self) -> str:
        lines = [
            f"kind: {self.kind.value}",
            f"relation: {self.relation.value}",
            f"left_result_sha256: {self.left.result_sha256}",
            f"right_result_sha256: {self.right.result_sha256}",
            f"same_config: {str(self.same_config).lower()}",
            f"same_execution_environment: {str(self.same_execution_environment).lower()}",
            f"same_traffic_evidence: {str(self.same_traffic_evidence).lower()}",
            f"same_source_evidence: {str(self.same_source_evidence).lower()}",
            f"same_terminal_evidence: {str(self.same_terminal_evidence).lower()}",
            f"same_metric_evidence: {str(self.same_metric_evidence).lower()}",
            f"difference_count: {self.difference_count}",
            f"difference_paths_truncated: {str(self.difference_paths_truncated).lower()}",
        ]
        if self.difference_paths:
            lines.append("difference_paths:")
            lines.extend(f"- {path}" for path in self.difference_paths)
        return "\n".join(lines) + "\n"


def compare_environmental_network_fifo_result_archives(
    left: EnvironmentalNetworkFIFOResultArchive,
    right: EnvironmentalNetworkFIFOResultArchive,
    *,
    max_difference_paths: int = _DEFAULT_MAX_DIFFERENCE_PATHS,
) -> EnvironmentalNetworkFIFOArchiveComparison:
    """Compare verified A154 evidence only; never rerun or reconstruct a simulator."""

    if not isinstance(left, EnvironmentalNetworkFIFOResultArchive) or not isinstance(
        right, EnvironmentalNetworkFIFOResultArchive
    ):
        raise TypeError(
            "left and right must be EnvironmentalNetworkFIFOResultArchive values"
        )
    if (
        not isinstance(max_difference_paths, int)
        or isinstance(max_difference_paths, bool)
        or max_difference_paths < 1
    ):
        raise ValueError("max_difference_paths must be a positive integer")
    if left.kind is not right.kind:
        raise EnvironmentalNetworkFIFOArchiveComparisonError(
            f"archive kinds differ: {left.kind.value} != {right.kind.value}"
        )
    if left.result_sha256 == right.result_sha256 and left.canonical_bytes != right.canonical_bytes:
        raise EnvironmentalNetworkFIFOArchiveComparisonError(
            "equal result SHA-256 values refer to different canonical bytes"
        )

    left_document = left.to_json_object()
    right_document = right.to_json_object()
    left_fp = _fingerprint(left, left_document)
    right_fp = _fingerprint(right, right_document)
    same_config = left_fp.config_sha256 == right_fp.config_sha256
    if left.canonical_bytes == right.canonical_bytes:
        relation = EnvironmentalNetworkFIFOArchiveComparisonRelation.IDENTICAL
    elif same_config:
        relation = EnvironmentalNetworkFIFOArchiveComparisonRelation.SAME_CONFIG_DIFFERENT_RESULT
    else:
        relation = EnvironmentalNetworkFIFOArchiveComparisonRelation.DIFFERENT_CONFIG

    paths: list[str] = []
    difference_count = _diff_json(
        left_document,
        right_document,
        path="",
        captured=paths,
        limit=max_difference_paths,
    )
    return EnvironmentalNetworkFIFOArchiveComparison(
        kind=left.kind,
        relation=relation,
        left=left_fp,
        right=right_fp,
        same_config=same_config,
        same_execution_environment=(
            left_fp.execution_environment_sha256 == right_fp.execution_environment_sha256
        ),
        same_traffic_evidence=(
            left_fp.traffic_evidence_sha256 == right_fp.traffic_evidence_sha256
        ),
        same_source_evidence=(
            left_fp.source_evidence_sha256 == right_fp.source_evidence_sha256
        ),
        same_terminal_evidence=(
            left_fp.terminal_evidence_sha256 == right_fp.terminal_evidence_sha256
        ),
        same_metric_evidence=(
            left_fp.metric_evidence_sha256 == right_fp.metric_evidence_sha256
        ),
        difference_count=difference_count,
        difference_paths=tuple(paths),
        difference_paths_truncated=difference_count > len(paths),
    )


def compare_environmental_network_fifo_result_archive_files(
    left_path: str | Path,
    right_path: str | Path,
    *,
    max_difference_paths: int = _DEFAULT_MAX_DIFFERENCE_PATHS,
    require_checksum: bool = True,
) -> EnvironmentalNetworkFIFOArchiveComparison:
    left = read_environmental_network_fifo_result_archive(
        left_path, require_checksum=require_checksum
    )
    right = read_environmental_network_fifo_result_archive(
        right_path, require_checksum=require_checksum
    )
    return compare_environmental_network_fifo_result_archives(
        left, right, max_difference_paths=max_difference_paths
    )


def _fingerprint(
    archive: EnvironmentalNetworkFIFOResultArchive,
    document: Mapping[str, Any],
) -> EnvironmentalNetworkFIFOArchiveEvidenceFingerprint:
    config_sha = document.get("config_sha256")
    environment = document.get("execution_environment")
    if not isinstance(config_sha, str) or not isinstance(environment, Mapping):
        raise EnvironmentalNetworkFIFOArchiveComparisonError(
            "verified FIFO archive lacks config/environment evidence"
        )
    return EnvironmentalNetworkFIFOArchiveEvidenceFingerprint(
        result_sha256=archive.result_sha256,
        config_sha256=config_sha,
        execution_environment_sha256=_sha_json(environment),
        traffic_evidence_sha256=_sha_json(_traffic_evidence(archive.kind, document)),
        source_evidence_sha256=_sha_json(_source_evidence(archive.kind, document)),
        terminal_evidence_sha256=_sha_json(_terminal_evidence(archive.kind, document)),
        metric_evidence_sha256=_sha_json(_metric_evidence(archive.kind, document)),
    )


def _traffic_evidence(
    kind: EnvironmentalNetworkFIFOResultArchiveKind, document: Mapping[str, Any]
) -> Any:
    if kind is EnvironmentalNetworkFIFOResultArchiveKind.SINGLE_RUN:
        return document["traffic"]
    return document["shared_traffic"]


def _source_evidence(
    kind: EnvironmentalNetworkFIFOResultArchiveKind, document: Mapping[str, Any]
) -> Any:
    if kind is EnvironmentalNetworkFIFOResultArchiveKind.SINGLE_RUN:
        return document["source"]
    return {
        "baseline": document["baseline"]["source"],
        "control": document["control"]["source"],
    }


def _terminal_evidence(
    kind: EnvironmentalNetworkFIFOResultArchiveKind, document: Mapping[str, Any]
) -> Any:
    if kind is EnvironmentalNetworkFIFOResultArchiveKind.SINGLE_RUN:
        return document["final_state"]
    return {
        "baseline": document["baseline"]["final_state"],
        "control": document["control"]["final_state"],
    }


def _summary_outcomes(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "realized_demand_count",
        "physical_admission_count",
        "pending_source_count",
        "due_pending_source_count",
        "mean_admitted_waiting_time_min",
        "max_admitted_waiting_time_min",
        "final_source_generation",
        "source_plan_count",
        "source_boundary_count",
        "final_time",
        "final_vessel_count",
        "final_vessel_ids",
        "fleet_observation_count",
        "stop_reason",
    )
    return {key: summary.get(key) for key in keys}


def _metric_evidence(
    kind: EnvironmentalNetworkFIFOResultArchiveKind, document: Mapping[str, Any]
) -> Any:
    if kind is EnvironmentalNetworkFIFOResultArchiveKind.SINGLE_RUN:
        return {
            "summary_outcomes": _summary_outcomes(document["summary"]),
            "final_queue_metrics": document["source"]["final_queue_metrics"],
        }
    return {
        "baseline_outcomes": _summary_outcomes(document["baseline"]["summary"]),
        "baseline_queue": document["baseline"]["source"]["final_queue_metrics"],
        "control_outcomes": _summary_outcomes(document["control"]["summary"]),
        "control_queue": document["control"]["source"]["final_queue_metrics"],
        "difference": document["difference"],
    }


def _diff_json(
    left: Any,
    right: Any,
    *,
    path: str,
    captured: list[str],
    limit: int,
) -> int:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        count = 0
        for key in sorted(set(left) | set(right)):
            child = _pointer_child(path, str(key))
            if key not in left or key not in right:
                count += 1
                _capture(captured, child, limit)
            else:
                count += _diff_json(
                    left[key], right[key], path=child, captured=captured, limit=limit
                )
        return count
    if _is_sequence(left) and _is_sequence(right):
        count = 0
        common = min(len(left), len(right))
        for index in range(common):
            count += _diff_json(
                left[index],
                right[index],
                path=_pointer_child(path, str(index)),
                captured=captured,
                limit=limit,
            )
        for index in range(common, max(len(left), len(right))):
            count += 1
            _capture(captured, _pointer_child(path, str(index)), limit)
        return count
    if left != right:
        _capture(captured, path or "/", limit)
        return 1
    return 0


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _pointer_child(path: str, token: str) -> str:
    escaped = token.replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def _capture(captured: list[str], path: str, limit: int) -> None:
    if len(captured) < limit:
        captured.append(path)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha_json(value: Any) -> str:
    return sha256(_canonical_json(value)).hexdigest()
