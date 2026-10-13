"""Read-only comparison reports over verified A144 environmental-network archives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .environmental_network_result_archive import (
    EnvironmentalNetworkResultArchive,
    EnvironmentalNetworkResultArchiveKind,
    read_environmental_network_result_archive,
)

_COMPARISON_FORMAT = "vessel-crossing-nextgen-environmental-network-archive-comparison"
_COMPARISON_SCHEMA_VERSION = 1
_DEFAULT_MAX_DIFFERENCE_PATHS = 128


class EnvironmentalNetworkArchiveComparisonError(RuntimeError):
    """Raised when two verified archives cannot be meaningfully compared."""


class EnvironmentalNetworkArchiveComparisonRelation(str, Enum):
    IDENTICAL = "IDENTICAL"
    SAME_CONFIG_DIFFERENT_RESULT = "SAME_CONFIG_DIFFERENT_RESULT"
    DIFFERENT_CONFIG = "DIFFERENT_CONFIG"


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkArchiveEvidenceFingerprint:
    result_sha256: str
    config_sha256: str
    execution_environment_sha256: str
    traffic_evidence_sha256: str
    terminal_evidence_sha256: str
    metric_evidence_sha256: str


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkArchiveComparison:
    kind: EnvironmentalNetworkResultArchiveKind
    relation: EnvironmentalNetworkArchiveComparisonRelation
    left: EnvironmentalNetworkArchiveEvidenceFingerprint
    right: EnvironmentalNetworkArchiveEvidenceFingerprint
    same_config: bool
    same_execution_environment: bool
    same_traffic_evidence: bool
    same_terminal_evidence: bool
    same_metric_evidence: bool
    difference_count: int
    difference_paths: tuple[str, ...]
    difference_paths_truncated: bool

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _COMPARISON_FORMAT,
            "schema_version": _COMPARISON_SCHEMA_VERSION,
            "kind": self.kind.value,
            "relation": self.relation.value,
            "left": asdict(self.left),
            "right": asdict(self.right),
            "same_config": self.same_config,
            "same_execution_environment": self.same_execution_environment,
            "same_traffic_evidence": self.same_traffic_evidence,
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
            f"same_terminal_evidence: {str(self.same_terminal_evidence).lower()}",
            f"same_metric_evidence: {str(self.same_metric_evidence).lower()}",
            f"difference_count: {self.difference_count}",
            f"difference_paths_truncated: {str(self.difference_paths_truncated).lower()}",
        ]
        if self.difference_paths:
            lines.append("difference_paths:")
            lines.extend(f"- {path}" for path in self.difference_paths)
        return "\n".join(lines) + "\n"


def compare_environmental_network_result_archives(
    left: EnvironmentalNetworkResultArchive,
    right: EnvironmentalNetworkResultArchive,
    *,
    max_difference_paths: int = _DEFAULT_MAX_DIFFERENCE_PATHS,
) -> EnvironmentalNetworkArchiveComparison:
    """Compare verified evidence only; never rerun or reconstruct a simulator."""

    if not isinstance(left, EnvironmentalNetworkResultArchive) or not isinstance(
        right, EnvironmentalNetworkResultArchive
    ):
        raise TypeError("left and right must be EnvironmentalNetworkResultArchive values")
    if not isinstance(max_difference_paths, int) or isinstance(max_difference_paths, bool) or max_difference_paths < 1:
        raise ValueError("max_difference_paths must be a positive integer")
    if left.kind is not right.kind:
        raise EnvironmentalNetworkArchiveComparisonError(
            f"archive kinds differ: {left.kind.value} != {right.kind.value}"
        )
    if left.result_sha256 == right.result_sha256 and left.canonical_bytes != right.canonical_bytes:
        raise EnvironmentalNetworkArchiveComparisonError(
            "equal result SHA-256 values refer to different canonical bytes"
        )

    left_document = left.to_json_object()
    right_document = right.to_json_object()
    left_fingerprint = _fingerprint(left, left_document)
    right_fingerprint = _fingerprint(right, right_document)
    same_config = left_fingerprint.config_sha256 == right_fingerprint.config_sha256

    if left.canonical_bytes == right.canonical_bytes:
        relation = EnvironmentalNetworkArchiveComparisonRelation.IDENTICAL
    elif same_config:
        relation = EnvironmentalNetworkArchiveComparisonRelation.SAME_CONFIG_DIFFERENT_RESULT
    else:
        relation = EnvironmentalNetworkArchiveComparisonRelation.DIFFERENT_CONFIG

    paths: list[str] = []
    difference_count = _diff_json(
        left_document,
        right_document,
        path="",
        captured=paths,
        limit=max_difference_paths,
    )
    return EnvironmentalNetworkArchiveComparison(
        kind=left.kind,
        relation=relation,
        left=left_fingerprint,
        right=right_fingerprint,
        same_config=same_config,
        same_execution_environment=(
            left_fingerprint.execution_environment_sha256
            == right_fingerprint.execution_environment_sha256
        ),
        same_traffic_evidence=(
            left_fingerprint.traffic_evidence_sha256 == right_fingerprint.traffic_evidence_sha256
        ),
        same_terminal_evidence=(
            left_fingerprint.terminal_evidence_sha256
            == right_fingerprint.terminal_evidence_sha256
        ),
        same_metric_evidence=(
            left_fingerprint.metric_evidence_sha256 == right_fingerprint.metric_evidence_sha256
        ),
        difference_count=difference_count,
        difference_paths=tuple(paths),
        difference_paths_truncated=difference_count > len(paths),
    )


def compare_environmental_network_result_archive_files(
    left_path: str | Path,
    right_path: str | Path,
    *,
    max_difference_paths: int = _DEFAULT_MAX_DIFFERENCE_PATHS,
    require_checksum: bool = True,
) -> EnvironmentalNetworkArchiveComparison:
    """Read/verify two disk archives, then compare only their immutable evidence."""

    left = read_environmental_network_result_archive(left_path, require_checksum=require_checksum)
    right = read_environmental_network_result_archive(right_path, require_checksum=require_checksum)
    return compare_environmental_network_result_archives(
        left,
        right,
        max_difference_paths=max_difference_paths,
    )


def _fingerprint(
    archive: EnvironmentalNetworkResultArchive,
    document: Mapping[str, Any],
) -> EnvironmentalNetworkArchiveEvidenceFingerprint:
    config_sha = document.get("config_sha256")
    if not isinstance(config_sha, str):  # defensive; A144 verification guarantees this
        raise EnvironmentalNetworkArchiveComparisonError("verified archive lacks config_sha256")
    environment = document.get("execution_environment")
    if not isinstance(environment, Mapping):
        raise EnvironmentalNetworkArchiveComparisonError(
            "verified archive lacks execution_environment"
        )
    return EnvironmentalNetworkArchiveEvidenceFingerprint(
        result_sha256=archive.result_sha256,
        config_sha256=config_sha,
        execution_environment_sha256=_sha_json(environment),
        traffic_evidence_sha256=_sha_json(_traffic_evidence(archive.kind, document)),
        terminal_evidence_sha256=_sha_json(_terminal_evidence(archive.kind, document)),
        metric_evidence_sha256=_sha_json(_metric_evidence(archive.kind, document)),
    )


def _traffic_evidence(kind: EnvironmentalNetworkResultArchiveKind, document: Mapping[str, Any]) -> Any:
    if kind in (
        EnvironmentalNetworkResultArchiveKind.SINGLE_RUN,
        EnvironmentalNetworkResultArchiveKind.PAIRED,
    ):
        return document["traffic"]
    return [
        {
            "case_id": case["case_id"],
            "traffic": None if case["result"] is None else case["result"]["traffic"],
        }
        for case in document["cases"]
    ]


def _terminal_evidence(kind: EnvironmentalNetworkResultArchiveKind, document: Mapping[str, Any]) -> Any:
    if kind is EnvironmentalNetworkResultArchiveKind.SINGLE_RUN:
        return document["final_state"]
    if kind is EnvironmentalNetworkResultArchiveKind.PAIRED:
        return {
            "baseline": {
                "final_snapshot": document["baseline"]["final_snapshot"],
                "final_commands": document["baseline"]["final_commands"],
            },
            "control": document["control"]["final_state"],
        }
    result: list[dict[str, Any]] = []
    for case in document["cases"]:
        paired = case["result"]
        if paired is None:
            terminal = None
        else:
            terminal = {
                "baseline": {
                    "final_snapshot": paired["baseline"]["final_snapshot"],
                    "final_commands": paired["baseline"]["final_commands"],
                },
                "control": paired["control"]["final_state"],
            }
        result.append({"case_id": case["case_id"], "terminal": terminal})
    return result


def _metric_evidence(kind: EnvironmentalNetworkResultArchiveKind, document: Mapping[str, Any]) -> Any:
    if kind is EnvironmentalNetworkResultArchiveKind.SINGLE_RUN:
        return document["summary"]
    if kind is EnvironmentalNetworkResultArchiveKind.PAIRED:
        return {
            "baseline_metrics": document["baseline"]["metrics"],
            "control_metrics": document["control_metrics"],
            "difference": document["difference"],
        }
    return document["aggregate"]


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
    return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes, bytearray))


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
