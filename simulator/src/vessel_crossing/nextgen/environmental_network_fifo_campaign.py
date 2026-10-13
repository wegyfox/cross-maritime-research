"""Fixed campaign/protocol packaging over A160 repeated-pair FIFO statistics.

A161 is deliberately orchestration-only.  A campaign is an explicit finite set of named
A160 configs.  Each point already owns its deterministic replication schedule, A159
estimand and 95% paired inference.  A161 therefore does not define metrics, derive extra
seeds, adapt replication counts, stop early, or select an implicit latest configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from ..archive import environment_manifest
from ..version import __version__
from .environmental_network_experiment import _canonical_json, _json_value
from .environmental_network_fifo_repeated_pair_statistics import (
    EnvironmentalNetworkFIFORepeatedPairStatisticsConfig,
    EnvironmentalNetworkFIFORepeatedPairStatisticsResult,
    run_environmental_network_fifo_repeated_pair_statistics,
)

_FIFO_CAMPAIGN_PROTOCOL_FORMAT = "vessel-crossing-nextgen-environmental-network-fifo-campaign-protocol"
_FIFO_CAMPAIGN_PROTOCOL_SCHEMA_VERSION = 1
_FIFO_CAMPAIGN_RESULT_FORMAT = "vessel-crossing-nextgen-environmental-network-fifo-campaign-result"
_FIFO_CAMPAIGN_RESULT_SCHEMA_VERSION = 1
_FIFO_CAMPAIGN_DOMAIN = b"vessel-crossing-nextgen-environmental-network-fifo-campaign\x00"


class EnvironmentalNetworkFIFOCampaignError(RuntimeError):
    """Raised when a fixed A161 FIFO campaign is invalid."""


class EnvironmentalNetworkFIFOCampaignIntegrityError(EnvironmentalNetworkFIFOCampaignError):
    """Raised when persisted A161 campaign evidence fails canonical/checksum checks."""


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOCampaignPoint:
    point_id: str
    statistics: EnvironmentalNetworkFIFORepeatedPairStatisticsConfig

    def __post_init__(self) -> None:
        point_id = self.point_id.strip()
        if not point_id:
            raise ValueError("point_id must be non-empty")
        if not isinstance(self.statistics, EnvironmentalNetworkFIFORepeatedPairStatisticsConfig):
            raise TypeError("statistics must be EnvironmentalNetworkFIFORepeatedPairStatisticsConfig")
        object.__setattr__(self, "point_id", point_id)

    def to_json_object(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "statistics_config_sha256": self.statistics.config_sha256,
            "statistics_experiment_id": self.statistics.experiment_id,
            "statistics": self.statistics.to_json_object(),
        }


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOCampaignProtocol:
    points: tuple[EnvironmentalNetworkFIFOCampaignPoint, ...]
    label: str | None = None

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if not points:
            raise ValueError("FIFO campaign requires at least one explicit point")
        if any(not isinstance(item, EnvironmentalNetworkFIFOCampaignPoint) for item in points):
            raise TypeError("points must contain EnvironmentalNetworkFIFOCampaignPoint values")
        ids = [item.point_id for item in points]
        if len(set(ids)) != len(ids):
            raise ValueError("FIFO campaign point_id values must be unique")
        if self.label is not None:
            label = self.label.strip()
            if not label:
                raise ValueError("label must be non-empty when supplied")
            object.__setattr__(self, "label", label)
        object.__setattr__(self, "points", tuple(sorted(points, key=lambda item: item.point_id)))

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _FIFO_CAMPAIGN_PROTOCOL_FORMAT,
            "schema_version": _FIFO_CAMPAIGN_PROTOCOL_SCHEMA_VERSION,
            "software_version": __version__,
            "label": self.label,
            "points": [item.to_json_object() for item in self.points],
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def protocol_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()

    @property
    def campaign_id(self) -> str:
        return sha256(_FIFO_CAMPAIGN_DOMAIN + self.to_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOCampaignPointResult:
    point_id: str
    statistics_config_sha256: str
    statistics_experiment_id: str
    statistics_result_sha256: str
    statistics: EnvironmentalNetworkFIFORepeatedPairStatisticsResult


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOCampaignResult:
    protocol: EnvironmentalNetworkFIFOCampaignProtocol
    points: tuple[EnvironmentalNetworkFIFOCampaignPointResult, ...]
    canonical_bytes: bytes = field(repr=False)

    @property
    def campaign_id(self) -> str:
        return self.protocol.campaign_id

    @property
    def protocol_sha256(self) -> str:
        return self.protocol.protocol_sha256

    def to_json_object(self) -> dict[str, Any]:
        obj = json.loads(self.canonical_bytes.decode("utf-8"))
        if not isinstance(obj, dict):  # pragma: no cover
            raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign result is not a JSON object")
        return obj

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes

    @property
    def result_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()


def run_environmental_network_fifo_campaign(
    protocol: EnvironmentalNetworkFIFOCampaignProtocol,
) -> EnvironmentalNetworkFIFOCampaignResult:
    if not isinstance(protocol, EnvironmentalNetworkFIFOCampaignProtocol):
        raise TypeError("protocol must be EnvironmentalNetworkFIFOCampaignProtocol")

    point_results: list[EnvironmentalNetworkFIFOCampaignPointResult] = []
    point_documents: list[dict[str, Any]] = []
    for point in protocol.points:
        result = run_environmental_network_fifo_repeated_pair_statistics(point.statistics)
        record = EnvironmentalNetworkFIFOCampaignPointResult(
            point_id=point.point_id,
            statistics_config_sha256=result.config_sha256,
            statistics_experiment_id=result.experiment_id,
            statistics_result_sha256=result.result_sha256,
            statistics=result,
        )
        point_results.append(record)
        point_documents.append({
            "point_id": record.point_id,
            "statistics_config_sha256": record.statistics_config_sha256,
            "statistics_experiment_id": record.statistics_experiment_id,
            "statistics_result_sha256": record.statistics_result_sha256,
            "statistics_result": result.to_json_object(),
        })

    document = {
        "format": _FIFO_CAMPAIGN_RESULT_FORMAT,
        "schema_version": _FIFO_CAMPAIGN_RESULT_SCHEMA_VERSION,
        "campaign_id": protocol.campaign_id,
        "protocol_sha256": protocol.protocol_sha256,
        "protocol": protocol.to_json_object(),
        "execution_environment": environment_manifest(),
        "points": point_documents,
    }
    canonical_bytes = _canonical_json(document)
    return EnvironmentalNetworkFIFOCampaignResult(
        protocol=protocol,
        points=tuple(point_results),
        canonical_bytes=canonical_bytes,
    )


def write_environmental_network_fifo_campaign_result(
    result: EnvironmentalNetworkFIFOCampaignResult,
    path: str | Path,
) -> tuple[Path, Path]:
    """Write canonical result bytes and an exact SHA-256 sidecar; never reconstruct runtime."""
    if not isinstance(result, EnvironmentalNetworkFIFOCampaignResult):
        raise TypeError("result must be EnvironmentalNetworkFIFOCampaignResult")
    target = Path(path)
    target.write_bytes(result.to_json_bytes())
    checksum = target.with_name(target.name + ".sha256")
    checksum.write_text(f"{result.result_sha256}  {target.name}\n", encoding="utf-8")
    return target, checksum


def verify_environmental_network_fifo_campaign_result(
    path: str | Path,
    checksum_path: str | Path | None = None,
) -> dict[str, Any]:
    """Non-executable checksum/canonical/internal-hash verification for an A161 file."""
    target = Path(path)
    raw = target.read_bytes()
    digest = sha256(raw).hexdigest()
    checksum = Path(checksum_path) if checksum_path is not None else target.with_name(target.name + ".sha256")
    line = checksum.read_text(encoding="utf-8").strip()
    expected = f"{digest}  {target.name}"
    if line != expected:
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign checksum sidecar mismatch")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign result is not valid JSON") from exc
    if not isinstance(obj, dict) or obj.get("format") != _FIFO_CAMPAIGN_RESULT_FORMAT:
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("unsupported FIFO campaign result format")
    if obj.get("schema_version") != _FIFO_CAMPAIGN_RESULT_SCHEMA_VERSION:
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("unsupported FIFO campaign result schema")
    if _canonical_json(obj) != raw:
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign result is not canonical JSON")
    protocol = obj.get("protocol")
    if not isinstance(protocol, dict):
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign protocol evidence is missing")
    if protocol.get("format") != _FIFO_CAMPAIGN_PROTOCOL_FORMAT:
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("unsupported FIFO campaign protocol format")
    if protocol.get("schema_version") != _FIFO_CAMPAIGN_PROTOCOL_SCHEMA_VERSION:
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("unsupported FIFO campaign protocol schema")
    protocol_bytes = _canonical_json(protocol)
    if sha256(protocol_bytes).hexdigest() != obj.get("protocol_sha256"):
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign protocol hash mismatch")
    expected_campaign_id = sha256(_FIFO_CAMPAIGN_DOMAIN + protocol_bytes).hexdigest()
    if obj.get("campaign_id") != expected_campaign_id:
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign identity mismatch")

    protocol_points = protocol.get("points")
    result_points = obj.get("points")
    if not isinstance(protocol_points, list) or not protocol_points:
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign protocol points are malformed")
    if not isinstance(result_points, list) or len(result_points) != len(protocol_points):
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign result points do not match protocol")

    protocol_by_id: dict[str, dict[str, Any]] = {}
    for point in protocol_points:
        if not isinstance(point, dict) or not isinstance(point.get("point_id"), str):
            raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign protocol point evidence is malformed")
        point_id = point["point_id"]
        if not point_id or point_id in protocol_by_id:
            raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign protocol point identities are invalid")
        protocol_by_id[point_id] = point

    seen_result_ids: set[str] = set()
    for point in result_points:
        if not isinstance(point, dict) or not isinstance(point.get("statistics_result"), dict):
            raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign point evidence is malformed")
        point_id = point.get("point_id")
        if not isinstance(point_id, str) or point_id in seen_result_ids or point_id not in protocol_by_id:
            raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign result point identity mismatch")
        seen_result_ids.add(point_id)
        protocol_point = protocol_by_id[point_id]
        if point.get("statistics_config_sha256") != protocol_point.get("statistics_config_sha256"):
            raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign point protocol config mismatch")
        if point.get("statistics_experiment_id") != protocol_point.get("statistics_experiment_id"):
            raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign point protocol experiment mismatch")

        stats = point["statistics_result"]
        stats_bytes = _canonical_json(stats)
        if sha256(stats_bytes).hexdigest() != point.get("statistics_result_sha256"):
            raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign point result hash mismatch")
        if stats.get("config_sha256") != point.get("statistics_config_sha256"):
            raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign point config hash mismatch")
        if stats.get("experiment_id") != point.get("statistics_experiment_id"):
            raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign point experiment identity mismatch")

    if seen_result_ids != set(protocol_by_id):
        raise EnvironmentalNetworkFIFOCampaignIntegrityError("campaign result point set mismatch")
    return obj
