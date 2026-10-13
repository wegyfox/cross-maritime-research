"""C13 benchmark-corpus lineage and coverage accounting.

This layer is evidence-only.  It records where benchmark scenarios came from and
separately audits what an explicitly declared target population is or is not covered
by.  In particular, an AIS-derived scenario can have excellent provenance while still
leaving non-AIS, small-craft, fishing or rare-event strata uncovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Any, Iterable, Mapping

from .scope import TestbedFeature


class CorpusSourceKind(str, Enum):
    AIS_TRACKS = "AIS_TRACKS"
    SENSOR_RECORDING = "SENSOR_RECORDING"
    VTS_RECORD = "VTS_RECORD"
    FIELD_OBSERVATION = "FIELD_OBSERVATION"
    INCIDENT_ACCIDENT_REPORT = "INCIDENT_ACCIDENT_REPORT"
    EXTERNAL_BENCHMARK = "EXTERNAL_BENCHMARK"
    SYNTHETIC_GENERATOR = "SYNTHETIC_GENERATOR"
    DOCUMENTED_ASSUMPTION = "DOCUMENTED_ASSUMPTION"
    OTHER = "OTHER"


class ScenarioDerivationKind(str, Enum):
    RAW_REPLAY = "RAW_REPLAY"
    FILTERED_REPLAY = "FILTERED_REPLAY"
    RECONSTRUCTION = "RECONSTRUCTION"
    SYNTHETIC_AUGMENTATION = "SYNTHETIC_AUGMENTATION"
    CALIBRATED_SYNTHETIC = "CALIBRATED_SYNTHETIC"
    FULLY_SYNTHETIC = "FULLY_SYNTHETIC"


class CorpusSupportClass(str, Enum):
    DIRECT_EMPIRICAL = "DIRECT_EMPIRICAL"
    EMPIRICALLY_DERIVED = "EMPIRICALLY_DERIVED"
    DOCUMENTARY_RECONSTRUCTION = "DOCUMENTARY_RECONSTRUCTION"
    EXTERNAL_BENCHMARK = "EXTERNAL_BENCHMARK"
    SYNTHETIC = "SYNTHETIC"
    ASSUMPTION_BASED = "ASSUMPTION_BASED"
    MIXED_OTHER = "MIXED_OTHER"


class CoverageDimension(str, Enum):
    VESSEL_CLASS = "VESSEL_CLASS"
    SIZE_CLASS = "SIZE_CLASS"
    AIS_AVAILABILITY = "AIS_AVAILABILITY"
    OPERATIONAL_ACTIVITY = "OPERATIONAL_ACTIVITY"
    BEHAVIOR_CLASS = "BEHAVIOR_CLASS"
    ENCOUNTER_TYPE = "ENCOUNTER_TYPE"
    GEOGRAPHIC_REGION = "GEOGRAPHIC_REGION"
    WATERWAY_TYPE = "WATERWAY_TYPE"
    TRAFFIC_DENSITY = "TRAFFIC_DENSITY"
    ENVIRONMENT = "ENVIRONMENT"
    VISIBILITY = "VISIBILITY"
    SEA_ROOM = "SEA_ROOM"
    INCIDENT_SEVERITY = "INCIDENT_SEVERITY"
    CUSTOM = "CUSTOM"


class CoverageStatus(str, Enum):
    MISSING = "MISSING"
    EMPIRICAL_GAP = "EMPIRICAL_GAP"
    INSUFFICIENT = "INSUFFICIENT"
    UNDERREPRESENTED = "UNDERREPRESENTED"
    PRESENT_UNWEIGHTED = "PRESENT_UNWEIGHTED"
    COVERED = "COVERED"
    OVERREPRESENTED = "OVERREPRESENTED"


_EMPIRICAL_SOURCE_KINDS = {
    CorpusSourceKind.AIS_TRACKS,
    CorpusSourceKind.SENSOR_RECORDING,
    CorpusSourceKind.VTS_RECORD,
    CorpusSourceKind.FIELD_OBSERVATION,
}


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")


def _canonical_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("canonical corpus data must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical corpus mapping keys must be strings")
            result[key] = _canonical_json_value(item)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError(f"unsupported canonical corpus value type: {type(value).__name__}")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_canonical_json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(_canonical_json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class CoverageTag:
    dimension: CoverageDimension
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("coverage tag value must be non-empty")


@dataclass(frozen=True, slots=True)
class CorpusSourceArtifact:
    source_id: str
    source_kind: CorpusSourceKind
    source_version: str
    source_uri: str
    content_sha256: str
    license_id: str
    parent_source_ids: tuple[str, ...] = ()
    known_exclusions: tuple[str, ...] = ()
    attributes_json: str = "{}"

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        source_kind: CorpusSourceKind,
        source_version: str,
        source_uri: str,
        content_sha256: str,
        license_id: str,
        parent_source_ids: Iterable[str] = (),
        known_exclusions: Iterable[str] = (),
        attributes: Mapping[str, Any] | None = None,
    ) -> "CorpusSourceArtifact":
        if not source_id or not source_version or not source_uri or not license_id:
            raise ValueError("source identity/version/uri/license fields must be non-empty")
        _validate_sha256(content_sha256, "content_sha256")
        parents = tuple(sorted(set(parent_source_ids)))
        exclusions = tuple(sorted(set(known_exclusions)))
        return cls(
            source_id=source_id,
            source_kind=source_kind,
            source_version=source_version,
            source_uri=source_uri,
            content_sha256=content_sha256.lower(),
            license_id=license_id,
            parent_source_ids=parents,
            known_exclusions=exclusions,
            attributes_json=_canonical_json(dict(attributes or {})),
        )

    @property
    def attributes(self) -> Mapping[str, Any]:
        return json.loads(self.attributes_json)

    @property
    def is_empirical(self) -> bool:
        return self.source_kind in _EMPIRICAL_SOURCE_KINDS

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "source_version": self.source_version,
            "source_uri": self.source_uri,
            "content_sha256": self.content_sha256,
            "license_id": self.license_id,
            "parent_source_ids": list(self.parent_source_ids),
            "known_exclusions": list(self.known_exclusions),
            "attributes": json.loads(self.attributes_json),
        })


@dataclass(frozen=True, slots=True)
class CorpusScenarioRecord:
    scenario_id: str
    derivation_kind: ScenarioDerivationKind
    source_artifact_ids: tuple[str, ...]
    adapter_id: str
    adapter_version: str
    derivation_parameters_json: str
    coverage_tags: tuple[CoverageTag, ...]
    protocol_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        scenario_id: str,
        derivation_kind: ScenarioDerivationKind,
        source_artifact_ids: Iterable[str],
        adapter_id: str,
        adapter_version: str,
        coverage_tags: Iterable[CoverageTag],
        derivation_parameters: Mapping[str, Any] | None = None,
        protocol_ids: Iterable[str] = (),
        notes: Iterable[str] = (),
    ) -> "CorpusScenarioRecord":
        if not scenario_id or not adapter_id or not adapter_version:
            raise ValueError("scenario_id, adapter_id and adapter_version must be non-empty")
        sources = tuple(sorted(set(source_artifact_ids)))
        if not sources:
            raise ValueError("a corpus scenario must reference at least one source artifact")
        tags = tuple(sorted(set(coverage_tags), key=lambda tag: (tag.dimension.value, tag.value)))
        if not tags:
            raise ValueError("coverage_tags must be explicit and non-empty")
        return cls(
            scenario_id=scenario_id,
            derivation_kind=derivation_kind,
            source_artifact_ids=sources,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            derivation_parameters_json=_canonical_json(dict(derivation_parameters or {})),
            coverage_tags=tags,
            protocol_ids=tuple(sorted(set(protocol_ids))),
            notes=tuple(sorted(set(notes))),
        )

    @property
    def derivation_parameters(self) -> Mapping[str, Any]:
        return json.loads(self.derivation_parameters_json)

    def has_all_tags(self, required: Iterable[CoverageTag]) -> bool:
        return set(required).issubset(self.coverage_tags)

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "scenario_id": self.scenario_id,
            "derivation_kind": self.derivation_kind.value,
            "source_artifact_ids": list(self.source_artifact_ids),
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "derivation_parameters": json.loads(self.derivation_parameters_json),
            "coverage_tags": [
                {"dimension": tag.dimension.value, "value": tag.value}
                for tag in self.coverage_tags
            ],
            "protocol_ids": list(self.protocol_ids),
            "notes": list(self.notes),
        })


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    corpus_id: str
    corpus_version: str
    sources: tuple[CorpusSourceArtifact, ...]
    scenarios: tuple[CorpusScenarioRecord, ...]
    component_id: TestbedFeature = TestbedFeature.C13_CORPUS_LINEAGE

    @classmethod
    def from_unsorted(
        cls,
        *,
        corpus_id: str,
        corpus_version: str,
        sources: Iterable[CorpusSourceArtifact],
        scenarios: Iterable[CorpusScenarioRecord],
    ) -> "CorpusManifest":
        if not corpus_id or not corpus_version:
            raise ValueError("corpus_id and corpus_version must be non-empty")
        ordered_sources = tuple(sorted(tuple(sources), key=lambda item: item.source_id))
        ordered_scenarios = tuple(sorted(tuple(scenarios), key=lambda item: item.scenario_id))
        source_ids = [item.source_id for item in ordered_sources]
        scenario_ids = [item.scenario_id for item in ordered_scenarios]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario_id values must be unique")
        known_sources = set(source_ids)
        for source in ordered_sources:
            if source.source_id in source.parent_source_ids:
                raise ValueError("source artifact cannot parent itself")
            missing = set(source.parent_source_ids) - known_sources
            if missing:
                raise ValueError(f"source {source.source_id} references unknown parents: {sorted(missing)}")
        _assert_acyclic_sources(ordered_sources)
        for scenario in ordered_scenarios:
            missing = set(scenario.source_artifact_ids) - known_sources
            if missing:
                raise ValueError(f"scenario {scenario.scenario_id} references unknown sources: {sorted(missing)}")
        return cls(corpus_id, corpus_version, ordered_sources, ordered_scenarios)

    def source(self, source_id: str) -> CorpusSourceArtifact:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(source_id)

    def scenario(self, scenario_id: str) -> CorpusScenarioRecord:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(scenario_id)

    def support_class(self, scenario: CorpusScenarioRecord) -> CorpusSupportClass:
        source_kinds = {self.source(source_id).source_kind for source_id in scenario.source_artifact_ids}
        has_empirical = any(kind in _EMPIRICAL_SOURCE_KINDS for kind in source_kinds)
        if scenario.derivation_kind in {ScenarioDerivationKind.RAW_REPLAY, ScenarioDerivationKind.FILTERED_REPLAY} and has_empirical:
            return CorpusSupportClass.DIRECT_EMPIRICAL
        if scenario.derivation_kind is ScenarioDerivationKind.SYNTHETIC_AUGMENTATION and has_empirical:
            return CorpusSupportClass.EMPIRICALLY_DERIVED
        if scenario.derivation_kind is ScenarioDerivationKind.RECONSTRUCTION:
            return CorpusSupportClass.DOCUMENTARY_RECONSTRUCTION
        if scenario.derivation_kind in {ScenarioDerivationKind.CALIBRATED_SYNTHETIC, ScenarioDerivationKind.FULLY_SYNTHETIC}:
            return CorpusSupportClass.SYNTHETIC
        if CorpusSourceKind.EXTERNAL_BENCHMARK in source_kinds:
            return CorpusSupportClass.EXTERNAL_BENCHMARK
        if source_kinds and source_kinds <= {CorpusSourceKind.DOCUMENTED_ASSUMPTION}:
            return CorpusSupportClass.ASSUMPTION_BASED
        return CorpusSupportClass.MIXED_OTHER

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "source_identities": [source.identity_sha256() for source in self.sources],
            "scenario_identities": [scenario.identity_sha256() for scenario in self.scenarios],
        })


@dataclass(frozen=True, slots=True)
class TargetPopulationStratum:
    stratum_id: str
    required_tags: tuple[CoverageTag, ...]
    minimum_total_scenarios: int = 1
    minimum_direct_empirical_scenarios: int = 0
    target_prevalence: float | None = None
    prevalence_tolerance: float = 0.05
    reference_evidence_ids: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        stratum_id: str,
        required_tags: Iterable[CoverageTag],
        minimum_total_scenarios: int = 1,
        minimum_direct_empirical_scenarios: int = 0,
        target_prevalence: float | None = None,
        prevalence_tolerance: float = 0.05,
        reference_evidence_ids: Iterable[str] = (),
    ) -> "TargetPopulationStratum":
        if not stratum_id:
            raise ValueError("stratum_id must be non-empty")
        tags = tuple(sorted(set(required_tags), key=lambda tag: (tag.dimension.value, tag.value)))
        if not tags:
            raise ValueError("target stratum must declare at least one required tag")
        if minimum_total_scenarios < 0 or minimum_direct_empirical_scenarios < 0:
            raise ValueError("minimum scenario counts must be non-negative")
        if target_prevalence is not None:
            if not isfinite(target_prevalence) or not 0.0 <= target_prevalence <= 1.0:
                raise ValueError("target_prevalence must be in [0, 1]")
        if not isfinite(prevalence_tolerance) or prevalence_tolerance < 0.0 or prevalence_tolerance > 1.0:
            raise ValueError("prevalence_tolerance must be in [0, 1]")
        refs = tuple(sorted(set(reference_evidence_ids)))
        if target_prevalence is not None and not refs:
            raise ValueError("target_prevalence requires reference_evidence_ids")
        return cls(
            stratum_id=stratum_id,
            required_tags=tags,
            minimum_total_scenarios=minimum_total_scenarios,
            minimum_direct_empirical_scenarios=minimum_direct_empirical_scenarios,
            target_prevalence=target_prevalence,
            prevalence_tolerance=prevalence_tolerance,
            reference_evidence_ids=refs,
        )

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "stratum_id": self.stratum_id,
            "required_tags": [
                {"dimension": tag.dimension.value, "value": tag.value}
                for tag in self.required_tags
            ],
            "minimum_total_scenarios": self.minimum_total_scenarios,
            "minimum_direct_empirical_scenarios": self.minimum_direct_empirical_scenarios,
            "target_prevalence": self.target_prevalence,
            "prevalence_tolerance": self.prevalence_tolerance,
            "reference_evidence_ids": list(self.reference_evidence_ids),
        })


@dataclass(frozen=True, slots=True)
class TargetPopulationDefinition:
    population_id: str
    population_version: str
    strata: tuple[TargetPopulationStratum, ...]

    @classmethod
    def from_unsorted(
        cls,
        *,
        population_id: str,
        population_version: str,
        strata: Iterable[TargetPopulationStratum],
    ) -> "TargetPopulationDefinition":
        if not population_id or not population_version:
            raise ValueError("population_id and population_version must be non-empty")
        ordered = tuple(sorted(tuple(strata), key=lambda item: item.stratum_id))
        ids = [item.stratum_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("stratum_id values must be unique")
        return cls(population_id, population_version, ordered)

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "population_id": self.population_id,
            "population_version": self.population_version,
            "stratum_identities": [item.identity_sha256() for item in self.strata],
        })


@dataclass(frozen=True, slots=True)
class CoverageStratumResult:
    stratum_id: str
    status: CoverageStatus
    matched_scenario_ids: tuple[str, ...]
    direct_empirical_scenario_ids: tuple[str, ...]
    support_counts: tuple[tuple[CorpusSupportClass, int], ...]
    corpus_prevalence: float
    target_prevalence: float | None
    prevalence_delta: float | None
    reason: str

    @property
    def matched_count(self) -> int:
        return len(self.matched_scenario_ids)

    @property
    def direct_empirical_count(self) -> int:
        return len(self.direct_empirical_scenario_ids)


@dataclass(frozen=True, slots=True)
class CorpusCoverageAudit:
    corpus_id: str
    corpus_version: str
    corpus_sha256: str
    population_id: str
    population_version: str
    population_sha256: str
    results: tuple[CoverageStratumResult, ...]
    component_id: TestbedFeature = TestbedFeature.C13_CORPUS_LINEAGE

    @property
    def gap_count(self) -> int:
        return sum(result.status in {
            CoverageStatus.MISSING,
            CoverageStatus.EMPIRICAL_GAP,
            CoverageStatus.INSUFFICIENT,
            CoverageStatus.UNDERREPRESENTED,
        } for result in self.results)

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "corpus_sha256": self.corpus_sha256,
            "population_sha256": self.population_sha256,
            "results": [
                {
                    "stratum_id": result.stratum_id,
                    "status": result.status.value,
                    "matched_scenario_ids": list(result.matched_scenario_ids),
                    "direct_empirical_scenario_ids": list(result.direct_empirical_scenario_ids),
                    "support_counts": [(kind.value, count) for kind, count in result.support_counts],
                    "corpus_prevalence": result.corpus_prevalence,
                    "target_prevalence": result.target_prevalence,
                    "prevalence_delta": result.prevalence_delta,
                    "reason": result.reason,
                }
                for result in self.results
            ],
        })


class CorpusCoverageEvaluator:
    component_id = TestbedFeature.C13_CORPUS_LINEAGE

    def evaluate(self, manifest: CorpusManifest, population: TargetPopulationDefinition) -> CorpusCoverageAudit:
        total = len(manifest.scenarios)
        results: list[CoverageStratumResult] = []
        for stratum in population.strata:
            matched = tuple(s for s in manifest.scenarios if s.has_all_tags(stratum.required_tags))
            direct = tuple(s for s in matched if manifest.support_class(s) is CorpusSupportClass.DIRECT_EMPIRICAL)
            counts = tuple(
                (kind, sum(manifest.support_class(s) is kind for s in matched))
                for kind in CorpusSupportClass
                if any(manifest.support_class(s) is kind for s in matched)
            )
            prevalence = (len(matched) / total) if total else 0.0
            delta = None if stratum.target_prevalence is None else prevalence - stratum.target_prevalence
            status, reason = _coverage_status(
                stratum=stratum,
                matched_count=len(matched),
                direct_empirical_count=len(direct),
                corpus_prevalence=prevalence,
            )
            results.append(CoverageStratumResult(
                stratum_id=stratum.stratum_id,
                status=status,
                matched_scenario_ids=tuple(s.scenario_id for s in matched),
                direct_empirical_scenario_ids=tuple(s.scenario_id for s in direct),
                support_counts=counts,
                corpus_prevalence=prevalence,
                target_prevalence=stratum.target_prevalence,
                prevalence_delta=delta,
                reason=reason,
            ))
        return CorpusCoverageAudit(
            corpus_id=manifest.corpus_id,
            corpus_version=manifest.corpus_version,
            corpus_sha256=manifest.identity_sha256(),
            population_id=population.population_id,
            population_version=population.population_version,
            population_sha256=population.identity_sha256(),
            results=tuple(results),
        )


class NoOpCorpusCoverageEvaluator:
    """Compatibility evaluator: records identities but makes no coverage claims."""

    component_id = TestbedFeature.C13_CORPUS_LINEAGE

    def evaluate(self, manifest: CorpusManifest, population: TargetPopulationDefinition) -> CorpusCoverageAudit:
        return CorpusCoverageAudit(
            corpus_id=manifest.corpus_id,
            corpus_version=manifest.corpus_version,
            corpus_sha256=manifest.identity_sha256(),
            population_id=population.population_id,
            population_version=population.population_version,
            population_sha256=population.identity_sha256(),
            results=(),
        )


def source_artifact_from_scenario_lineage(*, source_id: str, lineage: Any) -> CorpusSourceArtifact:
    """Adapt the C1 canonical lineage record without inferring population coverage tags."""

    kind_map = {
        "AIS": CorpusSourceKind.AIS_TRACKS,
        "AIS_REPLAY": CorpusSourceKind.AIS_TRACKS,
        "RECORDED_TRACK": CorpusSourceKind.SENSOR_RECORDING,
        "SYNTHETIC": CorpusSourceKind.SYNTHETIC_GENERATOR,
        "DNV_MARITIME_SCHEMA": CorpusSourceKind.EXTERNAL_BENCHMARK,
    }
    raw_kind = str(lineage.source_kind).upper()
    kind = kind_map.get(raw_kind, CorpusSourceKind.OTHER)
    return CorpusSourceArtifact.create(
        source_id=source_id,
        source_kind=kind,
        source_version=str(lineage.source_version or "unknown"),
        source_uri=str(lineage.source_uri or f"lineage://{source_id}"),
        content_sha256=str(lineage.source_sha256),
        license_id=str(lineage.license_id or "UNKNOWN"),
        attributes={
            "c1_adapter_id": str(lineage.adapter_id),
            "c1_attributes": dict(lineage.attributes),
        },
    )


def _coverage_status(
    *,
    stratum: TargetPopulationStratum,
    matched_count: int,
    direct_empirical_count: int,
    corpus_prevalence: float,
) -> tuple[CoverageStatus, str]:
    if matched_count == 0:
        return CoverageStatus.MISSING, "no corpus scenario matches the declared stratum"
    if direct_empirical_count < stratum.minimum_direct_empirical_scenarios:
        return (
            CoverageStatus.EMPIRICAL_GAP,
            f"direct empirical support {direct_empirical_count} < required {stratum.minimum_direct_empirical_scenarios}",
        )
    if matched_count < stratum.minimum_total_scenarios:
        return (
            CoverageStatus.INSUFFICIENT,
            f"matched scenarios {matched_count} < required {stratum.minimum_total_scenarios}",
        )
    if stratum.target_prevalence is None:
        return CoverageStatus.PRESENT_UNWEIGHTED, "present; no target prevalence was declared"
    delta = corpus_prevalence - stratum.target_prevalence
    if delta < -stratum.prevalence_tolerance:
        return CoverageStatus.UNDERREPRESENTED, "corpus prevalence is below the declared target prevalence band"
    if delta > stratum.prevalence_tolerance:
        return CoverageStatus.OVERREPRESENTED, "corpus prevalence is above the declared target prevalence band"
    return CoverageStatus.COVERED, "corpus prevalence is within the declared target prevalence band"


def _assert_acyclic_sources(sources: tuple[CorpusSourceArtifact, ...]) -> None:
    graph = {source.source_id: source.parent_source_ids for source in sources}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError("source lineage graph contains a cycle")
        visiting.add(node)
        for parent in graph[node]:
            visit(parent)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(graph):
        visit(node)
