"""Fail-closed prefix-only reuse for D0 diagnostic execution evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Mapping

from ..nextgen.reference_envelope import ReferenceEnvelopeLevel
from ..reporting.reference_envelope import IdealGapDecomposition, write_reference_envelope_artifacts
from .classifier import classify_diagnostic_bottleneck
from .contract import DiagnosticScenarioSpec
from .effects import estimate_ladder_effects
from .evidence import DiagnosticLevelEvidence, DiagnosticStudyEvidence
from .execution import (
    DiagnosticAdapterRegistry,
    DiagnosticExecutionContext,
    DiagnosticExecutionError,
    DiagnosticExecutionResult,
    _identity_level,
    validate_diagnostic_execution,
)
from .interventions import InterventionDisposition
from .product import DiagnosticProductRun
from .synthesis import synthesize_generic_ideal_gap_decomposition
from .views import write_diagnostic_decision_view_artifacts


class DiagnosticResumeError(ValueError):
    """Raised when cached execution identity cannot safely match current preflight."""


@dataclass(frozen=True, slots=True)
class DiagnosticResumeEvidenceArtifacts:
    directory: str
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticVerifiedPrefix:
    levels: tuple[DiagnosticLevelEvidence, ...]
    verified_through: ReferenceEnvelopeLevel | None
    stop_reason: str | None

    @property
    def complete(self) -> bool:
        return len(self.levels) == len(tuple(ReferenceEnvelopeLevel))


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        raw = target.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticResumeError(f"invalid run manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise DiagnosticResumeError("run manifest root must be an object")
    if payload.get("format") != "cross-diagnostic-run-manifest-v1":
        raise DiagnosticResumeError("unsupported run manifest format")
    if payload.get("schema_version") != 1:
        raise DiagnosticResumeError("unsupported run manifest schema_version")
    if raw != _canonical_json(payload):
        raise DiagnosticResumeError("run manifest is not canonical JSON")
    return payload


def _level_cache_path(directory: str | Path, level: ReferenceEnvelopeLevel) -> Path:
    return Path(directory) / "RUN_EVIDENCE" / f"{level.value}.json"


def write_diagnostic_resume_evidence(
    study: DiagnosticStudyEvidence,
    output_directory: str | Path,
) -> DiagnosticResumeEvidenceArtifacts:
    root = Path(output_directory) / "RUN_EVIDENCE"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for level in study.levels:
        path = root / f"{level.level.value}.json"
        path.write_bytes(level.to_json_bytes())
        paths.append(str(path))
    return DiagnosticResumeEvidenceArtifacts(str(root), tuple(paths))


def _load_level(path: Path) -> DiagnosticLevelEvidence:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticResumeError(f"invalid cached level evidence {path.name}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DiagnosticResumeError(f"cached level evidence {path.name} must be an object")
    if payload.get("format") != "cross-diagnostic-level-evidence-v1":
        raise DiagnosticResumeError(f"unsupported cached level evidence format: {path.name}")
    level_payload = dict(payload)
    level_payload.pop("format", None)
    level = DiagnosticLevelEvidence.from_json_object(level_payload)
    if raw != level.to_json_bytes():
        raise DiagnosticResumeError(f"cached level evidence is not canonical: {path.name}")
    return level


def _manifest_rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = payload.get("levels")
    if not isinstance(raw, list):
        raise DiagnosticResumeError("run manifest levels must be an array")
    rows: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise DiagnosticResumeError("run manifest level entry must be an object")
        level_id = item.get("level")
        if not isinstance(level_id, str) or level_id in rows:
            raise DiagnosticResumeError("run manifest level ids must be unique strings")
        rows[level_id] = item
    return rows


def _metric_values(level: DiagnosticLevelEvidence, key: str) -> list[Any]:
    return [item.metrics[key] for item in level.replications if key in item.metrics]


def _uniform_or_list(values: list[Any]) -> Any:
    if not values:
        return None
    if all(item == values[0] for item in values):
        return values[0]
    return values


def _expected_row_semantics(
    level: DiagnosticLevelEvidence,
    *,
    disposition: str,
    applied_intervention_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "level": level.level.value,
        "disposition": disposition,
        "intervention_id": level.intervention_id,
        "target_factor": level.target_factor,
        "applied_intervention_ids": list(applied_intervention_ids),
        "causal": level.causal,
        "identity_of": level.identity_of.value if level.identity_of is not None else None,
        "evidence_sha256": level.evidence_sha256,
        "replication_count": len(level.replications),
        "replication_ids": [item.replication_id for item in level.replications],
        "seed_schedule": _metric_values(level, "seed"),
        "canonical_config_sha256": _uniform_or_list(
            _metric_values(level, "canonical_config_sha256")
        ),
        "canonical_result_sha256": _uniform_or_list(
            _metric_values(level, "canonical_result_sha256")
        ),
    }


def _require_run_identity(payload, scenario, preflight) -> None:
    adapter = preflight.adapter
    identity = adapter.execution_identity
    if identity is None:
        raise DiagnosticResumeError("resume requires current explicit execution identity")
    if payload.get("scenario_id") != scenario.scenario_id:
        raise DiagnosticResumeError("cached scenario_id does not match current scenario")
    if payload.get("scenario_sha256") != scenario.scenario_sha256:
        raise DiagnosticResumeError("cached scenario SHA does not match current scenario")
    if payload.get("plan_sha256") != preflight.plan.plan_sha256:
        raise DiagnosticResumeError("cached plan SHA does not match current plan")
    if payload.get("runtime_adapter") != adapter.adapter_id:
        raise DiagnosticResumeError("cached runtime adapter does not match current adapter")
    cached_identity = payload.get("execution_identity")
    expected_identity = {
        "identity_id": identity.identity_id,
        "identity_sha256": identity.identity_sha256,
    }
    if cached_identity != expected_identity:
        raise DiagnosticResumeError("cached execution identity does not match current binding set")


def prepare_diagnostic_resume_prefix(
    scenario: DiagnosticScenarioSpec,
    *,
    registry: DiagnosticAdapterRegistry,
    manifest_path: str | Path,
    evidence_directory: str | Path,
) -> DiagnosticVerifiedPrefix:
    """Accept the longest verified B0->... prefix and reject sparse reuse."""

    preflight = validate_diagnostic_execution(scenario, registry=registry)
    manifest = _load_manifest(manifest_path)
    _require_run_identity(manifest, scenario, preflight)
    rows = _manifest_rows(manifest)
    runs: list[DiagnosticLevelEvidence] = []
    applied: list[str] = []

    baseline_level = ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK
    baseline_row = rows.get(baseline_level.value)
    if baseline_row is None:
        return DiagnosticVerifiedPrefix((), None, "B0_MANIFEST_ROW_MISSING")
    try:
        baseline = _load_level(_level_cache_path(evidence_directory, baseline_level))
    except DiagnosticResumeError as exc:
        return DiagnosticVerifiedPrefix((), None, str(exc))
    if baseline.level is not baseline_level:
        return DiagnosticVerifiedPrefix((), None, "B0_CACHED_LEVEL_MISMATCH")
    expected = _expected_row_semantics(
        baseline,
        disposition="BASELINE",
        applied_intervention_ids=(),
    )
    if any(baseline_row.get(key) != value for key, value in expected.items()):
        return DiagnosticVerifiedPrefix((), None, "B0_MANIFEST_EVIDENCE_MISMATCH")
    runs.append(baseline)

    for entry in preflight.plan.entries:
        previous = runs[-1]
        row = rows.get(entry.level.value)
        if row is None:
            return DiagnosticVerifiedPrefix(
                tuple(runs), runs[-1].level, f"{entry.level.value}_MANIFEST_ROW_MISSING"
            )
        if entry.disposition is InterventionDisposition.IDENTITY:
            candidate = _identity_level(previous, entry)
        else:
            applied.append(entry.intervention_id)
            try:
                candidate = _load_level(_level_cache_path(evidence_directory, entry.level))
            except DiagnosticResumeError as exc:
                return DiagnosticVerifiedPrefix(tuple(runs), runs[-1].level, str(exc))
        if candidate.level is not entry.level:
            return DiagnosticVerifiedPrefix(
                tuple(runs), runs[-1].level, f"{entry.level.value}_CACHED_LEVEL_MISMATCH"
            )
        expected = _expected_row_semantics(
            candidate,
            disposition=entry.disposition.value,
            applied_intervention_ids=tuple(applied),
        )
        if any(row.get(key) != value for key, value in expected.items()):
            return DiagnosticVerifiedPrefix(
                tuple(runs), runs[-1].level, f"{entry.level.value}_MANIFEST_EVIDENCE_MISMATCH"
            )
        runs.append(candidate)
    return DiagnosticVerifiedPrefix(tuple(runs), runs[-1].level, None)


def run_diagnostic_orchestrator_with_resume(
    scenario: DiagnosticScenarioSpec,
    *,
    registry: DiagnosticAdapterRegistry,
    prefix: DiagnosticVerifiedPrefix,
) -> DiagnosticExecutionResult:
    preflight = validate_diagnostic_execution(scenario, registry=registry)
    adapter = preflight.adapter
    plan = preflight.plan
    runs = list(prefix.levels)
    canonical_levels = tuple(ReferenceEnvelopeLevel)
    if tuple(item.level for item in runs) != canonical_levels[: len(runs)]:
        raise DiagnosticResumeError("verified prefix is not canonical B0->... order")

    applied: list[str] = []
    completed_entries = max(0, len(runs) - 1)
    for entry in plan.entries[:completed_entries]:
        if entry.disposition is not InterventionDisposition.IDENTITY:
            applied.append(entry.intervention_id)

    if not runs:
        baseline = adapter.run(
            DiagnosticExecutionContext(
                scenario,
                plan,
                ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK,
                None,
                None,
                (),
                (),
            )
        )
        runs.append(baseline)
        completed_entries = 0

    for entry in plan.entries[completed_entries:]:
        previous = runs[-1]
        if entry.disposition is InterventionDisposition.IDENTITY:
            runs.append(_identity_level(previous, entry))
            continue
        applied.append(entry.intervention_id)
        result = adapter.run(
            DiagnosticExecutionContext(
                scenario,
                plan,
                entry.level,
                entry,
                previous,
                tuple(runs),
                tuple(applied),
            )
        )
        if result.intervention_id != entry.intervention_id:
            raise DiagnosticExecutionError(
                f"adapter intervention identity mismatch at {entry.level.value}"
            )
        if result.target_factor != entry.target_factor:
            raise DiagnosticExecutionError(
                f"adapter target-factor identity mismatch at {entry.level.value}"
            )
        expected_causal = entry.disposition is InterventionDisposition.APPLICABLE
        if result.causal is not expected_causal:
            raise DiagnosticExecutionError(
                f"adapter causal/reference mismatch at {entry.level.value}"
            )
        runs.append(result)

    provenance: dict[str, Any] = {
        "scenario_sha256": scenario.scenario_sha256,
        "plan_sha256": plan.plan_sha256,
        "runtime_adapter": adapter.adapter_id,
    }
    if adapter.execution_identity is not None:
        provenance.update(
            {
                "execution_identity_id": adapter.execution_identity.identity_id,
                "execution_identity_sha256": adapter.execution_identity.identity_sha256,
            }
        )
    study = DiagnosticStudyEvidence(
        scenario.scenario_id,
        tuple(runs),
        claim_scope=scenario.claim_scope,
        provenance=provenance,
    )
    effects = estimate_ladder_effects(
        study,
        confidence_level=scenario.experiment.confidence_level,
    )
    classification = classify_diagnostic_bottleneck(effects)
    return DiagnosticExecutionResult(study, effects, classification)


def run_diagnostic_product_with_resume(
    scenario: DiagnosticScenarioSpec,
    *,
    registry: DiagnosticAdapterRegistry,
    prefix: DiagnosticVerifiedPrefix,
    output_directory: str | Path,
    date: str,
    scope: str,
    title: str | None = None,
) -> DiagnosticProductRun:
    execution = run_diagnostic_orchestrator_with_resume(
        scenario,
        registry=registry,
        prefix=prefix,
    )
    payload = synthesize_generic_ideal_gap_decomposition(
        execution.study,
        execution.effects,
        execution.classification,
        date=date,
        scope=scope,
    )
    decomposition = IdealGapDecomposition.from_json_object(payload)
    artifacts = write_reference_envelope_artifacts(
        decomposition,
        output_directory,
        title=title,
    )
    decision_view = write_diagnostic_decision_view_artifacts(
        decomposition,
        output_directory,
        title=(f"{title} — Decision View" if title else None),
    )
    return DiagnosticProductRun(execution, decomposition, artifacts, decision_view)
