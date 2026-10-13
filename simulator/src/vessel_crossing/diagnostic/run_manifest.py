"""Versioned product-run manifest for verified D0 execution provenance.

The run manifest binds an already completed product run to explicit scientific
execution identity.  It is not a cache and does not implement resume; A54 may use
this verified surface as the prerequisite for fail-closed prefix reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from ..version import __version__
from .contract import DiagnosticScenarioSpec
from .effects import diagnostic_effects_sha256
from .interventions import InterventionDisposition
from .planning import build_diagnostic_plan
from .product import DiagnosticProductRun

_FORMAT = "cross-diagnostic-run-manifest-v1"
_SCHEMA_VERSION = 1


class DiagnosticRunManifestError(ValueError):
    """Raised when reusable run-manifest provenance is absent or inconsistent."""


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


def _sha256_hex(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise DiagnosticRunManifestError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _execution_identity(provenance: Mapping[str, Any]) -> tuple[str, str]:
    identity_id = provenance.get("execution_identity_id")
    identity_sha = provenance.get("execution_identity_sha256")
    if not isinstance(identity_id, str) or not identity_id.strip():
        raise DiagnosticRunManifestError(
            "reusable product-run manifest requires explicit execution identity"
        )
    return identity_id.strip(), _sha256_hex(
        identity_sha,
        name="execution_identity_sha256",
    )


def _ordered_metric_values(level, key: str) -> list[Any]:
    values: list[Any] = []
    for replication in level.replications:
        if key in replication.metrics:
            values.append(replication.metrics[key])
    return values


def _uniform_or_list(values: list[Any]) -> Any:
    if not values:
        return None
    if all(item == values[0] for item in values):
        return values[0]
    return values


def build_diagnostic_product_run_manifest(
    run: DiagnosticProductRun,
    scenario: DiagnosticScenarioSpec,
    *,
    date: str,
    scope: str,
) -> dict[str, Any]:
    """Build the canonical run identity object without writing it."""

    if not isinstance(run, DiagnosticProductRun):
        raise TypeError("run must be DiagnosticProductRun")
    if not isinstance(scenario, DiagnosticScenarioSpec):
        raise TypeError("scenario must be DiagnosticScenarioSpec")
    if not isinstance(date, str) or not date.strip():
        raise DiagnosticRunManifestError("date must be non-empty")
    if not isinstance(scope, str) or not scope.strip():
        raise DiagnosticRunManifestError("scope must be non-empty")

    plan = build_diagnostic_plan(scenario)
    provenance = run.execution.study.provenance
    identity_id, identity_sha = _execution_identity(provenance)
    if provenance.get("scenario_sha256") != scenario.scenario_sha256:
        raise DiagnosticRunManifestError("study provenance scenario SHA does not match scenario")
    if provenance.get("plan_sha256") != plan.plan_sha256:
        raise DiagnosticRunManifestError("study provenance plan SHA does not match current plan")
    runtime_adapter = provenance.get("runtime_adapter")
    if not isinstance(runtime_adapter, str) or not runtime_adapter.strip():
        raise DiagnosticRunManifestError("study provenance runtime_adapter is missing")

    entries = {item.level: item for item in plan.entries}
    applied: list[str] = []
    level_rows: list[dict[str, Any]] = []
    for level_evidence in run.execution.study.levels:
        entry = entries.get(level_evidence.level)
        if entry is None:
            disposition = "BASELINE"
        else:
            disposition = entry.disposition.value
            if entry.disposition in {
                InterventionDisposition.APPLICABLE,
                InterventionDisposition.REFERENCE,
            }:
                applied.append(entry.intervention_id)
        seeds = _ordered_metric_values(level_evidence, "seed")
        config_hashes = _ordered_metric_values(level_evidence, "canonical_config_sha256")
        result_hashes = _ordered_metric_values(level_evidence, "canonical_result_sha256")
        level_rows.append(
            {
                "level": level_evidence.level.value,
                "disposition": disposition,
                "intervention_id": level_evidence.intervention_id,
                "target_factor": level_evidence.target_factor,
                "applied_intervention_ids": list(applied),
                "causal": level_evidence.causal,
                "identity_of": (
                    level_evidence.identity_of.value
                    if level_evidence.identity_of is not None
                    else None
                ),
                "evidence_sha256": level_evidence.evidence_sha256,
                "replication_count": len(level_evidence.replications),
                "replication_ids": [
                    item.replication_id for item in level_evidence.replications
                ],
                "seed_schedule": seeds,
                "canonical_config_sha256": _uniform_or_list(config_hashes),
                "canonical_result_sha256": _uniform_or_list(result_hashes),
            }
        )

    return {
        "format": _FORMAT,
        "schema_version": _SCHEMA_VERSION,
        "scenario_id": scenario.scenario_id,
        "scenario_sha256": scenario.scenario_sha256,
        "plan_sha256": plan.plan_sha256,
        "runtime_adapter": runtime_adapter.strip(),
        "execution_identity": {
            "identity_id": identity_id,
            "identity_sha256": identity_sha,
        },
        "product_version": __version__,
        "report_metadata": {"date": date.strip(), "scope": scope.strip()},
        "levels": level_rows,
        "downstream": {
            "study_evidence_sha256": run.execution.study.evidence_sha256,
            "effects_sha256": diagnostic_effects_sha256(run.execution.effects),
            "classification_sha256": run.execution.classification.classification_sha256,
            "decomposition_sha256": run.decomposition.result_sha256,
            "report_manifest_sha256": run.artifacts.manifest_sha256,
        },
    }


@dataclass(frozen=True, slots=True)
class DiagnosticRunManifestArtifacts:
    path: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class DiagnosticRunManifestVerification:
    passed: bool
    errors: tuple[str, ...]
    manifest_sha256: str | None = None


def write_diagnostic_product_run_manifest(
    run: DiagnosticProductRun,
    scenario: DiagnosticScenarioSpec,
    output_directory: str | Path,
    *,
    date: str,
    scope: str,
) -> DiagnosticRunManifestArtifacts:
    payload = build_diagnostic_product_run_manifest(run, scenario, date=date, scope=scope)
    raw = _canonical_json(payload)
    path = Path(output_directory) / "PRODUCT_RUN_MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return DiagnosticRunManifestArtifacts(str(path), sha256(raw).hexdigest())


def verify_diagnostic_product_run_manifest(
    path: str | Path,
    *,
    run: DiagnosticProductRun | None = None,
    scenario: DiagnosticScenarioSpec | None = None,
    date: str | None = None,
    scope: str | None = None,
) -> DiagnosticRunManifestVerification:
    target = Path(path)
    try:
        raw = target.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return DiagnosticRunManifestVerification(False, (f"invalid run manifest: {exc}",))
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return DiagnosticRunManifestVerification(False, ("run manifest root must be an object",))
    if payload.get("format") != _FORMAT:
        errors.append("unsupported run manifest format")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        errors.append("unsupported run manifest schema_version")
    canonical = _canonical_json(dict(payload))
    if raw != canonical:
        errors.append("run manifest is not canonical JSON")
    digest = sha256(canonical).hexdigest()

    if any(item is not None for item in (run, scenario, date, scope)):
        if run is None or scenario is None or date is None or scope is None:
            errors.append(
                "current-run verification requires run, scenario, date and scope together"
            )
        else:
            try:
                expected = build_diagnostic_product_run_manifest(
                    run,
                    scenario,
                    date=date,
                    scope=scope,
                )
            except (DiagnosticRunManifestError, TypeError, ValueError) as exc:
                errors.append(f"cannot build current run identity: {exc}")
            else:
                if dict(payload) != expected:
                    errors.append("run manifest does not match current run identity")
    return DiagnosticRunManifestVerification(not errors, tuple(errors), digest)
