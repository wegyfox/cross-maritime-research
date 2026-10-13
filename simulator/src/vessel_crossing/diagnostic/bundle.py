"""Portable offline-verifiable reproduction bundle for one completed D0 product run."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping
import zipfile

from ..nextgen.reference_envelope import ReferenceEnvelopeLevel
from ..reporting.reference_envelope import (
    IdealGapDecomposition,
    verify_reference_envelope_artifacts,
)
from ..version import __version__
from .contract import DiagnosticScenarioSpec
from .evidence import DiagnosticLevelEvidence
from .product import DiagnosticProductRun
from .run_manifest import verify_diagnostic_product_run_manifest

_FORMAT = "cross-diagnostic-reproduction-bundle-v1"
_SCHEMA_VERSION = 1
_BUNDLE_NAME = "DIAGNOSTIC_REPRODUCTION_BUNDLE.zip"
_MAX_FILES = 32
_MAX_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024


class DiagnosticReproductionBundleError(ValueError):
    """Raised when bundle creation prerequisites are missing or inconsistent."""


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


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _read(path: str | Path, *, name: str) -> bytes:
    target = Path(path)
    try:
        return target.read_bytes()
    except OSError as exc:
        raise DiagnosticReproductionBundleError(f"cannot read {name}: {exc}") from exc


def _file_record(raw: bytes) -> dict[str, Any]:
    return {"sha256": _sha(raw), "size_bytes": len(raw)}


def _load_json(raw: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticReproductionBundleError(f"invalid {name}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DiagnosticReproductionBundleError(f"{name} root must be an object")
    return payload


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def _write_deterministic_zip(path: Path, files: Mapping[str, bytes]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(files):
                archive.writestr(
                    _zip_info(name),
                    files[name],
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class DiagnosticReproductionBundleArtifacts:
    path: str
    bundle_sha256: str
    manifest_sha256: str
    file_count: int


@dataclass(frozen=True, slots=True)
class DiagnosticReproductionBundleVerification:
    passed: bool
    errors: tuple[str, ...]
    bundle_sha256: str | None = None
    manifest_sha256: str | None = None


def build_diagnostic_reproduction_bundle_payload(
    run: DiagnosticProductRun,
    scenario: DiagnosticScenarioSpec,
    output_directory: str | Path,
    *,
    date: str,
    scope: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Collect and cross-check current run artifacts before packaging them."""

    if not isinstance(run, DiagnosticProductRun):
        raise TypeError("run must be DiagnosticProductRun")
    if not isinstance(scenario, DiagnosticScenarioSpec):
        raise TypeError("scenario must be DiagnosticScenarioSpec")
    output_directory = Path(output_directory)

    report_verification = verify_reference_envelope_artifacts(output_directory)
    if not report_verification.passed:
        raise DiagnosticReproductionBundleError(
            "canonical report verification failed: " + "; ".join(report_verification.errors)
        )

    run_manifest_path = output_directory / "PRODUCT_RUN_MANIFEST.json"
    run_manifest_verification = verify_diagnostic_product_run_manifest(
        run_manifest_path,
        run=run,
        scenario=scenario,
        date=date,
        scope=scope,
    )
    if not run_manifest_verification.passed:
        raise DiagnosticReproductionBundleError(
            "product run manifest verification failed: "
            + "; ".join(run_manifest_verification.errors)
        )

    files: dict[str, bytes] = {
        "scenario/SCENARIO.json": scenario.to_json_bytes(),
        "run/PRODUCT_RUN_MANIFEST.json": _read(
            run_manifest_path,
            name="PRODUCT_RUN_MANIFEST.json",
        ),
        "report/diagnostics/IDEAL_GAP_DECOMPOSITION.json": _read(
            run.artifacts.diagnostics_path,
            name="canonical diagnostics",
        ),
        "report/REPORT.html": _read(run.artifacts.report_html_path, name="researcher report"),
        "report/REPORT_MANIFEST.json": _read(
            run.artifacts.manifest_path,
            name="report manifest",
        ),
        "report/DECISION_VIEW.json": _read(
            run.decision_view.json_path,
            name="decision view JSON",
        ),
        "report/DECISION_VIEW.html": _read(
            run.decision_view.html_path,
            name="decision view HTML",
        ),
    }

    for level in run.execution.study.levels:
        cache_path = output_directory / "RUN_EVIDENCE" / f"{level.level.value}.json"
        raw = _read(cache_path, name=f"RUN_EVIDENCE/{level.level.value}.json")
        if raw != level.to_json_bytes():
            raise DiagnosticReproductionBundleError(
                f"RUN_EVIDENCE/{level.level.value}.json does not match current study evidence"
            )
        files[f"run/RUN_EVIDENCE/{level.level.value}.json"] = raw

    if len(run.execution.study.levels) != len(tuple(ReferenceEnvelopeLevel)):
        raise DiagnosticReproductionBundleError("current study must contain exactly B0..B6")

    if _sha(files["report/DECISION_VIEW.json"]) != run.decision_view.json_sha256:
        raise DiagnosticReproductionBundleError(
            "DECISION_VIEW.json does not match current product-run artifact SHA-256"
        )
    if _sha(files["report/DECISION_VIEW.html"]) != run.decision_view.html_sha256:
        raise DiagnosticReproductionBundleError(
            "DECISION_VIEW.html does not match current product-run artifact SHA-256"
        )

    decision_json = _load_json(files["report/DECISION_VIEW.json"], name="DECISION_VIEW.json")
    source = decision_json.get("source")
    if not isinstance(source, Mapping):
        raise DiagnosticReproductionBundleError("DECISION_VIEW.json source must be an object")
    if source.get("decomposition_sha256") != run.decomposition.result_sha256:
        raise DiagnosticReproductionBundleError(
            "DECISION_VIEW.json does not reference current decomposition SHA-256"
        )

    run_manifest = _load_json(
        files["run/PRODUCT_RUN_MANIFEST.json"],
        name="PRODUCT_RUN_MANIFEST.json",
    )
    identity = run_manifest.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise DiagnosticReproductionBundleError(
            "reproduction bundle requires explicit execution identity"
        )

    payload = {
        "format": _FORMAT,
        "schema_version": _SCHEMA_VERSION,
        "product_version": __version__,
        "scenario_id": scenario.scenario_id,
        "scenario_sha256": scenario.scenario_sha256,
        "execution_identity": dict(identity),
        "decomposition_sha256": run.decomposition.result_sha256,
        "run_manifest_sha256": _sha(files["run/PRODUCT_RUN_MANIFEST.json"]),
        "report_manifest_sha256": _sha(files["report/REPORT_MANIFEST.json"]),
        "decision_view_sha256": _sha(files["report/DECISION_VIEW.json"]),
        "verification_scope": (
            "OFFLINE_INTEGRITY_AND_PROVENANCE_ONLY_NO_SCIENTIFIC_REEXECUTION"
        ),
        "files": {name: _file_record(raw) for name, raw in sorted(files.items())},
    }
    return payload, files


def write_diagnostic_reproduction_bundle(
    run: DiagnosticProductRun,
    scenario: DiagnosticScenarioSpec,
    output_directory: str | Path,
    *,
    date: str,
    scope: str,
    bundle_path: str | Path | None = None,
) -> DiagnosticReproductionBundleArtifacts:
    """Write one deterministic zip containing the complete bounded D0 run evidence."""

    manifest, files = build_diagnostic_reproduction_bundle_payload(
        run,
        scenario,
        output_directory,
        date=date,
        scope=scope,
    )
    manifest_bytes = _canonical_json(manifest)
    archive_files = dict(files)
    archive_files["BUNDLE_MANIFEST.json"] = manifest_bytes

    target = Path(bundle_path) if bundle_path is not None else Path(output_directory) / _BUNDLE_NAME
    _write_deterministic_zip(target, archive_files)
    raw = target.read_bytes()
    return DiagnosticReproductionBundleArtifacts(
        path=str(target),
        bundle_sha256=_sha(raw),
        manifest_sha256=_sha(manifest_bytes),
        file_count=len(archive_files),
    )


def _safe_member_name(name: str) -> bool:
    if not name or "\\" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def _bundle_file_bytes(archive: zipfile.ZipFile) -> tuple[dict[str, bytes], list[str]]:
    infos = archive.infolist()
    errors: list[str] = []
    names = [item.filename for item in infos]
    if len(names) > _MAX_FILES:
        errors.append("bundle contains too many members")
    if len(set(names)) != len(names):
        errors.append("bundle contains duplicate member names")
    total = 0
    files: dict[str, bytes] = {}
    for info in infos:
        if not _safe_member_name(info.filename):
            errors.append(f"unsafe bundle member name: {info.filename!r}")
            continue
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            errors.append(f"bundle symlink member is not allowed: {info.filename}")
            continue
        total += info.file_size
        if total > _MAX_TOTAL_UNCOMPRESSED_BYTES:
            errors.append("bundle exceeds maximum uncompressed size")
            break
        try:
            files[info.filename] = archive.read(info)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            errors.append(f"cannot read bundle member {info.filename}: {exc}")
    return files, errors


def verify_diagnostic_reproduction_bundle(
    bundle_path: str | Path,
) -> DiagnosticReproductionBundleVerification:
    """Verify bundle bytes and internal provenance links without executing science."""

    target = Path(bundle_path)
    try:
        bundle_raw = target.read_bytes()
        with zipfile.ZipFile(target, "r") as archive:
            files, errors = _bundle_file_bytes(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        return DiagnosticReproductionBundleVerification(
            False,
            (f"invalid reproduction bundle: {exc}",),
        )

    manifest_raw = files.get("BUNDLE_MANIFEST.json")
    if manifest_raw is None:
        errors.append("BUNDLE_MANIFEST.json is missing")
        return DiagnosticReproductionBundleVerification(
            False,
            tuple(errors),
            bundle_sha256=_sha(bundle_raw),
        )

    try:
        manifest = _load_json(manifest_raw, name="BUNDLE_MANIFEST.json")
    except DiagnosticReproductionBundleError as exc:
        errors.append(str(exc))
        return DiagnosticReproductionBundleVerification(
            False,
            tuple(errors),
            bundle_sha256=_sha(bundle_raw),
            manifest_sha256=_sha(manifest_raw),
        )

    if manifest_raw != _canonical_json(dict(manifest)):
        errors.append("BUNDLE_MANIFEST.json is not canonical JSON")
    if manifest.get("format") != _FORMAT:
        errors.append("unsupported reproduction bundle format")
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        errors.append("unsupported reproduction bundle schema_version")

    file_records = manifest.get("files")
    if not isinstance(file_records, Mapping):
        errors.append("bundle manifest files must be an object")
        file_records = {}
    expected_names = set(file_records) | {"BUNDLE_MANIFEST.json"}
    if set(files) != expected_names:
        errors.append("bundle member set does not exactly match bundle manifest")

    for name, record in file_records.items():
        raw = files.get(name)
        if raw is None:
            continue
        if not isinstance(record, Mapping):
            errors.append(f"invalid bundle file record: {name}")
            continue
        if record.get("sha256") != _sha(raw):
            errors.append(f"SHA-256 mismatch: {name}")
        if record.get("size_bytes") != len(raw):
            errors.append(f"size mismatch: {name}")

    _verify_bundle_cross_links(files, manifest, errors)
    return DiagnosticReproductionBundleVerification(
        not errors,
        tuple(errors),
        bundle_sha256=_sha(bundle_raw),
        manifest_sha256=_sha(manifest_raw),
    )


def _verify_bundle_cross_links(
    files: Mapping[str, bytes],
    manifest: Mapping[str, Any],
    errors: list[str],
) -> None:
    required = (
        "scenario/SCENARIO.json",
        "run/PRODUCT_RUN_MANIFEST.json",
        "report/diagnostics/IDEAL_GAP_DECOMPOSITION.json",
        "report/REPORT.html",
        "report/REPORT_MANIFEST.json",
        "report/DECISION_VIEW.json",
        "report/DECISION_VIEW.html",
    )
    missing = [name for name in required if name not in files]
    if missing:
        errors.extend(f"missing required bundle member: {name}" for name in missing)
        return

    try:
        scenario_payload = _load_json(files["scenario/SCENARIO.json"], name="SCENARIO.json")
        scenario = DiagnosticScenarioSpec.from_json_object(scenario_payload)
        if files["scenario/SCENARIO.json"] != scenario.to_json_bytes():
            errors.append("SCENARIO.json is not canonical diagnostic scenario JSON")
        if manifest.get("scenario_id") != scenario.scenario_id:
            errors.append("bundle manifest scenario_id does not match SCENARIO.json")
        if manifest.get("scenario_sha256") != scenario.scenario_sha256:
            errors.append("bundle manifest scenario_sha256 does not match SCENARIO.json")
    except (DiagnosticReproductionBundleError, TypeError, ValueError) as exc:
        errors.append(f"invalid bundled scenario contract: {exc}")
        scenario = None

    try:
        decomposition_payload = _load_json(
            files["report/diagnostics/IDEAL_GAP_DECOMPOSITION.json"],
            name="IDEAL_GAP_DECOMPOSITION.json",
        )
        decomposition = IdealGapDecomposition.from_json_object(decomposition_payload)
        if (
            files["report/diagnostics/IDEAL_GAP_DECOMPOSITION.json"]
            != decomposition.to_json_bytes()
        ):
            errors.append("bundled decomposition is not canonical JSON")
        if manifest.get("decomposition_sha256") != decomposition.result_sha256:
            errors.append("bundle manifest decomposition_sha256 mismatch")
    except (DiagnosticReproductionBundleError, TypeError, ValueError) as exc:
        errors.append(f"invalid bundled decomposition: {exc}")
        decomposition = None

    try:
        run_manifest = _load_json(
            files["run/PRODUCT_RUN_MANIFEST.json"],
            name="PRODUCT_RUN_MANIFEST.json",
        )
        run_raw = files["run/PRODUCT_RUN_MANIFEST.json"]
        if run_raw != _canonical_json(dict(run_manifest)):
            errors.append("bundled PRODUCT_RUN_MANIFEST.json is not canonical JSON")
        if manifest.get("run_manifest_sha256") != _sha(run_raw):
            errors.append("bundle run_manifest_sha256 mismatch")
        if run_manifest.get("scenario_sha256") != manifest.get("scenario_sha256"):
            errors.append("run manifest scenario_sha256 mismatch")
        if run_manifest.get("execution_identity") != manifest.get("execution_identity"):
            errors.append("execution identity mismatch between bundle and run manifest")
    except DiagnosticReproductionBundleError as exc:
        errors.append(str(exc))
        run_manifest = {}

    evidence_names = {
        level.value: f"run/RUN_EVIDENCE/{level.value}.json" for level in ReferenceEnvelopeLevel
    }
    if any(name not in files for name in evidence_names.values()):
        errors.append("bundle does not contain the complete B0-B6 RUN_EVIDENCE set")
    rows = run_manifest.get("levels") if isinstance(run_manifest, Mapping) else None
    row_by_level: dict[str, Mapping[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping) and isinstance(row.get("level"), str):
                row_by_level[row["level"]] = row
    else:
        errors.append("run manifest levels must be an array")
    for level_id, name in evidence_names.items():
        raw = files.get(name)
        if raw is None:
            continue
        try:
            evidence_payload = _load_json(raw, name=name)
            mutable = dict(evidence_payload)
            if mutable.pop("format", None) != "cross-diagnostic-level-evidence-v1":
                errors.append(f"unsupported level evidence format: {name}")
                continue
            evidence = DiagnosticLevelEvidence.from_json_object(mutable)
            if raw != evidence.to_json_bytes():
                errors.append(f"non-canonical level evidence: {name}")
            row = row_by_level.get(level_id)
            if row is None:
                errors.append(f"run manifest lacks level row: {level_id}")
            elif row.get("evidence_sha256") != evidence.evidence_sha256:
                errors.append(f"run manifest evidence SHA mismatch: {level_id}")
        except (DiagnosticReproductionBundleError, TypeError, ValueError) as exc:
            errors.append(f"invalid bundled level evidence {level_id}: {exc}")

    try:
        report_manifest = _load_json(
            files["report/REPORT_MANIFEST.json"],
            name="REPORT_MANIFEST.json",
        )
        report_raw = files["report/REPORT_MANIFEST.json"]
        if report_raw != _canonical_json(dict(report_manifest)):
            errors.append("bundled REPORT_MANIFEST.json is not canonical JSON")
        if manifest.get("report_manifest_sha256") != _sha(report_raw):
            errors.append("bundle report_manifest_sha256 mismatch")
        report_files = report_manifest.get("files")
        if isinstance(report_files, Mapping):
            for relative_name, record in report_files.items():
                member = f"report/{relative_name}"
                raw = files.get(member)
                if raw is None:
                    errors.append(f"report manifest member missing from bundle: {member}")
                    continue
                if not isinstance(record, Mapping):
                    errors.append(f"invalid report manifest record: {relative_name}")
                    continue
                if record.get("sha256") != _sha(raw):
                    errors.append(f"report manifest SHA mismatch: {relative_name}")
                if record.get("size_bytes") != len(raw):
                    errors.append(f"report manifest size mismatch: {relative_name}")
        else:
            errors.append("report manifest files must be an object")
        if decomposition is not None and report_manifest.get(
            "source_decomposition_canonical_sha256"
        ) != decomposition.result_sha256:
            errors.append("report manifest decomposition cross-link mismatch")
    except DiagnosticReproductionBundleError as exc:
        errors.append(str(exc))

    try:
        decision = _load_json(files["report/DECISION_VIEW.json"], name="DECISION_VIEW.json")
        decision_raw = files["report/DECISION_VIEW.json"]
        if decision_raw != _canonical_json(dict(decision)):
            errors.append("DECISION_VIEW.json is not canonical JSON")
        if manifest.get("decision_view_sha256") != _sha(decision_raw):
            errors.append("bundle decision_view_sha256 mismatch")
        source = decision.get("source")
        if decomposition is not None and (
            not isinstance(source, Mapping)
            or source.get("decomposition_sha256") != decomposition.result_sha256
        ):
            errors.append("decision view decomposition cross-link mismatch")
    except DiagnosticReproductionBundleError as exc:
        errors.append(str(exc))

    downstream = run_manifest.get("downstream") if isinstance(run_manifest, Mapping) else None
    if isinstance(downstream, Mapping) and decomposition is not None:
        if downstream.get("decomposition_sha256") != decomposition.result_sha256:
            errors.append("run manifest decomposition SHA mismatch")
        if downstream.get("report_manifest_sha256") != _sha(
            files["report/REPORT_MANIFEST.json"]
        ):
            errors.append("run manifest report-manifest SHA mismatch")
    elif run_manifest:
        errors.append("run manifest downstream object is missing")
