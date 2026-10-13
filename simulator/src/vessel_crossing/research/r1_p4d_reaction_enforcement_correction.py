"""Correction overlay for the R1 P4D same-route reaction-enforcement defect.

The scientific design is not changed.  The exact frozen V1 confirmatory input plan is
reconstructed and retained as the input authority.  Structurally unaffected physical
results may be reused with explicit parent provenance; every point whose non-zero HS
latency treatment can activate is rerun on the corrected runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from typing import Iterable, Mapping

from .r1_p4d_confirmatory_analysis import analyze_p4d_confirmatory_results
from .r1_p4d_confirmatory_execution import (
    P4D_CONFIRMATORY_CAMPAIGN_ID,
    P4D_CONFIRMATORY_RESULT_FORMAT,
    P4DConfirmatoryPlan,
    P4DConfirmatoryPoint,
    build_p4d_confirmatory_plan,
    run_p4d_confirmatory_point,
)

PARENT_SOURCE_ARCHIVE_SHA256 = "8cd08acaa5a2367df753c4d624d105e521c8fc3aa356e5f91792da0b1a40ece3"
PARENT_PROTOCOL_SHA256 = "c6e85adbfba5ee2b7f34cd37db6b38bc6767c77a1aa551a93024de13e804861c"
PARENT_STATISTICAL_PLAN_SHA256 = "ab4cdf4e11a4a016555e9791338495c73ec673146d140a9a50bc162c4a8da7f2"
PARENT_SEED_SCHEDULE_SHA256 = "2ab7962004bd3a298a9b39a9d05b6acfd892c17a41599bd2134ab4ccd62b9a14"
PARENT_QUALIFICATION_AUDIT_SHA256 = "21a96bd31a0aa7aa2bc5bf2810515533d64db1a8ae39d43224744768c778a0f4"
PARENT_PLAN_SHA256 = "d55f08768c3b58027002a54dc923e2d3df256ce1872df0835506edbee97475b0"
PARENT_FINAL_H2_RESULTS_SHA256 = "69341abadb421ad66ceb20f29845ade7c6252c58a1e4ddec53e3b50dd1bf6ab3"

CORRECTION_CAMPAIGN_ID = "R1_P4D_REACTION_ENFORCEMENT_CORRECTED_V4_PERFORMANCE_EQUIVALENT_2026-09-25"
CORRECTION_PLAN_FORMAT = "cross-r1-p4d-reaction-enforcement-correction-plan-v2"
CORRECTION_RESULT_FORMAT = "cross-r1-p4d-reaction-enforcement-corrected-point-v2"
CORRECTION_ANALYSIS_FORMAT = "cross-r1-p4d-reaction-enforcement-corrected-analysis-v2"
DEFECT_ID = "P4D-SR-REACTION-ENFORCEMENT-001"
REUSE_ACTION = "REUSE_PARENT_UNAFFECTED"
RERUN_ACTION = "RERUN_CORRECTED_RUNTIME"


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


@lru_cache(maxsize=1)
def exact_parent_plan() -> P4DConfirmatoryPlan:
    """Reconstruct and verify the exact preregistered 1056-point input plan."""
    plan = build_p4d_confirmatory_plan(
        source_archive_sha256=PARENT_SOURCE_ARCHIVE_SHA256,
        protocol_sha256=PARENT_PROTOCOL_SHA256,
        statistical_plan_sha256=PARENT_STATISTICAL_PLAN_SHA256,
        seed_schedule_sha256=PARENT_SEED_SCHEDULE_SHA256,
        qualification_postrun_audit_sha256=PARENT_QUALIFICATION_AUDIT_SHA256,
    )
    if plan.identity_sha256 != PARENT_PLAN_SHA256:
        raise RuntimeError("frozen parent P4D plan identity drift")
    return plan


def correction_action(point: P4DConfirmatoryPoint) -> str:
    """Return the outcome-independent correction action for one frozen point.

    NEUTRAL has no extra HS reaction allowance.  At P100 every vessel is AS, so the HS
    latency treatment is inactive.  Every non-zero-latency point below P100 is rerun
    conservatively, whether or not a same-route constraint happened to bind historically.
    """
    if point.arm == "NEUTRAL" or point.penetration == 1.0:
        return REUSE_ACTION
    return RERUN_ACTION


@dataclass(frozen=True, slots=True)
class P4DReactionEnforcementCorrectionPlan:
    corrected_source_archive_sha256: str
    defect_audit_sha256: str
    parent_plan_sha256: str = PARENT_PLAN_SHA256
    parent_final_results_sha256: str = PARENT_FINAL_H2_RESULTS_SHA256
    campaign_id: str = CORRECTION_CAMPAIGN_ID

    def __post_init__(self) -> None:
        for name in (
            "corrected_source_archive_sha256",
            "defect_audit_sha256",
            "parent_plan_sha256",
            "parent_final_results_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a 64-character SHA-256 string")
        if self.parent_plan_sha256 != exact_parent_plan().identity_sha256:
            raise ValueError("parent plan SHA-256 does not match exact frozen input plan")

    @property
    def parent_plan(self) -> P4DConfirmatoryPlan:
        return exact_parent_plan()

    @property
    def rerun_point_ids(self) -> tuple[str, ...]:
        return tuple(p.point_id for p in self.parent_plan.points if correction_action(p) == RERUN_ACTION)

    @property
    def reused_point_ids(self) -> tuple[str, ...]:
        return tuple(p.point_id for p in self.parent_plan.points if correction_action(p) == REUSE_ACTION)

    @property
    def document(self) -> dict[str, object]:
        parent = self.parent_plan
        return {
            "format": CORRECTION_PLAN_FORMAT,
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "defect_id": DEFECT_ID,
            "corrected_source_archive_sha256": self.corrected_source_archive_sha256,
            "defect_audit_sha256": self.defect_audit_sha256,
            "parent_campaign_id": P4D_CONFIRMATORY_CAMPAIGN_ID,
            "parent_plan_sha256": self.parent_plan_sha256,
            "parent_source_archive_sha256": PARENT_SOURCE_ARCHIVE_SHA256,
            "parent_final_results_sha256": self.parent_final_results_sha256,
            "scientific_design_change": False,
            "input_schedule_change": False,
            "statistical_analysis_change": False,
            "selection_rule": "RERUN iff arm != NEUTRAL and penetration < 1.0",
            "total_points": len(parent.points),
            "rerun_point_count": len(self.rerun_point_ids),
            "reused_point_count": len(self.reused_point_ids),
            "rerun_point_ids": list(self.rerun_point_ids),
            "reused_point_ids": list(self.reused_point_ids),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.document)

    @property
    def identity_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()


def _point_by_id(plan: P4DConfirmatoryPlan, point_id: str) -> P4DConfirmatoryPoint:
    try:
        return next(p for p in plan.points if p.point_id == point_id)
    except StopIteration as exc:
        raise ValueError(f"unknown parent point id: {point_id}") from exc


def _wrap_physical(
    plan: P4DReactionEnforcementCorrectionPlan,
    point: P4DConfirmatoryPoint,
    physical: Mapping[str, object],
    *,
    action: str,
    parent_result_sha256: str | None,
) -> dict[str, object]:
    if action != correction_action(point):
        raise ValueError("correction action does not match frozen structural selection rule")
    if action == REUSE_ACTION and (not isinstance(parent_result_sha256, str) or len(parent_result_sha256) != 64):
        raise ValueError("reused points require the exact parent point-result SHA-256")
    if action == RERUN_ACTION and parent_result_sha256 is not None:
        raise ValueError("rerun points must not masquerade as reused parent bytes")
    return {
        "format": CORRECTION_RESULT_FORMAT,
        "schema_version": 1,
        "campaign_id": plan.campaign_id,
        "correction_plan_sha256": plan.identity_sha256,
        "corrected_source_archive_sha256": plan.corrected_source_archive_sha256,
        "defect_id": DEFECT_ID,
        "parent_campaign_id": P4D_CONFIRMATORY_CAMPAIGN_ID,
        "parent_plan_sha256": PARENT_PLAN_SHA256,
        "parent_point_id": point.point_id,
        "parent_point_identity_sha256": point.compute_identity_sha256(),
        "correction_action": action,
        "parent_point_result_sha256": parent_result_sha256,
        "physical": json.loads(json.dumps(physical)),
    }


def rerun_corrected_point(
    plan: P4DReactionEnforcementCorrectionPlan,
    point_id: str,
) -> dict[str, object]:
    """Execute one structurally affected point with exact parent inputs on corrected code."""
    parent = plan.parent_plan
    point = _point_by_id(parent, point_id)
    if correction_action(point) != RERUN_ACTION:
        raise ValueError("point is structurally unaffected and must be reused, not rerun")
    # The old runner remains the exact input-identity oracle.  Only its physical runtime
    # is now corrected; the old outer envelope is discarded and replaced below.
    old_envelope = json.loads(run_p4d_confirmatory_point(parent, point))
    return _wrap_physical(
        plan,
        point,
        old_envelope["physical"],
        action=RERUN_ACTION,
        parent_result_sha256=None,
    )


def reuse_parent_point(
    plan: P4DReactionEnforcementCorrectionPlan,
    parent_document: Mapping[str, object],
    *,
    parent_result_sha256: str,
) -> dict[str, object]:
    """Wrap one structurally unaffected H2 point with explicit correction provenance."""
    parent = plan.parent_plan
    if parent_document.get("format") != P4D_CONFIRMATORY_RESULT_FORMAT:
        raise ValueError("unexpected parent P4D result format")
    if parent_document.get("campaign_id") != P4D_CONFIRMATORY_CAMPAIGN_ID:
        raise ValueError("unexpected parent campaign")
    if parent_document.get("plan_sha256") != PARENT_PLAN_SHA256:
        raise ValueError("parent point is not bound to the exact frozen plan")
    meta = parent_document.get("point")
    if not isinstance(meta, Mapping) or not isinstance(meta.get("point_id"), str):
        raise ValueError("parent point metadata missing")
    point = _point_by_id(parent, str(meta["point_id"]))
    if meta.get("identity_sha256") != point.compute_identity_sha256():
        raise ValueError("parent point identity mismatch")
    if correction_action(point) != REUSE_ACTION:
        raise ValueError("affected point cannot be reused by the correction overlay")
    physical = parent_document.get("physical")
    if not isinstance(physical, Mapping):
        raise ValueError("parent physical document missing")
    return _wrap_physical(
        plan,
        point,
        physical,
        action=REUSE_ACTION,
        parent_result_sha256=parent_result_sha256,
    )


def analyze_corrected_results(
    plan: P4DReactionEnforcementCorrectionPlan,
    documents: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the exact preregistered V1 analyzer to the corrected complete physical set."""
    parent = plan.parent_plan
    correction_plan_sha256 = plan.identity_sha256
    parent_plan_sha256 = parent.identity_sha256
    by_id: dict[str, Mapping[str, object]] = {}
    for document in documents:
        if document.get("format") != CORRECTION_RESULT_FORMAT:
            raise ValueError("unexpected corrected result format")
        if document.get("campaign_id") != plan.campaign_id:
            raise ValueError("corrected campaign mismatch")
        if document.get("correction_plan_sha256") != correction_plan_sha256:
            raise ValueError("correction plan mismatch")
        point_id = document.get("parent_point_id")
        if not isinstance(point_id, str) or point_id in by_id:
            raise ValueError("unknown or duplicate corrected point")
        point = _point_by_id(parent, point_id)
        if document.get("parent_point_identity_sha256") != point.compute_identity_sha256():
            raise ValueError("corrected parent-point identity mismatch")
        if document.get("correction_action") != correction_action(point):
            raise ValueError("corrected result action violates frozen selection rule")
        physical = document.get("physical")
        if not isinstance(physical, Mapping):
            raise ValueError("corrected physical document missing")
        by_id[point_id] = document
    if len(by_id) != len(parent.points):
        raise ValueError(f"complete 1056-point corrected set required; got {len(by_id)}")

    adapted = []
    for point in parent.points:
        corrected = by_id[point.point_id]
        adapted.append({
            "format": P4D_CONFIRMATORY_RESULT_FORMAT,
            "schema_version": 1,
            "campaign_id": parent.campaign_id,
            "plan_sha256": parent_plan_sha256,
            "source_archive_sha256": parent.source_archive_sha256,
            "point": dict(point.identity_payload(), identity_sha256=point.compute_identity_sha256()),
            "physical": corrected["physical"],
        })
    frozen = analyze_p4d_confirmatory_results(parent, adapted)
    return {
        "format": CORRECTION_ANALYSIS_FORMAT,
        "schema_version": 1,
        "campaign_id": plan.campaign_id,
        "correction_plan_sha256": correction_plan_sha256,
        "corrected_source_archive_sha256": plan.corrected_source_archive_sha256,
        "defect_id": DEFECT_ID,
        "parent_plan_sha256": PARENT_PLAN_SHA256,
        "reused_point_count": len(plan.reused_point_ids),
        "rerun_point_count": len(plan.rerun_point_ids),
        "analysis_method": "EXACT_FROZEN_V1_PREREGISTERED_ANALYZER_ON_CORRECTED_1056_POINT_PHYSICAL_SET",
        "scientific_design_change": False,
        "frozen_analysis": frozen,
    }


def _is_canonical_parent_result_member(name: str) -> bool:
    """Return whether *name* belongs to the frozen H2 canonical result namespace."""
    return (
        name.startswith("results/")
        and name.count("/") == 1
        and name.lower().endswith(".json")
        and not name.endswith("/")
    )


def load_parent_results_archive(archive_path: str | "Path") -> dict[str, tuple[dict[str, object], str]]:
    """Load and verify the exact frozen H2 result archive.

    Returns a mapping from point id to ``(parsed_document, sha256_of_stored_json_bytes)``.
    The archive identity is checked before any reuse decision is made.  JSON members that
    are not physical point results (analysis/manifests) are ignored.
    """
    from pathlib import Path
    from zipfile import ZipFile

    path = Path(archive_path)
    actual_archive_sha256 = sha256(path.read_bytes()).hexdigest()
    if actual_archive_sha256 != PARENT_FINAL_H2_RESULTS_SHA256:
        raise ValueError(
            "parent H2 archive SHA-256 mismatch: "
            f"expected {PARENT_FINAL_H2_RESULTS_SHA256}, got {actual_archive_sha256}"
        )

    parent = exact_parent_plan()
    expected_ids = {point.point_id for point in parent.points}
    results: dict[str, tuple[dict[str, object], str]] = {}
    with ZipFile(path, "r") as zf:
        names = zf.namelist()
        for name in names:
            # H2 intentionally preserves superseded invalid H1 records and several
            # provenance/audit JSON documents alongside the canonical result set.
            # Only the frozen ``results/<file>.json`` namespace is authoritative for
            # the 1056-point physical dataset.  Scanning JSON by content alone would
            # double-count the 44 superseded H1 records.
            if not _is_canonical_parent_result_member(name):
                continue
            raw = zf.read(name)
            sidecar_name = f"{name}.sha256"
            if sidecar_name not in names:
                raise ValueError(f"parent result {name!r} is missing SHA-256 sidecar")
            sidecar = zf.read(sidecar_name).decode("utf-8").strip().split()
            actual_json_sha256 = sha256(raw).hexdigest()
            if not sidecar or sidecar[0] != actual_json_sha256:
                raise ValueError(f"parent result {name!r} SHA-256 sidecar mismatch")
            try:
                document = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ValueError(f"parent result {name!r} is not valid UTF-8 JSON") from None
            if not isinstance(document, dict) or document.get("format") != P4D_CONFIRMATORY_RESULT_FORMAT:
                raise ValueError(f"parent result {name!r} has unexpected result format")
            meta = document.get("point")
            if not isinstance(meta, Mapping):
                raise ValueError(f"parent result {name!r} has no point metadata")
            point_id = meta.get("point_id")
            if not isinstance(point_id, str) or point_id not in expected_ids:
                raise ValueError(f"parent result {name!r} has unknown point id {point_id!r}")
            if point_id in results:
                raise ValueError(f"duplicate parent result for {point_id}")
            # reuse_parent_point performs the detailed plan/point identity checks.
            results[point_id] = (document, actual_json_sha256)

    if set(results) != expected_ids:
        missing = sorted(expected_ids - set(results))
        extra = sorted(set(results) - expected_ids)
        raise ValueError(
            "complete 1056-point parent H2 set required; "
            f"found {len(results)}, missing={missing[:5]}, extra={extra[:5]}"
        )
    return results


def write_correction_seed_set(
    plan: P4DReactionEnforcementCorrectionPlan,
    parent_results: Mapping[str, tuple[Mapping[str, object], str]],
    output_dir: str | "Path",
) -> dict[str, object]:
    """Write plan plus all 480 structurally unaffected reused envelopes.

    Existing files are accepted only when their bytes match the newly derived canonical
    bytes, making repeated preparation idempotent and fail-closed.
    """
    from pathlib import Path

    root = Path(output_dir)
    points_dir = root / "points"
    points_dir.mkdir(parents=True, exist_ok=True)
    plan_path = root / "CORRECTION_PLAN.json"
    plan_bytes = plan.to_json_bytes()
    if plan_path.exists() and plan_path.read_bytes() != plan_bytes:
        raise ValueError("existing correction plan differs from current plan identity")
    plan_path.write_bytes(plan_bytes)
    (root / "CORRECTION_PLAN.json.sha256").write_text(
        f"{plan.identity_sha256}  CORRECTION_PLAN.json\n", encoding="utf-8"
    )

    reused = 0
    for point_id in plan.reused_point_ids:
        try:
            parent_document, parent_sha256 = parent_results[point_id]
        except KeyError as exc:
            raise ValueError(f"missing parent result required for reuse: {point_id}") from exc
        corrected = reuse_parent_point(
            plan,
            parent_document,
            parent_result_sha256=parent_sha256,
        )
        raw = _canonical_json(corrected)
        path = points_dir / f"{point_id}.json"
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"existing corrected result conflicts with reused point {point_id}")
        path.write_bytes(raw)
        path.with_suffix(".json.sha256").write_text(
            f"{sha256(raw).hexdigest()}  {path.name}\n", encoding="utf-8"
        )
        reused += 1
    return {
        "correction_plan_sha256": plan.identity_sha256,
        "reused_point_count": reused,
        "rerun_point_count": len(plan.rerun_point_ids),
        "total_point_count": len(plan.parent_plan.points),
    }


def load_completed_corrected_points(
    plan: P4DReactionEnforcementCorrectionPlan,
    output_dir: str | "Path",
) -> dict[str, dict[str, object]]:
    """Load only canonical, provenance-valid completed corrected point files."""
    from pathlib import Path

    root = Path(output_dir) / "points"
    if not root.exists():
        return {}
    completed: dict[str, dict[str, object]] = {}
    for path in root.glob("P4DC-*.json"):
        raw = path.read_bytes()
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid corrected JSON: {path}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"corrected result must be an object: {path}")
        point_id = document.get("parent_point_id")
        if not isinstance(point_id, str) or path.name != f"{point_id}.json":
            raise ValueError(f"corrected result filename/point mismatch: {path}")
        point = _point_by_id(plan.parent_plan, point_id)
        if document.get("format") != CORRECTION_RESULT_FORMAT:
            raise ValueError(f"unexpected corrected format: {path}")
        if document.get("campaign_id") != plan.campaign_id:
            raise ValueError(f"corrected campaign mismatch: {path}")
        if document.get("correction_plan_sha256") != plan.identity_sha256:
            raise ValueError(f"corrected plan identity mismatch: {path}")
        if document.get("corrected_source_archive_sha256") != plan.corrected_source_archive_sha256:
            raise ValueError(f"corrected source identity mismatch: {path}")
        if document.get("parent_point_identity_sha256") != point.compute_identity_sha256():
            raise ValueError(f"corrected parent point identity mismatch: {path}")
        if document.get("correction_action") != correction_action(point):
            raise ValueError(f"corrected action mismatch: {path}")
        sidecar = path.with_suffix(".json.sha256")
        if not sidecar.exists():
            raise ValueError(f"missing result SHA sidecar: {path}")
        expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if expected != sha256(raw).hexdigest():
            raise ValueError(f"corrected result SHA mismatch: {path}")
        completed[point_id] = document
    return completed


def pending_rerun_point_ids(
    plan: P4DReactionEnforcementCorrectionPlan,
    output_dir: str | "Path",
) -> tuple[str, ...]:
    """Return affected point ids that do not yet have verified corrected artifacts."""
    completed = load_completed_corrected_points(plan, output_dir)
    return tuple(point_id for point_id in plan.rerun_point_ids if point_id not in completed)


def write_corrected_point_file(
    plan: P4DReactionEnforcementCorrectionPlan,
    document: Mapping[str, object],
    output_dir: str | "Path",
) -> str:
    """Atomically persist one corrected result and its SHA sidecar."""
    from pathlib import Path
    import os

    point_id = document.get("parent_point_id")
    if not isinstance(point_id, str):
        raise ValueError("corrected point document has no parent_point_id")
    point = _point_by_id(plan.parent_plan, point_id)
    if correction_action(point) != RERUN_ACTION:
        raise ValueError("write_corrected_point_file is reserved for rerun points")
    if document.get("correction_plan_sha256") != plan.identity_sha256:
        raise ValueError("corrected point is bound to a different correction plan")
    raw = _canonical_json(dict(document))
    root = Path(output_dir) / "points"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{point_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    digest = sha256(raw).hexdigest()
    sidecar = path.with_suffix(".json.sha256")
    tmp_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp_sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    os.replace(tmp_sidecar, sidecar)
    return digest


def finalize_corrected_campaign(
    plan: P4DReactionEnforcementCorrectionPlan,
    output_dir: str | "Path",
) -> dict[str, object]:
    """Run frozen analysis and write deterministic final artifacts for a complete overlay."""
    from pathlib import Path
    from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

    root = Path(output_dir)
    completed = load_completed_corrected_points(plan, root)
    expected_ids = {p.point_id for p in plan.parent_plan.points}
    if set(completed) != expected_ids:
        missing = sorted(expected_ids - set(completed))
        raise ValueError(
            f"cannot finalize incomplete correction campaign: {len(completed)}/1056; "
            f"first missing={missing[:5]}"
        )
    ordered_documents = [completed[p.point_id] for p in plan.parent_plan.points]
    analysis = analyze_corrected_results(plan, ordered_documents)
    analysis_raw = _canonical_json(analysis)
    analysis_path = root / "R1_P4D_REACTION_ENFORCEMENT_CORRECTED_FINAL_ANALYSIS.json"
    analysis_path.write_bytes(analysis_raw)
    analysis_sha = sha256(analysis_raw).hexdigest()
    analysis_path.with_suffix(".json.sha256").write_text(
        f"{analysis_sha}  {analysis_path.name}\n", encoding="utf-8"
    )

    manifest = {
        "format": "cross-r1-p4d-reaction-enforcement-corrected-complete-manifest-v1",
        "schema_version": 1,
        "campaign_id": plan.campaign_id,
        "correction_plan_sha256": plan.identity_sha256,
        "corrected_source_archive_sha256": plan.corrected_source_archive_sha256,
        "parent_final_results_sha256": plan.parent_final_results_sha256,
        "total_point_count": len(completed),
        "reused_point_count": len(plan.reused_point_ids),
        "rerun_point_count": len(plan.rerun_point_ids),
        "final_analysis_sha256": analysis_sha,
        "point_result_sha256": {
            point_id: sha256((root / "points" / f"{point_id}.json").read_bytes()).hexdigest()
            for point_id in sorted(completed)
        },
    }
    manifest_raw = _canonical_json(manifest)
    manifest_path = root / "COMPLETE_EXECUTION_MANIFEST.json"
    manifest_path.write_bytes(manifest_raw)
    manifest_sha = sha256(manifest_raw).hexdigest()
    manifest_path.with_suffix(".json.sha256").write_text(
        f"{manifest_sha}  {manifest_path.name}\n", encoding="utf-8"
    )

    archive_path = root / "R1_P4D_REACTION_ENFORCEMENT_CORRECTED_RESULTS_COMPLETE_V4.zip"
    members = [
        root / "CORRECTION_PLAN.json",
        root / "CORRECTION_PLAN.json.sha256",
        analysis_path,
        analysis_path.with_suffix(".json.sha256"),
        manifest_path,
        manifest_path.with_suffix(".json.sha256"),
        *sorted((root / "points").glob("*.json")),
        *sorted((root / "points").glob("*.json.sha256")),
    ]
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for path in members:
            rel = path.relative_to(root).as_posix()
            info = ZipInfo(rel, date_time=(2026, 9, 25, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    archive_sha = sha256(archive_path.read_bytes()).hexdigest()
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{archive_sha}  {archive_path.name}\n", encoding="utf-8"
    )
    return {
        "campaign_id": plan.campaign_id,
        "correction_plan_sha256": plan.identity_sha256,
        "point_count": len(completed),
        "reused_point_count": len(plan.reused_point_ids),
        "rerun_point_count": len(plan.rerun_point_ids),
        "final_analysis_sha256": analysis_sha,
        "complete_manifest_sha256": manifest_sha,
        "results_archive": str(archive_path),
        "results_archive_sha256": archive_sha,
    }
