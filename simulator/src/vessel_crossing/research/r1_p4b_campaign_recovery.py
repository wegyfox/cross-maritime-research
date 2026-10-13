"""Fail-closed durable recovery for the R1 P4B publication campaign.

This module is orchestration-only.  It never runs a scientific point and never
promotes a journal event to scientific completion.  Completion authority is an
exact result JSON + SHA-256 sidecar pair bound to the frozen execution plan.

The recovery contract deliberately distinguishes:

* committed result pair -- may be recovered into a checkpoint;
* incomplete/uncommitted debris -- must be retried, never promoted;
* apparently committed but inconsistent evidence -- fail closed.

That distinction makes process termination around durable-write boundaries
safe without changing any numerical/scientific semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from .r1_p4b_publication_execution import (
    P4B_CHECKPOINT_FORMAT,
    P4B_POINT_FORMAT,
    P4B_POINT_SCHEMA_VERSION,
    P4BPublicationExecutionPlan,
    P4BPhysicalPoint,
    initial_checkpoint,
    load_checkpoint,
    result_file_name,
    write_checkpoint_atomic,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class P4BCampaignRecoveryError(RuntimeError):
    """Raised when durable campaign evidence is inconsistent or ambiguous."""


@dataclass(frozen=True, slots=True)
class P4BCampaignRecoveryReport:
    """Non-scientific restart/reconciliation evidence."""

    checkpoint: dict[str, object]
    verified_checkpoint_point_ids: tuple[str, ...]
    recovered_uncheckpointed_point_ids: tuple[str, ...]
    retry_point_ids: tuple[str, ...]
    incomplete_artifact_point_ids: tuple[str, ...]
    stale_temporary_paths: tuple[str, ...]
    checkpoint_rewritten: bool


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _expected_point_document(point: P4BPhysicalPoint) -> dict[str, object]:
    # round-trip converts tuple-valued dataclass fields to the list representation
    # used by canonical JSON output.
    return json.loads(json.dumps(asdict(point), sort_keys=True))


def _read_sidecar(sidecar: Path, *, expected_file_name: str) -> str:
    try:
        text = sidecar.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise P4BCampaignRecoveryError(f"cannot read result SHA sidecar: {sidecar}") from exc
    parts = text.split()
    if len(parts) != 2 or not _SHA256_RE.fullmatch(parts[0]) or parts[1] != expected_file_name:
        raise P4BCampaignRecoveryError(f"malformed result SHA sidecar: {sidecar}")
    return parts[0]


def verify_committed_result(
    results_dir: Path,
    plan: P4BPublicationExecutionPlan,
    point: P4BPhysicalPoint,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Verify one committed result pair and its exact plan/point identity.

    Both the JSON and sidecar are required.  This function never repairs or
    guesses evidence.
    """
    result = results_dir / result_file_name(point)
    sidecar = Path(str(result) + ".sha256")
    if not result.is_file() or not sidecar.is_file():
        raise P4BCampaignRecoveryError(
            f"committed result pair incomplete for {point.point_id}"
        )
    try:
        raw = result.read_bytes()
    except OSError as exc:
        raise P4BCampaignRecoveryError(f"cannot read result: {result}") from exc
    actual = _sha256(raw)
    sidecar_sha = _read_sidecar(sidecar, expected_file_name=result.name)
    if actual != sidecar_sha:
        raise P4BCampaignRecoveryError(f"result SHA mismatch for {point.point_id}")
    if expected_sha256 is not None:
        if not _SHA256_RE.fullmatch(expected_sha256):
            raise P4BCampaignRecoveryError(
                f"checkpoint result SHA malformed for {point.point_id}"
            )
        if actual != expected_sha256:
            raise P4BCampaignRecoveryError(
                f"checkpoint/result SHA mismatch for {point.point_id}"
            )
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P4BCampaignRecoveryError(
            f"result JSON is truncated or malformed for {point.point_id}"
        ) from exc
    if not isinstance(document, dict):
        raise P4BCampaignRecoveryError(f"result JSON must be an object for {point.point_id}")
    if document.get("format") != P4B_POINT_FORMAT or document.get("schema_version") != P4B_POINT_SCHEMA_VERSION:
        raise P4BCampaignRecoveryError(f"result schema mismatch for {point.point_id}")
    if document.get("campaign_id") != plan.campaign_id:
        raise P4BCampaignRecoveryError(f"result campaign mismatch for {point.point_id}")
    if document.get("protocol_id") != plan.protocol_id:
        raise P4BCampaignRecoveryError(f"result protocol mismatch for {point.point_id}")
    if document.get("plan_sha256") != plan.identity_sha256:
        raise P4BCampaignRecoveryError(f"result plan mismatch for {point.point_id}")
    if document.get("source_archive_sha256") != plan.source_archive_sha256:
        raise P4BCampaignRecoveryError(f"result source mismatch for {point.point_id}")
    if document.get("point") != _expected_point_document(point):
        raise P4BCampaignRecoveryError(f"result point identity mismatch for {point.point_id}")
    return actual


def _validate_checkpoint_shape(
    checkpoint: Mapping[str, object],
    plan: P4BPublicationExecutionPlan,
) -> dict[str, dict[str, object]]:
    required = {
        "format",
        "schema_version",
        "campaign_id",
        "plan_sha256",
        "source_archive_sha256",
        "completed",
        "last_durable_execution_order",
        "status",
    }
    if set(checkpoint) != required:
        raise P4BCampaignRecoveryError("checkpoint schema mismatch")
    if checkpoint.get("format") != P4B_CHECKPOINT_FORMAT or checkpoint.get("schema_version") != 1:
        raise P4BCampaignRecoveryError("checkpoint format/version mismatch")
    completed_obj = checkpoint.get("completed")
    if not isinstance(completed_obj, dict):
        raise P4BCampaignRecoveryError("checkpoint completed must be an object")
    points = {point.point_id: point for point in plan.physical_points}
    completed: dict[str, dict[str, object]] = {}
    seen_files: set[str] = set()
    for point_id, entry_obj in completed_obj.items():
        if point_id not in points:
            raise P4BCampaignRecoveryError(f"checkpoint references unknown point {point_id!r}")
        if not isinstance(entry_obj, dict) or set(entry_obj) != {
            "execution_order",
            "result_sha256",
            "file_name",
        }:
            raise P4BCampaignRecoveryError(f"checkpoint entry schema mismatch for {point_id}")
        point = points[point_id]
        if entry_obj.get("execution_order") != point.execution_order:
            raise P4BCampaignRecoveryError(f"checkpoint execution_order mismatch for {point_id}")
        expected_name = result_file_name(point)
        if entry_obj.get("file_name") != expected_name:
            raise P4BCampaignRecoveryError(f"checkpoint file_name mismatch for {point_id}")
        result_sha = entry_obj.get("result_sha256")
        if not isinstance(result_sha, str) or not _SHA256_RE.fullmatch(result_sha):
            raise P4BCampaignRecoveryError(f"checkpoint result SHA malformed for {point_id}")
        if expected_name in seen_files:
            raise P4BCampaignRecoveryError("checkpoint contains duplicate result file names")
        seen_files.add(expected_name)
        completed[point_id] = dict(entry_obj)
    last = checkpoint.get("last_durable_execution_order")
    if isinstance(last, bool) or not isinstance(last, int) or last < 0:
        raise P4BCampaignRecoveryError("checkpoint last_durable_execution_order is invalid")
    expected_last = max((entry["execution_order"] for entry in completed.values()), default=0)
    if last != expected_last:
        raise P4BCampaignRecoveryError("checkpoint last_durable_execution_order mismatch")
    status = checkpoint.get("status")
    if status not in {"FROZEN_BEFORE_EXECUTION", "RUNNING", "COMPLETE"}:
        raise P4BCampaignRecoveryError("checkpoint status is invalid")
    if status == "COMPLETE" and len(completed) != len(plan.physical_points):
        raise P4BCampaignRecoveryError("checkpoint claims COMPLETE with missing results")
    return completed


def _planned_artifact_maps(plan: P4BPublicationExecutionPlan) -> tuple[dict[str, P4BPhysicalPoint], dict[str, P4BPhysicalPoint]]:
    by_id = {point.point_id: point for point in plan.physical_points}
    by_name = {result_file_name(point): point for point in plan.physical_points}
    return by_id, by_name


def _classify_directory_artifacts(
    results_dir: Path,
    plan: P4BPublicationExecutionPlan,
) -> tuple[set[str], set[str], tuple[str, ...]]:
    """Return committed-candidate IDs, incomplete IDs and stale temp paths.

    Unknown JSON/result-sidecar names fail closed because they make duplicate or
    cross-campaign evidence ambiguous.  `.tmp` debris is explicitly non-authority.
    """
    _by_id, by_name = _planned_artifact_maps(plan)
    committed_candidates: set[str] = set()
    incomplete: set[str] = set()
    stale_tmp: list[str] = []
    if not results_dir.exists():
        return committed_candidates, incomplete, ()
    try:
        children = tuple(results_dir.iterdir())
    except OSError as exc:
        raise P4BCampaignRecoveryError(f"cannot inspect results directory: {results_dir}") from exc
    for child in children:
        if child.is_dir():
            raise P4BCampaignRecoveryError(f"unexpected directory in results directory: {child.name}")
        name = child.name
        if name.endswith(".tmp"):
            stale_tmp.append(name)
            continue
        if name.endswith(".json.sha256"):
            base = name[:-7]  # remove '.sha256', keep '.json'
            point = by_name.get(base)
            if point is None:
                raise P4BCampaignRecoveryError(f"unexpected result sidecar: {name}")
            result = results_dir / base
            if result.is_file():
                committed_candidates.add(point.point_id)
            else:
                incomplete.add(point.point_id)
            continue
        if name.endswith(".json"):
            point = by_name.get(name)
            if point is None:
                # Read enough to provide a specific duplicate-identity failure when possible.
                try:
                    doc = json.loads(child.read_bytes())
                except Exception:
                    raise P4BCampaignRecoveryError(f"unexpected result JSON artifact: {name}")
                embedded = doc.get("point", {}).get("point_id") if isinstance(doc, dict) else None
                if embedded in _by_id:
                    raise P4BCampaignRecoveryError(
                        f"duplicate result identity for {embedded}: unexpected file {name}"
                    )
                raise P4BCampaignRecoveryError(f"unexpected result JSON artifact: {name}")
            sidecar = Path(str(child) + ".sha256")
            if sidecar.is_file():
                committed_candidates.add(point.point_id)
            else:
                incomplete.add(point.point_id)
            continue
        raise P4BCampaignRecoveryError(f"unexpected artifact in results directory: {name}")
    return committed_candidates, incomplete, tuple(sorted(stale_tmp))


def reconcile_publication_campaign(
    *,
    plan: P4BPublicationExecutionPlan,
    results_dir: Path,
    checkpoint_path: Path,
) -> P4BCampaignRecoveryReport:
    """Reconcile restart state without inventing a scientific completion.

    Valid result+sidecar pairs written before a process crash may be promoted
    into the checkpoint after exact identity verification.  Incomplete debris
    remains retryable.  Corrupt or ambiguous committed evidence fails closed.
    """
    checkpoint = load_checkpoint(checkpoint_path, plan)
    completed = _validate_checkpoint_shape(checkpoint, plan)
    by_id, _by_name = _planned_artifact_maps(plan)

    verified_checkpoint: list[str] = []
    for point_id, entry in sorted(
        completed.items(), key=lambda item: int(item[1]["execution_order"])
    ):
        verify_committed_result(
            results_dir,
            plan,
            by_id[point_id],
            expected_sha256=str(entry["result_sha256"]),
        )
        verified_checkpoint.append(point_id)

    committed_candidates, incomplete, stale_tmp = _classify_directory_artifacts(
        results_dir, plan
    )
    recovered: list[str] = []
    for point_id in sorted(
        committed_candidates - set(completed), key=lambda pid: by_id[pid].execution_order
    ):
        point = by_id[point_id]
        digest = verify_committed_result(results_dir, plan, point)
        completed[point_id] = {
            "execution_order": point.execution_order,
            "result_sha256": digest,
            "file_name": result_file_name(point),
        }
        recovered.append(point_id)

    all_ids = set(by_id)
    completed_ids = set(completed)
    retry = tuple(
        point.point_id
        for point in plan.physical_points
        if point.point_id not in completed_ids
    )
    incomplete_ids = tuple(
        point.point_id
        for point in plan.physical_points
        if point.point_id in incomplete and point.point_id not in completed_ids
    )

    expected_last = max(
        (int(entry["execution_order"]) for entry in completed.values()), default=0
    )
    expected_status = (
        "COMPLETE" if len(completed) == len(all_ids)
        else ("RUNNING" if completed else "FROZEN_BEFORE_EXECUTION")
    )
    reconciled = {
        "format": P4B_CHECKPOINT_FORMAT,
        "schema_version": 1,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.identity_sha256,
        "source_archive_sha256": plan.source_archive_sha256,
        "completed": completed,
        "last_durable_execution_order": expected_last,
        "status": expected_status,
    }
    rewritten = dict(checkpoint) != reconciled
    if rewritten:
        write_checkpoint_atomic(checkpoint_path, reconciled)

    return P4BCampaignRecoveryReport(
        checkpoint=reconciled,
        verified_checkpoint_point_ids=tuple(verified_checkpoint),
        recovered_uncheckpointed_point_ids=tuple(recovered),
        retry_point_ids=retry,
        incomplete_artifact_point_ids=incomplete_ids,
        stale_temporary_paths=stale_tmp,
        checkpoint_rewritten=rewritten,
    )
