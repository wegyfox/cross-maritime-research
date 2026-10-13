"""R1 P4B campaign execution, checkpointing, authorization, and parallel orchestration.

This is orchestration only.  The scientific one-point runner is isolated in
`r1_p4b_physical_runner`; immutable planning is isolated in
`r1_p4b_plan_materialization`.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from typing import Mapping

from vessel_crossing.research.execution_journal import (
    ExecutionEventKind,
    append_event,
    load_journal,
    reconcile_with_authoritative_completion,
)
from vessel_crossing.research.r1_p4b_experiment_spec import (
    P4B_AUTH_FORMAT,
    P4B_AUTH_SCHEMA_VERSION,
    P4B_CHECKPOINT_FORMAT,
    P4B_MAX_PARALLEL_SCIENTIFIC_WORKERS,
    _canonical_json,
    _sha_bytes,
)
from vessel_crossing.research.r1_p4b_physical_runner import run_publication_physical_point
from vessel_crossing.research.r1_p4b_plan_materialization import (
    P4BPublicationExecutionPlan,
    plan_from_json_bytes,
)
from vessel_crossing.research.r1_p4b_result_schema import (
    result_file_name,
    write_result_atomically,
    _atomic_write,
)

def initial_checkpoint(plan: P4BPublicationExecutionPlan) -> dict[str, object]:
    return {
        "format": P4B_CHECKPOINT_FORMAT,
        "schema_version": 1,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.identity_sha256,
        "source_archive_sha256": plan.source_archive_sha256,
        "completed": {},
        "last_durable_execution_order": 0,
        "status": "FROZEN_BEFORE_EXECUTION",
    }

def write_checkpoint_atomic(path: Path, checkpoint: Mapping[str, object]) -> str:
    payload = _canonical_json(dict(checkpoint))
    digest = _sha_bytes(payload)
    _atomic_write(path, payload)
    _atomic_write(Path(str(path) + ".sha256"), f"{digest}  {path.name}\n".encode("utf-8"))
    if _sha_bytes(path.read_bytes()) != digest:
        raise RuntimeError("checkpoint raw readback SHA mismatch")
    return digest

def load_checkpoint(path: Path, plan: P4BPublicationExecutionPlan) -> dict[str, object]:
    if not path.exists():
        return initial_checkpoint(plan)
    payload = path.read_bytes()
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.exists():
        raise RuntimeError("checkpoint SHA sidecar missing")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if _sha_bytes(payload) != expected:
        raise RuntimeError("checkpoint SHA mismatch")
    doc = json.loads(payload)
    if doc.get("campaign_id") != plan.campaign_id or doc.get("plan_sha256") != plan.identity_sha256:
        raise RuntimeError("checkpoint plan identity mismatch")
    if doc.get("source_archive_sha256") != plan.source_archive_sha256:
        raise RuntimeError("checkpoint source identity mismatch")
    return doc

def verify_execution_authorization(
    authorization: Mapping[str, object] | None,
    plan: P4BPublicationExecutionPlan,
) -> None:
    if authorization is None:
        raise PermissionError("scientific execution requires an external authorization file")
    if authorization.get("format") != P4B_AUTH_FORMAT or authorization.get("schema_version") != P4B_AUTH_SCHEMA_VERSION:
        raise PermissionError("invalid P4B authorization schema")
    if authorization.get("authorized") is not True:
        raise PermissionError("P4B scientific execution is not authorized")
    if authorization.get("campaign_id") != plan.campaign_id:
        raise PermissionError("authorization campaign mismatch")
    if authorization.get("plan_sha256") != plan.identity_sha256:
        raise PermissionError("authorization plan mismatch")
    if authorization.get("source_archive_sha256") != plan.source_archive_sha256:
        raise PermissionError("authorization source mismatch")
    if not authorization.get("authorization_id"):
        raise PermissionError("authorization_id is required")


def _worker(plan_bytes: bytes, point_index: int) -> tuple[int, bytes]:
    plan = plan_from_json_bytes(plan_bytes)
    point = plan.physical_points[point_index]
    return point_index, run_publication_physical_point(plan, point)


def execute_publication_campaign(
    plan_path: Path,
    results_dir: Path,
    checkpoint_path: Path,
    authorization_path: Path,
    journal_path: Path | None = None,
) -> None:
    """Execute/resume the frozen campaign with at most two scientific workers."""
    # Lazy import avoids a module cycle: recovery depends on checkpoint/result contracts.
    from vessel_crossing.research.r1_p4b_campaign_recovery import reconcile_publication_campaign

    plan_bytes = plan_path.read_bytes()
    plan = plan_from_json_bytes(plan_bytes)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    verify_execution_authorization(authorization, plan)

    recovery = reconcile_publication_campaign(
        plan=plan, results_dir=results_dir, checkpoint_path=checkpoint_path
    )
    checkpoint = recovery.checkpoint
    completed = dict(checkpoint.get("completed", {}))

    journal = journal_path or checkpoint_path.with_name(checkpoint_path.name + ".journal.json")
    session_id = f"{plan.campaign_id}:{plan.identity_sha256}"
    if journal.exists():
        snapshot = load_journal(journal)
        if snapshot.session_id != session_id:
            raise RuntimeError("execution journal session identity mismatch")
        verified, _retry = reconcile_with_authoritative_completion(snapshot, completed.keys())
        stale = set(snapshot.running_point_ids) | set(snapshot.failed_point_ids)
        for point_id in sorted(stale & set(verified)):
            append_event(
                journal,
                session_id=session_id,
                point_id=point_id,
                kind=ExecutionEventKind.COMPLETED,
                worker_id=f"recovery-pid-{os.getpid()}",
                detail="RECOVERED_FROM_DURABLE_AUTHORITY",
            )

    pending = [i for i, point in enumerate(plan.physical_points) if point.point_id not in completed]
    worker_id = f"parent-pid-{os.getpid()}"
    with ProcessPoolExecutor(max_workers=P4B_MAX_PARALLEL_SCIENTIFIC_WORKERS) as pool:
        for offset in range(0, len(pending), P4B_MAX_PARALLEL_SCIENTIFIC_WORKERS):
            batch = pending[offset:offset + P4B_MAX_PARALLEL_SCIENTIFIC_WORKERS]
            futures = []
            for index in batch:
                point = plan.physical_points[index]
                append_event(
                    journal,
                    session_id=session_id,
                    point_id=point.point_id,
                    kind=ExecutionEventKind.STARTED,
                    worker_id=worker_id,
                )
                futures.append((index, pool.submit(_worker, plan_bytes, index)))

            batch_results = []
            for index, future in futures:
                point = plan.physical_points[index]
                try:
                    batch_results.append(future.result())
                except Exception as exc:
                    append_event(
                        journal,
                        session_id=session_id,
                        point_id=point.point_id,
                        kind=ExecutionEventKind.FAILED,
                        worker_id=worker_id,
                        detail=type(exc).__name__,
                    )
                    raise

            try:
                for index, payload in sorted(batch_results):
                    point = plan.physical_points[index]
                    _path, digest = write_result_atomically(results_dir, point, payload)
                    completed[point.point_id] = {
                        "execution_order": point.execution_order,
                        "result_sha256": digest,
                        "file_name": result_file_name(point),
                    }
                last = max(entry["execution_order"] for entry in completed.values())
                checkpoint = {
                    "format": P4B_CHECKPOINT_FORMAT,
                    "schema_version": 1,
                    "campaign_id": plan.campaign_id,
                    "plan_sha256": plan.identity_sha256,
                    "source_archive_sha256": plan.source_archive_sha256,
                    "completed": completed,
                    "last_durable_execution_order": last,
                    "status": "RUNNING" if len(completed) < len(plan.physical_points) else "COMPLETE",
                }
                write_checkpoint_atomic(checkpoint_path, checkpoint)
            except Exception as exc:
                for index in batch:
                    point = plan.physical_points[index]
                    if point.point_id not in completed or point.point_id not in checkpoint.get("completed", {}):
                        append_event(
                            journal,
                            session_id=session_id,
                            point_id=point.point_id,
                            kind=ExecutionEventKind.FAILED,
                            worker_id=worker_id,
                            detail=type(exc).__name__,
                        )
                raise

            for index in batch:
                point = plan.physical_points[index]
                append_event(
                    journal,
                    session_id=session_id,
                    point_id=point.point_id,
                    kind=ExecutionEventKind.COMPLETED,
                    worker_id=worker_id,
                )
