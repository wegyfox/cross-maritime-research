"""Durable checkpoint/resume shell for A89 robustness campaign work units.

A90 is orchestration only.  It does not know how to mutate simulator physics or how to
aggregate sensitivity.  A caller supplies an identity-bearing work-unit executor.  This
layer guarantees canonical ordering, atomic per-unit result records, fail-closed resume,
and no recomputation of already validated work units.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .robustness_matrix import RobustnessCampaignPlan, RobustnessWorkUnit

ROBUSTNESS_CHECKPOINT_SCHEMA = "cross-psp-robustness-checkpoint-v1"
ROBUSTNESS_UNIT_RESULT_SCHEMA = "cross-psp-robustness-unit-result-v1"
ROBUSTNESS_FAILURE_SCHEMA = "cross-psp-robustness-unit-failure-v1"
ROBUSTNESS_EXECUTION_ALGORITHM_ID = "cross-psp-robustness-checkpoint-executor-v1"

STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_COMPLETE = "COMPLETE"
STATUS_PAUSED_RESUMABLE = "PAUSED_RESUMABLE"
STATUS_CANCELLED_RESUMABLE = "CANCELLED_RESUMABLE"
STATUS_FAILED_RESUMABLE = "FAILED_RESUMABLE"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RobustnessExecutionError(RuntimeError):
    """Raised when durable campaign execution cannot proceed safely."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RobustnessExecutionError(f"value is not canonical JSON: {exc}") from exc


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _validate_sha(name: str, value: object) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise RobustnessExecutionError(f"{name} must be a lowercase 64-character SHA-256")
    return text


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload) + b"\n"
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        try:
            fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustnessExecutionError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RobustnessExecutionError(f"{path} must contain a JSON object")
    return payload


def _unit_record_path(root: Path, unit: RobustnessWorkUnit) -> Path:
    return root / "units" / f"{unit.work_unit_index:08d}_{unit.work_unit_identity_sha256}.json"


def _failure_path(root: Path, unit: RobustnessWorkUnit) -> Path:
    failure_dir = root / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{unit.work_unit_index:08d}_{unit.work_unit_identity_sha256}_attempt_"
    attempts = []
    for path in failure_dir.glob(f"{prefix}*.json"):
        try:
            attempts.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    attempt = max(attempts, default=0) + 1
    return failure_dir / f"{prefix}{attempt:04d}.json"


def _static_identity(plan: RobustnessCampaignPlan, code_sha: str) -> dict[str, object]:
    return {
        "schema": ROBUSTNESS_CHECKPOINT_SCHEMA,
        "execution_algorithm_id": ROBUSTNESS_EXECUTION_ALGORITHM_ID,
        "matrix_algorithm_id": plan.algorithm_id,
        "campaign_identity_sha256": plan.campaign_identity_sha256,
        "baseline": plan.baseline.canonical_dict(),
        "execution_code_identity_sha256": code_sha,
        "work_unit_count": plan.work_unit_count,
    }


def _result_record(
    plan: RobustnessCampaignPlan,
    code_sha: str,
    unit: RobustnessWorkUnit,
    result: Mapping[str, Any],
) -> dict[str, object]:
    normalized_result = json.loads(_canonical_bytes(dict(result)))
    result_sha = _sha(normalized_result)
    material = {
        "schema": ROBUSTNESS_UNIT_RESULT_SCHEMA,
        "campaign_identity_sha256": plan.campaign_identity_sha256,
        "execution_code_identity_sha256": code_sha,
        "work_unit_index": unit.work_unit_index,
        "work_unit_identity_sha256": unit.work_unit_identity_sha256,
        "combination_identity_sha256": unit.combination_identity_sha256,
        "replication_index": unit.replication_index,
        "derived_seed": unit.derived_seed,
        "parameter_values": dict(unit.parameter_values),
        "result": normalized_result,
        "result_sha256": result_sha,
    }
    return {**material, "record_identity_sha256": _sha(material)}


def _validate_result_record(
    payload: Mapping[str, Any],
    *,
    plan: RobustnessCampaignPlan,
    code_sha: str,
    unit: RobustnessWorkUnit,
) -> dict[str, object]:
    if payload.get("schema") != ROBUSTNESS_UNIT_RESULT_SCHEMA:
        raise RobustnessExecutionError("unit result schema mismatch")
    checks = {
        "campaign_identity_sha256": plan.campaign_identity_sha256,
        "execution_code_identity_sha256": code_sha,
        "work_unit_index": unit.work_unit_index,
        "work_unit_identity_sha256": unit.work_unit_identity_sha256,
        "combination_identity_sha256": unit.combination_identity_sha256,
        "replication_index": unit.replication_index,
        "derived_seed": unit.derived_seed,
        "parameter_values": dict(unit.parameter_values),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RobustnessExecutionError(
                f"unit result identity mismatch for work unit {unit.work_unit_index}: {key}"
            )
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise RobustnessExecutionError(f"unit result {unit.work_unit_index} has invalid result payload")
    result_sha = _sha(dict(result))
    if payload.get("result_sha256") != result_sha:
        raise RobustnessExecutionError(f"unit result {unit.work_unit_index} result hash mismatch")
    material = {key: value for key, value in payload.items() if key != "record_identity_sha256"}
    if payload.get("record_identity_sha256") != _sha(material):
        raise RobustnessExecutionError(f"unit result {unit.work_unit_index} record hash mismatch")
    return dict(payload)


def _scan_completed(
    root: Path,
    plan: RobustnessCampaignPlan,
    code_sha: str,
) -> dict[int, dict[str, object]]:
    completed: dict[int, dict[str, object]] = {}
    units_dir = root / "units"
    if not units_dir.exists():
        return completed
    expected_by_name = {_unit_record_path(root, unit).name: unit for unit in plan.work_units}
    for path in sorted(units_dir.glob("*.json")):
        unit = expected_by_name.get(path.name)
        if unit is None:
            raise RobustnessExecutionError(f"unexpected unit result artifact {path.name}")
        payload = _validate_result_record(_read_json(path), plan=plan, code_sha=code_sha, unit=unit)
        if unit.work_unit_index in completed:
            raise RobustnessExecutionError(f"duplicate completed unit {unit.work_unit_index}")
        completed[unit.work_unit_index] = payload
    return completed


def _checkpoint_payload(
    *,
    plan: RobustnessCampaignPlan,
    code_sha: str,
    completed: Mapping[int, Mapping[str, Any]],
    status: str,
    last_failure: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    completed_rows = [
        {
            "work_unit_index": index,
            "work_unit_identity_sha256": payload["work_unit_identity_sha256"],
            "result_sha256": payload["result_sha256"],
            "record_identity_sha256": payload["record_identity_sha256"],
        }
        for index, payload in sorted(completed.items())
    ]
    remaining = [unit.work_unit_index for unit in plan.work_units if unit.work_unit_index not in completed]
    material = {
        **_static_identity(plan, code_sha),
        "status": status,
        "completed_count": len(completed_rows),
        "remaining_count": len(remaining),
        "next_work_unit_index": remaining[0] if remaining else None,
        "completed": completed_rows,
        "last_failure": dict(last_failure) if last_failure is not None else None,
    }
    return {**material, "checkpoint_identity_sha256": _sha(material)}



def _validate_checkpoint_static(
    payload: Mapping[str, Any],
    *,
    plan: RobustnessCampaignPlan,
    code_sha: str,
) -> None:
    static = _static_identity(plan, code_sha)
    for key, expected in static.items():
        if payload.get(key) != expected:
            raise RobustnessExecutionError(f"checkpoint identity mismatch: {key}")
    material = {key: value for key, value in payload.items() if key != "checkpoint_identity_sha256"}
    if payload.get("checkpoint_identity_sha256") != _sha(material):
        raise RobustnessExecutionError("checkpoint hash mismatch")

def _validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    plan: RobustnessCampaignPlan,
    code_sha: str,
    completed: Mapping[int, Mapping[str, Any]],
) -> None:
    static = _static_identity(plan, code_sha)
    for key, expected in static.items():
        if payload.get(key) != expected:
            raise RobustnessExecutionError(f"checkpoint identity mismatch: {key}")
    material = {key: value for key, value in payload.items() if key != "checkpoint_identity_sha256"}
    if payload.get("checkpoint_identity_sha256") != _sha(material):
        raise RobustnessExecutionError("checkpoint hash mismatch")
    rows = payload.get("completed")
    if not isinstance(rows, list):
        raise RobustnessExecutionError("checkpoint completed list is invalid")
    checkpoint_indices: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("work_unit_index"), int):
            raise RobustnessExecutionError("checkpoint completed row is invalid")
        index = int(row["work_unit_index"])
        checkpoint_indices.add(index)
        actual = completed.get(index)
        if actual is None:
            raise RobustnessExecutionError(
                f"checkpoint claims completed work unit {index} but durable unit record is missing"
            )
        for key in ("work_unit_identity_sha256", "result_sha256", "record_identity_sha256"):
            if row.get(key) != actual.get(key):
                raise RobustnessExecutionError(f"checkpoint completed row mismatch for unit {index}: {key}")
    # Extra durable unit records are allowed: they represent a crash after the atomic unit
    # record was committed but before the checkpoint replacement.  The next write repairs it.
    if len(checkpoint_indices) != len(rows):
        raise RobustnessExecutionError("checkpoint contains duplicate completed unit rows")


@dataclass(frozen=True, slots=True)
class RobustnessExecutionSummary:
    status: str
    campaign_identity_sha256: str
    execution_code_identity_sha256: str
    completed_count: int
    remaining_count: int
    checkpoint_path: str


def run_robustness_campaign(
    plan: RobustnessCampaignPlan,
    output_dir: str | os.PathLike[str],
    executor: Callable[[RobustnessWorkUnit], Mapping[str, Any]],
    *,
    execution_code_identity_sha256: str,
    should_cancel: Callable[[], bool] | None = None,
    max_new_units: int | None = None,
) -> RobustnessExecutionSummary:
    """Execute/resume canonical work units with atomic durable records.

    ``max_new_units`` is an operational chunking control only; it is not part of campaign
    identity.  It allows bounded runs in constrained environments without changing any
    scientific work unit or seed.
    """
    code_sha = _validate_sha("execution_code_identity_sha256", execution_code_identity_sha256)
    if max_new_units is not None:
        if isinstance(max_new_units, bool) or not isinstance(max_new_units, int) or max_new_units <= 0:
            raise RobustnessExecutionError("max_new_units must be a positive integer when provided")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "checkpoint.json"
    checkpoint_payload = None
    if checkpoint_path.exists():
        checkpoint_payload = _read_json(checkpoint_path)
        _validate_checkpoint_static(checkpoint_payload, plan=plan, code_sha=code_sha)
    completed = _scan_completed(root, plan, code_sha)
    if checkpoint_payload is not None:
        _validate_checkpoint(
            checkpoint_payload, plan=plan, code_sha=code_sha, completed=completed
        )
    else:
        unexpected = [path for path in root.iterdir() if path.name not in {"units", "failures"}]
        if unexpected:
            raise RobustnessExecutionError(
                f"output directory contains artifacts but no checkpoint: {[path.name for path in unexpected]}"
            )

    def write_checkpoint(status: str, failure: Mapping[str, Any] | None = None) -> None:
        _atomic_write_json(
            checkpoint_path,
            _checkpoint_payload(
                plan=plan,
                code_sha=code_sha,
                completed=completed,
                status=status,
                last_failure=failure,
            ),
        )

    if len(completed) == plan.work_unit_count:
        write_checkpoint(STATUS_COMPLETE)
        return RobustnessExecutionSummary(
            STATUS_COMPLETE,
            plan.campaign_identity_sha256,
            code_sha,
            len(completed),
            0,
            str(checkpoint_path),
        )

    write_checkpoint(STATUS_IN_PROGRESS)
    new_units = 0
    for unit in plan.work_units:
        if unit.work_unit_index in completed:
            continue
        if should_cancel is not None and should_cancel():
            write_checkpoint(STATUS_CANCELLED_RESUMABLE)
            return RobustnessExecutionSummary(
                STATUS_CANCELLED_RESUMABLE,
                plan.campaign_identity_sha256,
                code_sha,
                len(completed),
                plan.work_unit_count - len(completed),
                str(checkpoint_path),
            )
        if max_new_units is not None and new_units >= max_new_units:
            write_checkpoint(STATUS_PAUSED_RESUMABLE)
            return RobustnessExecutionSummary(
                STATUS_PAUSED_RESUMABLE,
                plan.campaign_identity_sha256,
                code_sha,
                len(completed),
                plan.work_unit_count - len(completed),
                str(checkpoint_path),
            )
        try:
            result = executor(unit)
            if not isinstance(result, Mapping):
                raise RobustnessExecutionError("work-unit executor must return a mapping")
            record = _result_record(plan, code_sha, unit, result)
        except KeyboardInterrupt:
            write_checkpoint(STATUS_PAUSED_RESUMABLE)
            raise
        except Exception as exc:
            failure_material = {
                "schema": ROBUSTNESS_FAILURE_SCHEMA,
                "campaign_identity_sha256": plan.campaign_identity_sha256,
                "execution_code_identity_sha256": code_sha,
                "work_unit_index": unit.work_unit_index,
                "work_unit_identity_sha256": unit.work_unit_identity_sha256,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            failure = {**failure_material, "failure_identity_sha256": _sha(failure_material)}
            _atomic_write_json(_failure_path(root, unit), failure)
            write_checkpoint(STATUS_FAILED_RESUMABLE, failure)
            if isinstance(exc, RobustnessExecutionError):
                raise
            raise RobustnessExecutionError(
                f"work unit {unit.work_unit_index} failed: {type(exc).__name__}: {exc}"
            ) from exc

        _atomic_write_json(_unit_record_path(root, unit), record)
        completed[unit.work_unit_index] = record
        new_units += 1
        write_checkpoint(STATUS_IN_PROGRESS)

    write_checkpoint(STATUS_COMPLETE)
    return RobustnessExecutionSummary(
        STATUS_COMPLETE,
        plan.campaign_identity_sha256,
        code_sha,
        len(completed),
        0,
        str(checkpoint_path),
    )
