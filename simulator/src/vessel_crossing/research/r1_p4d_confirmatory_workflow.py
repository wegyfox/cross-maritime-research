"""Deterministic workflow helpers for the frozen P4D executor lineage.

These helpers are evidence-oriented. They do not execute scientific points and
do not change simulator physics; they validate checkpoint/package identity and
make resume state explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile


class ConfirmatoryWorkflowError(RuntimeError):
    """Raised when checkpoint or resume evidence fails closed validation."""


@dataclass(frozen=True, slots=True)
class ResumeStatus:
    verified_ids: tuple[str, ...]
    pending_ids: tuple[str, ...]
    running_ids: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        return (
            f"verified {len(self.verified_ids)}, "
            f"pending {len(self.pending_ids)}, "
            f"running {len(self.running_ids)}"
        )


def sha256_file(path: str | Path) -> str:
    """Return SHA-256 for a file without loading the whole artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_deterministic_zip(
    output: str | Path,
    files: tuple[Path, ...],
    *,
    root: str | Path,
) -> str:
    """Create a byte-deterministic ZIP from explicitly selected evidence files."""
    destination = Path(output)
    root_path = Path(root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    ordered_files = sorted(
        (Path(path) for path in files),
        key=lambda path: path.relative_to(root_path).as_posix(),
    )
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in ordered_files:
            relative_path = path.relative_to(root_path).as_posix()
            info = zipfile.ZipInfo(relative_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())

    return sha256_file(destination)


def load_verified_checkpoint(
    path: str | Path,
    *,
    expected_plan_sha256: str,
    allow_overlay_id: str | None = None,
) -> dict[str, str]:
    """Load a strict checkpoint; overlay-derived state requires explicit approval."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfirmatoryWorkflowError("checkpoint must be a JSON object")

    allowed_fields = {"format", "plan_sha256", "completed", "overlay_id"}
    if (
        set(data) - allowed_fields
        or data.get("format") != "cross-confirmatory-checkpoint-v1"
    ):
        raise ConfirmatoryWorkflowError("unsupported checkpoint schema")
    if data.get("plan_sha256") != expected_plan_sha256:
        raise ConfirmatoryWorkflowError("checkpoint plan hash mismatch")

    overlay_id = data.get("overlay_id")
    if overlay_id is not None and overlay_id != allow_overlay_id:
        raise ConfirmatoryWorkflowError(
            "checkpoint overlay is not explicitly authorized"
        )

    completed = data.get("completed")
    if not isinstance(completed, dict):
        raise ConfirmatoryWorkflowError("checkpoint completed map is required")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in completed.items()
    ):
        raise ConfirmatoryWorkflowError(
            "checkpoint completed map must contain string point IDs and hashes"
        )
    return dict(completed)


def inspect_resume(
    *,
    expected_point_ids: tuple[str, ...],
    result_dir: str | Path,
    checkpoint_completed: dict[str, str],
    running_ids: tuple[str, ...] = (),
) -> ResumeStatus:
    """Verify checkpoint-bound result hashes and report pending work fail-closed."""
    expected = set(expected_point_ids)
    if len(expected) != len(expected_point_ids):
        raise ConfirmatoryWorkflowError("expected_point_ids must be unique")
    if set(checkpoint_completed) - expected:
        raise ConfirmatoryWorkflowError("checkpoint contains unknown point ids")

    verified: list[str] = []
    result_root = Path(result_dir)
    for point_id, expected_digest in sorted(checkpoint_completed.items()):
        path = result_root / f"{point_id}.json"
        if not path.is_file() or sha256_file(path) != expected_digest:
            raise ConfirmatoryWorkflowError(
                f"checkpoint-bound artifact invalid: {point_id}"
            )
        verified.append(point_id)

    pending = tuple(sorted(expected - set(verified)))
    running = tuple(sorted(set(running_ids) & set(pending)))
    return ResumeStatus(tuple(verified), pending, running)
