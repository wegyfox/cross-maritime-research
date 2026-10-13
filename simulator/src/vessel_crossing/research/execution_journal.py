"""Durable diagnostic journal for long-running research execution.

The journal is deliberately not completion authority. A checkpoint or verified
result artifact remains authoritative for completed scientific work. The journal
only records worker lifecycle and reconciliation evidence so a stopped process
can explain what was running, what failed, and what must be retried.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


class ExecutionJournalError(RuntimeError):
    """Raised when journal evidence is malformed or internally inconsistent."""


class ExecutionEventKind(str, Enum):
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    sequence: int
    point_id: str
    kind: ExecutionEventKind
    worker_id: str
    detail: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("sequence must be an integer")
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        if not self.point_id or not self.worker_id:
            raise ValueError("point_id and worker_id must be non-empty")
        if not isinstance(self.kind, ExecutionEventKind):
            raise TypeError("kind must be ExecutionEventKind")


@dataclass(frozen=True, slots=True)
class ExecutionJournalSnapshot:
    session_id: str
    events: tuple[ExecutionEvent, ...]

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")

    def _latest_states(self) -> dict[str, ExecutionEventKind]:
        states: dict[str, ExecutionEventKind] = {}
        for event in self.events:
            states[event.point_id] = event.kind
        return states

    @property
    def running_point_ids(self) -> tuple[str, ...]:
        states = self._latest_states()
        return tuple(
            sorted(
                point_id
                for point_id, state in states.items()
                if state is ExecutionEventKind.STARTED
            )
        )

    @property
    def failed_point_ids(self) -> tuple[str, ...]:
        states = self._latest_states()
        return tuple(
            sorted(
                point_id
                for point_id, state in states.items()
                if state is ExecutionEventKind.FAILED
            )
        )


def _canonical_bytes(payload: object) -> bytes:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return (rendered + "\n").encode("utf-8")


def _payload(snapshot: ExecutionJournalSnapshot) -> dict[str, object]:
    return {
        "format": "cross-execution-journal-v1",
        "session_id": snapshot.session_id,
        "events": [
            {
                "sequence": event.sequence,
                "point_id": event.point_id,
                "kind": event.kind.value,
                "worker_id": event.worker_id,
                "detail": event.detail,
            }
            for event in snapshot.events
        ],
    }


def snapshot_sha256(snapshot: ExecutionJournalSnapshot) -> str:
    """Return the canonical SHA-256 identity of a journal snapshot."""
    return hashlib.sha256(_canonical_bytes(_payload(snapshot))).hexdigest()


def write_journal(path: str | Path, snapshot: ExecutionJournalSnapshot) -> None:
    """Atomically replace the journal and its SHA-256 sidecar."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    raw = _canonical_bytes(_payload(snapshot))
    digest = hashlib.sha256(raw).hexdigest()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary_hash = destination.with_suffix(destination.suffix + ".sha256.tmp")

    temporary.write_bytes(raw)
    temporary_hash.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary, destination)
    os.replace(
        temporary_hash,
        destination.with_suffix(destination.suffix + ".sha256"),
    )


def load_journal(path: str | Path) -> ExecutionJournalSnapshot:
    """Load a journal only when its hash sidecar and strict schema are valid."""
    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if not source.is_file() or not sidecar.is_file():
        raise ExecutionJournalError("journal and hash sidecar are both required")

    raw = source.read_bytes()
    expected = sidecar.read_text(encoding="ascii").strip()
    actual = hashlib.sha256(raw).hexdigest()
    if expected != actual:
        raise ExecutionJournalError("journal hash mismatch")

    data = json.loads(raw)
    if (
        not isinstance(data, dict)
        or set(data) != {"format", "session_id", "events"}
        or data.get("format") != "cross-execution-journal-v1"
    ):
        raise ExecutionJournalError("unsupported journal schema")
    if not isinstance(data["events"], list):
        raise ExecutionJournalError("journal events must be a list")

    events: list[ExecutionEvent] = []
    for item in data["events"]:
        if not isinstance(item, dict) or set(item) != {
            "sequence",
            "point_id",
            "kind",
            "worker_id",
            "detail",
        }:
            raise ExecutionJournalError("unsupported event schema")
        try:
            kind = ExecutionEventKind(item["kind"])
        except (TypeError, ValueError) as exc:
            raise ExecutionJournalError("unknown execution event kind") from exc
        try:
            event = ExecutionEvent(
                sequence=item["sequence"],
                point_id=item["point_id"],
                kind=kind,
                worker_id=item["worker_id"],
                detail=item["detail"],
            )
        except (TypeError, ValueError) as exc:
            raise ExecutionJournalError("invalid execution event") from exc
        events.append(event)

    if [event.sequence for event in events] != list(range(1, len(events) + 1)):
        raise ExecutionJournalError("journal event sequence is not contiguous")

    try:
        return ExecutionJournalSnapshot(data["session_id"], tuple(events))
    except ValueError as exc:
        raise ExecutionJournalError("invalid journal header") from exc


def append_event(
    path: str | Path,
    *,
    session_id: str,
    point_id: str,
    kind: ExecutionEventKind,
    worker_id: str,
    detail: str = "",
) -> ExecutionJournalSnapshot:
    """Append one diagnostic event with an atomic full-file commit."""
    destination = Path(path)
    if destination.exists():
        current = load_journal(destination)
        if current.session_id != session_id:
            raise ExecutionJournalError("journal session_id mismatch")
    else:
        current = ExecutionJournalSnapshot(session_id, ())

    event = ExecutionEvent(
        sequence=len(current.events) + 1,
        point_id=point_id,
        kind=kind,
        worker_id=worker_id,
        detail=detail,
    )
    updated = ExecutionJournalSnapshot(
        session_id=session_id,
        events=current.events + (event,),
    )
    write_journal(destination, updated)
    return updated


def reconcile_with_authoritative_completion(
    snapshot: ExecutionJournalSnapshot,
    completed_point_ids: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return verified-completed IDs and retry IDs without promoting journal claims.

    A point is completed only because the caller supplied it as authoritative.
    Any point left STARTED/FAILED in the journal but absent from that set remains
    retryable.
    """
    completed = set(completed_point_ids)
    touched = {event.point_id for event in snapshot.events}
    verified = tuple(sorted(touched & completed))
    retry = tuple(
        sorted(
            (set(snapshot.running_point_ids) | set(snapshot.failed_point_ids))
            - completed
        )
    )
    return verified, retry
