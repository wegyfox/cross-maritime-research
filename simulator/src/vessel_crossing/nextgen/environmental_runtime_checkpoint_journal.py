"""Crash-safe periodic journal for A129 persistent environmental runtime checkpoints.

A130 adds *publication policy* around the exact committed-state checkpoint primitive from
A129.  It does not add another simulator clock, fleet observer, lifecycle writer, or owner.
A periodic journal boundary is merely an ordinary ``run_until`` caller horizon.  Once the
underlying A123 runtime has reached that quiescent horizon, A129 captures the exact object
graph and A130 publishes it with a two-phase local filesystem protocol:

1. fsync + atomic replace the immutable checkpoint file;
2. fsync + atomic replace ``journal.json`` making that checkpoint authoritative;
3. only then delete files retired by retention.

A crash after (1) but before (2) therefore leaves an unlisted orphan which resume ignores.
If the newest listed checkpoint is corrupt, resume walks backward to the newest valid entry,
atomically heals the manifest, and preserves the sequence high-water mark so an abandoned
or corrupt sequence id is never reused.

The journal is a trusted local artifact, exactly like A129 checkpoints.  It inherits A129's
same-version/Python/cloudpickle compatibility contract and pickle security model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Any

from ..version import __version__
from .environmental_idle_fleet_control import (
    EnvironmentalPersistentFleetControlResult,
    EnvironmentalPersistentFleetControlSession,
)
from .environmental_runtime_checkpoint import (
    EnvironmentalFleetRuntimeCheckpoint,
    EnvironmentalFleetRuntimeCheckpointError,
    capture_environmental_fleet_runtime_checkpoint,
    restore_environmental_fleet_runtime_checkpoint,
)

_EPS = 1e-10
_SCHEMA_VERSION = 1
_MANIFEST_NAME = "journal.json"
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d{16})\.vcchk$")


class EnvironmentalFleetRuntimeCheckpointJournalError(RuntimeError):
    """Base error for A130 journal creation/publication/resume failures."""


class EnvironmentalFleetRuntimeCheckpointJournalExists(EnvironmentalFleetRuntimeCheckpointJournalError):
    """A journal already exists at the requested directory."""


class EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(EnvironmentalFleetRuntimeCheckpointJournalError):
    """The authoritative journal manifest or all retained checkpoints are unusable."""


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeCheckpointJournalConfig:
    """Periodic publication and retention policy."""

    interval_min: float = 1.0
    retain_last: int = 4

    def __post_init__(self) -> None:
        if not isfinite(self.interval_min) or self.interval_min <= 0.0:
            raise ValueError("interval_min must be finite and positive")
        if isinstance(self.retain_last, bool) or not isinstance(self.retain_last, int):
            raise TypeError("retain_last must be an integer")
        if self.retain_last < 1:
            raise ValueError("retain_last must be at least 1")


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeCheckpointJournalEntry:
    """One immutable checkpoint file named by a never-reused sequence id."""

    sequence: int
    file_name: str
    world_time: float
    file_sha256: str
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeCheckpointJournalManifest:
    """Authoritative ordered journal index."""

    schema_version: int
    runtime_version: str
    interval_min: float
    retain_last: int
    next_sequence: int
    entries: tuple[EnvironmentalFleetRuntimeCheckpointJournalEntry, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeCheckpointJournalRecovery:
    """Evidence returned when resume had to roll back corrupt newest entries."""

    restored_sequence: int
    discarded_sequences: tuple[int, ...]


class EnvironmentalFleetRuntimeCheckpointJournal:
    """Run one persistent A123 runtime while periodically publishing A129 checkpoints."""

    def __init__(
        self,
        directory: Path,
        session: EnvironmentalPersistentFleetControlSession,
        manifest: EnvironmentalFleetRuntimeCheckpointJournalManifest,
        *,
        recovery: EnvironmentalFleetRuntimeCheckpointJournalRecovery | None = None,
    ) -> None:
        self.directory = directory
        self.session = session
        self.manifest = manifest
        self.config = EnvironmentalFleetRuntimeCheckpointJournalConfig(
            interval_min=manifest.interval_min,
            retain_last=manifest.retain_last,
        )
        self.recovery = recovery
        self._next_checkpoint_time = self._derive_next_checkpoint_time()

    @property
    def world(self):
        return self.session.world

    @property
    def entries(self) -> tuple[EnvironmentalFleetRuntimeCheckpointJournalEntry, ...]:
        return self.manifest.entries

    @property
    def next_checkpoint_time(self) -> float:
        return self._next_checkpoint_time

    def _derive_next_checkpoint_time(self) -> float:
        if not self.manifest.entries:
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                "journal manifest has no committed checkpoints"
            )
        return self.manifest.entries[-1].world_time + self.config.interval_min

    @classmethod
    def create(
        cls,
        directory: str | Path,
        session: EnvironmentalPersistentFleetControlSession,
        *,
        config: EnvironmentalFleetRuntimeCheckpointJournalConfig = EnvironmentalFleetRuntimeCheckpointJournalConfig(),
    ) -> "EnvironmentalFleetRuntimeCheckpointJournal":
        if not isinstance(session, EnvironmentalPersistentFleetControlSession):
            raise TypeError("session must be EnvironmentalPersistentFleetControlSession")
        if not isinstance(config, EnvironmentalFleetRuntimeCheckpointJournalConfig):
            raise TypeError("config must be EnvironmentalFleetRuntimeCheckpointJournalConfig")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / _MANIFEST_NAME
        existing_checkpoints = tuple(root.glob("checkpoint-*.vcchk"))
        existing_temps = tuple(root.glob("*.tmp"))
        if manifest_path.exists() or existing_checkpoints or existing_temps:
            raise EnvironmentalFleetRuntimeCheckpointJournalExists(
                f"journal directory {root} already contains checkpoint-journal state"
            )
        base = EnvironmentalFleetRuntimeCheckpointJournalManifest(
            schema_version=_SCHEMA_VERSION,
            runtime_version=__version__,
            interval_min=float(config.interval_min),
            retain_last=config.retain_last,
            next_sequence=1,
            entries=(),
        )
        journal = cls.__new__(cls)
        journal.directory = root
        journal.session = session
        journal.manifest = base
        journal.config = config
        journal.recovery = None
        journal._next_checkpoint_time = session.world.time
        journal._publish_checkpoint()
        journal._next_checkpoint_time = journal.manifest.entries[-1].world_time + config.interval_min
        return journal

    @classmethod
    def resume(cls, directory: str | Path) -> "EnvironmentalFleetRuntimeCheckpointJournal":
        root = Path(directory)
        manifest = _read_manifest(root / _MANIFEST_NAME)
        if manifest.schema_version != _SCHEMA_VERSION:
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                f"unsupported journal schema {manifest.schema_version}; expected {_SCHEMA_VERSION}"
            )
        if manifest.runtime_version != __version__:
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                f"journal runtime version {manifest.runtime_version!r} != current {__version__!r}"
            )
        if not manifest.entries:
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                "journal manifest has no committed checkpoints"
            )

        # Never reuse a sequence that has already appeared on disk, even if it was an
        # unlisted orphan from a crash or is about to be discarded as corrupt.
        disk_high_water = _checkpoint_sequence_high_water(root)
        next_sequence = max(manifest.next_sequence, disk_high_water + 1)

        restored: EnvironmentalPersistentFleetControlSession | None = None
        restored_index: int | None = None
        for index in range(len(manifest.entries) - 1, -1, -1):
            entry = manifest.entries[index]
            try:
                checkpoint = _read_listed_checkpoint(root, entry)
                restored = restore_environmental_fleet_runtime_checkpoint(checkpoint)
            except (
                OSError,
                EnvironmentalFleetRuntimeCheckpointError,
                EnvironmentalFleetRuntimeCheckpointJournalIntegrityError,
                ValueError,
                TypeError,
            ):
                continue
            restored_index = index
            break
        if restored is None or restored_index is None:
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                "no valid retained checkpoint can be restored"
            )

        discarded = tuple(item.sequence for item in manifest.entries[restored_index + 1 :])
        healed_entries = manifest.entries[: restored_index + 1]
        healed = EnvironmentalFleetRuntimeCheckpointJournalManifest(
            schema_version=manifest.schema_version,
            runtime_version=manifest.runtime_version,
            interval_min=manifest.interval_min,
            retain_last=manifest.retain_last,
            next_sequence=next_sequence,
            entries=healed_entries,
        )
        recovery = None
        if discarded or next_sequence != manifest.next_sequence:
            _atomic_write_json(root / _MANIFEST_NAME, _manifest_to_dict(healed))
            recovery = EnvironmentalFleetRuntimeCheckpointJournalRecovery(
                restored_sequence=healed_entries[-1].sequence,
                discarded_sequences=discarded,
            )
        return cls(root, restored, healed, recovery=recovery)

    def _publish_checkpoint(self) -> EnvironmentalFleetRuntimeCheckpointJournalEntry:
        checkpoint = capture_environmental_fleet_runtime_checkpoint(self.session)
        raw = checkpoint.to_bytes()
        sequence = self.manifest.next_sequence
        file_name = f"checkpoint-{sequence:016d}.vcchk"
        destination = self.directory / file_name
        if destination.exists():
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                f"checkpoint sequence {sequence} already exists on disk"
            )
        _atomic_write_bytes(destination, raw)
        entry = EnvironmentalFleetRuntimeCheckpointJournalEntry(
            sequence=sequence,
            file_name=file_name,
            world_time=float(self.session.world.time),
            file_sha256=sha256(raw).hexdigest(),
            payload_sha256=checkpoint.payload_sha256,
        )
        combined = self.manifest.entries + (entry,)
        retained = combined[-self.config.retain_last :]
        updated = EnvironmentalFleetRuntimeCheckpointJournalManifest(
            schema_version=_SCHEMA_VERSION,
            runtime_version=__version__,
            interval_min=float(self.config.interval_min),
            retain_last=self.config.retain_last,
            next_sequence=sequence + 1,
            entries=retained,
        )

        # The checkpoint is not authoritative until this atomic manifest publication.
        _atomic_write_json(self.directory / _MANIFEST_NAME, _manifest_to_dict(updated))
        self.manifest = updated

        # Retention cleanup occurs only after the new manifest is durable.  A crash here
        # can only leave an unlisted orphan, never remove an authoritative checkpoint.
        retained_names = {item.file_name for item in retained}
        for retired in combined:
            if retired.file_name not in retained_names:
                try:
                    (self.directory / retired.file_name).unlink()
                except FileNotFoundError:
                    pass
        _fsync_directory(self.directory)
        return entry

    def checkpoint_now(self) -> EnvironmentalFleetRuntimeCheckpointJournalEntry:
        """Publish the current already-quiescent committed boundary immediately."""

        entry = self._publish_checkpoint()
        self._next_checkpoint_time = entry.world_time + self.config.interval_min
        return entry

    def run_until(self, end_time: float) -> EnvironmentalPersistentFleetControlResult:
        """Advance through ordinary caller horizons and checkpoint at periodic horizons."""

        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")

        last_result: EnvironmentalPersistentFleetControlResult | None = None
        while self._next_checkpoint_time <= end_time + _EPS:
            boundary = max(self.world.time, self._next_checkpoint_time)
            last_result = self.session.run_until(boundary)
            if self.world.time < boundary - _EPS:
                # A non-END_TIME stop such as COMMAND_OVERRIDDEN retains authority for the
                # caller; do not invent a checkpoint horizon beyond the actual world time.
                return last_result
            self._publish_checkpoint()
            self._next_checkpoint_time += self.config.interval_min
            if self.world.time >= end_time - _EPS:
                return last_result

        last_result = self.session.run_until(end_time)
        return last_result


def create_environmental_fleet_runtime_checkpoint_journal(
    directory: str | Path,
    session: EnvironmentalPersistentFleetControlSession,
    *,
    config: EnvironmentalFleetRuntimeCheckpointJournalConfig = EnvironmentalFleetRuntimeCheckpointJournalConfig(),
) -> EnvironmentalFleetRuntimeCheckpointJournal:
    return EnvironmentalFleetRuntimeCheckpointJournal.create(directory, session, config=config)


def resume_environmental_fleet_runtime_checkpoint_journal(
    directory: str | Path,
) -> EnvironmentalFleetRuntimeCheckpointJournal:
    return EnvironmentalFleetRuntimeCheckpointJournal.resume(directory)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    _atomic_write_bytes(path, data)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform/filesystem specific
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platform/filesystem specific
        pass
    finally:
        os.close(descriptor)


def _manifest_to_dict(manifest: EnvironmentalFleetRuntimeCheckpointJournalManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "runtime_version": manifest.runtime_version,
        "interval_min": manifest.interval_min,
        "retain_last": manifest.retain_last,
        "next_sequence": manifest.next_sequence,
        "entries": [asdict(entry) for entry in manifest.entries],
    }


def _read_manifest(path: Path) -> EnvironmentalFleetRuntimeCheckpointJournalManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
            "checkpoint journal manifest is missing or invalid"
        ) from exc
    required = {
        "schema_version",
        "runtime_version",
        "interval_min",
        "retain_last",
        "next_sequence",
        "entries",
    }
    if not isinstance(raw, dict) or set(raw) != required or not isinstance(raw["entries"], list):
        raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
            "checkpoint journal manifest fields are invalid"
        )
    try:
        config = EnvironmentalFleetRuntimeCheckpointJournalConfig(
            interval_min=float(raw["interval_min"]),
            retain_last=int(raw["retain_last"]),
        )
        next_sequence = int(raw["next_sequence"])
        entries = tuple(
            EnvironmentalFleetRuntimeCheckpointJournalEntry(
                sequence=int(item["sequence"]),
                file_name=str(item["file_name"]),
                world_time=float(item["world_time"]),
                file_sha256=str(item["file_sha256"]),
                payload_sha256=str(item["payload_sha256"]),
            )
            for item in raw["entries"]
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
            "checkpoint journal manifest values are invalid"
        ) from exc
    if next_sequence < 1:
        raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError("next_sequence must be positive")
    previous = 0
    previous_time: float | None = None
    for entry in entries:
        if entry.sequence <= previous or entry.sequence >= next_sequence:
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                "journal checkpoint sequences are not strictly increasing below next_sequence"
            )
        if _CHECKPOINT_RE.fullmatch(entry.file_name) is None:
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                "journal checkpoint filename is not canonical"
            )
        if int(_CHECKPOINT_RE.fullmatch(entry.file_name).group(1)) != entry.sequence:  # type: ignore[union-attr]
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                "journal checkpoint filename disagrees with sequence"
            )
        if not isfinite(entry.world_time):
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                "journal checkpoint world_time must be finite"
            )
        if previous_time is not None and entry.world_time < previous_time - _EPS:
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                "journal checkpoint times move backwards"
            )
        if len(entry.file_sha256) != 64 or len(entry.payload_sha256) != 64:
            raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
                "journal checkpoint digest fields are invalid"
            )
        previous = entry.sequence
        previous_time = entry.world_time
    return EnvironmentalFleetRuntimeCheckpointJournalManifest(
        schema_version=int(raw["schema_version"]),
        runtime_version=str(raw["runtime_version"]),
        interval_min=config.interval_min,
        retain_last=config.retain_last,
        next_sequence=next_sequence,
        entries=entries,
    )


def _read_listed_checkpoint(
    root: Path,
    entry: EnvironmentalFleetRuntimeCheckpointJournalEntry,
) -> EnvironmentalFleetRuntimeCheckpoint:
    path = root / entry.file_name
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != entry.file_sha256:
        raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
            f"whole-file SHA-256 mismatch for checkpoint sequence {entry.sequence}"
        )
    checkpoint = EnvironmentalFleetRuntimeCheckpoint.from_bytes(raw)
    if checkpoint.payload_sha256 != entry.payload_sha256:
        raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
            f"payload SHA-256 disagrees with manifest for checkpoint sequence {entry.sequence}"
        )
    if abs(checkpoint.audit.world_time - entry.world_time) > _EPS:
        raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
            f"checkpoint world_time disagrees with manifest for sequence {entry.sequence}"
        )
    return checkpoint


def _checkpoint_sequence_high_water(root: Path) -> int:
    high = 0
    try:
        items = tuple(root.iterdir())
    except OSError as exc:
        raise EnvironmentalFleetRuntimeCheckpointJournalIntegrityError(
            "checkpoint journal directory cannot be read"
        ) from exc
    for item in items:
        match = _CHECKPOINT_RE.fullmatch(item.name)
        if match is not None:
            high = max(high, int(match.group(1)))
    return high
