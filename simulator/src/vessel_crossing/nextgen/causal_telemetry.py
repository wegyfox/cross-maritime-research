"""Append-only causal telemetry for reproducible full-network diagnostics.

The ledger is intentionally observation-only.  It has no authority over simulation time,
commands, arbitration, plant state, or encounter lifecycle.  Callers may record canonical
causal boundaries from any research layer and later compare paired/common-random-number
arms to locate the first divergence in the recorded causal stream.

The schema is generic by design: R2 pair-context exposure is one use, while later
perception, degradation, VTS, plant-fidelity, fault, and human/ROC studies can reuse the
same evidence surface without introducing article-specific telemetry forks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

_EPS = 1e-10
_FORMAT = "vessel-crossing-nextgen-causal-telemetry"
_SCHEMA_VERSION = 1


class CausalTelemetryError(RuntimeError):
    """Raised when causal telemetry evidence violates the append-only contract."""


class CausalEventKind(str, Enum):
    """Stable high-level classes for cross-research causal evidence."""

    SOURCE_DEMAND = "SOURCE_DEMAND"
    SOURCE_ADMISSION = "SOURCE_ADMISSION"
    SAME_ROUTE_EVALUATION = "SAME_ROUTE_EVALUATION"
    CONTROL_DECISION = "CONTROL_DECISION"
    CONTROL_TRANSITION = "CONTROL_TRANSITION"
    SAFETY_STATE = "SAFETY_STATE"
    PERCEPTION_UPDATE = "PERCEPTION_UPDATE"
    ENVIRONMENT_UPDATE = "ENVIRONMENT_UPDATE"
    FALLBACK_TRANSITION = "FALLBACK_TRANSITION"
    TERMINAL = "TERMINAL"
    CUSTOM = "CUSTOM"


class CausalDivergenceReason(str, Enum):
    """Reason that two ordered causal streams first cease to be identical."""

    EVENT_MISMATCH = "EVENT_MISMATCH"
    LEFT_EXHAUSTED = "LEFT_EXHAUSTED"
    RIGHT_EXHAUSTED = "RIGHT_EXHAUSTED"


def _validated_text(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty when supplied")
    return value


def _validated_sha256(value: str | None, *, name: str) -> str | None:
    value = _validated_text(value, name=name)
    if value is None:
        return None
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _json_safe(value: Any, *, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return float(value)
    if isinstance(value, tuple | list):
        return [_json_safe(item, path=f"{path}[]") for item in value]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            out[key] = _json_safe(item, path=f"{path}.{key}")
        return out
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class CausalEventRecord:
    """One immutable observation in an ordered causal ledger."""

    sequence_index: int
    time_min: float
    event_kind: CausalEventKind
    source_component: str
    entity_ids: tuple[str, ...] = ()
    owner: str | None = None
    reason: str | None = None
    truth_state_sha256: str | None = None
    observed_state_sha256: str | None = None
    action_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_index, int) or self.sequence_index < 0:
            raise ValueError("sequence_index must be a non-negative integer")
        if not isfinite(self.time_min) or self.time_min < 0.0:
            raise ValueError("time_min must be finite and non-negative")
        if not isinstance(self.event_kind, CausalEventKind):
            raise TypeError("event_kind must be CausalEventKind")
        source_component = _validated_text(self.source_component, name="source_component")
        assert source_component is not None
        object.__setattr__(self, "source_component", source_component)
        entity_ids = tuple(self.entity_ids)
        if any(not isinstance(item, str) or not item for item in entity_ids):
            raise ValueError("entity_ids must contain only non-empty strings")
        object.__setattr__(self, "entity_ids", entity_ids)
        object.__setattr__(self, "owner", _validated_text(self.owner, name="owner"))
        object.__setattr__(self, "reason", _validated_text(self.reason, name="reason"))
        object.__setattr__(
            self,
            "truth_state_sha256",
            _validated_sha256(self.truth_state_sha256, name="truth_state_sha256"),
        )
        object.__setattr__(
            self,
            "observed_state_sha256",
            _validated_sha256(self.observed_state_sha256, name="observed_state_sha256"),
        )
        object.__setattr__(
            self,
            "action_sha256",
            _validated_sha256(self.action_sha256, name="action_sha256"),
        )
        metadata = _json_safe(dict(self.metadata))
        assert isinstance(metadata, dict)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def to_json_object(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "time_min": self.time_min,
            "event_kind": self.event_kind.value,
            "source_component": self.source_component,
            "entity_ids": list(self.entity_ids),
            "owner": self.owner,
            "reason": self.reason,
            "truth_state_sha256": self.truth_state_sha256,
            "observed_state_sha256": self.observed_state_sha256,
            "action_sha256": self.action_sha256,
            "metadata": _json_safe(self.metadata),
        }

    def causal_signature_sha256(self) -> str:
        """Hash event semantics while deliberately excluding the sequence index."""
        document = self.to_json_object()
        document.pop("sequence_index")
        return sha256(_canonical_json(document)).hexdigest()


@dataclass(frozen=True, slots=True)
class CausalTelemetrySnapshot:
    """Frozen portable causal evidence for one simulation arm."""

    events: tuple[CausalEventRecord, ...]
    canonical_bytes: bytes = field(repr=False)

    def to_json_object(self) -> dict[str, Any]:
        value = json.loads(self.canonical_bytes.decode("utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - construction invariant
            raise CausalTelemetryError("frozen causal telemetry is not a JSON object")
        return value

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes

    @property
    def result_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()


class CausalTelemetryLedger:
    """Mutable append-only collector that never mutates simulation state."""

    def __init__(self) -> None:
        self._events: list[CausalEventRecord] = []

    @property
    def events(self) -> tuple[CausalEventRecord, ...]:
        return tuple(self._events)

    def append(
        self,
        *,
        time_min: float,
        event_kind: CausalEventKind,
        source_component: str,
        entity_ids: tuple[str, ...] = (),
        owner: str | None = None,
        reason: str | None = None,
        truth_state_sha256: str | None = None,
        observed_state_sha256: str | None = None,
        action_sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CausalEventRecord:
        if self._events and time_min < self._events[-1].time_min - _EPS:
            raise CausalTelemetryError(
                "causal telemetry time must be non-decreasing"
            )
        event = CausalEventRecord(
            sequence_index=len(self._events),
            time_min=time_min,
            event_kind=event_kind,
            source_component=source_component,
            entity_ids=entity_ids,
            owner=owner,
            reason=reason,
            truth_state_sha256=truth_state_sha256,
            observed_state_sha256=observed_state_sha256,
            action_sha256=action_sha256,
            metadata={} if metadata is None else metadata,
        )
        self._events.append(event)
        return event

    def freeze(self) -> CausalTelemetrySnapshot:
        events = tuple(self._events)
        document = {
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "event_count": len(events),
            "events": [event.to_json_object() for event in events],
        }
        return CausalTelemetrySnapshot(events, _canonical_json(document))


def causal_telemetry_snapshot_from_json_bytes(raw: bytes) -> CausalTelemetrySnapshot:
    """Reconstruct and validate a frozen telemetry snapshot from canonical JSON bytes."""
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or value.get("format") != _FORMAT:
        raise ValueError("unexpected causal telemetry format")
    if int(value.get("schema_version", -1)) != _SCHEMA_VERSION:
        raise ValueError("unsupported causal telemetry schema version")
    ledger = CausalTelemetryLedger()
    events = value.get("events")
    if not isinstance(events, list):
        raise ValueError("causal telemetry events must be a list")
    for index, row in enumerate(events):
        if not isinstance(row, dict) or int(row.get("sequence_index", -1)) != index:
            raise ValueError("causal telemetry sequence index drift")
        ledger.append(
            time_min=float(row["time_min"]),
            event_kind=CausalEventKind(str(row["event_kind"])),
            source_component=str(row["source_component"]),
            entity_ids=tuple(str(x) for x in row.get("entity_ids", [])),
            owner=row.get("owner"),
            reason=row.get("reason"),
            truth_state_sha256=row.get("truth_state_sha256"),
            observed_state_sha256=row.get("observed_state_sha256"),
            action_sha256=row.get("action_sha256"),
            metadata=row.get("metadata", {}),
        )
    snapshot = ledger.freeze()
    if snapshot.to_json_bytes() != raw:
        raise ValueError("causal telemetry canonical/semantic identity drift")
    return snapshot


@dataclass(frozen=True, slots=True)
class FirstCausalDivergence:
    """First ordered event at which two identically instrumented arms diverge."""

    sequence_index: int
    reason: CausalDivergenceReason
    left: CausalEventRecord | None
    right: CausalEventRecord | None

    @property
    def time_min(self) -> float | None:
        times = [
            event.time_min
            for event in (self.left, self.right)
            if event is not None
        ]
        return min(times) if times else None


def first_causal_divergence(
    left: CausalTelemetrySnapshot,
    right: CausalTelemetrySnapshot,
) -> FirstCausalDivergence | None:
    """Return the first semantic mismatch between two ordered causal streams.

    The comparison assumes both arms use the same instrumentation contract.  It is suited
    to paired/CRN experiments: before a treatment causes any physical/control difference,
    the two ledgers must remain semantically identical.  Once event streams diverge, only
    the first divergence is reported; later alignment is intentionally not guessed.
    """

    if not isinstance(left, CausalTelemetrySnapshot) or not isinstance(
        right, CausalTelemetrySnapshot
    ):
        raise TypeError("left and right must be CausalTelemetrySnapshot values")

    common = min(len(left.events), len(right.events))
    for index in range(common):
        left_event = left.events[index]
        right_event = right.events[index]
        if left_event.causal_signature_sha256() != right_event.causal_signature_sha256():
            return FirstCausalDivergence(
                index,
                CausalDivergenceReason.EVENT_MISMATCH,
                left_event,
                right_event,
            )

    if len(left.events) < len(right.events):
        return FirstCausalDivergence(
            common,
            CausalDivergenceReason.LEFT_EXHAUSTED,
            None,
            right.events[common],
        )
    if len(right.events) < len(left.events):
        return FirstCausalDivergence(
            common,
            CausalDivergenceReason.RIGHT_EXHAUSTED,
            left.events[common],
            None,
        )
    return None
