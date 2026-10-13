"""C18 explicit coordination/VTS information channel.

The channel transports declared information and intent.  It has no GroundTruthWorld
access and no actuator/plant authority.  VTS/peer messages therefore remain an
observable input to a controller rather than an omniscient coordinator shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .scope import TestbedFeature


class CoordinationMode(str, Enum):
    PEER_INTENT = "PEER_INTENT"
    VTS_ADVISORY = "VTS_ADVISORY"
    CENTRAL_CLEARANCE = "CENTRAL_CLEARANCE"


class CoordinationParticipantRole(str, Enum):
    VESSEL = "VESSEL"
    VTS = "VTS"
    SHORE_OPERATOR = "SHORE_OPERATOR"
    EXTERNAL_SYSTEM = "EXTERNAL_SYSTEM"


class CoordinationMessageKind(str, Enum):
    INTENT = "INTENT"
    POSITION_REPORT = "POSITION_REPORT"
    HAZARD_REPORT = "HAZARD_REPORT"
    ADVISORY = "ADVISORY"
    CLEARANCE_REQUEST = "CLEARANCE_REQUEST"
    CLEARANCE_GRANT = "CLEARANCE_GRANT"
    CLEARANCE_DENIAL = "CLEARANCE_DENIAL"
    WARNING = "WARNING"
    ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"


class InformationBasis(str, Enum):
    OBSERVED_WORLD = "OBSERVED_WORLD"
    DECLARED_INTENT = "DECLARED_INTENT"
    OPERATOR_INPUT = "OPERATOR_INPUT"
    EXTERNAL_VTS_FEED = "EXTERNAL_VTS_FEED"
    DOCUMENTED_ASSUMPTION = "DOCUMENTED_ASSUMPTION"


class TransmissionStatus(str, Enum):
    SCHEDULED = "SCHEDULED"
    DROPPED = "DROPPED"
    DELIVERED = "DELIVERED"
    EXPIRED = "EXPIRED"


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("coordination payload cannot contain NaN/inf")
        return value
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    raise TypeError(f"unsupported coordination payload type: {type(value).__name__}")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _unit_interval(identity: str, namespace: str) -> float:
    raw = hashlib.sha256(f"{namespace}:{identity}".encode()).digest()
    return int.from_bytes(raw[:8], "big") / float(2**64)


@dataclass(frozen=True, slots=True)
class CoordinationParticipant:
    participant_id: str
    role: CoordinationParticipantRole

    def __post_init__(self) -> None:
        if not self.participant_id:
            raise ValueError("participant_id must be non-empty")


@dataclass(frozen=True, slots=True)
class CoordinationMessage:
    message_id: str
    sender_id: str
    recipient_ids: tuple[str, ...]
    kind: CoordinationMessageKind
    information_basis: InformationBasis
    created_at_min: float
    valid_until_min: float | None
    payload: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        *,
        message_id: str,
        sender_id: str,
        recipient_ids: Iterable[str],
        kind: CoordinationMessageKind,
        information_basis: InformationBasis,
        created_at_min: float,
        payload: Mapping[str, Any],
        valid_until_min: float | None = None,
    ) -> "CoordinationMessage":
        recipients = tuple(sorted(set(recipient_ids)))
        if not message_id or not sender_id:
            raise ValueError("message_id and sender_id must be non-empty")
        if not recipients or any(not item for item in recipients):
            raise ValueError("recipient_ids must contain at least one non-empty id")
        if sender_id in recipients:
            raise ValueError("sender cannot be a message recipient")
        if not isfinite(created_at_min) or created_at_min < 0:
            raise ValueError("created_at_min must be finite and non-negative")
        if valid_until_min is not None:
            if not isfinite(valid_until_min) or valid_until_min < created_at_min:
                raise ValueError("valid_until_min must be >= created_at_min")
        canonical_payload = MappingProxyType(_canonical(payload))
        return cls(message_id, sender_id, recipients, kind, information_basis, created_at_min, valid_until_min, canonical_payload)

    def identity_sha256(self) -> str:
        return _sha({
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_ids": list(self.recipient_ids),
            "kind": self.kind.value,
            "information_basis": self.information_basis.value,
            "created_at_min": self.created_at_min,
            "valid_until_min": self.valid_until_min,
            "payload": dict(self.payload),
        })


@dataclass(frozen=True, slots=True)
class CoordinationChannelSpec:
    channel_id: str
    channel_version: str
    mode: CoordinationMode
    base_latency_min: float = 0.0
    max_additional_latency_min: float = 0.0
    loss_probability: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.channel_id or not self.channel_version:
            raise ValueError("channel_id and channel_version must be non-empty")
        for name in ("base_latency_min", "max_additional_latency_min"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isfinite(self.loss_probability) or not 0 <= self.loss_probability <= 1:
            raise ValueError("loss_probability must be in [0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def identity_sha256(self) -> str:
        return _sha({
            "channel_id": self.channel_id,
            "channel_version": self.channel_version,
            "mode": self.mode.value,
            "base_latency_min": self.base_latency_min,
            "max_additional_latency_min": self.max_additional_latency_min,
            "loss_probability": self.loss_probability,
            "seed": self.seed,
        })


@dataclass(frozen=True, slots=True)
class TransmissionRecord:
    message: CoordinationMessage
    channel_sha256: str
    scheduled_delivery_min: float | None
    dropped: bool
    drop_draw: float
    latency_draw: float
    component_id: TestbedFeature = TestbedFeature.C18_COORDINATION_CHANNEL

    def status_at(self, time_min: float) -> TransmissionStatus:
        if not isfinite(time_min) or time_min < 0:
            raise ValueError("time_min must be finite and non-negative")
        if self.dropped:
            return TransmissionStatus.DROPPED
        assert self.scheduled_delivery_min is not None
        if time_min < self.scheduled_delivery_min:
            return TransmissionStatus.SCHEDULED
        if self.message.valid_until_min is not None and time_min > self.message.valid_until_min:
            return TransmissionStatus.EXPIRED
        return TransmissionStatus.DELIVERED

    def identity_sha256(self) -> str:
        return _sha({
            "message_sha256": self.message.identity_sha256(),
            "channel_sha256": self.channel_sha256,
            "scheduled_delivery_min": self.scheduled_delivery_min,
            "dropped": self.dropped,
            "drop_draw": self.drop_draw,
            "latency_draw": self.latency_draw,
        })


@dataclass(frozen=True, slots=True)
class CoordinationInbox:
    recipient_id: str
    time_min: float
    delivered: tuple[TransmissionRecord, ...]
    component_id: TestbedFeature = TestbedFeature.C18_COORDINATION_CHANNEL

    def message_ids(self) -> tuple[str, ...]:
        return tuple(record.message.message_id for record in self.delivered)

    def identity_sha256(self) -> str:
        return _sha({
            "recipient_id": self.recipient_id,
            "time_min": self.time_min,
            "transmission_sha256": [r.identity_sha256() for r in self.delivered],
        })


class CoordinationChannel:
    component_id = TestbedFeature.C18_COORDINATION_CHANNEL

    def __init__(self, *, spec: CoordinationChannelSpec, participants: Iterable[CoordinationParticipant]) -> None:
        self.spec = spec
        parts = tuple(sorted(tuple(participants), key=lambda p: p.participant_id))
        ids = [p.participant_id for p in parts]
        if not parts or len(ids) != len(set(ids)):
            raise ValueError("participants must be non-empty with unique ids")
        self.participants = parts
        self._known = set(ids)
        self._records: dict[str, TransmissionRecord] = {}

    def publish(self, message: CoordinationMessage) -> TransmissionRecord:
        if message.sender_id not in self._known:
            raise ValueError("unknown coordination sender")
        unknown = set(message.recipient_ids) - self._known
        if unknown:
            raise ValueError(f"unknown coordination recipients: {sorted(unknown)}")
        if message.message_id in self._records:
            raise ValueError("duplicate coordination message_id")
        identity = message.identity_sha256()
        drop_draw = _unit_interval(identity, f"drop:{self.spec.seed}")
        latency_draw = _unit_interval(identity, f"latency:{self.spec.seed}")
        dropped = drop_draw < self.spec.loss_probability
        latency = self.spec.base_latency_min + self.spec.max_additional_latency_min * latency_draw
        delivery = None if dropped else message.created_at_min + latency
        record = TransmissionRecord(
            message=message,
            channel_sha256=self.spec.identity_sha256(),
            scheduled_delivery_min=delivery,
            dropped=dropped,
            drop_draw=drop_draw,
            latency_draw=latency_draw,
        )
        self._records[message.message_id] = record
        return record

    def inbox(self, recipient_id: str, *, time_min: float) -> CoordinationInbox:
        if recipient_id not in self._known:
            raise ValueError("unknown coordination recipient")
        delivered = tuple(sorted(
            (
                r for r in self._records.values()
                if recipient_id in r.message.recipient_ids and r.status_at(time_min) is TransmissionStatus.DELIVERED
            ),
            key=lambda r: (r.scheduled_delivery_min, r.message.message_id),
        ))
        return CoordinationInbox(recipient_id, time_min, delivered)

    def records(self) -> tuple[TransmissionRecord, ...]:
        return tuple(self._records[k] for k in sorted(self._records))


class NoOpCoordinationChannel:
    component_id = TestbedFeature.C18_COORDINATION_CHANNEL

    def inbox(self, recipient_id: str, *, time_min: float) -> CoordinationInbox:
        return CoordinationInbox(recipient_id, time_min, ())
