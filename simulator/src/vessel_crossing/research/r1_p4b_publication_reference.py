"""R1 P4B publication-reference primitives.

This module is intentionally independent of CROSS microscopic outcomes.  It provides:

* deterministic, nested AS/HS type assignment from a dedicated SHA-256 namespace; and
* a finite-horizon single-server queue embedding for an externally declared
  adjacent-class headway matrix.

The published headway values are reference-side inputs only.  They are never fed into
microscopic admission, collision avoidance, vessel dynamics, or behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Iterable, Mapping, Sequence


AS = "AS"
HS = "HS"
_ALLOWED_TYPES = frozenset((AS, HS))
_ALLOWED_SEED_NAMESPACES = frozenset(("TRAFFIC_ARRIVALS", "TYPE_ASSIGNMENT", "BEHAVIOR_STOCHASTICITY"))


def publication_seed(*, campaign_id: str, replication: int, namespace: str) -> int:
    """Derive one deterministic 63-bit publication seed from an explicit namespace."""
    if not campaign_id:
        raise ValueError("campaign_id must be non-empty")
    if replication < 1:
        raise ValueError("replication must be >= 1")
    if namespace not in _ALLOWED_SEED_NAMESPACES:
        raise ValueError(f"unsupported seed namespace: {namespace!r}")
    payload = f"cross-r1-p4b-seed-v1|{campaign_id}|R{replication:04d}|{namespace}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class ExogenousDemandEvent:
    """One exogenous demand event before physical source admission."""

    event_id: str
    arrival_min: float
    stream_id: str = "combined"

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if not self.stream_id:
            raise ValueError("stream_id must be non-empty")
        if not isfinite(self.arrival_min) or self.arrival_min < 0.0:
            raise ValueError("arrival_min must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TypedDemandEvent:
    event_id: str
    arrival_min: float
    stream_id: str
    vessel_type: str
    type_uniform: float

    def __post_init__(self) -> None:
        if self.vessel_type not in _ALLOWED_TYPES:
            raise ValueError(f"unsupported vessel_type: {self.vessel_type!r}")
        if not 0.0 <= self.type_uniform < 1.0:
            raise ValueError("type_uniform must lie in [0, 1)")


@dataclass(frozen=True, slots=True)
class PairHeadwayMatrix:
    """Ordered predecessor->follower headway matrix in minutes."""

    as_to_as_min: float
    as_to_hs_min: float
    hs_to_as_min: float
    hs_to_hs_min: float
    reference_id: str

    def __post_init__(self) -> None:
        if not self.reference_id:
            raise ValueError("reference_id must be non-empty")
        for name, value in (
            ("as_to_as_min", self.as_to_as_min),
            ("as_to_hs_min", self.as_to_hs_min),
            ("hs_to_as_min", self.hs_to_as_min),
            ("hs_to_hs_min", self.hs_to_hs_min),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def headway_min(self, predecessor: str, follower: str) -> float:
        if predecessor not in _ALLOWED_TYPES or follower not in _ALLOWED_TYPES:
            raise ValueError("predecessor/follower must be AS or HS")
        if predecessor == AS and follower == AS:
            return self.as_to_as_min
        if predecessor == AS and follower == HS:
            return self.as_to_hs_min
        if predecessor == HS and follower == AS:
            return self.hs_to_as_min
        return self.hs_to_hs_min

    @property
    def identity_sha256(self) -> str:
        payload = {
            "reference_id": self.reference_id,
            "orientation": "PREDECESSOR_TO_FOLLOWER",
            "as_to_as_min": self.as_to_as_min,
            "as_to_hs_min": self.as_to_hs_min,
            "hs_to_as_min": self.hs_to_as_min,
            "hs_to_hs_min": self.hs_to_hs_min,
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


GUO_2024_ILLUSTRATIVE = PairHeadwayMatrix(
    as_to_as_min=7.50,
    as_to_hs_min=9.00,
    hs_to_as_min=11.25,
    hs_to_hs_min=11.25,
    reference_id="GUO_2024_OE_314_119734_ILLUSTRATIVE_PAIR_HEADWAYS",
)


@dataclass(frozen=True, slots=True)
class MacroQueueRecord:
    event_id: str
    stream_id: str
    vessel_type: str
    arrival_min: float
    departure_min: float
    waiting_min: float
    predecessor_type: str | None
    required_headway_min: float | None


@dataclass(frozen=True, slots=True)
class MacroQueueSummary:
    reference_id: str
    reference_sha256: str
    horizon_end_min: float
    event_count: int
    completed_count: int
    backlog_count: int
    throughput_per_hour: float
    mean_waiting_min: float
    records: tuple[MacroQueueRecord, ...]


def _canonical_type_uniform(*, campaign_id: str, replication: int, event: ExogenousDemandEvent) -> float:
    if not campaign_id:
        raise ValueError("campaign_id must be non-empty")
    if replication < 1:
        raise ValueError("replication must be >= 1")
    type_assignment_seed = publication_seed(
        campaign_id=campaign_id, replication=replication, namespace="TYPE_ASSIGNMENT"
    )
    payload = {
        "namespace": "cross-r1-p4b-type-assignment-v1",
        "campaign_id": campaign_id,
        "replication": int(replication),
        "type_assignment_seed": type_assignment_seed,
        "event_id": event.event_id,
        "stream_id": event.stream_id,
        "arrival_min": format(event.arrival_min, ".17g"),
    }
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(2**64)


def assign_nested_types(
    events: Iterable[ExogenousDemandEvent],
    *,
    penetration: float,
    campaign_id: str,
    replication: int,
) -> tuple[TypedDemandEvent, ...]:
    """Assign AS iff u_type < penetration using a penetration-independent u_type.

    Reusing the same campaign/replication/events therefore makes the assigned AS set
    nested as penetration increases.
    """
    if not isfinite(penetration) or not 0.0 <= penetration <= 1.0:
        raise ValueError("penetration must lie in [0, 1]")
    typed = []
    for event in events:
        u = _canonical_type_uniform(campaign_id=campaign_id, replication=replication, event=event)
        typed.append(
            TypedDemandEvent(
                event_id=event.event_id,
                arrival_min=event.arrival_min,
                stream_id=event.stream_id,
                vessel_type=AS if u < penetration else HS,
                type_uniform=u,
            )
        )
    return tuple(typed)


def run_pair_headway_reference(
    events: Sequence[TypedDemandEvent],
    *,
    matrix: PairHeadwayMatrix = GUO_2024_ILLUSTRATIVE,
    horizon_end_min: float,
    measurement_start_min: float = 0.0,
) -> MacroQueueSummary:
    """Run a deterministic finite-horizon single-server pair-headway queue.

    Events are globally ordered by (arrival_min, event_id); this is the declared shared
    macro-server order.  For i>1:
        d_i = max(a_i, d_(i-1) + h(z_(i-1), z_i))
    The first event departs at its arrival time.
    """
    if not isfinite(horizon_end_min) or horizon_end_min <= 0.0:
        raise ValueError("horizon_end_min must be finite and positive")
    if not isfinite(measurement_start_min) or not 0.0 <= measurement_start_min < horizon_end_min:
        raise ValueError("measurement_start_min must be in [0, horizon_end_min)")

    ordered = sorted(events, key=lambda e: (e.arrival_min, e.event_id))
    if len({e.event_id for e in ordered}) != len(ordered):
        raise ValueError("event_id values must be unique")

    records: list[MacroQueueRecord] = []
    prev_type: str | None = None
    prev_departure: float | None = None
    for event in ordered:
        if event.arrival_min > horizon_end_min:
            # Keep demand events outside the evaluation horizon out of this reference run.
            continue
        if prev_type is None:
            required = None
            departure = event.arrival_min
        else:
            required = matrix.headway_min(prev_type, event.vessel_type)
            assert prev_departure is not None
            departure = max(event.arrival_min, prev_departure + required)
        records.append(
            MacroQueueRecord(
                event_id=event.event_id,
                stream_id=event.stream_id,
                vessel_type=event.vessel_type,
                arrival_min=event.arrival_min,
                departure_min=departure,
                waiting_min=departure - event.arrival_min,
                predecessor_type=prev_type,
                required_headway_min=required,
            )
        )
        prev_type = event.vessel_type
        prev_departure = departure

    completed = sum(measurement_start_min <= r.departure_min <= horizon_end_min for r in records)
    arrived_by_end = sum(r.arrival_min <= horizon_end_min for r in records)
    backlog = sum(r.arrival_min <= horizon_end_min < r.departure_min for r in records)
    duration_h = (horizon_end_min - measurement_start_min) / 60.0
    throughput = completed / duration_h
    window_records = [r for r in records if measurement_start_min <= r.arrival_min <= horizon_end_min]
    mean_wait = sum(r.waiting_min for r in window_records) / len(window_records) if window_records else 0.0
    if backlog != arrived_by_end - sum(r.departure_min <= horizon_end_min for r in records):
        raise AssertionError("macro backlog accounting invariant failed")
    return MacroQueueSummary(
        reference_id=matrix.reference_id,
        reference_sha256=matrix.identity_sha256,
        horizon_end_min=horizon_end_min,
        event_count=arrived_by_end,
        completed_count=completed,
        backlog_count=backlog,
        throughput_per_hour=throughput,
        mean_waiting_min=mean_wait,
        records=tuple(records),
    )
