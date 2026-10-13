"""R1 P4C matched-mechanism source-service primitives.

DESIGN/QUALIFICATION ONLY.

This module transfers the exact R1/P4B M1 service semantics into a microscopic-side
source gate without changing the frozen P4B implementation or result bytes.

IMPORTANT R1 GUARDRAIL: this is an opt-in *diagnostic mechanism-transfer control*, not
a replacement for the independent CROSS target model.  Ordinary R1 CROSS must continue
to generate admission/headway behavior endogenously and must not receive prescribed M1
headways.  P4C deliberately imposes them only in the labelled matched-control arm so that
construct-transfer error can be separated from additional physical admission delay.

The contract is intentionally narrow:

* one global service order sorted by ``(arrival_min, event_id)``;
* one predecessor->follower pair-headway matrix;
* one common logical service event, ``SOURCE_SERVICE_RELEASE``;
* no cross-stream bypass: only the current global head may be released;
* Layer-1 Micro-lite commits a head exactly at its headway-ready time;
* Layer-2 Full CROSS may commit later when physical source clearance permits, but a
  physically blocked global head remains head-of-line and later demands may not bypass it.

Consequently, with no physical blocking the release recursion is exactly the frozen M1
recursion.  Any later release in Full CROSS can then be attributed to microscopic-side
constraints layered *after* the matched service mechanism, rather than to a different
AS/HS autonomy mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Sequence

from .r1_p4b_publication_reference import (
    GUO_2024_ILLUSTRATIVE,
    PairHeadwayMatrix,
    TypedDemandEvent,
)

_EPS = 1e-10
SOURCE_SERVICE_RELEASE = "SOURCE_SERVICE_RELEASE"
P4C_MATCHED_MECHANISM_ID = "R1_P4C_MATCHED_GLOBAL_SOURCE_SERVICE_V1"


class P4CMatchedSourceServiceError(RuntimeError):
    """Raised when matched source-service state would violate the frozen M1 semantics."""


@dataclass(frozen=True, slots=True)
class MatchedSourceServiceRecord:
    event_id: str
    stream_id: str
    vessel_type: str
    arrival_min: float
    release_min: float
    waiting_min: float
    predecessor_type: str | None
    required_headway_min: float | None
    headway_ready_min: float
    service_event: str = SOURCE_SERVICE_RELEASE


@dataclass(frozen=True, slots=True)
class MatchedSourceServiceCandidate:
    """The *only* demand currently eligible to become a service release.

    ``headway_ready_min`` is the earliest time at which the matched pair-headway
    mechanism permits release.  Full CROSS is allowed to release later because of
    physical clearance, but never earlier and never by bypassing this candidate.
    """

    event_id: str
    stream_id: str
    vessel_type: str
    arrival_min: float
    predecessor_type: str | None
    required_headway_min: float | None
    headway_ready_min: float


@dataclass(frozen=True, slots=True)
class MatchedSourceServiceQueue:
    """Immutable global single-server gate matching the frozen M1 service discipline."""

    ordered_events: tuple[TypedDemandEvent, ...]
    matrix: PairHeadwayMatrix = GUO_2024_ILLUSTRATIVE
    next_index: int = 0
    records: tuple[MatchedSourceServiceRecord, ...] = ()

    def __post_init__(self) -> None:
        canonical = tuple(sorted(self.ordered_events, key=lambda e: (e.arrival_min, e.event_id)))
        if canonical != self.ordered_events:
            raise ValueError("ordered_events must be canonical global (arrival_min,event_id) order")
        if len({item.event_id for item in self.ordered_events}) != len(self.ordered_events):
            raise ValueError("event_id values must be unique")
        if self.next_index != len(self.records):
            raise P4CMatchedSourceServiceError("next_index must equal committed record count")
        if not 0 <= self.next_index <= len(self.ordered_events):
            raise ValueError("next_index outside queue bounds")
        for idx, record in enumerate(self.records):
            event = self.ordered_events[idx]
            if record.event_id != event.event_id:
                raise P4CMatchedSourceServiceError("record order differs from canonical global service order")
            if record.release_min < record.headway_ready_min - _EPS:
                raise P4CMatchedSourceServiceError("record released before matched headway was ready")
            if idx and record.release_min < self.records[idx - 1].release_min - _EPS:
                raise P4CMatchedSourceServiceError("service releases must be non-decreasing")

    @property
    def complete(self) -> bool:
        return self.next_index == len(self.ordered_events)

    @property
    def last_record(self) -> MatchedSourceServiceRecord | None:
        return None if not self.records else self.records[-1]

    @property
    def candidate(self) -> MatchedSourceServiceCandidate | None:
        if self.complete:
            return None
        event = self.ordered_events[self.next_index]
        predecessor = self.last_record
        if predecessor is None:
            required = None
            ready = event.arrival_min
            predecessor_type = None
        else:
            predecessor_type = predecessor.vessel_type
            required = self.matrix.headway_min(predecessor_type, event.vessel_type)
            ready = max(event.arrival_min, predecessor.release_min + required)
        return MatchedSourceServiceCandidate(
            event_id=event.event_id,
            stream_id=event.stream_id,
            vessel_type=event.vessel_type,
            arrival_min=event.arrival_min,
            predecessor_type=predecessor_type,
            required_headway_min=required,
            headway_ready_min=ready,
        )


def build_matched_source_service_queue(
    events: Sequence[TypedDemandEvent],
    *,
    matrix: PairHeadwayMatrix = GUO_2024_ILLUSTRATIVE,
    horizon_end_min: float | None = None,
) -> MatchedSourceServiceQueue:
    """Build the P4C gate from the same exogenous typed events used by M1.

    When ``horizon_end_min`` is supplied, demands after that horizon are excluded exactly
    like ``run_pair_headway_reference``.  Releases of included demands may still occur
    after the horizon and therefore constitute backlog.
    """

    if horizon_end_min is not None:
        if not isfinite(horizon_end_min) or horizon_end_min <= 0.0:
            raise ValueError("horizon_end_min must be finite and positive")
        selected = tuple(event for event in events if event.arrival_min <= horizon_end_min)
    else:
        selected = tuple(events)
    ordered = tuple(sorted(selected, key=lambda e: (e.arrival_min, e.event_id)))
    return MatchedSourceServiceQueue(ordered_events=ordered, matrix=matrix)


def commit_matched_source_release(
    queue: MatchedSourceServiceQueue,
    *,
    event_id: str,
    release_min: float,
) -> MatchedSourceServiceQueue:
    """Commit the current global head at or after its matched headway-ready time.

    Full CROSS should call this only after its native physical admission/clearance checks
    succeed.  Supplying any later event id is an explicit no-bypass violation.
    """

    if not isfinite(release_min):
        raise ValueError("release_min must be finite")
    candidate = queue.candidate
    if candidate is None:
        raise P4CMatchedSourceServiceError("cannot release from a complete matched queue")
    if event_id != candidate.event_id:
        raise P4CMatchedSourceServiceError(
            f"global head-of-line violation: expected {candidate.event_id!r}, got {event_id!r}"
        )
    if release_min < candidate.headway_ready_min - _EPS:
        raise P4CMatchedSourceServiceError(
            f"release {release_min} precedes matched headway-ready time {candidate.headway_ready_min}"
        )
    record = MatchedSourceServiceRecord(
        event_id=candidate.event_id,
        stream_id=candidate.stream_id,
        vessel_type=candidate.vessel_type,
        arrival_min=candidate.arrival_min,
        release_min=float(release_min),
        waiting_min=float(release_min) - candidate.arrival_min,
        predecessor_type=candidate.predecessor_type,
        required_headway_min=candidate.required_headway_min,
        headway_ready_min=candidate.headway_ready_min,
    )
    return replace(queue, next_index=queue.next_index + 1, records=queue.records + (record,))


def run_matched_micro_lite(
    events: Sequence[TypedDemandEvent],
    *,
    matrix: PairHeadwayMatrix = GUO_2024_ILLUSTRATIVE,
    horizon_end_min: float | None = None,
) -> MatchedSourceServiceQueue:
    """Run Layer-1 Micro-lite with no physical blocking.

    Every global head is committed exactly at ``headway_ready_min``.  This function is a
    qualification oracle, not a scientific Full CROSS execution path.
    """

    queue = build_matched_source_service_queue(events, matrix=matrix, horizon_end_min=horizon_end_min)
    while not queue.complete:
        candidate = queue.candidate
        assert candidate is not None
        queue = commit_matched_source_release(
            queue,
            event_id=candidate.event_id,
            release_min=candidate.headway_ready_min,
        )
    return queue


def matched_source_service_throughput_per_hour(
    records: Sequence[MatchedSourceServiceRecord],
    *,
    measurement_start_min: float,
    horizon_end_min: float,
) -> float:
    """Common P4C throughput: SOURCE_SERVICE_RELEASE events per measurement hour."""

    if not isfinite(measurement_start_min) or not isfinite(horizon_end_min):
        raise ValueError("measurement boundaries must be finite")
    if not 0.0 <= measurement_start_min < horizon_end_min:
        raise ValueError("measurement_start_min must be in [0,horizon_end_min)")
    completed = sum(
        measurement_start_min <= record.release_min <= horizon_end_min for record in records
    )
    return completed / ((horizon_end_min - measurement_start_min) / 60.0)
