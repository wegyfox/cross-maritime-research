"""Conflict-zone service-series metrics for R1 P4D.

The module is measurement-only.  It never feeds a reference headway into CROSS.  The
observed service series is defined by passive ``CONFLICT_CENTER_PASS`` observations.
An optional pair-headway matrix may be applied *after the run* as a labelled diagnostic
residual against the realized center-passage sequence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from statistics import mean
from typing import Iterable, Mapping, Sequence

from .r1_p4b_publication_reference import PairHeadwayMatrix
from .r1_p4d_conflict_observer import P4DConflictObservation

_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class P4DCenterServiceRecord:
    ordinal: int
    vessel_id: str
    stream_id: str
    vessel_type: str
    passage_time_min: float
    predecessor_vessel_id: str | None
    predecessor_type: str | None
    observed_headway_min: float | None
    reference_headway_min: float | None
    headway_residual_min: float | None
    arrival_rank: int
    conflict_request_time_min: float | None
    predecessor_server_busy: bool | None
    transition_class: str | None


def _event_field(event: object, name: str):
    if isinstance(event, Mapping):
        return event[name]
    return getattr(event, name)


def center_pass_service_summary(
    observations: Iterable[P4DConflictObservation],
    typed_events: Sequence[object],
    *,
    reference_matrix: PairHeadwayMatrix | None = None,
) -> dict[str, object]:
    """Summarize realized global service at the common center-passage boundary.

    ``typed_events`` supplies exogenous demand order/type only.  The actual service order
    comes exclusively from observed center passages.  Reordering is therefore visible
    rather than silently repaired to match the macro queue.
    """
    typed = list(typed_events)
    if len({_event_field(x, "event_id") for x in typed}) != len(typed):
        raise ValueError("typed event ids must be unique")
    type_by_id = {str(_event_field(x, "event_id")): str(_event_field(x, "vessel_type")) for x in typed}
    ordered_demand = sorted(
        typed,
        key=lambda x: (float(_event_field(x, "arrival_min")), str(_event_field(x, "event_id"))),
    )
    rank_by_id = {str(_event_field(x, "event_id")): i for i, x in enumerate(ordered_demand)}

    centers = sorted(
        (x for x in observations if x.role == "CONFLICT_CENTER_PASS"),
        key=lambda x: (x.observation_time_min, x.vessel_id),
    )
    if len({x.vessel_id for x in centers}) != len(centers):
        raise ValueError("a vessel may have at most one center-pass observation")
    missing = sorted(x.vessel_id for x in centers if x.vessel_id not in type_by_id)
    if missing:
        raise ValueError(f"center-pass vessels missing typed role: {missing!r}")
    requests = {
        x.vessel_id: float(x.observation_time_min)
        for x in observations
        if x.role == "CONFLICT_REQUEST"
    }

    records: list[P4DCenterServiceRecord] = []
    pair_rows: dict[str, list[tuple[float, float | None, float | None]]] = defaultdict(list)
    prev: P4DConflictObservation | None = None
    prev_type: str | None = None
    for i, obs in enumerate(centers, start=1):
        vessel_type = type_by_id[obs.vessel_id]
        headway = None if prev is None else float(obs.observation_time_min - prev.observation_time_min)
        if headway is not None and headway < -_EPS:
            raise AssertionError("center observations are not time ordered")
        required = None
        residual = None
        if headway is not None and reference_matrix is not None:
            assert prev_type is not None
            required = float(reference_matrix.headway_min(prev_type, vessel_type))
            residual = headway - required
            pair_rows[f"{prev_type}->{vessel_type}"].append((headway, required, residual))
        elif headway is not None:
            assert prev_type is not None
            pair_rows[f"{prev_type}->{vessel_type}"].append((headway, None, None))
        request_time = requests.get(obs.vessel_id)
        busy = None if prev is None or request_time is None else request_time <= float(prev.observation_time_min) + _EPS
        transition_class = None if prev is None else (
            "BUSY_AT_PREDECESSOR_PASS" if busy else "IDLE_OR_APPROACH_LIMITED"
        )
        records.append(P4DCenterServiceRecord(
            ordinal=i,
            vessel_id=obs.vessel_id,
            stream_id=obs.stream_id,
            vessel_type=vessel_type,
            passage_time_min=float(obs.observation_time_min),
            predecessor_vessel_id=None if prev is None else prev.vessel_id,
            predecessor_type=prev_type,
            observed_headway_min=headway,
            reference_headway_min=required,
            headway_residual_min=residual,
            arrival_rank=rank_by_id[obs.vessel_id],
            conflict_request_time_min=request_time,
            predecessor_server_busy=busy,
            transition_class=transition_class,
        ))
        prev = obs
        prev_type = vessel_type

    ranks = [r.arrival_rank for r in records]
    inversion_count = sum(1 for i in range(len(ranks)) for j in range(i + 1, len(ranks)) if ranks[i] > ranks[j])
    adjacent_overtake_count = sum(ranks[i] > ranks[i + 1] for i in range(len(ranks) - 1))
    headways = [r.observed_headway_min for r in records if r.observed_headway_min is not None]
    residuals = [r.headway_residual_min for r in records if r.headway_residual_min is not None]
    reference = [r.reference_headway_min for r in records if r.reference_headway_min is not None]
    busy_records = [r for r in records if r.predecessor_server_busy is True and r.observed_headway_min is not None]
    busy_headways = [float(r.observed_headway_min) for r in busy_records]
    busy_residuals = [float(r.headway_residual_min) for r in busy_records if r.headway_residual_min is not None]
    busy_reference = [float(r.reference_headway_min) for r in busy_records if r.reference_headway_min is not None]

    pair_summary = []
    for pair, rows in sorted(pair_rows.items()):
        obs_h = [x[0] for x in rows]
        req_h = [x[1] for x in rows if x[1] is not None]
        res_h = [x[2] for x in rows if x[2] is not None]
        pair_summary.append({
            "pair": pair,
            "count": len(rows),
            "mean_observed_headway_min": mean(obs_h),
            "mean_reference_headway_min": mean(req_h) if req_h else None,
            "mean_residual_min": mean(res_h) if res_h else None,
        })

    observed_span_rate = None
    if len(records) >= 2:
        span = records[-1].passage_time_min - records[0].passage_time_min
        observed_span_rate = None if span <= _EPS else 60.0 * (len(records) - 1) / span
    reference_span_rate = None
    if reference:
        ref_span = sum(reference)
        reference_span_rate = None if ref_span <= _EPS else 60.0 * len(reference) / ref_span

    return {
        "service_event": "CONFLICT_CENTER_PASS",
        "measurement_role": "OBSERVED_MICRO_SERVICE_SERIES_NO_REFERENCE_FORCING",
        "passage_count": len(records),
        "headway_count": len(headways),
        "mean_observed_headway_min": mean(headways) if headways else None,
        "min_observed_headway_min": min(headways) if headways else None,
        "max_observed_headway_min": max(headways) if headways else None,
        "observed_span_service_rate_per_hour": observed_span_rate,
        "busy_transition_count": len(busy_records),
        "idle_or_approach_limited_transition_count": len(headways) - len(busy_records),
        "mean_busy_headway_min": mean(busy_headways) if busy_headways else None,
        "min_busy_headway_min": min(busy_headways) if busy_headways else None,
        "max_busy_headway_min": max(busy_headways) if busy_headways else None,
        "arrival_order_inversion_count": inversion_count,
        "adjacent_overtake_count": adjacent_overtake_count,
        "reference_diagnostic": None if reference_matrix is None else {
            "reference_id": reference_matrix.reference_id,
            "reference_sha256": reference_matrix.identity_sha256,
            "semantic_status": "POST_HOC_DIAGNOSTIC_ONLY_NOT_TARGET_FORCING",
            "mean_reference_headway_min": mean(reference) if reference else None,
            "mean_headway_residual_min": mean(residuals) if residuals else None,
            "min_headway_residual_min": min(residuals) if residuals else None,
            "max_headway_residual_min": max(residuals) if residuals else None,
            "reference_span_service_rate_per_hour": reference_span_rate,
            "busy_only": {
                "transition_count": len(busy_records),
                "mean_reference_headway_min": mean(busy_reference) if busy_reference else None,
                "mean_observed_headway_min": mean(busy_headways) if busy_headways else None,
                "mean_headway_residual_min": mean(busy_residuals) if busy_residuals else None,
                "min_headway_residual_min": min(busy_residuals) if busy_residuals else None,
                "max_headway_residual_min": max(busy_residuals) if busy_residuals else None,
                "capacity_interpretation_rule": "ONLY_BUSY_TRANSITIONS_MAY_BE_USED_FOR_SERVICE_CAPACITY_DIAGNOSTICS",
            },
        },
        "pair_summary": pair_summary,
        "records": [asdict(x) for x in records],
    }
