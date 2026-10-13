"""Historical P4B evidence adapter for R1 P4D screening.

This module does not reinterpret P4B as a P4D confirmatory experiment.  It only uses the
already-frozen P4B P1 T0/T60 runs as *screening evidence* for the one microscopic axis
that is architecturally clean in P4D: command-response delay.

T0 is neutral command timing (AS=HS=0 min).  T60 is the same independent CROSS target
with HS=+1 min and AS=0 min.  No reference pair-headway is injected into CROSS in either
endpoint.  The adapter preserves terminal failures instead of silently dropping them.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Iterable, Mapping, Sequence

T0_ID = "T0_ZERO_EXTRA_LAG"
T60_ID = "T60_HS_PLUS_60S_STRESS"
P1_ID = "P1_FIRST_ORDER_RESPONSE_SYNTHETIC"


@dataclass(frozen=True, slots=True)
class HistoricalP4DPair:
    replication: int
    room_nm: float
    demand_per_hour: float
    penetration: float
    traffic_seed: int
    traffic_input_sha256: str
    t0_status: str
    t60_status: str
    t0_point_id: str
    t60_point_id: str
    metrics: Mapping[str, float | int | None]

    @property
    def both_success(self) -> bool:
        return self.t0_status == "SUCCESS" and self.t60_status == "SUCCESS"


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if isfinite(out) else None


def _metric(doc: Mapping[str, object], section: str, key: str) -> float | None:
    raw = doc.get(section)
    if not isinstance(raw, Mapping):
        return None
    return _finite_float(raw.get(key))


def _status(doc: Mapping[str, object]) -> str:
    terminal = doc.get("terminal")
    if not isinstance(terminal, Mapping):
        return "MISSING_TERMINAL"
    return str(terminal.get("status", "MISSING_STATUS"))


def _point(doc: Mapping[str, object]) -> Mapping[str, object]:
    point = doc.get("point")
    if not isinstance(point, Mapping):
        raise ValueError("result document missing point mapping")
    return point


def _nominal_route_duration_min(room_nm: float) -> float:
    # Frozen P4B route geometry is source=-room -> destination=+room at 0.1 NM/min.
    return 20.0 * room_nm


def _success_metrics(doc: Mapping[str, object], *, room_nm: float) -> dict[str, float | int | None]:
    if _status(doc) != "SUCCESS":
        return {}
    perf = doc.get("performance")
    resp = doc.get("response_diagnostics")
    safety = doc.get("safety")
    if not isinstance(perf, Mapping):
        return {}
    resp = resp if isinstance(resp, Mapping) else {}
    safety = safety if isinstance(safety, Mapping) else {}
    route = _finite_float(perf.get("evaluation_mean_route_duration_min_150"))
    nominal = _nominal_route_duration_min(room_nm)
    return {
        "source_wait_mean_min": _finite_float(perf.get("evaluation_mean_admitted_source_wait_min_150")),
        "source_wait_max_min": _finite_float(perf.get("evaluation_max_admitted_source_wait_min_150")),
        "source_backlog_150": _finite_float(perf.get("evaluation_source_backlog_count_150")),
        "route_duration_mean_min": route,
        "route_extra_mean_min": None if route is None else route - nominal,
        "system_duration_mean_min": _finite_float(perf.get("evaluation_mean_system_duration_min_150")),
        "source_throughput_per_hour_90": _finite_float(perf.get("primary_source_admission_throughput_per_hour_90")),
        "route_completed_150": _finite_float(perf.get("evaluation_route_completed_count_150")),
        "rebootstrap_count": _finite_float(resp.get("rebootstrap_count")),
        "fleet_observation_count": _finite_float(resp.get("fleet_observation_count")),
        "near_miss_count": _finite_float(safety.get("near_miss_episode_count")),
        "domain_intrusion_count": _finite_float(safety.get("safety_domain_intrusion_episode_count")),
        "hull_overlap_count": _finite_float(safety.get("modeled_hull_overlap_episode_count")),
        "unresolved_count": _finite_float(safety.get("unresolved_episode_count")),
        "min_center_separation_nm": _finite_float(safety.get("sampled_min_center_separation_nm")),
    }


def _difference(t60: Mapping[str, float | int | None], t0: Mapping[str, float | int | None], key: str) -> float | None:
    a = _finite_float(t60.get(key))
    b = _finite_float(t0.get(key))
    return None if a is None or b is None else a - b


def pair_p4b_t0_t60_results(result_docs: Iterable[Mapping[str, object]]) -> tuple[HistoricalP4DPair, ...]:
    """Pair every P1 T60 penetration with the matching neutral T0 CRN result."""
    t0: dict[tuple[int, float, float], Mapping[str, object]] = {}
    t60: list[Mapping[str, object]] = []
    for doc in result_docs:
        point = _point(doc)
        if str(point.get("plant_id")) != P1_ID:
            continue
        timing = str(point.get("timing_endpoint_id"))
        key = (
            int(point["replication"]),
            float(point["room_nm"]),
            float(point["demand_per_hour"]),
        )
        if timing == T0_ID:
            if key in t0:
                raise ValueError(f"duplicate T0 result for {key!r}")
            t0[key] = doc
        elif timing == T60_ID:
            t60.append(doc)

    pairs: list[HistoricalP4DPair] = []
    seen: set[tuple[int, float, float, float]] = set()
    for t60_doc in t60:
        p60 = _point(t60_doc)
        key3 = (int(p60["replication"]), float(p60["room_nm"]), float(p60["demand_per_hour"]))
        base = t0.get(key3)
        if base is None:
            raise ValueError(f"missing T0 counterpart for {key3!r}")
        penetration_raw = p60.get("penetration_execution")
        if penetration_raw is None:
            raise ValueError("T60 point missing penetration_execution")
        penetration = float(penetration_raw)
        key4 = (*key3, penetration)
        if key4 in seen:
            raise ValueError(f"duplicate T60 pair for {key4!r}")
        seen.add(key4)

        p0 = _point(base)
        if int(p0["traffic_seed"]) != int(p60["traffic_seed"]):
            raise ValueError(f"CRN traffic seed mismatch for {key4!r}")
        sha0 = str(p0.get("expected_traffic_input_sha256"))
        sha60 = str(p60.get("expected_traffic_input_sha256"))
        if sha0 != sha60:
            raise ValueError(f"CRN traffic SHA mismatch for {key4!r}")

        room = float(p60["room_nm"])
        m0 = _success_metrics(base, room_nm=room)
        m60 = _success_metrics(t60_doc, room_nm=room)
        metric_keys = sorted(set(m0) | set(m60))
        metrics: dict[str, float | int | None] = {}
        for metric in metric_keys:
            metrics[f"t0_{metric}"] = m0.get(metric)
            metrics[f"t60_{metric}"] = m60.get(metric)
            if metric not in {"min_center_separation_nm"}:
                metrics[f"delta_{metric}"] = _difference(m60, m0, metric)
        sep0 = _finite_float(m0.get("min_center_separation_nm"))
        sep60 = _finite_float(m60.get("min_center_separation_nm"))
        metrics["delta_min_center_separation_nm"] = None if sep0 is None or sep60 is None else sep60 - sep0

        pairs.append(HistoricalP4DPair(
            replication=key3[0], room_nm=key3[1], demand_per_hour=key3[2], penetration=penetration,
            traffic_seed=int(p60["traffic_seed"]), traffic_input_sha256=sha60,
            t0_status=_status(base), t60_status=_status(t60_doc),
            t0_point_id=str(p0["point_id"]), t60_point_id=str(p60["point_id"]), metrics=metrics,
        ))

    return tuple(sorted(pairs, key=lambda x: (x.room_nm, x.demand_per_hour, x.penetration, x.replication)))


def _mean_non_none(values: Sequence[float | None]) -> float | None:
    usable = [float(x) for x in values if x is not None and isfinite(float(x))]
    return mean(usable) if usable else None


def summarize_historical_command_delay_screen(pairs: Sequence[HistoricalP4DPair]) -> dict[str, object]:
    """Produce fail-visible cell summaries. Every cell keeps all 24 terminal outcomes."""
    grouped: dict[tuple[float, float, float], list[HistoricalP4DPair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.room_nm, pair.demand_per_hour, pair.penetration)].append(pair)

    cells: list[dict[str, object]] = []
    for (room, demand, penetration), rows in sorted(grouped.items()):
        status_pairs = Counter((r.t0_status, r.t60_status) for r in rows)
        both = [r for r in rows if r.both_success]
        delta_keys = sorted({k for r in both for k in r.metrics if k.startswith("delta_")})
        delta_means = {
            key: _mean_non_none([_finite_float(r.metrics.get(key)) for r in both])
            for key in delta_keys
        }
        changed = {
            "source_wait_mean": sum(abs(_finite_float(r.metrics.get("delta_source_wait_mean_min")) or 0.0) > 1e-10 for r in both),
            "route_duration": sum(abs(_finite_float(r.metrics.get("delta_route_duration_mean_min")) or 0.0) > 1e-10 for r in both),
            "rebootstrap": sum(abs(_finite_float(r.metrics.get("delta_rebootstrap_count")) or 0.0) > 1e-10 for r in both),
            "min_separation": sum(abs(_finite_float(r.metrics.get("delta_min_center_separation_nm")) or 0.0) > 1e-10 for r in both),
            "near_miss": sum(abs(_finite_float(r.metrics.get("delta_near_miss_count")) or 0.0) > 1e-10 for r in both),
        }
        cells.append({
            "room_nm": room,
            "demand_per_hour": demand,
            "penetration": penetration,
            "n_expected": 24,
            "n_pairs": len(rows),
            "n_both_success": len(both),
            "status_pairs": [
                {"t0": a, "t60": b, "count": count}
                for (a, b), count in sorted(status_pairs.items())
            ],
            "changed_replication_counts": changed,
            "mean_deltas_t60_minus_t0": delta_means,
        })

    overall_status = Counter((r.t0_status, r.t60_status) for r in pairs)
    both = [r for r in pairs if r.both_success]
    p100 = [r for r in pairs if abs(r.penetration - 1.0) <= 1e-12]
    p100_status_identity = sum(r.t0_status == r.t60_status for r in p100)
    p100_success = [r for r in p100 if r.both_success]
    p100_metric_identity = sum(
        all(
            value is None or abs(float(value)) <= 1e-10
            for key, value in r.metrics.items()
            if key.startswith("delta_")
        )
        for r in p100_success
    )
    return {
        "format": "cross-r1-p4d-historical-command-delay-screen-v1",
        "evidence_role": "SCREENING_ONLY_REUSE_OF_FROZEN_P4B_P1_T0_T60_NOT_P4D_CONFIRMATORY",
        "mechanism": {
            "T0": "AS=0 min, HS=0 min command delay",
            "T60": "AS=0 min, HS=1 min command delay",
            "reference_headway_forcing": False,
        },
        "pair_count": len(pairs),
        "both_success_count": len(both),
        "overall_status_pairs": [
            {"t0": a, "t60": b, "count": count}
            for (a, b), count in sorted(overall_status.items())
        ],
        "cell_count": len(cells),
        "p100_neutral_identity_control": {
            "pair_count": len(p100),
            "status_identity_count": p100_status_identity,
            "both_success_count": len(p100_success),
            "successful_metric_identity_count": p100_metric_identity,
            "pass": p100_status_identity == len(p100) and p100_metric_identity == len(p100_success),
        },
        "cells": cells,
    }
