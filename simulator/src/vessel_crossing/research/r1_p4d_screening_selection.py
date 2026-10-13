"""Deterministic regime selection from the P4D historical command-delay screen.

Selection is screening-only.  It uses coverage/activity rules rather than choosing cells
by the sign or magnitude of a desired scientific effect.  Fresh confirmatory seeds are
still required after the surface is frozen.
"""
from __future__ import annotations

from math import ceil
from typing import Mapping, Sequence

_REQUIRED_P = (0.0, 0.5, 1.0)
_ACTIVITY_KEYS = ("min_separation", "near_miss", "rebootstrap", "route_duration", "source_wait_mean")


def _cell_index(screen: Mapping[str, object]) -> dict[tuple[float, float, float], Mapping[str, object]]:
    cells = screen.get("cells")
    if not isinstance(cells, Sequence):
        raise ValueError("screen missing cells")
    out = {}
    for raw in cells:
        if not isinstance(raw, Mapping):
            raise ValueError("screen cell must be a mapping")
        key = (float(raw["room_nm"]), float(raw["demand_per_hour"]), float(raw["penetration"]))
        if key in out:
            raise ValueError(f"duplicate screen cell {key!r}")
        out[key] = raw
    return out


def _all_success(cell: Mapping[str, object]) -> bool:
    pairs = cell.get("status_pairs")
    return (
        int(cell.get("n_pairs", 0)) == 24
        and isinstance(pairs, Sequence)
        and len(pairs) == 1
        and pairs[0].get("t0") == "SUCCESS"
        and pairs[0].get("t60") == "SUCCESS"
        and int(pairs[0].get("count", 0)) == 24
    )


def _changed_max(cell: Mapping[str, object]) -> int:
    changed = cell.get("changed_replication_counts")
    if not isinstance(changed, Mapping):
        return 0
    return max((int(changed.get(k, 0)) for k in _ACTIVITY_KEYS), default=0)


def _changed_total(cell: Mapping[str, object]) -> int:
    changed = cell.get("changed_replication_counts")
    if not isinstance(changed, Mapping):
        return 0
    return sum(int(changed.get(k, 0)) for k in _ACTIVITY_KEYS)


def _status_reversal_count(cell: Mapping[str, object]) -> int:
    pairs = cell.get("status_pairs")
    if not isinstance(pairs, Sequence):
        return 0
    return sum(int(x.get("count", 0)) for x in pairs if x.get("t0") != x.get("t60"))


def select_p4d_screening_regimes(screen: Mapping[str, object]) -> dict[str, object]:
    """Select low, balanced-active and stress regimes with explicit deterministic rules."""
    index = _cell_index(screen)
    room_demand = sorted({(r, d) for r, d, _ in index})
    for r, d in room_demand:
        for p in _REQUIRED_P:
            if (r, d, p) not in index:
                raise ValueError(f"missing required penetration cell {(r, d, p)!r}")

    low_candidates = []
    active_candidates = []
    stress_candidates = []
    diagnostics = []
    for room, demand in room_demand:
        p0, p50, p100 = (index[(room, demand, p)] for p in _REQUIRED_P)
        zero_change = all(_changed_total(c) == 0 for c in (p0, p50, p100))
        stable_all_success = all(_all_success(c) for c in (p0, p50, p100))
        if zero_change and stable_all_success:
            low_candidates.append((demand, room))

        balanced_ok = True
        activation = []
        reversals = 0
        for c in (p0, p50):
            n = int(c.get("n_both_success", 0))
            threshold = ceil(0.25 * n) if n else 10**9
            activation.append(_changed_max(c) >= threshold)
            reversals += _status_reversal_count(c)
            if n < 18:
                balanced_ok = False
        if balanced_ok and all(activation):
            # Ranking is coverage-first, then fewer terminal reversals, then lower demand/room.
            min_success = min(int(p0["n_both_success"]), int(p50["n_both_success"]))
            active_candidates.append((-min_success, reversals, demand, room))

        stress_ok = demand == max(x[1] for x in room_demand)
        if stress_ok:
            n0 = int(p0.get("n_both_success", 0)); n50 = int(p50.get("n_both_success", 0))
            if min(n0, n50) >= 12 and _changed_max(p0) > 0 and _changed_max(p50) > 0:
                # Prefer more usable paired blocks, then tighter room; effect magnitude is not ranked.
                stress_candidates.append((-min(n0, n50), room))

        diagnostics.append({
            "room_nm": room,
            "demand_per_hour": demand,
            "p0_both_success": int(p0.get("n_both_success", 0)),
            "p50_both_success": int(p50.get("n_both_success", 0)),
            "p0_max_changed_replications": _changed_max(p0),
            "p50_max_changed_replications": _changed_max(p50),
            "p0_status_reversals": _status_reversal_count(p0),
            "p50_status_reversals": _status_reversal_count(p50),
        })

    if not low_candidates or not active_candidates or not stress_candidates:
        raise RuntimeError("historical screen does not contain all required P4D regime classes")
    low_demand, low_room = min(low_candidates)
    _, active_reversals, active_demand, active_room = min(active_candidates)
    _, stress_room = min(stress_candidates)
    stress_demand = max(x[1] for x in room_demand)

    return {
        "format": "cross-r1-p4d-screening-regime-selection-v1",
        "evidence_role": "SCREENING_ONLY_SELECTION_NOT_CONFIRMATORY",
        "selection_rules": {
            "low_response_control": "24/24 SUCCESS at P00/P50/P100 and zero changed proxy counts; choose lowest demand then room",
            "balanced_active": "P00/P50 each >=18 both-success and >=25% of successful blocks changed in at least one physical proxy; rank coverage, terminal stability, then lower demand/room",
            "stress": "highest demand, P00/P50 each >=12 both-success, nonzero physical-proxy activation; rank usable coverage then tighter room",
            "effect_direction_or_mean_magnitude_used_for_ranking": False,
        },
        "selected": {
            "LOW_RESPONSE_CONTROL": {"room_nm": low_room, "demand_per_hour": low_demand},
            "BALANCED_INTERACTION_ACTIVE": {
                "room_nm": active_room, "demand_per_hour": active_demand,
                "p0_p50_status_reversals": active_reversals,
            },
            "STRESS_SENSITIVITY": {"room_nm": stress_room, "demand_per_hour": stress_demand},
        },
        "diagnostics": diagnostics,
    }
