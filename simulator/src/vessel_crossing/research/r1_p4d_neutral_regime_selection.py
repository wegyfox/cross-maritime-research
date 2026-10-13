"""Neutral-only regime selection authority for R1 P4D.

This module intentionally ignores the historical P4B T60 response arm.  The old T60
adapter used debounce-like command semantics and is therefore not valid evidence for
the corrected P4D bounded-action-latency mechanism.

Only frozen P4B P1/T0 independent-CROSS evidence is used here to classify the baseline
physical regime.  Terminal failures remain visible.  Selection is based on baseline
interaction/feasibility, never on the sign or magnitude of a desired autonomy effect.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Iterable, Mapping, Sequence

P1_ID = "P1_FIRST_ORDER_RESPONSE_SYNTHETIC"
T0_ID = "T0_ZERO_EXTRA_LAG"
_EPS = 1e-9


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if isfinite(out) else None


def _terminal_status(doc: Mapping[str, object]) -> str:
    terminal = doc.get("terminal")
    if not isinstance(terminal, Mapping):
        return "MISSING_TERMINAL"
    return str(terminal.get("status", "MISSING_STATUS"))


def _point(doc: Mapping[str, object]) -> Mapping[str, object]:
    point = doc.get("point")
    if not isinstance(point, Mapping):
        raise ValueError("result document missing point mapping")
    return point


@dataclass(frozen=True, slots=True)
class NeutralPhysicalRow:
    replication: int
    room_nm: float
    demand_per_hour: float
    traffic_seed: int
    point_id: str
    status: str
    mean_source_wait_min: float | None = None
    mean_route_extra_min: float | None = None
    rebootstrap_count: int | None = None
    near_miss_count: int | None = None
    domain_intrusion_count: int | None = None
    hull_overlap_count: int | None = None
    unresolved_count: int | None = None
    min_center_separation_nm: float | None = None

    @property
    def success(self) -> bool:
        return self.status == "SUCCESS"

    @property
    def source_wait_active(self) -> bool:
        return self.success and self.mean_source_wait_min is not None and self.mean_source_wait_min > _EPS

    @property
    def route_perturbed(self) -> bool:
        return self.success and self.mean_route_extra_min is not None and abs(self.mean_route_extra_min) > _EPS

    @property
    def control_active(self) -> bool:
        return self.success and (self.rebootstrap_count or 0) > 0

    @property
    def safety_domain_active(self) -> bool:
        # Domain intrusion / hull overlap are geometric safety-envelope violations.
        # ``unresolved`` is not included here: a near-miss episode can remain open at
        # the evaluation boundary without any route/control perturbation.
        return self.success and any((x or 0) > 0 for x in (
            self.domain_intrusion_count, self.hull_overlap_count,
        ))

    @property
    def in_system_active(self) -> bool:
        # Near-miss alone is deliberately not an activation criterion: it can reflect a
        # close but completely nominal passage.  We require evidence that the native
        # dynamics/control/safety envelope actually became nontrivial.
        return self.route_perturbed or self.control_active or self.safety_domain_active


def neutral_rows_from_p4b_results(result_docs: Iterable[Mapping[str, object]]) -> tuple[NeutralPhysicalRow, ...]:
    rows: list[NeutralPhysicalRow] = []
    seen: set[tuple[int, float, float]] = set()
    for doc in result_docs:
        point = _point(doc)
        if str(point.get("plant_id")) != P1_ID or str(point.get("timing_endpoint_id")) != T0_ID:
            continue
        key = (int(point["replication"]), float(point["room_nm"]), float(point["demand_per_hour"]))
        if key in seen:
            raise ValueError(f"duplicate P1/T0 neutral result {key!r}")
        seen.add(key)
        status = _terminal_status(doc)
        kwargs: dict[str, object] = {}
        if status == "SUCCESS":
            perf = doc.get("performance") if isinstance(doc.get("performance"), Mapping) else {}
            response = doc.get("response_diagnostics") if isinstance(doc.get("response_diagnostics"), Mapping) else {}
            safety = doc.get("safety") if isinstance(doc.get("safety"), Mapping) else {}
            nominal = 20.0 * key[1]
            route = _finite(perf.get("evaluation_mean_route_duration_min_150"))
            kwargs = {
                "mean_source_wait_min": _finite(perf.get("evaluation_mean_admitted_source_wait_min_150")),
                "mean_route_extra_min": None if route is None else route - nominal,
                "rebootstrap_count": int(_finite(response.get("rebootstrap_count")) or 0),
                "near_miss_count": int(_finite(safety.get("near_miss_episode_count")) or 0),
                "domain_intrusion_count": int(_finite(safety.get("safety_domain_intrusion_episode_count")) or 0),
                "hull_overlap_count": int(_finite(safety.get("modeled_hull_overlap_episode_count")) or 0),
                "unresolved_count": int(_finite(safety.get("unresolved_episode_count")) or 0),
                "min_center_separation_nm": _finite(safety.get("sampled_min_center_separation_nm")),
            }
        rows.append(NeutralPhysicalRow(
            replication=key[0], room_nm=key[1], demand_per_hour=key[2],
            traffic_seed=int(point["traffic_seed"]), point_id=str(point["point_id"]), status=status,
            **kwargs,
        ))
    return tuple(sorted(rows, key=lambda x: (x.room_nm, x.demand_per_hour, x.replication)))


def summarize_neutral_physical_map(rows: Sequence[NeutralPhysicalRow]) -> dict[str, object]:
    grouped: dict[tuple[float, float], list[NeutralPhysicalRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.room_nm, row.demand_per_hour)].append(row)
    cells: list[dict[str, object]] = []
    for (room, demand), cell_rows in sorted(grouped.items()):
        statuses = Counter(x.status for x in cell_rows)
        success = [x for x in cell_rows if x.success]
        cells.append({
            "room_nm": room,
            "demand_per_hour": demand,
            "n_expected": 24,
            "n_rows": len(cell_rows),
            "status_counts": dict(sorted(statuses.items())),
            "success_count": len(success),
            "modeled_infeasible_count": statuses.get("MODELED_INFEASIBLE", 0),
            "source_wait_active_count": sum(x.source_wait_active for x in success),
            "near_miss_count_runs": sum((x.near_miss_count or 0) > 0 for x in success),
            "route_perturbed_count": sum(x.route_perturbed for x in success),
            "control_active_count": sum(x.control_active for x in success),
            "safety_domain_active_count": sum(x.safety_domain_active for x in success),
            "in_system_active_count": sum(x.in_system_active for x in success),
        })
    return {
        "format": "cross-r1-p4d-neutral-physical-map-v2",
        "evidence_role": "SCREENING_ONLY_FROZEN_P4B_P1_T0_NEUTRAL_INDEPENDENT_CROSS",
        "supersedes_for_regime_selection": "R1_P4D_HISTORICAL_COMMAND_DELAY_SCREEN_V1 / T60 effect-based selection",
        "row_count": len(rows),
        "cell_count": len(cells),
        "cells": cells,
    }


def select_neutral_regimes(neutral_map: Mapping[str, object]) -> dict[str, object]:
    raw_cells = neutral_map.get("cells")
    if not isinstance(raw_cells, Sequence):
        raise ValueError("neutral map missing cells")
    cells = [dict(x) for x in raw_cells if isinstance(x, Mapping)]
    if len(cells) != 9:
        raise ValueError(f"expected 9 neutral room/demand cells, got {len(cells)}")
    for c in cells:
        if int(c.get("n_rows", 0)) != 24:
            raise ValueError("neutral selection requires all 24 frozen replications per cell")

    low = [c for c in cells if int(c["success_count"]) == 24 and int(c["in_system_active_count"]) == 0]
    if not low:
        raise RuntimeError("no clean low-interaction neutral control")
    low_pick = min(low, key=lambda c: (float(c["demand_per_hour"]), float(c["room_nm"])))

    # Balanced means approximately half of successful neutral replications exhibit an
    # actual in-system perturbation, while at least 20/24 remain feasible.  Ranking by
    # distance from 50% characterizes baseline interaction richness, not treatment effect.
    balanced = []
    for c in cells:
        success = int(c["success_count"]); active = int(c["in_system_active_count"])
        if success < 20:
            continue
        threshold = ceil(0.25 * success)
        if active < threshold:
            continue
        frac = active / success
        balanced.append((abs(frac - 0.5), -success, float(c["demand_per_hour"]), float(c["room_nm"]), c))
    if not balanced:
        raise RuntimeError("no balanced interaction-active neutral regime")
    balanced_pick = min(balanced)[-1]

    # Stress is defined by neutral feasibility pressure first, then interaction activity.
    # This uses no treatment response.  Ties prefer the tighter room, a predeclared
    # geometry ordering rather than an observed treatment effect.
    stress_pick = min(
        cells,
        key=lambda c: (
            -int(c["modeled_infeasible_count"]),
            -int(c["in_system_active_count"]),
            float(c["room_nm"]),
            -float(c["demand_per_hour"]),
        ),
    )

    return {
        "format": "cross-r1-p4d-neutral-regime-selection-v2",
        "evidence_role": "SCREENING_ONLY_BASELINE_PHYSICAL_REGIME_SELECTION",
        "uses_historical_t60_effect": False,
        "effect_direction_or_treatment_magnitude_used_for_ranking": False,
        "selection_rules": {
            "low_control": "24/24 SUCCESS and 0 in-system-active neutral replications; choose lowest demand then room",
            "balanced_active": ">=20/24 SUCCESS and >=25% in-system-active; choose active fraction closest to 50%, then coverage, demand, room",
            "stress": "maximize neutral MODELED_INFEASIBLE count, then neutral in-system-active count; tie by tighter room then higher demand",
            "in_system_active_definition": "route perturbation OR rebootstrap/control activity OR safety-domain intrusion/hull overlap; near-miss-only, unresolved-episode-only, and source-wait-only do not qualify",
        },
        "selected": {
            "LOW_INTERACTION_CONTROL": {"room_nm": low_pick["room_nm"], "demand_per_hour": low_pick["demand_per_hour"]},
            "BALANCED_INTERACTION_ACTIVE": {"room_nm": balanced_pick["room_nm"], "demand_per_hour": balanced_pick["demand_per_hour"]},
            "STRESS_SENSITIVITY": {"room_nm": stress_pick["room_nm"], "demand_per_hour": stress_pick["demand_per_hour"]},
        },
        "cells": cells,
    }
