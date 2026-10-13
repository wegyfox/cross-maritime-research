"""Deterministic representative-fixture selection for P4D qualification.

The scientific regime (low / balanced / stress) is selected from frozen neutral T0
physics.  Qualification still needs concrete replications to exercise the corrected
P4D behavior mechanism.  This module chooses *computationally bounded* representatives
without looking at treatment-response sign or magnitude.

Eligibility is deliberately baseline-only:

* LOW: successful neutral replication with no in-system activation;
* BALANCED/STRESS: successful neutral replication with in-system activation;
* all: the P50 type assignment must contain both AS and HS;
* ranking: fewest generated traffic vessels, then lowest replication index.

Traffic count is an exogenous realization-complexity proxy.  It is not a treatment
outcome and therefore can be used to keep qualification runnable without selecting a
"nice" behavioral effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .r1_p4d_neutral_regime_selection import NeutralPhysicalRow


@dataclass(frozen=True, slots=True)
class P4DQualificationFixture:
    regime_id: str
    replication: int
    room_nm: float
    demand_per_hour: float
    traffic_seed: int
    traffic_count: int
    p50_as_count: int
    p50_hs_count: int
    baseline_in_system_active: bool
    baseline_point_id: str

    def __post_init__(self) -> None:
        if self.regime_id not in {
            "LOW_INTERACTION_CONTROL",
            "BALANCED_INTERACTION_ACTIVE",
            "STRESS_SENSITIVITY",
        }:
            raise ValueError("unsupported P4D qualification regime")
        if self.replication <= 0:
            raise ValueError("replication must be positive")
        if self.traffic_count <= 0:
            raise ValueError("traffic_count must be positive")
        if self.p50_as_count <= 0 or self.p50_hs_count <= 0:
            raise ValueError("P50 qualification fixture must contain both AS and HS")
        if self.p50_as_count + self.p50_hs_count != self.traffic_count:
            raise ValueError("P50 role counts must equal traffic_count")


def select_qualification_fixtures(
    rows: Iterable[NeutralPhysicalRow],
    *,
    selected_regimes: Mapping[str, Mapping[str, float]],
    traffic_count_by_replication_cell: Mapping[tuple[int, float, float], int],
    p50_role_counts_by_replication_cell: Mapping[tuple[int, float, float], tuple[int, int]],
) -> tuple[P4DQualificationFixture, ...]:
    """Choose one bounded representative for each neutral-selected regime.

    No treated P4D result is accepted by this function, so treatment-effect magnitude
    cannot influence fixture selection.
    """
    row_list = tuple(rows)
    out: list[P4DQualificationFixture] = []
    ordered_regimes = (
        "LOW_INTERACTION_CONTROL",
        "BALANCED_INTERACTION_ACTIVE",
        "STRESS_SENSITIVITY",
    )
    for regime_id in ordered_regimes:
        if regime_id not in selected_regimes:
            raise ValueError(f"missing selected regime {regime_id!r}")
        cell = selected_regimes[regime_id]
        room = float(cell["room_nm"])
        demand = float(cell["demand_per_hour"])
        require_active = regime_id != "LOW_INTERACTION_CONTROL"
        candidates: list[tuple[int, int, NeutralPhysicalRow, int, int]] = []
        for row in row_list:
            if abs(row.room_nm - room) > 1e-12 or abs(row.demand_per_hour - demand) > 1e-12:
                continue
            if not row.success:
                continue
            if bool(row.in_system_active) != require_active:
                continue
            key = (row.replication, row.room_nm, row.demand_per_hour)
            count = traffic_count_by_replication_cell.get(key)
            roles = p50_role_counts_by_replication_cell.get(key)
            if count is None or roles is None:
                continue
            as_count, hs_count = int(roles[0]), int(roles[1])
            if count <= 0 or as_count <= 0 or hs_count <= 0 or as_count + hs_count != count:
                continue
            candidates.append((int(count), row.replication, row, as_count, hs_count))
        if not candidates:
            raise RuntimeError(f"no eligible mixed-P50 qualification fixture for {regime_id}")
        count, _, row, as_count, hs_count = min(candidates, key=lambda x: (x[0], x[1]))
        out.append(P4DQualificationFixture(
            regime_id=regime_id,
            replication=row.replication,
            room_nm=row.room_nm,
            demand_per_hour=row.demand_per_hour,
            traffic_seed=row.traffic_seed,
            traffic_count=count,
            p50_as_count=as_count,
            p50_hs_count=hs_count,
            baseline_in_system_active=row.in_system_active,
            baseline_point_id=row.point_id,
        ))
    return tuple(out)


def fixture_selection_payload(fixtures: Sequence[P4DQualificationFixture]) -> dict[str, object]:
    return {
        "format": "cross-r1-p4d-qualification-fixture-selection-v1",
        "evidence_role": "QUALIFICATION_FIXTURE_SELECTION_ONLY_NOT_CONFIRMATORY",
        "selection_uses_treatment_effect": False,
        "selection_rules": {
            "low": "neutral SUCCESS + no in-system activation + mixed P50; minimize traffic count then replication",
            "balanced_stress": "neutral SUCCESS + in-system activation + mixed P50; minimize traffic count then replication",
            "complexity_proxy": "generated traffic vessel count only",
        },
        "fixtures": [
            {
                "regime_id": f.regime_id,
                "replication": f.replication,
                "room_nm": f.room_nm,
                "demand_per_hour": f.demand_per_hour,
                "traffic_seed": f.traffic_seed,
                "traffic_count": f.traffic_count,
                "p50_as_count": f.p50_as_count,
                "p50_hs_count": f.p50_hs_count,
                "baseline_in_system_active": f.baseline_in_system_active,
                "baseline_point_id": f.baseline_point_id,
            }
            for f in fixtures
        ],
    }
