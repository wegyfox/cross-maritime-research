"""Fresh fixture selection for P4D action-latency sensitivity qualification.

Every V1/V2 fixture whose treated result has been observed is excluded.  Selection uses
only frozen neutral T0 physical classification plus exogenous traffic complexity.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence
from .r1_p4d_neutral_regime_selection import NeutralPhysicalRow
from .r1_p4d_qualification_fixture_selection import P4DQualificationFixture


def select_fresh_qualification_fixtures_v3(
    rows: Iterable[NeutralPhysicalRow], *, selected_regimes: Mapping[str, Mapping[str,float]],
    traffic_count_by_replication_cell: Mapping[tuple[int,float,float],int],
    p50_role_counts_by_replication_cell: Mapping[tuple[int,float,float],tuple[int,int]],
    excluded_prior_observed_baseline_point_ids: Sequence[str],
) -> tuple[P4DQualificationFixture,...]:
    excluded=set(str(x) for x in excluded_prior_observed_baseline_point_ids)
    row_list=tuple(rows); out=[]
    for regime_id in ("LOW_INTERACTION_CONTROL","BALANCED_INTERACTION_ACTIVE","STRESS_SENSITIVITY"):
        cell=selected_regimes[regime_id]; room=float(cell["room_nm"]); demand=float(cell["demand_per_hour"]); require_active=regime_id!="LOW_INTERACTION_CONTROL"
        candidates=[]
        for row in row_list:
            if row.point_id in excluded or abs(row.room_nm-room)>1e-12 or abs(row.demand_per_hour-demand)>1e-12:
                continue
            if not row.success or bool(row.in_system_active)!=require_active:
                continue
            key=(row.replication,row.room_nm,row.demand_per_hour); count=traffic_count_by_replication_cell.get(key); roles=p50_role_counts_by_replication_cell.get(key)
            if count is None or roles is None: continue
            ac,hc=int(roles[0]),int(roles[1])
            # Mechanism-coverage rule frozen before treated execution: both roles need
            # at least two vessels so one role cannot be represented by a single trajectory.
            if count<=0 or ac<2 or hc<2 or ac+hc!=count: continue
            candidates.append((int(count),row.replication,row,ac,hc))
        if not candidates: raise RuntimeError(f"no fresh mixed-P50 V3 fixture with >=2 AS and >=2 HS for {regime_id}")
        count,_,row,ac,hc=min(candidates,key=lambda x:(x[0],x[1]))
        out.append(P4DQualificationFixture(regime_id=regime_id,replication=row.replication,room_nm=row.room_nm,demand_per_hour=row.demand_per_hour,traffic_seed=row.traffic_seed,traffic_count=count,p50_as_count=ac,p50_hs_count=hc,baseline_in_system_active=row.in_system_active,baseline_point_id=row.point_id))
    return tuple(out)


def fixture_selection_payload_v3(fixtures: Sequence[P4DQualificationFixture], *, excluded_prior_observed_baseline_point_ids: Sequence[str]) -> dict[str,object]:
    return {
        "format":"cross-r1-p4d-action-latency-qualification-fixture-selection-v3",
        "evidence_role":"PRE_FREEZE_SENSITIVITY_FIXTURE_SELECTION_ONLY_NOT_CONFIRMATORY",
        "selection_uses_treatment_effect":False,
        "prior_treated_fixture_reuse":False,
        "excluded_prior_observed_baseline_point_ids":sorted(str(x) for x in excluded_prior_observed_baseline_point_ids),
        "selection_rules":{
            "cell_authority":"R1_P4D_SCREENING_REGIME_SELECTION_V2_NEUTRAL_ONLY",
            "low":"neutral SUCCESS + no in-system activation + unseen treatment + at least 2 AS and 2 HS; minimize traffic count then replication",
            "balanced_stress":"neutral SUCCESS + in-system activation + unseen treatment + at least 2 AS and 2 HS; minimize traffic count then replication",
            "complexity_proxy":"generated traffic vessel count only",
        },
        "fixtures":[{
            "regime_id":f.regime_id,"replication":f.replication,"room_nm":f.room_nm,
            "demand_per_hour":f.demand_per_hour,"traffic_seed":f.traffic_seed,
            "traffic_count":f.traffic_count,"p50_as_count":f.p50_as_count,"p50_hs_count":f.p50_hs_count,
            "baseline_in_system_active":f.baseline_in_system_active,"baseline_point_id":f.baseline_point_id,
        } for f in fixtures],
    }
