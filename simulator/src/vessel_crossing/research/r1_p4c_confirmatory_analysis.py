"""Fail-closed preregistered analysis for the R1 P4C matched-mechanism campaign."""
from __future__ import annotations

from collections import Counter
import json
from math import isfinite, sqrt
from statistics import mean, stdev
from typing import Iterable, Mapping, Sequence

from scipy.stats import t as student_t

from vessel_crossing.research.r1_p4c_confirmatory_execution import (
    P4C_POINT_FORMAT,
    P4CExecutionPlan,
    P4CConceptualRow,
)
from vessel_crossing.research.r1_p4b_publication_execution import (
    P4B_EVALUATION_END_MIN,
    P4B_PENETRATIONS,
    P4B_PRIMARY_MEASUREMENT_END_MIN,
)

P4C_ANALYSIS_FORMAT = "cross-r1-p4c-confirmatory-analysis"
P4C_ANALYSIS_SCHEMA_VERSION = 1
P4C_PRIMARY_FAMILY_SIZE = 36
P4C_ALPHA = 0.05
P4C_PRIMARY_METRIC = "primary_source_service_throughput_per_hour_90"
P4C_PRIMARY_NONZERO_PENETRATIONS = tuple(p for p in P4B_PENETRATIONS if p > 0.0)
_EPS = 1e-12


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _sign(value: float | None) -> int | None:
    if value is None: return None
    if abs(value) <= _EPS: return 0
    return 1 if value > 0 else -1


def _normalize_terminal(document: Mapping[str, object] | None) -> str:
    if document is None: return "MISSING_RESULT"
    terminal=document.get("terminal")
    if not isinstance(terminal,Mapping): return "MALFORMED_TERMINAL"
    status=str(terminal.get("status"))
    if status=="SUCCESS":
        try: t=float(terminal.get("terminal_time_min"))
        except (TypeError,ValueError): return "SUCCESS_MALFORMED_TIME"
        return "SUCCESS" if abs(t-P4B_EVALUATION_END_MIN)<=_EPS else "SUCCESS_NOT_FULL_HORIZON"
    return status


def _performance_value(document: Mapping[str, object] | None, key: str) -> float | None:
    if _normalize_terminal(document)!="SUCCESS": return None
    perf=document.get("performance") if document is not None else None
    if not isinstance(perf,Mapping): return None
    value=perf.get(key)
    if not isinstance(value,(int,float)) or not isfinite(float(value)): return None
    return float(value)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    n=len(values)
    if n==0:
        return {"n_available":0,"mean":None,"sample_sd":None,"standard_error":None,"ci95_low":None,"ci95_high":None,"raw_p_value":None}
    mu=mean(values)
    if n<2:
        return {"n_available":n,"mean":mu,"sample_sd":None,"standard_error":None,"ci95_low":None,"ci95_high":None,"raw_p_value":None}
    if all(v==values[0] for v in values):
        raw=1.0 if values[0]==0.0 else 0.0
        return {"n_available":n,"mean":mu,"sample_sd":0.0,"standard_error":0.0,"ci95_low":mu,"ci95_high":mu,"raw_p_value":raw}
    sd=stdev(values); se=sd/sqrt(n); critical=float(student_t.ppf(0.975,df=n-1)); half=critical*se
    raw=float(2.0*student_t.sf(abs(mu/se),df=n-1))
    return {"n_available":n,"mean":mu,"sample_sd":sd,"standard_error":se,"ci95_low":mu-half,"ci95_high":mu+half,"raw_p_value":raw}


def _descriptive(values: Sequence[float]) -> dict[str, float | int | None]:
    d=_summary(values); d.pop("raw_p_value",None); return d


def _holm(raw: Sequence[float | None]) -> list[float]:
    if len(raw)!=P4C_PRIMARY_FAMILY_SIZE: raise ValueError("Holm family must contain all 36 P4C contrasts")
    family=[1.0 if p is None else float(p) for p in raw]
    order=sorted(range(len(family)),key=lambda i:(family[i],i)); adjusted=[1.0]*len(family); running=0.0; m=len(family)
    for rank,idx in enumerate(order):
        candidate=min(1.0,(m-rank)*family[idx]); running=max(running,candidate); adjusted[idx]=running
    return adjusted


def _index_results(plan: P4CExecutionPlan, documents: Iterable[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    points={p.point_id:p for p in plan.physical_points}; indexed={}; psha=plan.identity_sha256
    for d in documents:
        if d.get("format")!=P4C_POINT_FORMAT: raise ValueError("unexpected P4C result format")
        if d.get("campaign_id")!=plan.campaign_id or d.get("plan_sha256")!=psha: raise ValueError("P4C result campaign/plan mismatch")
        if d.get("source_archive_sha256")!=plan.source_archive_sha256: raise ValueError("P4C result source mismatch")
        po=d.get("point")
        if not isinstance(po,Mapping): raise ValueError("P4C result point metadata missing")
        pid=po.get("point_id")
        if not isinstance(pid,str) or pid not in points: raise ValueError("unknown P4C physical point")
        if pid in indexed: raise ValueError("duplicate P4C physical result")
        if po.get("identity_sha256")!=points[pid].identity_sha256: raise ValueError("P4C physical point identity mismatch")
        indexed[pid]=d
    if len(indexed)!=len(points):
        missing=len(points)-len(indexed); raise ValueError(f"P4C analysis requires complete result set; missing={missing}")
    return indexed


def _row_lookup(plan: P4CExecutionPlan) -> dict[tuple[object,...],P4CConceptualRow]:
    return {(r.replication,r.plant_id,r.room_nm,r.demand_per_hour,r.penetration):r for r in plan.conceptual_rows}


def _layer1_lookup(plan: P4CExecutionPlan) -> dict[tuple[int,float,float],Mapping[str,object]]:
    out={}
    for b in plan.input_bases:
        for row in b.layer1_by_penetration:
            if row.get("m1_micro_lite_exact_parity") is not True: raise ValueError("plan contains failed M1/Micro-lite parity")
            out[(b.replication,b.demand_per_hour,float(row["penetration"]))]=row
    return out


def analyze_p4c_results(plan: P4CExecutionPlan, documents: Iterable[Mapping[str,object]]) -> dict[str,object]:
    result_map=_index_results(plan,documents); rows=_row_lookup(plan); layer1=_layer1_lookup(plan)
    p1=next(p.plant_id for p in plan.physical_points if p.plant_id.startswith("P1_"))
    p2=next(p.plant_id for p in plan.physical_points if p.plant_id.startswith("P2_"))
    rooms=sorted({p.room_nm for p in plan.physical_points if p.plant_id==p1})
    demands=sorted({p.demand_per_hour for p in plan.physical_points if p.plant_id==p1})
    primary=[]
    for room in rooms:
        for demand in demands:
            for penetration in P4C_PRIMARY_NONZERO_PENETRATIONS:
                reps=[]; vals=[]
                for rep in range(1,plan.replication_count+1):
                    pr=rows[(rep,p1,room,demand,penetration)]; br=rows[(rep,p1,room,demand,0.0)]
                    pd=result_map[pr.physical_point_id]; bd=result_map[br.physical_point_id]
                    ps=_normalize_terminal(pd); bs=_normalize_terminal(bd)
                    pf=_performance_value(pd,P4C_PRIMARY_METRIC); bf=_performance_value(bd,P4C_PRIMARY_METRIC)
                    l1p=float(layer1[(rep,demand,penetration)]["primary_throughput_per_hour"])
                    l1b=float(layer1[(rep,demand,0.0)]["primary_throughput_per_hour"])
                    l1gain=l1p-l1b; l2gain=None if pf is None or bf is None else pf-bf
                    attenuation=None if l2gain is None else l2gain-l1gain
                    if attenuation is not None: vals.append(attenuation)
                    reps.append({"replication":rep,"penetration_terminal":ps,"baseline_terminal":bs,
                                 "layer1_gain":l1gain,"layer2_gain":l2gain,"A_T":attenuation})
                summary=_summary(vals)
                primary.append({"room_nm":room,"demand_per_hour":demand,"penetration":penetration,
                                "replications":reps,"summary":summary})
    adjusted=_holm([x["summary"]["raw_p_value"] for x in primary])
    for row,adj in zip(primary,adjusted):
        row["summary"]["holm_adjusted_p_value"]=adj
        row["summary"]["resolution"]="RESOLVED_NONZERO" if adj<=P4C_ALPHA else "NOT_RESOLVED_AT_CURRENT_PRECISION"
        row["summary"]["mean_sign"]=_sign(row["summary"]["mean"])

    # Descriptive P1 supporting cell summaries.
    supporting=[]
    for room in rooms:
        for demand in demands:
            for p in P4B_PENETRATIONS:
                docs=[result_map[rows[(rep,p1,room,demand,p)].physical_point_id] for rep in range(1,plan.replication_count+1)]
                supporting.append({
                    "room_nm":room,"demand_per_hour":demand,"penetration":p,
                    "terminal_counts":dict(Counter(_normalize_terminal(d) for d in docs)),
                    "route_completed_150":_descriptive([v for d in docs if (v:=_performance_value(d,"evaluation_route_completed_count_150")) is not None]),
                    "mean_system_duration_150":_descriptive([v for d in docs if (v:=_performance_value(d,"evaluation_mean_system_duration_min_150")) is not None]),
                    "physical_extra_delay_mean":_descriptive([v for d in docs if (v:=_performance_value(d,"matched_physical_extra_delay_mean_min")) is not None]),
                    "physical_extra_delay_max":_descriptive([v for d in docs if (v:=_performance_value(d,"matched_physical_extra_delay_max_min")) is not None]),
                })

    # Sparse P2 sentinel: terminal relation and A_T sign relation versus P1.
    sentinel=[]
    sentinel_cells=sorted({(r.room_nm,r.demand_per_hour,r.penetration) for r in plan.conceptual_rows if r.plant_id==p2})
    primary_map={(x["room_nm"],x["demand_per_hour"],x["penetration"]):x for x in primary}
    for room,demand,p in sentinel_cells:
        terminal_pairs=[]
        for rep in range(1,plan.replication_count+1):
            d1=result_map[rows[(rep,p1,room,demand,p)].physical_point_id]
            d2=result_map[rows[(rep,p2,room,demand,p)].physical_point_id]
            terminal_pairs.append(f"{_normalize_terminal(d1)}->{_normalize_terminal(d2)}")
        item={"room_nm":room,"demand_per_hour":demand,"penetration":p,
              "terminal_relation_counts":dict(Counter(terminal_pairs))}
        if p>0.0:
            p1row=primary_map[(room,demand,p)]
            p2vals=[]
            for rep in range(1,plan.replication_count+1):
                pr=rows[(rep,p2,room,demand,p)]; br=rows[(rep,p2,room,demand,0.0)]
                pd=result_map[pr.physical_point_id]; bd=result_map[br.physical_point_id]
                pf=_performance_value(pd,P4C_PRIMARY_METRIC); bf=_performance_value(bd,P4C_PRIMARY_METRIC)
                l1gain=float(layer1[(rep,demand,p)]["primary_throughput_per_hour"])-float(layer1[(rep,demand,0.0)]["primary_throughput_per_hour"])
                if pf is not None and bf is not None: p2vals.append((pf-bf)-l1gain)
            item["p1_mean_A_T_sign"]=_sign(p1row["summary"]["mean"])
            item["p2_mean_A_T_sign"]=_sign(mean(p2vals) if p2vals else None)
            item["p2_A_T_descriptive"]=_descriptive(p2vals)
        sentinel.append(item)

    return {
        "format":P4C_ANALYSIS_FORMAT,"schema_version":P4C_ANALYSIS_SCHEMA_VERSION,
        "campaign_id":plan.campaign_id,"plan_sha256":plan.identity_sha256,"source_archive_sha256":plan.source_archive_sha256,
        "primary_metric":P4C_PRIMARY_METRIC,"primary_measurement_window_min":[0.0,P4B_PRIMARY_MEASUREMENT_END_MIN],
        "evaluation_horizon_min":P4B_EVALUATION_END_MIN,"primary_family_size":P4C_PRIMARY_FAMILY_SIZE,
        "alpha":P4C_ALPHA,"matched_arm_role":plan.matched_arm_role,"behavior_timing":plan.behavior_timing,
        "transfer_fidelity_gate":"PASS_M1_EQUALS_MICROLITE_FOR_ALL_FROZEN_INPUT_BASES",
        "physical_result_count":len(result_map),"primary_contrasts":primary,
        "p1_supporting_cells":supporting,"p2_sentinel":sentinel,
    }


def analysis_to_json_bytes(analysis: Mapping[str,object]) -> bytes:
    return _canonical_json(analysis)
