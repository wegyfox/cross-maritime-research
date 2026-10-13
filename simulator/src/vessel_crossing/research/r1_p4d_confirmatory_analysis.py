"""Frozen analysis for R1 P4D interaction-active confirmatory campaign."""
from __future__ import annotations

from collections import Counter
from math import isfinite, sqrt
from statistics import mean, stdev
from typing import Iterable, Mapping, Sequence

from scipy.stats import binomtest, t as student_t

from .r1_p4d_confirmatory_execution import (
    P4D_CONFIRMATORY_ARMS,
    P4D_CONFIRMATORY_CAMPAIGN_ID,
    P4D_CONFIRMATORY_RESULT_FORMAT,
    P4DConfirmatoryPlan,
)

P4D_CONFIRMATORY_ANALYSIS_FORMAT = "cross-r1-p4d-confirmatory-analysis-v1"
P4D_PRIMARY_FAMILY_SIZE = 8
P4D_ALPHA = 0.05
_EPS = 1e-9


def _canonical_json(value: object) -> bytes:
    import json
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _index(plan: P4DConfirmatoryPlan, documents: Iterable[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    expected={p.point_id:p for p in plan.points}; out={}; psha=plan.identity_sha256
    for d in documents:
        if d.get("format") != P4D_CONFIRMATORY_RESULT_FORMAT: raise ValueError("unexpected P4D confirmatory result format")
        if d.get("campaign_id") != plan.campaign_id or d.get("plan_sha256") != psha: raise ValueError("campaign/plan mismatch")
        if d.get("source_archive_sha256") != plan.source_archive_sha256: raise ValueError("source mismatch")
        po=d.get("point")
        if not isinstance(po, Mapping): raise ValueError("missing point metadata")
        pid=po.get("point_id")
        if not isinstance(pid,str) or pid not in expected or pid in out: raise ValueError("unknown/duplicate point")
        if po.get("identity_sha256") != expected[pid].compute_identity_sha256(): raise ValueError("point identity mismatch")
        out[pid]=d
    if len(out) != len(expected): raise ValueError(f"complete 1056-point set required; got {len(out)}")
    return out


def _physical_signature(p: Mapping[str, object]) -> object:
    return {
        "terminal":p["terminal"], "source":p["source"], "lifecycle":p["lifecycle"], "route":p["route"],
        "conflict":p["conflict"], "center_service":p["center_service"], "response":p["response"], "safety":p["safety"],
    }


def _derive(p: Mapping[str, object]) -> dict[str, object]:
    term=p["terminal"]; t=float(term["terminal_time_min"])
    records=p["center_service"]["records"]
    cutoff=min(90.0,t)
    count=sum(float(r["passage_time_min"]) <= cutoff + _EPS for r in records)
    life=p["lifecycle"]; safety=p.get("safety")
    overlap=0 if safety is None else int(safety.get("modeled_hull_overlap_episode_count",0))
    valid150=(term["status"]=="SUCCESS" and abs(t-150.0)<=_EPS and bool(life.get("lifecycle_conserved")) and bool(life.get("ownership_conserved")) and overlap==0)
    return {"center_yield90_count":count,"center_yield90_per_hour":count/1.5,"operational_valid150":1 if valid150 else 0,
            "terminal_status":term["status"],"terminal_time_min":t,"infeasibility_type":term.get("infeasibility_type")}


def _paired_t(values: Sequence[float]) -> dict[str, object]:
    n=len(values); mu=mean(values)
    if n<2: raise ValueError("paired t summary requires n>=2")
    if all(v==values[0] for v in values):
        raw=1.0 if values[0]==0.0 else 0.0
        return {"n":n,"mean_difference":mu,"sample_sd":0.0,"standard_error":0.0,"ci95_low":mu,"ci95_high":mu,"raw_p_value":raw}
    sd=stdev(values); se=sd/sqrt(n); crit=float(student_t.ppf(.975,df=n-1)); raw=float(2*student_t.sf(abs(mu/se),df=n-1))
    return {"n":n,"mean_difference":mu,"sample_sd":sd,"standard_error":se,"ci95_low":mu-crit*se,"ci95_high":mu+crit*se,"raw_p_value":raw}


def _paired_binary(p100: Sequence[int], p00: Sequence[int]) -> dict[str, object]:
    diffs=[float(a-b) for a,b in zip(p100,p00)]; base=_paired_t(diffs)
    b=sum(a==1 and z==0 for a,z in zip(p100,p00)); c=sum(a==0 and z==1 for a,z in zip(p100,p00)); disc=b+c
    raw=1.0 if disc==0 else float(binomtest(b,disc,p=.5,alternative="two-sided").pvalue)
    base.update({"discordant_p100_valid_p00_invalid":b,"discordant_p100_invalid_p00_valid":c,"discordant_total":disc,"raw_p_value":raw})
    return base


def _holm(raw: Sequence[float]) -> list[float]:
    if len(raw)!=P4D_PRIMARY_FAMILY_SIZE: raise ValueError("Holm family size must be 8")
    order=sorted(range(len(raw)),key=lambda i:(raw[i],i)); out=[1.0]*len(raw); run=0.0; m=len(raw)
    for rank,idx in enumerate(order):
        run=max(run,min(1.0,(m-rank)*float(raw[idx]))); out[idx]=run
    return out


def analyze_p4d_confirmatory_results(plan: P4DConfirmatoryPlan, documents: Iterable[Mapping[str, object]]) -> dict[str, object]:
    indexed=_index(plan,documents)
    bykey={(p.replication,p.regime_id,p.penetration,p.arm):indexed[p.point_id] for p in plan.points}

    # Execution-validity gate: resource failures/accounting defects are not scientific outcomes.
    invalid=[]
    for p in plan.points:
        phys=indexed[p.point_id]["physical"]; term=phys["terminal"]; life=phys["lifecycle"]
        if term["status"] not in {"SUCCESS","MODELED_INFEASIBLE"}: invalid.append((p.point_id,"terminal_class"))
        if term.get("resource_unavailability_type") is not None: invalid.append((p.point_id,"resource_unavailable"))
        if not bool(life.get("lifecycle_conserved")) or not bool(life.get("ownership_conserved")) or not bool(life.get("all_admitted_accounted_for")):
            invalid.append((p.point_id,"lifecycle_or_ownership"))
    if invalid: raise ValueError(f"invalid confirmatory executions: {invalid[:8]}")

    # Hard leakage gate: at P100 latency is physically inactive because every vessel is AS.
    leakage=[]
    for regime in ("LOW_INTERACTION_CONTROL","BALANCED_INTERACTION_ACTIVE","STRESS_SENSITIVITY"):
        for rep in range(1,plan.replication_count+1):
            docs=[bykey[(rep,regime,1.0,a)]["physical"] for a in P4D_CONFIRMATORY_ARMS]
            sig0=_physical_signature(docs[0])
            if any(_physical_signature(d)!=sig0 for d in docs[1:]): leakage.append({"regime_id":regime,"replication":rep})
    if leakage: raise ValueError(f"P100 latency treatment leakage detected: {leakage[:8]}")

    primary=[]
    for arm in P4D_CONFIRMATORY_ARMS:
        ydiff=[]; p100v=[]; p00v=[]; reps=[]
        for rep in range(1,plan.replication_count+1):
            d100=_derive(bykey[(rep,"BALANCED_INTERACTION_ACTIVE",1.0,arm)]["physical"])
            d00=_derive(bykey[(rep,"BALANCED_INTERACTION_ACTIVE",0.0,arm)]["physical"])
            ydiff.append(float(d100["center_yield90_per_hour"])-float(d00["center_yield90_per_hour"]))
            p100v.append(int(d100["operational_valid150"])); p00v.append(int(d00["operational_valid150"]))
            reps.append({"replication":rep,"p100":d100,"p00":d00})
        ys=_paired_t(ydiff); vs=_paired_binary(p100v,p00v)
        primary.append({"arm":arm,"outcome":"CENTER_YIELD90_PER_HOUR","replications":reps,"summary":ys})
        primary.append({"arm":arm,"outcome":"OPERATIONAL_VALID150","replications":reps,"summary":vs})
    adjusted=_holm([float(x["summary"]["raw_p_value"]) for x in primary])
    for row,adj in zip(primary,adjusted):
        row["summary"]["holm_adjusted_p_value"]=adj
        row["summary"]["resolution"]="RESOLVED" if adj<=P4D_ALPHA else "NOT_RESOLVED_AT_CURRENT_PRECISION"

    # Secondary balanced gradient summaries.
    gradients=[]
    for arm in P4D_CONFIRMATORY_ARMS:
        rows=[]
        for pen in (0.0,.25,.5,.75,1.0):
            vals=[_derive(bykey[(rep,"BALANCED_INTERACTION_ACTIVE",pen,arm)]["physical"]) for rep in range(1,plan.replication_count+1)]
            rows.append({"penetration":pen,"mean_center_yield90_per_hour":mean(float(v["center_yield90_per_hour"]) for v in vals),
                         "operational_valid150_rate":mean(int(v["operational_valid150"]) for v in vals),
                         "terminal_counts":dict(Counter(str(v["terminal_status"]) for v in vals))})
        yield_reversals=sum(rows[i+1]["mean_center_yield90_per_hour"] < rows[i]["mean_center_yield90_per_hour"]-_EPS for i in range(4))
        valid_reversals=sum(rows[i+1]["operational_valid150_rate"] < rows[i]["operational_valid150_rate"]-_EPS for i in range(4))
        gradients.append({"arm":arm,"penetration_rows":rows,"adjacent_yield_reversal_count":yield_reversals,"adjacent_validity_reversal_count":valid_reversals})

    sentinels=[]
    for regime in ("LOW_INTERACTION_CONTROL","STRESS_SENSITIVITY"):
        for arm in P4D_CONFIRMATORY_ARMS:
            for pen in (0.0,.5,1.0):
                vals=[_derive(bykey[(rep,regime,pen,arm)]["physical"]) for rep in range(1,plan.replication_count+1)]
                sentinels.append({"regime_id":regime,"arm":arm,"penetration":pen,
                    "mean_center_yield90_per_hour":mean(float(v["center_yield90_per_hour"]) for v in vals),
                    "operational_valid150_rate":mean(int(v["operational_valid150"]) for v in vals),
                    "terminal_counts":dict(Counter(str(v["terminal_status"]) for v in vals))})

    return {
        "format":P4D_CONFIRMATORY_ANALYSIS_FORMAT,"schema_version":1,"campaign_id":P4D_CONFIRMATORY_CAMPAIGN_ID,
        "plan_sha256":plan.identity_sha256,"source_archive_sha256":plan.source_archive_sha256,
        "physical_result_count":len(indexed),"primary_family_size":P4D_PRIMARY_FAMILY_SIZE,"alpha":P4D_ALPHA,
        "p100_latency_leakage_gate":"PASS","primary_contrasts":primary,"balanced_gradients":gradients,"sentinel_summaries":sentinels,
    }


def analysis_to_json_bytes(analysis: Mapping[str, object]) -> bytes:
    return _canonical_json(analysis)
