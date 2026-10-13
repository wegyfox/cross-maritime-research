"""Fresh pre-confirmatory qualification for the P4D bounded-action-latency sensitivity axis.

This campaign is a new lineage after response-qualification V1/V2 failures.  It does not
reinterpret either failure.  The scientific construct is no longer a calibrated HS model;
it is a transparent bounded-action-latency sensitivity axis with fixed levels 0/15/30/60 s.

Modeled infeasibility is a valid physical/model outcome for the 30/60-s active-regime
arms, provided lifecycle/ownership is conserved and no resource failure occurs.  The 15-s
arm is the minimum non-zero activation level and must remain successful in both active
regimes before the axis can proceed to any confirmatory design.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

from vessel_crossing.nextgen import generate_poisson_network_traffic
from vessel_crossing.research.r1_p4b_behavior_qualification import traffic_input_sha256, typed_demand_for_penetration
from vessel_crossing.research.r1_p4b_publication_execution import build_publication_config

from .r1_p4d_action_latency_sensitivity import ARM_TO_LATENCY_MIN, arm_contract
from .r1_p4d_screening import P4DScreeningCase, run_p4d_screening_case

P4D_ACTION_LATENCY_QUAL_FORMAT="cross-r1-p4d-action-latency-qualification-plan-v1"
P4D_ACTION_LATENCY_RESULT_FORMAT="cross-r1-p4d-action-latency-qualification-result-v1"
P4D_ACTION_LATENCY_ANALYSIS_FORMAT="cross-r1-p4d-action-latency-qualification-analysis-v1"
P4D_ACTION_LATENCY_CAMPAIGN_ID="R1_P4D_ACTION_LATENCY_SENSITIVITY_QUALIFICATION_V1_2026-09-21"
P4D_ACTION_LATENCY_PENETRATION=0.5
P4D_ACTION_LATENCY_MAX_WORKERS=2
ARMS=tuple(ARM_TO_LATENCY_MIN)
REGIMES=("LOW_INTERACTION_CONTROL","BALANCED_INTERACTION_ACTIVE","STRESS_SENSITIVITY")
_EPS=1e-9


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode("utf-8")


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _typed_sha(typed: Sequence[object]) -> str:
    return _sha(_canonical_json([asdict(x) for x in typed]))


@dataclass(frozen=True, slots=True)
class P4DActionLatencyQualificationPoint:
    execution_order:int
    point_id:str
    regime_id:str
    arm:str
    replication:int
    room_nm:float
    demand_per_hour:float
    traffic_seed:int
    traffic_count:int
    p50_as_count:int
    p50_hs_count:int
    expected_traffic_input_sha256:str
    expected_typed_sha256:str
    expected_contract_sha256:str
    source_archive_sha256:str

    def __post_init__(self)->None:
        if self.execution_order<=0: raise ValueError("execution_order must be positive")
        if self.regime_id not in REGIMES: raise ValueError("unsupported regime")
        if self.arm not in ARMS: raise ValueError("unsupported arm")
        if self.traffic_count<=0 or self.p50_as_count<=0 or self.p50_hs_count<=0:
            raise ValueError("qualification points require mixed non-empty P50 traffic")
        if self.p50_as_count+self.p50_hs_count!=self.traffic_count:
            raise ValueError("role counts must equal traffic_count")

    @property
    def identity_payload(self)->dict[str,object]: return asdict(self)
    @property
    def identity_sha256(self)->str: return _sha(_canonical_json(self.identity_payload))


@dataclass(frozen=True, slots=True)
class P4DActionLatencyQualificationPlan:
    source_archive_sha256:str
    fixture_selection_sha256:str
    sensitivity_axis_sha256:str
    points:tuple[P4DActionLatencyQualificationPoint,...]

    @property
    def document(self)->dict[str,object]:
        return {
            "format":P4D_ACTION_LATENCY_QUAL_FORMAT,"schema_version":1,
            "campaign_id":P4D_ACTION_LATENCY_CAMPAIGN_ID,
            "evidence_role":"PRE_CONFIRMATORY_MODEL_SENSITIVITY_QUALIFICATION_ONLY",
            "scientific_confirmatory_interpretation_authorized":False,
            "penetration":P4D_ACTION_LATENCY_PENETRATION,
            "max_parallel_workers":P4D_ACTION_LATENCY_MAX_WORKERS,
            "source_archive_sha256":self.source_archive_sha256,
            "fixture_selection_sha256":self.fixture_selection_sha256,
            "sensitivity_axis_sha256":self.sensitivity_axis_sha256,
            "qualification_rules":qualification_rules(),
            "points":[dict(p.identity_payload,identity_sha256=p.identity_sha256) for p in self.points],
        }
    def to_json_bytes(self)->bytes: return _canonical_json(self.document)
    @property
    def identity_sha256(self)->str: return _sha(self.to_json_bytes())


def qualification_rules()->dict[str,object]:
    return {
        "lineage":{
            "response_qualification_v1":"FAILED_IMMUTABLE",
            "response_qualification_v2":"FAILED_IMMUTABLE",
            "v2_failure_reason":"LAT60 stress arm reached MODELED_INFEASIBLE under a frozen success-only gate",
        },
        "all_points":[
            "exactly 12 fresh planned points: 3 regimes x 4 fixed latency arms",
            "traffic input and nested P50 role assignment must match the frozen plan",
            "RESOURCE_UNAVAILABLE and unexpected exceptions are invalid qualification outcomes",
            "modeled hull overlap is forbidden before any terminal",
            "every admitted vessel must remain lifecycle- and ownership-conserved",
        ],
        "low_control":[
            "all four latency arms must reach SUCCESS at t=150",
            "all generated demands must be admitted",
            "all four arms must remain interaction-inactive",
            "source timeline, route records, safety evidence and terminal lifecycle must equal NEUTRAL",
        ],
        "balanced_stress_neutral":[
            "NEUTRAL must reach SUCCESS at t=150, admit all generated demands and remain interaction-active",
            "NEUTRAL must contain at least one CENTER_PASS passage",
        ],
        "balanced_stress_nonzero":[
            "LAT15 must reach SUCCESS at t=150 and remain interaction-active",
            "LAT30/LAT60 may be SUCCESS or MODELED_INFEASIBLE",
            "any SUCCESS nonzero arm must remain interaction-active, admit all demands and change at least one post-admission physical metric",
            "any MODELED_INFEASIBLE nonzero arm must show interaction before the terminal, retain at least one CENTER_PASS passage, and conserve lifecycle/ownership",
            "source-timeline change is reported but is not by itself a failure; pure source-pre-metering collapse is rejected by the interaction-active gate",
        ],
        "directional_or_monotonic_effect_requirement":"NONE",
        "parameter_selection_from_observed_effect":"FORBIDDEN; all 0/15/30/60-s levels are retained as a sensitivity axis",
    }


def build_action_latency_qualification_plan(fixture_selection:Mapping[str,object],*,source_archive_sha256:str,fixture_selection_sha256:str,sensitivity_axis_sha256:str)->P4DActionLatencyQualificationPlan:
    fixtures=fixture_selection.get("fixtures")
    if not isinstance(fixtures,Sequence) or len(fixtures)!=3: raise ValueError("fixture selection must contain exactly three fixtures")
    points=[]; order=0
    for f in fixtures:
        rep=int(f["replication"]);room=float(f["room_nm"]);demand=float(f["demand_per_hour"]);seed=int(f["traffic_seed"]);regime=str(f["regime_id"])
        cfg=build_publication_config(room_nm=room,demand_per_hour=demand,traffic_seed=seed)
        traffic=generate_poisson_network_traffic(cfg.network,cfg.streams,horizon=cfg.scenario.traffic_horizon_min,seed=cfg.scenario.seed,start_time=cfg.scenario.start_time)
        typed=typed_demand_for_penetration(traffic,penetration=P4D_ACTION_LATENCY_PENETRATION,campaign_id=P4D_ACTION_LATENCY_CAMPAIGN_ID,replication=rep)
        ac=sum(x.vessel_type=="AS" for x in typed); hc=sum(x.vessel_type=="HS" for x in typed)
        if ac<=0 or hc<=0: raise RuntimeError(f"lost mixed P50 assignment for {regime}")
        for arm in ARMS:
            order+=1; c=arm_contract(arm)
            points.append(P4DActionLatencyQualificationPoint(
                execution_order=order, point_id=f"P4DALS-{regime}-R{rep:02d}-P050-{arm}",regime_id=regime,arm=arm,
                replication=rep,room_nm=room,demand_per_hour=demand,traffic_seed=seed,traffic_count=traffic.count,p50_as_count=ac,p50_hs_count=hc,
                expected_traffic_input_sha256=traffic_input_sha256(traffic),expected_typed_sha256=_typed_sha(typed),expected_contract_sha256=c.identity_sha256,
                source_archive_sha256=source_archive_sha256,
            ))
    return P4DActionLatencyQualificationPlan(source_archive_sha256,fixture_selection_sha256,sensitivity_axis_sha256,tuple(points))


def plan_from_json_bytes(raw:bytes)->P4DActionLatencyQualificationPlan:
    d=json.loads(raw)
    if d.get("format")!=P4D_ACTION_LATENCY_QUAL_FORMAT: raise ValueError("unexpected plan format")
    pts=[]
    for item in d["points"]:
        row=dict(item);row.pop("identity_sha256",None);pts.append(P4DActionLatencyQualificationPoint(**row))
    p=P4DActionLatencyQualificationPlan(str(d["source_archive_sha256"]),str(d["fixture_selection_sha256"]),str(d["sensitivity_axis_sha256"]),tuple(pts))
    if p.to_json_bytes()!=raw: raise ValueError("plan canonical/semantic identity drift")
    return p


def run_action_latency_qualification_point(plan:P4DActionLatencyQualificationPlan,point:P4DActionLatencyQualificationPoint)->bytes:
    c=arm_contract(point.arm)
    if c.identity_sha256!=point.expected_contract_sha256: raise RuntimeError("contract identity drift")
    case=P4DScreeningCase(case_id=point.point_id,room_nm=point.room_nm,demand_per_hour=point.demand_per_hour,traffic_seed=point.traffic_seed,replication=point.replication,penetration=P4D_ACTION_LATENCY_PENETRATION,contract=c,campaign_id=P4D_ACTION_LATENCY_CAMPAIGN_ID)
    s=run_p4d_screening_case(case).document
    if s["traffic_input_sha256"]!=point.expected_traffic_input_sha256: raise RuntimeError("traffic identity drift")
    if _sha(_canonical_json(s["typed_roles"]))!=point.expected_typed_sha256: raise RuntimeError("typed-role identity drift")
    return _canonical_json({
        "format":P4D_ACTION_LATENCY_RESULT_FORMAT,"schema_version":1,"campaign_id":P4D_ACTION_LATENCY_CAMPAIGN_ID,
        "evidence_role":"PRE_CONFIRMATORY_MODEL_SENSITIVITY_QUALIFICATION_ONLY","scientific_confirmatory_interpretation_authorized":False,
        "plan_sha256":plan.identity_sha256,"source_archive_sha256":plan.source_archive_sha256,
        "point":dict(point.identity_payload,identity_sha256=point.identity_sha256),"screening":s,
    })


def _source_timeline(s:Mapping[str,object])->object: return s["source"]["admissions"]

def _post_signature(s:Mapping[str,object])->dict[str,object]:
    safety=s.get("safety")
    return {"route":s["route"]["records"],"conflict":s["conflict"]["gate_rows"],"rebootstrap":s["response"]["rebootstrap_count"],"minsep":None if safety is None else safety.get("sampled_min_center_separation_nm"),"center":s["center_service"]["records"],"lifecycle":s.get("lifecycle")}


def analyze_action_latency_qualification_results(plan:P4DActionLatencyQualificationPlan,result_documents:Sequence[Mapping[str,object]])->bytes:
    expected={p.point_id:p for p in plan.points}
    if len(result_documents)!=len(expected): raise ValueError(f"qualification requires {len(expected)} results, got {len(result_documents)}")
    by_id={}
    for d in result_documents:
        if d.get("format")!=P4D_ACTION_LATENCY_RESULT_FORMAT: raise ValueError("unexpected result format")
        if d.get("plan_sha256")!=plan.identity_sha256 or d.get("source_archive_sha256")!=plan.source_archive_sha256: raise ValueError("result plan/source mismatch")
        point=d.get("point"); pid=str(point["point_id"])
        if pid in by_id or pid not in expected: raise ValueError(f"duplicate/unexpected point {pid}")
        if point.get("identity_sha256")!=expected[pid].identity_sha256: raise ValueError(f"point identity drift {pid}")
        by_id[pid]=d
    checks=[]; overall=True; summaries=[]
    for regime in REGIMES:
        arm_docs={arm:by_id[next(p.point_id for p in plan.points if p.regime_id==regime and p.arm==arm)]["screening"] for arm in ARMS}
        neutral=arm_docs["NEUTRAL"]
        local=[]
        for arm,s in arm_docs.items():
            p=next(p for p in plan.points if p.regime_id==regime and p.arm==arm)
            term=s["terminal"]; safety=s.get("safety"); life=s.get("lifecycle",{})
            common={
                "traffic_identity":s["traffic_input_sha256"]==p.expected_traffic_input_sha256,
                "typed_identity":_sha(_canonical_json(s["typed_roles"]))==p.expected_typed_sha256,
                "terminal_class_valid":term["status"] in {"SUCCESS","MODELED_INFEASIBLE"},
                "no_resource_unavailability":term.get("resource_unavailability_type") is None,
                "no_hull_overlap":safety is None or int(safety.get("modeled_hull_overlap_episode_count",0))==0,
                "lifecycle_conserved":bool(life.get("lifecycle_conserved",False)),
                "ownership_conserved":bool(life.get("ownership_conserved",False)),
                "all_admitted_accounted_for":bool(life.get("all_admitted_accounted_for",False)),
            }
            for name,passed in common.items(): checks.append({"regime_id":regime,"arm":arm,"check":name,"pass":bool(passed)}); local.append(bool(passed))
        typed_same=all(arm_docs[a]["typed_roles"]==neutral["typed_roles"] for a in ARMS)
        checks.append({"regime_id":regime,"arm":"ALL","check":"typed_roles_identical_across_arms","pass":typed_same});local.append(typed_same)
        if regime=="LOW_INTERACTION_CONTROL":
            base_sig=_post_signature(neutral); base_source=_source_timeline(neutral); base_safety=neutral.get("safety")
            for arm,s in arm_docs.items():
                special={
                    "success_full_horizon":s["terminal"]["status"]=="SUCCESS" and abs(float(s["terminal"]["terminal_time_min"])-150.0)<=_EPS,
                    "all_demands_admitted":int(s["source"]["admission_count"])==int(s["source"]["demand_count"]),
                    "interaction_inactive":not bool(s["response"]["interaction_active"]),
                    "source_timeline_identical_to_neutral":_source_timeline(s)==base_source,
                    "post_admission_signature_identical_to_neutral":_post_signature(s)==base_sig,
                    "safety_identical_to_neutral":s.get("safety")==base_safety,
                }
                for name,passed in special.items(): checks.append({"regime_id":regime,"arm":arm,"check":name,"pass":bool(passed)});local.append(bool(passed))
        else:
            neutral_checks={
                "neutral_success_full_horizon":neutral["terminal"]["status"]=="SUCCESS" and abs(float(neutral["terminal"]["terminal_time_min"])-150.0)<=_EPS,
                "neutral_all_demands_admitted":int(neutral["source"]["admission_count"])==int(neutral["source"]["demand_count"]),
                "neutral_interaction_active":bool(neutral["response"]["interaction_active"]),
                "neutral_center_passage_present":int(neutral["center_service"]["passage_count"])>0,
            }
            for name,passed in neutral_checks.items(): checks.append({"regime_id":regime,"arm":"NEUTRAL","check":name,"pass":bool(passed)});local.append(bool(passed))
            for arm in ("LAT15","LAT30","LAT60"):
                s=arm_docs[arm]; status=s["terminal"]["status"]; active=bool(s["response"]["interaction_active"]); passage=int(s["center_service"]["passage_count"])>0
                special={"interaction_active":active,"center_passage_present":passage}
                if arm=="LAT15": special.update({
                    "minimum_nonzero_level_success_full_horizon":status=="SUCCESS" and abs(float(s["terminal"]["terminal_time_min"])-150.0)<=_EPS,
                    "minimum_nonzero_level_all_demands_admitted":int(s["source"]["admission_count"])==int(s["source"]["demand_count"]),
                    "post_admission_physics_changed":_post_signature(s)!=_post_signature(neutral),
                })
                elif status=="SUCCESS": special.update({
                    "successful_nonzero_all_demands_admitted":int(s["source"]["admission_count"])==int(s["source"]["demand_count"]),
                    "post_admission_physics_changed":_post_signature(s)!=_post_signature(neutral),
                })
                else: special.update({
                    "modeled_infeasible_has_type":bool(s["terminal"].get("infeasibility_type")),
                    "modeled_infeasible_before_full_horizon":float(s["terminal"]["terminal_time_min"])<150.0-_EPS,
                })
                for name,passed in special.items(): checks.append({"regime_id":regime,"arm":arm,"check":name,"pass":bool(passed)});local.append(bool(passed))
        rp=all(local);overall=overall and rp
        summaries.append({
            "regime_id":regime,"qualification_pass":rp,
            "arms":{arm:{"status":s["terminal"]["status"],"terminal_time_min":s["terminal"]["terminal_time_min"],"infeasibility_type":s["terminal"].get("infeasibility_type"),"interaction_active":bool(s["response"]["interaction_active"]),"admission_count":s["source"]["admission_count"],"demand_count":s["source"]["demand_count"],"route_completion_count":s["route"]["completion_count"],"active_count":s.get("lifecycle",{}).get("active_count"),"mean_source_wait_min":s["source"]["mean_admitted_waiting_time_min"],"rebootstrap_count":s["response"]["rebootstrap_count"],"min_center_separation_nm":None if s.get("safety") is None else s["safety"].get("sampled_min_center_separation_nm"),"source_timeline_changed_vs_neutral":_source_timeline(s)!=_source_timeline(neutral),"post_admission_changed_vs_neutral":_post_signature(s)!=_post_signature(neutral)} for arm,s in arm_docs.items()}
        })
    return _canonical_json({
        "format":P4D_ACTION_LATENCY_ANALYSIS_FORMAT,"schema_version":1,"campaign_id":P4D_ACTION_LATENCY_CAMPAIGN_ID,
        "evidence_role":"PRE_CONFIRMATORY_MODEL_SENSITIVITY_QUALIFICATION_ONLY","scientific_confirmatory_interpretation_authorized":False,
        "plan_sha256":plan.identity_sha256,"source_archive_sha256":plan.source_archive_sha256,"qualification_pass":overall,
        "checks":checks,"regime_summary":summaries,"qualification_rules":qualification_rules(),
    })
