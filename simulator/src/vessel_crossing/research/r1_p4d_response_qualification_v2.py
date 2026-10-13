"""Pre-freeze response-mechanism qualification campaign V2 for R1 P4D.

This is intentionally *not* a confirmatory scientific campaign.  It exercises the
candidate bounded-action-latency mechanism in three neutral-selected physical regimes
before any P4D protocol/source freeze is allowed.

The six points are paired by regime: NEUTRAL vs CANDIDATE at P50 on an identical traffic
realization and identical nested AS/HS assignment.  Qualification rules are frozen here
before balanced/stress candidate results are inspected.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping, Sequence

from vessel_crossing.nextgen import generate_poisson_network_traffic
from vessel_crossing.research.r1_p4b_behavior_qualification import (
    traffic_input_sha256,
    typed_demand_for_penetration,
)
from vessel_crossing.research.r1_p4b_publication_execution import build_publication_config

from .r1_p4d_response_hypothesis import candidate_response_contract
from .r1_p4d_screening import (
    P4DScreeningCase,
    qualification_contracts,
    run_p4d_screening_case,
)

P4D_RESPONSE_QUALIFICATION_FORMAT = "cross-r1-p4d-response-qualification-plan-v2"
P4D_RESPONSE_QUALIFICATION_RESULT_FORMAT = "cross-r1-p4d-response-qualification-result-v2"
P4D_RESPONSE_QUALIFICATION_ANALYSIS_FORMAT = "cross-r1-p4d-response-qualification-analysis-v2"
P4D_RESPONSE_QUALIFICATION_CAMPAIGN_ID = "R1_P4D_RESPONSE_QUALIFICATION_V2_2026-09-21"
P4D_RESPONSE_QUALIFICATION_PENETRATION = 0.5
P4D_RESPONSE_QUALIFICATION_MAX_WORKERS = 2
_EPS = 1e-9


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _typed_sha(typed: Sequence[object]) -> str:
    return _sha(_canonical_json([asdict(x) for x in typed]))


@dataclass(frozen=True, slots=True)
class P4DResponseQualificationPoint:
    execution_order: int
    point_id: str
    regime_id: str
    arm: str
    replication: int
    room_nm: float
    demand_per_hour: float
    traffic_seed: int
    traffic_count: int
    p50_as_count: int
    p50_hs_count: int
    expected_traffic_input_sha256: str
    expected_typed_sha256: str
    expected_contract_sha256: str
    source_archive_sha256: str

    def __post_init__(self) -> None:
        if self.execution_order <= 0:
            raise ValueError("execution_order must be positive")
        if self.regime_id not in {
            "LOW_INTERACTION_CONTROL", "BALANCED_INTERACTION_ACTIVE", "STRESS_SENSITIVITY"
        }:
            raise ValueError("unsupported regime_id")
        if self.arm not in {"NEUTRAL", "CANDIDATE"}:
            raise ValueError("arm must be NEUTRAL or CANDIDATE")
        if self.traffic_count <= 0 or self.p50_as_count <= 0 or self.p50_hs_count <= 0:
            raise ValueError("qualification points require a non-empty mixed P50 realization")
        if self.p50_as_count + self.p50_hs_count != self.traffic_count:
            raise ValueError("role counts must equal traffic_count")

    @property
    def identity_payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def identity_sha256(self) -> str:
        return _sha(_canonical_json(self.identity_payload))


@dataclass(frozen=True, slots=True)
class P4DResponseQualificationPlan:
    source_archive_sha256: str
    fixture_selection_sha256: str
    response_hypothesis_sha256: str
    points: tuple[P4DResponseQualificationPoint, ...]

    @property
    def document(self) -> dict[str, object]:
        return {
            "format": P4D_RESPONSE_QUALIFICATION_FORMAT,
            "schema_version": 1,
            "campaign_id": P4D_RESPONSE_QUALIFICATION_CAMPAIGN_ID,
            "evidence_role": "PRE_FREEZE_QUALIFICATION_ONLY_NOT_CONFIRMATORY",
            "scientific_confirmatory_interpretation_authorized": False,
            "penetration": P4D_RESPONSE_QUALIFICATION_PENETRATION,
            "max_parallel_workers": P4D_RESPONSE_QUALIFICATION_MAX_WORKERS,
            "source_archive_sha256": self.source_archive_sha256,
            "fixture_selection_sha256": self.fixture_selection_sha256,
            "response_hypothesis_sha256": self.response_hypothesis_sha256,
            "qualification_rules": qualification_rules(),
            "points": [dict(x.identity_payload, identity_sha256=x.identity_sha256) for x in self.points],
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.document)

    @property
    def identity_sha256(self) -> str:
        return _sha(self.to_json_bytes())


def qualification_rules() -> dict[str, object]:
    """V2 rules frozen after V1 failure and before any fresh V2 treated execution."""
    return {
        "all_points": [
            "exactly six fresh planned points must be present",
            "terminal status must be SUCCESS at the full 150-min horizon",
            "traffic input and typed P50 assignment must match the frozen V2 plan",
            "all generated demands must be admitted by the evaluation horizon",
            "no modeled hull-overlap episode is allowed",
            "terminal lifecycle must conserve every admitted vessel",
            "every still-active admitted vessel must have exactly one live owner",
            "route completion by 150 is NOT required; active in-system backlog is retained as an outcome",
        ],
        "low_control": [
            "NEUTRAL and CANDIDATE must both remain interaction_active=false",
            "source admission timeline must be physically identical between arms",
            "route-duration records must be physically identical between arms",
            "safety summary must be physically identical between arms",
            "terminal lifecycle audit must be physically identical between arms",
        ],
        "balanced_and_stress": [
            "NEUTRAL must be interaction_active=true",
            "CANDIDATE must remain interaction_active=true (no interaction collapse by source pre-metering)",
            "both arms must retain at least one CENTER_PASS passage",
            "CANDIDATE must change at least one post-admission physical metric relative to NEUTRAL",
            "active in-system vessels at 150 are admissible only when lifecycle/ownership conservation passes",
        ],
        "post_admission_activation_metrics": [
            "route records", "conflict gate observations", "rebootstrap count",
            "sampled minimum center separation", "center-service records", "terminal lifecycle state",
        ],
        "directional_effect_requirement": "NONE; sign/magnitude is not a qualification criterion",
        "v1_status": "V1 remains FAILED; V2 does not reinterpret V1 results",
    }


def _contract_for_arm(arm: str):
    if arm == "NEUTRAL":
        return qualification_contracts()["NEUTRAL"]
    if arm == "CANDIDATE":
        return candidate_response_contract()
    raise KeyError(arm)


def build_response_qualification_plan(
    fixture_selection: Mapping[str, object],
    *,
    source_archive_sha256: str,
    fixture_selection_sha256: str,
    response_hypothesis_sha256: str,
) -> P4DResponseQualificationPlan:
    fixtures = fixture_selection.get("fixtures")
    if not isinstance(fixtures, Sequence) or len(fixtures) != 3:
        raise ValueError("qualification fixture selection must contain exactly three fixtures")
    points: list[P4DResponseQualificationPoint] = []
    order = 0
    for fixture in fixtures:
        if not isinstance(fixture, Mapping):
            raise TypeError("fixture rows must be mappings")
        rep = int(fixture["replication"]); room = float(fixture["room_nm"]); demand = float(fixture["demand_per_hour"])
        seed = int(fixture["traffic_seed"]); regime = str(fixture["regime_id"])
        cfg = build_publication_config(room_nm=room, demand_per_hour=demand, traffic_seed=seed)
        traffic = generate_poisson_network_traffic(
            cfg.network, cfg.streams,
            horizon=cfg.scenario.traffic_horizon_min,
            seed=cfg.scenario.seed,
            start_time=cfg.scenario.start_time,
        )
        typed = typed_demand_for_penetration(
            traffic,
            penetration=P4D_RESPONSE_QUALIFICATION_PENETRATION,
            campaign_id=P4D_RESPONSE_QUALIFICATION_CAMPAIGN_ID,
            replication=rep,
        )
        as_count = sum(x.vessel_type == "AS" for x in typed)
        hs_count = sum(x.vessel_type == "HS" for x in typed)
        if as_count <= 0 or hs_count <= 0:
            raise RuntimeError(f"qualification campaign lost mixed P50 assignment for {regime}")
        for arm in ("NEUTRAL", "CANDIDATE"):
            order += 1
            contract = _contract_for_arm(arm)
            points.append(P4DResponseQualificationPoint(
                execution_order=order,
                point_id=f"P4DQ-{regime}-R{rep:02d}-P050-{arm}",
                regime_id=regime,
                arm=arm,
                replication=rep,
                room_nm=room,
                demand_per_hour=demand,
                traffic_seed=seed,
                traffic_count=traffic.count,
                p50_as_count=as_count,
                p50_hs_count=hs_count,
                expected_traffic_input_sha256=traffic_input_sha256(traffic),
                expected_typed_sha256=_typed_sha(typed),
                expected_contract_sha256=contract.identity_sha256,
                source_archive_sha256=source_archive_sha256,
            ))
    return P4DResponseQualificationPlan(
        source_archive_sha256=source_archive_sha256,
        fixture_selection_sha256=fixture_selection_sha256,
        response_hypothesis_sha256=response_hypothesis_sha256,
        points=tuple(points),
    )


def plan_from_json_bytes(raw: bytes) -> P4DResponseQualificationPlan:
    d = json.loads(raw)
    if d.get("format") != P4D_RESPONSE_QUALIFICATION_FORMAT:
        raise ValueError("unexpected P4D response qualification plan format")
    pts=[]
    for item in d["points"]:
        row=dict(item); row.pop("identity_sha256",None)
        pts.append(P4DResponseQualificationPoint(**row))
    plan=P4DResponseQualificationPlan(
        source_archive_sha256=str(d["source_archive_sha256"]),
        fixture_selection_sha256=str(d["fixture_selection_sha256"]),
        response_hypothesis_sha256=str(d["response_hypothesis_sha256"]),
        points=tuple(pts),
    )
    # Reject semantic drift in any field/rule/order.
    if plan.to_json_bytes() != raw:
        raise ValueError("qualification plan canonical/semantic identity drift")
    return plan


def run_response_qualification_point(
    plan: P4DResponseQualificationPlan,
    point: P4DResponseQualificationPoint,
) -> bytes:
    if point.source_archive_sha256 != plan.source_archive_sha256:
        raise RuntimeError("point/source identity mismatch")
    contract=_contract_for_arm(point.arm)
    if contract.identity_sha256 != point.expected_contract_sha256:
        raise RuntimeError("qualification contract identity drift")
    case=P4DScreeningCase(
        case_id=point.point_id,
        room_nm=point.room_nm,
        demand_per_hour=point.demand_per_hour,
        traffic_seed=point.traffic_seed,
        replication=point.replication,
        penetration=P4D_RESPONSE_QUALIFICATION_PENETRATION,
        contract=contract,
        campaign_id=P4D_RESPONSE_QUALIFICATION_CAMPAIGN_ID,
    )
    screening=run_p4d_screening_case(case)
    sd=screening.document
    if sd["traffic_input_sha256"] != point.expected_traffic_input_sha256:
        raise RuntimeError("qualification traffic identity drift")
    typed_sha=_sha(_canonical_json(sd["typed_roles"]))
    if typed_sha != point.expected_typed_sha256:
        raise RuntimeError("qualification typed-role identity drift")
    out={
        "format": P4D_RESPONSE_QUALIFICATION_RESULT_FORMAT,
        "schema_version": 1,
        "campaign_id": P4D_RESPONSE_QUALIFICATION_CAMPAIGN_ID,
        "evidence_role": "PRE_FREEZE_QUALIFICATION_ONLY_NOT_CONFIRMATORY",
        "scientific_confirmatory_interpretation_authorized": False,
        "plan_sha256": plan.identity_sha256,
        "source_archive_sha256": plan.source_archive_sha256,
        "point": dict(point.identity_payload, identity_sha256=point.identity_sha256),
        "screening_result_sha256": screening.result_sha256,
        "screening": sd,
    }
    return _canonical_json(out)


def _physical_source_timeline(screening: Mapping[str, object]) -> object:
    return screening["source"]["admissions"]


def _post_admission_signature(screening: Mapping[str, object]) -> dict[str, object]:
    safety=screening.get("safety")
    return {
        "route_records": screening["route"]["records"],
        "conflict_gate_rows": screening["conflict"]["gate_rows"],
        "rebootstrap_count": screening["response"]["rebootstrap_count"],
        "sampled_min_center_separation_nm": None if safety is None else safety.get("sampled_min_center_separation_nm"),
        "center_service_records": screening["center_service"]["records"],
        "terminal_lifecycle": screening.get("lifecycle"),
    }


def analyze_response_qualification_results(
    plan: P4DResponseQualificationPlan,
    result_documents: Sequence[Mapping[str, object]],
) -> bytes:
    expected={p.point_id:p for p in plan.points}
    if len(result_documents) != len(expected):
        raise ValueError(f"qualification requires {len(expected)} results, got {len(result_documents)}")
    by_id={}
    for doc in result_documents:
        if doc.get("format") != P4D_RESPONSE_QUALIFICATION_RESULT_FORMAT:
            raise ValueError("unexpected qualification result format")
        if doc.get("plan_sha256") != plan.identity_sha256 or doc.get("source_archive_sha256") != plan.source_archive_sha256:
            raise ValueError("qualification result plan/source identity mismatch")
        point=doc.get("point")
        if not isinstance(point, Mapping):
            raise ValueError("qualification result missing point")
        pid=str(point["point_id"])
        if pid in by_id or pid not in expected:
            raise ValueError(f"duplicate or unexpected point {pid}")
        if point.get("identity_sha256") != expected[pid].identity_sha256:
            raise ValueError(f"point identity drift for {pid}")
        by_id[pid]=doc

    checks=[]
    overall=True
    paired=[]
    for regime in ("LOW_INTERACTION_CONTROL","BALANCED_INTERACTION_ACTIVE","STRESS_SENSITIVITY"):
        p_neutral=next(p for p in plan.points if p.regime_id==regime and p.arm=="NEUTRAL")
        p_candidate=next(p for p in plan.points if p.regime_id==regime and p.arm=="CANDIDATE")
        n=by_id[p_neutral.point_id]["screening"]; c=by_id[p_candidate.point_id]["screening"]
        # Common hard gates.
        local=[]
        for arm_name,s,p in (("NEUTRAL",n,p_neutral),("CANDIDATE",c,p_candidate)):
            terminal=s["terminal"]
            safety=s.get("safety")
            hard={
                "terminal_success_full_horizon": terminal["status"]=="SUCCESS" and abs(float(terminal["terminal_time_min"])-150.0)<=_EPS,
                "traffic_identity": s["traffic_input_sha256"]==p.expected_traffic_input_sha256,
                "typed_identity": _sha(_canonical_json(s["typed_roles"]))==p.expected_typed_sha256,
                "all_demands_admitted": int(s["source"]["admission_count"])==int(s["source"]["demand_count"]),
                "no_hull_overlap": safety is None or int(safety["modeled_hull_overlap_episode_count"])==0,
                "lifecycle_conserved": bool(s.get("lifecycle",{}).get("lifecycle_conserved",False)),
                "ownership_conserved": bool(s.get("lifecycle",{}).get("ownership_conserved",False)),
                "all_admitted_accounted_for": bool(s.get("lifecycle",{}).get("all_admitted_accounted_for",False)),
            }
            for name,passed in hard.items():
                checks.append({"regime_id":regime,"arm":arm_name,"check":name,"pass":bool(passed)})
                local.append(bool(passed))
        typed_same=n["typed_roles"]==c["typed_roles"]
        checks.append({"regime_id":regime,"arm":"PAIR","check":"paired_typed_roles_identical","pass":typed_same})
        local.append(typed_same)

        n_active=bool(n["response"]["interaction_active"]); c_active=bool(c["response"]["interaction_active"])
        post_n=_post_admission_signature(n); post_c=_post_admission_signature(c)
        downstream_changed=post_n != post_c
        if regime=="LOW_INTERACTION_CONTROL":
            special={
                "neutral_interaction_inactive": not n_active,
                "candidate_interaction_inactive": not c_active,
                "source_timeline_identical": _physical_source_timeline(n)==_physical_source_timeline(c),
                "route_records_identical": n["route"]["records"]==c["route"]["records"],
                "safety_identical": n.get("safety")==c.get("safety"),
                "terminal_lifecycle_identical": n.get("lifecycle")==c.get("lifecycle"),
            }
        else:
            special={
                "neutral_interaction_active": n_active,
                "candidate_interaction_active": c_active,
                "neutral_center_passage_present": int(n["center_service"]["passage_count"])>0,
                "candidate_center_passage_present": int(c["center_service"]["passage_count"])>0,
                "post_admission_physics_changed": downstream_changed,
            }
        for name,passed in special.items():
            checks.append({"regime_id":regime,"arm":"PAIR","check":name,"pass":bool(passed)})
            local.append(bool(passed))
        regime_pass=all(local)
        overall=overall and regime_pass
        paired.append({
            "regime_id":regime,
            "qualification_pass":regime_pass,
            "neutral_interaction_active":n_active,
            "candidate_interaction_active":c_active,
            "source_timeline_changed":_physical_source_timeline(n)!=_physical_source_timeline(c),
            "post_admission_physics_changed":downstream_changed,
            "neutral_mean_source_wait_min":n["source"]["mean_admitted_waiting_time_min"],
            "candidate_mean_source_wait_min":c["source"]["mean_admitted_waiting_time_min"],
            "neutral_mean_route_extra_min":n["route"]["mean_route_extra_min"],
            "candidate_mean_route_extra_min":c["route"]["mean_route_extra_min"],
            "neutral_rebootstrap_count":n["response"]["rebootstrap_count"],
            "candidate_rebootstrap_count":c["response"]["rebootstrap_count"],
            "neutral_min_center_separation_nm":None if n.get("safety") is None else n["safety"]["sampled_min_center_separation_nm"],
            "candidate_min_center_separation_nm":None if c.get("safety") is None else c["safety"]["sampled_min_center_separation_nm"],
            "neutral_route_completion_count":n["route"]["completion_count"],
            "candidate_route_completion_count":c["route"]["completion_count"],
            "neutral_terminal_active_count":n.get("lifecycle",{}).get("active_count"),
            "candidate_terminal_active_count":c.get("lifecycle",{}).get("active_count"),
        })
    out={
        "format":P4D_RESPONSE_QUALIFICATION_ANALYSIS_FORMAT,
        "schema_version":1,
        "campaign_id":P4D_RESPONSE_QUALIFICATION_CAMPAIGN_ID,
        "evidence_role":"PRE_FREEZE_QUALIFICATION_ONLY_NOT_CONFIRMATORY",
        "scientific_confirmatory_interpretation_authorized":False,
        "plan_sha256":plan.identity_sha256,
        "source_archive_sha256":plan.source_archive_sha256,
        "qualification_pass":overall,
        "checks":checks,
        "paired_regime_summary":paired,
        "qualification_rules":qualification_rules(),
    }
    return _canonical_json(out)
