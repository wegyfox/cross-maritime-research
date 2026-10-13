"""R1 P4D interaction-active confirmatory execution plan and point runner.

The independent CROSS target is preserved: no macro/reference headway is forced into the
physical runtime.  The only treatment is the qualified bounded extra action-latency
sensitivity axis applied to HS maneuver-target updates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Mapping, Sequence

from vessel_crossing.nextgen import generate_poisson_network_traffic
from vessel_crossing.research.r1_p4b_behavior_qualification import traffic_input_sha256, typed_demand_for_penetration
from vessel_crossing.research.r1_p4b_publication_execution import build_publication_config
from vessel_crossing.research.r1_p4b_publication_reference import publication_seed

from .r1_p4d_action_latency_sensitivity import ARM_TO_LATENCY_MIN, arm_contract
from .r1_p4d_screening import P4DScreeningCase, run_p4d_screening_case

P4D_CONFIRMATORY_CAMPAIGN_ID = "R1_P4D_INTERACTION_ACTIVE_CONFIRMATORY_V1_2026-09-21"
P4D_CONFIRMATORY_PLAN_FORMAT = "cross-r1-p4d-confirmatory-plan-v1"
P4D_CONFIRMATORY_RESULT_FORMAT = "cross-r1-p4d-confirmatory-point-v1"
P4D_CONFIRMATORY_REPLICATIONS = 24
P4D_CONFIRMATORY_MAX_WORKERS = 2
P4D_CONFIRMATORY_ARMS = tuple(ARM_TO_LATENCY_MIN)
P4D_BALANCED_PENETRATIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
P4D_SENTINEL_PENETRATIONS = (0.0, 0.50, 1.0)
P4D_CONFIRMATORY_REGIMES = {
    "LOW_INTERACTION_CONTROL": {"room_nm": 1.300, "demand_per_hour": 4.0, "penetrations": P4D_SENTINEL_PENETRATIONS, "role": "NEGATIVE_CONTROL"},
    "BALANCED_INTERACTION_ACTIVE": {"room_nm": 1.375, "demand_per_hour": 6.0, "penetrations": P4D_BALANCED_PENETRATIONS, "role": "PRIMARY"},
    "STRESS_SENSITIVITY": {"room_nm": 1.375, "demand_per_hour": 8.0, "penetrations": P4D_SENTINEL_PENETRATIONS, "role": "ROBUSTNESS_SENTINEL"},
}


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _typed_sha(typed: Sequence[object]) -> str:
    rows=[]
    for x in typed:
        if isinstance(x, Mapping):
            rows.append(dict(x))
        else:
            rows.append(asdict(x))
    return _sha(_canonical_json(rows))


def _ptag(p: float) -> str:
    return f"P{int(round(p * 100)):03d}"


@dataclass(frozen=True, slots=True)
class P4DConfirmatoryPoint:
    execution_order: int
    point_id: str
    regime_id: str
    regime_role: str
    arm: str
    replication: int
    room_nm: float
    demand_per_hour: float
    penetration: float
    traffic_seed: int
    type_assignment_seed: int
    behavior_stochasticity_seed: int
    expected_traffic_count: int
    expected_traffic_input_sha256: str
    expected_typed_sha256: str
    expected_contract_sha256: str
    source_archive_sha256: str
    identity_sha256: str = field(default="")

    def identity_payload(self) -> dict[str, object]:
        d = asdict(self)
        d.pop("identity_sha256", None)
        return d

    def compute_identity_sha256(self) -> str:
        return _sha(_canonical_json(self.identity_payload()))

    def __post_init__(self) -> None:
        if self.execution_order < 1:
            raise ValueError("execution_order must be positive")
        if self.regime_id not in P4D_CONFIRMATORY_REGIMES:
            raise ValueError("unknown confirmatory regime")
        if self.arm not in P4D_CONFIRMATORY_ARMS:
            raise ValueError("unknown action-latency arm")
        if self.replication < 1 or self.replication > P4D_CONFIRMATORY_REPLICATIONS:
            raise ValueError("replication outside frozen range")
        if self.penetration not in P4D_CONFIRMATORY_REGIMES[self.regime_id]["penetrations"]:
            raise ValueError("penetration outside frozen regime grid")
        if self.identity_sha256 and self.identity_sha256 != self.compute_identity_sha256():
            raise ValueError("point identity mismatch")


@dataclass(frozen=True, slots=True)
class P4DConfirmatoryPlan:
    source_archive_sha256: str
    protocol_sha256: str
    statistical_plan_sha256: str
    seed_schedule_sha256: str
    qualification_postrun_audit_sha256: str
    points: tuple[P4DConfirmatoryPoint, ...]
    campaign_id: str = P4D_CONFIRMATORY_CAMPAIGN_ID
    replication_count: int = P4D_CONFIRMATORY_REPLICATIONS
    max_parallel_scientific_workers: int = P4D_CONFIRMATORY_MAX_WORKERS
    scientific_execution_authorization: str = "EXTERNAL_AUTHORIZATION_REQUIRED"

    @property
    def document(self) -> dict[str, object]:
        return {
            "format": P4D_CONFIRMATORY_PLAN_FORMAT,
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "source_archive_sha256": self.source_archive_sha256,
            "protocol_sha256": self.protocol_sha256,
            "statistical_plan_sha256": self.statistical_plan_sha256,
            "seed_schedule_sha256": self.seed_schedule_sha256,
            "qualification_postrun_audit_sha256": self.qualification_postrun_audit_sha256,
            "replication_count": self.replication_count,
            "max_parallel_scientific_workers": self.max_parallel_scientific_workers,
            "scientific_execution_authorization": self.scientific_execution_authorization,
            "primary_measurement_window_min": [0.0, 90.0],
            "evaluation_horizon_min": 150.0,
            "primary_service_event": "CONFLICT_CENTER_PASS",
            "regimes": P4D_CONFIRMATORY_REGIMES,
            "action_latency_arms_min": ARM_TO_LATENCY_MIN,
            "points": [dict(p.identity_payload(), identity_sha256=p.compute_identity_sha256()) for p in self.points],
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.document)

    @property
    def identity_sha256(self) -> str:
        return _sha(self.to_json_bytes())


def _seed(replication: int, namespace: str) -> int:
    return publication_seed(campaign_id=P4D_CONFIRMATORY_CAMPAIGN_ID, replication=replication, namespace=namespace)


def build_p4d_confirmatory_plan(*, source_archive_sha256: str, protocol_sha256: str, statistical_plan_sha256: str,
                                seed_schedule_sha256: str, qualification_postrun_audit_sha256: str) -> P4DConfirmatoryPlan:
    points: list[P4DConfirmatoryPoint] = []
    order = 0
    for rep in range(1, P4D_CONFIRMATORY_REPLICATIONS + 1):
        traffic_seed = _seed(rep, "TRAFFIC_ARRIVALS")
        type_seed = _seed(rep, "TYPE_ASSIGNMENT")
        behavior_seed = _seed(rep, "BEHAVIOR_STOCHASTICITY")
        for regime_id, spec in P4D_CONFIRMATORY_REGIMES.items():
            room = float(spec["room_nm"]); demand = float(spec["demand_per_hour"])
            cfg = build_publication_config(room_nm=room, demand_per_hour=demand, traffic_seed=traffic_seed)
            traffic = generate_poisson_network_traffic(cfg.network, cfg.streams, horizon=cfg.scenario.traffic_horizon_min,
                                                       seed=cfg.scenario.seed, start_time=cfg.scenario.start_time)
            tsha = traffic_input_sha256(traffic)
            for penetration in spec["penetrations"]:
                typed = typed_demand_for_penetration(traffic, penetration=float(penetration),
                                                      campaign_id=P4D_CONFIRMATORY_CAMPAIGN_ID, replication=rep)
                tysha = _typed_sha(typed)
                for arm in P4D_CONFIRMATORY_ARMS:
                    order += 1
                    c = arm_contract(arm)
                    p = P4DConfirmatoryPoint(
                        execution_order=order,
                        point_id=f"P4DC-R{rep:02d}-{regime_id}-{_ptag(float(penetration))}-{arm}",
                        regime_id=regime_id,
                        regime_role=str(spec["role"]),
                        arm=arm,
                        replication=rep,
                        room_nm=room,
                        demand_per_hour=demand,
                        penetration=float(penetration),
                        traffic_seed=traffic_seed,
                        type_assignment_seed=type_seed,
                        behavior_stochasticity_seed=behavior_seed,
                        expected_traffic_count=traffic.count,
                        expected_traffic_input_sha256=tsha,
                        expected_typed_sha256=tysha,
                        expected_contract_sha256=c.identity_sha256,
                        source_archive_sha256=source_archive_sha256,
                    )
                    points.append(P4DConfirmatoryPoint(**dict(p.identity_payload(), identity_sha256=p.compute_identity_sha256())))
    if len(points) != 1056:
        raise RuntimeError(f"frozen P4D confirmatory plan must contain 1056 points, got {len(points)}")
    return P4DConfirmatoryPlan(source_archive_sha256, protocol_sha256, statistical_plan_sha256,
                               seed_schedule_sha256, qualification_postrun_audit_sha256, tuple(points))


def plan_from_json_bytes(raw: bytes) -> P4DConfirmatoryPlan:
    d = json.loads(raw)
    if d.get("format") != P4D_CONFIRMATORY_PLAN_FORMAT:
        raise ValueError("unexpected P4D confirmatory plan format")
    pts = []
    for row in d["points"]:
        r = dict(row); pts.append(P4DConfirmatoryPoint(**r))
    plan = P4DConfirmatoryPlan(
        source_archive_sha256=str(d["source_archive_sha256"]),
        protocol_sha256=str(d["protocol_sha256"]),
        statistical_plan_sha256=str(d["statistical_plan_sha256"]),
        seed_schedule_sha256=str(d["seed_schedule_sha256"]),
        qualification_postrun_audit_sha256=str(d["qualification_postrun_audit_sha256"]),
        points=tuple(pts), campaign_id=str(d["campaign_id"]), replication_count=int(d["replication_count"]),
        max_parallel_scientific_workers=int(d["max_parallel_scientific_workers"]),
        scientific_execution_authorization=str(d["scientific_execution_authorization"]),
    )
    if plan.to_json_bytes() != raw:
        raise ValueError("confirmatory plan canonical/semantic identity drift")
    return plan


def _confirmatory_physical_document(screening: Mapping[str, object]) -> dict[str, object]:
    d = json.loads(json.dumps(screening))
    d["evidence_role"] = "R1_P4D_CONFIRMATORY_INDEPENDENT_CROSS_POINT"
    d["scientific_confirmatory_interpretation_authorized"] = True
    return d


def run_p4d_confirmatory_point(plan: P4DConfirmatoryPlan, point: P4DConfirmatoryPoint) -> bytes:
    if point.source_archive_sha256 != plan.source_archive_sha256:
        raise RuntimeError("point/source identity drift")
    contract = arm_contract(point.arm)
    if contract.identity_sha256 != point.expected_contract_sha256:
        raise RuntimeError("contract identity drift")
    case = P4DScreeningCase(
        case_id=point.point_id, room_nm=point.room_nm, demand_per_hour=point.demand_per_hour,
        traffic_seed=point.traffic_seed, replication=point.replication, penetration=point.penetration,
        contract=contract, campaign_id=plan.campaign_id,
    )
    physical = _confirmatory_physical_document(run_p4d_screening_case(case).document)
    if physical["traffic_input_sha256"] != point.expected_traffic_input_sha256:
        raise RuntimeError("traffic identity drift")
    if int(physical["traffic_count"]) != point.expected_traffic_count:
        raise RuntimeError("traffic count drift")
    if _typed_sha(tuple(physical["typed_roles"])) != point.expected_typed_sha256:
        raise RuntimeError("typed-role identity drift")
    doc = {
        "format": P4D_CONFIRMATORY_RESULT_FORMAT,
        "schema_version": 1,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.identity_sha256,
        "source_archive_sha256": plan.source_archive_sha256,
        "point": dict(point.identity_payload(), identity_sha256=point.compute_identity_sha256()),
        "physical": physical,
    }
    return _canonical_json(doc)

