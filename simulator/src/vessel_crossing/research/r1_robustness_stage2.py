"""Stage-2 plant-fidelity sensitivity design over the closed Stage-1 V4 runtime.

Stage-2A intentionally fixes vessel geometry at the original 120 x 20 m engineering
baseline and varies the four already-frozen operational-domain policies against two
synthetic response plants.  The Stage-1 V4 A162 outcomes are references, not new
executions.  This avoids implying that one synthetic manoeuvring profile is calibrated
or geometrically scalable across the 90--240 m Stage-1 hull sweep.

All richer plants in this module are MODEL-FORM SENSITIVITY ONLY.  In particular the
3.5 kn steerage point encodes the mariner-informed qualitative low-speed concern supplied
for this project; it is not asserted to be a universal or experimentally validated
threshold for all ships.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from typing import Any

from vessel_crossing.nextgen import EnvironmentalNetworkSourceFutureIngressConfig
from vessel_crossing.nextgen.environmental_joint_exact_search import EnvironmentalJointExactSearchConfig
from vessel_crossing.nextgen.rule13_progress_coordination import Rule13ProgressCoordinationPolicy
from vessel_crossing.testbed.plant_fidelity import (
    FirstOrderResponseParameters,
    FirstOrderResponsePlantAdapter,
    SteerageAuthorityPoint,
    TabulatedSteerageResponseParameters,
    TabulatedSteerageResponsePlantAdapter,
)

from .environmental_plant_fidelity import ResponseLagEnvironmentalMotionBackend
from .r1_robustness_matrix import RobustnessStage1Point, run_stage1_point, stage1_design

STAGE2A_EVIDENCE_ROLE = "ENGINEERING_PLANT_MODEL_FORM_SENSITIVITY_NOT_CALIBRATED_VESSEL_CLAIM"
STAGE2A_GEOMETRY_ID = "G120x020"
STAGE2A_PREDICTION_HORIZON_MIN = 12.0
STAGE2A_TRAJECTORY_CACHE_MAX_ENTRIES = 256


@dataclass(frozen=True, slots=True)
class Stage2APlantProfile:
    plant_id: str
    family: str
    parameters: dict[str, Any]
    evidence_role: str = STAGE2A_EVIDENCE_ROLE

    @property
    def identity_sha256(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Stage2APoint:
    point_id: str
    stage1_point_id: str
    domain_policy_id: str
    plant: Stage2APlantProfile

    @property
    def identity_sha256(self) -> str:
        payload = {
            "point_id": self.point_id,
            "stage1_point_id": self.stage1_point_id,
            "domain_policy_id": self.domain_policy_id,
            "plant_sha256": self.plant.identity_sha256,
            "evidence_role": STAGE2A_EVIDENCE_ROLE,
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def first_order_profile() -> Stage2APlantProfile:
    p = FirstOrderResponseParameters(
        yaw_time_constant_min=0.50,
        speed_time_constant_min=1.00,
        heading_gain_per_min=1.00,
        max_step_min=0.02,
    )
    return Stage2APlantProfile("P1_FIRST_ORDER_RESPONSE_SYNTHETIC", "FIRST_ORDER_RESPONSE", asdict(p))


def tabulated_steerage_profile() -> Stage2APlantProfile:
    p = TabulatedSteerageResponseParameters(
        steerage_curve=(
            SteerageAuthorityPoint(0.0, 0.0),
            SteerageAuthorityPoint(3.5, 0.0),
            SteerageAuthorityPoint(6.0, 0.5),
            SteerageAuthorityPoint(10.0, 1.0),
        ),
        steering_response_delay_min=0.20,
        yaw_time_constant_min=0.50,
        speed_time_constant_min=1.20,
        heading_gain_per_min=1.00,
        max_step_min=0.02,
    )
    payload = asdict(p)
    payload["steerage_curve"] = [asdict(x) for x in p.steerage_curve]
    return Stage2APlantProfile("P2_TABULATED_STEERAGE_SYNTHETIC", "TABULATED_STEERAGE_RESPONSE", payload)


def stage2a_profiles() -> tuple[Stage2APlantProfile, ...]:
    return (first_order_profile(), tabulated_steerage_profile())


def _baseline_geometry_points() -> tuple[RobustnessStage1Point, ...]:
    return tuple(p for p in stage1_design() if p.geometry_id == STAGE2A_GEOMETRY_ID)


def stage2a_design() -> tuple[Stage2APoint, ...]:
    out = []
    for p in sorted(_baseline_geometry_points(), key=lambda x: x.domain_policy_id):
        for plant in stage2a_profiles():
            out.append(Stage2APoint(
                point_id=f"R1ROB-S2A-{p.domain_policy_id}-{plant.plant_id}",
                stage1_point_id=p.point_id,
                domain_policy_id=p.domain_policy_id,
                plant=plant,
            ))
    return tuple(out)


def stage2a_design_sha256() -> str:
    return sha256(json.dumps([
        {"point_id": p.point_id, "point_sha256": p.identity_sha256} for p in stage2a_design()
    ], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stage1_point_for(point: Stage2APoint) -> RobustnessStage1Point:
    return next(p for p in _baseline_geometry_points() if p.point_id == point.stage1_point_id)


def build_v4_config(point: RobustnessStage1Point, base_builder) -> Any:
    """Apply the immutable V4 execution method to a materialized Stage-1 point."""
    cfg = base_builder(point)
    sc = cfg.scenario
    rc = sc.recapture
    jc = rc.joint_cycle
    rp = jc.replanning
    sem = replace(
        rp.semantic_config,
        rule13_progress_coordination=Rule13ProgressCoordinationPolicy.FRONTIER_DEPENDENCIES,
    )
    rp = replace(
        rp,
        semantic_config=sem,
        exact_search_config=EnvironmentalJointExactSearchConfig(
            max_pair_evidence=10_000,
            max_branch_nodes=10_000,
        ),
    )
    jc = replace(jc, replanning=rp)
    rc = replace(rc, joint_cycle=jc)
    sc = replace(sc, recapture=rc)
    return replace(cfg, scenario=sc, label=point.point_id + "-S2A-V4-runtime")


def backend_for(profile: Stage2APlantProfile) -> ResponseLagEnvironmentalMotionBackend:
    if profile.family == "FIRST_ORDER_RESPONSE":
        p = FirstOrderResponseParameters(**profile.parameters)
        plant = FirstOrderResponsePlantAdapter(p)
    elif profile.family == "TABULATED_STEERAGE_RESPONSE":
        raw = dict(profile.parameters)
        raw["steerage_curve"] = tuple(SteerageAuthorityPoint(**x) for x in raw["steerage_curve"])
        p = TabulatedSteerageResponseParameters(**raw)
        plant = TabulatedSteerageResponsePlantAdapter(p)
    else:
        raise ValueError(f"unsupported Stage-2A plant family: {profile.family}")
    return ResponseLagEnvironmentalMotionBackend(
        plant,
        default_prediction_horizon_min=STAGE2A_PREDICTION_HORIZON_MIN,
        trajectory_cache_max_entries=STAGE2A_TRAJECTORY_CACHE_MAX_ENTRIES,
    )


def run_stage2a_point(point: Stage2APoint, *, base_builder) -> dict[str, Any]:
    stage1_point = stage1_point_for(point)
    cfg = build_v4_config(stage1_point, base_builder)
    pair = cfg.scenario.recapture.joint_cycle.replanning.semantic_config.physics.pair_clearance
    guard = EnvironmentalNetworkSourceFutureIngressConfig(pair_clearance=pair)
    backend = backend_for(point.plant)
    result = run_stage1_point(
        stage1_point,
        future_ingress_clearance=guard,
        motion_backend=backend,
        materialized_config=cfg,
        bounded_safety_telemetry=True,
    )
    return {
        "stage2a_evidence_role": STAGE2A_EVIDENCE_ROLE,
        "stage2a_point_id": point.point_id,
        "stage2a_point_sha256": point.identity_sha256,
        "stage2a_design_sha256": stage2a_design_sha256(),
        "plant_id": point.plant.plant_id,
        "plant_sha256": point.plant.identity_sha256,
        "motion_backend_id": backend.backend_id,
        "v4_config_sha256": cfg.config_sha256,
        "safety_telemetry_mode": "ONLINE_BOUNDED_EXACT_SUMMARY",
        "trajectory_cache_max_entries": STAGE2A_TRAJECTORY_CACHE_MAX_ENTRIES,
        "stage1_result": result,
    }
