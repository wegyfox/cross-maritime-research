"""Pure configuration/preflight layer for the RC2 interactive wizard.

This module edits the supported diagnostic scenario surface only.  It does not run the
scientific model and it does not duplicate dynamic-physics validation; final domain
validation is delegated to ``analysis`` + ``q5_integration``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from pathlib import Path
from typing import Any, Mapping

import yaml

from vessel_crossing.diagnostic.contract import load_diagnostic_scenario
from vessel_crossing.diagnostic.planning import build_diagnostic_plan
from vessel_crossing.diagnostic.interventions import InterventionDisposition

from .analysis import AnalysisValidation, load_baseline_analysis_scenario, validate_analysis_scenario
from .q5_integration import (
    A69_SPEC_SHA256,
    A70_IMPLEMENTATION_MANIFEST_SHA256,
    prepare_q5_scenario,
)
from .reference_cache import (
    DynamicReferenceCache,
    ReferenceCacheError,
    build_reference_cache_identity,
    default_reference_cache_dir,
)

WIZARD_CONFIG_SCHEMA = "cross-psp-wizard-config-v1"
# Measured A82 qualification-machine calibration, not a promise for another machine.
A82_FRESH_REFERENCE_CALIBRATION_SECONDS = 1699.341
A83_CACHE_HIT_DIAGNOSTIC_CALIBRATION_SECONDS = 21.48


class WizardConfigError(ValueError):
    """Raised when wizard input cannot be represented on the supported RC2 surface."""


@dataclass(frozen=True, slots=True)
class StreamPhysicsInput:
    stream_id: str
    source_to_center_nm: float
    center_to_exit_nm: float
    heading_deg: float
    cruise_speed_knots: float
    maximum_speed_knots: float
    length_m: float
    beam_m: float
    max_acceleration_knots_per_min: float
    max_deceleration_knots_per_min: float
    max_turn_rate_deg_per_min: float
    safety_domain: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrafficInput:
    alpha_rate_per_hour: float = 6.0
    beta_rate_per_hour: float = 6.0
    seed: int = 359
    traffic_horizon_min: float = 15.0
    end_time_min: float = 55.0
    replications: int = 2
    confidence_level: float = 0.95
    seed_namespace: str = "cross-psp-q5-v1"


@dataclass(frozen=True, slots=True)
class WizardPreflight:
    validation: AnalysisValidation
    scope_status: str
    physics_identity_sha256: str
    cache_identity_sha256: str
    verified_reference_cache_hit: bool
    diagnostic_replication_runs: int
    dynamic_reference_runs_if_cache_miss: int
    estimated_wall_clock_seconds: tuple[float, float]
    estimate_basis: str


def _finite_positive(name: str, value: float) -> float:
    out = float(value)
    if not isfinite(out) or not (out > 0.0):
        raise WizardConfigError(f"{name} must be positive")
    return out


def _heading_unit(heading_deg: float) -> tuple[float, float]:
    rad = float(heading_deg) * pi / 180.0
    return sin(rad), cos(rad)


def _snap_zero(value: float, *, tolerance: float = 1e-12) -> float:
    out = float(value)
    return 0.0 if abs(out) <= tolerance else out


def _point(distance_nm: float, heading_deg: float, *, source: bool) -> list[float]:
    ux, uy = _heading_unit(heading_deg)
    sign = -1.0 if source else 1.0
    distance = float(distance_nm)
    return [_snap_zero(sign * distance * ux), _snap_zero(sign * distance * uy)]


def _stream_by_id(mapping: dict[str, Any], stream_id: str) -> dict[str, Any]:
    streams = mapping["model"]["parameters"]["streams"]
    rows = [row for row in streams if row.get("stream_id") == stream_id]
    if len(rows) != 1:
        raise WizardConfigError(f"baseline must contain exactly one stream {stream_id!r}")
    return rows[0]


def _edge_by_id(mapping: dict[str, Any], edge_id: str) -> dict[str, Any]:
    edges = mapping["model"]["parameters"]["network"]["edges"]
    rows = [row for row in edges if row.get("edge_id") == edge_id]
    if len(rows) != 1:
        raise WizardConfigError(f"baseline must contain exactly one edge {edge_id!r}")
    return rows[0]


def _node_by_id(mapping: dict[str, Any], node_id: str) -> dict[str, Any]:
    nodes = mapping["model"]["parameters"]["network"]["nodes"]
    rows = [row for row in nodes if row.get("node_id") == node_id]
    if len(rows) != 1:
        raise WizardConfigError(f"baseline must contain exactly one node {node_id!r}")
    return rows[0]


def _apply_traffic(mapping: dict[str, Any], traffic: TrafficInput) -> None:
    alpha = _stream_by_id(mapping, "alpha")
    beta = _stream_by_id(mapping, "beta")
    alpha["rate_per_hour"] = _finite_positive("alpha rate_per_hour", traffic.alpha_rate_per_hour)
    beta["rate_per_hour"] = _finite_positive("beta rate_per_hour", traffic.beta_rate_per_hour)
    scenario = mapping["model"]["parameters"]["scenario"]
    scenario["seed"] = int(traffic.seed)
    scenario["traffic_horizon_min"] = _finite_positive("traffic_horizon_min", traffic.traffic_horizon_min)
    scenario["end_time"] = _finite_positive("end_time", traffic.end_time_min)
    experiment = mapping["experiment"]
    experiment["replications"] = int(traffic.replications)
    experiment["confidence_level"] = float(traffic.confidence_level)
    experiment["seed_namespace"] = str(traffic.seed_namespace)


def build_traffic_scenario(traffic: TrafficInput) -> dict[str, Any]:
    mapping = deepcopy(load_baseline_analysis_scenario())
    _apply_traffic(mapping, traffic)
    return mapping


def _apply_stream_physics(mapping: dict[str, Any], spec: StreamPhysicsInput) -> None:
    stream_id = spec.stream_id
    stream = _stream_by_id(mapping, stream_id)
    edge_id = "ALPHA" if stream_id == "alpha" else "BETA"
    edge = _edge_by_id(mapping, edge_id)
    source_node = _node_by_id(mapping, edge["from_node_id"])
    exit_node = _node_by_id(mapping, edge["to_node_id"])
    source = _point(_finite_positive("source_to_center_nm", spec.source_to_center_nm), spec.heading_deg, source=True)
    exit_ = _point(_finite_positive("center_to_exit_nm", spec.center_to_exit_nm), spec.heading_deg, source=False)
    source_node["position_nm"] = source
    exit_node["position_nm"] = exit_
    edge["waypoints_nm"] = [source, exit_]

    class_id = f"wizard-{stream_id}"
    stream["vessel_class_id"] = class_id
    stream["cruise_speed_knots"] = _finite_positive("cruise_speed_knots", spec.cruise_speed_knots)
    mapping["model"]["parameters"]["vessel_classes"][class_id] = {
        "dimensions_m": {
            "length_m": _finite_positive("length_m", spec.length_m),
            "beam_m": _finite_positive("beam_m", spec.beam_m),
        },
        "maneuver_limits": {
            "max_acceleration_knots_per_min": _finite_positive(
                "max_acceleration_knots_per_min", spec.max_acceleration_knots_per_min
            ),
            "max_deceleration_knots_per_min": _finite_positive(
                "max_deceleration_knots_per_min", spec.max_deceleration_knots_per_min
            ),
            "max_turn_rate_deg_per_min": _finite_positive(
                "max_turn_rate_deg_per_min", spec.max_turn_rate_deg_per_min
            ),
        },
        "safety_domain": dict(spec.safety_domain),
        "maximum_speed_knots": _finite_positive("maximum_speed_knots", spec.maximum_speed_knots),
    }


def build_physics_scenario(
    traffic: TrafficInput,
    alpha: StreamPhysicsInput,
    beta: StreamPhysicsInput,
    *,
    alpha_b6_share: float = 0.5,
) -> dict[str, Any]:
    if alpha.stream_id != "alpha" or beta.stream_id != "beta":
        raise WizardConfigError("physics inputs must be alpha then beta")
    share = float(alpha_b6_share)
    if not (0.0 < share < 1.0):
        raise WizardConfigError("alpha B6 share must be strictly between 0 and 1")
    mapping = build_traffic_scenario(traffic)
    _apply_stream_physics(mapping, alpha)
    _apply_stream_physics(mapping, beta)
    classes = mapping["model"]["parameters"]["vessel_classes"]
    alpha_class = classes["wizard-alpha"]
    beta_class = classes["wizard-beta"]
    baseline_class = classes.get("q5")
    if alpha_class == beta_class:
        shared_id = "q5" if alpha_class == baseline_class else "wizard-shared"
        classes[shared_id] = alpha_class
        _stream_by_id(mapping, "alpha")["vessel_class_id"] = shared_id
        _stream_by_id(mapping, "beta")["vessel_class_id"] = shared_id
        active = {shared_id}
    else:
        active = {"wizard-alpha", "wizard-beta"}
    mapping["model"]["parameters"]["vessel_classes"] = {
        key: value for key, value in classes.items() if key in active
    }
    if share == 0.5:
        mapping["metadata"].pop("b6_target_stream_shares", None)
    else:
        mapping["metadata"]["b6_target_stream_shares"] = {
            "alpha": share,
            "beta": 1.0 - share,
        }
    return mapping


def write_scenario(mapping: Mapping[str, Any], destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(dict(mapping), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def preflight_scenario(path: Path, *, cache_dir: Path | None = None) -> WizardPreflight:
    validation = validate_analysis_scenario(path)
    scenario = load_diagnostic_scenario(path)
    prepared = prepare_q5_scenario(scenario)
    plan = build_diagnostic_plan(scenario)
    applicable = sum(
        1 for entry in plan.entries if entry.disposition is InterventionDisposition.APPLICABLE
    )
    # B0 + each applicable physical/controller level is simulated per replication.
    diagnostic_runs = (1 + applicable) * int(plan.replications)

    identity = build_reference_cache_identity(
        physics_identity_sha256=prepared.physical.physics_identity_sha256,
        profile=prepared.physical.profile,
        runtime=prepared.reference_runtime,
        b6_spec_sha256=A69_SPEC_SHA256,
        implementation_manifest_sha256=A70_IMPLEMENTATION_MANIFEST_SHA256,
    )
    cache = DynamicReferenceCache(cache_dir or default_reference_cache_dir())
    try:
        cache_hit = cache.load(identity) is not None
    except (ReferenceCacheError, OSError):
        cache_hit = False

    # Calibration scaling is intentionally simple and explicit.  The diagnostic calibration
    # was measured at 2 reps / 6 simulated B0+B4+B5 runs.  It is not extrapolated by horizon.
    diag_seconds = A83_CACHE_HIT_DIAGNOSTIC_CALIBRATION_SECONDS * max(1.0, diagnostic_runs / 6.0)
    if cache_hit:
        lo, hi = diag_seconds * 0.7, diag_seconds * 2.0
        basis = "A83 cache-hit diagnostic calibration; machine/scenario dependent"
        ref_runs = 0
    else:
        ref = A82_FRESH_REFERENCE_CALIBRATION_SECONDS
        lo, hi = diag_seconds * 0.7 + ref * 0.8, diag_seconds * 2.0 + ref * 1.5
        basis = "A82 fresh-reference + A83 diagnostic calibration; machine/scenario dependent"
        ref_runs = 6
    return WizardPreflight(
        validation=validation,
        scope_status=prepared.physical.scope_status,
        physics_identity_sha256=prepared.physical.physics_identity_sha256,
        cache_identity_sha256=identity.sha256,
        verified_reference_cache_hit=cache_hit,
        diagnostic_replication_runs=diagnostic_runs,
        dynamic_reference_runs_if_cache_miss=ref_runs,
        estimated_wall_clock_seconds=(lo, hi),
        estimate_basis=basis,
    )
