"""Production A93 work-unit runtime adapter for public robustness campaigns.

The adapter is intentionally narrow.  A89 work-unit parameters are applied only through
an explicit versioned binding vocabulary, then executed through the existing A88/Q5
registries.  No generic object/YAML path mutation is accepted.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from vessel_crossing.diagnostic.contract import load_diagnostic_scenario
from vessel_crossing.diagnostic.execution import run_diagnostic_orchestrator
from vessel_crossing.nextgen.reference_envelope import ReferenceEnvelopeLevel

from .custom_physics_binding import (
    BoundCustomPhysicsScenario,
    bind_distribution_scenario,
    bind_fixed_scenario,
    bind_fleet_mixture_scenario,
    build_bound_reference_cache_identity,
)
from .fleet_mixtures import FleetMixturePlan, fleet_mixture_plan_from_mapping
from .physics_distributions import PhysicsDistributionPlan, physics_distribution_plan_from_mapping
from .q5_integration import build_q5_bindings, build_q5_registry, prepare_q5_scenario
from .robustness_matrix import RobustnessWorkUnit
from .sampled_q5_integration import build_sampled_q5_bindings, build_sampled_q5_registry

ROBUSTNESS_RUNTIME_SCHEMA = "cross-psp-robustness-runtime-result-v1"
ROBUSTNESS_RUNTIME_ADAPTER_ID = "cross-psp-q5-robustness-runtime-v1"
ROBUSTNESS_PARAMETER_BINDING_SCHEMA = "cross-psp-robustness-parameter-bindings-v1"
ROBUSTNESS_REPLICATION_SEMANTICS_SCHEMA = "cross-psp-robustness-replication-semantics-v1"

_FIXED_STREAM_RE = re.compile(r"^stream\.([A-Za-z0-9_-]+)\.([a-z0-9_]+)$")
_DISTRIBUTION_RE = re.compile(
    r"^distribution\.([A-Za-z0-9_-]+)\.(bounds\.(?:lower|upper)|parameters\.(?:mean|stddev|mode))$"
)
_FLEET_RE = re.compile(r"^fleet\.([A-Za-z0-9_-]+)\.class\.([^\.]+)\.([a-z0-9_]+)$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

_FIXED_CLASS_FIELDS = MappingProxyType({
    "length_m": ("dimensions_m", "length_m"),
    "beam_m": ("dimensions_m", "beam_m"),
    "maximum_speed_knots": (None, "maximum_speed_knots"),
    "max_acceleration_knots_per_min": ("maneuver_limits", "max_acceleration_knots_per_min"),
    "max_deceleration_knots_per_min": ("maneuver_limits", "max_deceleration_knots_per_min"),
    "max_turn_rate_deg_per_min": ("maneuver_limits", "max_turn_rate_deg_per_min"),
    "safety_radius_nm": ("safety_domain", "radius_nm"),
})
_FLEET_PROFILE_FIELDS = MappingProxyType({
    "length_m": ("dimensions_m", "length_m"),
    "beam_m": ("dimensions_m", "beam_m"),
    "cruise_speed_knots": ("speed", "cruise_speed_knots"),
    "maximum_speed_knots": ("speed", "maximum_speed_knots"),
    "max_acceleration_knots_per_min": ("maneuver_limits", "max_acceleration_knots_per_min"),
    "max_deceleration_knots_per_min": ("maneuver_limits", "max_deceleration_knots_per_min"),
    "max_turn_rate_deg_per_min": ("maneuver_limits", "max_turn_rate_deg_per_min"),
    "safety_ahead_nm": ("safety_domain", "ahead_nm"),
    "safety_astern_nm": ("safety_domain", "astern_nm"),
    "safety_port_nm": ("safety_domain", "port_nm"),
    "safety_starboard_nm": ("safety_domain", "starboard_nm"),
})


class RobustnessRuntimeError(RuntimeError):
    """Raised when a work unit cannot be bound or executed without ambiguity."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RobustnessRuntimeError(f"value is not canonical JSON: {exc}") from exc


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _sha256(name: str, value: object) -> str:
    text = str(value).strip().lower()
    if not _SHA_RE.fullmatch(text):
        raise RobustnessRuntimeError(f"{name} must be a lowercase 64-character SHA-256")
    return text


def _decimal(name: str, value: object) -> Decimal:
    if isinstance(value, bool):
        raise RobustnessRuntimeError(f"{name} must be finite numeric")
    try:
        out = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RobustnessRuntimeError(f"{name} must be finite numeric") from exc
    if not out.is_finite():
        raise RobustnessRuntimeError(f"{name} must be finite numeric")
    return out


def _derive_seed(master: int, namespace: str) -> int:
    material = {
        "schema": ROBUSTNESS_RUNTIME_ADAPTER_ID,
        "work_unit_seed": int(master),
        "namespace": namespace,
    }
    return int.from_bytes(sha256(_canonical_bytes(material)).digest()[:8], "big", signed=False)


def parameter_binding_manifest() -> dict[str, object]:
    return {
        "schema": ROBUSTNESS_PARAMETER_BINDING_SCHEMA,
        "fixed": {
            "pattern": "stream.<stream_id>.<field>",
            "fields": ["cruise_speed_knots", *_FIXED_CLASS_FIELDS.keys()],
        },
        "distributions": {
            "pattern": "distribution.<stream_id>.<field>",
            "fields": [
                "bounds.lower",
                "bounds.upper",
                "parameters.mean",
                "parameters.stddev",
                "parameters.mode",
            ],
        },
        "fleet_mixture": {
            "pattern": "fleet.<stream_id>.class.<class_id>.<field>",
            "fields": ["weight", *_FLEET_PROFILE_FIELDS.keys()],
        },
    }


def runtime_adapter_code_identity_sha256() -> str:
    """Hash the adapter and the execution/binding source files that define its semantics."""
    root = Path(__file__).resolve().parent
    files = (
        "robustness_runtime.py",
        "custom_physics_binding.py",
        "sampled_q5_integration.py",
        "q5_integration.py",
        "physics_distributions.py",
        "fleet_mixtures.py",
    )
    source_hashes: dict[str, str] = {}
    for name in files:
        path = root / name
        try:
            source_hashes[name] = sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RobustnessRuntimeError(f"cannot hash runtime source {path}: {exc}") from exc
    return _sha({
        "adapter_id": ROBUSTNESS_RUNTIME_ADAPTER_ID,
        "binding_manifest": parameter_binding_manifest(),
        "source_sha256": source_hashes,
    })


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RobustnessRuntimeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise RobustnessRuntimeError(f"{label} {path} must contain a mapping")
    return dict(raw)


def _scenario_parts(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    try:
        params = payload["model"]["parameters"]
        streams = params["streams"]
        classes = params["vessel_classes"]
    except (KeyError, TypeError) as exc:
        raise RobustnessRuntimeError("baseline scenario is missing canonical model.parameters fields") from exc
    if not isinstance(params, dict) or not isinstance(streams, list) or not isinstance(classes, dict):
        raise RobustnessRuntimeError("baseline scenario model.parameters shape is unsupported")
    return params, streams, classes


def _stream_row(streams: list[dict[str, Any]], stream_id: str) -> dict[str, Any]:
    rows = [row for row in streams if isinstance(row, dict) and row.get("stream_id") == stream_id]
    if len(rows) != 1:
        raise RobustnessRuntimeError(f"stream binding requires exactly one stream {stream_id!r}")
    return rows[0]


def _set_nested_numeric(container: dict[str, Any], path: tuple[str | None, str], value: Decimal, *, label: str) -> None:
    parent_name, field = path
    target = container if parent_name is None else container.get(parent_name)
    if not isinstance(target, dict) or field not in target:
        raise RobustnessRuntimeError(f"{label} is not supported by the baseline representation")
    target[field] = float(value)


def _class_scalar(class_payload: Mapping[str, Any], path: tuple[str | None, str], *, label: str) -> Decimal:
    parent_name, field = path
    target = class_payload if parent_name is None else class_payload.get(parent_name)
    if not isinstance(target, Mapping) or field not in target:
        raise RobustnessRuntimeError(f"{label} is not supported by the baseline representation")
    return _decimal(label, target[field])


def _clone_class_for_stream(
    *,
    streams: list[dict[str, Any]],
    classes: dict[str, Any],
    stream: dict[str, Any],
    stream_id: str,
    combination_sha: str,
) -> dict[str, Any]:
    class_id = str(stream.get("vessel_class_id", ""))
    original = classes.get(class_id)
    if not isinstance(original, Mapping):
        raise RobustnessRuntimeError(f"stream {stream_id!r} references missing vessel class {class_id!r}")
    references = [row for row in streams if isinstance(row, Mapping) and row.get("vessel_class_id") == class_id]
    if len(references) <= 1:
        return original  # type: ignore[return-value]
    clone_id = f"{class_id}__a93_{stream_id}_{combination_sha[:12]}"
    previous = classes.get(clone_id)
    if previous is None:
        classes[clone_id] = deepcopy(dict(original))
    elif previous != original:
        raise RobustnessRuntimeError(f"generated vessel class collision for {clone_id!r}")
    stream["vessel_class_id"] = clone_id
    return classes[clone_id]


def _apply_fixed_parameters(payload: dict[str, Any], unit: RobustnessWorkUnit) -> None:
    _, streams, classes = _scenario_parts(payload)
    for parameter_id, raw_value in sorted(unit.parameter_values.items()):
        match = _FIXED_STREAM_RE.fullmatch(parameter_id)
        if match is None:
            raise RobustnessRuntimeError(
                f"unsupported fixed parameter binding {parameter_id!r}; expected stream.<stream_id>.<supported_field>"
            )
        stream_id, field = match.groups()
        value = _decimal(parameter_id, raw_value)
        stream = _stream_row(streams, stream_id)
        if field == "cruise_speed_knots":
            if "cruise_speed_knots" not in stream:
                raise RobustnessRuntimeError(f"{parameter_id!r} is absent from the baseline stream")
            stream["cruise_speed_knots"] = float(value)
            continue
        path = _FIXED_CLASS_FIELDS.get(field)
        if path is None:
            raise RobustnessRuntimeError(
                f"unsupported fixed parameter field {field!r}; supported={sorted(['cruise_speed_knots', *_FIXED_CLASS_FIELDS])}"
            )
        class_id = str(stream.get("vessel_class_id", ""))
        current = classes.get(class_id)
        if not isinstance(current, Mapping):
            raise RobustnessRuntimeError(f"stream {stream_id!r} references missing vessel class {class_id!r}")
        baseline_value = _class_scalar(current, path, label=parameter_id)
        if value == baseline_value:
            continue
        target = _clone_class_for_stream(
            streams=streams,
            classes=classes,
            stream=stream,
            stream_id=stream_id,
            combination_sha=unit.combination_identity_sha256,
        )
        _set_nested_numeric(target, path, value, label=parameter_id)


def _find_distribution_stream(plan: dict[str, Any], stream_id: str) -> dict[str, Any]:
    streams = plan.get("streams")
    if not isinstance(streams, list):
        raise RobustnessRuntimeError("distribution plan streams must be a sequence")
    rows = [row for row in streams if isinstance(row, dict) and row.get("stream_id") == stream_id]
    if len(rows) != 1:
        raise RobustnessRuntimeError(f"distribution binding requires exactly one stream {stream_id!r}")
    return rows[0]


def _apply_distribution_parameters(plan: dict[str, Any], unit: RobustnessWorkUnit) -> None:
    for parameter_id, raw_value in sorted(unit.parameter_values.items()):
        match = _DISTRIBUTION_RE.fullmatch(parameter_id)
        if match is None:
            raise RobustnessRuntimeError(
                f"unsupported distribution parameter binding {parameter_id!r}; see {ROBUSTNESS_PARAMETER_BINDING_SCHEMA}"
            )
        stream_id, field_path = match.groups()
        value = _decimal(parameter_id, raw_value)
        row = _find_distribution_stream(plan, stream_id)
        distribution = row.get("distribution")
        if not isinstance(distribution, dict):
            raise RobustnessRuntimeError(f"distribution stream {stream_id!r} has invalid distribution mapping")
        group, field = field_path.split(".", 1)
        target = distribution.get(group)
        if not isinstance(target, dict) or field not in target:
            raise RobustnessRuntimeError(
                f"{parameter_id!r} is not declared by the baseline distribution family"
            )
        target[field] = float(value)


def _fleet_alias(class_id: str, stream_id: str, combination_sha: str) -> str:
    return f"{class_id}__a93_{stream_id}_{combination_sha[:12]}"


def _find_fleet_member(
    plan: dict[str, Any],
    stream_id: str,
    class_id: str,
    *,
    combination_sha: str,
) -> dict[str, Any]:
    streams = plan.get("streams")
    if not isinstance(streams, list):
        raise RobustnessRuntimeError("fleet-mixture streams must be a sequence")
    stream_rows = [row for row in streams if isinstance(row, dict) and row.get("stream_id") == stream_id]
    if len(stream_rows) != 1:
        raise RobustnessRuntimeError(f"fleet binding requires exactly one stream {stream_id!r}")
    classes = stream_rows[0].get("classes")
    if not isinstance(classes, list):
        raise RobustnessRuntimeError(f"fleet stream {stream_id!r} classes must be a sequence")
    accepted_ids = {class_id, _fleet_alias(class_id, stream_id, combination_sha)}
    members = [
        row for row in classes
        if isinstance(row, dict)
        and isinstance(row.get("profile"), Mapping)
        and row["profile"].get("class_id") in accepted_ids
    ]
    if len(members) != 1:
        raise RobustnessRuntimeError(
            f"fleet binding requires exactly one class {class_id!r} in stream {stream_id!r}"
        )
    return members[0]


def _fleet_class_is_shared(plan: Mapping[str, Any], stream_id: str, class_id: str) -> bool:
    streams = plan.get("streams")
    if not isinstance(streams, list):
        return False
    count = 0
    for stream in streams:
        if not isinstance(stream, Mapping):
            continue
        classes = stream.get("classes")
        if not isinstance(classes, list):
            continue
        for member in classes:
            if (
                isinstance(member, Mapping)
                and isinstance(member.get("profile"), Mapping)
                and member["profile"].get("class_id") == class_id
            ):
                count += 1
    return count > 1


def _apply_fleet_parameters(plan: dict[str, Any], unit: RobustnessWorkUnit) -> None:
    for parameter_id, raw_value in sorted(unit.parameter_values.items()):
        match = _FLEET_RE.fullmatch(parameter_id)
        if match is None:
            raise RobustnessRuntimeError(
                f"unsupported fleet-mixture parameter binding {parameter_id!r}; see {ROBUSTNESS_PARAMETER_BINDING_SCHEMA}"
            )
        stream_id, class_id, field = match.groups()
        value = _decimal(parameter_id, raw_value)
        member = _find_fleet_member(
            plan, stream_id, class_id, combination_sha=unit.combination_identity_sha256
        )
        if field == "weight":
            member["weight"] = str(value)
            continue
        path = _FLEET_PROFILE_FIELDS.get(field)
        if path is None:
            raise RobustnessRuntimeError(
                f"unsupported fleet-mixture class field {field!r}; supported={sorted(['weight', *_FLEET_PROFILE_FIELDS])}"
            )
        profile = member.get("profile")
        if not isinstance(profile, dict):
            raise RobustnessRuntimeError(f"fleet member {class_id!r} has invalid profile")
        current = _class_scalar(profile, path, label=parameter_id)
        if value == current:
            continue
        if _fleet_class_is_shared(plan, stream_id, class_id):
            profile["class_id"] = _fleet_alias(
                class_id, stream_id, unit.combination_identity_sha256
            )
        _set_nested_numeric(profile, path, value, label=parameter_id)


def _set_traffic_seed(scenario_payload: dict[str, Any], seed: int) -> None:
    params, _, _ = _scenario_parts(scenario_payload)
    scenario = params.get("scenario")
    if not isinstance(scenario, dict) or "seed" not in scenario:
        raise RobustnessRuntimeError("baseline scenario does not expose model.parameters.scenario.seed")
    scenario["seed"] = int(seed)


def _write_scenario(payload: Mapping[str, Any], root: Path) -> Path:
    path = root / "scenario.yaml"
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _level_payload(level) -> dict[str, object]:
    return {
        "primary_kpi": float(level.primary_kpi),
        "aggregate_metrics": dict(level.aggregate_metrics),
        "causal": bool(level.causal),
        "intervention_id": level.intervention_id,
        "reason": level.reason,
    }


def _result_mapping(
    *,
    unit: RobustnessWorkUnit,
    mode: str,
    baseline_physics_sha: str,
    bound: BoundCustomPhysicsScenario,
    execution_identity_sha: str,
    traffic_seed: int,
    sample_seed: int | None,
    q5_confidence_level: float | None,
    q5_paired: bool | None,
    result,
) -> dict[str, object]:
    b0 = result.study.level(ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK)
    b4 = result.study.level(ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION)
    b5 = result.study.level(ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE)
    b6 = result.study.level(ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE)
    cache_identity = build_bound_reference_cache_identity(bound)
    b0_kpi = float(b0.primary_kpi)
    b4_kpi = float(b4.primary_kpi)
    b5_kpi = float(b5.primary_kpi)
    b6_kpi = float(b6.primary_kpi)
    internal_replications = int(b5.aggregate_metrics.get("replication_count", 1))
    internal_seed_schedule = [int(value) for value in b5.aggregate_metrics.get("seed_schedule", ())]
    if len(internal_seed_schedule) != internal_replications:
        raise RobustnessRuntimeError(
            "Q5 B5 replication metadata is inconsistent: replication_count does not match seed_schedule"
        )
    campaign_axis = "NONE_FIXED_Q5_INTERNAL_ONLY" if mode == "fixed" else "SAMPLED_PHYSICS_SEED"
    return {
        "schema": ROBUSTNESS_RUNTIME_SCHEMA,
        "runtime_adapter_id": ROBUSTNESS_RUNTIME_ADAPTER_ID,
        "work_unit_identity_sha256": unit.work_unit_identity_sha256,
        "combination_identity_sha256": unit.combination_identity_sha256,
        "replication_index": unit.replication_index,
        "custom_physics_mode": mode,
        "seeds": {
            "work_unit_seed": unit.derived_seed,
            "traffic_seed": traffic_seed,
            "sample_seed": sample_seed,
        },
        "replication_semantics": {
            "schema": ROBUSTNESS_REPLICATION_SEMANTICS_SCHEMA,
            "campaign_replication_index": unit.replication_index,
            "campaign_replication_axis": campaign_axis,
            "q5_internal_replication_axis": "TRAFFIC_REALIZATION",
            "q5_internal_replication_count": internal_replications,
            "q5_internal_seed_schedule": internal_seed_schedule,
            "q5_internal_paired": q5_paired,
            "q5_internal_confidence_level": q5_confidence_level,
            "uncertainty_note": (
                "Fixed-physics campaign replications are not a second traffic-seed axis; "
                "Q5 internal paired replications provide the traffic-realization layer."
                if mode == "fixed" else
                "Campaign replications vary sampled-physics seed; Q5 internal paired replications "
                "remain the traffic-realization layer."
            ),
        },
        "identities": {
            "baseline_physics_identity_sha256": baseline_physics_sha,
            "derived_physics_identity_sha256": bound.physics_identity_sha256,
            "derived_plan_identity_sha256": bound.plan_identity_sha256,
            "q5_execution_identity_sha256": execution_identity_sha,
            "reference_cache_identity_sha256": cache_identity.sha256,
            "runtime_adapter_code_identity_sha256": runtime_adapter_code_identity_sha256(),
        },
        "scope_status": bound.scope_status,
        "metrics": {
            "primary": b5_kpi,
            "b0_mean_service_yield_per_hour": b0_kpi,
            "b4_mean_service_yield_per_hour": b4_kpi,
            "b5_mean_service_yield_per_hour": b5_kpi,
            "b6_reference_rate_per_hour": b6_kpi,
            "b4_delta_vs_b0_per_hour": b4_kpi - b0_kpi,
            "b5_delta_vs_b4_per_hour": b5_kpi - b4_kpi,
            "b5_delta_vs_b0_per_hour": b5_kpi - b0_kpi,
        },
        "levels": {
            "B0": _level_payload(b0),
            "B4": _level_payload(b4),
            "B5": _level_payload(b5),
            "B6": _level_payload(b6),
        },
        "claim_boundary": {
            "automatic_qualification_promotion": False,
            "causal": False,
            "field_calibrated": False,
            "deployment_recommendation": False,
            "global_optimum_claim": False,
        },
    }


@dataclass(frozen=True, slots=True)
class PreparedRobustnessWorkUnitRuntime:
    unit: RobustnessWorkUnit
    mode: str
    scenario: object
    bound: BoundCustomPhysicsScenario
    registry: object
    execution_identity_sha256: str
    baseline_physics_identity_sha256: str
    traffic_seed: int
    sample_seed: int | None
    derived_plan_identity_sha256: str | None

    def identity_dict(self) -> dict[str, object]:
        cache_identity = build_bound_reference_cache_identity(self.bound)
        return {
            "work_unit_identity_sha256": self.unit.work_unit_identity_sha256,
            "combination_identity_sha256": self.unit.combination_identity_sha256,
            "custom_physics_mode": self.mode,
            "baseline_physics_identity_sha256": self.baseline_physics_identity_sha256,
            "derived_physics_identity_sha256": self.bound.physics_identity_sha256,
            "derived_plan_identity_sha256": self.derived_plan_identity_sha256,
            "q5_execution_identity_sha256": self.execution_identity_sha256,
            "reference_cache_identity_sha256": cache_identity.sha256,
            "traffic_seed": self.traffic_seed,
            "sample_seed": self.sample_seed,
        }


@dataclass(frozen=True, slots=True)
class RobustnessRuntimeAdapter:
    baseline_source: Mapping[str, object]
    reference_cache_dir: Path | None = None

    def __post_init__(self) -> None:
        source = dict(self.baseline_source)
        required = {
            "mode",
            "scenario_path",
            "plan_path",
            "expected_physics_identity_sha256",
            "expected_plan_identity_sha256",
        }
        if set(source) != required:
            raise RobustnessRuntimeError(
                "baseline_source fields mismatch; "
                f"missing={sorted(required - set(source))}, extra={sorted(set(source) - required)}"
            )
        mode = str(source["mode"])
        if mode not in {"fixed", "distributions", "fleet_mixture"}:
            raise RobustnessRuntimeError(f"unsupported baseline mode {mode!r}")
        object.__setattr__(self, "baseline_source", MappingProxyType(source))
        if self.reference_cache_dir is not None:
            object.__setattr__(self, "reference_cache_dir", Path(self.reference_cache_dir))
        self._validate_baseline()

    @property
    def code_identity_sha256(self) -> str:
        return runtime_adapter_code_identity_sha256()

    def _baseline_objects(self):
        source = self.baseline_source
        scenario_path = Path(str(source["scenario_path"]))
        if not scenario_path.is_file():
            raise RobustnessRuntimeError(f"baseline scenario not found: {scenario_path}")
        scenario = load_diagnostic_scenario(scenario_path)
        prepared = prepare_q5_scenario(scenario)
        mode = str(source["mode"])
        plan_raw: dict[str, Any] | None = None
        plan: PhysicsDistributionPlan | FleetMixturePlan | None = None
        if mode == "fixed":
            if source["plan_path"] is not None:
                raise RobustnessRuntimeError("fixed baseline must have plan_path=null")
            bound = bind_fixed_scenario(prepared)
        elif mode == "distributions":
            if source["plan_path"] is None:
                raise RobustnessRuntimeError("distributions baseline requires plan_path")
            plan_raw = _load_yaml_mapping(Path(str(source["plan_path"])), label="distribution plan")
            plan = physics_distribution_plan_from_mapping(plan_raw)
            bound = bind_distribution_scenario(prepared, plan)
        else:
            if source["plan_path"] is None:
                raise RobustnessRuntimeError("fleet_mixture baseline requires plan_path")
            plan_raw = _load_yaml_mapping(Path(str(source["plan_path"])), label="fleet-mixture plan")
            plan = fleet_mixture_plan_from_mapping(plan_raw)
            bound = bind_fleet_mixture_scenario(prepared, plan)
        return scenario_path, prepared, plan_raw, plan, bound

    def _validate_baseline(self) -> None:
        _, _, _, plan, bound = self._baseline_objects()
        expected_physics = _sha256(
            "baseline_source.expected_physics_identity_sha256",
            self.baseline_source["expected_physics_identity_sha256"],
        )
        if bound.physics_identity_sha256 != expected_physics:
            raise RobustnessRuntimeError(
                "baseline physics identity mismatch: baseline files changed after campaign configuration"
            )
        expected_plan = self.baseline_source["expected_plan_identity_sha256"]
        if plan is None:
            if expected_plan is not None:
                raise RobustnessRuntimeError("fixed baseline must have expected_plan_identity_sha256=null")
        else:
            if expected_plan is None:
                raise RobustnessRuntimeError("sampled baseline requires expected_plan_identity_sha256")
            if plan.plan_identity_sha256 != _sha256(
                "baseline_source.expected_plan_identity_sha256", expected_plan
            ):
                raise RobustnessRuntimeError(
                    "baseline plan identity mismatch: baseline plan changed after campaign configuration"
                )

    def prepare(self, unit: RobustnessWorkUnit) -> PreparedRobustnessWorkUnitRuntime:
        """Derive and validate one complete work-unit scenario/plan without executing it."""
        if not isinstance(unit, RobustnessWorkUnit):
            raise RobustnessRuntimeError("runtime adapter requires an A89 RobustnessWorkUnit")
        mode = str(self.baseline_source["mode"])
        if mode == "fixed" and unit.replication_index != 0:
            raise RobustnessRuntimeError(
                "fixed-physics campaign replication_index must be 0: Q5 already owns the paired "
                "traffic-replication layer; duplicate outer fixed replications are unsupported"
            )
        scenario_path, _, plan_raw, _, baseline_bound = self._baseline_objects()
        scenario_payload = deepcopy(_load_yaml_mapping(scenario_path, label="baseline scenario"))
        params, _, _ = _scenario_parts(scenario_payload)
        scenario_config = params.get("scenario")
        if not isinstance(scenario_config, Mapping) or "seed" not in scenario_config:
            raise RobustnessRuntimeError("baseline scenario does not expose model.parameters.scenario.seed")
        traffic_seed = int(scenario_config["seed"])
        sample_seed: int | None = None

        if mode == "fixed":
            _apply_fixed_parameters(scenario_payload, unit)
        elif mode == "distributions":
            if plan_raw is None:
                raise RobustnessRuntimeError("distribution baseline plan unexpectedly missing")
            plan_raw = deepcopy(plan_raw)
            _apply_distribution_parameters(plan_raw, unit)
            sample_seed = _derive_seed(unit.derived_seed, "sample")
            plan_raw["master_seed"] = sample_seed
        elif mode == "fleet_mixture":
            if plan_raw is None:
                raise RobustnessRuntimeError("fleet-mixture baseline plan unexpectedly missing")
            plan_raw = deepcopy(plan_raw)
            _apply_fleet_parameters(plan_raw, unit)
            sample_seed = _derive_seed(unit.derived_seed, "sample")
            plan_raw["master_seed"] = sample_seed
        else:  # pragma: no cover - constructor rejects this
            raise RobustnessRuntimeError(f"unsupported baseline mode {mode!r}")

        try:
            with tempfile.TemporaryDirectory(prefix="cross_psp_a93_") as tmp_name:
                scenario_file = _write_scenario(scenario_payload, Path(tmp_name))
                scenario = load_diagnostic_scenario(scenario_file)
                prepared = prepare_q5_scenario(scenario)
                if mode == "fixed":
                    bound = bind_fixed_scenario(prepared)
                    bindings = build_q5_bindings(
                        scenario, reference_cache_dir=self.reference_cache_dir
                    )
                    registry = build_q5_registry(
                        scenario, reference_cache_dir=self.reference_cache_dir
                    )
                elif mode == "distributions":
                    assert plan_raw is not None
                    distribution_plan = physics_distribution_plan_from_mapping(plan_raw)
                    bound = bind_distribution_scenario(prepared, distribution_plan)
                    bindings = build_sampled_q5_bindings(
                        bound,
                        distribution_plan=distribution_plan,
                        reference_cache_dir=self.reference_cache_dir,
                    )
                    registry = build_sampled_q5_registry(
                        bound,
                        distribution_plan=distribution_plan,
                        reference_cache_dir=self.reference_cache_dir,
                    )
                else:
                    assert plan_raw is not None
                    fleet_plan = fleet_mixture_plan_from_mapping(plan_raw)
                    bound = bind_fleet_mixture_scenario(prepared, fleet_plan)
                    bindings = build_sampled_q5_bindings(
                        bound,
                        fleet_mixture_plan=fleet_plan,
                        reference_cache_dir=self.reference_cache_dir,
                    )
                    registry = build_sampled_q5_registry(
                        bound,
                        fleet_mixture_plan=fleet_plan,
                        reference_cache_dir=self.reference_cache_dir,
                    )
                identity = bindings.execution_identity
                if identity is None:
                    raise RobustnessRuntimeError("Q5 binding did not provide an execution identity")
        except RobustnessRuntimeError:
            raise
        except Exception as exc:
            raise RobustnessRuntimeError(
                f"work unit {unit.work_unit_index} failed during supported Q5 preparation: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return PreparedRobustnessWorkUnitRuntime(
            unit=unit,
            mode=mode,
            scenario=scenario,
            bound=bound,
            registry=registry,
            execution_identity_sha256=identity.identity_sha256,
            baseline_physics_identity_sha256=baseline_bound.physics_identity_sha256,
            traffic_seed=traffic_seed,
            sample_seed=sample_seed,
            derived_plan_identity_sha256=bound.plan_identity_sha256,
        )

    def execute(self, unit: RobustnessWorkUnit) -> Mapping[str, object]:
        prepared = self.prepare(unit)
        try:
            result = run_diagnostic_orchestrator(prepared.scenario, registry=prepared.registry)
        except Exception as exc:
            raise RobustnessRuntimeError(
                f"work unit {unit.work_unit_index} failed in supported Q5 runtime: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        experiment = getattr(prepared.scenario, "experiment", None)
        confidence_level = getattr(experiment, "confidence_level", None)
        paired = getattr(experiment, "paired", None)
        return _result_mapping(
            unit=unit,
            mode=prepared.mode,
            baseline_physics_sha=prepared.baseline_physics_identity_sha256,
            bound=prepared.bound,
            execution_identity_sha=prepared.execution_identity_sha256,
            traffic_seed=prepared.traffic_seed,
            sample_seed=prepared.sample_seed,
            q5_confidence_level=None if confidence_level is None else float(confidence_level),
            q5_paired=None if paired is None else bool(paired),
            result=result,
        )


def build_robustness_runtime_adapter(
    baseline_source: Mapping[str, object],
    *,
    reference_cache_dir: str | Path | None = None,
) -> RobustnessRuntimeAdapter:
    return RobustnessRuntimeAdapter(
        baseline_source=dict(baseline_source),
        reference_cache_dir=None if reference_cache_dir is None else Path(reference_cache_dir),
    )
