"""Fail-closed scientific-profile contract for the first production PSP track.

The contract remains fail-closed for B4/B5/B6 runtime bindings.  A64 adds only the
separately qualified safe conflict-slot solver and deterministic cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Mapping

import yaml

from vessel_crossing.version import __version__ as _base_framework_version

PROFILE_CONTRACT_SCHEMA = "cross-two-stream-profile-contract-v4"
PROFILE_ID = "cross-two-stream-crossing-profile-v1"
PROFILE_VERSION = "0.1.0a7"
BASE_FRAMEWORK_VERSION = "1.1.0a36"
RUNTIME_ADAPTER_ID = "nextgen.environmental_network_fifo.v1"
ROUTE_MODEL_ID = "straight-orthogonal-single-conflict-center-v1"
ENVIRONMENT_MODEL_ID = "canonical-uniform-drift-2d-v1"
PLANNED_SLOT_PREDICTION_SCOPE_ID = "zero-drift-straight-nominal-v1"
SLOT_SOLVER_CONTRACT_ID = "cross-safe-conflict-slot-grid-contract-v1"
B4_POLICY_CONTRACT_ID = "cross-b4-earliest-safe-slot-contract-v1"
B5_POLICY_CONTRACT_ID = "cross-b5-bounded-slot-cost-staging-contract-v1"
B6_REFERENCE_CONTRACT_ID = "cross-b6-balanced-saturated-reference-contract-v3"
REFERENCE_ARBITER_CONTRACT_ID = "cross-b6-reference-fair-slot-arbiter-contract-v1"


class ProfileContractError(ValueError):
    """Raised when a PSP contract leaves the qualified v1 domain."""


def _finite_positive(name: str, value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileContractError(f"{name} must be a finite positive number") from exc
    if not isfinite(out) or out <= 0.0:
        raise ProfileContractError(f"{name} must be a finite positive number")
    return out


def _finite_nonnegative(name: str, value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileContractError(f"{name} must be finite and non-negative") from exc
    if not isfinite(out) or out < 0.0:
        raise ProfileContractError(f"{name} must be finite and non-negative")
    return out


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProfileContractError(f"{name} must be an integer >= {minimum}")
    return value


def _strict_keys(name: str, value: Mapping[str, object], expected: set[str]) -> None:
    keys = set(value)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise ProfileContractError(f"{name} keys mismatch; missing={missing}, extra={extra}")


@dataclass(frozen=True, slots=True)
class StreamGeometry:
    stream_id: str
    source_to_center_distance_nm: float
    center_to_exit_distance_nm: float

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not self.stream_id.strip():
            raise ProfileContractError("stream_id must be a non-empty string")
        object.__setattr__(
            self,
            "source_to_center_distance_nm",
            _finite_positive("source_to_center_distance_nm", self.source_to_center_distance_nm),
        )
        object.__setattr__(
            self,
            "center_to_exit_distance_nm",
            _finite_positive("center_to_exit_distance_nm", self.center_to_exit_distance_nm),
        )


@dataclass(frozen=True, slots=True)
class StreamShare:
    stream_id: str
    share: float

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not self.stream_id.strip():
            raise ProfileContractError("stream share stream_id must be a non-empty string")
        share = _finite_positive("stream share", self.share)
        if share >= 1.0:
            raise ProfileContractError("each stream share must be < 1")
        object.__setattr__(self, "share", share)


@dataclass(frozen=True, slots=True)
class SafeSlotNumericalContract:
    separation_grid_min: float
    max_search_min: float
    clearance_time_tolerance_min: float
    clearance_margin_tolerance_nm: float
    max_clearance_subdivisions: int

    def __post_init__(self) -> None:
        grid = _finite_positive("separation_grid_min", self.separation_grid_min)
        max_search = _finite_positive("max_search_min", self.max_search_min)
        time_tolerance = _finite_positive(
            "clearance_time_tolerance_min", self.clearance_time_tolerance_min
        )
        margin_tolerance = _finite_nonnegative(
            "clearance_margin_tolerance_nm", self.clearance_margin_tolerance_nm
        )
        subdivisions = _integer(
            "max_clearance_subdivisions", self.max_clearance_subdivisions, minimum=1
        )
        if grid > max_search:
            raise ProfileContractError("separation_grid_min must not exceed max_search_min")
        object.__setattr__(self, "separation_grid_min", grid)
        object.__setattr__(self, "max_search_min", max_search)
        object.__setattr__(self, "clearance_time_tolerance_min", time_tolerance)
        object.__setattr__(self, "clearance_margin_tolerance_nm", margin_tolerance)
        object.__setattr__(self, "max_clearance_subdivisions", subdivisions)


@dataclass(frozen=True, slots=True)
class B5FairnessContract:
    lookahead_depth: int
    max_bypass_per_demand: int
    max_extra_wait_min: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "lookahead_depth", _integer("lookahead_depth", self.lookahead_depth, minimum=2))
        object.__setattr__(
            self,
            "max_bypass_per_demand",
            _integer("max_bypass_per_demand", self.max_bypass_per_demand, minimum=1),
        )
        object.__setattr__(
            self,
            "max_extra_wait_min",
            _finite_positive("max_extra_wait_min", self.max_extra_wait_min),
        )


@dataclass(frozen=True, slots=True)
class B6SupplyTemplateEntry:
    profile_class_id: str
    through_water_heading_deg: float
    through_water_speed_nm_per_min: float

    def __post_init__(self) -> None:
        if not isinstance(self.profile_class_id, str) or not self.profile_class_id.strip():
            raise ProfileContractError("B6 supply profile_class_id must be a non-empty string")
        heading = float(self.through_water_heading_deg)
        if not isfinite(heading):
            raise ProfileContractError("B6 supply heading must be finite")
        object.__setattr__(self, "through_water_heading_deg", heading % 360.0)
        object.__setattr__(
            self,
            "through_water_speed_nm_per_min",
            _finite_positive("B6 supply through_water_speed_nm_per_min", self.through_water_speed_nm_per_min),
        )


@dataclass(frozen=True, slots=True)
class B6StreamSupplyTemplate:
    stream_id: str
    cycle: tuple[B6SupplyTemplateEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not self.stream_id.strip():
            raise ProfileContractError("B6 supply template stream_id must be a non-empty string")
        if not self.cycle:
            raise ProfileContractError("B6 supply template cycle must be non-empty")
        headings = {round(item.through_water_heading_deg, 12) for item in self.cycle}
        if len(headings) != 1:
            raise ProfileContractError("B6 v1 requires one fixed heading per stream supply template")


@dataclass(frozen=True, slots=True)
class B6ReferenceContract:
    stream_shares: tuple[StreamShare, StreamShare]
    supply_templates: tuple[B6StreamSupplyTemplate, B6StreamSupplyTemplate]
    max_share_deficit_passages: float
    fairness_comparison_tolerance: float
    balance_tolerance_fraction: float
    warmup_min: float
    measurement_duration_min: float
    drain_min: float
    saturated_demands_per_stream: int

    def __post_init__(self) -> None:
        if len(self.stream_shares) != 2:
            raise ProfileContractError("B6 requires exactly two stream shares")
        ids = tuple(item.stream_id for item in self.stream_shares)
        if len(set(ids)) != 2:
            raise ProfileContractError("B6 stream shares must name two distinct streams")
        if abs(sum(item.share for item in self.stream_shares) - 1.0) > 1e-12:
            raise ProfileContractError("B6 stream shares must sum to 1")
        if len(self.supply_templates) != 2:
            raise ProfileContractError("B6 requires exactly two supply templates")
        template_ids = tuple(item.stream_id for item in self.supply_templates)
        if len(set(template_ids)) != 2:
            raise ProfileContractError("B6 supply templates must name two distinct streams")
        heading_by_stream = {item.stream_id: item.cycle[0].through_water_heading_deg for item in self.supply_templates}
        h = tuple(heading_by_stream[item.stream_id] for item in self.supply_templates)
        raw = abs((h[0] - h[1]) % 360.0)
        angle = min(raw, 360.0 - raw)
        if abs(angle - 90.0) > 1e-9:
            raise ProfileContractError("B6 supply template headings must be orthogonal in v1")
        object.__setattr__(
            self,
            "max_share_deficit_passages",
            _finite_positive("max_share_deficit_passages", self.max_share_deficit_passages),
        )
        fair_eps = _finite_nonnegative("fairness_comparison_tolerance", self.fairness_comparison_tolerance)
        if fair_eps >= self.max_share_deficit_passages:
            raise ProfileContractError("fairness_comparison_tolerance must be smaller than max_share_deficit_passages")
        object.__setattr__(self, "fairness_comparison_tolerance", fair_eps)
        tolerance = _finite_nonnegative(
            "balance_tolerance_fraction", self.balance_tolerance_fraction
        )
        if tolerance >= 0.5:
            raise ProfileContractError("balance_tolerance_fraction must be < 0.5")
        object.__setattr__(self, "balance_tolerance_fraction", tolerance)
        object.__setattr__(self, "warmup_min", _finite_nonnegative("warmup_min", self.warmup_min))
        object.__setattr__(
            self,
            "measurement_duration_min",
            _finite_positive("measurement_duration_min", self.measurement_duration_min),
        )
        object.__setattr__(self, "drain_min", _finite_nonnegative("drain_min", self.drain_min))
        object.__setattr__(
            self,
            "saturated_demands_per_stream",
            _integer("saturated_demands_per_stream", self.saturated_demands_per_stream, minimum=2),
        )


@dataclass(frozen=True, slots=True)
class TwoStreamProfileContract:
    schema: str
    profile_id: str
    profile_version: str
    base_framework_version: str
    runtime_adapter_id: str
    route_model_id: str
    environment_model_id: str
    planned_slot_prediction_scope_id: str
    crossing_angle_deg: float
    streams: tuple[StreamGeometry, StreamGeometry]
    slot_solver: SafeSlotNumericalContract
    b5_fairness: B5FairnessContract
    b6_reference: B6ReferenceContract

    def __post_init__(self) -> None:
        expected = {
            "schema": PROFILE_CONTRACT_SCHEMA,
            "profile_id": PROFILE_ID,
            "profile_version": PROFILE_VERSION,
            "base_framework_version": BASE_FRAMEWORK_VERSION,
            "runtime_adapter_id": RUNTIME_ADAPTER_ID,
            "route_model_id": ROUTE_MODEL_ID,
            "environment_model_id": ENVIRONMENT_MODEL_ID,
            "planned_slot_prediction_scope_id": PLANNED_SLOT_PREDICTION_SCOPE_ID,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise ProfileContractError(f"{name} must equal {required!r}")
        angle = _finite_positive("crossing_angle_deg", self.crossing_angle_deg)
        if abs(angle - 90.0) > 1e-12:
            raise ProfileContractError("profile v1 supports an orthogonal 90-degree crossing only")
        object.__setattr__(self, "crossing_angle_deg", angle)
        if len(self.streams) != 2:
            raise ProfileContractError("profile v1 requires exactly two streams")
        stream_ids = tuple(item.stream_id for item in self.streams)
        if len(set(stream_ids)) != 2:
            raise ProfileContractError("profile stream IDs must be distinct")
        share_ids = tuple(item.stream_id for item in self.b6_reference.stream_shares)
        if set(share_ids) != set(stream_ids):
            raise ProfileContractError("B6 stream-share IDs must exactly match profile stream IDs")
        template_ids = tuple(item.stream_id for item in self.b6_reference.supply_templates)
        if set(template_ids) != set(stream_ids):
            raise ProfileContractError("B6 supply-template IDs must exactly match profile stream IDs")

    @property
    def component_contract_ids(self) -> Mapping[str, str]:
        return {
            "slot_solver": SLOT_SOLVER_CONTRACT_ID,
            "b4_policy": B4_POLICY_CONTRACT_ID,
            "b5_policy": B5_POLICY_CONTRACT_ID,
            "b6_reference": B6_REFERENCE_CONTRACT_ID,
            "reference_arbiter": REFERENCE_ARBITER_CONTRACT_ID,
        }

    def canonical_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "base_framework_version": self.base_framework_version,
            "runtime_adapter_id": self.runtime_adapter_id,
            "route_model_id": self.route_model_id,
            "environment_model_id": self.environment_model_id,
            "planned_slot_prediction_scope_id": self.planned_slot_prediction_scope_id,
            "crossing_angle_deg": self.crossing_angle_deg,
            "streams": [
                {
                    "stream_id": item.stream_id,
                    "source_to_center_distance_nm": item.source_to_center_distance_nm,
                    "center_to_exit_distance_nm": item.center_to_exit_distance_nm,
                }
                for item in self.streams
            ],
            "slot_solver": {
                "separation_grid_min": self.slot_solver.separation_grid_min,
                "max_search_min": self.slot_solver.max_search_min,
                "clearance_time_tolerance_min": self.slot_solver.clearance_time_tolerance_min,
                "clearance_margin_tolerance_nm": self.slot_solver.clearance_margin_tolerance_nm,
                "max_clearance_subdivisions": self.slot_solver.max_clearance_subdivisions,
            },
            "b5_fairness": {
                "lookahead_depth": self.b5_fairness.lookahead_depth,
                "max_bypass_per_demand": self.b5_fairness.max_bypass_per_demand,
                "max_extra_wait_min": self.b5_fairness.max_extra_wait_min,
            },
            "b6_reference": {
                "stream_shares": [
                    {"stream_id": item.stream_id, "share": item.share}
                    for item in self.b6_reference.stream_shares
                ],
                "supply_templates": [
                    {
                        "stream_id": item.stream_id,
                        "cycle": [
                            {
                                "profile_class_id": entry.profile_class_id,
                                "through_water_heading_deg": entry.through_water_heading_deg,
                                "through_water_speed_nm_per_min": entry.through_water_speed_nm_per_min,
                            }
                            for entry in item.cycle
                        ],
                    }
                    for item in self.b6_reference.supply_templates
                ],
                "max_share_deficit_passages": self.b6_reference.max_share_deficit_passages,
                "fairness_comparison_tolerance": self.b6_reference.fairness_comparison_tolerance,
                "balance_tolerance_fraction": self.b6_reference.balance_tolerance_fraction,
                "warmup_min": self.b6_reference.warmup_min,
                "measurement_duration_min": self.b6_reference.measurement_duration_min,
                "drain_min": self.b6_reference.drain_min,
                "saturated_demands_per_stream": self.b6_reference.saturated_demands_per_stream,
            },
            "component_contract_ids": dict(self.component_contract_ids),
        }

    @property
    def contract_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return sha256(payload).hexdigest()


def _stream_geometry(value: object) -> StreamGeometry:
    if not isinstance(value, Mapping):
        raise ProfileContractError("each stream must be an object")
    _strict_keys("stream", value, {"stream_id", "source_to_center_distance_nm", "center_to_exit_distance_nm"})
    return StreamGeometry(
        stream_id=value["stream_id"],
        source_to_center_distance_nm=value["source_to_center_distance_nm"],
        center_to_exit_distance_nm=value["center_to_exit_distance_nm"],
    )


def _stream_share(value: object) -> StreamShare:
    if not isinstance(value, Mapping):
        raise ProfileContractError("each stream share must be an object")
    _strict_keys("stream share", value, {"stream_id", "share"})
    return StreamShare(stream_id=value["stream_id"], share=value["share"])


def _b6_supply_entry(value: object) -> B6SupplyTemplateEntry:
    if not isinstance(value, Mapping):
        raise ProfileContractError("each B6 supply cycle entry must be an object")
    _strict_keys(
        "B6 supply cycle entry",
        value,
        {"profile_class_id", "through_water_heading_deg", "through_water_speed_nm_per_min"},
    )
    return B6SupplyTemplateEntry(
        profile_class_id=value["profile_class_id"],
        through_water_heading_deg=value["through_water_heading_deg"],
        through_water_speed_nm_per_min=value["through_water_speed_nm_per_min"],
    )


def _b6_supply_template(value: object) -> B6StreamSupplyTemplate:
    if not isinstance(value, Mapping):
        raise ProfileContractError("each B6 supply template must be an object")
    _strict_keys("B6 supply template", value, {"stream_id", "cycle"})
    cycle = value["cycle"]
    if not isinstance(cycle, list) or not cycle:
        raise ProfileContractError("B6 supply template cycle must be a non-empty list")
    return B6StreamSupplyTemplate(
        stream_id=value["stream_id"],
        cycle=tuple(_b6_supply_entry(item) for item in cycle),
    )


def profile_contract_from_mapping(value: Mapping[str, object]) -> TwoStreamProfileContract:
    if not isinstance(value, Mapping):
        raise ProfileContractError("profile contract root must be an object")
    expected = {
        "schema",
        "profile_id",
        "profile_version",
        "base_framework_version",
        "runtime_adapter_id",
        "route_model_id",
        "environment_model_id",
        "planned_slot_prediction_scope_id",
        "crossing_angle_deg",
        "streams",
        "slot_solver",
        "b5_fairness",
        "b6_reference",
    }
    _strict_keys("profile", value, expected)

    streams_raw = value["streams"]
    if not isinstance(streams_raw, list) or len(streams_raw) != 2:
        raise ProfileContractError("streams must be a list of exactly two objects")
    streams = tuple(_stream_geometry(item) for item in streams_raw)

    slot = value["slot_solver"]
    if not isinstance(slot, Mapping):
        raise ProfileContractError("slot_solver must be an object")
    _strict_keys(
        "slot_solver",
        slot,
        {
            "separation_grid_min",
            "max_search_min",
            "clearance_time_tolerance_min",
            "clearance_margin_tolerance_nm",
            "max_clearance_subdivisions",
        },
    )

    fairness = value["b5_fairness"]
    if not isinstance(fairness, Mapping):
        raise ProfileContractError("b5_fairness must be an object")
    _strict_keys(
        "b5_fairness",
        fairness,
        {"lookahead_depth", "max_bypass_per_demand", "max_extra_wait_min"},
    )

    b6 = value["b6_reference"]
    if not isinstance(b6, Mapping):
        raise ProfileContractError("b6_reference must be an object")
    _strict_keys(
        "b6_reference",
        b6,
        {
            "stream_shares",
            "supply_templates",
            "max_share_deficit_passages",
            "fairness_comparison_tolerance",
            "balance_tolerance_fraction",
            "warmup_min",
            "measurement_duration_min",
            "drain_min",
            "saturated_demands_per_stream",
        },
    )
    shares_raw = b6["stream_shares"]
    if not isinstance(shares_raw, list) or len(shares_raw) != 2:
        raise ProfileContractError("b6_reference.stream_shares must contain exactly two objects")
    templates_raw = b6["supply_templates"]
    if not isinstance(templates_raw, list) or len(templates_raw) != 2:
        raise ProfileContractError("b6_reference.supply_templates must contain exactly two objects")

    return TwoStreamProfileContract(
        schema=value["schema"],
        profile_id=value["profile_id"],
        profile_version=value["profile_version"],
        base_framework_version=value["base_framework_version"],
        runtime_adapter_id=value["runtime_adapter_id"],
        route_model_id=value["route_model_id"],
        environment_model_id=value["environment_model_id"],
        planned_slot_prediction_scope_id=value["planned_slot_prediction_scope_id"],
        crossing_angle_deg=value["crossing_angle_deg"],
        streams=streams,  # type: ignore[arg-type]
        slot_solver=SafeSlotNumericalContract(
            separation_grid_min=slot["separation_grid_min"],
            max_search_min=slot["max_search_min"],
            clearance_time_tolerance_min=slot["clearance_time_tolerance_min"],
            clearance_margin_tolerance_nm=slot["clearance_margin_tolerance_nm"],
            max_clearance_subdivisions=slot["max_clearance_subdivisions"],
        ),
        b5_fairness=B5FairnessContract(
            lookahead_depth=fairness["lookahead_depth"],
            max_bypass_per_demand=fairness["max_bypass_per_demand"],
            max_extra_wait_min=fairness["max_extra_wait_min"],
        ),
        b6_reference=B6ReferenceContract(
            stream_shares=tuple(_stream_share(item) for item in shares_raw),  # type: ignore[arg-type]
            supply_templates=tuple(_b6_supply_template(item) for item in templates_raw),  # type: ignore[arg-type]
            max_share_deficit_passages=b6["max_share_deficit_passages"],
            fairness_comparison_tolerance=b6["fairness_comparison_tolerance"],
            balance_tolerance_fraction=b6["balance_tolerance_fraction"],
            warmup_min=b6["warmup_min"],
            measurement_duration_min=b6["measurement_duration_min"],
            drain_min=b6["drain_min"],
            saturated_demands_per_stream=b6["saturated_demands_per_stream"],
        ),
    )


def load_profile_contract(path: str | Path) -> TwoStreamProfileContract:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ProfileContractError("profile contract must use .json, .yaml, or .yml")
    if not isinstance(data, Mapping):
        raise ProfileContractError("profile contract root must be an object")
    return profile_contract_from_mapping(data)


def assert_compatible_base_framework() -> None:
    if _base_framework_version != BASE_FRAMEWORK_VERSION:
        raise RuntimeError(
            "profile is qualified only for vessel-crossing-simulator "
            f"{BASE_FRAMEWORK_VERSION}; loaded {_base_framework_version}"
        )
