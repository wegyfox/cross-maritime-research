"""Frozen protocol for completing the trajectory experiment planned in 2020."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LegacyTrajectoryProtocol:
    protocol_id: str
    baseline_policy: str
    comparison_policy: str
    route_names: tuple[str, ...]
    total_demand_rates_per_minute: tuple[float, ...]
    horizon_minutes: float
    desired_speed_knots: float
    safety_min_dcpa_nm: float
    safety_tcpa_horizon_minutes: float
    max_acceleration_knots_per_minute: float
    max_deceleration_knots_per_minute: float
    same_stream_min_spacing_nm: float
    conflict_zone_x_min_nm: float
    conflict_zone_x_max_nm: float
    conflict_zone_y_min_nm: float
    conflict_zone_y_max_nm: float
    reduced_speed_fraction: float
    base_seed: int
    replications: int
    confidence_level: float
    metrics: tuple[str, ...]
    exploratory_design_note: str

    def __post_init__(self) -> None:
        if not self.protocol_id.strip():
            raise ValueError("protocol_id must be non-empty")
        if (self.baseline_policy, self.comparison_policy) != (
            "simplified_rule15_speed",
            "stern_passer_yield_speed",
        ):
            raise ValueError("v1 policy identities are frozen")
        if self.route_names != ("A", "B", "C", "D"):
            raise ValueError("v1 route order is frozen to A/B/C/D")
        if not self.total_demand_rates_per_minute or tuple(sorted(self.total_demand_rates_per_minute)) != self.total_demand_rates_per_minute:
            raise ValueError("demand grid must be non-empty and sorted")
        if len(set(self.total_demand_rates_per_minute)) != len(self.total_demand_rates_per_minute):
            raise ValueError("demand rates must be unique")
        numeric_positive = (
            self.horizon_minutes,
            self.desired_speed_knots,
            self.safety_min_dcpa_nm,
            self.safety_tcpa_horizon_minutes,
            self.max_acceleration_knots_per_minute,
            self.max_deceleration_knots_per_minute,
            self.same_stream_min_spacing_nm,
        )
        if any(not isfinite(value) or value <= 0 for value in numeric_positive):
            raise ValueError("frozen positive numerical parameters must be finite and positive")
        if any(not isfinite(value) or value <= 0 for value in self.total_demand_rates_per_minute):
            raise ValueError("demand rates must be finite and positive")
        if not 0 < self.reduced_speed_fraction < 1:
            raise ValueError("reduced_speed_fraction must be between zero and one")
        if self.base_seed < 0 or self.replications < 2:
            raise ValueError("base_seed must be non-negative and replications >= 2")
        if self.confidence_level != 0.95:
            raise ValueError("v1 implementation freezes 95% Student-t intervals")
        required = {
            "crossing_throughput_per_hour",
            "maneuvers_per_generated",
            "stops_per_generated",
            "mean_route_delay_minutes",
            "mean_source_waiting_delay_minutes",
        }
        if set(self.metrics) != required:
            raise ValueError("v1 metric set is frozen")
        if not self.exploratory_design_note.strip():
            raise ValueError("exploratory_design_note must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def pretty_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    def fingerprint_sha256(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LegacyTrajectoryProtocol":
        raw = dict(data)
        raw["route_names"] = tuple(raw["route_names"])
        raw["total_demand_rates_per_minute"] = tuple(raw["total_demand_rates_per_minute"])
        raw["metrics"] = tuple(raw["metrics"])
        return cls(**raw)

    @classmethod
    def from_json(cls, text: str) -> "LegacyTrajectoryProtocol":
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("protocol JSON root must be an object")
        return cls.from_dict(value)


def build_legacy_trajectory_protocol_v1() -> LegacyTrajectoryProtocol:
    return LegacyTrajectoryProtocol(
        protocol_id="legacy-trajectory-rule15-vs-stern-yield-v1",
        baseline_policy="simplified_rule15_speed",
        comparison_policy="stern_passer_yield_speed",
        route_names=("A", "B", "C", "D"),
        # Design frozen after a clearly labeled 3-rep x 360-min exploratory
        # run.  No confirmatory statistics were used as an acceptance target.
        total_demand_rates_per_minute=(0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12),
        horizon_minutes=1440.0,
        desired_speed_knots=6.0,
        safety_min_dcpa_nm=1.0,
        safety_tcpa_horizon_minutes=120.0,
        max_acceleration_knots_per_minute=1.0,
        max_deceleration_knots_per_minute=1.0,
        same_stream_min_spacing_nm=1.0,
        conflict_zone_x_min_nm=4.0,
        conflict_zone_x_max_nm=6.0,
        conflict_zone_y_min_nm=4.0,
        conflict_zone_y_max_nm=6.0,
        reduced_speed_fraction=2.0 / 3.0,
        base_seed=20260905,
        replications=30,
        confidence_level=0.95,
        metrics=(
            "crossing_throughput_per_hour",
            "maneuvers_per_generated",
            "stops_per_generated",
            "mean_route_delay_minutes",
            "mean_source_waiting_delay_minutes",
        ),
        exploratory_design_note=(
            "Demand grid selected after a labeled engineering/exploratory sweep "
            "at 0.02/0.04/0.08/0.12 per minute with 3 replications and a 360-minute "
            "horizon; exploratory numerical effects are not confirmatory findings "
            "and are not stopping or acceptance criteria."
        ),
    )


def save_legacy_trajectory_protocol(protocol: LegacyTrajectoryProtocol, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(protocol.pretty_json(), encoding="utf-8")
    return path


def load_legacy_trajectory_protocol(path: str | Path) -> LegacyTrajectoryProtocol:
    return LegacyTrajectoryProtocol.from_json(Path(path).read_text(encoding="utf-8"))
