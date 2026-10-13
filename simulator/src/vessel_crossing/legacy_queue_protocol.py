"""Frozen protocol for the legacy-faithful two-stream single-server study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .legacy_capacity import LegacyCapacityReference, legacy_capacity_reference


@dataclass(frozen=True, slots=True)
class LegacyQueueReplicationProtocol:
    protocol_id: str
    title: str
    hypothesis: str
    fcfs_scheduler: str
    priority_scheduler: str
    cpa_limit_nm: float
    speed_knots: float
    ordinary_multiplier: float
    privileged_multiplier: float
    privileged_fraction: float
    horizon_minutes: float
    demand_rates_per_hour: tuple[float, ...]
    base_seed: int
    replications: int
    confidence_level: float
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    paired_common_random_numbers: bool
    report_all_demand_points: bool
    report_both_percentage_denominators: bool
    deterministic_table_reference_speed_knots: float
    deterministic_table_reference_cpa_limits_nm: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "title",
            "hypothesis",
            "fcfs_scheduler",
            "priority_scheduler",
            "primary_metric",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.fcfs_scheduler == self.priority_scheduler:
            raise ValueError("scheduler identities must differ")
        if self.cpa_limit_nm <= 0 or self.speed_knots <= 0:
            raise ValueError("CPA limit and speed must be positive")
        if self.ordinary_multiplier <= 0 or self.privileged_multiplier <= 0:
            raise ValueError("service multipliers must be positive")
        if not 0 <= self.privileged_fraction <= 1:
            raise ValueError("privileged_fraction must be between 0 and 1")
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if not self.demand_rates_per_hour or any(rate <= 0 for rate in self.demand_rates_per_hour):
            raise ValueError("demand rates must be positive")
        if any(right <= left for left, right in zip(self.demand_rates_per_hour, self.demand_rates_per_hour[1:])):
            raise ValueError("demand rates must be strictly increasing")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        if self.replications < 2:
            raise ValueError("replications must be at least 2")
        if abs(self.confidence_level - 0.95) > 1e-12:
            raise ValueError("current queue summaries are fixed to 95% confidence")
        if len(set(self.secondary_metrics)) != len(self.secondary_metrics):
            raise ValueError("secondary metrics must be unique")
        if self.primary_metric in self.secondary_metrics:
            raise ValueError("primary metric must not be duplicated as secondary")
        if not self.paired_common_random_numbers:
            raise ValueError("paired common random numbers are required")
        if not self.report_all_demand_points:
            raise ValueError("all pre-specified demand points must be reported")
        if not self.report_both_percentage_denominators:
            raise ValueError("both 20%/25% denominator conventions must be reported")
        if self.deterministic_table_reference_speed_knots <= 0:
            raise ValueError("table reference speed must be positive")
        refs = self.deterministic_table_reference_cpa_limits_nm
        if not refs or any(value <= 0 for value in refs):
            raise ValueError("table reference CPA values must be positive")
        if any(right <= left for left, right in zip(refs, refs[1:])):
            raise ValueError("table reference CPA values must be strictly increasing")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def pretty_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    def fingerprint_sha256(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def capacity_reference(self) -> LegacyCapacityReference:
        return legacy_capacity_reference(
            cpa_limit_nm=self.cpa_limit_nm,
            speed_knots=self.speed_knots,
            privileged_fraction=self.privileged_fraction,
            ordinary_multiplier=self.ordinary_multiplier,
            privileged_multiplier=self.privileged_multiplier,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LegacyQueueReplicationProtocol":
        raw = dict(value)
        raw["demand_rates_per_hour"] = tuple(raw["demand_rates_per_hour"])
        raw["secondary_metrics"] = tuple(raw["secondary_metrics"])
        raw["deterministic_table_reference_cpa_limits_nm"] = tuple(
            raw["deterministic_table_reference_cpa_limits_nm"]
        )
        return cls(**raw)

    @classmethod
    def from_json(cls, text: str) -> "LegacyQueueReplicationProtocol":
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("protocol JSON root must be an object")
        return cls.from_dict(value)


def build_legacy_single_server_protocol_v1() -> LegacyQueueReplicationProtocol:
    return LegacyQueueReplicationProtocol(
        protocol_id="legacy-single-server-fcfs-vs-priority-reservation-v1",
        title="Legacy two-stream FCFS vs priority-reservation replication",
        hypothesis=(
            "Under the recovered equal-stream single-server assumptions, the deterministic "
            "Table-2 capacity contrast should reproduce the source 80% priority/FCFS ratio. "
            "Separately, under frozen equal-split Poisson demand, FCFS should not have lower "
            "finite-window throughput than the reconstructed priority-reservation discipline."
        ),
        fcfs_scheduler="legacy_single_server_fcfs",
        priority_scheduler="legacy_single_server_priority_reservation",
        cpa_limit_nm=1.0,
        speed_knots=6.0,
        ordinary_multiplier=3.0,
        privileged_multiplier=4.5,
        privileged_fraction=0.5,
        horizon_minutes=10080.0,
        demand_rates_per_hour=(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4),
        base_seed=2026090326,
        replications=500,
        confidence_level=0.95,
        primary_metric="throughput_difference_fcfs_minus_priority_per_hour",
        secondary_metrics=(
            "fcfs_throughput_per_hour",
            "priority_throughput_per_hour",
            "fcfs_mean_waiting_time",
            "priority_mean_waiting_time",
            "fcfs_post_window_clearance_time",
            "priority_post_window_clearance_time",
            "fcfs_completion_fraction_in_window",
            "priority_completion_fraction_in_window",
        ),
        paired_common_random_numbers=True,
        report_all_demand_points=True,
        report_both_percentage_denominators=True,
        deterministic_table_reference_speed_knots=12.0,
        deterministic_table_reference_cpa_limits_nm=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    )


def save_legacy_queue_protocol(protocol: LegacyQueueReplicationProtocol, path: str | Path) -> None:
    Path(path).write_text(protocol.pretty_json(), encoding="utf-8")


def load_legacy_queue_protocol(path: str | Path) -> LegacyQueueReplicationProtocol:
    return LegacyQueueReplicationProtocol.from_json(Path(path).read_text(encoding="utf-8"))
