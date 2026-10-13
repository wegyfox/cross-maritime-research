"""Serializable experiment protocols for reproducible paper-grade runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .experiments import DemandSweepPoint
from .kinematics import knots_to_nm_per_minute
from .saturation import SaturationThresholds
from .scenarios import PoissonScenarioConfig, RouteDemand


@dataclass(frozen=True, slots=True)
class NamedSaturationProfile:
    name: str
    thresholds: SaturationThresholds

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("saturation profile name must be non-empty")


@dataclass(frozen=True, slots=True)
class ReplicationProtocol:
    """Frozen, serializable specification for one policy-comparison study."""

    protocol_id: str
    title: str
    hypothesis: str
    baseline_policy: str
    comparison_policy: str
    route_names: tuple[str, ...]
    desired_speed_knots: float
    horizon_minutes: float
    total_demand_rates_per_minute: tuple[float, ...]
    base_seed: int
    initial_replications: int
    replication_batch_size: int
    max_replications: int
    target_primary_ci_half_width: float
    confidence_level: float
    ci_precision_scope: str
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    capacity_proxy_definition: str
    legacy_reference_percent: float
    report_all_demand_points: bool
    paired_common_random_numbers: bool
    run_generated_cohort_to_clearance: bool
    primary_saturation_profile: str
    saturation_profiles: tuple[NamedSaturationProfile, ...]

    def __post_init__(self) -> None:
        text_fields = {
            "protocol_id": self.protocol_id,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "baseline_policy": self.baseline_policy,
            "comparison_policy": self.comparison_policy,
            "primary_metric": self.primary_metric,
            "ci_precision_scope": self.ci_precision_scope,
            "capacity_proxy_definition": self.capacity_proxy_definition,
            "primary_saturation_profile": self.primary_saturation_profile,
        }
        for name, value in text_fields.items():
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.baseline_policy == self.comparison_policy:
            raise ValueError("baseline and comparison policies must differ")
        if not self.route_names or any(not name.strip() for name in self.route_names):
            raise ValueError("route_names must be non-empty")
        if len(set(self.route_names)) != len(self.route_names):
            raise ValueError("route_names must be unique")
        if self.desired_speed_knots <= 0:
            raise ValueError("desired_speed_knots must be positive")
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        rates = self.total_demand_rates_per_minute
        if not rates or any(rate <= 0 for rate in rates):
            raise ValueError("total demand rates must be positive")
        if any(right <= left for left, right in zip(rates, rates[1:])):
            raise ValueError("total demand rates must be strictly increasing")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        if self.initial_replications < 2:
            raise ValueError("initial_replications must be >= 2")
        if self.replication_batch_size < 1:
            raise ValueError("replication_batch_size must be >= 1")
        if self.max_replications < self.initial_replications:
            raise ValueError("max_replications cannot be below initial_replications")
        if self.target_primary_ci_half_width <= 0:
            raise ValueError("target_primary_ci_half_width must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if abs(self.confidence_level - 0.95) > 1e-12:
            raise ValueError("current replication summaries are fixed to 95% confidence")
        if self.legacy_reference_percent <= 0:
            raise ValueError("legacy_reference_percent must be positive")
        if not self.report_all_demand_points:
            raise ValueError("paper-grade protocol must report all pre-specified demand points")
        if not self.paired_common_random_numbers:
            raise ValueError("this protocol family requires paired common random numbers")
        if not self.run_generated_cohort_to_clearance:
            raise ValueError("this protocol family requires run-to-clear cohort handling")

        if len(set(self.secondary_metrics)) != len(self.secondary_metrics):
            raise ValueError("secondary_metrics must be unique")
        if self.primary_metric in self.secondary_metrics:
            raise ValueError("primary metric must not be duplicated as secondary")

        profile_names = [profile.name for profile in self.saturation_profiles]
        if not profile_names or len(profile_names) != len(set(profile_names)):
            raise ValueError("saturation profile names must be non-empty and unique")
        if self.primary_saturation_profile not in profile_names:
            raise ValueError("primary saturation profile must exist")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with no runtime-only objects."""
        return asdict(self)

    def canonical_json(self) -> str:
        """Stable byte-equivalent JSON used for protocol fingerprints."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def pretty_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    def fingerprint_sha256(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def saturation_profile(self, name: str | None = None) -> NamedSaturationProfile:
        target = self.primary_saturation_profile if name is None else name
        for profile in self.saturation_profiles:
            if profile.name == target:
                return profile
        raise KeyError(target)

    def demand_points(self) -> tuple[DemandSweepPoint, ...]:
        """Build the frozen balanced route-demand grid from this protocol."""
        per_route_divisor = len(self.route_names)
        speed = knots_to_nm_per_minute(self.desired_speed_knots)
        return tuple(
            DemandSweepPoint(
                name=f"total_{total_rate:.3f}_per_min",
                scenario_config=PoissonScenarioConfig(
                    horizon=self.horizon_minutes,
                    desired_speed=speed,
                    route_demands=tuple(
                        RouteDemand(route_name, total_rate / per_route_divisor)
                        for route_name in self.route_names
                    ),
                ),
            )
            for total_rate in self.total_demand_rates_per_minute
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplicationProtocol":
        raw = dict(data)
        raw["route_names"] = tuple(raw["route_names"])
        raw["total_demand_rates_per_minute"] = tuple(raw["total_demand_rates_per_minute"])
        raw["secondary_metrics"] = tuple(raw["secondary_metrics"])
        raw["saturation_profiles"] = tuple(
            NamedSaturationProfile(
                name=item["name"],
                thresholds=SaturationThresholds(**item["thresholds"]),
            )
            for item in raw["saturation_profiles"]
        )
        return cls(**raw)

    @classmethod
    def from_json(cls, text: str) -> "ReplicationProtocol":
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("protocol JSON root must be an object")
        return cls.from_dict(value)


def load_replication_protocol(path: str | Path) -> ReplicationProtocol:
    return ReplicationProtocol.from_json(Path(path).read_text(encoding="utf-8"))


def save_replication_protocol(protocol: ReplicationProtocol, path: str | Path) -> None:
    Path(path).write_text(protocol.pretty_json(), encoding="utf-8")


def build_legacy_fcfs_vs_rule15_protocol_v1() -> ReplicationProtocol:
    """Return the first frozen balanced four-stream replication protocol."""

    strict = NamedSaturationProfile(
        "strict",
        SaturationThresholds(
            min_crossing_service_fraction=0.95,
            max_mean_active_approach_count=3.0,
            max_mean_conflict_zone_entry_delay=10.0,
            max_post_window_conflict_zone_clearance_time=60.0,
            max_starved_route_count_in_window=0.0,
            plateau_relative_throughput_gain=0.03,
            decline_relative_tolerance=0.02,
            min_congestion_reasons_for_saturation=2,
        ),
    )
    nominal = NamedSaturationProfile(
        "nominal",
        SaturationThresholds(
            min_crossing_service_fraction=0.90,
            max_mean_active_approach_count=4.0,
            max_mean_conflict_zone_entry_delay=20.0,
            max_post_window_conflict_zone_clearance_time=120.0,
            max_starved_route_count_in_window=0.0,
            plateau_relative_throughput_gain=0.05,
            decline_relative_tolerance=0.02,
            min_congestion_reasons_for_saturation=2,
        ),
    )
    lenient = NamedSaturationProfile(
        "lenient",
        SaturationThresholds(
            min_crossing_service_fraction=0.80,
            max_mean_active_approach_count=6.0,
            max_mean_conflict_zone_entry_delay=30.0,
            max_post_window_conflict_zone_clearance_time=180.0,
            max_starved_route_count_in_window=0.0,
            plateau_relative_throughput_gain=0.08,
            decline_relative_tolerance=0.03,
            min_congestion_reasons_for_saturation=2,
        ),
    )

    return ReplicationProtocol(
        protocol_id="legacy-fcfs-vs-rule15-v1",
        title="Balanced four-stream FCFS vs simplified Rule-15 crossing replication",
        hypothesis=(
            "Under a frozen balanced four-stream demand grid, the FCFS speed-control "
            "baseline may improve crossing throughput and/or delay relative to the "
            "simplified Rule-15 speed-control baseline; the legacy ~20% capacity claim "
            "is treated as a hypothesis and may fail to replicate."
        ),
        baseline_policy="simplified_rule15_speed",
        comparison_policy="fcfs_speed",
        route_names=("A", "B", "C", "D"),
        desired_speed_knots=6.0,
        horizon_minutes=1440.0,
        total_demand_rates_per_minute=(0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24),
        base_seed=20260903,
        initial_replications=30,
        replication_batch_size=10,
        max_replications=120,
        target_primary_ci_half_width=0.25,
        confidence_level=0.95,
        ci_precision_scope="paired primary-metric CI half-width at every demand point",
        primary_metric="crossing_throughput_per_hour",
        secondary_metrics=(
            "crossing_service_fraction_in_window",
            "mean_conflict_zone_entry_delay",
            "mean_conflict_zone_exit_delay",
            "mean_active_approach_count",
            "peak_active_approach_backlog",
            "post_window_conflict_zone_clearance_time",
            "service_fraction_jain_index",
            "starved_route_count_in_window",
            "max_abs_exit_share_error",
        ),
        capacity_proxy_definition=(
            "highest pre-specified total demand rate classified stable or strained "
            "under the nominal saturation profile"
        ),
        legacy_reference_percent=20.0,
        report_all_demand_points=True,
        paired_common_random_numbers=True,
        run_generated_cohort_to_clearance=True,
        primary_saturation_profile="nominal",
        saturation_profiles=(strict, nominal, lenient),
    )


@dataclass(frozen=True, slots=True)
class ExecutableModelParameters:
    """Numerical model parameters required to execute a study protocol.

    v0.19 froze the statistical study design but left several simulation
    defaults outside the protocol fingerprint.  The executable protocol keeps
    those choices explicit so a confirmatory run cannot silently change when a
    code default changes.
    """

    geometry_name: str
    safety_min_dcpa_nm: float
    safety_tcpa_horizon_minutes: float | None
    max_acceleration_knots_per_minute: float
    max_deceleration_knots_per_minute: float
    same_stream_min_spacing_nm: float
    conflict_zone_x_min_nm: float
    conflict_zone_x_max_nm: float
    conflict_zone_y_min_nm: float
    conflict_zone_y_max_nm: float
    policy_reduced_speed_fraction: float

    def __post_init__(self) -> None:
        if not self.geometry_name.strip():
            raise ValueError("geometry_name must be non-empty")
        if self.safety_min_dcpa_nm <= 0:
            raise ValueError("safety_min_dcpa_nm must be positive")
        if (
            self.safety_tcpa_horizon_minutes is not None
            and self.safety_tcpa_horizon_minutes <= 0
        ):
            raise ValueError("safety_tcpa_horizon_minutes must be positive when set")
        if self.max_acceleration_knots_per_minute < 0:
            raise ValueError("max acceleration must be non-negative")
        if self.max_deceleration_knots_per_minute < 0:
            raise ValueError("max deceleration must be non-negative")
        if self.same_stream_min_spacing_nm < 0:
            raise ValueError("same_stream_min_spacing_nm must be non-negative")
        if self.conflict_zone_x_max_nm <= self.conflict_zone_x_min_nm:
            raise ValueError("conflict zone x bounds are invalid")
        if self.conflict_zone_y_max_nm <= self.conflict_zone_y_min_nm:
            raise ValueError("conflict zone y bounds are invalid")
        if not 0 < self.policy_reduced_speed_fraction < 1:
            raise ValueError("policy_reduced_speed_fraction must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class ExecutableReplicationProtocol:
    """Study protocol plus every numerical model parameter needed to run it."""

    execution_protocol_id: str
    study: ReplicationProtocol
    model: ExecutableModelParameters
    supersedes_study_fingerprint: str | None = None
    supersession_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.execution_protocol_id.strip():
            raise ValueError("execution_protocol_id must be non-empty")
        if self.supersedes_study_fingerprint is not None:
            if len(self.supersedes_study_fingerprint) != 64:
                raise ValueError("supersedes_study_fingerprint must be SHA-256 hex")
            if not self.supersession_reason or not self.supersession_reason.strip():
                raise ValueError("supersession_reason is required when superseding")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def pretty_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    def fingerprint_sha256(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutableReplicationProtocol":
        raw = dict(data)
        study = ReplicationProtocol.from_dict(raw.pop("study"))
        model = ExecutableModelParameters(**raw.pop("model"))
        return cls(study=study, model=model, **raw)

    @classmethod
    def from_json(cls, text: str) -> "ExecutableReplicationProtocol":
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("executable protocol JSON root must be an object")
        return cls.from_dict(value)


def load_executable_replication_protocol(
    path: str | Path,
) -> ExecutableReplicationProtocol:
    return ExecutableReplicationProtocol.from_json(
        Path(path).read_text(encoding="utf-8")
    )


def save_executable_replication_protocol(
    protocol: ExecutableReplicationProtocol,
    path: str | Path,
) -> None:
    Path(path).write_text(protocol.pretty_json(), encoding="utf-8")


def build_legacy_fcfs_vs_rule15_execution_protocol_v2() -> ExecutableReplicationProtocol:
    """Execution-ready successor to the v1 study-only protocol.

    v1 remains immutable.  v2 exists because model parameters were discovered
    to be outside the v1 fingerprint before the confirmatory run was executed.
    """

    study = build_legacy_fcfs_vs_rule15_protocol_v1()
    return ExecutableReplicationProtocol(
        execution_protocol_id="legacy-fcfs-vs-rule15-v2",
        study=study,
        model=ExecutableModelParameters(
            geometry_name="diploma_reference_nm",
            safety_min_dcpa_nm=1.0,
            safety_tcpa_horizon_minutes=120.0,
            max_acceleration_knots_per_minute=1.0,
            max_deceleration_knots_per_minute=1.0,
            same_stream_min_spacing_nm=1.0,
            conflict_zone_x_min_nm=4.0,
            conflict_zone_x_max_nm=6.0,
            conflict_zone_y_min_nm=4.0,
            conflict_zone_y_max_nm=6.0,
            policy_reduced_speed_fraction=2.0 / 3.0,
        ),
        supersedes_study_fingerprint=study.fingerprint_sha256(),
        supersession_reason=(
            "The v1 study protocol froze statistical design but omitted numerical "
            "simulation/policy parameters. It was superseded before any confirmatory "
            "run so execution cannot depend on mutable code defaults."
        ),
    )
