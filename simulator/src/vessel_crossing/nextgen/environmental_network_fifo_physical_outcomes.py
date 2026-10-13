"""Single-pair physical outcome evidence for the corrected FIFO network model.

A159 deliberately sits *after* A153.  A153 already proves one exact CRN demand
realization and arm-local FIFO queue/runtime outcomes, but it does not freeze route
completion or continuous maritime safety outcomes.  This module executes the same two
A152 arms on one shared traffic realization while attaching read-only A139 continuous
safety telemetry.  It then freezes a separately named result family containing:

* the complete nested A153 paired evidence;
* demand/admission/completion timing for every completed route;
* per-arm route completion and duration metrics;
* continuous hull/domain/near-miss/unresolved safety intervals and mission summaries;
* control-minus-baseline physical differences.

It adds no campaign, confidence interval, stopping rule, JSON-to-runtime loader, restore,
or resume surface.  Those belong to later protocol layers after this per-pair estimand is
frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any

from ..archive import environment_manifest
from ..version import __version__
from .environmental_network_experiment import _canonical_json, _json_value
from .environmental_network_fifo_experiment import (
    EnvironmentalNetworkFIFOExperimentResult,
    _run_environmental_network_fifo_experiment_with_traffic,
)
from .environmental_network_fifo_paired_experiment import (
    EnvironmentalNetworkFIFOPairedExperimentConfig,
    EnvironmentalNetworkFIFOPairedExperimentResult,
    _assemble_environmental_network_fifo_paired_result,
)
from .environmental_network_paired_experiment import (
    EnvironmentalNetworkRouteCompletion,
    _MetricEnvironmentalWorld,
)
from .mission_safety import (
    MissionSafetyExperimentSummary,
    TimedPairSafetyIntervalMetric,
    summarize_mission_safety,
)
from .network_traffic import PoissonNetworkTrafficRealization, generate_poisson_network_traffic
from .safety_metrics import PairSafetyMetricConfig

_EPS = 1e-10
_FIFO_PHYSICAL_CONFIG_FORMAT = "vessel-crossing-nextgen-environmental-network-fifo-physical-paired-experiment"
_FIFO_PHYSICAL_CONFIG_SCHEMA_VERSION = 1
_FIFO_PHYSICAL_RESULT_FORMAT = "vessel-crossing-nextgen-environmental-network-fifo-physical-paired-result"
_FIFO_PHYSICAL_RESULT_SCHEMA_VERSION = 1
_FIFO_PHYSICAL_DOMAIN = b"vessel-crossing-nextgen-environmental-network-fifo-physical-pair\x00"


class EnvironmentalNetworkFIFOPhysicalOutcomeError(RuntimeError):
    """Raised when A159 physical evidence cannot be constructed consistently."""


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOPhysicalOutcomeConfig:
    """One A153 FIFO pair plus the continuous safety metric resolution for A159."""

    paired: EnvironmentalNetworkFIFOPairedExperimentConfig
    safety: PairSafetyMetricConfig
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paired, EnvironmentalNetworkFIFOPairedExperimentConfig):
            raise TypeError("paired must be EnvironmentalNetworkFIFOPairedExperimentConfig")
        if not isinstance(self.safety, PairSafetyMetricConfig):
            raise TypeError("safety must be PairSafetyMetricConfig")
        if self.label is not None:
            label = self.label.strip()
            if not label:
                raise ValueError("label must be non-empty when supplied")
            object.__setattr__(self, "label", label)

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _FIFO_PHYSICAL_CONFIG_FORMAT,
            "schema_version": _FIFO_PHYSICAL_CONFIG_SCHEMA_VERSION,
            "label": self.label,
            "paired_config_sha256": self.paired.config_sha256,
            "paired_pair_id": self.paired.pair_id,
            "paired": self.paired.to_json_object(),
            "safety": _json_value(self.safety),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def config_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()

    @property
    def pair_id(self) -> str:
        material = (
            _FIFO_PHYSICAL_DOMAIN
            + __version__.encode("utf-8")
            + b"\x00"
            + self.config_sha256.encode("ascii")
        )
        return sha256(material).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOPhysicalArmMetrics:
    """Physical outcomes for one FIFO arm at the common caller horizon."""

    realized_demand_count: int
    physical_admission_count: int
    route_completed_count: int
    route_completion_fraction_of_demand: float | None
    route_completion_fraction_of_admitted: float | None
    route_completion_rate_per_hour: float | None
    mean_route_duration_min: float | None
    mean_system_duration_min: float | None
    final_vessel_count: int
    safety: MissionSafetyExperimentSummary


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOPhysicalDifference:
    """Control-minus-baseline physical outcome differences."""

    route_completed_count: int
    route_completion_fraction_of_demand: float | None
    route_completion_fraction_of_admitted: float | None
    route_completion_rate_per_hour: float | None
    mean_route_duration_min: float | None
    mean_system_duration_min: float | None
    final_vessel_count: int
    modeled_hull_overlap_episode_count: int
    safety_domain_intrusion_episode_count: int
    near_miss_episode_count: int
    unresolved_episode_count: int
    nonclear_pair_exposure_min: float


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOPhysicalArmEvidence:
    result: EnvironmentalNetworkFIFOExperimentResult
    completions: tuple[EnvironmentalNetworkRouteCompletion, ...]
    safety_intervals: tuple[TimedPairSafetyIntervalMetric, ...]
    metrics: EnvironmentalNetworkFIFOPhysicalArmMetrics


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOPhysicalOutcomeResult:
    """A159 frozen evidence over one exact A153 CRN pair."""

    config: EnvironmentalNetworkFIFOPhysicalOutcomeConfig
    paired: EnvironmentalNetworkFIFOPairedExperimentResult
    baseline: EnvironmentalNetworkFIFOPhysicalArmEvidence
    control: EnvironmentalNetworkFIFOPhysicalArmEvidence
    difference: EnvironmentalNetworkFIFOPhysicalDifference
    canonical_bytes: bytes = field(repr=False)

    @property
    def pair_id(self) -> str:
        return self.config.pair_id

    @property
    def config_sha256(self) -> str:
        return self.config.config_sha256

    def to_json_object(self) -> dict[str, Any]:
        document = json.loads(self.canonical_bytes.decode("utf-8"))
        if not isinstance(document, dict):  # pragma: no cover - construction invariant
            raise EnvironmentalNetworkFIFOPhysicalOutcomeError(
                "frozen physical-outcome result is not a JSON object"
            )
        return document

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes

    @property
    def result_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()


def _route_completions(
    result: EnvironmentalNetworkFIFOExperimentResult,
) -> tuple[EnvironmentalNetworkRouteCompletion, ...]:
    admissions = {item.vessel_id: item for item in result.run.reservoir.admissions}
    items: list[EnvironmentalNetworkRouteCompletion] = []
    seen: set[str] = set()
    persistent = result.runtime_start.session.runtime_session.persistent
    for disposition in persistent.disposition_results:
        for vessel in disposition.dispositions:
            if vessel.vessel_id in seen:
                raise EnvironmentalNetworkFIFOPhysicalOutcomeError(
                    f"duplicate route completion evidence for {vessel.vessel_id!r}"
                )
            admission = admissions.get(vessel.vessel_id)
            if admission is None:
                raise EnvironmentalNetworkFIFOPhysicalOutcomeError(
                    f"route completion lacks FIFO admission evidence for {vessel.vessel_id!r}"
                )
            seen.add(vessel.vessel_id)
            items.append(
                EnvironmentalNetworkRouteCompletion(
                    vessel_id=vessel.vessel_id,
                    demand_time_min=admission.demand_time,
                    admission_time_min=admission.admission_time,
                    completion_time_min=disposition.committed_observation_time,
                    end_action=vessel.action,
                )
            )
    return tuple(sorted(items, key=lambda item: (item.completion_time_min, item.vessel_id)))


def _arm_metrics(
    result: EnvironmentalNetworkFIFOExperimentResult,
    completions: tuple[EnvironmentalNetworkRouteCompletion, ...],
    intervals: tuple[TimedPairSafetyIntervalMetric, ...],
) -> EnvironmentalNetworkFIFOPhysicalArmMetrics:
    demand_count = result.summary.realized_demand_count
    admission_count = result.summary.physical_admission_count
    completion_count = len(completions)
    duration = result.summary.end_time - result.summary.start_time
    return EnvironmentalNetworkFIFOPhysicalArmMetrics(
        realized_demand_count=demand_count,
        physical_admission_count=admission_count,
        route_completed_count=completion_count,
        route_completion_fraction_of_demand=(
            None if demand_count == 0 else completion_count / demand_count
        ),
        route_completion_fraction_of_admitted=(
            None if admission_count == 0 else completion_count / admission_count
        ),
        route_completion_rate_per_hour=(
            None if duration <= _EPS else completion_count / duration * 60.0
        ),
        mean_route_duration_min=(
            None
            if not completions
            else sum(item.route_duration_min for item in completions) / completion_count
        ),
        mean_system_duration_min=(
            None
            if not completions
            else sum(item.system_duration_min for item in completions) / completion_count
        ),
        final_vessel_count=result.summary.final_vessel_count,
        safety=summarize_mission_safety(intervals),
    )


def _difference(
    baseline: EnvironmentalNetworkFIFOPhysicalArmMetrics,
    control: EnvironmentalNetworkFIFOPhysicalArmMetrics,
) -> EnvironmentalNetworkFIFOPhysicalDifference:
    def optional_delta(left: float | None, right: float | None) -> float | None:
        return None if left is None or right is None else left - right

    cs = control.safety
    bs = baseline.safety
    return EnvironmentalNetworkFIFOPhysicalDifference(
        route_completed_count=control.route_completed_count - baseline.route_completed_count,
        route_completion_fraction_of_demand=optional_delta(
            control.route_completion_fraction_of_demand,
            baseline.route_completion_fraction_of_demand,
        ),
        route_completion_fraction_of_admitted=optional_delta(
            control.route_completion_fraction_of_admitted,
            baseline.route_completion_fraction_of_admitted,
        ),
        route_completion_rate_per_hour=optional_delta(
            control.route_completion_rate_per_hour,
            baseline.route_completion_rate_per_hour,
        ),
        mean_route_duration_min=optional_delta(
            control.mean_route_duration_min,
            baseline.mean_route_duration_min,
        ),
        mean_system_duration_min=optional_delta(
            control.mean_system_duration_min,
            baseline.mean_system_duration_min,
        ),
        final_vessel_count=control.final_vessel_count - baseline.final_vessel_count,
        modeled_hull_overlap_episode_count=(
            cs.modeled_hull_overlap_episode_count - bs.modeled_hull_overlap_episode_count
        ),
        safety_domain_intrusion_episode_count=(
            cs.safety_domain_intrusion_episode_count - bs.safety_domain_intrusion_episode_count
        ),
        near_miss_episode_count=cs.near_miss_episode_count - bs.near_miss_episode_count,
        unresolved_episode_count=cs.unresolved_episode_count - bs.unresolved_episode_count,
        nonclear_pair_exposure_min=(
            cs.nonclear_pair_exposure_min - bs.nonclear_pair_exposure_min
        ),
    )


def _run_arm_with_metrics(
    config,
    traffic: PoissonNetworkTrafficRealization,
    safety: PairSafetyMetricConfig,
) -> tuple[EnvironmentalNetworkFIFOExperimentResult, tuple[TimedPairSafetyIntervalMetric, ...]]:
    holder: dict[str, _MetricEnvironmentalWorld] = {}

    def world_factory(profiles):
        world = _MetricEnvironmentalWorld(
            (),
            limits={},
            environmental_drift_velocity={},
            start_time=config.scenario.start_time,
            profile_catalog=profiles,
            safety_config=safety,
        )
        holder["world"] = world
        return world

    result = _run_environmental_network_fifo_experiment_with_traffic(
        config,
        traffic,
        _world_factory=world_factory,
    )
    world = holder.get("world")
    if world is None:  # pragma: no cover - construction invariant
        raise EnvironmentalNetworkFIFOPhysicalOutcomeError("telemetry world was not constructed")
    return result, world.safety_intervals


def run_environmental_network_fifo_physical_outcomes(
    config: EnvironmentalNetworkFIFOPhysicalOutcomeConfig,
) -> EnvironmentalNetworkFIFOPhysicalOutcomeResult:
    """Run one exact-CRN A153 pair and freeze A159 route/safety outcomes."""

    if not isinstance(config, EnvironmentalNetworkFIFOPhysicalOutcomeConfig):
        raise TypeError("config must be EnvironmentalNetworkFIFOPhysicalOutcomeConfig")

    pair = config.paired
    base = pair.baseline
    scenario = base.scenario
    traffic = generate_poisson_network_traffic(
        base.network,
        base.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    baseline_result, baseline_intervals = _run_arm_with_metrics(
        pair.baseline, traffic, config.safety
    )
    control_result, control_intervals = _run_arm_with_metrics(
        pair.control, traffic, config.safety
    )
    paired = _assemble_environmental_network_fifo_paired_result(
        pair, traffic, baseline_result, control_result
    )

    baseline_completions = _route_completions(baseline_result)
    control_completions = _route_completions(control_result)
    baseline_metrics = _arm_metrics(
        baseline_result, baseline_completions, baseline_intervals
    )
    control_metrics = _arm_metrics(
        control_result, control_completions, control_intervals
    )
    difference = _difference(baseline_metrics, control_metrics)

    baseline = EnvironmentalNetworkFIFOPhysicalArmEvidence(
        result=baseline_result,
        completions=baseline_completions,
        safety_intervals=baseline_intervals,
        metrics=baseline_metrics,
    )
    control = EnvironmentalNetworkFIFOPhysicalArmEvidence(
        result=control_result,
        completions=control_completions,
        safety_intervals=control_intervals,
        metrics=control_metrics,
    )
    document = {
        "format": _FIFO_PHYSICAL_RESULT_FORMAT,
        "schema_version": _FIFO_PHYSICAL_RESULT_SCHEMA_VERSION,
        "pair_id": config.pair_id,
        "config_sha256": config.config_sha256,
        "config": config.to_json_object(),
        "execution_environment": environment_manifest(),
        "paired_fifo": paired.to_json_object(),
        "physical_outcomes": {
            "baseline": {
                "completions": _json_value(baseline_completions),
                "safety_intervals": _json_value(baseline_intervals),
                "metrics": _json_value(baseline_metrics),
            },
            "control": {
                "completions": _json_value(control_completions),
                "safety_intervals": _json_value(control_intervals),
                "metrics": _json_value(control_metrics),
            },
            "difference": _json_value(difference),
        },
    }
    canonical_bytes = _canonical_json(document)
    return EnvironmentalNetworkFIFOPhysicalOutcomeResult(
        config=config,
        paired=paired,
        baseline=baseline,
        control=control,
        difference=difference,
        canonical_bytes=canonical_bytes,
    )
