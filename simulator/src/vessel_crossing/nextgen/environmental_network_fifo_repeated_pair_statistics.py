"""Repeated-pair statistical inference over the frozen A159 FIFO estimand.

A160 does not define new per-pair outcomes.  Every replication executes one complete
A159 physical-outcome pair at a deterministic child seed.  The scientific vector is
exactly the union of:

* A159 control-minus-baseline physical route/safety outcomes; and
* A153 control-minus-baseline source-queue outcomes.

Runtime counters remain diagnostic evidence and are deliberately excluded.  Optional
per-pair outcomes retain explicit missingness; summaries never silently drop missing
replications without recording the count.  Two-sided confidence intervals are fixed to
95% Student-t intervals over the observed replication-level deltas.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from math import sqrt
from typing import Any, Iterable

from scipy.stats import t as student_t

from ..archive import environment_manifest
from ..experiments import replication_seeds
from ..version import __version__
from .environmental_network_experiment import _canonical_json, _json_value
from .environmental_network_fifo_paired_experiment import (
    EnvironmentalNetworkFIFOPairedExperimentConfig,
)
from .environmental_network_fifo_physical_outcomes import (
    EnvironmentalNetworkFIFOPhysicalOutcomeConfig,
    EnvironmentalNetworkFIFOPhysicalOutcomeResult,
    run_environmental_network_fifo_physical_outcomes,
)

_FIFO_REPEATED_CONFIG_FORMAT = (
    "vessel-crossing-nextgen-environmental-network-fifo-repeated-pair-experiment"
)
_FIFO_REPEATED_CONFIG_SCHEMA_VERSION = 1
_FIFO_REPEATED_RESULT_FORMAT = (
    "vessel-crossing-nextgen-environmental-network-fifo-repeated-pair-result"
)
_FIFO_REPEATED_RESULT_SCHEMA_VERSION = 1
_FIFO_REPEATED_DOMAIN = b"vessel-crossing-nextgen-environmental-network-fifo-repeated-pair\x00"
_CONFIDENCE_LEVEL = 0.95
_EPS = 1e-12


class EnvironmentalNetworkFIFORepeatedPairStatisticsError(RuntimeError):
    """Raised when A160 repeated-pair evidence cannot be built consistently."""


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFORepeatedPairStatisticsConfig:
    """One frozen A159 estimand plus a deterministic replication schedule.

    The seed already present in ``physical.paired`` is the root seed for A160.  A160
    never executes that seed directly; it derives deterministic independent child seeds
    with :func:`vessel_crossing.experiments.replication_seeds`.
    """

    physical: EnvironmentalNetworkFIFOPhysicalOutcomeConfig
    replications: int
    confidence_level: float = _CONFIDENCE_LEVEL
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.physical, EnvironmentalNetworkFIFOPhysicalOutcomeConfig):
            raise TypeError("physical must be EnvironmentalNetworkFIFOPhysicalOutcomeConfig")
        if isinstance(self.replications, bool) or not isinstance(self.replications, int):
            raise TypeError("replications must be an integer")
        if self.replications < 2:
            raise ValueError("replications must be at least 2")
        if abs(self.confidence_level - _CONFIDENCE_LEVEL) > _EPS:
            raise ValueError("A160 freezes two-sided 95% Student-t intervals")
        if self.label is not None:
            label = self.label.strip()
            if not label:
                raise ValueError("label must be non-empty when supplied")
            object.__setattr__(self, "label", label)

    @property
    def base_seed(self) -> int:
        return self.physical.paired.baseline.scenario.seed

    @property
    def child_seeds(self) -> tuple[int, ...]:
        return replication_seeds(base_seed=self.base_seed, count=self.replications)

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _FIFO_REPEATED_CONFIG_FORMAT,
            "schema_version": _FIFO_REPEATED_CONFIG_SCHEMA_VERSION,
            "label": self.label,
            "confidence_level": self.confidence_level,
            "replications": self.replications,
            "base_seed": self.base_seed,
            "child_seeds": list(self.child_seeds),
            "physical_estimand_config_sha256": self.physical.config_sha256,
            "physical_estimand_pair_id": self.physical.pair_id,
            "physical": self.physical.to_json_object(),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def config_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()

    @property
    def experiment_id(self) -> str:
        material = (
            _FIFO_REPEATED_DOMAIN
            + __version__.encode("utf-8")
            + b"\x00"
            + self.config_sha256.encode("ascii")
        )
        return sha256(material).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFORepeatedPairDifferenceVector:
    """Complete A160 scientific control-minus-baseline vector for one replication."""

    route_completed_count: float
    route_completion_fraction_of_demand: float | None
    route_completion_fraction_of_admitted: float | None
    route_completion_rate_per_hour: float | None
    mean_route_duration_min: float | None
    mean_system_duration_min: float | None
    final_vessel_count: float
    modeled_hull_overlap_episode_count: float
    safety_domain_intrusion_episode_count: float
    near_miss_episode_count: float
    unresolved_episode_count: float
    nonclear_pair_exposure_min: float
    source_admission_count: float
    source_pending_count: float
    source_due_pending_count: float
    source_mean_admitted_waiting_time_min: float | None
    source_max_admitted_waiting_time_min: float | None
    source_mean_due_waiting_time_min: float | None
    source_max_due_waiting_time_min: float | None


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFORepeatedPairReplication:
    replication_index: int
    seed: int
    physical_pair_id: str
    physical_config_sha256: str
    physical_result_sha256: str
    difference: EnvironmentalNetworkFIFORepeatedPairDifferenceVector


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFORepeatedPairScalarSummary:
    """One metric summary with explicit missingness across all requested replications."""

    metric_name: str
    replication_count: int
    observed_count: int
    missing_count: int
    mean: float | None
    sample_std: float | None
    standard_error: float | None
    confidence_level: float
    ci_low: float | None
    ci_high: float | None


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFORepeatedPairStatisticsResult:
    config: EnvironmentalNetworkFIFORepeatedPairStatisticsConfig
    replications: tuple[EnvironmentalNetworkFIFORepeatedPairReplication, ...]
    metric_summaries: tuple[EnvironmentalNetworkFIFORepeatedPairScalarSummary, ...]
    canonical_bytes: bytes = field(repr=False)

    @property
    def experiment_id(self) -> str:
        return self.config.experiment_id

    @property
    def config_sha256(self) -> str:
        return self.config.config_sha256

    def to_json_object(self) -> dict[str, Any]:
        document = json.loads(self.canonical_bytes.decode("utf-8"))
        if not isinstance(document, dict):  # pragma: no cover - construction invariant
            raise EnvironmentalNetworkFIFORepeatedPairStatisticsError(
                "frozen repeated-pair result is not a JSON object"
            )
        return document

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes

    @property
    def result_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()

    def summary_for(self, metric_name: str) -> EnvironmentalNetworkFIFORepeatedPairScalarSummary:
        for item in self.metric_summaries:
            if item.metric_name == metric_name:
                return item
        raise KeyError(metric_name)


_METRIC_NAMES = tuple(EnvironmentalNetworkFIFORepeatedPairDifferenceVector.__dataclass_fields__)


def _physical_config_with_seed(
    config: EnvironmentalNetworkFIFOPhysicalOutcomeConfig,
    seed: int,
) -> EnvironmentalNetworkFIFOPhysicalOutcomeConfig:
    pair = config.paired
    baseline = replace(
        pair.baseline,
        scenario=replace(pair.baseline.scenario, seed=seed),
    )
    control = replace(
        pair.control,
        scenario=replace(pair.control.scenario, seed=seed),
    )
    paired = EnvironmentalNetworkFIFOPairedExperimentConfig(
        baseline,
        control,
        label=pair.label,
    )
    return replace(config, paired=paired)


def _difference_vector(
    result: EnvironmentalNetworkFIFOPhysicalOutcomeResult,
) -> EnvironmentalNetworkFIFORepeatedPairDifferenceVector:
    physical = result.difference
    source = result.paired.difference.source_queue
    return EnvironmentalNetworkFIFORepeatedPairDifferenceVector(
        route_completed_count=float(physical.route_completed_count),
        route_completion_fraction_of_demand=physical.route_completion_fraction_of_demand,
        route_completion_fraction_of_admitted=physical.route_completion_fraction_of_admitted,
        route_completion_rate_per_hour=physical.route_completion_rate_per_hour,
        mean_route_duration_min=physical.mean_route_duration_min,
        mean_system_duration_min=physical.mean_system_duration_min,
        final_vessel_count=float(physical.final_vessel_count),
        modeled_hull_overlap_episode_count=float(physical.modeled_hull_overlap_episode_count),
        safety_domain_intrusion_episode_count=float(physical.safety_domain_intrusion_episode_count),
        near_miss_episode_count=float(physical.near_miss_episode_count),
        unresolved_episode_count=float(physical.unresolved_episode_count),
        nonclear_pair_exposure_min=physical.nonclear_pair_exposure_min,
        source_admission_count=float(source.admission_count_delta),
        source_pending_count=float(source.pending_count_delta),
        source_due_pending_count=float(source.due_pending_count_delta),
        source_mean_admitted_waiting_time_min=source.mean_admitted_waiting_time_min_delta,
        source_max_admitted_waiting_time_min=source.max_admitted_waiting_time_min_delta,
        source_mean_due_waiting_time_min=source.mean_due_waiting_time_min_delta,
        source_max_due_waiting_time_min=source.max_due_waiting_time_min_delta,
    )


def _scalar_summary(
    metric_name: str,
    values: Iterable[float | None],
    *,
    confidence_level: float = _CONFIDENCE_LEVEL,
) -> EnvironmentalNetworkFIFORepeatedPairScalarSummary:
    all_values = tuple(values)
    observed = tuple(float(value) for value in all_values if value is not None)
    replication_count = len(all_values)
    observed_count = len(observed)
    missing_count = replication_count - observed_count
    if observed_count == 0:
        return EnvironmentalNetworkFIFORepeatedPairScalarSummary(
            metric_name,
            replication_count,
            0,
            missing_count,
            None,
            None,
            None,
            confidence_level,
            None,
            None,
        )
    mean = sum(observed) / observed_count
    if observed_count < 2:
        return EnvironmentalNetworkFIFORepeatedPairScalarSummary(
            metric_name,
            replication_count,
            observed_count,
            missing_count,
            mean,
            None,
            None,
            confidence_level,
            None,
            None,
        )
    variance = sum((value - mean) ** 2 for value in observed) / (observed_count - 1)
    sample_std = sqrt(variance)
    standard_error = sample_std / sqrt(observed_count)
    critical = float(
        student_t.ppf((1.0 + confidence_level) / 2.0, df=observed_count - 1)
    )
    half_width = critical * standard_error
    return EnvironmentalNetworkFIFORepeatedPairScalarSummary(
        metric_name,
        replication_count,
        observed_count,
        missing_count,
        mean,
        sample_std,
        standard_error,
        confidence_level,
        mean - half_width,
        mean + half_width,
    )


def run_environmental_network_fifo_repeated_pair_statistics(
    config: EnvironmentalNetworkFIFORepeatedPairStatisticsConfig,
) -> EnvironmentalNetworkFIFORepeatedPairStatisticsResult:
    """Execute deterministic A159 replications and summarize the fixed paired vector."""

    if not isinstance(config, EnvironmentalNetworkFIFORepeatedPairStatisticsConfig):
        raise TypeError("config must be EnvironmentalNetworkFIFORepeatedPairStatisticsConfig")

    records: list[EnvironmentalNetworkFIFORepeatedPairReplication] = []
    vectors: list[EnvironmentalNetworkFIFORepeatedPairDifferenceVector] = []
    for index, seed in enumerate(config.child_seeds):
        child_config = _physical_config_with_seed(config.physical, seed)
        physical_result = run_environmental_network_fifo_physical_outcomes(child_config)
        # A159 is the CRN authority.  Recheck the object identity invariant at the A160
        # composition boundary so a future refactor cannot silently weaken it.
        traffic = physical_result.paired.shared_traffic
        if (
            physical_result.baseline.result.traffic is not traffic
            or physical_result.control.result.traffic is not traffic
        ):
            raise EnvironmentalNetworkFIFORepeatedPairStatisticsError(
                "A160 replication did not retain one exact A159 CRN traffic object"
            )
        vector = _difference_vector(physical_result)
        vectors.append(vector)
        records.append(
            EnvironmentalNetworkFIFORepeatedPairReplication(
                replication_index=index,
                seed=seed,
                physical_pair_id=physical_result.pair_id,
                physical_config_sha256=physical_result.config_sha256,
                physical_result_sha256=physical_result.result_sha256,
                difference=vector,
            )
        )

    summaries = tuple(
        _scalar_summary(
            metric_name,
            (getattr(vector, metric_name) for vector in vectors),
            confidence_level=config.confidence_level,
        )
        for metric_name in _METRIC_NAMES
    )
    document = {
        "format": _FIFO_REPEATED_RESULT_FORMAT,
        "schema_version": _FIFO_REPEATED_RESULT_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "config_sha256": config.config_sha256,
        "config": config.to_json_object(),
        "execution_environment": environment_manifest(),
        "replications": _json_value(tuple(records)),
        "metric_summaries": _json_value(summaries),
    }
    canonical_bytes = _canonical_json(document)
    return EnvironmentalNetworkFIFORepeatedPairStatisticsResult(
        config=config,
        replications=tuple(records),
        metric_summaries=summaries,
        canonical_bytes=canonical_bytes,
    )
