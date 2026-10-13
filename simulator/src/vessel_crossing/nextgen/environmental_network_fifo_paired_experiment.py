"""Paired common-random-number experiments for the separately named FIFO source model.

A153 shares exactly one exogenous Poisson traffic realization between two A152 FIFO
experiment arms.  Each arm owns its deterministic source admission/waiting response.
The frozen A139/A141 comparative schemas are intentionally not reused or modified.
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
    EnvironmentalNetworkFIFOExperimentConfig,
    EnvironmentalNetworkFIFOExperimentResult,
    _run_environmental_network_fifo_experiment_with_traffic,
)
from .environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceQueueMetricDifference,
    compare_source_queue_metrics,
)
from .network_traffic import PoissonNetworkTrafficRealization, generate_poisson_network_traffic

_FIFO_PAIR_CONFIG_FORMAT = "vessel-crossing-nextgen-environmental-network-fifo-paired-experiment"
_FIFO_PAIR_CONFIG_SCHEMA_VERSION = 1
_FIFO_PAIR_RESULT_FORMAT = "vessel-crossing-nextgen-environmental-network-fifo-paired-result"
_FIFO_PAIR_RESULT_SCHEMA_VERSION = 1
_FIFO_PAIR_DOMAIN = b"vessel-crossing-nextgen-environmental-network-fifo-pair\x00"


class EnvironmentalNetworkFIFOPairedExperimentError(RuntimeError):
    """Raised when two FIFO arms do not form one valid CRN comparison."""


def _shared_exogenous_basis(config: EnvironmentalNetworkFIFOExperimentConfig) -> dict[str, Any]:
    scenario = config.scenario
    return {
        "network": _json_value(config.network),
        "streams": _json_value(config.streams),
        "seed": scenario.seed,
        "start_time": scenario.start_time,
        "traffic_horizon_min": scenario.traffic_horizon_min,
        "end_time": float(scenario.end_time),
        "default_environmental_drift_velocity": _json_value(
            scenario.default_environmental_drift_velocity
        ),
        "stream_environmental_drift_velocity": _json_value(
            scenario.stream_environmental_drift_velocity
        ),
        "admission_policy": config.admission_policy.value,
        "source_recheck": _json_value(config.recheck),
    }


def _shared_basis_sha256(config: EnvironmentalNetworkFIFOExperimentConfig) -> str:
    return sha256(_canonical_json(_shared_exogenous_basis(config))).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOPairedExperimentConfig:
    """Two treatment configs over one exact exogenous FIFO demand/environment basis."""

    baseline: EnvironmentalNetworkFIFOExperimentConfig
    control: EnvironmentalNetworkFIFOExperimentConfig
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, EnvironmentalNetworkFIFOExperimentConfig):
            raise TypeError("baseline must be EnvironmentalNetworkFIFOExperimentConfig")
        if not isinstance(self.control, EnvironmentalNetworkFIFOExperimentConfig):
            raise TypeError("control must be EnvironmentalNetworkFIFOExperimentConfig")
        left = _shared_exogenous_basis(self.baseline)
        right = _shared_exogenous_basis(self.control)
        if left != right:
            raise EnvironmentalNetworkFIFOPairedExperimentError(
                "FIFO paired arms must share the exact exogenous network/traffic/environment/recheck basis"
            )
        if self.label is not None:
            label = self.label.strip()
            if not label:
                raise ValueError("label must be non-empty when supplied")
            object.__setattr__(self, "label", label)

    @property
    def shared_basis_sha256(self) -> str:
        return _shared_basis_sha256(self.baseline)

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _FIFO_PAIR_CONFIG_FORMAT,
            "schema_version": _FIFO_PAIR_CONFIG_SCHEMA_VERSION,
            "label": self.label,
            "shared_basis_sha256": self.shared_basis_sha256,
            "shared_exogenous_basis": _shared_exogenous_basis(self.baseline),
            "baseline": self.baseline.to_json_object(),
            "control": self.control.to_json_object(),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def config_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()

    @property
    def pair_id(self) -> str:
        material = (
            _FIFO_PAIR_DOMAIN
            + __version__.encode("utf-8")
            + b"\x00"
            + self.config_sha256.encode("ascii")
        )
        return sha256(material).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOPairedDifference:
    """Control-minus-baseline paired outcomes at the common physical horizon."""

    source_queue: EnvironmentalNetworkSourceQueueMetricDifference
    final_vessel_count_delta: int
    fleet_observation_count_delta: int
    source_plan_count_delta: int
    source_boundary_count_delta: int
    final_source_generation_delta: int


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOPairedExperimentResult:
    """A153 canonical evidence plus the two frozen A152 arm results."""

    config: EnvironmentalNetworkFIFOPairedExperimentConfig
    shared_traffic: PoissonNetworkTrafficRealization
    baseline: EnvironmentalNetworkFIFOExperimentResult
    control: EnvironmentalNetworkFIFOExperimentResult
    difference: EnvironmentalNetworkFIFOPairedDifference
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
            raise EnvironmentalNetworkFIFOPairedExperimentError(
                "frozen paired result document is not a JSON object"
            )
        return document

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes

    @property
    def result_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()


def _assemble_environmental_network_fifo_paired_result(
    config: EnvironmentalNetworkFIFOPairedExperimentConfig,
    traffic: PoissonNetworkTrafficRealization,
    baseline: EnvironmentalNetworkFIFOExperimentResult,
    control: EnvironmentalNetworkFIFOExperimentResult,
) -> EnvironmentalNetworkFIFOPairedExperimentResult:
    """Package-private A159 hook: freeze A153 evidence from already executed CRN arms."""

    if baseline.traffic is not traffic or control.traffic is not traffic:
        raise EnvironmentalNetworkFIFOPairedExperimentError(
            "paired FIFO arms did not retain the exact shared traffic realization"
        )
    if baseline.config != config.baseline or control.config != config.control:
        raise EnvironmentalNetworkFIFOPairedExperimentError(
            "paired FIFO arm results do not belong to the supplied pair config"
        )

    source_queue = compare_source_queue_metrics(
        baseline.final_queue_metrics,
        control.final_queue_metrics,
    )
    difference = EnvironmentalNetworkFIFOPairedDifference(
        source_queue=source_queue,
        final_vessel_count_delta=(
            control.summary.final_vessel_count - baseline.summary.final_vessel_count
        ),
        fleet_observation_count_delta=(
            control.summary.fleet_observation_count - baseline.summary.fleet_observation_count
        ),
        source_plan_count_delta=(
            control.summary.source_plan_count - baseline.summary.source_plan_count
        ),
        source_boundary_count_delta=(
            control.summary.source_boundary_count - baseline.summary.source_boundary_count
        ),
        final_source_generation_delta=(
            control.summary.final_source_generation - baseline.summary.final_source_generation
        ),
    )
    document = {
        "format": _FIFO_PAIR_RESULT_FORMAT,
        "schema_version": _FIFO_PAIR_RESULT_SCHEMA_VERSION,
        "pair_id": config.pair_id,
        "config_sha256": config.config_sha256,
        "shared_basis_sha256": config.shared_basis_sha256,
        "config": config.to_json_object(),
        "execution_environment": environment_manifest(),
        "shared_traffic": _json_value(traffic),
        "baseline": baseline.to_json_object(),
        "control": control.to_json_object(),
        "difference": _json_value(difference),
    }
    canonical_bytes = _canonical_json(document)
    return EnvironmentalNetworkFIFOPairedExperimentResult(
        config=config,
        shared_traffic=traffic,
        baseline=baseline,
        control=control,
        difference=difference,
        canonical_bytes=canonical_bytes,
    )


def run_environmental_network_fifo_paired_experiment(
    config: EnvironmentalNetworkFIFOPairedExperimentConfig,
) -> EnvironmentalNetworkFIFOPairedExperimentResult:
    """Generate CRN traffic once, then execute independent arm-local FIFO responses."""

    if not isinstance(config, EnvironmentalNetworkFIFOPairedExperimentConfig):
        raise TypeError("config must be EnvironmentalNetworkFIFOPairedExperimentConfig")

    base = config.baseline
    scenario = base.scenario
    traffic = generate_poisson_network_traffic(
        base.network,
        base.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    baseline = _run_environmental_network_fifo_experiment_with_traffic(base, traffic)
    control = _run_environmental_network_fifo_experiment_with_traffic(config.control, traffic)
    return _assemble_environmental_network_fifo_paired_result(
        config, traffic, baseline, control
    )
