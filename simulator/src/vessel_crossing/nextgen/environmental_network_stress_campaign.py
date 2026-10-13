"""Fixed multi-seed/environment stress campaigns over the A139 paired protocol.

A140 is deliberately an experiment-orchestration layer, not a new simulator model.  It
reuses one immutable A139 paired template and expands only two exogenous dimensions:

* the stochastic traffic seed;
* a named environmental-drift condition.

Each case is still an exact common-random-number baseline/control comparison.  Expected
model infeasibility (for example, a PAIR/JOINT grid with no physically safe survivor) is
recorded as an explicit case outcome instead of silently dropping that seed from the
campaign.  Unexpected programming/configuration failures remain exceptions and abort the
campaign.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from math import fsum
from types import MappingProxyType
from typing import Mapping

from ..archive import environment_manifest
from ..kinematics import Vector2
from ..version import __version__
from .environmental_candidate_generation import EnvironmentalCandidateGenerationUnavailable
from .environmental_joint_arbiter import NoEnvironmentalJointTuple
from .environmental_joint_physics import EnvironmentalJointPhysicsUnavailable
from .environmental_joint_replanning import EnvironmentalJointReplanningUnavailable
from .environmental_joint_semantics import EnvironmentalJointSemanticUnavailable
from .environmental_network_experiment import _canonical_json, _json_value
from .environmental_network_paired_experiment import (
    EnvironmentalNetworkPairedExperimentConfig,
    EnvironmentalNetworkPairedExperimentResult,
    run_environmental_network_paired_experiment,
)
from .environmental_pair_arbiter import NoEnvironmentalPairCandidate
from .environmental_pair_clearance import EnvironmentalPairTransitionUnavailable
from .environmental_pair_replanning import EnvironmentalPairReplanningUnavailable
from .environmental_route_handoff import EnvironmentalGroundVelocityUnavailable

_CAMPAIGN_FORMAT = "vessel-crossing-nextgen-environmental-network-stress-campaign"
_CAMPAIGN_SCHEMA_VERSION = 1
_CAMPAIGN_DOMAIN = b"vessel-crossing-nextgen-environmental-network-stress-campaign\x00"
_CASE_DOMAIN = b"vessel-crossing-nextgen-environmental-network-stress-case\x00"
_CAMPAIGN_RESULT_FORMAT = "vessel-crossing-nextgen-environmental-network-stress-result"
_CAMPAIGN_RESULT_SCHEMA_VERSION = 1


class EnvironmentalNetworkStressCampaignError(RuntimeError):
    """Raised when an A140 campaign definition is invalid."""


def _finite_drift(value: Vector2) -> Vector2:
    if len(value) != 2:
        raise ValueError("environmental drift must contain exactly two components")
    x, y = float(value[0]), float(value[1])
    if not (abs(x) < float("inf") and abs(y) < float("inf")):
        raise ValueError("environmental drift components must be finite")
    return (x, y)


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkStressEnvironment:
    """One named current/drift condition applied to every seed in a campaign."""

    environment_id: str
    default_environmental_drift_velocity: Vector2 = (0.0, 0.0)
    stream_environmental_drift_velocity: Mapping[str, Vector2] = MappingProxyType({})

    def __post_init__(self) -> None:
        environment_id = self.environment_id.strip()
        if not environment_id:
            raise ValueError("environment_id must be non-empty")
        stream_drifts = {
            str(stream_id): _finite_drift(drift)
            for stream_id, drift in self.stream_environmental_drift_velocity.items()
        }
        if any(not item for item in stream_drifts):
            raise ValueError("stream environmental drift keys must be non-empty")
        object.__setattr__(self, "environment_id", environment_id)
        object.__setattr__(
            self,
            "default_environmental_drift_velocity",
            _finite_drift(self.default_environmental_drift_velocity),
        )
        object.__setattr__(
            self,
            "stream_environmental_drift_velocity",
            MappingProxyType(stream_drifts),
        )


class EnvironmentalNetworkStressCaseStatus(str, Enum):
    SUCCESS = "SUCCESS"
    MODELED_INFEASIBLE = "MODELED_INFEASIBLE"


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkStressCampaignConfig:
    """Immutable Cartesian product of fixed seeds and named environment conditions."""

    paired_template: EnvironmentalNetworkPairedExperimentConfig
    seeds: tuple[int, ...]
    environments: tuple[EnvironmentalNetworkStressEnvironment, ...]
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paired_template, EnvironmentalNetworkPairedExperimentConfig):
            raise TypeError("paired_template must be EnvironmentalNetworkPairedExperimentConfig")
        seeds = tuple(self.seeds)
        if not seeds:
            raise ValueError("stress campaign requires at least one seed")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
            raise TypeError("stress campaign seeds must be integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("stress campaign seeds must be unique")
        environments = tuple(self.environments)
        if not environments:
            raise ValueError("stress campaign requires at least one environment")
        if any(not isinstance(item, EnvironmentalNetworkStressEnvironment) for item in environments):
            raise TypeError("environments must contain EnvironmentalNetworkStressEnvironment values")
        ids = [item.environment_id for item in environments]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate stress environment_id")

        stream_ids = {item.stream_id for item in self.paired_template.control.streams}
        for environment in environments:
            unknown = set(environment.stream_environmental_drift_velocity) - stream_ids
            if unknown:
                raise EnvironmentalNetworkStressCampaignError(
                    "environment drift override refers to unknown stream ids: "
                    + ", ".join(sorted(unknown))
                )

        # Fail before any stochastic draw when the experiment template itself contains a
        # disconnected network request.
        for stream in self.paired_template.control.streams:
            self.paired_template.control.network.resolve(stream.request)

        if self.label is not None:
            label = self.label.strip()
            if not label:
                raise ValueError("label must be non-empty when supplied")
            object.__setattr__(self, "label", label)
        object.__setattr__(self, "seeds", tuple(sorted(seeds)))
        object.__setattr__(
            self,
            "environments",
            tuple(sorted(environments, key=lambda item: item.environment_id)),
        )

    def _template_identity_object(self) -> dict[str, object]:
        # The template seed and template drift values are placeholders: every A140 case
        # replaces them explicitly.  They must therefore not perturb campaign identity.
        control = self.paired_template.control.to_json_object()
        scenario = dict(control["scenario"])
        scenario.pop("seed", None)
        scenario.pop("default_environmental_drift_velocity", None)
        scenario.pop("stream_environmental_drift_velocity", None)
        control["scenario"] = scenario
        return {
            "control_template": control,
            "safety": _json_value(self.paired_template.safety),
            "baseline_policy": "nominal_route_no_encounter_avoidance",
        }

    def to_json_object(self) -> dict[str, object]:
        return {
            "format": _CAMPAIGN_FORMAT,
            "schema_version": _CAMPAIGN_SCHEMA_VERSION,
            "label": self.label,
            "software_version": __version__,
            "paired_template": self._template_identity_object(),
            "seeds": list(self.seeds),
            "environments": _json_value(self.environments),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def config_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()

    @property
    def campaign_id(self) -> str:
        return sha256(_CAMPAIGN_DOMAIN + self.to_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkStressCaseResult:
    case_id: str
    seed: int
    environment_id: str
    paired_config: EnvironmentalNetworkPairedExperimentConfig
    status: EnvironmentalNetworkStressCaseStatus
    result: EnvironmentalNetworkPairedExperimentResult | None = None
    infeasibility_type: str | None = None
    infeasibility_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is EnvironmentalNetworkStressCaseStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkStressAggregate:
    total_case_count: int
    successful_case_count: int
    modeled_infeasible_case_count: int
    baseline_modeled_hull_overlap_episode_count: int
    control_modeled_hull_overlap_episode_count: int
    modeled_hull_overlap_episode_delta: int
    baseline_safety_domain_intrusion_episode_count: int
    control_safety_domain_intrusion_episode_count: int
    safety_domain_intrusion_episode_delta: int
    baseline_near_miss_episode_count: int
    control_near_miss_episode_count: int
    near_miss_episode_delta: int
    baseline_unresolved_episode_count: int
    control_unresolved_episode_count: int
    unresolved_episode_delta: int
    baseline_nonclear_pair_exposure_min: float
    control_nonclear_pair_exposure_min: float
    nonclear_pair_exposure_delta_min: float
    baseline_route_completed_count: int
    control_route_completed_count: int
    route_completed_count_delta: int
    mean_route_completion_rate_delta_per_hour: float | None
    mean_route_duration_delta_min: float | None


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkStressCampaignResult:
    config: EnvironmentalNetworkStressCampaignConfig
    campaign_id: str
    cases: tuple[EnvironmentalNetworkStressCaseResult, ...]
    aggregate: EnvironmentalNetworkStressAggregate

    def to_json_object(self) -> dict[str, object]:
        case_documents: list[dict[str, object]] = []
        for case in self.cases:
            case_documents.append(
                {
                    "case_id": case.case_id,
                    "seed": case.seed,
                    "environment_id": case.environment_id,
                    "paired_config": case.paired_config.to_json_object(),
                    "status": case.status.value,
                    "result": None if case.result is None else case.result.to_json_object(),
                    "infeasibility_type": case.infeasibility_type,
                    "infeasibility_message": case.infeasibility_message,
                }
            )
        return {
            "format": _CAMPAIGN_RESULT_FORMAT,
            "schema_version": _CAMPAIGN_RESULT_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "config_sha256": self.config.config_sha256,
            "config": self.config.to_json_object(),
            "execution_environment": environment_manifest(),
            "cases": case_documents,
            "aggregate": _json_value(self.aggregate),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def result_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()


_MODELED_INFEASIBILITY_TYPES = (
    NoEnvironmentalPairCandidate,
    NoEnvironmentalJointTuple,
    EnvironmentalCandidateGenerationUnavailable,
    EnvironmentalPairTransitionUnavailable,
    EnvironmentalJointPhysicsUnavailable,
    EnvironmentalJointSemanticUnavailable,
    EnvironmentalPairReplanningUnavailable,
    EnvironmentalJointReplanningUnavailable,
    EnvironmentalGroundVelocityUnavailable,
)


def _modeled_infeasibility(exc: BaseException) -> BaseException | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _MODELED_INFEASIBILITY_TYPES):
            return current
        current = current.__cause__ or current.__context__
    return None


def _case_config(
    config: EnvironmentalNetworkStressCampaignConfig,
    *,
    seed: int,
    environment: EnvironmentalNetworkStressEnvironment,
) -> EnvironmentalNetworkPairedExperimentConfig:
    control = config.paired_template.control
    scenario = replace(
        control.scenario,
        seed=seed,
        default_environmental_drift_velocity=environment.default_environmental_drift_velocity,
        stream_environmental_drift_velocity=dict(environment.stream_environmental_drift_velocity),
    )
    return EnvironmentalNetworkPairedExperimentConfig(
        replace(control, scenario=scenario),
        config.paired_template.safety,
    )


def _case_id(campaign_id: str, paired: EnvironmentalNetworkPairedExperimentConfig, environment_id: str) -> str:
    payload = _canonical_json(
        {
            "campaign_id": campaign_id,
            "environment_id": environment_id,
            "pair_id": paired.pair_id,
        }
    )
    return sha256(_CASE_DOMAIN + payload).hexdigest()


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fsum(present) / len(present) if present else None


def _aggregate(cases: tuple[EnvironmentalNetworkStressCaseResult, ...]) -> EnvironmentalNetworkStressAggregate:
    results = [item.result for item in cases if item.result is not None]
    baseline_hull = sum(item.baseline.metrics.safety.modeled_hull_overlap_episode_count for item in results)
    control_hull = sum(item.control_metrics.safety.modeled_hull_overlap_episode_count for item in results)
    baseline_domain = sum(item.baseline.metrics.safety.safety_domain_intrusion_episode_count for item in results)
    control_domain = sum(item.control_metrics.safety.safety_domain_intrusion_episode_count for item in results)
    baseline_near = sum(item.baseline.metrics.safety.near_miss_episode_count for item in results)
    control_near = sum(item.control_metrics.safety.near_miss_episode_count for item in results)
    baseline_unresolved = sum(item.baseline.metrics.safety.unresolved_episode_count for item in results)
    control_unresolved = sum(item.control_metrics.safety.unresolved_episode_count for item in results)
    baseline_exposure = fsum(item.baseline.metrics.safety.nonclear_pair_exposure_min for item in results)
    control_exposure = fsum(item.control_metrics.safety.nonclear_pair_exposure_min for item in results)
    baseline_completed = sum(item.baseline.metrics.route_completed_count for item in results)
    control_completed = sum(item.control_metrics.route_completed_count for item in results)
    return EnvironmentalNetworkStressAggregate(
        total_case_count=len(cases),
        successful_case_count=len(results),
        modeled_infeasible_case_count=len(cases) - len(results),
        baseline_modeled_hull_overlap_episode_count=baseline_hull,
        control_modeled_hull_overlap_episode_count=control_hull,
        modeled_hull_overlap_episode_delta=control_hull - baseline_hull,
        baseline_safety_domain_intrusion_episode_count=baseline_domain,
        control_safety_domain_intrusion_episode_count=control_domain,
        safety_domain_intrusion_episode_delta=control_domain - baseline_domain,
        baseline_near_miss_episode_count=baseline_near,
        control_near_miss_episode_count=control_near,
        near_miss_episode_delta=control_near - baseline_near,
        baseline_unresolved_episode_count=baseline_unresolved,
        control_unresolved_episode_count=control_unresolved,
        unresolved_episode_delta=control_unresolved - baseline_unresolved,
        baseline_nonclear_pair_exposure_min=baseline_exposure,
        control_nonclear_pair_exposure_min=control_exposure,
        nonclear_pair_exposure_delta_min=control_exposure - baseline_exposure,
        baseline_route_completed_count=baseline_completed,
        control_route_completed_count=control_completed,
        route_completed_count_delta=control_completed - baseline_completed,
        mean_route_completion_rate_delta_per_hour=_mean_optional(
            [item.difference.route_completion_rate_per_hour for item in results]
        ),
        mean_route_duration_delta_min=_mean_optional(
            [item.difference.mean_route_duration_min for item in results]
        ),
    )


def run_environmental_network_stress_campaign(
    config: EnvironmentalNetworkStressCampaignConfig,
) -> EnvironmentalNetworkStressCampaignResult:
    """Run the fixed seed × environment matrix, preserving modeled infeasibility."""

    if not isinstance(config, EnvironmentalNetworkStressCampaignConfig):
        raise TypeError("config must be EnvironmentalNetworkStressCampaignConfig")
    cases: list[EnvironmentalNetworkStressCaseResult] = []
    for environment in config.environments:
        for seed in config.seeds:
            paired = _case_config(config, seed=seed, environment=environment)
            case_id = _case_id(config.campaign_id, paired, environment.environment_id)
            try:
                result = run_environmental_network_paired_experiment(paired)
            except Exception as exc:
                modeled = _modeled_infeasibility(exc)
                if modeled is None:
                    raise
                cases.append(
                    EnvironmentalNetworkStressCaseResult(
                        case_id=case_id,
                        seed=seed,
                        environment_id=environment.environment_id,
                        paired_config=paired,
                        status=EnvironmentalNetworkStressCaseStatus.MODELED_INFEASIBLE,
                        infeasibility_type=type(modeled).__name__,
                        infeasibility_message=str(modeled),
                    )
                )
            else:
                cases.append(
                    EnvironmentalNetworkStressCaseResult(
                        case_id=case_id,
                        seed=seed,
                        environment_id=environment.environment_id,
                        paired_config=paired,
                        status=EnvironmentalNetworkStressCaseStatus.SUCCESS,
                        result=result,
                    )
                )
    frozen = tuple(cases)
    return EnvironmentalNetworkStressCampaignResult(
        config=config,
        campaign_id=config.campaign_id,
        cases=frozen,
        aggregate=_aggregate(frozen),
    )
