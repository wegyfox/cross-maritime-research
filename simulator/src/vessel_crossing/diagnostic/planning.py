"""Deterministic D0 diagnostic planning above the scenario/intervention contracts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from .contract import DiagnosticScenarioSpec
from .interventions import DEFAULT_INTERVENTION_REGISTRY, InterventionPlanEntry, InterventionRegistry

_FORMAT = "cross-diagnostic-plan-v1"
_SCHEMA_VERSION = 1


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class DiagnosticPlan:
    scenario_id: str
    scenario_sha256: str
    entries: tuple[InterventionPlanEntry, ...]
    model_adapters: tuple[tuple[str, str], ...]
    paired: bool
    replications: int
    confidence_level: float
    seed_namespace: str

    def to_json_object(self) -> dict[str, object]:
        return {
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "scenario_sha256": self.scenario_sha256,
            "model_adapters": dict(self.model_adapters),
            "experiment": {
                "paired": self.paired,
                "replications": self.replications,
                "confidence_level": self.confidence_level,
                "seed_namespace": self.seed_namespace,
            },
            "reference_ladder_plan": [item.to_json_object() for item in self.entries],
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def plan_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()


def build_diagnostic_plan(
    scenario: DiagnosticScenarioSpec,
    *,
    registry: InterventionRegistry = DEFAULT_INTERVENTION_REGISTRY,
) -> DiagnosticPlan:
    entries = registry.plan(
        scenario.capabilities,
        requested_interventions=scenario.requested_interventions,
    )
    return DiagnosticPlan(
        scenario_id=scenario.scenario_id,
        scenario_sha256=scenario.scenario_sha256,
        entries=entries,
        model_adapters=tuple(sorted(scenario.model_adapters.items())),
        paired=scenario.experiment.paired,
        replications=scenario.experiment.replications,
        confidence_level=scenario.experiment.confidence_level,
        seed_namespace=scenario.experiment.seed_namespace,
    )
