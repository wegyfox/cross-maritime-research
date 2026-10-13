"""Executable D0 orchestration boundary to already-existing canonical runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..nextgen.reference_envelope import ReferenceEnvelopeLevel
from .classifier import DiagnosticBottleneckClassification, classify_diagnostic_bottleneck
from .contract import DiagnosticScenarioSpec
from .effects import DiagnosticEffectEstimate, estimate_ladder_effects
from .evidence import DiagnosticLevelEvidence, DiagnosticStudyEvidence
from .interventions import InterventionDisposition, InterventionPlanEntry
from .planning import DiagnosticPlan, build_diagnostic_plan


class DiagnosticExecutionError(ValueError):
    """Raised when a D0 execution cannot safely use the declared canonical adapter."""


@dataclass(frozen=True, slots=True)
class DiagnosticExecutionContext:
    scenario: DiagnosticScenarioSpec
    plan: DiagnosticPlan
    level: ReferenceEnvelopeLevel
    plan_entry: InterventionPlanEntry | None
    previous_level: DiagnosticLevelEvidence | None
    previous_runs: tuple[DiagnosticLevelEvidence, ...]
    applied_intervention_ids: tuple[str, ...]


DiagnosticAdapterRunner = Callable[[DiagnosticExecutionContext], DiagnosticLevelEvidence]


@dataclass(frozen=True, slots=True)
class DiagnosticExecutionIdentity:
    """Explicit integrator-owned identity for one concrete scientific binding set."""

    identity_id: str
    components: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = self.identity_id.strip() if isinstance(self.identity_id, str) else ""
        if not value:
            raise DiagnosticExecutionError("execution identity_id must be non-empty")
        try:
            components = json.loads(
                json.dumps(
                    dict(self.components),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise DiagnosticExecutionError(
                "execution identity components must be JSON portable"
            ) from exc
        if not isinstance(components, dict) or not components:
            raise DiagnosticExecutionError(
                "execution identity components must be a non-empty object"
            )
        object.__setattr__(self, "identity_id", value)
        object.__setattr__(self, "components", MappingProxyType(components))

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": "cross-diagnostic-execution-identity-v1",
            "schema_version": 1,
            "identity_id": self.identity_id,
            "components": dict(self.components),
        }

    def to_json_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_json_object(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    @property
    def identity_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class DiagnosticExecutionAdapter:
    adapter_id: str
    runner: DiagnosticAdapterRunner
    supported_interventions: frozenset[str] = frozenset()
    supported_references: frozenset[str] = frozenset()
    supported_intervention_sequences: frozenset[tuple[str, ...]] = frozenset()
    execution_identity: DiagnosticExecutionIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_id, str) or not self.adapter_id.strip():
            raise DiagnosticExecutionError("adapter_id must be non-empty")
        object.__setattr__(self, "adapter_id", self.adapter_id.strip())
        if not callable(self.runner):
            raise TypeError("runner must be callable")
        if self.execution_identity is not None and not isinstance(
            self.execution_identity, DiagnosticExecutionIdentity
        ):
            raise TypeError("execution_identity must be DiagnosticExecutionIdentity")
        sequences: set[tuple[str, ...]] = set()
        for raw in self.supported_intervention_sequences:
            sequence = tuple(raw)
            if not sequence or any(not isinstance(item, str) or not item.strip() for item in sequence):
                raise DiagnosticExecutionError(
                    "supported intervention sequences must contain non-empty intervention ids"
                )
            sequence = tuple(item.strip() for item in sequence)
            if len(sequence) != len(set(sequence)):
                raise DiagnosticExecutionError(
                    "supported intervention sequences must not repeat an intervention id"
                )
            sequences.add(sequence)
        object.__setattr__(self, "supported_intervention_sequences", frozenset(sequences))

    def run(self, context: DiagnosticExecutionContext) -> DiagnosticLevelEvidence:
        result = self.runner(context)
        if not isinstance(result, DiagnosticLevelEvidence):
            raise DiagnosticExecutionError("runtime adapter must return DiagnosticLevelEvidence")
        if result.level is not context.level:
            raise DiagnosticExecutionError(
                f"adapter returned {result.level.value} while {context.level.value} was requested"
            )
        return result


class DiagnosticAdapterRegistry:
    """Immutable versioned lookup for product-to-runtime execution adapters."""

    def __init__(self, adapters: tuple[DiagnosticExecutionAdapter, ...] = ()) -> None:
        ids = [item.adapter_id for item in adapters]
        if len(ids) != len(set(ids)):
            raise DiagnosticExecutionError("diagnostic adapter ids must be unique")
        self._adapters: Mapping[str, DiagnosticExecutionAdapter] = MappingProxyType(
            {item.adapter_id: item for item in adapters}
        )

    def resolve(self, adapter_id: str) -> DiagnosticExecutionAdapter:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            raise DiagnosticExecutionError(
                f"no registered diagnostic runtime adapter {adapter_id!r}"
            ) from exc

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


DEFAULT_DIAGNOSTIC_ADAPTER_REGISTRY = DiagnosticAdapterRegistry()


@dataclass(frozen=True, slots=True)
class DiagnosticExecutionPreflight:
    scenario: DiagnosticScenarioSpec
    plan: DiagnosticPlan
    adapter: DiagnosticExecutionAdapter


@dataclass(frozen=True, slots=True)
class DiagnosticExecutionResult:
    study: DiagnosticStudyEvidence
    effects: tuple[DiagnosticEffectEstimate, ...]
    classification: DiagnosticBottleneckClassification


def validate_diagnostic_execution(
    scenario: DiagnosticScenarioSpec,
    *,
    registry: DiagnosticAdapterRegistry = DEFAULT_DIAGNOSTIC_ADAPTER_REGISTRY,
) -> DiagnosticExecutionPreflight:
    plan = build_diagnostic_plan(scenario)
    runtime_id = scenario.model_adapters.get("runtime")
    if runtime_id is None:
        raise DiagnosticExecutionError("model.adapters.runtime is required for execution")
    adapter = registry.resolve(runtime_id)
    cumulative_interventions: list[str] = []
    for entry in plan.entries:
        if entry.disposition in {
            InterventionDisposition.NOT_REQUESTED,
            InterventionDisposition.UNSUPPORTED,
        }:
            raise DiagnosticExecutionError(
                f"incomplete B0-B6 execution ladder at {entry.level.value}: {entry.disposition.value}"
            )
        if entry.disposition is InterventionDisposition.APPLICABLE:
            cumulative_interventions.append(entry.intervention_id)
            sequence = tuple(cumulative_interventions)
            if adapter.supported_intervention_sequences:
                if sequence not in adapter.supported_intervention_sequences:
                    raise DiagnosticExecutionError(
                        f"runtime {runtime_id!r} does not support cumulative intervention sequence {sequence!r}"
                    )
            elif entry.intervention_id not in adapter.supported_interventions:
                raise DiagnosticExecutionError(
                    f"runtime {runtime_id!r} does not support declared intervention {entry.intervention_id!r}"
                )
        if (
            entry.disposition is InterventionDisposition.REFERENCE
            and entry.intervention_id not in adapter.supported_references
        ):
            raise DiagnosticExecutionError(
                f"runtime {runtime_id!r} does not support declared reference {entry.intervention_id!r}"
            )
    return DiagnosticExecutionPreflight(scenario, plan, adapter)


def _identity_level(
    previous: DiagnosticLevelEvidence,
    entry: InterventionPlanEntry,
) -> DiagnosticLevelEvidence:
    return DiagnosticLevelEvidence(
        entry.level,
        {"service_yield_per_hour": previous.primary_kpi},
        previous.replications,
        intervention_id=entry.intervention_id,
        target_factor=entry.target_factor,
        causal=False,
        identity_of=previous.level,
        reason=entry.reason,
    )


def run_diagnostic_orchestrator(
    scenario: DiagnosticScenarioSpec,
    *,
    registry: DiagnosticAdapterRegistry = DEFAULT_DIAGNOSTIC_ADAPTER_REGISTRY,
) -> DiagnosticExecutionResult:
    preflight = validate_diagnostic_execution(scenario, registry=registry)
    adapter = preflight.adapter
    plan = preflight.plan
    runs: list[DiagnosticLevelEvidence] = []
    baseline_context = DiagnosticExecutionContext(
        scenario,
        plan,
        ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK,
        None,
        None,
        (),
        (),
    )
    baseline = adapter.run(baseline_context)
    runs.append(baseline)
    applied: list[str] = []

    for entry in plan.entries:
        previous = runs[-1]
        if entry.disposition is InterventionDisposition.IDENTITY:
            runs.append(_identity_level(previous, entry))
            continue
        applied.append(entry.intervention_id)
        context = DiagnosticExecutionContext(
            scenario,
            plan,
            entry.level,
            entry,
            previous,
            tuple(runs),
            tuple(applied),
        )
        result = adapter.run(context)
        if result.intervention_id != entry.intervention_id:
            raise DiagnosticExecutionError(
                f"adapter intervention identity mismatch at {entry.level.value}"
            )
        if result.target_factor != entry.target_factor:
            raise DiagnosticExecutionError(
                f"adapter target-factor identity mismatch at {entry.level.value}"
            )
        expected_causal = entry.disposition is InterventionDisposition.APPLICABLE
        if result.causal is not expected_causal:
            raise DiagnosticExecutionError(
                f"adapter causal/reference mismatch at {entry.level.value}"
            )
        runs.append(result)

    provenance: dict[str, Any] = {
        "scenario_sha256": scenario.scenario_sha256,
        "plan_sha256": plan.plan_sha256,
        "runtime_adapter": adapter.adapter_id,
    }
    if adapter.execution_identity is not None:
        provenance.update(
            {
                "execution_identity_id": adapter.execution_identity.identity_id,
                "execution_identity_sha256": adapter.execution_identity.identity_sha256,
            }
        )
    study = DiagnosticStudyEvidence(
        scenario.scenario_id,
        tuple(runs),
        claim_scope=scenario.claim_scope,
        provenance=provenance,
    )
    effects = estimate_ladder_effects(
        study, confidence_level=scenario.experiment.confidence_level
    )
    classification = classify_diagnostic_bottleneck(effects)
    return DiagnosticExecutionResult(study, effects, classification)
