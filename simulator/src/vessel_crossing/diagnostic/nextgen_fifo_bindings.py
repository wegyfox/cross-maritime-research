"""Runtime bindings for the D0 NextGen environmental-network FIFO adapter.

This module is deliberately *not* a second simulator.  It binds the generic diagnostic
execution surface to concrete, already-existing runtime compositions.  Capability flags
in a scenario never create executable support by themselves: intervention/reference
support is derived only from concrete registered factories/runners.

The actual YAML -> canonical NextGen configuration translator and level executor live
above these immutable bindings.  Keeping the registry separate makes unsupported
scientific/runtime semantics fail closed before any campaign is launched.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from ..nextgen.environmental_network_fifo_experiment import EnvironmentalNetworkFIFOExperimentConfig
from ..nextgen.environmental_network_source_fifo_runtime import (
    EnvironmentalNetworkSourceAdmissionController,
)
from .contract import DiagnosticScenarioSpec
from .evidence import DiagnosticLevelEvidence
from .execution import (
    DiagnosticAdapterRunner,
    DiagnosticExecutionAdapter,
    DiagnosticExecutionContext,
    DiagnosticExecutionIdentity,
)

NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_ADAPTER_ID = "nextgen.environmental_network_fifo.v1"


class NextgenFIFOBindingError(ValueError):
    """Raised when a runtime binding registry is ambiguous or incomplete."""


class NextgenFIFOConfigBuilder(Protocol):
    """Translate one validated D0 scenario into an existing canonical FIFO config."""

    def __call__(
        self, scenario: DiagnosticScenarioSpec
    ) -> EnvironmentalNetworkFIFOExperimentConfig: ...


class NextgenFIFOControllerFactory(Protocol):
    """Build a concrete canonical source-management controller for one level execution.

    The final P5 executor may pass runtime objects through ``runtime_context``.  Keeping
    this context as an opaque product-layer object avoids weakening or modifying the
    canonical NextGen controller protocol merely for D0 orchestration.
    """

    def __call__(
        self,
        context: DiagnosticExecutionContext,
        runtime_context: object,
    ) -> EnvironmentalNetworkSourceAdmissionController: ...


class NextgenFIFOReferenceRunner(Protocol):
    """Execute one concrete reference model and return portable diagnostic evidence."""

    def __call__(self, context: DiagnosticExecutionContext) -> DiagnosticLevelEvidence: ...


@dataclass(frozen=True, slots=True)
class NextgenFIFOInterventionBinding:
    intervention_id: str
    controller_factory: NextgenFIFOControllerFactory

    def __post_init__(self) -> None:
        value = self.intervention_id.strip() if isinstance(self.intervention_id, str) else ""
        if not value:
            raise NextgenFIFOBindingError("intervention_id must be non-empty")
        if not callable(self.controller_factory):
            raise TypeError("controller_factory must be callable")
        object.__setattr__(self, "intervention_id", value)


@dataclass(frozen=True, slots=True)
class NextgenFIFOCompositionBinding:
    """One concrete controller factory for an exact cumulative intervention sequence.

    This does not compose independent controllers.  The factory returns one canonical
    source-admission controller whose internal ordering/state-sharing semantics are owned
    and qualified by the concrete implementation.
    """

    intervention_ids: tuple[str, ...]
    controller_factory: NextgenFIFOControllerFactory

    def __post_init__(self) -> None:
        ids = tuple(
            item.strip() if isinstance(item, str) else "" for item in self.intervention_ids
        )
        if len(ids) < 2 or any(not item for item in ids):
            raise NextgenFIFOBindingError(
                "composition intervention_ids must contain at least two non-empty ids"
            )
        if len(ids) != len(set(ids)):
            raise NextgenFIFOBindingError(
                "composition intervention_ids must not contain duplicates"
            )
        if not callable(self.controller_factory):
            raise TypeError("composition controller_factory must be callable")
        object.__setattr__(self, "intervention_ids", ids)


@dataclass(frozen=True, slots=True)
class NextgenFIFOReferenceBinding:
    reference_id: str
    runner: NextgenFIFOReferenceRunner

    def __post_init__(self) -> None:
        value = self.reference_id.strip() if isinstance(self.reference_id, str) else ""
        if not value:
            raise NextgenFIFOBindingError("reference_id must be non-empty")
        if not callable(self.runner):
            raise TypeError("reference runner must be callable")
        object.__setattr__(self, "reference_id", value)


class NextgenFIFOBindingRegistry:
    """Immutable concrete implementation registry for the D0 FIFO adapter.

    This registry is intentionally distinct from :class:`InterventionRegistry`: the
    semantic registry says *what* a ladder step means, while this registry proves that a
    concrete runtime implementation has actually been supplied for this adapter.
    """

    def __init__(
        self,
        *,
        config_builder: NextgenFIFOConfigBuilder,
        interventions: tuple[NextgenFIFOInterventionBinding, ...] = (),
        compositions: tuple[NextgenFIFOCompositionBinding, ...] = (),
        references: tuple[NextgenFIFOReferenceBinding, ...] = (),
        execution_identity: DiagnosticExecutionIdentity | None = None,
    ) -> None:
        if not callable(config_builder):
            raise TypeError("config_builder must be callable")
        if execution_identity is not None and not isinstance(
            execution_identity, DiagnosticExecutionIdentity
        ):
            raise TypeError("execution_identity must be DiagnosticExecutionIdentity")
        intervention_ids = [item.intervention_id for item in interventions]
        composition_ids = [item.intervention_ids for item in compositions]
        reference_ids = [item.reference_id for item in references]
        if len(intervention_ids) != len(set(intervention_ids)):
            raise NextgenFIFOBindingError("duplicate intervention runtime bindings")
        if len(composition_ids) != len(set(composition_ids)):
            raise NextgenFIFOBindingError("duplicate cumulative intervention runtime bindings")
        if len(reference_ids) != len(set(reference_ids)):
            raise NextgenFIFOBindingError("duplicate reference runtime bindings")
        supported_ids = set(intervention_ids) | {
            intervention_id
            for sequence in composition_ids
            for intervention_id in sequence
        }
        overlap = supported_ids & set(reference_ids)
        if overlap:
            raise NextgenFIFOBindingError(
                f"runtime binding ids cannot be both intervention and reference: {sorted(overlap)}"
            )
        self._config_builder = config_builder
        self._execution_identity = execution_identity
        self._interventions: Mapping[str, NextgenFIFOInterventionBinding] = MappingProxyType(
            {item.intervention_id: item for item in interventions}
        )
        self._compositions: Mapping[
            tuple[str, ...], NextgenFIFOCompositionBinding
        ] = MappingProxyType({item.intervention_ids: item for item in compositions})
        self._references: Mapping[str, NextgenFIFOReferenceBinding] = MappingProxyType(
            {item.reference_id: item for item in references}
        )

    @property
    def config_builder(self) -> NextgenFIFOConfigBuilder:
        return self._config_builder

    @property
    def execution_identity(self) -> DiagnosticExecutionIdentity | None:
        return self._execution_identity

    @property
    def intervention_ids(self) -> frozenset[str]:
        return frozenset(self._interventions)

    @property
    def supported_intervention_ids(self) -> frozenset[str]:
        return frozenset(
            set(self._interventions)
            | {item for sequence in self._compositions for item in sequence}
        )

    @property
    def intervention_sequences(self) -> frozenset[tuple[str, ...]]:
        return frozenset(
            {(intervention_id,) for intervention_id in self._interventions}
            | set(self._compositions)
        )

    @property
    def reference_ids(self) -> frozenset[str]:
        return frozenset(self._references)

    def intervention(self, intervention_id: str) -> NextgenFIFOInterventionBinding:
        try:
            return self._interventions[intervention_id]
        except KeyError as exc:
            raise NextgenFIFOBindingError(
                f"no concrete NextGen FIFO intervention binding {intervention_id!r}"
            ) from exc

    def composition(
        self, intervention_ids: tuple[str, ...]
    ) -> NextgenFIFOCompositionBinding:
        try:
            return self._compositions[tuple(intervention_ids)]
        except KeyError as exc:
            raise NextgenFIFOBindingError(
                f"no concrete NextGen FIFO cumulative intervention binding {tuple(intervention_ids)!r}"
            ) from exc

    def reference(self, reference_id: str) -> NextgenFIFOReferenceBinding:
        try:
            return self._references[reference_id]
        except KeyError as exc:
            raise NextgenFIFOBindingError(
                f"no concrete NextGen FIFO reference binding {reference_id!r}"
            ) from exc


def build_nextgen_fifo_execution_adapter(
    bindings: NextgenFIFOBindingRegistry,
    *,
    runner: DiagnosticAdapterRunner,
) -> DiagnosticExecutionAdapter:
    """Create the versioned D0 adapter with support derived from concrete bindings only."""

    if not isinstance(bindings, NextgenFIFOBindingRegistry):
        raise TypeError("bindings must be NextgenFIFOBindingRegistry")
    if not callable(runner):
        raise TypeError("runner must be callable")
    return DiagnosticExecutionAdapter(
        NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_ADAPTER_ID,
        runner,
        supported_interventions=bindings.supported_intervention_ids,
        supported_references=bindings.reference_ids,
        supported_intervention_sequences=bindings.intervention_sequences,
        execution_identity=bindings.execution_identity,
    )
