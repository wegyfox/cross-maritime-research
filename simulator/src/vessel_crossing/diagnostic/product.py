"""One-command diagnostic product workflow above explicit runtime bindings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..reporting.reference_envelope import (
    IdealGapDecomposition,
    ReferenceEnvelopeArtifacts,
    write_reference_envelope_artifacts,
)
from .contract import DiagnosticScenarioSpec
from .execution import (
    DiagnosticAdapterRegistry,
    DiagnosticExecutionResult,
    run_diagnostic_orchestrator,
)
from .synthesis import synthesize_generic_ideal_gap_decomposition
from .views import DiagnosticDecisionViewArtifacts, write_diagnostic_decision_view_artifacts


@dataclass(frozen=True, slots=True)
class DiagnosticProductRun:
    """Qualified product result plus canonical report artifacts."""

    execution: DiagnosticExecutionResult
    decomposition: IdealGapDecomposition
    artifacts: ReferenceEnvelopeArtifacts
    decision_view: DiagnosticDecisionViewArtifacts


def run_diagnostic_product(
    scenario: DiagnosticScenarioSpec,
    *,
    registry: DiagnosticAdapterRegistry,
    output_directory: str | Path,
    date: str,
    scope: str,
    title: str | None = None,
) -> DiagnosticProductRun:
    """Execute, diagnose, synthesize and render one explicit D0 scenario.

    ``registry`` is mandatory by design.  Scientific/runtime implementations are never
    inferred from scenario capability flags or silently populated by this product layer.
    """

    if not isinstance(scenario, DiagnosticScenarioSpec):
        raise TypeError("scenario must be DiagnosticScenarioSpec")
    if not isinstance(registry, DiagnosticAdapterRegistry):
        raise TypeError("registry must be DiagnosticAdapterRegistry")

    execution = run_diagnostic_orchestrator(scenario, registry=registry)
    payload = synthesize_generic_ideal_gap_decomposition(
        execution.study,
        execution.effects,
        execution.classification,
        date=date,
        scope=scope,
    )
    decomposition = IdealGapDecomposition.from_json_object(payload)
    artifacts = write_reference_envelope_artifacts(
        decomposition,
        output_directory,
        title=title,
    )
    decision_view = write_diagnostic_decision_view_artifacts(
        decomposition,
        output_directory,
        title=(f"{title} — Decision View" if title else None),
    )
    return DiagnosticProductRun(execution, decomposition, artifacts, decision_view)
