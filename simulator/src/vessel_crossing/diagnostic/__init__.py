"""Reusable scenario-to-diagnosis product contracts above the frozen CROSS runtime."""

from .contract import (
    DiagnosticCapabilities,
    DiagnosticExperimentSpec,
    DiagnosticScenarioContractError,
    DiagnosticScenarioSpec,
    load_diagnostic_scenario,
)
from .interventions import (
    DEFAULT_INTERVENTION_REGISTRY,
    InterventionDefinition,
    InterventionDisposition,
    InterventionPlanEntry,
    InterventionRegistry,
)
from .planning import DiagnosticPlan, build_diagnostic_plan

__all__ = [
    "DiagnosticCapabilities",
    "DiagnosticExperimentSpec",
    "DiagnosticScenarioContractError",
    "DiagnosticScenarioSpec",
    "load_diagnostic_scenario",
    "DEFAULT_INTERVENTION_REGISTRY",
    "InterventionDefinition",
    "InterventionDisposition",
    "InterventionPlanEntry",
    "InterventionRegistry",
    "DiagnosticPlan",
    "build_diagnostic_plan",
]

from .archive_adapter import (
    LegacyNetworkFifoArchiveSpec,
    LegacyNetworkFifoEvidenceBundle,
    build_legacy_network_fifo_evidence,
)
from .classifier import DiagnosticBottleneckClassification, classify_diagnostic_bottleneck
from .effects import DiagnosticEffectEstimate, estimate_ladder_effects
from .evidence import (
    DiagnosticEvidenceError,
    DiagnosticLevelEvidence,
    DiagnosticReplicationEvidence,
    DiagnosticStudyEvidence,
)
from .synthesis import (
    DiagnosticSynthesisProfile,
    r2_frozen_acceptance_synthesis_profile,
    synthesize_generic_ideal_gap_decomposition,
    synthesize_ideal_gap_decomposition,
)

__all__ += [
    "LegacyNetworkFifoArchiveSpec",
    "LegacyNetworkFifoEvidenceBundle",
    "build_legacy_network_fifo_evidence",
    "DiagnosticBottleneckClassification",
    "classify_diagnostic_bottleneck",
    "DiagnosticEffectEstimate",
    "estimate_ladder_effects",
    "DiagnosticEvidenceError",
    "DiagnosticLevelEvidence",
    "DiagnosticReplicationEvidence",
    "DiagnosticStudyEvidence",
    "DiagnosticSynthesisProfile",
    "r2_frozen_acceptance_synthesis_profile",
    "synthesize_generic_ideal_gap_decomposition",
    "synthesize_ideal_gap_decomposition",
]

from .product import DiagnosticProductRun, run_diagnostic_product

__all__ += [
    "DiagnosticProductRun",
    "run_diagnostic_product",
]

from .execution import (
    DEFAULT_DIAGNOSTIC_ADAPTER_REGISTRY,
    DiagnosticAdapterRegistry,
    DiagnosticExecutionAdapter,
    DiagnosticExecutionContext,
    DiagnosticExecutionError,
    DiagnosticExecutionIdentity,
    DiagnosticExecutionPreflight,
    DiagnosticExecutionResult,
    run_diagnostic_orchestrator,
    validate_diagnostic_execution,
)

__all__ += [
    "DEFAULT_DIAGNOSTIC_ADAPTER_REGISTRY",
    "DiagnosticAdapterRegistry",
    "DiagnosticExecutionAdapter",
    "DiagnosticExecutionContext",
    "DiagnosticExecutionError",
    "DiagnosticExecutionIdentity",
    "DiagnosticExecutionPreflight",
    "DiagnosticExecutionResult",
    "run_diagnostic_orchestrator",
    "validate_diagnostic_execution",
]

from .run_manifest import (
    DiagnosticRunManifestArtifacts,
    DiagnosticRunManifestError,
    DiagnosticRunManifestVerification,
    build_diagnostic_product_run_manifest,
    verify_diagnostic_product_run_manifest,
    write_diagnostic_product_run_manifest,
)

__all__ += [
    "DiagnosticRunManifestArtifacts",
    "DiagnosticRunManifestError",
    "DiagnosticRunManifestVerification",
    "build_diagnostic_product_run_manifest",
    "verify_diagnostic_product_run_manifest",
    "write_diagnostic_product_run_manifest",
]

from .resume import (
    DiagnosticResumeError,
    DiagnosticResumeEvidenceArtifacts,
    DiagnosticVerifiedPrefix,
    prepare_diagnostic_resume_prefix,
    run_diagnostic_orchestrator_with_resume,
    run_diagnostic_product_with_resume,
    write_diagnostic_resume_evidence,
)

__all__ += [
    "DiagnosticResumeError",
    "DiagnosticResumeEvidenceArtifacts",
    "DiagnosticVerifiedPrefix",
    "prepare_diagnostic_resume_prefix",
    "run_diagnostic_orchestrator_with_resume",
    "run_diagnostic_product_with_resume",
    "write_diagnostic_resume_evidence",
]

from .views import (
    DiagnosticDecisionViewArtifacts,
    build_diagnostic_decision_view,
    render_diagnostic_decision_view_html,
    write_diagnostic_decision_view_artifacts,
)

__all__ += [
    "DiagnosticDecisionViewArtifacts",
    "build_diagnostic_decision_view",
    "render_diagnostic_decision_view_html",
    "write_diagnostic_decision_view_artifacts",
]

from .bundle import (
    DiagnosticReproductionBundleArtifacts,
    DiagnosticReproductionBundleError,
    DiagnosticReproductionBundleVerification,
    build_diagnostic_reproduction_bundle_payload,
    verify_diagnostic_reproduction_bundle,
    write_diagnostic_reproduction_bundle,
)

__all__ += [
    "DiagnosticReproductionBundleArtifacts",
    "DiagnosticReproductionBundleError",
    "DiagnosticReproductionBundleVerification",
    "build_diagnostic_reproduction_bundle_payload",
    "verify_diagnostic_reproduction_bundle",
    "write_diagnostic_reproduction_bundle",
]
