"""Frozen C0-C19 research-testbed architecture manifest.

This module does not enable any optional feature.  It records which generic modules
compose the closed base architecture and the invariants that remain binding when later
research protocols branch from the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from typing import Iterable

from .scope import (
    A162_BASE_ARCHIVE,
    A162_BASE_SHA256,
    A162_BASE_VERSION,
    DEFAULT_FEATURE_FLAGS,
    SCOPE_NAME,
    TESTBED_SCOPE_VERSION,
    TestbedFeature,
)


@dataclass(frozen=True, slots=True)
class BaselineComponent:
    feature: TestbedFeature
    module: str
    default_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.module.startswith("vessel_crossing.testbed."):
            raise ValueError("baseline module must live under vessel_crossing.testbed")
        if self.default_enabled:
            raise ValueError("closed baseline components must remain disabled by default")


BASELINE_COMPONENTS: tuple[BaselineComponent, ...] = (
    BaselineComponent(TestbedFeature.C1_SCENARIO_INTERCHANGE, "vessel_crossing.testbed.scenario"),
    BaselineComponent(TestbedFeature.C2_WORLD_OBJECTS_CONSTRAINTS, "vessel_crossing.testbed.world_objects"),
    BaselineComponent(TestbedFeature.C3_BEHAVIOR_MODELS, "vessel_crossing.testbed.behavior"),
    BaselineComponent(TestbedFeature.C4_OBSERVED_WORLD_BOUNDARY, "vessel_crossing.testbed.observation"),
    BaselineComponent(TestbedFeature.C5_TRACK_LIFECYCLE, "vessel_crossing.testbed.track_lifecycle"),
    BaselineComponent(TestbedFeature.C6_ENVIRONMENT_BELIEF, "vessel_crossing.testbed.environment_belief"),
    BaselineComponent(TestbedFeature.C7_FAULT_INJECTION, "vessel_crossing.testbed.fault_injection"),
    BaselineComponent(TestbedFeature.C8_ODD_PREDICATES, "vessel_crossing.testbed.odd"),
    BaselineComponent(TestbedFeature.C9_FALLBACK_STATE_MACHINE, "vessel_crossing.testbed.operational_state"),
    BaselineComponent(TestbedFeature.C10_RECOVERABILITY, "vessel_crossing.testbed.recoverability"),
    BaselineComponent(TestbedFeature.C11_FACTOR_COMPOSITION, "vessel_crossing.testbed.factor_composition"),
    BaselineComponent(TestbedFeature.C12_EVALUATION_ORACLES, "vessel_crossing.testbed.evaluation_oracles"),
    BaselineComponent(TestbedFeature.C13_CORPUS_LINEAGE, "vessel_crossing.testbed.corpus_lineage"),
    BaselineComponent(TestbedFeature.C14_MACRO_REFERENCE_MODEL, "vessel_crossing.testbed.macro_reference"),
    BaselineComponent(TestbedFeature.C15_EXTERNAL_CONTROLLER, "vessel_crossing.testbed.external_controller"),
    BaselineComponent(TestbedFeature.C16_PLANT_FIDELITY, "vessel_crossing.testbed.plant_fidelity"),
    BaselineComponent(TestbedFeature.C17_REAL_DATA_CALIBRATION, "vessel_crossing.testbed.real_data_calibration"),
    BaselineComponent(TestbedFeature.C18_COORDINATION_CHANNEL, "vessel_crossing.testbed.coordination"),
    BaselineComponent(TestbedFeature.C19_HUMAN_ROC, "vessel_crossing.testbed.human_roc"),
)


BASELINE_INVARIANTS: tuple[str, ...] = (
    "no-article-specific-physics-fork",
    "optional-layers-disabled-or-noop-preserve-prior-semantics",
    "truth-perception-decision-plant-evaluation-separated",
    "external-tools-cross-versioned-adapters-or-cosimulation-boundaries",
    "seed-protocol-result-and-negative-evidence-identities-are-durable",
    "no-gpl-copy-into-permissive-core-without-explicit-license-decision",
    "research-claims-belong-to-protocols-not-architecture-names",
    "modeling-a-related-concept-does-not-imply-colreg-iso-or-certification-compliance",
)


@dataclass(frozen=True, slots=True)
class ResearchTestbedBaselineManifest:
    scope_name: str
    scope_version: str
    base_version: str
    base_archive: str
    base_sha256: str
    components: tuple[BaselineComponent, ...]
    invariants: tuple[str, ...]
    architecture_closed: bool = True

    def identity_sha256(self) -> str:
        payload = {
            "scope_name": self.scope_name,
            "scope_version": self.scope_version,
            "base_version": self.base_version,
            "base_archive": self.base_archive,
            "base_sha256": self.base_sha256,
            "architecture_closed": self.architecture_closed,
            "components": [
                {
                    "feature": item.feature.value,
                    "module": item.module,
                    "default_enabled": item.default_enabled,
                }
                for item in self.components
            ],
            "invariants": list(self.invariants),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    def validate(self, *, import_modules: bool = False) -> None:
        features = tuple(item.feature for item in self.components)
        if len(features) != 19 or set(features) != set(TestbedFeature):
            raise ValueError("baseline must cover C1..C19 exactly once")
        if len(features) != len(set(features)):
            raise ValueError("baseline features must be unique")
        if not self.architecture_closed:
            raise ValueError("baseline closure manifest must be marked closed")
        if self.base_sha256 != A162_BASE_SHA256:
            raise ValueError("baseline A162 hash differs from immutable C0 boundary")
        if any(DEFAULT_FEATURE_FLAGS[item.feature] for item in self.components):
            raise ValueError("baseline feature registry must remain disabled by default")
        if import_modules:
            for item in self.components:
                importlib.import_module(item.module)


def current_baseline_manifest() -> ResearchTestbedBaselineManifest:
    manifest = ResearchTestbedBaselineManifest(
        scope_name=SCOPE_NAME,
        scope_version=TESTBED_SCOPE_VERSION,
        base_version=A162_BASE_VERSION,
        base_archive=A162_BASE_ARCHIVE,
        base_sha256=A162_BASE_SHA256,
        components=BASELINE_COMPONENTS,
        invariants=BASELINE_INVARIANTS,
    )
    manifest.validate()
    return manifest


BASELINE_MANIFEST = current_baseline_manifest()
