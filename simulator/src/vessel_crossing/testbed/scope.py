"""Declared scope identity and feature registry for the post-A162 research testbed.

This module is intentionally infrastructure-only. Importing it must not alter any
A162 physics, controller, stochastic, archive, or statistical semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, TypeVar

from vessel_crossing.version import __version__ as _PACKAGE_VERSION


TESTBED_SCOPE_VERSION = _PACKAGE_VERSION
A162_BASE_VERSION = "1.0.0a162"
A162_BASE_ARCHIVE = "vessel-crossing-simulator_v1.0.0a162.zip"
A162_BASE_SHA256 = "21c749ad4116d0b1533182304861808cec28b8085958008a71a556efed11ec3f"
SCOPE_NAME = "cross-maritime-autonomy-vv-testbed"


class TestbedFeature(str, Enum):
    """Stable component identifiers from the engineering handoff C1..C19."""

    __test__ = False

    C1_SCENARIO_INTERCHANGE = "C1_SCENARIO_INTERCHANGE"
    C2_WORLD_OBJECTS_CONSTRAINTS = "C2_WORLD_OBJECTS_CONSTRAINTS"
    C3_BEHAVIOR_MODELS = "C3_BEHAVIOR_MODELS"
    C4_OBSERVED_WORLD_BOUNDARY = "C4_OBSERVED_WORLD_BOUNDARY"
    C5_TRACK_LIFECYCLE = "C5_TRACK_LIFECYCLE"
    C6_ENVIRONMENT_BELIEF = "C6_ENVIRONMENT_BELIEF"
    C7_FAULT_INJECTION = "C7_FAULT_INJECTION"
    C8_ODD_PREDICATES = "C8_ODD_PREDICATES"
    C9_FALLBACK_STATE_MACHINE = "C9_FALLBACK_STATE_MACHINE"
    C10_RECOVERABILITY = "C10_RECOVERABILITY"
    C11_FACTOR_COMPOSITION = "C11_FACTOR_COMPOSITION"
    C12_EVALUATION_ORACLES = "C12_EVALUATION_ORACLES"
    C13_CORPUS_LINEAGE = "C13_CORPUS_LINEAGE"
    C14_MACRO_REFERENCE_MODEL = "C14_MACRO_REFERENCE_MODEL"
    C15_EXTERNAL_CONTROLLER = "C15_EXTERNAL_CONTROLLER"
    C16_PLANT_FIDELITY = "C16_PLANT_FIDELITY"
    C17_REAL_DATA_CALIBRATION = "C17_REAL_DATA_CALIBRATION"
    C18_COORDINATION_CHANNEL = "C18_COORDINATION_CHANNEL"
    C19_HUMAN_ROC = "C19_HUMAN_ROC"


@dataclass(frozen=True, slots=True)
class ScopeIdentity:
    scope_name: str
    scope_version: str
    base_version: str
    base_archive: str
    base_sha256: str


SCOPE_IDENTITY = ScopeIdentity(
    scope_name=SCOPE_NAME,
    scope_version=TESTBED_SCOPE_VERSION,
    base_version=A162_BASE_VERSION,
    base_archive=A162_BASE_ARCHIVE,
    base_sha256=A162_BASE_SHA256,
)


def disabled_feature_flags() -> Mapping[TestbedFeature, bool]:
    """Return the immutable C0 default: every post-A162 capability disabled."""

    return MappingProxyType({feature: False for feature in TestbedFeature})


DEFAULT_FEATURE_FLAGS = disabled_feature_flags()


def validate_feature_flags(flags: Mapping[TestbedFeature | str, bool]) -> Mapping[TestbedFeature, bool]:
    """Normalize an explicit complete feature map and reject unknown/implicit state."""

    normalized: dict[TestbedFeature, bool] = {}
    for key, value in flags.items():
        try:
            feature = key if isinstance(key, TestbedFeature) else TestbedFeature(str(key))
        except ValueError as exc:
            raise ValueError(f"unknown testbed feature {key!r}") from exc
        if not isinstance(value, bool):
            raise TypeError(f"feature {feature.value} must be bool")
        normalized[feature] = value
    missing = set(TestbedFeature) - set(normalized)
    if missing:
        names = ", ".join(sorted(feature.value for feature in missing))
        raise ValueError(f"missing explicit feature flags: {names}")
    return MappingProxyType(normalized)


T = TypeVar("T")


class ComponentAdapter(Protocol[T]):
    """Minimal boundary contract for optional external/testbed components."""

    component_id: TestbedFeature

    def apply(self, value: T) -> T:
        ...


@dataclass(frozen=True, slots=True)
class NoOpAdapter:
    """C0 compatibility adapter: preserve the exact input object and semantics."""

    component_id: TestbedFeature

    def apply(self, value: T) -> T:
        return value


def result_provenance(*, protocol_id: str, result_kind: str, extras: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Create canonical provenance fields required for every new-scope result."""

    if not protocol_id:
        raise ValueError("protocol_id must be non-empty")
    if not result_kind:
        raise ValueError("result_kind must be non-empty")
    payload: dict[str, Any] = {
        "scope_name": SCOPE_IDENTITY.scope_name,
        "scope_version": SCOPE_IDENTITY.scope_version,
        "base_version": SCOPE_IDENTITY.base_version,
        "base_archive": SCOPE_IDENTITY.base_archive,
        "base_sha256": SCOPE_IDENTITY.base_sha256,
        "protocol_id": protocol_id,
        "result_kind": result_kind,
    }
    if extras:
        reserved = set(payload).intersection(extras)
        if reserved:
            raise ValueError(f"extras may not replace provenance fields: {sorted(reserved)}")
        payload.update(extras)
    return MappingProxyType(payload)
