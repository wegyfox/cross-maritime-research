"""Versioned file-level contract for the CROSS Diagnostic Product.

This module is product/orchestration infrastructure.  It does not define a second
physical scenario model: the ``model.adapters`` and ``model.parameters`` sections bind
user-facing configuration to explicit existing C1/C2/C3/.../NextGen adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

_FORMAT = "cross-diagnostic-scenario-v1"
_SCHEMA_VERSION = 1


class DiagnosticScenarioContractError(ValueError):
    """Raised when a diagnostic scenario document is incomplete or invalid."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DiagnosticScenarioContractError(f"{name} must be JSON portable") from exc


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DiagnosticScenarioContractError(f"{name} must be an object")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiagnosticScenarioContractError(f"{name} must be a non-empty string")
    return value.strip()


def _string_set(value: Any, *, name: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple)):
        raise DiagnosticScenarioContractError(f"{name} must be an array of strings")
    out: set[str] = set()
    for index, item in enumerate(value):
        out.add(_nonempty_string(item, name=f"{name}[{index}]"))
    return frozenset(out)


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, (list, tuple)):
        raise DiagnosticScenarioContractError(f"{name} must be an array of strings")
    out = tuple(_nonempty_string(item, name=f"{name}[{index}]") for index, item in enumerate(value))
    if len(set(out)) != len(out):
        raise DiagnosticScenarioContractError(f"{name} must not contain duplicates")
    return out


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise DiagnosticScenarioContractError(f"{name} must be boolean")
    return value


@dataclass(frozen=True, slots=True)
class DiagnosticCapabilities:
    """Model capabilities used only to plan applicable/identity/reference steps.

    These flags describe what the configured model represents or can execute.  They are
    not outcome claims and must not encode effect sizes or bottleneck answers.
    """

    represented_factors: frozenset[str] = frozenset()
    canonical_factors: frozenset[str] = frozenset()
    executable_interventions: frozenset[str] = frozenset()
    reference_models: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.canonical_factors.issubset(self.represented_factors):
            extra = sorted(self.canonical_factors - self.represented_factors)
            raise DiagnosticScenarioContractError(
                f"canonical_factors must also be represented_factors: {extra}"
            )

    @classmethod
    def from_json_object(cls, payload: Mapping[str, Any]) -> "DiagnosticCapabilities":
        return cls(
            represented_factors=_string_set(
                payload.get("represented_factors"), name="capabilities.represented_factors"
            ),
            canonical_factors=_string_set(
                payload.get("canonical_factors"), name="capabilities.canonical_factors"
            ),
            executable_interventions=_string_set(
                payload.get("executable_interventions"),
                name="capabilities.executable_interventions",
            ),
            reference_models=_string_set(
                payload.get("reference_models"), name="capabilities.reference_models"
            ),
        )

    def to_json_object(self) -> dict[str, Any]:
        return {
            "represented_factors": sorted(self.represented_factors),
            "canonical_factors": sorted(self.canonical_factors),
            "executable_interventions": sorted(self.executable_interventions),
            "reference_models": sorted(self.reference_models),
        }


@dataclass(frozen=True, slots=True)
class DiagnosticExperimentSpec:
    paired: bool
    replications: int
    confidence_level: float = 0.95
    seed_namespace: str = "diagnostic-default"
    resume: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.paired, bool):
            raise DiagnosticScenarioContractError("experiment.paired must be boolean")
        if not isinstance(self.replications, int) or isinstance(self.replications, bool) or self.replications <= 0:
            raise DiagnosticScenarioContractError("experiment.replications must be a positive integer")
        if not isinstance(self.confidence_level, (int, float)) or isinstance(self.confidence_level, bool):
            raise DiagnosticScenarioContractError("experiment.confidence_level must be numeric")
        confidence = float(self.confidence_level)
        if not 0.0 < confidence < 1.0:
            raise DiagnosticScenarioContractError("experiment.confidence_level must be within (0, 1)")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self,
            "seed_namespace",
            _nonempty_string(self.seed_namespace, name="experiment.seed_namespace"),
        )
        if not isinstance(self.resume, bool):
            raise DiagnosticScenarioContractError("experiment.resume must be boolean")

    @classmethod
    def from_json_object(cls, payload: Mapping[str, Any]) -> "DiagnosticExperimentSpec":
        return cls(
            paired=_bool(payload.get("paired", True), name="experiment.paired"),
            replications=payload.get("replications", 1),
            confidence_level=payload.get("confidence_level", 0.95),
            seed_namespace=payload.get("seed_namespace", "diagnostic-default"),
            resume=_bool(payload.get("resume", True), name="experiment.resume"),
        )

    def to_json_object(self) -> dict[str, Any]:
        return {
            "paired": self.paired,
            "replications": self.replications,
            "confidence_level": self.confidence_level,
            "seed_namespace": self.seed_namespace,
            "resume": self.resume,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticScenarioSpec:
    scenario_id: str
    model_adapters: Mapping[str, str]
    model_parameters: Mapping[str, Any]
    capabilities: DiagnosticCapabilities
    experiment: DiagnosticExperimentSpec
    requested_interventions: tuple[str, ...]
    title: str | None = None
    description: str | None = None
    claim_scope: Mapping[str, bool] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _nonempty_string(self.scenario_id, name="scenario_id"))
        if self.title is not None:
            object.__setattr__(self, "title", _nonempty_string(self.title, name="title"))
        if self.description is not None:
            object.__setattr__(self, "description", _nonempty_string(self.description, name="description"))

        adapters: dict[str, str] = {}
        for key, value in dict(self.model_adapters).items():
            adapters[_nonempty_string(key, name="model.adapters key")] = _nonempty_string(
                value, name=f"model.adapters[{key!r}]"
            )
        if not adapters:
            raise DiagnosticScenarioContractError("model.adapters must contain at least one adapter")
        object.__setattr__(self, "model_adapters", MappingProxyType(dict(sorted(adapters.items()))))
        object.__setattr__(
            self,
            "model_parameters",
            MappingProxyType(_json_copy(dict(self.model_parameters), name="model.parameters")),
        )
        if not isinstance(self.capabilities, DiagnosticCapabilities):
            raise TypeError("capabilities must be DiagnosticCapabilities")
        if not isinstance(self.experiment, DiagnosticExperimentSpec):
            raise TypeError("experiment must be DiagnosticExperimentSpec")

        requested = tuple(self.requested_interventions)
        if len(set(requested)) != len(requested):
            raise DiagnosticScenarioContractError("requested_interventions must not contain duplicates")
        object.__setattr__(self, "requested_interventions", requested)

        scope = dict(self.claim_scope)
        allowed_scope = {"field_calibrated", "deployment_recommendation"}
        unknown = set(scope) - allowed_scope
        if unknown:
            raise DiagnosticScenarioContractError(f"unknown claim_scope fields: {sorted(unknown)}")
        normalized_scope = {
            "field_calibrated": _bool(scope.get("field_calibrated", False), name="claim_scope.field_calibrated"),
            "deployment_recommendation": _bool(
                scope.get("deployment_recommendation", False),
                name="claim_scope.deployment_recommendation",
            ),
        }
        object.__setattr__(self, "claim_scope", MappingProxyType(normalized_scope))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_json_copy(dict(self.metadata), name="metadata")),
        )

    @classmethod
    def from_json_object(cls, payload: Mapping[str, Any]) -> "DiagnosticScenarioSpec":
        source = _json_copy(dict(payload), name="scenario document")
        if source.get("format") != _FORMAT:
            raise DiagnosticScenarioContractError(
                f"unsupported format {source.get('format')!r}; expected {_FORMAT!r}"
            )
        if source.get("schema_version") != _SCHEMA_VERSION:
            raise DiagnosticScenarioContractError(
                f"unsupported schema_version {source.get('schema_version')!r}; expected {_SCHEMA_VERSION}"
            )
        model = _mapping(source.get("model"), name="model")
        adapters = _mapping(model.get("adapters"), name="model.adapters")
        parameters = _mapping(model.get("parameters", {}), name="model.parameters")
        capabilities = DiagnosticCapabilities.from_json_object(
            _mapping(source.get("capabilities", {}), name="capabilities")
        )
        experiment = DiagnosticExperimentSpec.from_json_object(
            _mapping(source.get("experiment", {}), name="experiment")
        )
        requested = _string_tuple(
            source.get("requested_interventions", ()), name="requested_interventions"
        )
        return cls(
            scenario_id=source.get("scenario_id"),
            title=source.get("title"),
            description=source.get("description"),
            model_adapters={str(key): value for key, value in adapters.items()},
            model_parameters=parameters,
            capabilities=capabilities,
            experiment=experiment,
            requested_interventions=requested,
            claim_scope=_mapping(source.get("claim_scope", {}), name="claim_scope"),
            metadata=_mapping(source.get("metadata", {}), name="metadata"),
        )

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "title": self.title,
            "description": self.description,
            "model": {
                "adapters": dict(self.model_adapters),
                "parameters": dict(self.model_parameters),
            },
            "capabilities": self.capabilities.to_json_object(),
            "experiment": self.experiment.to_json_object(),
            "requested_interventions": list(self.requested_interventions),
            "claim_scope": dict(self.claim_scope),
            "metadata": dict(self.metadata),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def scenario_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()



def load_diagnostic_scenario(path: str | Path) -> DiagnosticScenarioSpec:
    target = Path(path)
    raw = target.read_text(encoding="utf-8")
    suffix = target.suffix.lower()
    try:
        if suffix == ".json":
            payload = json.loads(raw)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(raw)
        else:
            raise DiagnosticScenarioContractError("scenario file must use .json, .yaml or .yml")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise DiagnosticScenarioContractError(f"invalid scenario syntax: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DiagnosticScenarioContractError("scenario document root must be an object")
    return DiagnosticScenarioSpec.from_json_object(payload)
