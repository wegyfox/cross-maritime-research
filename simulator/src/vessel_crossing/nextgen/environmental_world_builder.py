"""Public deterministic construction seam for environmental experiment worlds.

M0-05 graduates the safe subset of the historical private ``_world_factory`` hook to
an explicit, identity-bearing interface.  The builder is deliberately narrower than an
arbitrary callback: callers declare stable builder/configuration/backend identities and
receive one immutable build context.  The runtime validates those declarations against
the returned empty world before any source admission or vessel state is committed.

This module is architecture-only.  It does not change motion equations, planning,
safety policy, stochastic demand, controller timing or any scientific parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Protocol, runtime_checkable

from .environmental_candidate_transition import (
    EnvironmentalMotionBackend,
    NO_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID,
)
from .environmental_world import EnvironmentalWorld2D
from .vessel_classes import VesselProfileCatalog

_WORLD_BUILDER_FORMAT = "vessel-crossing-nextgen-environmental-world-builder"
_WORLD_BUILDER_SCHEMA_VERSION = 1
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class EnvironmentalWorldBuilderError(RuntimeError):
    """Raised when public world-builder identity or construction invariants fail."""


@dataclass(frozen=True, slots=True)
class EnvironmentalWorldBuildContext:
    """Minimal runtime context supplied to one public world builder."""

    start_time: float
    profile_catalog: VesselProfileCatalog

    def __post_init__(self) -> None:
        if not isfinite(self.start_time) or self.start_time < 0.0:
            raise ValueError("start_time must be finite and non-negative")
        if not isinstance(self.profile_catalog, VesselProfileCatalog):
            raise TypeError("profile_catalog must be VesselProfileCatalog")


@dataclass(frozen=True, slots=True)
class EnvironmentalWorldBuilderIdentity:
    """Canonical provenance for a public deterministic world builder."""

    builder_id: str
    configuration_sha256: str
    motion_backend_id: str | None
    motion_state_contract_id: str
    identity_sha256: str

    def to_json_object(self) -> dict[str, object]:
        return {
            "format": _WORLD_BUILDER_FORMAT,
            "schema_version": _WORLD_BUILDER_SCHEMA_VERSION,
            "builder_id": self.builder_id,
            "configuration_sha256": self.configuration_sha256,
            "motion_backend_id": self.motion_backend_id,
            "motion_state_contract_id": self.motion_state_contract_id,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentalWorldBuildResult:
    """Validated world plus the exact identity bound into result provenance."""

    world: EnvironmentalWorld2D
    identity: EnvironmentalWorldBuilderIdentity


@runtime_checkable
class EnvironmentalWorldBuilder(Protocol):
    """Stable public construction interface used by future experiment adapters.

    Implementations must expose immutable identity components and construct exactly one
    empty world aligned to the supplied start time.  The public executor validates that
    the returned motion backend agrees with the declared backend/state-contract
    identity and that the supplied profile catalog is not mutated during construction.
    """

    builder_id: str
    configuration_sha256: str
    expected_motion_backend_id: str | None
    expected_motion_state_contract_id: str

    def build(self, context: EnvironmentalWorldBuildContext) -> EnvironmentalWorld2D: ...


def environmental_world_builder_configuration_sha256(value: object) -> str:
    """Hash one JSON-serializable builder configuration with canonical encoding."""

    try:
        encoded = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("world-builder configuration must be JSON serializable") from exc
    return sha256(encoded).hexdigest()


def _identity_payload(
    *,
    builder_id: str,
    configuration_sha256: str,
    motion_backend_id: str | None,
    motion_state_contract_id: str,
) -> dict[str, object]:
    return {
        "format": _WORLD_BUILDER_FORMAT,
        "schema_version": _WORLD_BUILDER_SCHEMA_VERSION,
        "builder_id": builder_id,
        "configuration_sha256": configuration_sha256,
        "motion_backend_id": motion_backend_id,
        "motion_state_contract_id": motion_state_contract_id,
    }


def _canonical_identity_sha256(payload: dict[str, object]) -> str:
    return sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()


def environmental_world_builder_identity(
    builder: EnvironmentalWorldBuilder,
) -> EnvironmentalWorldBuilderIdentity:
    """Validate and freeze the identity advertised by one builder instance."""

    if not isinstance(builder, EnvironmentalWorldBuilder):
        raise TypeError("world_builder must implement EnvironmentalWorldBuilder")

    builder_id = builder.builder_id
    if not isinstance(builder_id, str) or not builder_id or builder_id != builder_id.strip():
        raise EnvironmentalWorldBuilderError(
            "builder_id must be a non-empty, already-trimmed string"
        )

    configuration_sha256 = builder.configuration_sha256
    if not isinstance(configuration_sha256, str) or not _HEX64.fullmatch(configuration_sha256):
        raise EnvironmentalWorldBuilderError(
            "configuration_sha256 must be a lowercase 64-hex SHA-256"
        )

    motion_backend_id = builder.expected_motion_backend_id
    if motion_backend_id is not None and (
        not isinstance(motion_backend_id, str)
        or not motion_backend_id
        or motion_backend_id != motion_backend_id.strip()
    ):
        raise EnvironmentalWorldBuilderError(
            "expected_motion_backend_id must be None or a non-empty trimmed string"
        )

    motion_state_contract_id = builder.expected_motion_state_contract_id
    if (
        not isinstance(motion_state_contract_id, str)
        or not motion_state_contract_id
        or motion_state_contract_id != motion_state_contract_id.strip()
    ):
        raise EnvironmentalWorldBuilderError(
            "expected_motion_state_contract_id must be a non-empty trimmed string"
        )
    if motion_backend_id is None and (
        motion_state_contract_id != NO_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID
    ):
        raise EnvironmentalWorldBuilderError(
            "a builder without a motion backend must declare the no-sidecar state contract"
        )

    payload = _identity_payload(
        builder_id=builder_id,
        configuration_sha256=configuration_sha256,
        motion_backend_id=motion_backend_id,
        motion_state_contract_id=motion_state_contract_id,
    )
    return EnvironmentalWorldBuilderIdentity(
        builder_id=builder_id,
        configuration_sha256=configuration_sha256,
        motion_backend_id=motion_backend_id,
        motion_state_contract_id=motion_state_contract_id,
        identity_sha256=_canonical_identity_sha256(payload),
    )


def preflight_environmental_world_builder(
    builder: EnvironmentalWorldBuilder,
) -> EnvironmentalWorldBuilderIdentity:
    """Validate public builder identity without constructing or mutating runtime state."""

    return environmental_world_builder_identity(builder)


def validate_environmental_world_builder_identity_document(value: object) -> None:
    """Fail closed on malformed/tampered archived public builder provenance."""

    if not isinstance(value, dict):
        raise EnvironmentalWorldBuilderError("world-builder provenance must be an object")
    if value.get("format") != _WORLD_BUILDER_FORMAT:
        raise EnvironmentalWorldBuilderError("unsupported world-builder provenance format")
    if value.get("schema_version") != _WORLD_BUILDER_SCHEMA_VERSION:
        raise EnvironmentalWorldBuilderError("unsupported world-builder provenance schema")

    builder_id = value.get("builder_id")
    configuration_sha256 = value.get("configuration_sha256")
    motion_backend_id = value.get("motion_backend_id")
    motion_state_contract_id = value.get("motion_state_contract_id")
    identity_sha256 = value.get("identity_sha256")

    if not isinstance(builder_id, str) or not builder_id or builder_id != builder_id.strip():
        raise EnvironmentalWorldBuilderError("archived builder_id is invalid")
    if not isinstance(configuration_sha256, str) or not _HEX64.fullmatch(configuration_sha256):
        raise EnvironmentalWorldBuilderError("archived configuration_sha256 is invalid")
    if motion_backend_id is not None and (
        not isinstance(motion_backend_id, str)
        or not motion_backend_id
        or motion_backend_id != motion_backend_id.strip()
    ):
        raise EnvironmentalWorldBuilderError("archived motion_backend_id is invalid")
    if (
        not isinstance(motion_state_contract_id, str)
        or not motion_state_contract_id
        or motion_state_contract_id != motion_state_contract_id.strip()
    ):
        raise EnvironmentalWorldBuilderError("archived motion_state_contract_id is invalid")
    if motion_backend_id is None and (
        motion_state_contract_id != NO_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID
    ):
        raise EnvironmentalWorldBuilderError(
            "archived no-backend builder must use the no-sidecar state contract"
        )
    if not isinstance(identity_sha256, str) or not _HEX64.fullmatch(identity_sha256):
        raise EnvironmentalWorldBuilderError("archived identity_sha256 is invalid")

    expected = _canonical_identity_sha256(
        _identity_payload(
            builder_id=builder_id,
            configuration_sha256=configuration_sha256,
            motion_backend_id=motion_backend_id,
            motion_state_contract_id=motion_state_contract_id,
        )
    )
    if identity_sha256 != expected:
        raise EnvironmentalWorldBuilderError("world-builder identity SHA-256 mismatch")


@dataclass(frozen=True, slots=True)
class EnvironmentalMotionBackendWorldBuilder:
    """Standard public builder for an empty world with one explicit motion backend.

    Supplying ``motion_backend=None`` intentionally builds the historical parity world.
    The builder object is meant for one experiment construction; the experiment owns
    the returned world and may exercise backend caches/telemetry during execution.
    """

    builder_id: str
    configuration_sha256: str
    motion_backend: EnvironmentalMotionBackend | None = None

    @property
    def expected_motion_backend_id(self) -> str | None:
        return None if self.motion_backend is None else self.motion_backend.backend_id

    @property
    def expected_motion_state_contract_id(self) -> str:
        return (
            NO_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID
            if self.motion_backend is None
            else self.motion_backend.backend_state_contract_id
        )

    def __post_init__(self) -> None:
        if self.motion_backend is not None and not isinstance(
            self.motion_backend, EnvironmentalMotionBackend
        ):
            raise TypeError("motion_backend must implement EnvironmentalMotionBackend")
        # Freeze/validate identity at construction so malformed builders fail before run.
        environmental_world_builder_identity(self)

    def build(self, context: EnvironmentalWorldBuildContext) -> EnvironmentalWorld2D:
        if not isinstance(context, EnvironmentalWorldBuildContext):
            raise TypeError("context must be EnvironmentalWorldBuildContext")
        return EnvironmentalWorld2D(
            (),
            limits={},
            environmental_drift_velocity={},
            start_time=context.start_time,
            motion_backend=self.motion_backend,
        )


def build_environmental_world(
    builder: EnvironmentalWorldBuilder,
    context: EnvironmentalWorldBuildContext,
) -> EnvironmentalWorldBuildResult:
    """Build and validate one public extension world before runtime state exists."""

    if not isinstance(context, EnvironmentalWorldBuildContext):
        raise TypeError("context must be EnvironmentalWorldBuildContext")
    identity_before = environmental_world_builder_identity(builder)

    profile_ids_before = context.profile_catalog.vessel_ids
    if profile_ids_before:
        raise EnvironmentalWorldBuilderError(
            "public world builder requires an empty profile catalog at construction"
        )

    world = builder.build(context)
    identity_after = environmental_world_builder_identity(builder)
    if identity_after != identity_before:
        raise EnvironmentalWorldBuilderError(
            "world-builder identity mutated during construction"
        )
    if context.profile_catalog.vessel_ids != profile_ids_before:
        raise EnvironmentalWorldBuilderError(
            "world builder must not mutate the experiment profile catalog"
        )
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world_builder.build() must return EnvironmentalWorld2D")
    if abs(world.time - context.start_time) > 1e-10:
        raise EnvironmentalWorldBuilderError(
            "world builder returned a world not aligned to experiment start_time"
        )
    if world.snapshot().vessels:
        raise EnvironmentalWorldBuilderError(
            "world builder must return an empty world before source admission"
        )

    backend = world.motion_backend
    if identity_before.motion_backend_id is None:
        if backend is not None:
            raise EnvironmentalWorldBuilderError(
                "world builder returned a motion backend despite declaring none"
            )
    else:
        if backend is None:
            raise EnvironmentalWorldBuilderError(
                "world builder declared a motion backend but returned none"
            )
        if not isinstance(backend, EnvironmentalMotionBackend):
            raise EnvironmentalWorldBuilderError(
                "world builder returned an invalid motion backend"
            )
        if backend.backend_id != identity_before.motion_backend_id:
            raise EnvironmentalWorldBuilderError(
                "returned motion backend_id disagrees with world-builder identity"
            )
        if (
            backend.backend_state_contract_id
            != identity_before.motion_state_contract_id
        ):
            raise EnvironmentalWorldBuilderError(
                "returned motion-state contract disagrees with world-builder identity"
            )

    return EnvironmentalWorldBuildResult(world=world, identity=identity_before)
