"""Identity-bearing same-route-following extension over the frozen A152 FIFO run.

The historical A152 config/result schema remains unchanged.  This wrapper freezes the
identity of one reusable same-route following mechanism and then executes A152 through a
package-private supervisor seam.  It is intentionally generic: the R2 pair-context
supervisor is the first caller, while later longitudinal behavior models may expose the
same identity contract without creating another full-network simulator fork.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Mapping, Protocol

from ..archive import environment_manifest
from ..version import __version__
from .environmental_network_experiment import _canonical_json, _json_value
from .environmental_network_fifo_experiment import (
    EnvironmentalNetworkFIFOExperimentConfig,
    EnvironmentalNetworkFIFOExperimentResult,
    _run_environmental_network_fifo_experiment_with_traffic,
)
from .network_traffic import PoissonNetworkTrafficRealization, generate_poisson_network_traffic
from .same_route_following import SameRouteFollowingSupervisor

_FORMAT = "vessel-crossing-nextgen-environmental-network-fifo-following-extension"
_SCHEMA_VERSION = 1
_RESULT_FORMAT = "vessel-crossing-nextgen-environmental-network-fifo-following-extension-result"
_RESULT_SCHEMA_VERSION = 1
_DOMAIN = b"vessel-crossing-nextgen-fifo-following-extension\x00"


class IdentityBearingSameRouteFollowing(Protocol):
    """Runtime identity contract required by the extension wrapper."""

    mechanism_id: str
    config_identity_sha256: str


class EnvironmentalNetworkFIFOFollowingExtensionError(RuntimeError):
    """Raised when a same-route extension cannot be bound or frozen safely."""


@dataclass(frozen=True, slots=True)
class SameRouteFollowingMechanismBinding:
    """Portable identity/provenance of one same-route following mechanism."""

    mechanism_id: str
    config_identity_sha256: str
    config_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.mechanism_id, str) or not self.mechanism_id:
            raise ValueError("mechanism_id must be a non-empty string")
        digest = self.config_identity_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise ValueError("config_identity_sha256 must be a lowercase SHA-256 digest")
        payload = json.loads(json.dumps(dict(self.config_payload), sort_keys=True))
        object.__setattr__(self, "config_payload", payload)
        actual = sha256(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        if actual != digest:
            raise ValueError(
                "config_payload does not reproduce config_identity_sha256"
            )

    def to_json_object(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "config_identity_sha256": self.config_identity_sha256,
            "config_payload": self.config_payload,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOFollowingExtensionConfig:
    """One frozen A152 base config plus one identity-bearing following mechanism."""

    base: EnvironmentalNetworkFIFOExperimentConfig
    following: SameRouteFollowingMechanismBinding
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base, EnvironmentalNetworkFIFOExperimentConfig):
            raise TypeError("base must be EnvironmentalNetworkFIFOExperimentConfig")
        if not isinstance(self.following, SameRouteFollowingMechanismBinding):
            raise TypeError("following must be SameRouteFollowingMechanismBinding")
        if self.label is not None:
            label = self.label.strip()
            if not label:
                raise ValueError("label must be non-empty when supplied")
            object.__setattr__(self, "label", label)

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "label": self.label,
            "base_config_sha256": self.base.config_sha256,
            "base": self.base.to_json_object(),
            "following": self.following.to_json_object(),
            "application_scope": "DYNAMIC_INGRESS_IF_ENABLED_AND_PERSISTENT_SAME_ROUTE",
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def config_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()

    @property
    def run_id(self) -> str:
        return sha256(
            _DOMAIN
            + __version__.encode("utf-8")
            + b"\x00"
            + self.config_sha256.encode("ascii")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOFollowingExtensionResult:
    """Portable identity-bearing wrapper over one complete A152 arm result."""

    config: EnvironmentalNetworkFIFOFollowingExtensionConfig
    inner: EnvironmentalNetworkFIFOExperimentResult
    canonical_bytes: bytes = field(repr=False)

    @property
    def run_id(self) -> str:
        return self.config.run_id

    @property
    def config_sha256(self) -> str:
        return self.config.config_sha256

    @property
    def result_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes

    def to_json_object(self) -> dict[str, Any]:
        value = json.loads(self.canonical_bytes.decode("utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - construction invariant
            raise EnvironmentalNetworkFIFOFollowingExtensionError(
                "frozen following-extension result is not a JSON object"
            )
        return value


def mechanism_binding_from_supervisor(
    supervisor: SameRouteFollowingSupervisor,
    *,
    config_payload: Mapping[str, Any],
) -> SameRouteFollowingMechanismBinding:
    """Create a binding only when runtime and payload identities agree."""

    if not isinstance(supervisor, SameRouteFollowingSupervisor):
        raise TypeError("supervisor must be SameRouteFollowingSupervisor")
    mechanism_id = getattr(supervisor, "mechanism_id", None)
    config_identity_sha256 = getattr(supervisor, "config_identity_sha256", None)
    if not isinstance(mechanism_id, str) or not mechanism_id:
        raise EnvironmentalNetworkFIFOFollowingExtensionError(
            "supervisor lacks a stable mechanism_id"
        )
    if not isinstance(config_identity_sha256, str):
        raise EnvironmentalNetworkFIFOFollowingExtensionError(
            "supervisor lacks a stable config_identity_sha256"
        )
    return SameRouteFollowingMechanismBinding(
        mechanism_id=mechanism_id,
        config_identity_sha256=config_identity_sha256,
        config_payload=config_payload,
    )


def _validate_runtime_binding(
    config: EnvironmentalNetworkFIFOFollowingExtensionConfig,
    supervisor: SameRouteFollowingSupervisor,
) -> None:
    if not isinstance(supervisor, SameRouteFollowingSupervisor):
        raise TypeError("supervisor must be SameRouteFollowingSupervisor")
    mechanism_id = getattr(supervisor, "mechanism_id", None)
    digest = getattr(supervisor, "config_identity_sha256", None)
    if mechanism_id != config.following.mechanism_id:
        raise EnvironmentalNetworkFIFOFollowingExtensionError(
            "runtime supervisor mechanism_id disagrees with frozen binding"
        )
    if digest != config.following.config_identity_sha256:
        raise EnvironmentalNetworkFIFOFollowingExtensionError(
            "runtime supervisor config identity disagrees with frozen binding"
        )


def run_environmental_network_fifo_following_extension_with_traffic(
    config: EnvironmentalNetworkFIFOFollowingExtensionConfig,
    traffic: PoissonNetworkTrafficRealization,
    supervisor: SameRouteFollowingSupervisor,
) -> EnvironmentalNetworkFIFOFollowingExtensionResult:
    """Execute one pre-realized full-network arm with frozen following provenance."""

    if not isinstance(config, EnvironmentalNetworkFIFOFollowingExtensionConfig):
        raise TypeError("config must be EnvironmentalNetworkFIFOFollowingExtensionConfig")
    _validate_runtime_binding(config, supervisor)
    inner = _run_environmental_network_fifo_experiment_with_traffic(
        config.base,
        traffic,
        _same_route_following_override=supervisor,
    )
    document = {
        "format": _RESULT_FORMAT,
        "schema_version": _RESULT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "config_sha256": config.config_sha256,
        "config": config.to_json_object(),
        "execution_environment": environment_manifest(),
        "inner_a152_result_sha256": inner.result_sha256,
        "inner_a152": inner.to_json_object(),
    }
    return EnvironmentalNetworkFIFOFollowingExtensionResult(
        config=config,
        inner=inner,
        canonical_bytes=_canonical_json(document),
    )


def run_environmental_network_fifo_following_extension(
    config: EnvironmentalNetworkFIFOFollowingExtensionConfig,
    supervisor: SameRouteFollowingSupervisor,
) -> EnvironmentalNetworkFIFOFollowingExtensionResult:
    """Generate the base seeded traffic once, then execute the following extension."""

    if not isinstance(config, EnvironmentalNetworkFIFOFollowingExtensionConfig):
        raise TypeError("config must be EnvironmentalNetworkFIFOFollowingExtensionConfig")
    base = config.base
    scenario = base.scenario
    traffic = generate_poisson_network_traffic(
        base.network,
        base.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    return run_environmental_network_fifo_following_extension_with_traffic(
        config, traffic, supervisor
    )
