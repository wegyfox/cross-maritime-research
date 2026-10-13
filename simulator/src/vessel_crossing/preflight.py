"""Fail-fast validation for frozen executable replication protocols."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

from .executor import GeometryFactory, PolicyFactory, build_execution_components
from .experiments import PAIRABLE_METRICS
from .metrics import AggregateMetrics
from .protocol import ExecutableReplicationProtocol, load_executable_replication_protocol


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    execution_protocol_id: str
    execution_protocol_fingerprint_sha256: str
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)


def fingerprint_sidecar_path(protocol_path: str | Path) -> Path:
    path = Path(protocol_path)
    return path.with_name(path.name + ".fingerprint.sha256")


def read_expected_fingerprint(path: str | Path) -> str:
    value = Path(path).read_text(encoding="utf-8").strip().split()[0]
    if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError("expected fingerprint must be a 64-character SHA-256 hex digest")
    return value.lower()


def _check(name: str, function) -> PreflightCheck:
    try:
        detail = function()
    except Exception as exc:  # report all independent preflight failures
        return PreflightCheck(name, False, f"{type(exc).__name__}: {exc}")
    return PreflightCheck(name, True, str(detail))


def preflight_executable_protocol(
    protocol: ExecutableReplicationProtocol,
    *,
    expected_fingerprint: str | None = None,
    policy_registry: Mapping[str, PolicyFactory] | None = None,
    geometry_registry: Mapping[str, GeometryFactory] | None = None,
) -> PreflightReport:
    """Validate one protocol before any stochastic compute starts."""

    actual_fingerprint = protocol.fingerprint_sha256()
    checks: list[PreflightCheck] = []

    def semantic_fingerprint_detail() -> str:
        if expected_fingerprint is None:
            return f"computed={actual_fingerprint}; no external expected fingerprint supplied"
        expected = expected_fingerprint.lower()
        if actual_fingerprint != expected:
            raise ValueError(
                f"semantic fingerprint mismatch: expected {expected}, got {actual_fingerprint}"
            )
        return f"matches {actual_fingerprint}"

    checks.append(_check("semantic_protocol_fingerprint", semantic_fingerprint_detail))

    def supersession_detail() -> str:
        frozen = protocol.supersedes_study_fingerprint
        current = protocol.study.fingerprint_sha256()
        if frozen is not None and frozen != current:
            raise ValueError(
                "supersedes_study_fingerprint does not match embedded study fingerprint"
            )
        return f"embedded_study_fingerprint={current}"

    checks.append(_check("embedded_study_fingerprint", supersession_detail))

    components: dict[str, object] = {}

    def component_detail() -> str:
        geometry, simulation_config, controllers = build_execution_components(
            protocol,
            policy_registry=policy_registry,
            geometry_registry=geometry_registry,
        )
        components["geometry"] = geometry
        components["simulation_config"] = simulation_config
        components["controllers"] = controllers
        return (
            f"geometry={protocol.model.geometry_name}; policies="
            f"{','.join(controller.policy.name for controller in controllers)}"
        )

    checks.append(_check("component_registry_resolution", component_detail))

    def route_detail() -> str:
        geometry = components.get("geometry")
        if geometry is None:
            geometry, simulation_config, controllers = build_execution_components(
                protocol,
                policy_registry=policy_registry,
                geometry_registry=geometry_registry,
            )
            components.update(
                geometry=geometry,
                simulation_config=simulation_config,
                controllers=controllers,
            )
        missing: list[str] = []
        for route_name in protocol.study.route_names:
            try:
                geometry.by_name(route_name)  # type: ignore[attr-defined]
            except KeyError:
                missing.append(route_name)
        if missing:
            raise ValueError(f"routes missing from geometry: {missing}")
        return f"routes={','.join(protocol.study.route_names)}"

    checks.append(_check("study_routes_exist", route_detail))

    def zone_detail() -> str:
        geometry = components.get("geometry")
        config = components.get("simulation_config")
        if geometry is None or config is None:
            geometry, config, controllers = build_execution_components(
                protocol,
                policy_registry=policy_registry,
                geometry_registry=geometry_registry,
            )
            components.update(
                geometry=geometry,
                simulation_config=config,
                controllers=controllers,
            )
        zone = config.conflict_zone  # type: ignore[attr-defined]
        if zone is None:
            raise ValueError("execution protocol does not resolve a conflict zone")
        intervals: list[str] = []
        for route_name in protocol.study.route_names:
            route = geometry.by_name(route_name)  # type: ignore[attr-defined]
            interval = zone.route_interval(route)
            if interval is None or interval.exit_progress <= interval.entry_progress:
                raise ValueError(f"route {route_name!r} does not cross conflict zone")
            intervals.append(
                f"{route_name}:{interval.entry_progress:.6g}-{interval.exit_progress:.6g}"
            )
        return "; ".join(intervals)

    checks.append(_check("conflict_zone_route_intersections", zone_detail))

    def primary_metric_detail() -> str:
        metric = protocol.study.primary_metric
        if metric not in PAIRABLE_METRICS:
            raise ValueError(f"primary metric {metric!r} is not pairable")
        return metric

    checks.append(_check("primary_metric_pairable", primary_metric_detail))

    aggregate_fields = {field.name for field in fields(AggregateMetrics)}

    def secondary_metric_detail() -> str:
        unknown = [
            metric
            for metric in protocol.study.secondary_metrics
            if metric not in aggregate_fields
        ]
        if unknown:
            raise ValueError(f"unknown AggregateMetrics fields: {unknown}")
        return f"{len(protocol.study.secondary_metrics)} metrics"

    checks.append(_check("secondary_metric_names", secondary_metric_detail))

    def saturation_detail() -> str:
        names = tuple(profile.name for profile in protocol.study.saturation_profiles)
        if protocol.study.primary_saturation_profile not in names:
            raise ValueError("primary saturation profile is missing")
        return (
            f"profiles={','.join(names)}; primary={protocol.study.primary_saturation_profile}"
        )

    checks.append(_check("saturation_profiles", saturation_detail))

    def replication_schedule_detail() -> str:
        study = protocol.study
        if study.initial_replications > study.max_replications:
            raise ValueError("initial_replications exceeds max_replications")
        if study.replication_batch_size <= 0:
            raise ValueError("replication_batch_size must be positive")
        remaining = study.max_replications - study.initial_replications
        batches = (remaining + study.replication_batch_size - 1) // study.replication_batch_size
        return (
            f"initial={study.initial_replications}; batch={study.replication_batch_size}; "
            f"max={study.max_replications}; at_most_additional_batches={batches}"
        )

    checks.append(_check("replication_schedule", replication_schedule_detail))

    return PreflightReport(
        execution_protocol_id=protocol.execution_protocol_id,
        execution_protocol_fingerprint_sha256=actual_fingerprint,
        checks=tuple(checks),
    )


def preflight_protocol_file(
    protocol_path: str | Path,
    *,
    expected_fingerprint: str | None = None,
    require_fingerprint_sidecar: bool = True,
    policy_registry: Mapping[str, PolicyFactory] | None = None,
    geometry_registry: Mapping[str, GeometryFactory] | None = None,
) -> PreflightReport:
    """Load and validate a protocol file with optional semantic sidecar enforcement."""

    protocol_path = Path(protocol_path)
    protocol = load_executable_replication_protocol(protocol_path)
    expected = expected_fingerprint
    if expected is None and require_fingerprint_sidecar:
        sidecar = fingerprint_sidecar_path(protocol_path)
        if not sidecar.exists():
            return PreflightReport(
                execution_protocol_id=protocol.execution_protocol_id,
                execution_protocol_fingerprint_sha256=protocol.fingerprint_sha256(),
                checks=(
                    PreflightCheck(
                        "semantic_protocol_fingerprint",
                        False,
                        f"fingerprint sidecar not found: {sidecar}",
                    ),
                ),
            )
        expected = read_expected_fingerprint(sidecar)
    return preflight_executable_protocol(
        protocol,
        expected_fingerprint=expected,
        policy_registry=policy_registry,
        geometry_registry=geometry_registry,
    )
