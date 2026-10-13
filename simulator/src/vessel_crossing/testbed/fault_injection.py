"""C7 deterministic unified degradation / fault injection bus.

Fault injection is infrastructure, not a novelty claim.  This module provides one
canonical schedule and audit identity for controlled degradations across observation,
planner, communication, actuator and environmental-belief layers.  Thin adapters turn
active fault specs into layer-specific effects without coupling the A162 physics kernel
to testbed fault semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Iterable

from vessel_crossing.nextgen.dynamics import ManeuverLimits

from .environment_belief import (
    EnvironmentalBeliefProvider,
    EnvironmentalBeliefSnapshot,
    EnvironmentalTruthField,
)
from .scope import TestbedFeature
from .track_lifecycle import DetectionTargetClass

_EPS = 1e-12


class FaultTargetComponent(str, Enum):
    OBSERVATION = "OBSERVATION"
    NAV_DATA = "NAV_DATA"
    PLANNER = "PLANNER"
    COMMUNICATION = "COMMUNICATION"
    ACTUATOR = "ACTUATOR"
    ENVIRONMENT_BELIEF = "ENVIRONMENT_BELIEF"


class FaultType(str, Enum):
    OBSERVATION_DROPOUT = "OBSERVATION_DROPOUT"
    OBSERVATION_STALE = "OBSERVATION_STALE"
    OBSERVATION_BIAS = "OBSERVATION_BIAS"
    OBSERVATION_LATENCY = "OBSERVATION_LATENCY"
    AIS_CORRUPTION = "AIS_CORRUPTION"
    GNSS_CORRUPTION = "GNSS_CORRUPTION"
    PLANNER_STALE = "PLANNER_STALE"
    PLANNER_NO_RESPONSE = "PLANNER_NO_RESPONSE"
    PLANNER_INVALID_RESPONSE = "PLANNER_INVALID_RESPONSE"
    COMMUNICATION_DELAY = "COMMUNICATION_DELAY"
    COMMUNICATION_LOSS = "COMMUNICATION_LOSS"
    ACTUATOR_DELAY = "ACTUATOR_DELAY"
    ACTUATOR_SATURATION = "ACTUATOR_SATURATION"
    REDUCED_TURN_AUTHORITY = "REDUCED_TURN_AUTHORITY"
    PROPULSION_DEGRADATION = "PROPULSION_DEGRADATION"
    ENVIRONMENT_ESTIMATE_FAULT = "ENVIRONMENT_ESTIMATE_FAULT"


_EXPECTED_COMPONENT = {
    FaultType.OBSERVATION_DROPOUT: FaultTargetComponent.OBSERVATION,
    FaultType.OBSERVATION_STALE: FaultTargetComponent.OBSERVATION,
    FaultType.OBSERVATION_BIAS: FaultTargetComponent.OBSERVATION,
    FaultType.OBSERVATION_LATENCY: FaultTargetComponent.OBSERVATION,
    FaultType.AIS_CORRUPTION: FaultTargetComponent.NAV_DATA,
    FaultType.GNSS_CORRUPTION: FaultTargetComponent.NAV_DATA,
    FaultType.PLANNER_STALE: FaultTargetComponent.PLANNER,
    FaultType.PLANNER_NO_RESPONSE: FaultTargetComponent.PLANNER,
    FaultType.PLANNER_INVALID_RESPONSE: FaultTargetComponent.PLANNER,
    FaultType.COMMUNICATION_DELAY: FaultTargetComponent.COMMUNICATION,
    FaultType.COMMUNICATION_LOSS: FaultTargetComponent.COMMUNICATION,
    FaultType.ACTUATOR_DELAY: FaultTargetComponent.ACTUATOR,
    FaultType.ACTUATOR_SATURATION: FaultTargetComponent.ACTUATOR,
    FaultType.REDUCED_TURN_AUTHORITY: FaultTargetComponent.ACTUATOR,
    FaultType.PROPULSION_DEGRADATION: FaultTargetComponent.ACTUATOR,
    FaultType.ENVIRONMENT_ESTIMATE_FAULT: FaultTargetComponent.ENVIRONMENT_BELIEF,
}


def _param_value(parameters: tuple[tuple[str, float | str | bool], ...], key: str, default: float) -> float:
    for name, value in parameters:
        if name == key:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"fault parameter {key!r} must be numeric")
            value = float(value)
            if not isfinite(value):
                raise ValueError(f"fault parameter {key!r} must be finite")
            return value
    return default


@dataclass(frozen=True, slots=True)
class FaultSpec:
    fault_id: str
    fault_type: FaultType
    target_component: FaultTargetComponent
    target_id: str
    start_min: float
    end_min: float
    severity: float
    parameters: tuple[tuple[str, float | str | bool], ...] = ()
    source: str = "DECLARED"
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.fault_id or not self.target_id or not self.source:
            raise ValueError("fault_id, target_id and source must be non-empty")
        if not isfinite(self.start_min) or self.start_min < 0.0:
            raise ValueError("start_min must be finite and non-negative")
        if not isfinite(self.end_min) or self.end_min <= self.start_min:
            raise ValueError("end_min must be finite and greater than start_min")
        if not isfinite(self.severity) or not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be within [0, 1]")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.target_component is not _EXPECTED_COMPONENT[self.fault_type]:
            raise ValueError(
                f"{self.fault_type.value} must target {_EXPECTED_COMPONENT[self.fault_type].value}"
            )
        names = [name for name, _ in self.parameters]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("fault parameters must be unique and sorted by name")
        for name, value in self.parameters:
            if not name:
                raise ValueError("fault parameter names must be non-empty")
            if not isinstance(value, (float, str, bool, int)):
                raise TypeError("fault parameter values must be scalar")
            if isinstance(value, float) and not isfinite(value):
                raise ValueError("fault float parameters must be finite")

    @classmethod
    def from_mapping(
        cls,
        *,
        fault_id: str,
        fault_type: FaultType,
        target_component: FaultTargetComponent,
        target_id: str,
        start_min: float,
        end_min: float,
        severity: float,
        parameters: dict[str, float | str | bool] | None = None,
        source: str = "DECLARED",
        seed: int | None = None,
    ) -> "FaultSpec":
        return cls(
            fault_id=fault_id,
            fault_type=fault_type,
            target_component=target_component,
            target_id=target_id,
            start_min=start_min,
            end_min=end_min,
            severity=severity,
            parameters=tuple(sorted((parameters or {}).items())),
            source=source,
            seed=seed,
        )

    def active_at(self, time_min: float) -> bool:
        return self.start_min <= time_min < self.end_min

    def identity_sha256(self) -> str:
        return _sha256_payload(self.audit_payload())

    def audit_payload(self) -> dict[str, object]:
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "target_component": self.target_component.value,
            "target_id": self.target_id,
            "start_min": self.start_min,
            "end_min": self.end_min,
            "severity": self.severity,
            "parameters": [[k, v] for k, v in self.parameters],
            "source": self.source,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class FaultAuditRecord:
    fault_id: str
    fault_sha256: str
    fault_type: FaultType
    target_component: FaultTargetComponent
    target_id: str
    start_min: float
    end_min: float
    severity: float
    source: str
    seed: int | None


@dataclass(frozen=True, slots=True)
class FaultInjectionBus:
    faults: tuple[FaultSpec, ...] = ()
    bus_id: str = "cross-c7/fault-bus/v1"
    component_id = TestbedFeature.C7_FAULT_INJECTION

    def __post_init__(self) -> None:
        if not self.bus_id:
            raise ValueError("bus_id must be non-empty")
        ids = [f.fault_id for f in self.faults]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("faults must be unique and sorted by fault_id")

    @classmethod
    def from_unsorted(
        cls, faults: Iterable[FaultSpec], *, bus_id: str = "cross-c7/fault-bus/v1"
    ) -> "FaultInjectionBus":
        return cls(tuple(sorted(tuple(faults), key=lambda f: f.fault_id)), bus_id=bus_id)

    def active_faults(
        self,
        *,
        time_min: float,
        component: FaultTargetComponent | None = None,
        target_id: str | None = None,
    ) -> tuple[FaultSpec, ...]:
        if not isfinite(time_min) or time_min < 0.0:
            raise ValueError("time_min must be finite and non-negative")
        return tuple(
            fault
            for fault in self.faults
            if fault.active_at(time_min)
            and (component is None or fault.target_component is component)
            and (target_id is None or fault.target_id in {target_id, "*"})
        )

    def audit_records(self) -> tuple[FaultAuditRecord, ...]:
        return tuple(
            FaultAuditRecord(
                fault_id=f.fault_id,
                fault_sha256=f.identity_sha256(),
                fault_type=f.fault_type,
                target_component=f.target_component,
                target_id=f.target_id,
                start_min=f.start_min,
                end_min=f.end_min,
                severity=f.severity,
                source=f.source,
                seed=f.seed,
            )
            for f in self.faults
        )

    def configuration_sha256(self) -> str:
        return _sha256_payload(
            {"bus_id": self.bus_id, "faults": [f.audit_payload() for f in self.faults]}
        )

    def observation_effect(self, *, time_min: float, target_id: str) -> "ObservationFaultEffect":
        effect = ObservationFaultEffect()
        faults = self.active_faults(
            time_min=time_min, component=FaultTargetComponent.OBSERVATION, target_id=target_id
        ) + self.active_faults(
            time_min=time_min, component=FaultTargetComponent.NAV_DATA, target_id=target_id
        )
        for fault in faults:
            if fault.fault_type is FaultType.OBSERVATION_DROPOUT:
                effect = replace(effect, dropout=effect.dropout or fault.severity > 0.0)
            elif fault.fault_type is FaultType.OBSERVATION_STALE:
                effect = replace(
                    effect,
                    extra_stale_age_min=max(
                        effect.extra_stale_age_min,
                        _param_value(fault.parameters, "stale_age_min", fault.severity),
                    ),
                )
            elif fault.fault_type is FaultType.OBSERVATION_LATENCY:
                effect = replace(
                    effect,
                    extra_latency_min=effect.extra_latency_min
                    + _param_value(fault.parameters, "latency_min", fault.severity),
                )
            elif fault.fault_type in {FaultType.OBSERVATION_BIAS, FaultType.GNSS_CORRUPTION}:
                effect = replace(
                    effect,
                    x_bias_nm=effect.x_bias_nm + _param_value(fault.parameters, "x_bias_nm", 0.0),
                    y_bias_nm=effect.y_bias_nm + _param_value(fault.parameters, "y_bias_nm", 0.0),
                    data_corruption_severity=max(effect.data_corruption_severity, fault.severity),
                )
            elif fault.fault_type is FaultType.AIS_CORRUPTION:
                effect = replace(
                    effect,
                    data_corruption_severity=max(effect.data_corruption_severity, fault.severity),
                )
        return effect

    def planner_effect(self, *, time_min: float, planner_id: str) -> "PlannerFaultEffect":
        result = PlannerFaultEffect()
        for fault in self.active_faults(
            time_min=time_min, component=FaultTargetComponent.PLANNER, target_id=planner_id
        ):
            if fault.fault_type is FaultType.PLANNER_STALE:
                result = replace(result, stale=True)
            elif fault.fault_type is FaultType.PLANNER_NO_RESPONSE:
                result = replace(result, no_response=True)
            elif fault.fault_type is FaultType.PLANNER_INVALID_RESPONSE:
                result = replace(result, invalid_response=True)
        return result

    def communication_effect(self, *, time_min: float, channel_id: str) -> "CommunicationFaultEffect":
        result = CommunicationFaultEffect()
        for fault in self.active_faults(
            time_min=time_min, component=FaultTargetComponent.COMMUNICATION, target_id=channel_id
        ):
            if fault.fault_type is FaultType.COMMUNICATION_LOSS:
                result = replace(result, lost=result.lost or fault.severity > 0.0)
            elif fault.fault_type is FaultType.COMMUNICATION_DELAY:
                result = replace(
                    result,
                    extra_delay_min=result.extra_delay_min
                    + _param_value(fault.parameters, "delay_min", fault.severity),
                )
        return result

    def actuator_effect(self, *, time_min: float, vessel_id: str) -> "ActuatorFaultEffect":
        result = ActuatorFaultEffect()
        for fault in self.active_faults(
            time_min=time_min, component=FaultTargetComponent.ACTUATOR, target_id=vessel_id
        ):
            if fault.fault_type is FaultType.ACTUATOR_DELAY:
                result = replace(
                    result,
                    extra_delay_min=result.extra_delay_min
                    + _param_value(fault.parameters, "delay_min", fault.severity),
                )
            elif fault.fault_type is FaultType.ACTUATOR_SATURATION:
                result = replace(
                    result,
                    command_authority_factor=min(
                        result.command_authority_factor,
                        _param_value(fault.parameters, "authority_factor", 1.0 - fault.severity),
                    ),
                )
            elif fault.fault_type is FaultType.REDUCED_TURN_AUTHORITY:
                result = replace(
                    result,
                    turn_authority_factor=min(
                        result.turn_authority_factor,
                        _param_value(fault.parameters, "turn_factor", 1.0 - fault.severity),
                    ),
                )
            elif fault.fault_type is FaultType.PROPULSION_DEGRADATION:
                result = replace(
                    result,
                    propulsion_authority_factor=min(
                        result.propulsion_authority_factor,
                        _param_value(fault.parameters, "propulsion_factor", 1.0 - fault.severity),
                    ),
                )
        return result.validated()

    def environment_effect(self, *, time_min: float, target_id: str = "environment") -> "EnvironmentEstimateFaultEffect":
        result = EnvironmentEstimateFaultEffect()
        for fault in self.active_faults(
            time_min=time_min,
            component=FaultTargetComponent.ENVIRONMENT_BELIEF,
            target_id=target_id,
        ):
            result = replace(
                result,
                extra_delay_min=result.extra_delay_min
                + _param_value(fault.parameters, "delay_min", 0.0),
                current_east_bias_nm_per_min=result.current_east_bias_nm_per_min
                + _param_value(fault.parameters, "current_east_bias_nm_per_min", 0.0),
                current_north_bias_nm_per_min=result.current_north_bias_nm_per_min
                + _param_value(fault.parameters, "current_north_bias_nm_per_min", 0.0),
                wind_east_bias_nm_per_min=result.wind_east_bias_nm_per_min
                + _param_value(fault.parameters, "wind_east_bias_nm_per_min", 0.0),
                wind_north_bias_nm_per_min=result.wind_north_bias_nm_per_min
                + _param_value(fault.parameters, "wind_north_bias_nm_per_min", 0.0),
                corruption_severity=max(result.corruption_severity, fault.severity),
            )
        return result


@dataclass(frozen=True, slots=True)
class ObservationFaultEffect:
    dropout: bool = False
    extra_stale_age_min: float = 0.0
    extra_latency_min: float = 0.0
    x_bias_nm: float = 0.0
    y_bias_nm: float = 0.0
    data_corruption_severity: float = 0.0


@dataclass(frozen=True, slots=True)
class PlannerFaultEffect:
    stale: bool = False
    no_response: bool = False
    invalid_response: bool = False


@dataclass(frozen=True, slots=True)
class CommunicationFaultEffect:
    lost: bool = False
    extra_delay_min: float = 0.0


@dataclass(frozen=True, slots=True)
class ActuatorFaultEffect:
    extra_delay_min: float = 0.0
    command_authority_factor: float = 1.0
    turn_authority_factor: float = 1.0
    propulsion_authority_factor: float = 1.0

    def validated(self) -> "ActuatorFaultEffect":
        for name in (
            "command_authority_factor", "turn_authority_factor", "propulsion_authority_factor"
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if not isfinite(self.extra_delay_min) or self.extra_delay_min < 0.0:
            raise ValueError("extra_delay_min must be finite and non-negative")
        return self


@dataclass(frozen=True, slots=True)
class EnvironmentEstimateFaultEffect:
    extra_delay_min: float = 0.0
    current_east_bias_nm_per_min: float = 0.0
    current_north_bias_nm_per_min: float = 0.0
    wind_east_bias_nm_per_min: float = 0.0
    wind_north_bias_nm_per_min: float = 0.0
    corruption_severity: float = 0.0


@dataclass(frozen=True, slots=True)
class FaultBusVisibilityHook:
    """C5 adapter: observation-dropout faults become visibility misses."""

    bus: FaultInjectionBus
    hook_id: str = "cross-c7/fault-bus-visibility/v1"

    def visible(
        self,
        *,
        time_min: float,
        entity_id: str,
        target_class: DetectionTargetClass,
    ) -> bool:
        del target_class
        return not self.bus.observation_effect(time_min=time_min, target_id=entity_id).dropout


@dataclass(frozen=True, slots=True)
class FaultAwareEnvironmentalBeliefProvider:
    """C6 adapter applying C7 estimate delay/bias without changing field truth."""

    delegate: EnvironmentalBeliefProvider
    bus: FaultInjectionBus
    target_id: str = "environment"
    provider_id: str = "cross-c7/fault-aware-environment-belief/v1"
    component_id = TestbedFeature.C7_FAULT_INJECTION

    def observe(
        self,
        field: EnvironmentalTruthField,
        *,
        time_min: float,
        x_east_nm: float,
        y_north_nm: float,
    ) -> EnvironmentalBeliefSnapshot:
        effect = self.bus.environment_effect(time_min=time_min, target_id=self.target_id)
        delegated_time = max(0.0, time_min - effect.extra_delay_min)
        base = self.delegate.observe(
            field,
            time_min=delegated_time,
            x_east_nm=x_east_nm,
            y_north_nm=y_north_nm,
        )
        return replace(
            base,
            observation_time_min=time_min,
            estimated_current_velocity_nm_per_min=(
                base.estimated_current_velocity_nm_per_min[0] + effect.current_east_bias_nm_per_min,
                base.estimated_current_velocity_nm_per_min[1] + effect.current_north_bias_nm_per_min,
            ),
            estimated_wind_velocity_nm_per_min=(
                base.estimated_wind_velocity_nm_per_min[0] + effect.wind_east_bias_nm_per_min,
                base.estimated_wind_velocity_nm_per_min[1] + effect.wind_north_bias_nm_per_min,
            ),
            provider_id=self.provider_id,
        )


def degraded_maneuver_limits(base: ManeuverLimits, effect: ActuatorFaultEffect) -> ManeuverLimits:
    effect = effect.validated()
    return ManeuverLimits(
        max_acceleration=base.max_acceleration * effect.propulsion_authority_factor,
        max_deceleration=base.max_deceleration * effect.command_authority_factor,
        max_turn_rate_deg_per_min=base.max_turn_rate_deg_per_min * effect.turn_authority_factor,
    )


def _sha256_payload(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
