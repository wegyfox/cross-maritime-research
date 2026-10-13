"""C16 plant-fidelity boundary for controller-independent vessel dynamics.

The plant consumes a normalized RequestedManeuver and physical state.  It never
reads ObservedWorld and never decides what maneuver should be requested.  This
keeps controller, actuator/plant realization and evaluation evidence separable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from math import atan2, cos, degrees, exp, hypot, isfinite, pi, sin
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from vessel_crossing.kinematics import knots_to_nm_per_minute
from vessel_crossing.nextgen.dynamics import (
    ManeuverCommand,
    ManeuverLimits,
    Vessel2DState,
    advance_toward_command,
    normalize_heading,
    signed_heading_delta,
)

from .behavior import RequestedManeuver
from .scope import TestbedFeature

_EPS = 1e-12


class PlantModelKind(str, Enum):
    A162_KINEMATIC = "A162_KINEMATIC"
    FIRST_ORDER_RESPONSE = "FIRST_ORDER_RESPONSE"
    TABULATED_STEERAGE_RESPONSE = "TABULATED_STEERAGE_RESPONSE"
    LOW_ORDER_ACTUATED = "LOW_ORDER_ACTUATED"
    LOW_ORDER_SWAY_YAW = "LOW_ORDER_SWAY_YAW"
    EXTERNAL_COSIM = "EXTERNAL_COSIM"


class PlantStepStatus(str, Enum):
    OK = "OK"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    PLANT_ERROR = "PLANT_ERROR"


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("canonical plant payload cannot contain NaN/inf")
        return value
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    raise TypeError(f"unsupported canonical plant payload type: {type(value).__name__}")


def _sha(payload: Any) -> str:
    raw = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class PlantSpec:
    kind: PlantModelKind
    model_id: str
    model_version: str
    adapter_id: str
    parameters: tuple[tuple[str, float | str | bool], ...] = ()
    _identity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_version or not self.adapter_id:
            raise ValueError("model_id, model_version and adapter_id must be non-empty")
        if tuple(sorted(self.parameters)) != self.parameters:
            raise ValueError("plant parameters must be sorted for canonical identity")
        _canonical(dict(self.parameters))
        object.__setattr__(
            self,
            "_identity_sha256",
            _sha({
                "kind": self.kind.value,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "adapter_id": self.adapter_id,
                "parameters": [list(v) for v in self.parameters],
            }),
        )

    def identity_sha256(self) -> str:
        """Return the immutable spec identity without re-canonicalizing static data."""
        return self._identity_sha256


@dataclass(frozen=True, slots=True)
class PlantActuatorState:
    """Actual low-level actuator state exposed by a richer plant.

    Rudder sign is adapter-defined and must be documented by the concrete
    external transport. Propeller rate is signed so an astern-capable backend
    can preserve actual shaft direction without abusing ``Vessel2DState``.
    """

    rudder_angle_deg: float
    propeller_rate_rps: float

    def __post_init__(self) -> None:
        if not isfinite(self.rudder_angle_deg):
            raise ValueError("rudder_angle_deg must be finite")
        if not isfinite(self.propeller_rate_rps):
            raise ValueError("propeller_rate_rps must be finite")

    def canonical_payload(self) -> Mapping[str, float]:
        return MappingProxyType({
            "rudder_angle_deg": self.rudder_angle_deg,
            "propeller_rate_rps": self.propeller_rate_rps,
        })


@dataclass(frozen=True, slots=True)
class PlantActuatorTransientState:
    """Internal actuator memory needed for finite reversal/response dynamics.

    ``last_nonzero_propeller_sign`` remembers the previously established shaft
    direction after the actual propeller rate reaches zero.  A reversal order
    therefore cannot cross instantaneously through zero and immediately build
    opposite thrust. ``pending_reversal_sign`` is -1/0/+1 and binds a currently
    active reversal hold to the requested direction.
    """

    last_nonzero_propeller_sign: int = 0
    pending_reversal_sign: int = 0
    reversal_hold_remaining_min: float = 0.0

    def __post_init__(self) -> None:
        if self.last_nonzero_propeller_sign not in (-1, 0, 1):
            raise ValueError("last_nonzero_propeller_sign must be -1, 0 or 1")
        if self.pending_reversal_sign not in (-1, 0, 1):
            raise ValueError("pending_reversal_sign must be -1, 0 or 1")
        if not isfinite(self.reversal_hold_remaining_min) or self.reversal_hold_remaining_min < 0.0:
            raise ValueError("reversal_hold_remaining_min must be finite and non-negative")
        if self.pending_reversal_sign == 0 and self.reversal_hold_remaining_min > _EPS:
            raise ValueError("non-zero reversal hold requires pending_reversal_sign")

    def canonical_payload(self) -> Mapping[str, float | int]:
        return MappingProxyType({
            "last_nonzero_propeller_sign": self.last_nonzero_propeller_sign,
            "pending_reversal_sign": self.pending_reversal_sign,
            "reversal_hold_remaining_min": self.reversal_hold_remaining_min,
        })


@dataclass(frozen=True, slots=True)
class PlantActuatorCommand:
    """Controller-independent actuator command for external/richer plants.

    This is deliberately separate from :class:`RequestedManeuver`: the latter
    is a navigation-level heading/speed request, while MMG-like plants consume
    physical rudder/propeller inputs after an actuator/autopilot layer.
    """

    issued_at_min: float
    target_rudder_angle_deg: float
    target_propeller_rate_rps: float
    source: str

    def __post_init__(self) -> None:
        if not isfinite(self.issued_at_min) or self.issued_at_min < 0.0:
            raise ValueError("issued_at_min must be finite and non-negative")
        if not isfinite(self.target_rudder_angle_deg):
            raise ValueError("target_rudder_angle_deg must be finite")
        if not isfinite(self.target_propeller_rate_rps):
            raise ValueError("target_propeller_rate_rps must be finite")
        if not self.source:
            raise ValueError("source must be non-empty")

    def canonical_payload(self) -> Mapping[str, float | str]:
        return MappingProxyType({
            "issued_at_min": self.issued_at_min,
            "target_rudder_angle_deg": self.target_rudder_angle_deg,
            "target_propeller_rate_rps": self.target_propeller_rate_rps,
            "source": self.source,
        })


@dataclass(frozen=True, slots=True)
class PlantDynamicState:
    vessel: Vessel2DState
    yaw_rate_deg_per_min: float = 0.0
    surge_acceleration_nm_per_min2: float = 0.0
    opaque_state: tuple[tuple[str, float | str | bool], ...] = ()
    # Richer 3-DOF state is optional so all pre-MR identities remain unchanged.
    # MMG convention used here: +u is forward; +v is to port.
    surge_velocity_nm_per_min: float | None = None
    sway_velocity_nm_per_min: float | None = None
    actuator_state: PlantActuatorState | None = None
    actuator_transient_state: PlantActuatorTransientState | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.yaw_rate_deg_per_min):
            raise ValueError("yaw_rate_deg_per_min must be finite")
        if not isfinite(self.surge_acceleration_nm_per_min2):
            raise ValueError("surge_acceleration_nm_per_min2 must be finite")
        if (self.surge_velocity_nm_per_min is None) != (self.sway_velocity_nm_per_min is None):
            raise ValueError("surge and sway velocity must be supplied together")
        if self.surge_velocity_nm_per_min is not None:
            if not isfinite(self.surge_velocity_nm_per_min) or not isfinite(self.sway_velocity_nm_per_min):
                raise ValueError("surge/sway velocity must be finite")
            projected_speed = hypot(self.surge_velocity_nm_per_min, self.sway_velocity_nm_per_min)
            if abs(projected_speed - self.vessel.speed) > 1e-9:
                raise ValueError("Vessel2DState.speed must equal the magnitude of supplied surge/sway velocity")
        if tuple(sorted(self.opaque_state)) != self.opaque_state:
            raise ValueError("opaque_state must be sorted")
        _canonical(dict(self.opaque_state))

    @property
    def has_body_velocity(self) -> bool:
        return self.surge_velocity_nm_per_min is not None

    @property
    def earth_velocity_nm_per_min(self) -> tuple[float, float]:
        """Return east/north velocity while preserving MMG body-frame drift.

        For rich state, +sway is port.  For legacy state, this is exactly the
        inherited ``Vessel2DState.velocity`` projection.
        """
        if not self.has_body_velocity:
            return self.vessel.velocity
        theta = self.vessel.heading_deg * pi / 180.0
        u = self.surge_velocity_nm_per_min
        v = self.sway_velocity_nm_per_min
        return (u * sin(theta) - v * cos(theta), u * cos(theta) + v * sin(theta))

    @property
    def course_over_ground_deg(self) -> float:
        east, north = self.earth_velocity_nm_per_min
        if hypot(east, north) <= _EPS:
            return self.vessel.heading_deg
        return normalize_heading(degrees(atan2(east, north)))

    def identity_sha256(self) -> str:
        payload: dict[str, Any] = {
            "vessel_id": self.vessel.vessel_id,
            "position_nm": list(self.vessel.position),
            "heading_deg": self.vessel.heading_deg,
            "speed_nm_per_min": self.vessel.speed,
            "yaw_rate_deg_per_min": self.yaw_rate_deg_per_min,
            "surge_acceleration_nm_per_min2": self.surge_acceleration_nm_per_min2,
            "opaque_state": [list(v) for v in self.opaque_state],
        }
        # Deliberately omit absent rich-state fields: this preserves every
        # historical state identity/hash from before MANEUVERING_REALISM.
        if self.has_body_velocity:
            payload["surge_velocity_nm_per_min"] = self.surge_velocity_nm_per_min
            payload["sway_velocity_nm_per_min"] = self.sway_velocity_nm_per_min
        if self.actuator_state is not None:
            payload["actuator_state"] = dict(self.actuator_state.canonical_payload())
        if self.actuator_transient_state is not None:
            payload["actuator_transient_state"] = dict(self.actuator_transient_state.canonical_payload())
        return _sha(payload)


@dataclass(frozen=True, slots=True)
class PlantStepRequest:
    time_min: float
    duration_min: float
    state: PlantDynamicState
    maneuver: RequestedManeuver
    limits: ManeuverLimits
    physical_environment: Mapping[str, Any] | None = None
    component_id: TestbedFeature = TestbedFeature.C16_PLANT_FIDELITY
    actuator_command: PlantActuatorCommand | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.time_min) or self.time_min < 0:
            raise ValueError("time_min must be finite and non-negative")
        if not isfinite(self.duration_min) or self.duration_min < 0:
            raise ValueError("duration_min must be finite and non-negative")
        if self.maneuver.requested_at_min > self.time_min + _EPS:
            raise ValueError("maneuver cannot be requested in the future")
        if self.actuator_command is not None and self.actuator_command.issued_at_min > self.time_min + _EPS:
            raise ValueError("actuator command cannot be issued in the future")
        if self.physical_environment is not None:
            object.__setattr__(self, "physical_environment", MappingProxyType(_canonical(self.physical_environment)))

    def canonical_payload(self) -> Mapping[str, Any]:
        env = None if self.physical_environment is None else dict(self.physical_environment)
        payload: dict[str, Any] = {
            "time_min": self.time_min,
            "duration_min": self.duration_min,
            "state_sha256": self.state.identity_sha256(),
            "maneuver": {
                "requested_at_min": self.maneuver.requested_at_min,
                "target_heading_deg": self.maneuver.target_heading_deg,
                "target_speed_knots": self.maneuver.target_speed_knots,
                "source": self.maneuver.source,
            },
            "limits": {
                "max_acceleration": self.limits.max_acceleration,
                "max_deceleration": self.limits.max_deceleration,
                "max_turn_rate_deg_per_min": self.limits.max_turn_rate_deg_per_min,
            },
            "physical_environment": env,
        }
        # Preserve all historical request hashes when no low-level actuation is
        # present.  External MMG requests bind the actuator command explicitly.
        if self.actuator_command is not None:
            payload["actuator_command"] = dict(self.actuator_command.canonical_payload())
        return MappingProxyType(_canonical(payload))

    def identity_sha256(self) -> str:
        """Hash an already-canonical request payload without a second recursive walk."""
        raw = json.dumps(
            dict(self.canonical_payload()), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class PlantStepResult:
    status: PlantStepStatus
    spec_sha256: str
    request_sha256: str
    state: PlantDynamicState | None
    distance_nm: float | None
    integration_steps: int | None
    diagnostics: tuple[tuple[str, str], ...] = ()
    component_id: TestbedFeature = TestbedFeature.C16_PLANT_FIDELITY

    def __post_init__(self) -> None:
        if self.status is PlantStepStatus.OK and self.state is None:
            raise ValueError("OK plant result requires state")
        if self.status is not PlantStepStatus.OK and self.state is not None:
            raise ValueError("non-OK plant result cannot expose a trusted state")
        if self.distance_nm is not None and (not isfinite(self.distance_nm) or self.distance_nm < 0):
            raise ValueError("distance_nm must be finite and non-negative")
        if self.integration_steps is not None and self.integration_steps < 0:
            raise ValueError("integration_steps must be non-negative")
        if tuple(sorted(self.diagnostics)) != self.diagnostics:
            raise ValueError("diagnostics must be sorted")

    @property
    def ok(self) -> bool:
        return self.status is PlantStepStatus.OK

    def identity_sha256(self) -> str:
        return _sha({
            "status": self.status.value,
            "spec_sha256": self.spec_sha256,
            "request_sha256": self.request_sha256,
            "state_sha256": None if self.state is None else self.state.identity_sha256(),
            "distance_nm": self.distance_nm,
            "integration_steps": self.integration_steps,
            "diagnostics": [list(v) for v in self.diagnostics],
        })


class PlantAdapter(Protocol):
    component_id: TestbedFeature
    spec: PlantSpec

    def step(self, request: PlantStepRequest) -> PlantStepResult:
        ...


class A162KinematicPlantAdapter:
    """Exact adapter over the inherited A162 bounded-command integrator."""
    component_id = TestbedFeature.C16_PLANT_FIDELITY

    def __init__(self) -> None:
        self.spec = PlantSpec(
            kind=PlantModelKind.A162_KINEMATIC,
            model_id="a162-bounded-command-kinematics",
            model_version="1.0.0a162",
            adapter_id="cross-c16-a162-kinematic-adapter-v1",
        )

    def step(self, request: PlantStepRequest) -> PlantStepResult:
        command = ManeuverCommand.from_knots(
            target_heading_deg=request.maneuver.target_heading_deg,
            target_speed_knots=request.maneuver.target_speed_knots,
        )
        advanced = advance_toward_command(request.state.vessel, command, request.duration_min, request.limits)
        state = PlantDynamicState(
            vessel=advanced.state,
            yaw_rate_deg_per_min=0.0,
            surge_acceleration_nm_per_min2=0.0,
        )
        return PlantStepResult(
            status=PlantStepStatus.OK,
            spec_sha256=self.spec.identity_sha256(),
            request_sha256=request.identity_sha256(),
            state=state,
            distance_nm=advanced.distance,
            integration_steps=advanced.segments,
        )


@dataclass(frozen=True, slots=True)
class FirstOrderResponseParameters:
    yaw_time_constant_min: float = 0.5
    speed_time_constant_min: float = 1.0
    heading_gain_per_min: float = 1.0
    max_step_min: float = 0.02

    def __post_init__(self) -> None:
        for name in ("yaw_time_constant_min", "speed_time_constant_min", "heading_gain_per_min", "max_step_min"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class SteerageAuthorityPoint:
    speed_knots: float
    yaw_authority_fraction: float

    def __post_init__(self) -> None:
        if not isfinite(self.speed_knots) or self.speed_knots < 0.0:
            raise ValueError("speed_knots must be finite and non-negative")
        if (
            not isfinite(self.yaw_authority_fraction)
            or self.yaw_authority_fraction < 0.0
            or self.yaw_authority_fraction > 1.0
        ):
            raise ValueError("yaw_authority_fraction must be finite in [0, 1]")


@dataclass(frozen=True, slots=True)
class TabulatedSteerageResponseParameters:
    """Explicit low-order manoeuvring profile for MR-1/MR-2 sensitivity work.

    The steerage curve is deliberately supplied by the caller.  CROSS does not bake in
    a universal minimum-steerage speed because that threshold is vessel/configuration
    specific and should ultimately come from manoeuvring data or a validated plant.
    """

    steerage_curve: tuple[SteerageAuthorityPoint, ...]
    steering_response_delay_min: float
    yaw_time_constant_min: float
    speed_time_constant_min: float
    heading_gain_per_min: float = 1.0
    max_step_min: float = 0.02

    def __post_init__(self) -> None:
        if len(self.steerage_curve) < 2:
            raise ValueError("steerage_curve must contain at least two points")
        speeds = tuple(point.speed_knots for point in self.steerage_curve)
        if any(b <= a for a, b in zip(speeds, speeds[1:])):
            raise ValueError("steerage_curve speeds must be strictly increasing")
        if self.steerage_curve[-1].yaw_authority_fraction <= 0.0:
            raise ValueError("steerage_curve must expose positive authority at its high-speed end")
        if not isfinite(self.steering_response_delay_min) or self.steering_response_delay_min < 0.0:
            raise ValueError("steering_response_delay_min must be finite and non-negative")
        for name in ("yaw_time_constant_min", "speed_time_constant_min", "heading_gain_per_min", "max_step_min"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def authority_at(self, speed_knots: float) -> float:
        if not isfinite(speed_knots) or speed_knots < 0.0:
            raise ValueError("speed_knots must be finite and non-negative")
        points = self.steerage_curve
        if speed_knots <= points[0].speed_knots:
            return points[0].yaw_authority_fraction
        if speed_knots >= points[-1].speed_knots:
            return points[-1].yaw_authority_fraction
        for first, second in zip(points, points[1:]):
            if speed_knots <= second.speed_knots:
                span = second.speed_knots - first.speed_knots
                weight = (speed_knots - first.speed_knots) / span
                return first.yaw_authority_fraction + weight * (
                    second.yaw_authority_fraction - first.yaw_authority_fraction
                )
        raise AssertionError("steerage interpolation fell through")


@dataclass(frozen=True, slots=True)
class PropulsionMapPoint:
    """One signed shaft-rate / steady surge-speed operating point.

    Positive values mean ahead propulsion/surge, negative values astern.  The
    map is deliberately vessel/profile supplied: CROSS does not infer an
    astern performance curve from ahead data.
    """

    propeller_rate_rps: float
    equilibrium_surge_knots: float

    def __post_init__(self) -> None:
        if not isfinite(self.propeller_rate_rps):
            raise ValueError("propeller_rate_rps must be finite")
        if not isfinite(self.equilibrium_surge_knots):
            raise ValueError("equilibrium_surge_knots must be finite")


@dataclass(frozen=True, slots=True)
class SwayYawCouplingParameters:
    """Optional low-order MR-5 sideslip/initial-kick sensitivity coefficients.

    Positive sway follows the C16/MMG convention: port.  Therefore a positive
    ``initial_kick_gain_knots_per_rudder_fraction`` makes a positive/starboard
    rudder order initially translate the reference point to port, opposite the
    developing starboard turn.  Signs are explicit profile inputs because the
    detailed response depends on hull/propeller/rudder geometry and loading.
    """

    sway_time_constant_min: float
    initial_kick_gain_knots_per_rudder_fraction: float
    initial_kick_decay_min: float
    yaw_sideslip_gain_knots_per_deg_per_min: float
    sway_to_yaw_gain_deg_per_min2_per_knot: float = 0.0
    max_abs_sway_knots: float = 2.0

    def __post_init__(self) -> None:
        for name in ("sway_time_constant_min", "initial_kick_decay_min", "max_abs_sway_knots"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "initial_kick_gain_knots_per_rudder_fraction",
            "yaw_sideslip_gain_knots_per_deg_per_min",
            "sway_to_yaw_gain_deg_per_min2_per_knot",
        ):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class LowOrderActuatedPlantParameters:
    """MR-1/MR-2/MR-4 low-order vessel/actuator sensitivity profile.

    This model is intentionally not labelled MMG.  It adds finite rudder and
    propeller actuator motion, propulsion/reversal delay and signed surge while
    retaining a low-order yaw response.  All values are explicit profile data;
    no conventional-ship constant is silently baked into the plant.
    """

    steerage_curve: tuple[SteerageAuthorityPoint, ...]
    propulsion_map: tuple[PropulsionMapPoint, ...]
    max_rudder_angle_deg: float
    rudder_rate_deg_per_min: float
    rudder_response_delay_min: float
    propeller_rate_slew_rps_per_min: float
    propulsion_response_delay_min: float
    reversal_delay_min: float
    yaw_time_constant_min: float
    surge_time_constant_min: float
    max_astern_acceleration_knots_per_min: float
    sway_yaw: SwayYawCouplingParameters | None = None
    autopilot_full_rudder_error_deg: float = 30.0
    max_step_min: float = 0.01

    def __post_init__(self) -> None:
        if len(self.steerage_curve) < 2:
            raise ValueError("steerage_curve must contain at least two points")
        steerage_speeds = tuple(point.speed_knots for point in self.steerage_curve)
        if any(b <= a for a, b in zip(steerage_speeds, steerage_speeds[1:])):
            raise ValueError("steerage_curve speeds must be strictly increasing")
        if len(self.propulsion_map) < 3:
            raise ValueError("propulsion_map must contain at least three points including astern/stop/ahead")
        prop_rates = tuple(point.propeller_rate_rps for point in self.propulsion_map)
        surge_speeds = tuple(point.equilibrium_surge_knots for point in self.propulsion_map)
        if any(b <= a for a, b in zip(prop_rates, prop_rates[1:])):
            raise ValueError("propulsion_map propeller rates must be strictly increasing")
        if any(b <= a for a, b in zip(surge_speeds, surge_speeds[1:])):
            raise ValueError("propulsion_map surge speeds must be strictly increasing")
        if not any(abs(p.propeller_rate_rps) <= _EPS and abs(p.equilibrium_surge_knots) <= _EPS
                   for p in self.propulsion_map):
            raise ValueError("propulsion_map must contain an explicit zero/stop point")
        for name in (
            "max_rudder_angle_deg", "rudder_rate_deg_per_min", "propeller_rate_slew_rps_per_min",
            "yaw_time_constant_min", "surge_time_constant_min", "autopilot_full_rudder_error_deg",
            "max_step_min",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("rudder_response_delay_min", "propulsion_response_delay_min", "reversal_delay_min"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not isfinite(self.max_astern_acceleration_knots_per_min)
            or self.max_astern_acceleration_knots_per_min <= 0.0
        ):
            raise ValueError("max_astern_acceleration_knots_per_min must be finite and positive")

    def steerage_authority_at(self, speed_knots: float) -> float:
        if not isfinite(speed_knots) or speed_knots < 0.0:
            raise ValueError("speed_knots must be finite and non-negative")
        points = self.steerage_curve
        if speed_knots <= points[0].speed_knots:
            return points[0].yaw_authority_fraction
        if speed_knots >= points[-1].speed_knots:
            return points[-1].yaw_authority_fraction
        for first, second in zip(points, points[1:]):
            if speed_knots <= second.speed_knots:
                weight = (speed_knots - first.speed_knots) / (second.speed_knots - first.speed_knots)
                return first.yaw_authority_fraction + weight * (
                    second.yaw_authority_fraction - first.yaw_authority_fraction
                )
        raise AssertionError("steerage interpolation fell through")

    def equilibrium_surge_knots_at(self, propeller_rate_rps: float) -> float:
        if not isfinite(propeller_rate_rps):
            raise ValueError("propeller_rate_rps must be finite")
        points = self.propulsion_map
        if propeller_rate_rps <= points[0].propeller_rate_rps:
            return points[0].equilibrium_surge_knots
        if propeller_rate_rps >= points[-1].propeller_rate_rps:
            return points[-1].equilibrium_surge_knots
        for first, second in zip(points, points[1:]):
            if propeller_rate_rps <= second.propeller_rate_rps:
                weight = (propeller_rate_rps - first.propeller_rate_rps) / (
                    second.propeller_rate_rps - first.propeller_rate_rps
                )
                return first.equilibrium_surge_knots + weight * (
                    second.equilibrium_surge_knots - first.equilibrium_surge_knots
                )
        raise AssertionError("propulsion interpolation fell through")

    def propeller_rate_for_nonnegative_speed_knots(self, speed_knots: float) -> float:
        """Invert the ahead branch for the navigation-level speed autopilot."""
        if not isfinite(speed_knots) or speed_knots < 0.0:
            raise ValueError("speed_knots must be finite and non-negative")
        points = tuple(p for p in self.propulsion_map if p.equilibrium_surge_knots >= -_EPS)
        if not points:
            raise ValueError("propulsion_map contains no non-negative ahead branch")
        if speed_knots <= points[0].equilibrium_surge_knots:
            return points[0].propeller_rate_rps
        if speed_knots >= points[-1].equilibrium_surge_knots:
            return points[-1].propeller_rate_rps
        for first, second in zip(points, points[1:]):
            if speed_knots <= second.equilibrium_surge_knots:
                weight = (speed_knots - first.equilibrium_surge_knots) / (
                    second.equilibrium_surge_knots - first.equilibrium_surge_knots
                )
                return first.propeller_rate_rps + weight * (
                    second.propeller_rate_rps - first.propeller_rate_rps
                )
        raise AssertionError("propulsion inverse interpolation fell through")


def _sign(value: float) -> int:
    if value > _EPS:
        return 1
    if value < -_EPS:
        return -1
    return 0


def _move_toward(current: float, target: float, max_rate_per_min: float, dt_min: float) -> float:
    delta = target - current
    bound = max_rate_per_min * dt_min
    if abs(delta) <= bound:
        return target
    return current + (bound if delta > 0.0 else -bound)


class LowOrderActuatedPlantAdapter:
    """Finite actuator + signed-surge sensitivity plant for MR-1/MR-2/MR-4.

    Navigation requests are converted by a deliberately simple declared
    autopilot into rudder/ahead-propeller targets.  Tests or external bridges may
    instead provide ``PlantActuatorCommand`` directly, including negative shaft
    rate for an astern order.  Actual rudder/shaft states move with finite slew
    rates and explicit delays; shaft reversal must pass through zero and wait the
    configured reversal delay before opposite thrust can build.

    Sway/pivot/kick and hydrodynamic MMG forces remain outside this adapter and
    are explicitly left for MR-5 / external co-simulation.
    """

    component_id = TestbedFeature.C16_PLANT_FIDELITY

    def __init__(self, parameters: LowOrderActuatedPlantParameters) -> None:
        if not isinstance(parameters, LowOrderActuatedPlantParameters):
            raise TypeError("parameters must be LowOrderActuatedPlantParameters")
        self.parameters = parameters
        steerage_json = json.dumps(
            [[p.speed_knots, p.yaw_authority_fraction] for p in parameters.steerage_curve],
            separators=(",", ":"),
        )
        propulsion_json = json.dumps(
            [[p.propeller_rate_rps, p.equilibrium_surge_knots] for p in parameters.propulsion_map],
            separators=(",", ":"),
        )
        parameter_payload = {
                "autopilot_full_rudder_error_deg": parameters.autopilot_full_rudder_error_deg,
                "max_astern_acceleration_knots_per_min": parameters.max_astern_acceleration_knots_per_min,
                "max_rudder_angle_deg": parameters.max_rudder_angle_deg,
                "max_step_min": parameters.max_step_min,
                "propeller_rate_slew_rps_per_min": parameters.propeller_rate_slew_rps_per_min,
                "propulsion_map_json": propulsion_json,
                "propulsion_response_delay_min": parameters.propulsion_response_delay_min,
                "reversal_delay_min": parameters.reversal_delay_min,
                "rudder_rate_deg_per_min": parameters.rudder_rate_deg_per_min,
                "rudder_response_delay_min": parameters.rudder_response_delay_min,
                "steerage_curve_json": steerage_json,
                "surge_time_constant_min": parameters.surge_time_constant_min,
                "yaw_time_constant_min": parameters.yaw_time_constant_min,
        }
        if parameters.sway_yaw is not None:
            sy = parameters.sway_yaw
            parameter_payload.update({
                "sway_initial_kick_decay_min": sy.initial_kick_decay_min,
                "sway_initial_kick_gain_knots_per_rudder_fraction": sy.initial_kick_gain_knots_per_rudder_fraction,
                "sway_max_abs_knots": sy.max_abs_sway_knots,
                "sway_time_constant_min": sy.sway_time_constant_min,
                "sway_to_yaw_gain_deg_per_min2_per_knot": sy.sway_to_yaw_gain_deg_per_min2_per_knot,
                "yaw_sideslip_gain_knots_per_deg_per_min": sy.yaw_sideslip_gain_knots_per_deg_per_min,
            })
        self.spec = PlantSpec(
            kind=(PlantModelKind.LOW_ORDER_SWAY_YAW if parameters.sway_yaw is not None else PlantModelKind.LOW_ORDER_ACTUATED),
            model_id=("cross-low-order-sway-yaw-plant" if parameters.sway_yaw is not None else "cross-low-order-actuated-plant"),
            model_version="1",
            adapter_id=("cross-c16-low-order-sway-yaw-v1" if parameters.sway_yaw is not None else "cross-c16-low-order-actuated-v1"),
            parameters=tuple(sorted(parameter_payload.items())),
        )

    def _autopilot_command(self, request: PlantStepRequest) -> PlantActuatorCommand:
        p = self.parameters
        error = signed_heading_delta(request.state.vessel.heading_deg, request.maneuver.target_heading_deg)
        rudder_fraction = max(-1.0, min(1.0, error / p.autopilot_full_rudder_error_deg))
        propeller_rate = p.propeller_rate_for_nonnegative_speed_knots(request.maneuver.target_speed_knots)
        return PlantActuatorCommand(
            issued_at_min=request.maneuver.requested_at_min,
            target_rudder_angle_deg=rudder_fraction * p.max_rudder_angle_deg,
            target_propeller_rate_rps=propeller_rate,
            source=f"autopilot:{request.maneuver.source}",
        )

    def step(self, request: PlantStepRequest) -> PlantStepResult:
        p = self.parameters
        command = request.actuator_command or self._autopilot_command(request)
        if abs(command.target_rudder_angle_deg) > p.max_rudder_angle_deg + _EPS:
            return PlantStepResult(
                status=PlantStepStatus.INVALID_RESPONSE,
                spec_sha256=self.spec.identity_sha256(),
                request_sha256=request.identity_sha256(),
                state=None,
                distance_nm=None,
                integration_steps=None,
                diagnostics=(("error", "target_rudder_angle_exceeds_profile_limit"),),
            )

        vessel = request.state.vessel
        x, y = vessel.position
        heading = vessel.heading_deg
        yaw_rate = request.state.yaw_rate_deg_per_min
        if request.state.has_body_velocity:
            surge = request.state.surge_velocity_nm_per_min
            sway = request.state.sway_velocity_nm_per_min
            assert surge is not None and sway is not None
        else:
            surge = vessel.speed
            sway = 0.0

        actuator = request.state.actuator_state or PlantActuatorState(0.0, 0.0)
        transient = request.state.actuator_transient_state or PlantActuatorTransientState(
            last_nonzero_propeller_sign=_sign(actuator.propeller_rate_rps),
        )
        rudder = actuator.rudder_angle_deg
        propeller = actuator.propeller_rate_rps
        last_sign = transient.last_nonzero_propeller_sign
        pending_sign = transient.pending_reversal_sign
        reversal_hold = transient.reversal_hold_remaining_min

        remaining = request.duration_min
        elapsed = 0.0
        distance = 0.0
        steps = 0
        last_accel = request.state.surge_acceleration_nm_per_min2
        astern_accel = knots_to_nm_per_minute(p.max_astern_acceleration_knots_per_min)

        while remaining > _EPS:
            dt = min(p.max_step_min, remaining)
            now_mid = request.time_min + elapsed + 0.5 * dt
            command_age = now_mid - command.issued_at_min

            if command_age + _EPS >= p.rudder_response_delay_min:
                rudder = _move_toward(rudder, command.target_rudder_angle_deg, p.rudder_rate_deg_per_min, dt)

            target_propeller = command.target_propeller_rate_rps
            target_sign = _sign(target_propeller)
            if command_age + _EPS < p.propulsion_response_delay_min:
                effective_propeller_target = propeller
            else:
                current_sign = _sign(propeller)
                established_sign = current_sign or last_sign
                reversing = target_sign != 0 and established_sign != 0 and target_sign != established_sign
                if reversing and abs(propeller) > _EPS:
                    effective_propeller_target = 0.0
                elif reversing:
                    if pending_sign != target_sign:
                        pending_sign = target_sign
                        reversal_hold = p.reversal_delay_min
                    if reversal_hold > _EPS:
                        effective_propeller_target = 0.0
                        reversal_hold = max(0.0, reversal_hold - dt)
                    else:
                        effective_propeller_target = target_propeller
                else:
                    pending_sign = 0
                    reversal_hold = 0.0
                    effective_propeller_target = target_propeller

                propeller = _move_toward(
                    propeller,
                    effective_propeller_target,
                    p.propeller_rate_slew_rps_per_min,
                    dt,
                )
                prop_sign = _sign(propeller)
                if prop_sign != 0:
                    last_sign = prop_sign
                    if pending_sign == prop_sign:
                        pending_sign = 0
                        reversal_hold = 0.0

            target_surge_knots = p.equilibrium_surge_knots_at(propeller)
            target_surge = knots_to_nm_per_minute(abs(target_surge_knots)) * _sign(target_surge_knots)
            raw_accel = (target_surge - surge) / p.surge_time_constant_min
            if raw_accel < 0.0:
                if surge <= 0.0 and target_surge < surge:
                    accel = max(-astern_accel, raw_accel)
                else:
                    accel = max(-request.limits.max_deceleration, raw_accel)
            else:
                if surge < 0.0 and target_surge > surge:
                    accel = min(request.limits.max_deceleration, raw_accel)
                else:
                    accel = min(request.limits.max_acceleration, raw_accel)

            mid_surge = surge + 0.5 * accel * dt
            new_surge = surge + accel * dt
            if (target_surge - surge) * (target_surge - new_surge) < 0.0:
                new_surge = target_surge

            authority = p.steerage_authority_at(abs(mid_surge) * 60.0)
            rudder_fraction = max(-1.0, min(1.0, rudder / p.max_rudder_angle_deg))
            desired_yaw = request.limits.max_turn_rate_deg_per_min * authority * rudder_fraction

            sy = p.sway_yaw
            if sy is None:
                sway_accel = 0.0
                mid_sway = sway
            else:
                steering_age = max(0.0, command_age - p.rudder_response_delay_min)
                kick_knots = (
                    sy.initial_kick_gain_knots_per_rudder_fraction
                    * rudder_fraction
                    * authority
                    * exp(-steering_age / sy.initial_kick_decay_min)
                )
                steady_sideslip_knots = sy.yaw_sideslip_gain_knots_per_deg_per_min * yaw_rate
                desired_sway_knots = max(
                    -sy.max_abs_sway_knots,
                    min(sy.max_abs_sway_knots, kick_knots + steady_sideslip_knots),
                )
                desired_sway = knots_to_nm_per_minute(abs(desired_sway_knots)) * _sign(desired_sway_knots)
                sway_accel = (desired_sway - sway) / sy.sway_time_constant_min
                mid_sway = sway + 0.5 * sway_accel * dt

            yaw_accel = (desired_yaw - yaw_rate) / p.yaw_time_constant_min
            if sy is not None:
                yaw_accel += sy.sway_to_yaw_gain_deg_per_min2_per_knot * (mid_sway * 60.0)
            mid_yaw = yaw_rate + 0.5 * yaw_accel * dt
            new_yaw = yaw_rate + yaw_accel * dt
            yaw_bound = request.limits.max_turn_rate_deg_per_min * authority
            new_yaw = max(-yaw_bound, min(yaw_bound, new_yaw))

            mid_heading = normalize_heading(heading + 0.5 * mid_yaw * dt)
            theta = mid_heading * pi / 180.0
            east = mid_surge * sin(theta) - mid_sway * cos(theta)
            north = mid_surge * cos(theta) + mid_sway * sin(theta)
            x += east * dt
            y += north * dt
            distance += hypot(east, north) * dt
            heading = normalize_heading(heading + mid_yaw * dt)
            surge = new_surge
            if sy is not None:
                sway = sway + sway_accel * dt
                max_sway = knots_to_nm_per_minute(sy.max_abs_sway_knots)
                sway = max(-max_sway, min(max_sway, sway))
            yaw_rate = new_yaw
            last_accel = accel
            remaining -= dt
            elapsed += dt
            steps += 1
            if steps > 1_000_000:
                raise RuntimeError("unexpected plant integration step count")

        out_speed = hypot(surge, sway)
        out = PlantDynamicState(
            vessel=Vessel2DState(vessel.vessel_id, (x, y), heading, out_speed),
            yaw_rate_deg_per_min=yaw_rate,
            surge_acceleration_nm_per_min2=last_accel,
            opaque_state=request.state.opaque_state,
            surge_velocity_nm_per_min=surge,
            sway_velocity_nm_per_min=sway,
            actuator_state=PlantActuatorState(rudder, propeller),
            actuator_transient_state=PlantActuatorTransientState(
                last_nonzero_propeller_sign=last_sign,
                pending_reversal_sign=pending_sign,
                reversal_hold_remaining_min=reversal_hold,
            ),
        )
        return PlantStepResult(
            status=PlantStepStatus.OK,
            spec_sha256=self.spec.identity_sha256(),
            request_sha256=request.identity_sha256(),
            state=out,
            distance_nm=distance,
            integration_steps=steps,
            diagnostics=tuple(sorted((
                ("actuation", "finite_rudder_and_propeller_slew_with_reversal_hold"),
                ("model_scope", "mr1_mr2_mr4_low_order_not_validated_mmg"),
                ("sway_model", "low_order_profile_coupling" if p.sway_yaw is not None else "not_generated_mr5_pending"),
            ))),
        )


class TabulatedSteerageResponsePlantAdapter:
    """Speed-dependent steerage + steering/yaw lag sensitivity plant.

    This closes only MR-1/MR-2.  It still has no sway/pivot/kick, astern propulsion or
    vessel-specific turning/stopping table solver and must not be labelled as MMG.
    """

    component_id = TestbedFeature.C16_PLANT_FIDELITY

    def __init__(self, parameters: TabulatedSteerageResponseParameters) -> None:
        if not isinstance(parameters, TabulatedSteerageResponseParameters):
            raise TypeError("parameters must be TabulatedSteerageResponseParameters")
        self.parameters = parameters
        curve_json = json.dumps(
            [[p.speed_knots, p.yaw_authority_fraction] for p in parameters.steerage_curve],
            separators=(",", ":"),
        )
        self.spec = PlantSpec(
            kind=PlantModelKind.TABULATED_STEERAGE_RESPONSE,
            model_id="cross-tabulated-steerage-response-plant",
            model_version="1",
            adapter_id="cross-c16-tabulated-steerage-response-v1",
            parameters=tuple(sorted({
                "heading_gain_per_min": parameters.heading_gain_per_min,
                "max_step_min": parameters.max_step_min,
                "speed_time_constant_min": parameters.speed_time_constant_min,
                "steerage_curve_json": curve_json,
                "steering_response_delay_min": parameters.steering_response_delay_min,
                "yaw_time_constant_min": parameters.yaw_time_constant_min,
            }.items())),
        )

    def step(self, request: PlantStepRequest) -> PlantStepResult:
        p = self.parameters
        vessel = request.state.vessel
        x, y = vessel.position
        heading = vessel.heading_deg
        speed = vessel.speed
        yaw_rate = request.state.yaw_rate_deg_per_min
        target_speed = knots_to_nm_per_minute(request.maneuver.target_speed_knots)
        remaining = request.duration_min
        elapsed = 0.0
        distance = 0.0
        steps = 0
        last_accel = request.state.surge_acceleration_nm_per_min2

        while remaining > _EPS:
            dt = min(p.max_step_min, remaining)
            raw_accel = (target_speed - speed) / p.speed_time_constant_min
            accel = max(-request.limits.max_deceleration, min(request.limits.max_acceleration, raw_accel))
            mid_speed = max(0.0, speed + 0.5 * accel * dt)
            new_speed = max(0.0, speed + accel * dt)
            if (target_speed - speed) * (target_speed - new_speed) < 0:
                new_speed = target_speed

            mid_speed_knots = mid_speed * 60.0
            authority = p.authority_at(mid_speed_knots)
            command_age_mid = request.time_min + elapsed + 0.5 * dt - request.maneuver.requested_at_min
            heading_error = signed_heading_delta(heading, request.maneuver.target_heading_deg)
            if command_age_mid + _EPS < p.steering_response_delay_min:
                desired_yaw = 0.0
            else:
                desired_yaw = p.heading_gain_per_min * heading_error
                desired_yaw = max(
                    -request.limits.max_turn_rate_deg_per_min * authority,
                    min(request.limits.max_turn_rate_deg_per_min * authority, desired_yaw),
                )

            yaw_accel = (desired_yaw - yaw_rate) / p.yaw_time_constant_min
            mid_yaw = yaw_rate + 0.5 * yaw_accel * dt
            new_yaw = yaw_rate + yaw_accel * dt
            yaw_bound = request.limits.max_turn_rate_deg_per_min * authority
            new_yaw = max(-yaw_bound, min(yaw_bound, new_yaw))

            mid_heading = normalize_heading(heading + 0.5 * mid_yaw * dt)
            theta = mid_heading * pi / 180.0
            ds = mid_speed * dt
            x += sin(theta) * ds
            y += cos(theta) * ds
            distance += ds
            heading = normalize_heading(heading + mid_yaw * dt)
            speed = new_speed
            yaw_rate = new_yaw
            last_accel = accel
            remaining -= dt
            elapsed += dt
            steps += 1
            if steps > 1_000_000:
                raise RuntimeError("unexpected plant integration step count")

        out = PlantDynamicState(
            vessel=Vessel2DState(vessel.vessel_id, (x, y), heading, speed),
            yaw_rate_deg_per_min=yaw_rate,
            surge_acceleration_nm_per_min2=last_accel,
            opaque_state=request.state.opaque_state,
        )
        return PlantStepResult(
            status=PlantStepStatus.OK,
            spec_sha256=self.spec.identity_sha256(),
            request_sha256=request.identity_sha256(),
            state=out,
            distance_nm=distance,
            integration_steps=steps,
            diagnostics=tuple(sorted((
                ("model_scope", "mr1_mr2_low_order_not_validated_mmg"),
                ("steerage_profile", "caller_supplied_piecewise_linear"),
            ))),
        )


@dataclass(slots=True)
class _FirstOrderPrefixCacheEntry:
    """Exact native-step prefixes for one frozen first-order request family."""

    template: PlantStepRequest
    states: list[PlantDynamicState]
    distance_prefix_nm: list[float]


class FirstOrderResponsePlantAdapter:
    """Low-order response-lag plant for sensitivity tests, not a validated MMG model.

    Heading error commands a bounded desired yaw rate. Actual yaw rate follows it
    with a first-order lag. Surge speed follows the requested speed with a
    first-order lag, additionally bounded by the inherited acceleration limits.
    Deterministic midpoint integration is used with an explicit maximum step.
    """
    component_id = TestbedFeature.C16_PLANT_FIDELITY

    def __init__(self, parameters: FirstOrderResponseParameters | None = None) -> None:
        self.parameters = parameters or FirstOrderResponseParameters()
        p = self.parameters
        self.spec = PlantSpec(
            kind=PlantModelKind.FIRST_ORDER_RESPONSE,
            model_id="cross-first-order-response-plant",
            model_version="1",
            adapter_id="cross-c16-first-order-response-v1",
            parameters=tuple(sorted({
                "heading_gain_per_min": p.heading_gain_per_min,
                "max_step_min": p.max_step_min,
                "speed_time_constant_min": p.speed_time_constant_min,
                "yaw_time_constant_min": p.yaw_time_constant_min,
            }.items())),
        )
        # PlantSpec is immutable; cache its canonical digest once instead of
        # re-serializing and hashing the same specification on every integration step.
        self._spec_sha256 = self.spec.identity_sha256()

        # Delayed behavior prediction can issue many frozen-start requests that differ
        # only by duration.  Keep this implementation cache strictly inside the
        # first-order plant so upper-layer trajectory semantics remain unchanged.
        # A family is promoted only on second sight; one-off continuation requests
        # therefore execute the original numerical core without partition overhead.
        self._prefix_seen: dict[tuple[Any, ...], None] = {}
        self._prefix_cache: dict[tuple[Any, ...], _FirstOrderPrefixCacheEntry] = {}
        self._duration_partition_cache: dict[float, tuple[int, float]] = {}

    _PREFIX_CACHE_MAX_FAMILIES = 64
    _PREFIX_SEEN_MAX_FAMILIES = 256
    _PREFIX_CACHE_MAX_FULL_STEPS = 1024
    _DURATION_PARTITION_CACHE_MAX = 4096

    @staticmethod
    def _prefix_family_key(request: PlantStepRequest) -> tuple[Any, ...]:
        """Return the exact frozen request identity except for duration.

        The immutable state object is retained as part of the key, so there is no
        object-id reuse hazard. Rich environment/actuator requests are never admitted.
        """
        return (
            request.time_min,
            request.state,
            request.maneuver,
            request.limits,
            request.component_id,
        )

    def _duration_partition(self, duration_min: float) -> tuple[int, float] | None:
        """Replay the original repeated-subtraction step partition exactly.

        No floor/divmod shortcut is used: the residual duration must be the same
        floating-point value that the uncached integrator would obtain.
        """
        cached = self._duration_partition_cache.get(duration_min)
        if cached is not None:
            return cached

        remaining = duration_min
        full_steps = 0
        max_step = self.parameters.max_step_min
        while remaining > _EPS:
            dt = min(max_step, remaining)
            if dt < max_step:
                break
            remaining -= dt
            full_steps += 1
            if full_steps > self._PREFIX_CACHE_MAX_FULL_STEPS:
                return None

        residual = remaining if remaining > _EPS else 0.0
        result = (full_steps, residual)
        if len(self._duration_partition_cache) >= self._DURATION_PARTITION_CACHE_MAX:
            self._duration_partition_cache.pop(next(iter(self._duration_partition_cache)))
        self._duration_partition_cache[duration_min] = result
        return result

    @staticmethod
    def _request_from_prefix(
        template: PlantStepRequest,
        *,
        state: PlantDynamicState,
        duration_min: float,
    ) -> PlantStepRequest:
        return PlantStepRequest(
            time_min=template.time_min,
            duration_min=duration_min,
            state=state,
            maneuver=template.maneuver,
            limits=template.limits,
            physical_environment=template.physical_environment,
            component_id=template.component_id,
            actuator_command=template.actuator_command,
        )

    def _integrate_dynamics_uncached(
        self, request: PlantStepRequest
    ) -> tuple[PlantDynamicState, float, int]:
        """Run the exact numerical integrator without provenance serialization.

        The public :meth:`step` method remains the provenance-bearing API.  Research
        trajectory prediction may call this private numerical core when it only needs
        the physical state/distance/step count; both paths execute this same code.
        """
        p = self.parameters
        vessel = request.state.vessel
        x, y = vessel.position
        heading = vessel.heading_deg
        speed = vessel.speed
        yaw_rate = request.state.yaw_rate_deg_per_min
        target_speed = knots_to_nm_per_minute(request.maneuver.target_speed_knots)
        remaining = request.duration_min
        distance = 0.0
        steps = 0
        last_accel = request.state.surge_acceleration_nm_per_min2

        while remaining > _EPS:
            dt = min(p.max_step_min, remaining)
            heading_error = signed_heading_delta(heading, request.maneuver.target_heading_deg)
            desired_yaw = max(-request.limits.max_turn_rate_deg_per_min,
                              min(request.limits.max_turn_rate_deg_per_min, p.heading_gain_per_min * heading_error))
            yaw_accel = (desired_yaw - yaw_rate) / p.yaw_time_constant_min
            mid_yaw = yaw_rate + 0.5 * yaw_accel * dt
            new_yaw = yaw_rate + yaw_accel * dt
            new_yaw = max(-request.limits.max_turn_rate_deg_per_min,
                          min(request.limits.max_turn_rate_deg_per_min, new_yaw))

            raw_accel = (target_speed - speed) / p.speed_time_constant_min
            accel = max(-request.limits.max_deceleration, min(request.limits.max_acceleration, raw_accel))
            mid_speed = max(0.0, speed + 0.5 * accel * dt)
            new_speed = max(0.0, speed + accel * dt)
            if (target_speed - speed) * (target_speed - new_speed) < 0:
                new_speed = target_speed

            mid_heading = normalize_heading(heading + 0.5 * mid_yaw * dt)
            theta = mid_heading * pi / 180.0
            ds = mid_speed * dt
            x += sin(theta) * ds
            y += cos(theta) * ds
            distance += ds
            heading = normalize_heading(heading + mid_yaw * dt)
            speed = new_speed
            yaw_rate = new_yaw
            last_accel = accel
            remaining -= dt
            steps += 1
            if steps > 1_000_000:
                raise RuntimeError("unexpected plant integration step count")

        out = PlantDynamicState(
            vessel=Vessel2DState(vessel.vessel_id, (x, y), heading, speed),
            yaw_rate_deg_per_min=yaw_rate,
            surge_acceleration_nm_per_min2=last_accel,
            opaque_state=request.state.opaque_state,
        )
        return out, distance, steps

    def _integrate_dynamics(
        self, request: PlantStepRequest
    ) -> tuple[PlantDynamicState, float, int]:
        """Run the exact numerical core with bounded frozen-prefix reuse."""
        if request.physical_environment is not None or request.actuator_command is not None:
            return self._integrate_dynamics_uncached(request)

        family = self._prefix_family_key(request)
        entry = self._prefix_cache.get(family)
        if entry is None:
            if family not in self._prefix_seen:
                if len(self._prefix_seen) >= self._PREFIX_SEEN_MAX_FAMILIES:
                    self._prefix_seen.pop(next(iter(self._prefix_seen)))
                self._prefix_seen[family] = None
                return self._integrate_dynamics_uncached(request)

            partition = self._duration_partition(request.duration_min)
            if partition is None:
                return self._integrate_dynamics_uncached(request)

            self._prefix_seen.pop(family, None)
            if len(self._prefix_cache) >= self._PREFIX_CACHE_MAX_FAMILIES:
                self._prefix_cache.pop(next(iter(self._prefix_cache)))
            entry = _FirstOrderPrefixCacheEntry(
                template=request,
                states=[request.state],
                distance_prefix_nm=[0.0],
            )
            self._prefix_cache[family] = entry
        else:
            partition = self._duration_partition(request.duration_min)
            if partition is None:
                return self._integrate_dynamics_uncached(request)

        full_steps, residual = partition
        max_step = self.parameters.max_step_min

        while len(entry.states) - 1 < full_steps:
            step_request = self._request_from_prefix(
                entry.template,
                state=entry.states[-1],
                duration_min=max_step,
            )
            state, distance, steps = self._integrate_dynamics_uncached(step_request)
            if steps != 1:
                # Defensive fallback: native max-step integration must remain one step.
                # Discard the current implementation cache entry before falling back so
                # no partially built prefix can be reused by a later request.
                self._prefix_cache.pop(family, None)
                return self._integrate_dynamics_uncached(request)
            entry.states.append(state)
            entry.distance_prefix_nm.append(entry.distance_prefix_nm[-1] + distance)

        prefix_state = entry.states[full_steps]
        prefix_distance = entry.distance_prefix_nm[full_steps]
        if residual <= _EPS:
            return prefix_state, prefix_distance, full_steps

        residual_request = self._request_from_prefix(
            entry.template,
            state=prefix_state,
            duration_min=residual,
        )
        state, residual_distance, residual_steps = self._integrate_dynamics_uncached(residual_request)
        return state, prefix_distance + residual_distance, full_steps + residual_steps

    def step(self, request: PlantStepRequest) -> PlantStepResult:
        out, distance, steps = self._integrate_dynamics(request)
        return PlantStepResult(
            status=PlantStepStatus.OK,
            spec_sha256=self._spec_sha256,
            request_sha256=request.identity_sha256(),
            state=out,
            distance_nm=distance,
            integration_steps=steps,
            diagnostics=(("model_scope", "low_order_sensitivity_not_validated_mmg"),),
        )


@dataclass(frozen=True, slots=True)
class ExternalPlantOutcome:
    payload: Mapping[str, Any] | None
    error: str | None = None


class ExternalPlantTransport(Protocol):
    transport_id: str

    def exchange(self, request: PlantStepRequest) -> ExternalPlantOutcome:
        ...


class ExternalCoSimulationPlantAdapter:
    """Adapter boundary for MMG/OpenModelica/other co-simulation plants."""
    component_id = TestbedFeature.C16_PLANT_FIDELITY

    def __init__(self, *, spec: PlantSpec, transport: ExternalPlantTransport) -> None:
        if spec.kind is not PlantModelKind.EXTERNAL_COSIM:
            raise ValueError("external co-simulation adapter requires EXTERNAL_COSIM spec")
        if not getattr(transport, "transport_id", ""):
            raise ValueError("external plant transport_id must be non-empty")
        self.spec = spec
        self.transport = transport

    def step(self, request: PlantStepRequest) -> PlantStepResult:
        base = dict(spec_sha256=self.spec.identity_sha256(), request_sha256=request.identity_sha256())
        try:
            outcome = self.transport.exchange(request)
        except Exception as exc:  # adapter fault containment boundary
            return PlantStepResult(status=PlantStepStatus.PLANT_ERROR, state=None, distance_nm=None,
                                   integration_steps=None, diagnostics=(("error", type(exc).__name__),), **base)
        if outcome.error is not None:
            return PlantStepResult(status=PlantStepStatus.PLANT_ERROR, state=None, distance_nm=None,
                                   integration_steps=None, diagnostics=(("error", str(outcome.error)),), **base)
        if outcome.payload is None:
            return PlantStepResult(status=PlantStepStatus.INVALID_RESPONSE, state=None, distance_nm=None,
                                   integration_steps=None, diagnostics=(("error", "missing_payload"),), **base)
        try:
            payload = _canonical(outcome.payload)
            if payload.get("vessel_id") != request.state.vessel.vessel_id:
                raise ValueError("external plant returned a different vessel_id")
            pos = payload["position_nm"]
            if not isinstance(pos, list) or len(pos) != 2:
                raise ValueError("position_nm must contain two coordinates")
            surge_knots = payload.get("surge_velocity_knots")
            sway_knots = payload.get("sway_velocity_knots")
            if (surge_knots is None) != (sway_knots is None):
                raise ValueError("surge_velocity_knots and sway_velocity_knots must be supplied together")
            surge_velocity = None
            sway_velocity = None
            if surge_knots is not None:
                surge_velocity = knots_to_nm_per_minute(float(surge_knots))
                sway_velocity = knots_to_nm_per_minute(float(sway_knots))
                projected_speed = hypot(surge_velocity, sway_velocity)
                if "speed_knots" in payload:
                    legacy_speed = knots_to_nm_per_minute(float(payload["speed_knots"]))
                    if abs(projected_speed - legacy_speed) > 1e-9:
                        raise ValueError("speed_knots is inconsistent with surge/sway velocity")
                vessel_speed = projected_speed
            else:
                vessel_speed = knots_to_nm_per_minute(float(payload["speed_knots"]))

            vessel = Vessel2DState(
                vessel_id=payload["vessel_id"],
                position=(float(pos[0]), float(pos[1])),
                heading_deg=float(payload["heading_deg"]),
                speed=vessel_speed,
            )
            opaque_raw = payload.get("opaque_state", {})
            if not isinstance(opaque_raw, Mapping):
                raise ValueError("opaque_state must be a mapping")
            opaque = tuple(sorted((str(k), v) for k, v in opaque_raw.items()))
            actuator_raw = payload.get("actuator_state")
            actuator_state = None
            if actuator_raw is not None:
                if not isinstance(actuator_raw, Mapping):
                    raise ValueError("actuator_state must be a mapping")
                actuator_state = PlantActuatorState(
                    rudder_angle_deg=float(actuator_raw["rudder_angle_deg"]),
                    propeller_rate_rps=float(actuator_raw["propeller_rate_rps"]),
                )
            state = PlantDynamicState(
                vessel=vessel,
                yaw_rate_deg_per_min=float(payload.get("yaw_rate_deg_per_min", 0.0)),
                surge_acceleration_nm_per_min2=float(payload.get("surge_acceleration_nm_per_min2", 0.0)),
                opaque_state=opaque,
                surge_velocity_nm_per_min=surge_velocity,
                sway_velocity_nm_per_min=sway_velocity,
                actuator_state=actuator_state,
            )
            distance = payload.get("distance_nm")
            steps = payload.get("integration_steps")
            return PlantStepResult(
                status=PlantStepStatus.OK,
                state=state,
                distance_nm=None if distance is None else float(distance),
                integration_steps=None if steps is None else int(steps),
                diagnostics=(("transport_id", str(self.transport.transport_id)),),
                **base,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return PlantStepResult(status=PlantStepStatus.INVALID_RESPONSE, state=None, distance_nm=None,
                                   integration_steps=None, diagnostics=(("error", str(exc)),), **base)


class NoOpPlantAdapter:
    """Compatibility mode: do not execute a C16 plant step."""
    component_id = TestbedFeature.C16_PLANT_FIDELITY
    spec = PlantSpec(
        kind=PlantModelKind.A162_KINEMATIC,
        model_id="noop",
        model_version="1",
        adapter_id="cross-c16-noop-v1",
    )

    def step(self, request: PlantStepRequest) -> None:
        return None
