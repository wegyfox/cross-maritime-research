"""Research-only environmental motion backend over C16 low-order vessel plants.

This bridge is deliberately outside :mod:`vessel_crossing.nextgen`: it lets Stage-2
sensitivity experiments use the same richer C16 plant in both world execution and A84
candidate prediction without making the immutable next-gen parity kernel depend on the
testbed layer.

The backend is an engineering sensitivity mechanism, not a calibrated vessel model.
Its identity binds the complete C16 plant specification and the explicit default
prediction horizon used when a caller does not supply one.
"""
from __future__ import annotations

from bisect import bisect_right, insort
from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Any

from vessel_crossing.kinematics import Vector2
from vessel_crossing.nextgen.dynamics import ManeuverLimits, Vessel2DState
from vessel_crossing.nextgen.environmental_candidate_transition import (
    EnvironmentalCommandTrajectory,
    EnvironmentalMotionAdvance,
    validate_environmental_motion_state,
)
from vessel_crossing.nextgen.environmental_dynamics import (
    EnvironmentalAdvanceResult,
    WaterRelativeManeuverCommand,
    WaterRelativeVesselState,
)
from vessel_crossing.nextgen.environmental_evidence import (
    EnvironmentalVesselMotionEvidence,
    environmental_vessel_motion_evidence,
)
from vessel_crossing.testbed.behavior import RequestedManeuver
from vessel_crossing.testbed.plant_fidelity import (
    FirstOrderResponsePlantAdapter,
    LowOrderActuatedPlantAdapter,
    LowOrderActuatedPlantParameters,
    PlantActuatorState,
    PlantActuatorTransientState,
    PlantDynamicState,
    PlantStepRequest,
    PlantStepStatus,
)

_EPS = 1e-12

LOW_ORDER_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID = (
    "CROSS_RESEARCH_LOW_ORDER_ENVIRONMENTAL_STATE/V1"
)
RESPONSE_LAG_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID = (
    "CROSS_RESEARCH_RESPONSE_LAG_ENVIRONMENTAL_STATE/V1"
)


def _command_key(command: WaterRelativeManeuverCommand) -> tuple[float, float]:
    return (command.target_through_water_heading_deg, command.target_through_water_speed)


def _same_command(
    first: tuple[float, float] | None,
    second: tuple[float, float],
) -> bool:
    if first is None:
        return False
    # Headings are already normalized; command-generation uses deterministic floats.
    return abs(first[0] - second[0]) <= _EPS and abs(first[1] - second[1]) <= _EPS


def _shift_dynamic_state(state: PlantDynamicState, dx: float, dy: float) -> PlantDynamicState:
    vessel = state.vessel
    shifted = Vessel2DState(
        vessel_id=vessel.vessel_id,
        position=(vessel.position[0] + dx, vessel.position[1] + dy),
        heading_deg=vessel.heading_deg,
        speed=vessel.speed,
    )
    return PlantDynamicState(
        vessel=shifted,
        yaw_rate_deg_per_min=state.yaw_rate_deg_per_min,
        surge_acceleration_nm_per_min2=state.surge_acceleration_nm_per_min2,
        opaque_state=state.opaque_state,
        surge_velocity_nm_per_min=state.surge_velocity_nm_per_min,
        sway_velocity_nm_per_min=state.sway_velocity_nm_per_min,
        actuator_state=state.actuator_state,
        actuator_transient_state=state.actuator_transient_state,
    )


@dataclass(frozen=True, slots=True)
class LowOrderEnvironmentalMotionState:
    plant_state: PlantDynamicState
    active_command: tuple[float, float] | None
    command_age_min: float

    @property
    def backend_state_contract_id(self) -> str:
        return LOW_ORDER_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID

    def __post_init__(self) -> None:
        if not isfinite(self.command_age_min) or self.command_age_min < 0.0:
            raise ValueError("command_age_min must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LowOrderEnvironmentalTrajectory:
    initial_state: WaterRelativeVesselState
    environmental_drift_velocity: Vector2
    command: WaterRelativeManeuverCommand
    limits: ManeuverLimits
    transition_duration_min: float
    start_motion: EnvironmentalVesselMotionEvidence
    _backend: "LowOrderEnvironmentalMotionBackend"
    _motion_state: LowOrderEnvironmentalMotionState
    _cache: dict[float, tuple[WaterRelativeVesselState, LowOrderEnvironmentalMotionState]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        # Mutable cache is telemetry/performance state only; it is not part of the
        # trajectory identity or modeled dynamics.  Seed it with the frozen snapshot.
        self._cache[0.0] = (self.initial_state, self._motion_state)

    @property
    def reachable(self) -> bool:
        return True

    @property
    def transition_duration_is_target_completion(self) -> bool:
        # Richer plant trajectories use ``transition_duration_min`` as the bounded
        # prediction/certification horizon.  Unlike the historical parity A84
        # transition, that horizon is not a promise that the water-relative command
        # target is physically reached at its end.
        return False

    @property
    def final_state(self) -> WaterRelativeVesselState:
        return self.state_at_or_continue(self.transition_duration_min)

    @property
    def ground_speed_bound(self) -> float:
        return self._backend.ground_speed_bound_nm_per_min(self.environmental_drift_velocity)

    @property
    def turn_rate_bound_deg_per_min(self) -> float:
        # LowOrderActuatedPlantAdapter clamps actual yaw to the inherited maneuver bound.
        return self.limits.max_turn_rate_deg_per_min

    def state_at(self, elapsed_min: float) -> WaterRelativeVesselState:
        if elapsed_min < -_EPS or elapsed_min > self.transition_duration_min + _EPS:
            raise ValueError("elapsed_min must lie inside the candidate trajectory horizon")
        return self.state_at_or_continue(min(max(float(elapsed_min), 0.0), self.transition_duration_min))

    def state_at_or_continue(self, elapsed_min: float) -> WaterRelativeVesselState:
        if not isfinite(elapsed_min) or elapsed_min < -_EPS:
            raise ValueError("elapsed_min must be finite and non-negative")
        target = max(0.0, float(elapsed_min))
        key = round(target, 12)
        cached = self._cache.get(key)
        if cached is not None:
            return cached[0]

        # Build a dense deterministic prefix once. Domain-certification queries are
        # recursive/non-monotonic; caching every native plant step avoids re-integrating
        # 0..t independently for each midpoint while preserving the exact fixed-step
        # low-order integrator semantics.
        prior_key = max(k for k in self._cache if k <= key + _EPS)
        public, motion = self._cache[prior_key]
        current = prior_key
        # Chunking is a cache/performance choice only: ``plant.step`` still performs
        # the exact same native ``max_step_min`` integration internally.
        cache_chunk = max(0.25, self._backend.parameters.max_step_min)
        while current < target - _EPS:
            dt = min(cache_chunk, target - current)
            advanced = self._backend._simulate_from_frozen_state(  # noqa: SLF001
                public, self.command, dt, self.limits,
                self.environmental_drift_velocity, motion,
            )
            current = min(target, current + dt)
            public = advanced.advance.state
            motion = advanced.motion_state
            self._cache[round(current, 12)] = (public, motion)
        self._cache[key] = (public, motion)
        return public

    def state_and_motion_at_or_continue(self, elapsed_min: float):
        """Return public state and opaque plant sidecar from one cached continuation path."""
        state = self.state_at_or_continue(elapsed_min)
        key = round(max(0.0, float(elapsed_min)), 12)
        cached_state, motion_state = self._cache[key]
        # Keep the public value returned by state_at_or_continue as the authority and
        # assert that the paired cache entry is exactly the same object/value path.
        if cached_state != state:
            raise RuntimeError("trajectory state/sidecar cache pair is inconsistent")
        validate_environmental_motion_state(
            self._backend,
            motion_state,
            boundary="low-order trajectory continuation",
        )
        return state, motion_state

    def motion_at(self, elapsed_min: float) -> EnvironmentalVesselMotionEvidence:
        state = self.state_at(elapsed_min)
        return environmental_vessel_motion_evidence(state, self.environmental_drift_velocity)


class LowOrderEnvironmentalMotionBackend:
    """Closed-loop environmental backend over the C16 low-order actuated plant.

    The same backend object is used by world execution and candidate prediction.  Its
    sidecar remembers actuator/body-dynamics state and command age, so event slicing does
    not re-apply rudder/propulsion response delays on every physical interval.
    """

    backend_state_contract_id = LOW_ORDER_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID

    def __init__(
        self,
        parameters: LowOrderActuatedPlantParameters,
        *,
        default_prediction_horizon_min: float,
    ) -> None:
        if not isinstance(parameters, LowOrderActuatedPlantParameters):
            raise TypeError("parameters must be LowOrderActuatedPlantParameters")
        if not isfinite(default_prediction_horizon_min) or default_prediction_horizon_min <= 0.0:
            raise ValueError("default_prediction_horizon_min must be finite and positive")
        self.parameters = parameters
        self.plant = LowOrderActuatedPlantAdapter(parameters)
        self.default_prediction_horizon_min = float(default_prediction_horizon_min)
        self.backend_id = (
            "CROSS_RESEARCH_LOW_ORDER_ENVIRONMENTAL/"
            + self.plant.spec.identity_sha256()
            + f"/H{self.default_prediction_horizon_min:.12g}"
        )

    def initialize_motion_state(
        self,
        state: WaterRelativeVesselState,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
    ) -> LowOrderEnvironmentalMotionState:
        del limits, environmental_drift_velocity
        speed_knots = state.through_water_speed * 60.0
        propeller = self.parameters.propeller_rate_for_nonnegative_speed_knots(speed_knots)
        body = PlantDynamicState(
            vessel=Vessel2DState(
                state.vessel_id,
                state.position,
                state.through_water_heading_deg,
                state.through_water_speed,
            ),
            surge_velocity_nm_per_min=state.through_water_speed,
            sway_velocity_nm_per_min=0.0,
            actuator_state=PlantActuatorState(0.0, propeller),
            actuator_transient_state=PlantActuatorTransientState(
                last_nonzero_propeller_sign=(1 if propeller > _EPS else 0),
            ),
        )
        settled_age = max(
            self.parameters.rudder_response_delay_min,
            self.parameters.propulsion_response_delay_min,
        ) + self.parameters.max_step_min
        return LowOrderEnvironmentalMotionState(
            plant_state=body,
            active_command=(state.through_water_heading_deg, state.through_water_speed),
            command_age_min=settled_age,
        )

    def ground_speed_bound_nm_per_min(self, drift: Vector2) -> float:
        surge_knots = max(abs(p.equilibrium_surge_knots) for p in self.parameters.propulsion_map)
        sway_knots = 0.0 if self.parameters.sway_yaw is None else self.parameters.sway_yaw.max_abs_sway_knots
        water_bound = hypot(surge_knots, sway_knots) / 60.0
        return water_bound + hypot(float(drift[0]), float(drift[1]))

    def _request(
        self,
        state: LowOrderEnvironmentalMotionState,
        command: WaterRelativeManeuverCommand,
        duration_min: float,
        limits: ManeuverLimits,
    ) -> tuple[PlantStepRequest, float]:
        key = _command_key(command)
        age = state.command_age_min if _same_command(state.active_command, key) else 0.0
        maneuver = RequestedManeuver(
            requested_at_min=0.0,
            target_heading_deg=command.target_through_water_heading_deg,
            target_speed_knots=command.target_through_water_speed * 60.0,
            source="environmental-motion-backend",
        )
        return (
            PlantStepRequest(
                time_min=age,
                duration_min=duration_min,
                state=state.plant_state,
                maneuver=maneuver,
                limits=limits,
            ),
            age,
        )

    def _simulate_from_frozen_state(
        self,
        state: WaterRelativeVesselState,
        command: WaterRelativeManeuverCommand,
        duration_min: float,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
        motion_state: LowOrderEnvironmentalMotionState,
    ) -> EnvironmentalMotionAdvance:
        if not isinstance(motion_state, LowOrderEnvironmentalMotionState):
            raise TypeError("low-order environmental backend requires LowOrderEnvironmentalMotionState")
        if duration_min < 0.0 or not isfinite(duration_min):
            raise ValueError("duration_min must be finite and non-negative")
        # Planning context validates that sidecar and public state are snapshotted
        # together. Keep an explicit check here because a stale sidecar would otherwise
        # make candidate/execution evidence silently diverge.
        ps = motion_state.plant_state.vessel
        if (
            ps.vessel_id != state.vessel_id
            or abs(ps.position[0] - state.position[0]) > 1e-10
            or abs(ps.position[1] - state.position[1]) > 1e-10
            or abs(ps.heading_deg - state.through_water_heading_deg) > 1e-10
            or abs(ps.speed - state.through_water_speed) > 1e-10
        ):
            raise ValueError("public environmental state and low-order sidecar are inconsistent")

        request, starting_age = self._request(motion_state, command, duration_min, limits)
        stepped = self.plant.step(request)
        if stepped.status is not PlantStepStatus.OK or stepped.state is None:
            raise RuntimeError(f"low-order plant step failed: {stepped.diagnostics}")
        dx = float(environmental_drift_velocity[0]) * duration_min
        dy = float(environmental_drift_velocity[1]) * duration_min
        shifted = _shift_dynamic_state(stepped.state, dx, dy)
        final = WaterRelativeVesselState(
            vessel_id=state.vessel_id,
            position=shifted.vessel.position,
            through_water_heading_deg=shifted.vessel.heading_deg,
            through_water_speed=shifted.vessel.speed,
        )
        ground_displacement = (
            final.position[0] - state.position[0],
            final.position[1] - state.position[1],
        )
        advance = EnvironmentalAdvanceResult(
            state=final,
            duration=duration_min,
            segments=0 if stepped.integration_steps is None else stepped.integration_steps,
            through_water_distance=0.0 if stepped.distance_nm is None else stepped.distance_nm,
            environmental_displacement=(dx, dy),
            ground_displacement=ground_displacement,
        )
        return EnvironmentalMotionAdvance(
            advance=advance,
            motion_state=LowOrderEnvironmentalMotionState(
                plant_state=shifted,
                active_command=_command_key(command),
                command_age_min=starting_age + duration_min,
            ),
        )

    def advance(
        self,
        state: WaterRelativeVesselState,
        command: WaterRelativeManeuverCommand,
        duration_min: float,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
        motion_state: LowOrderEnvironmentalMotionState,
    ) -> EnvironmentalMotionAdvance:
        validate_environmental_motion_state(
            self, motion_state, boundary="low-order backend advance"
        )
        return self._simulate_from_frozen_state(
            state,
            command,
            duration_min,
            limits,
            environmental_drift_velocity,
            motion_state,
        )

    def build_trajectory(
        self,
        state: WaterRelativeVesselState,
        environmental_drift_velocity: Vector2,
        command: WaterRelativeManeuverCommand,
        limits: ManeuverLimits,
        motion_state: LowOrderEnvironmentalMotionState,
        *,
        evaluation_horizon_min: float | None = None,
    ) -> EnvironmentalCommandTrajectory:
        validate_environmental_motion_state(
            self, motion_state, boundary="low-order backend trajectory"
        )
        if not isinstance(motion_state, LowOrderEnvironmentalMotionState):
            raise TypeError("low-order environmental backend requires LowOrderEnvironmentalMotionState")
        horizon = self.default_prediction_horizon_min if evaluation_horizon_min is None else float(evaluation_horizon_min)
        if not isfinite(horizon) or horizon <= 0.0:
            raise ValueError("evaluation horizon must be finite and positive")
        return LowOrderEnvironmentalTrajectory(
            initial_state=state,
            environmental_drift_velocity=(
                float(environmental_drift_velocity[0]),
                float(environmental_drift_velocity[1]),
            ),
            command=command,
            limits=limits,
            transition_duration_min=horizon,
            start_motion=environmental_vessel_motion_evidence(state, environmental_drift_velocity),
            _backend=self,
            _motion_state=motion_state,
        )

@dataclass(frozen=True, slots=True)
class ResponseLagEnvironmentalMotionState:
    plant_state: PlantDynamicState
    active_command: tuple[float, float] | None
    command_age_min: float

    @property
    def backend_state_contract_id(self) -> str:
        return RESPONSE_LAG_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID

    def __post_init__(self) -> None:
        if not isfinite(self.command_age_min) or self.command_age_min < 0.0:
            raise ValueError("command_age_min must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ResponseLagEnvironmentalTrajectory:
    initial_state: WaterRelativeVesselState
    environmental_drift_velocity: Vector2
    command: WaterRelativeManeuverCommand
    limits: ManeuverLimits
    transition_duration_min: float
    start_motion: EnvironmentalVesselMotionEvidence
    _backend: "ResponseLagEnvironmentalMotionBackend"
    _motion_state: ResponseLagEnvironmentalMotionState
    _cache: dict[float, tuple[WaterRelativeVesselState, ResponseLagEnvironmentalMotionState]] = field(
        default_factory=dict, compare=False, repr=False
    )
    _cache_keys: list[float] = field(default_factory=list, compare=False, repr=False)

    def __post_init__(self) -> None:
        self._cache[0.0] = (self.initial_state, self._motion_state)
        if not self._cache_keys:
            self._cache_keys.append(0.0)

    def _cache_state(
        self,
        key: float,
        state: WaterRelativeVesselState,
        motion: ResponseLagEnvironmentalMotionState,
    ) -> None:
        if key not in self._cache:
            insort(self._cache_keys, key)
        self._cache[key] = (state, motion)

    @property
    def reachable(self) -> bool:
        return True

    @property
    def transition_duration_is_target_completion(self) -> bool:
        # Richer plant trajectories use ``transition_duration_min`` as the bounded
        # prediction/certification horizon.  Unlike the historical parity A84
        # transition, that horizon is not a promise that the water-relative command
        # target is physically reached at its end.
        return False

    @property
    def final_state(self) -> WaterRelativeVesselState:
        return self.state_at_or_continue(self.transition_duration_min)

    @property
    def ground_speed_bound(self) -> float:
        water = max(
            self.initial_state.through_water_speed,
            self.command.target_through_water_speed,
        )
        return water + hypot(*self.environmental_drift_velocity)

    @property
    def turn_rate_bound_deg_per_min(self) -> float:
        return self.limits.max_turn_rate_deg_per_min

    def state_at(self, elapsed_min: float) -> WaterRelativeVesselState:
        if elapsed_min < -_EPS or elapsed_min > self.transition_duration_min + _EPS:
            raise ValueError("elapsed_min must lie inside the candidate trajectory horizon")
        return self.state_at_or_continue(min(max(float(elapsed_min), 0.0), self.transition_duration_min))

    def state_at_or_continue(self, elapsed_min: float) -> WaterRelativeVesselState:
        if not isfinite(elapsed_min) or elapsed_min < -_EPS:
            raise ValueError("elapsed_min must be finite and non-negative")
        target = max(0.0, float(elapsed_min))
        key = round(target, 12)
        cached = self._cache.get(key)
        if cached is not None:
            return cached[0]
        prior_index = bisect_right(self._cache_keys, key + _EPS) - 1
        if prior_index < 0:
            raise RuntimeError("response-lag trajectory cache lost its frozen-state checkpoint")
        prior_key = self._cache_keys[prior_index]
        public, motion = self._cache[prior_key]
        current = prior_key
        cache_chunk = 0.5
        while current < target - _EPS:
            dt = min(cache_chunk, target - current)
            advanced = self._backend._simulate_from_frozen_state(  # noqa: SLF001
                public, self.command, dt, self.limits,
                self.environmental_drift_velocity, motion,
            )
            current = min(target, current + dt)
            public = advanced.advance.state
            motion = advanced.motion_state
            self._cache_state(round(current, 12), public, motion)
        self._cache_state(key, public, motion)
        return public

    def state_and_motion_at_or_continue(self, elapsed_min: float):
        """Return public state and opaque plant sidecar from one cached continuation path."""
        state = self.state_at_or_continue(elapsed_min)
        key = round(max(0.0, float(elapsed_min)), 12)
        cached_state, motion_state = self._cache[key]
        # Keep the public value returned by state_at_or_continue as the authority and
        # assert that the paired cache entry is exactly the same object/value path.
        if cached_state != state:
            raise RuntimeError("trajectory state/sidecar cache pair is inconsistent")
        validate_environmental_motion_state(
            self._backend,
            motion_state,
            boundary="response-lag trajectory continuation",
        )
        return state, motion_state

    def motion_at(self, elapsed_min: float) -> EnvironmentalVesselMotionEvidence:
        return environmental_vessel_motion_evidence(
            self.state_at(elapsed_min), self.environmental_drift_velocity
        )


class ResponseLagEnvironmentalMotionBackend:
    """Research bridge for C16 first-order/tabulated response plants.

    The wrapped adapter must expose deterministic ``step`` and ``spec`` attributes and
    must not require explicit low-level actuator state.  This backend is intended for
    Stage-2A model-form sensitivity; it is not a calibrated hydrodynamic model.
    """

    backend_state_contract_id = RESPONSE_LAG_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID

    def __init__(
        self,
        plant_adapter: Any,
        *,
        default_prediction_horizon_min: float,
        trajectory_cache_max_entries: int = 20_000,
    ) -> None:
        if not hasattr(plant_adapter, "step") or not hasattr(plant_adapter, "spec"):
            raise TypeError("plant_adapter must expose step() and spec")
        if not isfinite(default_prediction_horizon_min) or default_prediction_horizon_min <= 0.0:
            raise ValueError("default_prediction_horizon_min must be finite and positive")
        if isinstance(trajectory_cache_max_entries, bool) or not isinstance(trajectory_cache_max_entries, int):
            raise TypeError("trajectory_cache_max_entries must be an integer")
        if trajectory_cache_max_entries < 1:
            raise ValueError("trajectory_cache_max_entries must be at least 1")
        self.plant = plant_adapter
        self.default_prediction_horizon_min = float(default_prediction_horizon_min)
        self.backend_id = (
            "CROSS_RESEARCH_RESPONSE_ENVIRONMENTAL/"
            + self.plant.spec.identity_sha256()
            + f"/H{self.default_prediction_horizon_min:.12g}"
        )
        # Exact deterministic memoization only.  The complete physical/query state is
        # part of the key, so this cannot alias trajectories across different planner
        # states.  The cap prevents a long campaign from accumulating unbounded cache.
        self._trajectory_cache: dict[tuple[Any, ...], ResponseLagEnvironmentalTrajectory] = {}
        self._trajectory_cache_max_entries = trajectory_cache_max_entries

    def initialize_motion_state(
        self,
        state: WaterRelativeVesselState,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
    ) -> ResponseLagEnvironmentalMotionState:
        del limits, environmental_drift_velocity
        body = PlantDynamicState(
            vessel=Vessel2DState(
                state.vessel_id, state.position,
                state.through_water_heading_deg, state.through_water_speed,
            )
        )
        return ResponseLagEnvironmentalMotionState(
            plant_state=body,
            active_command=(state.through_water_heading_deg, state.through_water_speed),
            command_age_min=1.0e6,
        )

    def _simulate_from_frozen_state(
        self,
        state: WaterRelativeVesselState,
        command: WaterRelativeManeuverCommand,
        duration_min: float,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
        motion_state: ResponseLagEnvironmentalMotionState,
    ) -> EnvironmentalMotionAdvance:
        if not isinstance(motion_state, ResponseLagEnvironmentalMotionState):
            raise TypeError("response backend requires ResponseLagEnvironmentalMotionState")
        ps = motion_state.plant_state.vessel
        if (
            ps.vessel_id != state.vessel_id
            or abs(ps.position[0] - state.position[0]) > 1e-10
            or abs(ps.position[1] - state.position[1]) > 1e-10
            or abs(ps.heading_deg - state.through_water_heading_deg) > 1e-10
            or abs(ps.speed - state.through_water_speed) > 1e-10
        ):
            raise ValueError("public environmental state and response sidecar are inconsistent")
        key = _command_key(command)
        age = motion_state.command_age_min if _same_command(motion_state.active_command, key) else 0.0
        maneuver = RequestedManeuver(
            requested_at_min=0.0,
            target_heading_deg=command.target_through_water_heading_deg,
            target_speed_knots=command.target_through_water_speed * 60.0,
            source="environmental-response-backend",
        )
        request = PlantStepRequest(
            time_min=age,
            duration_min=duration_min,
            state=motion_state.plant_state,
            maneuver=maneuver,
            limits=limits,
        )
        if isinstance(self.plant, FirstOrderResponsePlantAdapter):
            stepped_state, stepped_distance, stepped_steps = self.plant._integrate_dynamics(request)  # noqa: SLF001
        else:
            stepped = self.plant.step(request)
            if stepped.status is not PlantStepStatus.OK or stepped.state is None:
                raise RuntimeError(f"response plant step failed: {stepped.diagnostics}")
            stepped_state = stepped.state
            stepped_distance = stepped.distance_nm
            stepped_steps = stepped.integration_steps
        dx = float(environmental_drift_velocity[0]) * duration_min
        dy = float(environmental_drift_velocity[1]) * duration_min
        shifted = _shift_dynamic_state(stepped_state, dx, dy)
        final = WaterRelativeVesselState(
            vessel_id=state.vessel_id,
            position=shifted.vessel.position,
            through_water_heading_deg=shifted.vessel.heading_deg,
            through_water_speed=shifted.vessel.speed,
        )
        advance = EnvironmentalAdvanceResult(
            state=final,
            duration=duration_min,
            segments=0 if stepped_steps is None else stepped_steps,
            through_water_distance=0.0 if stepped_distance is None else stepped_distance,
            environmental_displacement=(dx, dy),
            ground_displacement=(final.position[0]-state.position[0], final.position[1]-state.position[1]),
        )
        return EnvironmentalMotionAdvance(
            advance=advance,
            motion_state=ResponseLagEnvironmentalMotionState(
                plant_state=shifted,
                active_command=key,
                command_age_min=age + duration_min,
            ),
        )

    def advance(
        self,
        state: WaterRelativeVesselState,
        command: WaterRelativeManeuverCommand,
        duration_min: float,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
        motion_state: ResponseLagEnvironmentalMotionState,
    ) -> EnvironmentalMotionAdvance:
        validate_environmental_motion_state(
            self, motion_state, boundary="response-lag backend advance"
        )
        return self._simulate_from_frozen_state(
            state, command, duration_min, limits, environmental_drift_velocity, motion_state
        )

    def build_trajectory(
        self,
        state: WaterRelativeVesselState,
        environmental_drift_velocity: Vector2,
        command: WaterRelativeManeuverCommand,
        limits: ManeuverLimits,
        motion_state: ResponseLagEnvironmentalMotionState,
        *,
        evaluation_horizon_min: float | None = None,
    ) -> EnvironmentalCommandTrajectory:
        validate_environmental_motion_state(
            self, motion_state, boundary="response-lag backend trajectory"
        )
        if not isinstance(motion_state, ResponseLagEnvironmentalMotionState):
            raise TypeError("response backend requires ResponseLagEnvironmentalMotionState")
        horizon = self.default_prediction_horizon_min if evaluation_horizon_min is None else float(evaluation_horizon_min)
        if not isfinite(horizon) or horizon <= 0.0:
            raise ValueError("evaluation horizon must be finite and positive")
        drift = (float(environmental_drift_velocity[0]), float(environmental_drift_velocity[1]))
        key = (state, drift, command, limits, motion_state, round(horizon, 12))
        cached = self._trajectory_cache.get(key)
        if cached is not None:
            return cached
        trajectory = ResponseLagEnvironmentalTrajectory(
            initial_state=state,
            environmental_drift_velocity=drift,
            command=command,
            limits=limits,
            transition_duration_min=horizon,
            start_motion=environmental_vessel_motion_evidence(state, environmental_drift_velocity),
            _backend=self,
            _motion_state=motion_state,
        )
        if len(self._trajectory_cache) >= self._trajectory_cache_max_entries:
            self._trajectory_cache.clear()
        self._trajectory_cache[key] = trajectory
        return trajectory
