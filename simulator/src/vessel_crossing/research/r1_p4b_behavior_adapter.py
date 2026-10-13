"""R1 P4B behavior-command adapter for heterogeneous AS/HS execution.

This is an opt-in research-layer composition over the validated environmental motion
backend.  It does not modify the closed A162/P4B runtime.  The same wrapper is used by
candidate prediction and world execution, so a declared command-response delay is part
of both safety certification and realized motion.

The adapter intentionally models only *command response timing*.  It does not claim
that a numeric delay is a complete human model, COLREG implementation, or production
autonomy model.  Publication values remain a separate protocol-grounding decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import hypot, isfinite
from types import MappingProxyType
from typing import Any, Mapping
from weakref import WeakValueDictionary

from vessel_crossing.kinematics import Vector2
from vessel_crossing.nextgen.dynamics import ManeuverLimits
from vessel_crossing.nextgen.environmental_candidate_transition import (
    EnvironmentalCommandTrajectory,
    EnvironmentalMotionAdvance,
    EnvironmentalMotionBackend,
    EnvironmentalMotionState,
    StatefulEnvironmentalCommandTrajectory,
    environmental_motion_state_contract_id,
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
from vessel_crossing.testbed.behavior import BehaviorModelSpec, BehaviorProfile

_EPS = 1e-12
P4B_BEHAVIOR_MOTION_STATE_CONTRACT_PREFIX = "R1_P4B_BEHAVIOR_MOTION_STATE/V1"


def _p4b_behavior_motion_state_contract_id(base_contract_id: str) -> str:
    if not isinstance(base_contract_id, str) or not base_contract_id:
        raise TypeError("base motion-state contract id must be a non-empty string")
    return f"{P4B_BEHAVIOR_MOTION_STATE_CONTRACT_PREFIX}/{base_contract_id}"


_DELAYED_TIME_CACHE_MAX = 4096
_ALLOWED_ROLE_PROFILES = frozenset(
    (BehaviorProfile.COMPLIANT_MERCHANT, BehaviorProfile.AUTONOMOUS_POLICY)
)


@dataclass(frozen=True, slots=True)
class P4BBehaviorBinding:
    """One vessel's declared microscopic role and command-response timing.

    ``command_delay_min`` is a behavior/controller response delay above the physical
    plant.  It is not a plant time constant.  A value of zero is exact pass-through.
    """

    profile: BehaviorProfile
    adapter_id: str
    command_delay_min: float = 0.0

    def __post_init__(self) -> None:
        if self.profile not in _ALLOWED_ROLE_PROFILES:
            raise ValueError("P4B behavior binding must be COMPLIANT_MERCHANT or AUTONOMOUS_POLICY")
        if not self.adapter_id:
            raise ValueError("adapter_id must be non-empty")
        if not isfinite(self.command_delay_min) or self.command_delay_min < 0.0:
            raise ValueError("command_delay_min must be finite and non-negative")

    @property
    def spec(self) -> BehaviorModelSpec:
        return BehaviorModelSpec(
            self.profile,
            self.adapter_id,
            parameters=(("command_delay_min", format(self.command_delay_min, ".17g")),),
        )

    @property
    def identity_sha256(self) -> str:
        return self.spec.identity_sha256()


@dataclass(frozen=True, slots=True)
class P4BBehaviorMotionState:
    """Contract-owned world sidecar for command-response state above a base plant backend."""

    base_motion_state: EnvironmentalMotionState | None
    effective_command: WaterRelativeManeuverCommand
    pending_requested_command: WaterRelativeManeuverCommand | None
    pending_age_min: float
    binding_sha256: str

    @property
    def backend_state_contract_id(self) -> str:
        return _p4b_behavior_motion_state_contract_id(
            environmental_motion_state_contract_id(self.base_motion_state)
        )

    def __post_init__(self) -> None:
        if not isinstance(self.effective_command, WaterRelativeManeuverCommand):
            raise TypeError("effective_command must be WaterRelativeManeuverCommand")
        if self.pending_requested_command is not None and not isinstance(
            self.pending_requested_command, WaterRelativeManeuverCommand
        ):
            raise TypeError("pending_requested_command must be WaterRelativeManeuverCommand or None")
        if not isfinite(self.pending_age_min) or self.pending_age_min < 0.0:
            raise ValueError("pending_age_min must be finite and non-negative")
        if not self.binding_sha256:
            raise ValueError("binding_sha256 must be non-empty")


def _sum_advance_results(
    initial: WaterRelativeVesselState,
    parts: tuple[EnvironmentalAdvanceResult, ...],
) -> EnvironmentalAdvanceResult:
    if not parts:
        return EnvironmentalAdvanceResult(
            state=initial,
            duration=0.0,
            segments=0,
            through_water_distance=0.0,
            environmental_displacement=(0.0, 0.0),
            ground_displacement=(0.0, 0.0),
        )
    final = parts[-1].state
    return EnvironmentalAdvanceResult(
        state=final,
        duration=sum(item.duration for item in parts),
        segments=sum(item.segments for item in parts),
        through_water_distance=sum(item.through_water_distance for item in parts),
        environmental_displacement=(
            sum(item.environmental_displacement[0] for item in parts),
            sum(item.environmental_displacement[1] for item in parts),
        ),
        ground_displacement=(final.position[0] - initial.position[0], final.position[1] - initial.position[1]),
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class P4BBehaviorCommandTrajectory:
    """Candidate trajectory under one frozen requested command and behavior sidecar."""

    initial_state: WaterRelativeVesselState
    environmental_drift_velocity: Vector2
    command: WaterRelativeManeuverCommand
    limits: ManeuverLimits
    transition_duration_min: float
    start_motion: EnvironmentalVesselMotionEvidence
    _backend: "P4BBehaviorEnvironmentalMotionBackend"
    _motion_state: P4BBehaviorMotionState
    _cache: dict[float, tuple[WaterRelativeVesselState, P4BBehaviorMotionState]] = field(
        default_factory=dict, compare=False, repr=False
    )
    _canonical_time_targets: dict[float, float] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        self._cache[0.0] = (self.initial_state, self._motion_state)

    def _remember_pair(
        self,
        key: float,
        pair: tuple[WaterRelativeVesselState, P4BBehaviorMotionState],
    ) -> None:
        if key == 0.0:
            self._cache[key] = pair
            return
        if len(self._cache) >= _DELAYED_TIME_CACHE_MAX:
            # 0.0 is inserted first and remains the frozen anchor.  Evicting any
            # other exact memoized timestamp changes only hit-rate: a later query
            # deterministically recomputes from the same frozen initial state.
            oldest = next((item for item in self._cache if item != 0.0), None)
            if oldest is not None:
                self._cache.pop(oldest, None)
        self._cache[key] = pair

    def _pair_at_or_continue(
        self, elapsed_min: float
    ) -> tuple[WaterRelativeVesselState, P4BBehaviorMotionState]:
        if not isfinite(elapsed_min) or elapsed_min < -_EPS:
            raise ValueError("elapsed_min must be finite and non-negative")
        target = max(0.0, float(elapsed_min))
        key = round(target, 12)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Preserve the historical rounded-key first-writer semantics exactly.
        # The original cache computed the first raw timestamp associated with a
        # rounded key and reused that result forever.  Heavy state/sidecar pairs may
        # now be evicted, so retain only that first raw timestamp as lightweight
        # canonical metadata.  Any later recomputation uses the same raw timestamp.
        canonical_target = self._canonical_time_targets.get(key)
        first_sighting = canonical_target is None
        if first_sighting:
            canonical_target = target
            self._canonical_time_targets[key] = canonical_target

        # Pure frozen-start query: prediction must not depend on the order in which
        # candidate states were previously requested.  Do not retain a heavy result
        # for one-off adaptive timestamps; the second sighting promotes it into the
        # bounded exact memoization cache.
        advanced = self._backend._advance_from_frozen_state(  # noqa: SLF001
            self.initial_state,
            self.command,
            canonical_target,
            self.limits,
            self.environmental_drift_velocity,
            self._motion_state,
        )
        pair = (advanced.advance.state, advanced.motion_state)
        if not first_sighting:
            self._remember_pair(key, pair)
        return pair

    @property
    def reachable(self) -> bool:
        return True

    @property
    def transition_duration_is_target_completion(self) -> bool:
        return False

    @property
    def final_state(self) -> WaterRelativeVesselState:
        return self.state_at_or_continue(self.transition_duration_min)

    @property
    def ground_speed_bound(self) -> float:
        speeds = [
            self.initial_state.through_water_speed,
            self.command.target_through_water_speed,
            self._motion_state.effective_command.target_through_water_speed,
        ]
        if self._motion_state.pending_requested_command is not None:
            speeds.append(self._motion_state.pending_requested_command.target_through_water_speed)
        return max(speeds) + hypot(*self.environmental_drift_velocity)

    @property
    def turn_rate_bound_deg_per_min(self) -> float:
        return self.limits.max_turn_rate_deg_per_min

    def state_at(self, elapsed_min: float) -> WaterRelativeVesselState:
        if elapsed_min < -_EPS or elapsed_min > self.transition_duration_min + _EPS:
            raise ValueError("elapsed_min must lie inside the candidate trajectory horizon")
        return self.state_at_or_continue(min(max(float(elapsed_min), 0.0), self.transition_duration_min))

    def state_at_or_continue(self, elapsed_min: float) -> WaterRelativeVesselState:
        return self._pair_at_or_continue(elapsed_min)[0]

    def state_and_motion_at_or_continue(
        self, elapsed_min: float
    ) -> tuple[WaterRelativeVesselState, P4BBehaviorMotionState]:
        """Return public state and behavior-wrapped sidecar from one frozen path."""
        state, motion_state = self._pair_at_or_continue(elapsed_min)
        validate_environmental_motion_state(
            self._backend,
            motion_state,
            boundary="P4B behavior trajectory continuation",
        )
        return state, motion_state

    def motion_at(self, elapsed_min: float) -> EnvironmentalVesselMotionEvidence:
        return environmental_vessel_motion_evidence(
            self.state_at(elapsed_min), self.environmental_drift_velocity
        )


@dataclass(frozen=True, slots=True)
class P4BBehaviorDelegatedCommandTrajectory:
    """Behavior-preserving proxy over a base trajectory when no delay is pending.

    The base plant remains authoritative for candidate-state prediction, including its
    own cached integration path.  The proxy's paired continuation method re-wraps the
    base plant sidecar in :class:`P4BBehaviorMotionState`, so callers that consume a
    trajectory's richer state+sidecar continuation can safely feed that sidecar back
    into the behavior backend rather than accidentally bypassing the composition.
    """

    _base_trajectory: EnvironmentalCommandTrajectory
    _backend: "P4BBehaviorEnvironmentalMotionBackend"
    _motion_state: P4BBehaviorMotionState

    def __post_init__(self) -> None:
        if self._motion_state.pending_requested_command is not None:
            raise ValueError("delegated behavior trajectory requires no pending command")

    @property
    def initial_state(self) -> WaterRelativeVesselState:
        return self._base_trajectory.initial_state

    @property
    def environmental_drift_velocity(self) -> Vector2:
        return self._base_trajectory.environmental_drift_velocity

    @property
    def command(self) -> WaterRelativeManeuverCommand:
        return self._base_trajectory.command

    @property
    def limits(self) -> ManeuverLimits:
        return self._base_trajectory.limits

    @property
    def transition_duration_min(self) -> float:
        return self._base_trajectory.transition_duration_min

    @property
    def start_motion(self) -> EnvironmentalVesselMotionEvidence:
        return self._base_trajectory.start_motion

    @property
    def reachable(self) -> bool:
        return self._base_trajectory.reachable

    @property
    def transition_duration_is_target_completion(self) -> bool:
        return bool(getattr(self._base_trajectory, "transition_duration_is_target_completion", False))

    @property
    def final_state(self) -> WaterRelativeVesselState | None:
        return self._base_trajectory.final_state

    @property
    def ground_speed_bound(self) -> float:
        return self._base_trajectory.ground_speed_bound

    @property
    def turn_rate_bound_deg_per_min(self) -> float:
        return self._base_trajectory.turn_rate_bound_deg_per_min

    def state_at(self, elapsed_min: float) -> WaterRelativeVesselState:
        return self._base_trajectory.state_at(elapsed_min)

    def state_at_or_continue(self, elapsed_min: float) -> WaterRelativeVesselState:
        return self._base_trajectory.state_at_or_continue(elapsed_min)

    def state_and_motion_at_or_continue(
        self, elapsed_min: float
    ) -> tuple[WaterRelativeVesselState, P4BBehaviorMotionState]:
        if isinstance(self._base_trajectory, StatefulEnvironmentalCommandTrajectory):
            state, base_motion_state = self._base_trajectory.state_and_motion_at_or_continue(
                elapsed_min
            )
            base_motion_state = validate_environmental_motion_state(
                self._backend.base_backend,
                base_motion_state,
                boundary="P4B delegated base trajectory continuation",
            )
        else:
            state = self._base_trajectory.state_at_or_continue(elapsed_min)
            advanced = self._backend.base_backend.advance(
                self.initial_state,
                self._motion_state.effective_command,
                max(0.0, float(elapsed_min)),
                self.limits,
                self.environmental_drift_velocity,
                self._motion_state.base_motion_state,
            )
            if advanced.advance.state != state:
                raise RuntimeError(
                    "delegated base trajectory and base sidecar continuation disagree"
                )
            base_motion_state = validate_environmental_motion_state(
                self._backend.base_backend,
                advanced.motion_state,
                boundary="P4B delegated fallback continuation",
            )
        wrapped = P4BBehaviorMotionState(
            base_motion_state=base_motion_state,
            effective_command=self._motion_state.effective_command,
            pending_requested_command=None,
            pending_age_min=0.0,
            binding_sha256=self._motion_state.binding_sha256,
        )
        validate_environmental_motion_state(
            self._backend,
            wrapped,
            boundary="P4B delegated trajectory continuation",
        )
        return state, wrapped

    def motion_at(self, elapsed_min: float) -> EnvironmentalVesselMotionEvidence:
        return self._base_trajectory.motion_at(elapsed_min)


class P4BBehaviorEnvironmentalMotionBackend:
    """Opt-in behavior-response composition over any validated environmental backend."""

    def __init__(
        self,
        base_backend: EnvironmentalMotionBackend,
        bindings: Mapping[str, P4BBehaviorBinding],
        *,
        default_prediction_horizon_min: float = 12.0,
    ) -> None:
        if not isinstance(base_backend, EnvironmentalMotionBackend):
            raise TypeError("base_backend must implement EnvironmentalMotionBackend")
        if not isinstance(bindings, Mapping):
            raise TypeError("bindings must be a mapping")
        frozen = dict(bindings)
        if any(not vessel_id for vessel_id in frozen):
            raise ValueError("binding vessel ids must be non-empty")
        if any(not isinstance(item, P4BBehaviorBinding) for item in frozen.values()):
            raise TypeError("bindings must contain P4BBehaviorBinding values")
        if not isfinite(default_prediction_horizon_min) or default_prediction_horizon_min <= 0.0:
            raise ValueError("default_prediction_horizon_min must be finite and positive")
        self.base_backend = base_backend
        self.backend_state_contract_id = _p4b_behavior_motion_state_contract_id(
            base_backend.backend_state_contract_id
        )
        self.bindings = MappingProxyType(frozen)
        self.default_prediction_horizon_min = float(default_prediction_horizon_min)
        # Exact implementation-only memoization for delayed behavior trajectories.
        # The wrapped response backend already memoizes no-delay trajectories, but the
        # delayed path must remain above that backend and would otherwise construct a
        # fresh P4BBehaviorCommandTrajectory (and therefore an empty time-state cache)
        # for every identical planner query.  Weak-value reuse keeps identical live
        # trajectories shared (notably across one retained pair-grid batch) without
        # extending the lifetime of their potentially large adaptive time-state caches.
        # Unsupported generic backend sidecars keep the historical uncached behavior.
        self._delayed_trajectory_cache: WeakValueDictionary[
            tuple[Any, ...], P4BBehaviorCommandTrajectory
        ] = WeakValueDictionary()
        payload = {
            "base_backend_id": base_backend.backend_id,
            "bindings": [
                [vessel_id, frozen[vessel_id].identity_sha256]
                for vessel_id in sorted(frozen)
            ],
            "default_prediction_horizon_min": self.default_prediction_horizon_min,
        }
        digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.backend_id = f"R1_P4B_BEHAVIOR_RESPONSE/{digest}"

    def binding_for(self, vessel_id: str) -> P4BBehaviorBinding:
        try:
            return self.bindings[vessel_id]
        except KeyError as exc:
            raise KeyError(f"no P4B behavior binding for vessel {vessel_id!r}") from exc

    def initialize_motion_state(
        self,
        state: WaterRelativeVesselState,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
    ) -> P4BBehaviorMotionState:
        binding = self.binding_for(state.vessel_id)
        base_state = validate_environmental_motion_state(
            self.base_backend,
            self.base_backend.initialize_motion_state(
                state, limits, environmental_drift_velocity
            ),
            boundary=f"P4B base initialize {state.vessel_id!r}",
        )
        hold = WaterRelativeManeuverCommand(
            target_through_water_heading_deg=state.through_water_heading_deg,
            target_through_water_speed=state.through_water_speed,
        )
        motion_state = P4BBehaviorMotionState(
            base_motion_state=base_state,
            effective_command=hold,
            pending_requested_command=None,
            pending_age_min=0.0,
            binding_sha256=binding.identity_sha256,
        )
        validate_environmental_motion_state(
            self,
            motion_state,
            boundary=f"P4B initialize {state.vessel_id!r}",
        )
        return motion_state

    def _validate_motion_state(
        self,
        state: WaterRelativeVesselState,
        motion_state: P4BBehaviorMotionState,
    ) -> P4BBehaviorBinding:
        validate_environmental_motion_state(
            self,
            motion_state,
            boundary=f"P4B behavior state {state.vessel_id!r}",
        )
        if not isinstance(motion_state, P4BBehaviorMotionState):
            raise TypeError("behavior backend requires P4BBehaviorMotionState")
        validate_environmental_motion_state(
            self.base_backend,
            motion_state.base_motion_state,
            boundary=f"P4B embedded base state {state.vessel_id!r}",
        )
        binding = self.binding_for(state.vessel_id)
        if motion_state.binding_sha256 != binding.identity_sha256:
            raise ValueError("behavior binding identity changed after motion-state initialization")
        return binding

    @staticmethod
    def _reconcile_request(
        requested: WaterRelativeManeuverCommand,
        motion_state: P4BBehaviorMotionState,
        *,
        delay_min: float,
    ) -> P4BBehaviorMotionState:
        if delay_min <= _EPS:
            return P4BBehaviorMotionState(
                motion_state.base_motion_state, requested, None, 0.0, motion_state.binding_sha256
            )
        if requested == motion_state.effective_command:
            return P4BBehaviorMotionState(
                motion_state.base_motion_state,
                motion_state.effective_command,
                None,
                0.0,
                motion_state.binding_sha256,
            )
        if requested == motion_state.pending_requested_command:
            return motion_state
        return P4BBehaviorMotionState(
            motion_state.base_motion_state,
            motion_state.effective_command,
            requested,
            0.0,
            motion_state.binding_sha256,
        )

    def _advance_from_frozen_state(
        self,
        state: WaterRelativeVesselState,
        requested: WaterRelativeManeuverCommand,
        duration_min: float,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
        motion_state: P4BBehaviorMotionState,
    ) -> EnvironmentalMotionAdvance:
        if not isfinite(duration_min) or duration_min < 0.0:
            raise ValueError("duration_min must be finite and non-negative")
        binding = self._validate_motion_state(state, motion_state)
        working = self._reconcile_request(requested, motion_state, delay_min=binding.command_delay_min)
        if duration_min <= _EPS:
            return EnvironmentalMotionAdvance(_sum_advance_results(state, ()), working)

        public = state
        parts: list[EnvironmentalAdvanceResult] = []
        remaining_duration = float(duration_min)

        if working.pending_requested_command is not None:
            remaining_delay = max(0.0, binding.command_delay_min - working.pending_age_min)
            hold_dt = min(remaining_duration, remaining_delay)
            if hold_dt > _EPS:
                base = self.base_backend.advance(
                    public,
                    working.effective_command,
                    hold_dt,
                    limits,
                    environmental_drift_velocity,
                    working.base_motion_state,
                )
                parts.append(base.advance)
                public = base.advance.state
                working = P4BBehaviorMotionState(
                    base.motion_state,
                    working.effective_command,
                    working.pending_requested_command,
                    working.pending_age_min + hold_dt,
                    working.binding_sha256,
                )
                remaining_duration -= hold_dt

            if working.pending_age_min >= binding.command_delay_min - _EPS:
                assert working.pending_requested_command is not None
                working = P4BBehaviorMotionState(
                    working.base_motion_state,
                    working.pending_requested_command,
                    None,
                    0.0,
                    working.binding_sha256,
                )

        if remaining_duration > _EPS:
            base = self.base_backend.advance(
                public,
                working.effective_command,
                remaining_duration,
                limits,
                environmental_drift_velocity,
                working.base_motion_state,
            )
            parts.append(base.advance)
            working = P4BBehaviorMotionState(
                base.motion_state,
                working.effective_command,
                working.pending_requested_command,
                working.pending_age_min,
                working.binding_sha256,
            )

        return EnvironmentalMotionAdvance(
            advance=_sum_advance_results(state, tuple(parts)),
            motion_state=working,
        )

    def advance(
        self,
        state: WaterRelativeVesselState,
        command: WaterRelativeManeuverCommand,
        duration_min: float,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
        motion_state: P4BBehaviorMotionState,
    ) -> EnvironmentalMotionAdvance:
        return self._advance_from_frozen_state(
            state, command, duration_min, limits, environmental_drift_velocity, motion_state
        )

    def build_trajectory(
        self,
        state: WaterRelativeVesselState,
        environmental_drift_velocity: Vector2,
        command: WaterRelativeManeuverCommand,
        limits: ManeuverLimits,
        motion_state: P4BBehaviorMotionState,
        *,
        evaluation_horizon_min: float | None = None,
    ) -> EnvironmentalCommandTrajectory:
        binding = self._validate_motion_state(state, motion_state)
        prepared = self._reconcile_request(command, motion_state, delay_min=binding.command_delay_min)
        if prepared.pending_requested_command is None:
            base_trajectory = self.base_backend.build_trajectory(
                state,
                environmental_drift_velocity,
                prepared.effective_command,
                limits,
                prepared.base_motion_state,
                evaluation_horizon_min=evaluation_horizon_min,
            )
            return P4BBehaviorDelegatedCommandTrajectory(
                _base_trajectory=base_trajectory,
                _backend=self,
                _motion_state=prepared,
            )

        horizon = self.default_prediction_horizon_min if evaluation_horizon_min is None else float(evaluation_horizon_min)
        if not isfinite(horizon) or horizon <= 0.0:
            raise ValueError("evaluation horizon must be finite and positive")
        drift = (float(environmental_drift_velocity[0]), float(environmental_drift_velocity[1]))
        cache_key = (state, drift, command, limits, prepared, horizon)
        try:
            hash(cache_key)
        except TypeError:
            cache_key = None
        if cache_key is not None:
            cached = self._delayed_trajectory_cache.get(cache_key)
            if cached is not None:
                return cached
        trajectory = P4BBehaviorCommandTrajectory(
            initial_state=state,
            environmental_drift_velocity=drift,
            command=command,
            limits=limits,
            transition_duration_min=horizon,
            start_motion=environmental_vessel_motion_evidence(state, drift),
            _backend=self,
            _motion_state=prepared,
        )
        if cache_key is not None:
            self._delayed_trajectory_cache[cache_key] = trajectory
        return trajectory
