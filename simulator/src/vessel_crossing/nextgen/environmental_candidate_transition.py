"""One-vessel bounded environmental candidate transition evidence.

A84 is deliberately below pair/joint safety screening. It evaluates how one explicit
water-relative candidate command evolves under bounded turn/acceleration dynamics plus
one fixed Earth-frame drift vector. The result keeps hull/water-relative control state
and ground-motion evidence separate and provides exact state queries over the transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Protocol, runtime_checkable

from ..kinematics import Vector2
from .dynamics import ManeuverLimits
from .environmental_dynamics import (
    EnvironmentalAdvanceResult,
    WaterRelativeManeuverCommand,
    WaterRelativeVesselState,
    advance_with_uniform_drift,
    water_relative_command_transition_duration,
)
from .environmental_evidence import (
    EnvironmentalVesselMotionEvidence,
    environmental_vessel_motion_evidence,
)

_EPS = 1e-12


NO_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID = "CROSS_ENVIRONMENTAL_MOTION_STATE/NONE/V1"


class EnvironmentalMotionStateOwnershipError(TypeError):
    """Raised when a motion sidecar is composed with the wrong backend contract."""


@runtime_checkable
class EnvironmentalMotionState(Protocol):
    """Structural ownership marker for persistent environmental backend sidecars."""

    @property
    def backend_state_contract_id(self) -> str: ...


def environmental_motion_state_contract_id(
    motion_state: EnvironmentalMotionState | None,
) -> str:
    """Return the immutable ownership contract carried by one backend sidecar."""

    if motion_state is None:
        return NO_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID
    if not isinstance(motion_state, EnvironmentalMotionState):
        raise EnvironmentalMotionStateOwnershipError(
            "motion_state must implement EnvironmentalMotionState or be None"
        )
    contract_id = motion_state.backend_state_contract_id
    if not isinstance(contract_id, str) or not contract_id:
        raise EnvironmentalMotionStateOwnershipError(
            "motion_state backend_state_contract_id must be a non-empty string"
        )
    return contract_id


@runtime_checkable
class EnvironmentalCommandTrajectory(Protocol):
    """Structural A84/A85 trajectory contract for one explicit command.

    The historical :class:`EnvironmentalCandidateTransition` is the default PARITY
    implementation.  Richer Stage-2 plants may implement this protocol without adding
    fields to the historical dataclass or changing its canonical/evidence shape.
    """

    initial_state: WaterRelativeVesselState
    environmental_drift_velocity: Vector2
    command: WaterRelativeManeuverCommand
    limits: ManeuverLimits
    transition_duration_min: float
    start_motion: EnvironmentalVesselMotionEvidence

    @property
    def reachable(self) -> bool: ...

    @property
    def final_state(self) -> WaterRelativeVesselState | None: ...

    @property
    def ground_speed_bound(self) -> float: ...

    @property
    def turn_rate_bound_deg_per_min(self) -> float: ...

    def state_at(self, elapsed_min: float) -> WaterRelativeVesselState: ...

    def state_at_or_continue(self, elapsed_min: float) -> WaterRelativeVesselState: ...

    def motion_at(self, elapsed_min: float) -> EnvironmentalVesselMotionEvidence: ...


@runtime_checkable
class StatefulEnvironmentalCommandTrajectory(Protocol):
    """Explicit richer trajectory capability returning state and backend sidecar together.

    Consumers already hold an :class:`EnvironmentalCommandTrajectory`; this protocol
    makes the optional state+motion continuation seam explicit and runtime-checkable
    instead of discovering it through ad-hoc ``getattr`` calls.  Refactor-1/M0-03
    binds every returned sidecar to an explicit backend state-contract identity.
    """

    def state_and_motion_at_or_continue(
        self, elapsed_min: float
    ) -> tuple[WaterRelativeVesselState, EnvironmentalMotionState | None]: ...


@dataclass(frozen=True, slots=True)
class EnvironmentalMotionAdvance:
    """One backend world-advance result plus contract-owned persistent plant state."""

    advance: EnvironmentalAdvanceResult
    motion_state: EnvironmentalMotionState | None = None


@runtime_checkable
class EnvironmentalMotionBackend(Protocol):
    """Shared execution/prediction backend contract for environmental dynamics.

    ``EnvironmentalWorld2D`` owns the contract-bound per-vessel ``motion_state`` returned by
    this backend. Planning captures that state and A84 asks the same backend for a
    deterministic candidate trajectory. The historical parity backend uses no sidecar.
    """

    backend_id: str
    backend_state_contract_id: str

    def initialize_motion_state(
        self,
        state: WaterRelativeVesselState,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
    ) -> EnvironmentalMotionState | None: ...

    def advance(
        self,
        state: WaterRelativeVesselState,
        command: WaterRelativeManeuverCommand,
        duration_min: float,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
        motion_state: EnvironmentalMotionState | None,
    ) -> EnvironmentalMotionAdvance: ...

    def build_trajectory(
        self,
        state: WaterRelativeVesselState,
        environmental_drift_velocity: Vector2,
        command: WaterRelativeManeuverCommand,
        limits: ManeuverLimits,
        motion_state: EnvironmentalMotionState | None,
        *,
        evaluation_horizon_min: float | None = None,
    ) -> EnvironmentalCommandTrajectory: ...


def validate_environmental_motion_state(
    backend: EnvironmentalMotionBackend,
    motion_state: EnvironmentalMotionState | None,
    *,
    boundary: str = "environmental motion boundary",
) -> EnvironmentalMotionState | None:
    """Fail fast when a sidecar is owned by a different backend state contract."""

    if not isinstance(backend, EnvironmentalMotionBackend):
        raise TypeError("backend must implement EnvironmentalMotionBackend")
    expected = backend.backend_state_contract_id
    if not isinstance(expected, str) or not expected:
        raise EnvironmentalMotionStateOwnershipError(
            f"{boundary}: backend_state_contract_id must be a non-empty string"
        )
    observed = environmental_motion_state_contract_id(motion_state)
    if observed != expected:
        raise EnvironmentalMotionStateOwnershipError(
            f"{boundary}: motion-state contract mismatch: expected {expected!r}, observed {observed!r}"
        )
    return motion_state


@dataclass(frozen=True, slots=True)
class EnvironmentalCandidateTransition:
    """Exact one-vessel transition to a water-relative candidate command."""

    initial_state: WaterRelativeVesselState
    environmental_drift_velocity: Vector2
    command: WaterRelativeManeuverCommand
    limits: ManeuverLimits
    transition_duration_min: float
    start_motion: EnvironmentalVesselMotionEvidence
    advance: EnvironmentalAdvanceResult | None
    final_motion: EnvironmentalVesselMotionEvidence | None

    @property
    def reachable(self) -> bool:
        return isfinite(self.transition_duration_min)

    @property
    def final_state(self) -> WaterRelativeVesselState | None:
        return None if self.advance is None else self.advance.state

    @property
    def through_water_speed_bound(self) -> float:
        return max(
            self.initial_state.through_water_speed,
            self.command.target_through_water_speed,
        )

    @property
    def ground_speed_bound(self) -> float:
        """Conservative ground-speed magnitude bound over the transition."""

        dx, dy = self.environmental_drift_velocity
        return self.through_water_speed_bound + hypot(dx, dy)

    @property
    def turn_rate_bound_deg_per_min(self) -> float:
        """Conservative hull-heading rate bound used by oriented-domain certification."""

        return self.limits.max_turn_rate_deg_per_min

    def state_at(self, elapsed_min: float) -> WaterRelativeVesselState:
        """Return exact state inside the reachable transition interval."""

        if not self.reachable:
            raise ValueError("candidate transition is physically unreachable")
        if elapsed_min < -_EPS or elapsed_min > self.transition_duration_min + _EPS:
            raise ValueError("elapsed_min must lie inside the candidate transition")
        elapsed = min(max(float(elapsed_min), 0.0), self.transition_duration_min)
        return advance_with_uniform_drift(
            self.initial_state,
            self.command,
            elapsed,
            self.limits,
            environmental_drift_velocity=self.environmental_drift_velocity,
        ).state

    def state_at_or_continue(self, elapsed_min: float) -> WaterRelativeVesselState:
        """Return historical parity state for any non-negative elapsed time.

        A85 historically continued the same bounded command after the nominal transition
        completion.  Keeping that behavior on the default implementation lets pair-level
        code depend on the trajectory contract without changing old numerical evidence.
        """

        if not self.reachable:
            raise ValueError("candidate transition is physically unreachable")
        if elapsed_min < -_EPS:
            raise ValueError("elapsed_min must be non-negative")
        return advance_with_uniform_drift(
            self.initial_state,
            self.command,
            max(float(elapsed_min), 0.0),
            self.limits,
            environmental_drift_velocity=self.environmental_drift_velocity,
        ).state

    def motion_at(self, elapsed_min: float) -> EnvironmentalVesselMotionEvidence:
        state = self.state_at(elapsed_min)
        return environmental_vessel_motion_evidence(
            state,
            self.environmental_drift_velocity,
        )


def _build_parity_environmental_candidate_transition(
    state: WaterRelativeVesselState,
    environmental_drift_velocity: Vector2,
    command: WaterRelativeManeuverCommand,
    limits: ManeuverLimits,
) -> EnvironmentalCandidateTransition:
    """Build the historical A84 bounded-command transition exactly."""

    duration = water_relative_command_transition_duration(state, command, limits)
    start_motion = environmental_vessel_motion_evidence(
        state,
        environmental_drift_velocity,
    )
    if not isfinite(duration):
        return EnvironmentalCandidateTransition(
            state,
            start_motion.environmental_drift_velocity,
            command,
            limits,
            duration,
            start_motion,
            None,
            None,
        )

    advance = advance_with_uniform_drift(
        state,
        command,
        duration,
        limits,
        environmental_drift_velocity=start_motion.environmental_drift_velocity,
    )
    final_motion = environmental_vessel_motion_evidence(
        advance.state,
        start_motion.environmental_drift_velocity,
    )
    return EnvironmentalCandidateTransition(
        state,
        start_motion.environmental_drift_velocity,
        command,
        limits,
        duration,
        start_motion,
        advance,
        final_motion,
    )

class ParityEnvironmentalMotionBackend:
    """Historical environmental motion backend with no persistent sidecar state."""

    backend_id = "A162_ENVIRONMENTAL_PARITY"
    backend_state_contract_id = NO_ENVIRONMENTAL_MOTION_STATE_CONTRACT_ID

    def initialize_motion_state(
        self,
        state: WaterRelativeVesselState,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
    ) -> None:
        return None

    def advance(
        self,
        state: WaterRelativeVesselState,
        command: WaterRelativeManeuverCommand,
        duration_min: float,
        limits: ManeuverLimits,
        environmental_drift_velocity: Vector2,
        motion_state: EnvironmentalMotionState | None,
    ) -> EnvironmentalMotionAdvance:
        validate_environmental_motion_state(
            self, motion_state, boundary="parity backend advance"
        )
        return EnvironmentalMotionAdvance(
            advance_with_uniform_drift(
                state,
                command,
                duration_min,
                limits,
                environmental_drift_velocity=environmental_drift_velocity,
            ),
            None,
        )

    def build_trajectory(
        self,
        state: WaterRelativeVesselState,
        environmental_drift_velocity: Vector2,
        command: WaterRelativeManeuverCommand,
        limits: ManeuverLimits,
        motion_state: EnvironmentalMotionState | None,
        *,
        evaluation_horizon_min: float | None = None,
    ) -> EnvironmentalCommandTrajectory:
        validate_environmental_motion_state(
            self, motion_state, boundary="parity backend trajectory"
        )
        return _build_parity_environmental_candidate_transition(
            state, environmental_drift_velocity, command, limits
        )


PARITY_ENVIRONMENTAL_MOTION_BACKEND = ParityEnvironmentalMotionBackend()


def build_environmental_candidate_transition(
    state: WaterRelativeVesselState,
    environmental_drift_velocity: Vector2,
    command: WaterRelativeManeuverCommand,
    limits: ManeuverLimits,
    *,
    motion_backend: EnvironmentalMotionBackend | None = None,
    motion_state: EnvironmentalMotionState | None = None,
    evaluation_horizon_min: float | None = None,
) -> EnvironmentalCommandTrajectory:
    """Build one candidate trajectory through the selected environmental backend.

    The omitted/default backend is exactly the historical A84 implementation.  Richer
    backends are explicit opt-ins and receive the current contract-owned world sidecar state.
    """

    backend = PARITY_ENVIRONMENTAL_MOTION_BACKEND if motion_backend is None else motion_backend
    if not isinstance(backend, EnvironmentalMotionBackend):
        raise TypeError("motion_backend must implement EnvironmentalMotionBackend")
    validate_environmental_motion_state(
        backend, motion_state, boundary="candidate trajectory input"
    )
    return backend.build_trajectory(
        state,
        environmental_drift_velocity,
        command,
        limits,
        motion_state,
        evaluation_horizon_min=evaluation_horizon_min,
    )

