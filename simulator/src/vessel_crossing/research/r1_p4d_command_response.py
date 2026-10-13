"""P4D bounded command-response latency over the validated plant backend.

P4B's frozen research adapter intentionally remains untouched for reproducibility.  Its
pending-command timer restarts whenever the requested command changes.  In a continuously
updated controller that behaves like a debounce and can starve command application.

P4D uses a different, explicitly versioned contract: the latency clock starts when the
requested command first diverges from the currently effective command.  While that clock
runs, later controller updates replace the pending target but do not erase elapsed response
time.  If the request returns to the effective command, the pending response is cancelled.
This is a bounded action-latency abstraction, not a human reaction-time distribution and
not a macroscopic headway.
"""
from __future__ import annotations

from hashlib import sha256

from vessel_crossing.nextgen.environmental_dynamics import WaterRelativeManeuverCommand
from vessel_crossing.research.r1_p4b_behavior_adapter import (
    P4BBehaviorEnvironmentalMotionBackend,
    P4BBehaviorMotionState,
)

_EPS = 1e-12
P4D_COMMAND_RESPONSE_SEMANTICS_ID = "R1_P4D_BOUNDED_ACTION_LATENCY_V1"


class P4DBoundedCommandResponseEnvironmentalMotionBackend(P4BBehaviorEnvironmentalMotionBackend):
    """Behavior backend with non-resetting elapsed latency under pending-target updates."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        digest = sha256(
            f"{P4D_COMMAND_RESPONSE_SEMANTICS_ID}|{self.backend_id}".encode("utf-8")
        ).hexdigest()
        self.backend_id = f"{P4D_COMMAND_RESPONSE_SEMANTICS_ID}/{digest}"

    @staticmethod
    def _reconcile_request(
        requested: WaterRelativeManeuverCommand,
        motion_state: P4BBehaviorMotionState,
        *,
        delay_min: float,
    ) -> P4BBehaviorMotionState:
        if delay_min <= _EPS:
            return P4BBehaviorMotionState(
                motion_state.base_motion_state,
                requested,
                None,
                0.0,
                motion_state.binding_sha256,
            )
        if requested == motion_state.effective_command:
            # Controller returned to the command already being executed: no delayed
            # action remains to be taken.
            return P4BBehaviorMotionState(
                motion_state.base_motion_state,
                motion_state.effective_command,
                None,
                0.0,
                motion_state.binding_sha256,
            )
        if motion_state.pending_requested_command is None:
            return P4BBehaviorMotionState(
                motion_state.base_motion_state,
                motion_state.effective_command,
                requested,
                0.0,
                motion_state.binding_sha256,
            )
        if requested == motion_state.pending_requested_command:
            return motion_state
        # Update the target that will be acted on at the already-running response
        # deadline.  Crucially, do not reset pending_age_min.
        return P4BBehaviorMotionState(
            motion_state.base_motion_state,
            motion_state.effective_command,
            requested,
            motion_state.pending_age_min,
            motion_state.binding_sha256,
        )
