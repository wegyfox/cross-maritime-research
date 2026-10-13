"""C9 degraded/fallback operational state machine.

C9 is intentionally separate from C8 evaluation and from controller/plant code.  It
consumes explicit operational signals, records deterministic transition causes and
assigns authority ownership.  The state machine itself is generic infrastructure and
is not a novelty, COLREG, ISO, or certification claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Iterable, Mapping

from .odd import ODDEvaluation, ODDStatus
from .scope import TestbedFeature

_EPS = 1e-12


class OperationalState(str, Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    ODD_MARGIN = "ODD_MARGIN"
    ODD_EXIT = "ODD_EXIT"
    FALLBACK = "FALLBACK"
    REMOTE_TAKEOVER = "REMOTE_TAKEOVER"
    SAFE_STATE = "SAFE_STATE"


class AuthorityOwner(str, Enum):
    AUTONOMOUS_CONTROLLER = "AUTONOMOUS_CONTROLLER"
    FALLBACK_CONTROLLER = "FALLBACK_CONTROLLER"
    REMOTE_OPERATOR = "REMOTE_OPERATOR"
    SAFE_STATE_CONTROLLER = "SAFE_STATE_CONTROLLER"


class TransitionCause(str, Enum):
    DEGRADATION_ENTERED = "DEGRADATION_ENTERED"
    DEGRADATION_CLEARED = "DEGRADATION_CLEARED"
    ODD_MARGIN_ENTERED = "ODD_MARGIN_ENTERED"
    ODD_MARGIN_CLEARED = "ODD_MARGIN_CLEARED"
    ODD_EXIT_DETECTED = "ODD_EXIT_DETECTED"
    ODD_EXIT_FALLBACK = "ODD_EXIT_FALLBACK"
    FALLBACK_REQUESTED = "FALLBACK_REQUESTED"
    REMOTE_TAKEOVER_ACTIVATED = "REMOTE_TAKEOVER_ACTIVATED"
    REMOTE_TAKEOVER_TIMEOUT = "REMOTE_TAKEOVER_TIMEOUT"
    NO_ADMISSIBLE_ACTION = "NO_ADMISSIBLE_ACTION"
    SAFE_STATE_REACHED = "SAFE_STATE_REACHED"


_AUTHORITY_BY_STATE = {
    OperationalState.NORMAL: AuthorityOwner.AUTONOMOUS_CONTROLLER,
    OperationalState.DEGRADED: AuthorityOwner.AUTONOMOUS_CONTROLLER,
    OperationalState.ODD_MARGIN: AuthorityOwner.AUTONOMOUS_CONTROLLER,
    OperationalState.ODD_EXIT: AuthorityOwner.AUTONOMOUS_CONTROLLER,
    OperationalState.FALLBACK: AuthorityOwner.FALLBACK_CONTROLLER,
    OperationalState.REMOTE_TAKEOVER: AuthorityOwner.REMOTE_OPERATOR,
    OperationalState.SAFE_STATE: AuthorityOwner.SAFE_STATE_CONTROLLER,
}


@dataclass(frozen=True, slots=True)
class OperationalSignals:
    time_min: float
    odd_status: ODDStatus = ODDStatus.INSIDE
    degradation_active: bool = False
    fallback_requested: bool = False
    remote_takeover_acknowledged: bool = False
    admissible_action_available: bool = True
    safe_state_reached: bool = False
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.time_min) or self.time_min < 0.0:
            raise ValueError("time_min must be finite and non-negative")
        ids = tuple(self.evidence_ids)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("evidence_ids must be unique and sorted")

    @classmethod
    def from_odd_evaluation(
        cls,
        evaluation: ODDEvaluation,
        *,
        time_min: float,
        degradation_active: bool = False,
        fallback_requested: bool = False,
        remote_takeover_acknowledged: bool = False,
        admissible_action_available: bool = True,
        safe_state_reached: bool = False,
        extra_evidence_ids: Iterable[str] = (),
    ) -> "OperationalSignals":
        ids = {
            f"odd-context:{evaluation.context_sha256}",
            f"odd-envelope:{evaluation.envelope_sha256}",
            *tuple(extra_evidence_ids),
        }
        return cls(
            time_min=float(time_min),
            odd_status=evaluation.status,
            degradation_active=bool(degradation_active),
            fallback_requested=bool(fallback_requested),
            remote_takeover_acknowledged=bool(remote_takeover_acknowledged),
            admissible_action_available=bool(admissible_action_available),
            safe_state_reached=bool(safe_state_reached),
            evidence_ids=tuple(sorted(ids)),
        )


@dataclass(frozen=True, slots=True)
class OperationalStateMachineConfig:
    fallback_delay_after_odd_exit_min: float = 0.0
    remote_takeover_enabled: bool = False
    remote_ack_timeout_min: float | None = None
    takeover_activation_delay_min: float = 0.0
    config_id: str = "cross-c9/default-state-machine/v1"

    def __post_init__(self) -> None:
        if not self.config_id:
            raise ValueError("config_id must be non-empty")
        if not isfinite(self.fallback_delay_after_odd_exit_min) or self.fallback_delay_after_odd_exit_min < 0.0:
            raise ValueError("fallback_delay_after_odd_exit_min must be finite and non-negative")
        if not isfinite(self.takeover_activation_delay_min) or self.takeover_activation_delay_min < 0.0:
            raise ValueError("takeover_activation_delay_min must be finite and non-negative")
        if self.remote_ack_timeout_min is not None:
            if not isfinite(self.remote_ack_timeout_min) or self.remote_ack_timeout_min <= 0.0:
                raise ValueError("remote_ack_timeout_min must be finite and positive when set")
            if not self.remote_takeover_enabled:
                raise ValueError("remote_ack_timeout_min requires remote_takeover_enabled")

    def configuration_sha256(self) -> str:
        return _sha256_payload(
            {
                "config_id": self.config_id,
                "fallback_delay_after_odd_exit_min": self.fallback_delay_after_odd_exit_min,
                "remote_takeover_enabled": self.remote_takeover_enabled,
                "remote_ack_timeout_min": self.remote_ack_timeout_min,
                "takeover_activation_delay_min": self.takeover_activation_delay_min,
            }
        )


@dataclass(frozen=True, slots=True)
class OperationalTransition:
    time_min: float
    from_state: OperationalState
    to_state: OperationalState
    cause: TransitionCause
    authority_before: AuthorityOwner
    authority_after: AuthorityOwner
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.time_min) or self.time_min < 0.0:
            raise ValueError("transition time_min must be finite and non-negative")
        if self.from_state is self.to_state:
            raise ValueError("transition must change state")
        ids = tuple(self.evidence_ids)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("transition evidence_ids must be unique and sorted")

    def identity_sha256(self) -> str:
        return _sha256_payload(
            {
                "time_min": self.time_min,
                "from_state": self.from_state.value,
                "to_state": self.to_state.value,
                "cause": self.cause.value,
                "authority_before": self.authority_before.value,
                "authority_after": self.authority_after.value,
                "evidence_ids": list(self.evidence_ids),
            }
        )


@dataclass(frozen=True, slots=True)
class OperationalStateSnapshot:
    time_min: float
    state: OperationalState
    authority_owner: AuthorityOwner
    state_entered_time_min: float
    takeover_request_time_min: float | None
    takeover_ack_time_min: float | None
    latest_odd_status: ODDStatus
    degradation_active: bool
    transition_count: int
    transition_history_sha256: str
    config_sha256: str
    component_id: TestbedFeature = TestbedFeature.C9_FALLBACK_STATE_MACHINE


class OperationalStateMachine:
    """Deterministic stateful C9 transition engine.

    Once FALLBACK is reached, automatic recovery to autonomous authority is deliberately
    not performed.  Returning authority after a fallback/takeover episode requires an
    explicit future reset/reinitialization protocol rather than an implicit status flip.
    """

    component_id = TestbedFeature.C9_FALLBACK_STATE_MACHINE

    def __init__(self, config: OperationalStateMachineConfig | None = None) -> None:
        self.config = config or OperationalStateMachineConfig()
        self.state = OperationalState.NORMAL
        self.state_entered_time_min = 0.0
        self.last_time_min: float | None = None
        self.takeover_request_time_min: float | None = None
        self.takeover_ack_time_min: float | None = None
        self.latest_odd_status = ODDStatus.INSIDE
        self.degradation_active = False
        self._transitions: list[OperationalTransition] = []

    @property
    def authority_owner(self) -> AuthorityOwner:
        return _AUTHORITY_BY_STATE[self.state]

    @property
    def transitions(self) -> tuple[OperationalTransition, ...]:
        return tuple(self._transitions)

    def update(self, signals: OperationalSignals) -> OperationalStateSnapshot:
        if self.last_time_min is not None and signals.time_min + _EPS < self.last_time_min:
            raise ValueError("operational state machine requires monotonic time")
        self.last_time_min = signals.time_min
        self.latest_odd_status = signals.odd_status
        self.degradation_active = signals.degradation_active

        # Fail closed regardless of current state or remote-takeover configuration.
        if not signals.admissible_action_available:
            self._transition_if_needed(
                OperationalState.SAFE_STATE, TransitionCause.NO_ADMISSIBLE_ACTION, signals
            )
            return self.snapshot(signals.time_min)
        if signals.safe_state_reached:
            self._transition_if_needed(
                OperationalState.SAFE_STATE, TransitionCause.SAFE_STATE_REACHED, signals
            )
            return self.snapshot(signals.time_min)

        if self.state is OperationalState.SAFE_STATE:
            return self.snapshot(signals.time_min)
        if self.state is OperationalState.REMOTE_TAKEOVER:
            return self.snapshot(signals.time_min)

        if self.state is OperationalState.FALLBACK:
            self._advance_remote_takeover(signals)
            return self.snapshot(signals.time_min)

        if self.state is OperationalState.ODD_EXIT:
            if signals.fallback_requested or (
                signals.time_min - self.state_entered_time_min
                >= self.config.fallback_delay_after_odd_exit_min - _EPS
            ):
                cause = (
                    TransitionCause.FALLBACK_REQUESTED
                    if signals.fallback_requested
                    else TransitionCause.ODD_EXIT_FALLBACK
                )
                self._enter_fallback(cause, signals)
                self._advance_remote_takeover(signals)
            return self.snapshot(signals.time_min)

        if signals.fallback_requested:
            self._enter_fallback(TransitionCause.FALLBACK_REQUESTED, signals)
            self._advance_remote_takeover(signals)
            return self.snapshot(signals.time_min)

        if signals.odd_status is ODDStatus.OUTSIDE:
            self._transition_if_needed(
                OperationalState.ODD_EXIT, TransitionCause.ODD_EXIT_DETECTED, signals
            )
            if self.config.fallback_delay_after_odd_exit_min <= _EPS:
                self._enter_fallback(TransitionCause.ODD_EXIT_FALLBACK, signals)
                self._advance_remote_takeover(signals)
            return self.snapshot(signals.time_min)

        if signals.odd_status is ODDStatus.MARGIN:
            self._transition_if_needed(
                OperationalState.ODD_MARGIN, TransitionCause.ODD_MARGIN_ENTERED, signals
            )
            return self.snapshot(signals.time_min)

        # ODD is inside.  Before fallback, degradation is the remaining state axis.
        if signals.degradation_active:
            cause = (
                TransitionCause.ODD_MARGIN_CLEARED
                if self.state is OperationalState.ODD_MARGIN
                else TransitionCause.DEGRADATION_ENTERED
            )
            self._transition_if_needed(OperationalState.DEGRADED, cause, signals)
        else:
            cause = (
                TransitionCause.ODD_MARGIN_CLEARED
                if self.state is OperationalState.ODD_MARGIN
                else TransitionCause.DEGRADATION_CLEARED
            )
            self._transition_if_needed(OperationalState.NORMAL, cause, signals)
        return self.snapshot(signals.time_min)

    def snapshot(self, time_min: float | None = None) -> OperationalStateSnapshot:
        if time_min is None:
            time_min = self.last_time_min if self.last_time_min is not None else 0.0
        history_hash = _sha256_payload(
            {"transitions": [transition.identity_sha256() for transition in self._transitions]}
        )
        return OperationalStateSnapshot(
            time_min=float(time_min),
            state=self.state,
            authority_owner=self.authority_owner,
            state_entered_time_min=self.state_entered_time_min,
            takeover_request_time_min=self.takeover_request_time_min,
            takeover_ack_time_min=self.takeover_ack_time_min,
            latest_odd_status=self.latest_odd_status,
            degradation_active=self.degradation_active,
            transition_count=len(self._transitions),
            transition_history_sha256=history_hash,
            config_sha256=self.config.configuration_sha256(),
        )

    def _enter_fallback(self, cause: TransitionCause, signals: OperationalSignals) -> None:
        self._transition_if_needed(OperationalState.FALLBACK, cause, signals)
        if self.config.remote_takeover_enabled and self.takeover_request_time_min is None:
            self.takeover_request_time_min = signals.time_min

    def _advance_remote_takeover(self, signals: OperationalSignals) -> None:
        if self.state is not OperationalState.FALLBACK or not self.config.remote_takeover_enabled:
            return
        if self.takeover_request_time_min is None:
            self.takeover_request_time_min = signals.time_min
        if signals.remote_takeover_acknowledged and self.takeover_ack_time_min is None:
            self.takeover_ack_time_min = signals.time_min
        if self.takeover_ack_time_min is not None:
            if signals.time_min - self.takeover_ack_time_min >= self.config.takeover_activation_delay_min - _EPS:
                self._transition_if_needed(
                    OperationalState.REMOTE_TAKEOVER,
                    TransitionCause.REMOTE_TAKEOVER_ACTIVATED,
                    signals,
                )
                return
        if (
            self.takeover_ack_time_min is None
            and self.config.remote_ack_timeout_min is not None
            and signals.time_min - self.takeover_request_time_min >= self.config.remote_ack_timeout_min - _EPS
        ):
            self._transition_if_needed(
                OperationalState.SAFE_STATE,
                TransitionCause.REMOTE_TAKEOVER_TIMEOUT,
                signals,
            )

    def _transition_if_needed(
        self,
        target: OperationalState,
        cause: TransitionCause,
        signals: OperationalSignals,
    ) -> None:
        if self.state is target:
            return
        before = self.state
        authority_before = self.authority_owner
        self.state = target
        self.state_entered_time_min = signals.time_min
        self._transitions.append(
            OperationalTransition(
                time_min=signals.time_min,
                from_state=before,
                to_state=target,
                cause=cause,
                authority_before=authority_before,
                authority_after=self.authority_owner,
                evidence_ids=signals.evidence_ids,
            )
        )


@dataclass(frozen=True, slots=True)
class NoOpOperationalStateMachine:
    """Pre-C9 compatibility mode: no authority/state intervention."""

    machine_id: str = "cross-c9/noop-state-machine/v1"
    component_id = TestbedFeature.C9_FALLBACK_STATE_MACHINE

    def update(self, signals: OperationalSignals) -> OperationalStateSnapshot:
        return OperationalStateSnapshot(
            time_min=signals.time_min,
            state=OperationalState.NORMAL,
            authority_owner=AuthorityOwner.AUTONOMOUS_CONTROLLER,
            state_entered_time_min=0.0,
            takeover_request_time_min=None,
            takeover_ack_time_min=None,
            latest_odd_status=signals.odd_status,
            degradation_active=signals.degradation_active,
            transition_count=0,
            transition_history_sha256=_sha256_payload({"machine_id": self.machine_id, "history": []}),
            config_sha256=_sha256_payload({"machine_id": self.machine_id, "mode": "NOOP"}),
        )


def _sha256_payload(payload: Mapping[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
