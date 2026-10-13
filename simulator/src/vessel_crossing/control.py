"""Adapter between immutable policies and the stateful simulation engine."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isinf

from .policies import (
    ControlPolicy,
    EncounterObserver,
    PolicySnapshot,
    SpeedCommand,
    validate_speed_commands,
)
from .safety import RiskReason, time_to_exit_circular_domain
from .vessel import Vessel


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Auditable result of one policy call at one simulation instant."""

    snapshot: PolicySnapshot
    commands: tuple[SpeedCommand, ...]


@dataclass(frozen=True, slots=True)
class PolicyController:
    """Pure adapter: observe -> decide -> validate.

    It intentionally has no reference to ``Simulation`` and cannot apply its
    own commands.  The stateful engine remains responsible for execution and
    event rescheduling.
    """

    observer: EncounterObserver
    policy: ControlPolicy

    def evaluate(
        self,
        *,
        time: float,
        vessels: tuple[Vessel, ...] | list[Vessel],
    ) -> PolicyEvaluation:
        snapshot = self.observer.build_snapshot(time=time, vessels=vessels)
        commands = tuple(self.policy.decide(snapshot))
        commands = validate_speed_commands(snapshot, commands)
        return PolicyEvaluation(snapshot=snapshot, commands=commands)

    def next_horizon_review_delay(self, evaluation: PolicyEvaluation) -> float:
        """Return earliest analytical delay until an unsafe CPA enters horizon.

        Only encounters currently classified ``BEYOND_HORIZON`` whose DCPA is
        at or inside the configured domain need a future horizon-entry review.
        Returns ``math.inf`` when no such review is required.
        """

        horizon = self.observer.safety_domain.tcpa_horizon
        if horizon is None:
            return inf

        best = inf
        for encounter in evaluation.snapshot.encounters:
            assessment = encounter.assessment
            if assessment.reason is not RiskReason.BEYOND_HORIZON:
                continue
            cpa = assessment.closest_approach
            if cpa.dcpa > self.observer.safety_domain.min_dcpa:
                continue
            delay = cpa.raw_tcpa - horizon
            if delay > 0:
                best = min(best, delay)
        return best

    def next_danger_review_delay(self, evaluation: PolicyEvaluation) -> float:
        """Return next useful review time for a currently dangerous encounter."""

        radius = self.observer.safety_domain.min_dcpa
        best = inf
        for encounter in evaluation.snapshot.dangerous_encounters:
            cpa = encounter.assessment.closest_approach
            if encounter.assessment.reason is RiskReason.PREDICTED_CONFLICT:
                if cpa.raw_tcpa > 1e-10:
                    best = min(best, cpa.raw_tcpa)
                continue

            if encounter.assessment.reason is RiskReason.CURRENT_DOMAIN_VIOLATION:
                first = evaluation.snapshot.vessel(encounter.first_vessel_id)
                second = evaluation.snapshot.vessel(encounter.second_vessel_id)
                exit_delay = time_to_exit_circular_domain(
                    first.motion_state,
                    second.motion_state,
                    radius,
                )
                best = min(best, exit_delay)

        return best

    def next_recovery_review_delay(self, evaluation: PolicyEvaluation) -> float:
        """Return earliest CPA boundary for a currently reduced vessel.

        A speed reduction can turn a predicted conflict into a safe miss before
        the vessels have actually passed one another.  Policies intentionally
        keep that reduction installed through closest approach.  This review
        lets the engine reconsider recovery at the analytical CPA instant rather
        than waiting for an unrelated completion event or polling in fixed steps.
        """

        best = inf
        for encounter in evaluation.snapshot.encounters:
            cpa = encounter.assessment.closest_approach
            if isinf(cpa.raw_tcpa) or cpa.raw_tcpa <= 1e-10:
                continue
            first = evaluation.snapshot.vessel(encounter.first_vessel_id)
            second = evaluation.snapshot.vessel(encounter.second_vessel_id)
            first_target = first.desired_speed if first.target_speed is None else first.target_speed
            second_target = second.desired_speed if second.target_speed is None else second.target_speed
            if (
                first_target < first.desired_speed - 1e-12
                or second_target < second.desired_speed - 1e-12
            ):
                best = min(best, cpa.raw_tcpa)
        return best
