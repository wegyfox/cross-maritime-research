"""Persistent mixed-stage environmental fleet orchestration over A119/A120/A121.

A122 removes the caller-side choreography between already-defined ownership layers.  It
does not introduce a new planner, registry rule, route controller or physical clock.
Instead one wrapper repeatedly runs the existing A119 mixed scheduler and consumes its
explicit ownership boundaries in-place:

- ``RECAPTURE_REQUIRED`` -> A120 atomic mixed-stage rebootstrap;
- newly released avoidance channels -> immediate fleet-released route admission from
  the same committed A119 observation and exact A120 command tuple;
- ``ROUTE_DISPOSITION_REQUIRED`` -> A121 fleet-safe route-end disposition;
- terminal/end-time/override results -> return to the caller.

No boundary is re-observed at the same timestamp.  The same A117/A119 shared
``EnvironmentalWorld2D`` clock and canonical registry evidence remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping

from .environmental_fleet_multi_owner_recapture import EnvironmentalFleetMultiOwnerRecaptureConfig
from .environmental_mixed_owner_control import (
    EnvironmentalMixedOwnerControlError,
    EnvironmentalMixedOwnerControlResult,
    EnvironmentalMixedOwnerControlSession,
    EnvironmentalMixedOwnerStopReason,
    EnvironmentalMixedRouteStage,
)
from .environmental_mixed_owner_rebootstrap import (
    EnvironmentalMixedOwnerRebootstrapResult,
    _affected_seed_vessels,
    _replacement_owners,
    rebootstrap_environmental_mixed_owner_control,
)
from .environmental_route_end_disposition import (
    EnvironmentalRouteEndDispositionResult,
    apply_environmental_route_end_dispositions,
)
from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
from .vessel_classes import VesselProfileCatalog


class EnvironmentalPersistentMixedControlError(RuntimeError):
    """Base error for invalid A122 persistent mixed-stage orchestration."""


class EnvironmentalPersistentMixedStopReason(str, Enum):
    END_TIME = "end_time"
    COMMAND_OVERRIDDEN = "command_overridden"
    ALL_CONTROL_COMPLETE = "all_control_complete"


@dataclass(frozen=True, slots=True)
class EnvironmentalPersistentMixedControlConfig:
    max_stage_transitions_per_run: int = 256

    def __post_init__(self) -> None:
        if self.max_stage_transitions_per_run < 1:
            raise ValueError("max_stage_transitions_per_run must be at least 1")


@dataclass(frozen=True, slots=True)
class EnvironmentalPersistentMixedControlResult:
    stop_reason: EnvironmentalPersistentMixedStopReason
    terminal_mixed_result: EnvironmentalMixedOwnerControlResult
    rebootstrap_results: tuple[EnvironmentalMixedOwnerRebootstrapResult, ...]
    disposition_results: tuple[EnvironmentalRouteEndDispositionResult, ...]
    auto_admitted_route_owner_ids: tuple[str, ...]
    stage_transition_count: int

    @property
    def snapshot(self):
        return self.terminal_mixed_result.snapshot


class EnvironmentalPersistentMixedControlSession:
    """Automatically consume A119/A120/A121 boundaries on one shared environmental clock."""

    def __init__(
        self,
        mixed_session: EnvironmentalMixedOwnerControlSession,
        route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
        *,
        recapture_config: EnvironmentalFleetMultiOwnerRecaptureConfig,
        config: EnvironmentalPersistentMixedControlConfig = EnvironmentalPersistentMixedControlConfig(),
        profile_catalog: VesselProfileCatalog | None = None,
    ) -> None:
        if not isinstance(mixed_session, EnvironmentalMixedOwnerControlSession):
            raise TypeError("mixed_session must be EnvironmentalMixedOwnerControlSession")
        if not isinstance(route_targets, Mapping):
            raise TypeError("route_targets must be a mapping")
        if not isinstance(recapture_config, EnvironmentalFleetMultiOwnerRecaptureConfig):
            raise TypeError("recapture_config must be EnvironmentalFleetMultiOwnerRecaptureConfig")
        if not isinstance(config, EnvironmentalPersistentMixedControlConfig):
            raise TypeError("config must be EnvironmentalPersistentMixedControlConfig")

        self.mixed_session = mixed_session
        self.world = mixed_session.world
        self.route_targets = route_targets
        self.recapture_config = recapture_config
        self.config = config
        self.profile_catalog = profile_catalog
        self._rebootstrap_results: list[EnvironmentalMixedOwnerRebootstrapResult] = []
        self._disposition_results: list[EnvironmentalRouteEndDispositionResult] = []
        self._auto_admitted_route_owner_ids: list[str] = []
        self._release_sequence = 0

        # A122 is the authoritative caller-owned target catalogue for future route
        # admissions.  Existing route owners are already bound and remain unchanged.
        self.mixed_session.route_targets = route_targets

    @property
    def rebootstrap_results(self) -> tuple[EnvironmentalMixedOwnerRebootstrapResult, ...]:
        return tuple(self._rebootstrap_results)

    @property
    def disposition_results(self) -> tuple[EnvironmentalRouteEndDispositionResult, ...]:
        return tuple(self._disposition_results)

    @property
    def auto_admitted_route_owner_ids(self) -> tuple[str, ...]:
        return tuple(self._auto_admitted_route_owner_ids)

    def schedule_arrival(self, **kwargs) -> None:
        self.mixed_session.schedule_arrival(**kwargs)

    def schedule_removal(self, **kwargs) -> None:
        self.mixed_session.schedule_removal(**kwargs)

    def _validate_route_targets(self, vessel_ids: tuple[str, ...]) -> None:
        for vessel_id in vessel_ids:
            try:
                target = self.route_targets[vessel_id]
            except KeyError as exc:
                raise EnvironmentalPersistentMixedControlError(
                    f"missing route target for automatically released vessel {vessel_id!r}"
                ) from exc
            if not isinstance(target, EnvironmentalRouteHandoffTarget):
                raise TypeError(
                    f"route target for {vessel_id!r} must be EnvironmentalRouteHandoffTarget"
                )
            if target.vessel_id != vessel_id:
                raise EnvironmentalPersistentMixedControlError(
                    f"route target key {vessel_id!r} disagrees with target vessel id {target.vessel_id!r}"
                )

    def _potential_released_ids(
        self, source: EnvironmentalMixedOwnerControlResult
    ) -> tuple[str, ...]:
        """Mirror A120's release set read-only so route admission can be preflighted first."""

        seed = _affected_seed_vessels(self.mixed_session, source)
        replacement = _replacement_owners(source, seed)
        replacement_ids = {v for owner in replacement for v in owner.vessel_ids}
        current_ids = {v.vessel_id for v in self.world.snapshot().vessels}
        old_avoidance = self.mixed_session._avoidance_started()  # noqa: SLF001
        recapture_ids = set(source.recapture_owner_ids)
        retained = tuple(
            item
            for item in old_avoidance
            if item.owner.component_id not in recapture_ids
            and not replacement_ids.intersection(item.owner.vessel_ids)
        )
        affected_old_ids = {
            vessel_id
            for item in old_avoidance
            if item not in retained
            for vessel_id in item.owner.vessel_ids
        }
        return tuple(sorted((affected_old_ids - replacement_ids).intersection(current_ids)))

    def _admit_released_after_rebootstrap(
        self, result: EnvironmentalMixedOwnerRebootstrapResult
    ) -> tuple[str, ...]:
        released = result.released_vessel_ids
        if not released:
            return ()
        if tuple(v for v, _ in result.released_commands) != released:
            raise EnvironmentalPersistentMixedControlError(
                "A120 released command evidence does not match its released vessel tuple"
            )
        observation = result.source.recapture_observation
        if observation is None:
            raise EnvironmentalPersistentMixedControlError(
                "A120 source lacks the canonical observation required for route admission"
            )

        claimed = {
            vessel_id
            for owner in self.mixed_session._route_owners.values()  # noqa: SLF001
            if owner.stage is not EnvironmentalMixedRouteStage.COMPLETE
            for vessel_id in owner.vessel_ids
        }
        overlap = claimed.intersection(released)
        if overlap:
            raise EnvironmentalPersistentMixedControlError(
                f"released channels already have route authority: {sorted(overlap)!r}"
            )

        self._release_sequence += 1
        owner_id = (
            f"released:auto:{self._release_sequence}:"
            + "|".join(released)
        )
        owner = self.mixed_session._make_released_route_owner(  # noqa: SLF001
            owner_id,
            released,
            observation,
            result.snapshot_after,
            dict(result.released_commands),
        )
        self.mixed_session._route_owners[owner_id] = owner  # noqa: SLF001
        self.mixed_session._validate_disjoint_channels()  # noqa: SLF001
        self._auto_admitted_route_owner_ids.append(owner_id)
        return (owner_id,)

    @staticmethod
    def _terminal_reason(
        result: EnvironmentalMixedOwnerControlResult,
    ) -> EnvironmentalPersistentMixedStopReason:
        if result.stop_reason is EnvironmentalMixedOwnerStopReason.END_TIME:
            return EnvironmentalPersistentMixedStopReason.END_TIME
        if result.stop_reason is EnvironmentalMixedOwnerStopReason.COMMAND_OVERRIDDEN:
            return EnvironmentalPersistentMixedStopReason.COMMAND_OVERRIDDEN
        if result.stop_reason is EnvironmentalMixedOwnerStopReason.ALL_CONTROL_COMPLETE:
            return EnvironmentalPersistentMixedStopReason.ALL_CONTROL_COMPLETE
        raise AssertionError(result.stop_reason)

    def _result(
        self,
        terminal: EnvironmentalMixedOwnerControlResult,
        transitions: int,
    ) -> EnvironmentalPersistentMixedControlResult:
        return EnvironmentalPersistentMixedControlResult(
            stop_reason=self._terminal_reason(terminal),
            terminal_mixed_result=terminal,
            rebootstrap_results=self.rebootstrap_results,
            disposition_results=self.disposition_results,
            auto_admitted_route_owner_ids=self.auto_admitted_route_owner_ids,
            stage_transition_count=transitions,
        )

    def run_until(self, end_time: float) -> EnvironmentalPersistentMixedControlResult:
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.world.time:
            raise ValueError("end_time must not move backwards")

        transitions = 0
        while True:
            result = self.mixed_session.run_until(end_time)
            if result.stop_reason in {
                EnvironmentalMixedOwnerStopReason.END_TIME,
                EnvironmentalMixedOwnerStopReason.COMMAND_OVERRIDDEN,
                EnvironmentalMixedOwnerStopReason.ALL_CONTROL_COMPLETE,
            }:
                return self._result(result, transitions)

            if transitions >= self.config.max_stage_transitions_per_run:
                raise EnvironmentalPersistentMixedControlError(
                    "A122 exceeded max_stage_transitions_per_run before reaching a caller/terminal boundary"
                )

            if result.stop_reason is EnvironmentalMixedOwnerStopReason.RECAPTURE_REQUIRED:
                potential_released = self._potential_released_ids(result)
                # Route admission is part of A122's promised continuation.  Validate all
                # required targets before A120 is allowed to mutate avoidance commands.
                self._validate_route_targets(potential_released)
                recaptured = rebootstrap_environmental_mixed_owner_control(
                    self.mixed_session,
                    result,
                    self.route_targets,
                    config=self.recapture_config,
                    profile_catalog=self.profile_catalog,
                )
                self._rebootstrap_results.append(recaptured)
                self._admit_released_after_rebootstrap(recaptured)
                # Rebootstrap commits ordinary ownership before the temporary source
                # transaction owner is retired. No extra command write occurs here.
                self.mixed_session._reconcile_pending_source_control_ownership()  # noqa: SLF001
                transitions += 1
                continue

            if result.stop_reason is EnvironmentalMixedOwnerStopReason.ROUTE_DISPOSITION_REQUIRED:
                disposed = apply_environmental_route_end_dispositions(
                    self.mixed_session,
                    result,
                )
                self._disposition_results.append(disposed)
                transitions += 1
                continue

            raise EnvironmentalMixedOwnerControlError(
                f"A122 does not know how to consume mixed stop reason {result.stop_reason!r}"
            )
