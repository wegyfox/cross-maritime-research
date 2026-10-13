"""Stateful explicit-review supervisor for a multi-vessel joint encounter component.

A29 applies one snapshot-bound command tuple.  This layer repeats that process only at
explicit review events.  It consumes an already-existing registry update for the
current timestamp, avoiding a duplicate navigation observation, and after each review
rebuilds supervision admission and connected components.

The supervisor does not silently chase a changed topology.  If the original active
component splits, merges, gains/loses active edges, or gains/loses vessels, execution
stops with ``TOPOLOGY_CHANGED`` so an upstream fleet orchestrator can construct a new
component/intent/physics plan.  Only total disappearance of overlapping active
responsibility is ``COMPONENT_RELEASED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Iterable

from .encounter_components import (
    ActiveEncounterComponent,
    EncounterComponentBatch,
    EncounterComponentKind,
    build_active_encounter_components,
)
from .encounter_registry import EncounterRegistry, EncounterRegistryUpdate, PairKey
from .joint_execution import JointExecutionPlan, apply_joint_execution_plan
from .responsibility import ResponsibilityContext
from .supervision_admission import admit_supervision
from .world import World2D, WorldSnapshot

_EPS = 1e-10


class JointSupervisionError(RuntimeError):
    """Base class for fail-closed joint-supervision errors."""


class NoJointSupervisionPlan(JointSupervisionError):
    """Raised when the component is still active but the replanner returns no plan."""


class JointSupervisionPlanMismatch(JointSupervisionError):
    """Raised when a submitted plan belongs to a different component topology."""


@dataclass(frozen=True, slots=True)
class JointSupervisionConfig:
    review_interval_min: float
    max_reviews: int = 64

    def __post_init__(self) -> None:
        if not isfinite(self.review_interval_min) or self.review_interval_min <= 0.0:
            raise ValueError("review_interval_min must be finite and positive")
        if self.max_reviews < 1:
            raise ValueError("max_reviews must be at least 1")


@dataclass(frozen=True, slots=True)
class JointSupervisionPlan:
    pair_keys: tuple[PairKey, ...]
    execution_plan: JointExecutionPlan

    def __post_init__(self) -> None:
        if not self.pair_keys:
            raise ValueError("joint supervision plan requires at least one active pair")
        if tuple(sorted(self.pair_keys)) != self.pair_keys or len(set(self.pair_keys)) != len(self.pair_keys):
            raise ValueError("pair_keys must be unique and canonically sorted")
        vessel_set = set(self.execution_plan.vessel_ids)
        for first_id, second_id in self.pair_keys:
            if first_id >= second_id:
                raise ValueError("pair_keys must use canonical ascending vessel ids")
            if first_id not in vessel_set or second_id not in vessel_set:
                raise ValueError("pair key contains vessel outside execution plan")


class JointSupervisionStopReason(str, Enum):
    COMPONENT_RELEASED = "component_released"
    TOPOLOGY_CHANGED = "topology_changed"
    END_TIME = "end_time"


@dataclass(frozen=True, slots=True)
class JointSupervisionStep:
    start_time: float
    target_review_time: float
    actual_end_time: float
    component_before: ActiveEncounterComponent
    plan: JointSupervisionPlan
    snapshot_after: WorldSnapshot
    review_performed: bool
    registry_update_after: EncounterRegistryUpdate | None
    components_after: EncounterComponentBatch | None


@dataclass(frozen=True, slots=True)
class JointSupervisionResult:
    snapshot: WorldSnapshot
    last_registry_update: EncounterRegistryUpdate
    last_components: EncounterComponentBatch
    component: ActiveEncounterComponent | None
    steps: tuple[JointSupervisionStep, ...]
    stop_reason: JointSupervisionStopReason

    @property
    def handoff_ready(self) -> bool:
        return self.stop_reason is JointSupervisionStopReason.COMPONENT_RELEASED


JointReplanner = Callable[
    [WorldSnapshot, EncounterRegistryUpdate, ActiveEncounterComponent],
    JointSupervisionPlan | None,
]


def bind_joint_supervision_plan(
    component: ActiveEncounterComponent,
    execution_plan: JointExecutionPlan,
) -> JointSupervisionPlan:
    """Bind an a29 plan to the exact active pair topology used by supervision."""
    if component.component_id != execution_plan.component_id:
        raise JointSupervisionPlanMismatch("execution plan component_id does not match component")
    if component.vessel_ids != execution_plan.vessel_ids:
        raise JointSupervisionPlanMismatch("execution plan vessel order does not match component")
    return JointSupervisionPlan(component.pair_keys, execution_plan)


class JointAvoidanceExecutionSupervisor:
    """Execute/replan one multi-vessel component at explicit registry review events."""

    def __init__(
        self,
        world: World2D,
        registry: EncounterRegistry,
        *,
        config: JointSupervisionConfig,
    ) -> None:
        self.world = world
        self.registry = registry
        self.config = config

    @staticmethod
    def _components(update: EncounterRegistryUpdate) -> EncounterComponentBatch:
        return build_active_encounter_components(admit_supervision(update))

    @staticmethod
    def _same_component(
        component: ActiveEncounterComponent,
        candidate: ActiveEncounterComponent,
    ) -> bool:
        return (
            candidate.vessel_ids == component.vessel_ids
            and candidate.pair_keys == component.pair_keys
        )

    @staticmethod
    def _find_exact(
        batch: EncounterComponentBatch,
        component: ActiveEncounterComponent,
    ) -> ActiveEncounterComponent | None:
        for candidate in batch.components:
            if JointAvoidanceExecutionSupervisor._same_component(component, candidate):
                return candidate
        return None

    @staticmethod
    def _overlapping_components(
        batch: EncounterComponentBatch,
        vessel_ids: tuple[str, ...],
    ) -> tuple[ActiveEncounterComponent, ...]:
        original = set(vessel_ids)
        return tuple(
            component
            for component in batch.components
            if original.intersection(component.vessel_ids)
        )

    def _validate_initial(
        self,
        initial_update: EncounterRegistryUpdate,
        initial_component: ActiveEncounterComponent,
    ) -> EncounterComponentBatch:
        if abs(initial_update.time_min - self.world.time) > _EPS:
            raise ValueError("initial registry update time must match current world time")
        if initial_component.kind is not EncounterComponentKind.MULTI_VESSEL:
            raise ValueError("joint supervisor requires a MULTI_VESSEL active component")
        batch = self._components(initial_update)
        if self._find_exact(batch, initial_component) is None:
            raise ValueError("initial component is not present in the supplied registry update")
        snapshot = self.world.snapshot()
        for vessel_id in initial_component.vessel_ids:
            snapshot.by_id(vessel_id)
        return batch

    @staticmethod
    def _validate_plan(
        plan: JointSupervisionPlan,
        component: ActiveEncounterComponent,
    ) -> None:
        execution = plan.execution_plan
        if execution.component_id != component.component_id:
            raise JointSupervisionPlanMismatch("plan component_id does not match active component")
        if execution.vessel_ids != component.vessel_ids:
            raise JointSupervisionPlanMismatch("plan vessel order does not match active component")
        if plan.pair_keys != component.pair_keys:
            raise JointSupervisionPlanMismatch("plan pair topology does not match active component")

    def run_until_release(
        self,
        initial_update: EncounterRegistryUpdate,
        initial_component: ActiveEncounterComponent,
        *,
        replanner: JointReplanner,
        end_time: float,
        uncertain_pairs: Iterable[PairKey] = (),
        responsibility_context: ResponsibilityContext | None = None,
    ) -> JointSupervisionResult:
        """Execute at explicit reviews until release, topology change, or ``end_time``."""
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        uncertain_pairs = tuple(uncertain_pairs)
        components = self._validate_initial(initial_update, initial_component)
        current_component = self._find_exact(components, initial_component)
        assert current_component is not None
        update = initial_update
        steps: list[JointSupervisionStep] = []

        if self.world.time >= end_time - _EPS:
            return JointSupervisionResult(
                self.world.snapshot(), update, components, current_component, (), JointSupervisionStopReason.END_TIME
            )

        for _ in range(self.config.max_reviews):
            snapshot_before = self.world.snapshot()
            plan = replanner(snapshot_before, update, current_component)
            if plan is None:
                raise NoJointSupervisionPlan(
                    "active multi-vessel component remains but replanner returned no joint plan"
                )
            self._validate_plan(plan, current_component)
            apply_joint_execution_plan(self.world, plan.execution_plan)

            start_time = self.world.time
            target_review = start_time + self.config.review_interval_min
            if end_time < target_review - _EPS:
                self.world.run_until(end_time)
                step = JointSupervisionStep(
                    start_time=start_time,
                    target_review_time=target_review,
                    actual_end_time=end_time,
                    component_before=current_component,
                    plan=plan,
                    snapshot_after=self.world.snapshot(),
                    review_performed=False,
                    registry_update_after=None,
                    components_after=None,
                )
                steps.append(step)
                return JointSupervisionResult(
                    self.world.snapshot(),
                    update,
                    components,
                    current_component,
                    tuple(steps),
                    JointSupervisionStopReason.END_TIME,
                )

            self.world.run_until(target_review)
            update_after = self.registry.update(
                self.world,
                uncertain_pairs=uncertain_pairs,
                responsibility_context=responsibility_context,
            )
            components_after = self._components(update_after)
            exact = self._find_exact(components_after, current_component)
            overlapping = self._overlapping_components(components_after, current_component.vessel_ids)
            step = JointSupervisionStep(
                start_time=start_time,
                target_review_time=target_review,
                actual_end_time=target_review,
                component_before=current_component,
                plan=plan,
                snapshot_after=self.world.snapshot(),
                review_performed=True,
                registry_update_after=update_after,
                components_after=components_after,
            )
            steps.append(step)

            if exact is not None:
                update = update_after
                components = components_after
                current_component = exact
                if self.world.time >= end_time - _EPS:
                    return JointSupervisionResult(
                        self.world.snapshot(),
                        update,
                        components,
                        current_component,
                        tuple(steps),
                        JointSupervisionStopReason.END_TIME,
                    )
                continue

            if not overlapping:
                return JointSupervisionResult(
                    self.world.snapshot(),
                    update_after,
                    components_after,
                    None,
                    tuple(steps),
                    JointSupervisionStopReason.COMPONENT_RELEASED,
                )

            return JointSupervisionResult(
                self.world.snapshot(),
                update_after,
                components_after,
                None,
                tuple(steps),
                JointSupervisionStopReason.TOPOLOGY_CHANGED,
            )

        raise JointSupervisionError("joint supervision exceeded max_reviews")
