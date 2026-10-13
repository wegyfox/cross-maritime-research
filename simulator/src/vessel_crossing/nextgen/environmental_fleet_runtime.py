"""Cold-start constructor for the persistent environmental fleet runtime.

A124 removes the final historical-bootstrap requirement from A123.  A caller can now
start with a fresh ``EnvironmentalWorld2D`` and ``EncounterRegistry`` at any non-negative
world time.  One canonical fleet observation is committed immediately.  From that exact
evidence the runtime either atomically bootstraps all initial PAIR/JOINT owners or enters
A123 idle supervision with no command owner.

The constructor does not invent a second clock or a parallel observation session.  It
creates an ownerless A117 coordinator and empty A119 mixed stage around the already-
committed observation, then composes A122 and A123 exactly as the post-recapture path does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .encounter_classification import EncounterClassificationConfig
from .encounter_registry import EncounterRegistry, PairKey
from .environmental_encounter_classification import EnvironmentalEncounterDirectionBasis
from .environmental_fleet_multi_owner_recapture import EnvironmentalFleetMultiOwnerRecaptureConfig
from .environmental_fleet_ownership import (
    EnvironmentalFleetOwnershipObservation,
    observe_environmental_fleet_ownership,
)
from .environmental_idle_fleet_control import (
    EnvironmentalIdleFleetBootstrapResult,
    EnvironmentalIdleFleetControlConfig,
    EnvironmentalIdleFleetReason,
    EnvironmentalPersistentFleetControlSession,
    bootstrap_environmental_idle_fleet_owners,
)
from .environmental_mixed_owner_control import (
    EnvironmentalMixedOwnerControlConfig,
    EnvironmentalMixedOwnerControlSession,
)
from .environmental_multi_owner_control import (
    EnvironmentalMultiOwnerControlConfig,
    EnvironmentalMultiOwnerControlSession,
)
from .environmental_persistent_mixed_control import (
    EnvironmentalPersistentMixedControlConfig,
    EnvironmentalPersistentMixedControlSession,
)
from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
from .environmental_world import EnvironmentalWorld2D
from .responsibility import ResponsibilityContext
from .vessel_classes import VesselProfileCatalog
from .same_route_following import SameRouteFollowingSupervisor

_EPS = 1e-10


class EnvironmentalFleetRuntimeStartError(RuntimeError):
    """Raised when a cold-start runtime cannot establish one canonical initial state."""


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeConfig:
    multi_owner: EnvironmentalMultiOwnerControlConfig = EnvironmentalMultiOwnerControlConfig()
    mixed: EnvironmentalMixedOwnerControlConfig = EnvironmentalMixedOwnerControlConfig()
    persistent: EnvironmentalPersistentMixedControlConfig = EnvironmentalPersistentMixedControlConfig()
    idle: EnvironmentalIdleFleetControlConfig = EnvironmentalIdleFleetControlConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.multi_owner, EnvironmentalMultiOwnerControlConfig):
            raise TypeError("multi_owner must be EnvironmentalMultiOwnerControlConfig")
        if not isinstance(self.mixed, EnvironmentalMixedOwnerControlConfig):
            raise TypeError("mixed must be EnvironmentalMixedOwnerControlConfig")
        if not isinstance(self.persistent, EnvironmentalPersistentMixedControlConfig):
            raise TypeError("persistent must be EnvironmentalPersistentMixedControlConfig")
        if not isinstance(self.idle, EnvironmentalIdleFleetControlConfig):
            raise TypeError("idle must be EnvironmentalIdleFleetControlConfig")


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStartResult:
    session: EnvironmentalPersistentFleetControlSession
    initial_observation: EnvironmentalFleetOwnershipObservation
    initial_bootstrap: EnvironmentalIdleFleetBootstrapResult | None

    @property
    def initial_owner_ids(self) -> tuple[str, ...]:
        if self.initial_bootstrap is None:
            return ()
        return self.initial_bootstrap.owner_ids


def start_environmental_fleet_runtime(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    *,
    recapture_config: EnvironmentalFleetMultiOwnerRecaptureConfig,
    config: EnvironmentalFleetRuntimeConfig = EnvironmentalFleetRuntimeConfig(),
    uncertain_pairs: Iterable[PairKey] = (),
    responsibility_context: ResponsibilityContext | None = None,
    classification_config: EncounterClassificationConfig | None = None,
    direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
    profile_catalog: VesselProfileCatalog | None = None,
    persistent_same_route_following: SameRouteFollowingSupervisor | None = None,
) -> EnvironmentalFleetRuntimeStartResult:
    """Create the complete A117–A123 runtime from a fresh world/registry boundary."""

    if not isinstance(registry, EncounterRegistry):
        raise TypeError("registry must be EncounterRegistry")
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    if not isinstance(route_targets, Mapping):
        raise TypeError("route_targets must be a mapping")
    if not isinstance(recapture_config, EnvironmentalFleetMultiOwnerRecaptureConfig):
        raise TypeError("recapture_config must be EnvironmentalFleetMultiOwnerRecaptureConfig")
    if not isinstance(config, EnvironmentalFleetRuntimeConfig):
        raise TypeError("config must be EnvironmentalFleetRuntimeConfig")
    if persistent_same_route_following is not None and not isinstance(
        persistent_same_route_following, SameRouteFollowingSupervisor
    ):
        raise TypeError(
            "persistent_same_route_following must be SameRouteFollowingSupervisor or None"
        )
    if getattr(registry, "_last_update_time", None) is not None:
        raise EnvironmentalFleetRuntimeStartError(
            "cold-start runtime requires a fresh EncounterRegistry with no prior update"
        )

    uncertain_pairs = tuple(uncertain_pairs)
    # The persistent runtime owns a mutable mission-target catalogue.  A125 future ingress
    # can therefore install its route target as part of admission even when the caller
    # supplied an immutable Mapping at cold start.
    runtime_route_targets = dict(route_targets)
    if recapture_config.same_route_following is not None:
        registry.bind_responsibility_delegation_resolver(
            recapture_config.same_route_following.delegation_resolver(runtime_route_targets)
        )
    initial = observe_environmental_fleet_ownership(
        registry,
        world,
        uncertain_pairs=uncertain_pairs,
        responsibility_context=responsibility_context,
        classification_config=classification_config,
        direction_basis=direction_basis,
    )
    if abs(initial.time_min - world.time) > _EPS:  # defensive; observation constructor also aligns
        raise EnvironmentalFleetRuntimeStartError(
            "initial canonical fleet observation does not match world time"
        )

    coordinator = EnvironmentalMultiOwnerControlSession.from_idle_observation(
        registry,
        world,
        initial,
        uncertain_pairs=uncertain_pairs,
        responsibility_context=responsibility_context,
        classification_config=classification_config,
        direction_basis=direction_basis,
        config=config.multi_owner,
    )
    mixed = EnvironmentalMixedOwnerControlSession.from_idle_coordinator(
        coordinator,
        runtime_route_targets,
        config=config.mixed,
        profile_catalog=profile_catalog,
        # One persistent following supervisor must also constrain ordinary route-command
        # writes.  Otherwise a periodic persistent safety write can be overwritten at the
        # next route-navigation refresh by a weaker/default supervisor, silently dropping
        # microscopic reaction/observation allowances between fleet reviews.
        same_route_following=(
            persistent_same_route_following
            if persistent_same_route_following is not None
            else recapture_config.same_route_following
        ),
        persistent_same_route_following=persistent_same_route_following,
    )
    persistent = EnvironmentalPersistentMixedControlSession(
        mixed,
        runtime_route_targets,
        recapture_config=recapture_config,
        config=config.persistent,
        profile_catalog=profile_catalog,
    )
    session = EnvironmentalPersistentFleetControlSession(
        persistent,
        recapture_config=recapture_config,
        config=config.idle,
        profile_catalog=profile_catalog,
    )

    bootstrap: EnvironmentalIdleFleetBootstrapResult | None = None
    owner_ids: tuple[str, ...] = ()
    if initial.dispatch.owners:
        bootstrap = bootstrap_environmental_idle_fleet_owners(
            persistent,
            initial,
            config=recapture_config,
            profile_catalog=profile_catalog,
        )
        session._bootstrap_results.append(bootstrap)  # noqa: SLF001 - same package orchestration state
        owner_ids = bootstrap.owner_ids
        session._idle_active = False  # noqa: SLF001
    else:
        session._idle_active = True  # noqa: SLF001
        session._next_idle_review_time = world.time + config.idle.idle_review_interval_min  # noqa: SLF001

    session._record_idle_boundary(  # noqa: SLF001
        reasons=(EnvironmentalIdleFleetReason.INITIAL,),
        membership_event_count=0,
        removed_vessel_ids=(),
        observation=initial,
        bootstrapped_owner_ids=owner_ids,
    )
    return EnvironmentalFleetRuntimeStartResult(session, initial, bootstrap)
