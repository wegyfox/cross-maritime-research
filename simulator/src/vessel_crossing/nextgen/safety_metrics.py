"""Continuous pair safety-outcome metrics, separate from avoidance decisions.

The avoidance stack answers whether a proposed command is safe enough to execute.  This
module answers a different question after a physical interval is defined: what outcome
evidence did the modeled pair produce?  It never feeds a result back into arbitration or
command ownership.

Two explicit geometries are kept separate:

* a symmetric ellipse derived from vessel length/beam is used only as a *modeled hull
  footprint* for overlap evidence;
* the class ``safety_domain`` remains the larger operational domain used for domain
  intrusion / near-miss measurement.

Both interval checks reuse the conservative continuous-time domain-clearance
certificate.  A sampled threshold breach is observed evidence; a positive conservative
lower bound is a certificate of clearance.  If neither is available, the outcome is
reported as unresolved rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable

from .domain_clearance import DomainClearanceCertification, certify_pair_domain_clearance
from .dynamics import ManeuverCommand, ManeuverLimits, Vessel2DState, advance_toward_command
from .ship_domain import ShipDomain
from .vessel_classes import VesselClassProfile, VesselDimensions

_EPS = 1e-12


class SafetyMetricError(RuntimeError):
    """Base class for fail-closed safety metric errors."""


class SafetyMetricUnavailable(SafetyMetricError):
    """Raised when required profile/domain evidence is missing or inconsistent."""


class PairSafetyOutcome(str, Enum):
    MODELED_HULL_OVERLAP = "modeled_hull_overlap"
    HULL_CLEARANCE_UNRESOLVED = "hull_clearance_unresolved"
    SAFETY_DOMAIN_INTRUSION = "safety_domain_intrusion"
    SAFETY_DOMAIN_CLEARANCE_UNRESOLVED = "safety_domain_clearance_unresolved"
    NEAR_MISS = "near_miss"
    NEAR_MISS_UNRESOLVED = "near_miss_unresolved"
    CLEAR = "clear"


@dataclass(frozen=True, slots=True)
class PairSafetyMetricConfig:
    """Numerical resolution plus an explicit, non-regulatory near-miss margin."""

    near_miss_margin_nm: float
    time_tolerance_min: float = 1e-4
    margin_tolerance_nm: float = 1e-5
    max_subdivisions: int = 20

    def __post_init__(self) -> None:
        values = (self.near_miss_margin_nm, self.time_tolerance_min, self.margin_tolerance_nm)
        if not all(isfinite(value) for value in values):
            raise ValueError("safety metric numerical inputs must be finite")
        if self.near_miss_margin_nm < 0.0:
            raise ValueError("near_miss_margin_nm must be non-negative")
        if self.time_tolerance_min <= 0.0:
            raise ValueError("time_tolerance_min must be positive")
        if self.margin_tolerance_nm < 0.0:
            raise ValueError("margin_tolerance_nm must be non-negative")
        if self.max_subdivisions < 1:
            raise ValueError("max_subdivisions must be at least 1")


@dataclass(frozen=True, slots=True)
class PairSafetyIntervalMetric:
    first_vessel_id: str
    second_vessel_id: str
    duration_min: float
    outcome: PairSafetyOutcome
    hull_clearance: DomainClearanceCertification
    safety_domain_clearance: DomainClearanceCertification
    near_miss_margin_nm: float

    @property
    def pair_key(self) -> tuple[str, str]:
        return tuple(sorted((self.first_vessel_id, self.second_vessel_id)))  # type: ignore[return-value]

    @property
    def sampled_min_center_separation_nm(self) -> float:
        return min(
            self.hull_clearance.sampled_min_separation_nm,
            self.safety_domain_clearance.sampled_min_separation_nm,
        )

    @property
    def sampled_min_hull_margin_nm(self) -> float:
        return self.hull_clearance.sampled_min_margin_nm

    @property
    def sampled_min_safety_domain_margin_nm(self) -> float:
        return self.safety_domain_clearance.sampled_min_margin_nm


@dataclass(frozen=True, slots=True)
class FleetSafetyMetricSummary:
    interval_count: int
    distinct_pair_count: int
    modeled_hull_overlap_count: int
    safety_domain_intrusion_count: int
    near_miss_count: int
    unresolved_count: int
    clear_count: int
    sampled_min_center_separation_nm: float | None
    sampled_min_hull_margin_nm: float | None
    sampled_min_safety_domain_margin_nm: float | None


def modeled_hull_domain(dimensions: VesselDimensions) -> ShipDomain:
    """Return the explicit symmetric ellipse used as the a64 hull-footprint model."""

    half_length = 0.5 * dimensions.length_nm
    half_beam = 0.5 * dimensions.beam_nm
    return ShipDomain(
        ahead_nm=half_length,
        astern_nm=half_length,
        port_nm=half_beam,
        starboard_nm=half_beam,
    )


def _validate_profile_command(
    *,
    state: Vessel2DState,
    command: ManeuverCommand,
    profile: VesselClassProfile,
) -> None:
    if profile.speed_envelope is not None:
        if not profile.speed_envelope.permits(state.speed):
            raise SafetyMetricUnavailable(
                f"current speed exceeds profile speed envelope for {state.vessel_id}"
            )
        if not profile.speed_envelope.permits(command.target_speed):
            raise SafetyMetricUnavailable(
                f"command target speed exceeds profile speed envelope for {state.vessel_id}"
            )
    if profile.safety_domain is None:
        raise SafetyMetricUnavailable(
            f"safety domain is required for near-miss metrics for {state.vessel_id}"
        )


def _certificate(
    *,
    first_state: Vessel2DState,
    second_state: Vessel2DState,
    first_command: ManeuverCommand,
    second_command: ManeuverCommand,
    first_limits: ManeuverLimits,
    second_limits: ManeuverLimits,
    first_domain: ShipDomain,
    second_domain: ShipDomain,
    duration_min: float,
    config: PairSafetyMetricConfig,
) -> DomainClearanceCertification:
    return certify_pair_domain_clearance(
        first_state_at=lambda t: advance_toward_command(
            first_state, first_command, t, first_limits
        ).state,
        second_state_at=lambda t: advance_toward_command(
            second_state, second_command, t, second_limits
        ).state,
        first_domain=first_domain,
        second_domain=second_domain,
        duration_min=duration_min,
        first_speed_bound_nm_per_min=max(first_state.speed, first_command.target_speed),
        second_speed_bound_nm_per_min=max(second_state.speed, second_command.target_speed),
        first_turn_rate_bound_deg_per_min=first_limits.max_turn_rate_deg_per_min,
        second_turn_rate_bound_deg_per_min=second_limits.max_turn_rate_deg_per_min,
        time_tolerance_min=config.time_tolerance_min,
        margin_tolerance_nm=config.margin_tolerance_nm,
        max_subdivisions=config.max_subdivisions,
    )


def evaluate_pair_safety_interval(
    *,
    first_state: Vessel2DState,
    second_state: Vessel2DState,
    first_command: ManeuverCommand,
    second_command: ManeuverCommand,
    first_limits: ManeuverLimits,
    second_limits: ManeuverLimits,
    first_profile: VesselClassProfile,
    second_profile: VesselClassProfile,
    duration_min: float,
    config: PairSafetyMetricConfig,
) -> PairSafetyIntervalMetric:
    """Measure one pair over an explicit bounded-command physical interval."""

    if first_state.vessel_id == second_state.vessel_id:
        raise ValueError("pair safety metric requires two distinct vessel IDs")
    if not isfinite(duration_min) or duration_min < 0.0:
        raise ValueError("duration_min must be finite and non-negative")

    _validate_profile_command(state=first_state, command=first_command, profile=first_profile)
    _validate_profile_command(state=second_state, command=second_command, profile=second_profile)

    hull = _certificate(
        first_state=first_state,
        second_state=second_state,
        first_command=first_command,
        second_command=second_command,
        first_limits=first_limits,
        second_limits=second_limits,
        first_domain=modeled_hull_domain(first_profile.dimensions),
        second_domain=modeled_hull_domain(second_profile.dimensions),
        duration_min=duration_min,
        config=config,
    )
    safety = _certificate(
        first_state=first_state,
        second_state=second_state,
        first_command=first_command,
        second_command=second_command,
        first_limits=first_limits,
        second_limits=second_limits,
        first_domain=first_profile.safety_domain,  # type: ignore[arg-type]
        second_domain=second_profile.safety_domain,  # type: ignore[arg-type]
        duration_min=duration_min,
        config=config,
    )

    # Severity ordering is explicit.  "Observed" means the adaptive certificate sampled
    # an actual threshold breach.  Failure to certify clear is never upgraded to a breach.
    if hull.sampled_min_margin_nm <= _EPS:
        outcome = PairSafetyOutcome.MODELED_HULL_OVERLAP
    elif hull.conservative_lower_bound_nm <= _EPS:
        outcome = PairSafetyOutcome.HULL_CLEARANCE_UNRESOLVED
    elif safety.sampled_min_margin_nm <= _EPS:
        outcome = PairSafetyOutcome.SAFETY_DOMAIN_INTRUSION
    elif safety.conservative_lower_bound_nm <= _EPS:
        outcome = PairSafetyOutcome.SAFETY_DOMAIN_CLEARANCE_UNRESOLVED
    elif safety.sampled_min_margin_nm <= config.near_miss_margin_nm + _EPS:
        outcome = PairSafetyOutcome.NEAR_MISS
    elif safety.conservative_lower_bound_nm <= config.near_miss_margin_nm + _EPS:
        outcome = PairSafetyOutcome.NEAR_MISS_UNRESOLVED
    else:
        outcome = PairSafetyOutcome.CLEAR

    return PairSafetyIntervalMetric(
        first_vessel_id=first_state.vessel_id,
        second_vessel_id=second_state.vessel_id,
        duration_min=duration_min,
        outcome=outcome,
        hull_clearance=hull,
        safety_domain_clearance=safety,
        near_miss_margin_nm=config.near_miss_margin_nm,
    )


def summarize_fleet_safety_metrics(
    metrics: Iterable[PairSafetyIntervalMetric],
) -> FleetSafetyMetricSummary:
    items = tuple(metrics)
    unresolved = {
        PairSafetyOutcome.HULL_CLEARANCE_UNRESOLVED,
        PairSafetyOutcome.SAFETY_DOMAIN_CLEARANCE_UNRESOLVED,
        PairSafetyOutcome.NEAR_MISS_UNRESOLVED,
    }

    def minimum(values: Iterable[float]) -> float | None:
        collected = tuple(values)
        return None if not collected else min(collected)

    return FleetSafetyMetricSummary(
        interval_count=len(items),
        distinct_pair_count=len({item.pair_key for item in items}),
        modeled_hull_overlap_count=sum(
            item.outcome is PairSafetyOutcome.MODELED_HULL_OVERLAP for item in items
        ),
        safety_domain_intrusion_count=sum(
            item.outcome is PairSafetyOutcome.SAFETY_DOMAIN_INTRUSION for item in items
        ),
        near_miss_count=sum(item.outcome is PairSafetyOutcome.NEAR_MISS for item in items),
        unresolved_count=sum(item.outcome in unresolved for item in items),
        clear_count=sum(item.outcome is PairSafetyOutcome.CLEAR for item in items),
        sampled_min_center_separation_nm=minimum(
            item.sampled_min_center_separation_nm for item in items
        ),
        sampled_min_hull_margin_nm=minimum(item.sampled_min_hull_margin_nm for item in items),
        sampled_min_safety_domain_margin_nm=minimum(
            item.sampled_min_safety_domain_margin_nm for item in items
        ),
    )
