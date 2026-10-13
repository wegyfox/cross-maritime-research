"""Strict Stage C longitudinal reference binding primitives.

This module is deliberately separate from the historical P4B reference implementation.
Frozen P4B/P4D evidence must remain reproducible with its original code and identities.
Stage C uses explicit physical roles (``leader`` and ``follower``) so that source notation
cannot be confused with the simulator's internal ordering convention.

The current Guo et al. (2024) binding is source-backed only for its running-example
interval values and follower->leader orientation.  Its exact operational measurement
event and complete Section 3 capacity-bound formulation remain unresolved, so this
binding is intentionally not eligible for Stage C scientific freeze.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Final

AS: Final = "AS"
HS: Final = "HS"
_ALLOWED_TYPES: Final = frozenset((AS, HS))

DIRECT_SOURCE_VALUE: Final = "DIRECT_SOURCE_VALUE"
DERIVED_FROM_SOURCE: Final = "DERIVED_FROM_SOURCE"
ILLUSTRATIVE_EXOGENOUS_REFERENCE_VALUE: Final = "ILLUSTRATIVE_EXOGENOUS_REFERENCE_VALUE"
UNRESOLVED: Final = "UNRESOLVED"
NOT_JUSTIFIED: Final = "NOT_JUSTIFIED"
NOT_READY: Final = "NOT_READY"


@dataclass(frozen=True, slots=True)
class SourcePairInterval:
    """One source-declared interval keyed by follower and leader roles."""

    follower_type: str
    leader_type: str
    interval_min: float
    provenance: str
    scientific_role: str
    source_location: str

    def __post_init__(self) -> None:
        if self.follower_type not in _ALLOWED_TYPES or self.leader_type not in _ALLOWED_TYPES:
            raise ValueError("follower_type and leader_type must be AS or HS")
        if not isfinite(self.interval_min) or self.interval_min <= 0.0:
            raise ValueError("interval_min must be finite and positive")
        for name in ("provenance", "scientific_role", "source_location"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class StageCReferenceBinding:
    """Immutable source binding with explicit semantic readiness metadata.

    The source pair orientation is fixed as ``follower -> leader``.  Consumers that use
    a leader-first API must call :meth:`interval_for_leader_follower`, which performs the
    role translation explicitly rather than relying on ambiguous arrow notation.
    """

    reference_id: str
    source_title: str
    year: int
    doi: str
    source_pair_orientation: str
    source_term: str
    exact_measurement_event: str
    cross_measurement_mapping: str
    full_capacity_formulation_status: str
    stage_c_binding_status: str
    intervals: tuple[SourcePairInterval, ...]

    def __post_init__(self) -> None:
        if not self.reference_id or not self.source_title or not self.doi:
            raise ValueError("reference_id, source_title, and doi must be non-empty")
        if self.year < 1900:
            raise ValueError("year is outside the supported range")
        if self.source_pair_orientation != "FOLLOWER_TO_LEADER":
            raise ValueError("Stage C source_pair_orientation must be FOLLOWER_TO_LEADER")
        if not isinstance(self.intervals, tuple):
            raise TypeError("intervals must be an immutable tuple")
        if not all(isinstance(item, SourcePairInterval) for item in self.intervals):
            raise TypeError("intervals must contain SourcePairInterval values")
        keys = [(item.follower_type, item.leader_type) for item in self.intervals]
        expected = {(a, b) for a in _ALLOWED_TYPES for b in _ALLOWED_TYPES}
        if set(keys) != expected or len(keys) != len(expected):
            raise ValueError("intervals must contain each AS/HS follower-leader pair exactly once")

    def interval_for_source_roles(self, *, follower_type: str, leader_type: str) -> float:
        """Return the source interval using the source's follower-first role convention."""
        if follower_type not in _ALLOWED_TYPES or leader_type not in _ALLOWED_TYPES:
            raise ValueError("follower_type and leader_type must be AS or HS")
        for item in self.intervals:
            if item.follower_type == follower_type and item.leader_type == leader_type:
                return item.interval_min
        raise AssertionError("validated reference binding is missing a source-role pair")

    def interval_for_leader_follower(self, *, leader_type: str, follower_type: str) -> float:
        """Return the interval for a leader-first simulator/API convention.

        This is an explicit transpose of the source key convention, not a change to the
        source values themselves.
        """
        return self.interval_for_source_roles(follower_type=follower_type, leader_type=leader_type)

    @property
    def scientific_freeze_ready(self) -> bool:
        """Whether the binding satisfies the Stage C scientific-freeze semantic gate."""
        return (
            self.stage_c_binding_status == "READY_FOR_FREEZE"
            and self.exact_measurement_event != UNRESOLVED
            and self.cross_measurement_mapping != NOT_JUSTIFIED
            and self.full_capacity_formulation_status != UNRESOLVED
        )

    def require_scientific_freeze_ready(self) -> None:
        """Fail closed when unresolved semantics would make Stage C claims ambiguous."""
        if not self.scientific_freeze_ready:
            raise RuntimeError(
                "Stage C reference binding is not ready for scientific freeze: "
                "measurement semantics and/or full capacity formulation remain unresolved"
            )

    @property
    def identity_sha256(self) -> str:
        payload = {
            "reference_id": self.reference_id,
            "source_title": self.source_title,
            "year": self.year,
            "doi": self.doi,
            "source_pair_orientation": self.source_pair_orientation,
            "source_term": self.source_term,
            "exact_measurement_event": self.exact_measurement_event,
            "cross_measurement_mapping": self.cross_measurement_mapping,
            "full_capacity_formulation_status": self.full_capacity_formulation_status,
            "stage_c_binding_status": self.stage_c_binding_status,
            "intervals": [
                {
                    "follower_type": item.follower_type,
                    "leader_type": item.leader_type,
                    "interval_min": item.interval_min,
                    "provenance": item.provenance,
                    "scientific_role": item.scientific_role,
                    "source_location": item.source_location,
                }
                for item in sorted(self.intervals, key=lambda x: (x.follower_type, x.leader_type))
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()


def running_example_capacity_ships_per_day(mean_interval_min: float) -> float:
    """Reproduce the Guo running-example capacity arithmetic only.

    This helper must not be treated as the unresolved complete Section 3 capacity-bound
    formulation.
    """
    if not isfinite(mean_interval_min) or mean_interval_min <= 0.0:
        raise ValueError("mean_interval_min must be finite and positive")
    return 1440.0 / mean_interval_min


GUO_2024_RUNNING_EXAMPLE: Final = StageCReferenceBinding(
    reference_id="GUO_2024_OE_119734_RUNNING_EXAMPLE",
    source_title=(
        "Mixed traffic conditions of autonomous and human-driven ships: "
        "Assessing channel traffic capacity bounds and optimizing channel management"
    ),
    year=2024,
    doi="10.1016/j.oceaneng.2024.119734",
    source_pair_orientation="FOLLOWER_TO_LEADER",
    source_term="safety time interval",
    exact_measurement_event=UNRESOLVED,
    cross_measurement_mapping=NOT_JUSTIFIED,
    full_capacity_formulation_status=UNRESOLVED,
    stage_c_binding_status=NOT_READY,
    intervals=(
        SourcePairInterval(AS, AS, 7.50, DIRECT_SOURCE_VALUE, ILLUSTRATIVE_EXOGENOUS_REFERENCE_VALUE, "Introduction, Running example"),
        SourcePairInterval(AS, HS, 9.00, DIRECT_SOURCE_VALUE, ILLUSTRATIVE_EXOGENOUS_REFERENCE_VALUE, "Introduction, Running example"),
        SourcePairInterval(HS, AS, 11.25, DIRECT_SOURCE_VALUE, ILLUSTRATIVE_EXOGENOUS_REFERENCE_VALUE, "Introduction, Running example"),
        SourcePairInterval(HS, HS, 11.25, DIRECT_SOURCE_VALUE, ILLUSTRATIVE_EXOGENOUS_REFERENCE_VALUE, "Introduction, Running example"),
    ),
)

# Stage C C1b: source-audit addendum binding.  This is deliberately a separate identity
# from GUO_2024_RUNNING_EXAMPLE: the addendum supports a longitudinal ship-following
# passage-headway construct and a CROSS reference-line operationalization, while the
# exact geometric marker and verbatim Section-3 equations remain unavailable.
READY_WITH_LIMITATIONS: Final = "READY_WITH_LIMITATIONS"
LONGITUDINAL_REFERENCE_LINE_PASS: Final = "LONGITUDINAL_REFERENCE_LINE_PASS"


def derived_large_n_capacity_bounds_ships_per_day(p_as: float) -> tuple[float, float]:
    """Return derived (lower, upper) running-example capacity bounds.

    These equations reproduce the source's 50% running example but are not represented
    as verbatim Guo Section-3 equations.  ``p_as`` is the AS penetration fraction.
    """
    if not isfinite(p_as) or not 0.0 <= p_as <= 1.0:
        raise ValueError("p_as must be finite and within [0, 1]")
    upper_headway = 11.25 - 3.75 * p_as
    if p_as <= 0.5:
        lower_headway = 11.25 - 2.25 * p_as
    else:
        lower_headway = 12.75 - 5.25 * p_as
    return 1440.0 / lower_headway, 1440.0 / upper_headway


GUO_2024_RUNNING_EXAMPLE_DERIVED_V1: Final = StageCReferenceBinding(
    reference_id="GUO_2024_RUNNING_EXAMPLE_DERIVED_V1",
    source_title=(
        "Mixed traffic conditions of autonomous and human-driven ships: "
        "Assessing channel traffic capacity bounds and optimizing channel management"
    ),
    year=2024,
    doi="10.1016/j.oceaneng.2024.119734",
    source_pair_orientation="FOLLOWER_TO_LEADER",
    source_term="safety time interval / longitudinal temporal ship-following passage headway",
    # This is the CROSS operationalization of the now-qualified measurement construct,
    # not a claim that Guo names this exact geometric marker.
    exact_measurement_event=LONGITUDINAL_REFERENCE_LINE_PASS,
    cross_measurement_mapping=READY_WITH_LIMITATIONS,
    full_capacity_formulation_status="DERIVED_RUNNING_EXAMPLE_BOUNDS_NOT_EXACT_SECTION3",
    stage_c_binding_status=READY_WITH_LIMITATIONS,
    intervals=GUO_2024_RUNNING_EXAMPLE.intervals,
)
