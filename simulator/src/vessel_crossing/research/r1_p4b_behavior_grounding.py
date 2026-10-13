"""R1 P4B publication behavior-timing envelope.

This module encodes the B2 grounding decision.  It intentionally does *not* encode a
probability distribution for human response time and does not claim a real-OOW reaction
time.  The two endpoint values are a protocol sensitivity envelope above the validated
physical plant:

- T0: zero additional command-response delay for both AS and HS;
- T60: zero additional AS delay and +60 s additional HS delay.

T60 is a literature-motivated stress scale, not an empirical upper confidence bound for
a one-sided human reaction-delay distribution.  Publication interpretation must compare
endpoints separately and label conclusions timing-sensitive when they disagree.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from vessel_crossing.testbed.behavior import BehaviorProfile

from .r1_p4b_behavior_adapter import P4BBehaviorBinding


@dataclass(frozen=True, slots=True)
class P4BBehaviorTimingEndpoint:
    endpoint_id: str
    as_command_delay_min: float
    hs_command_delay_min: float
    interpretation: str

    def __post_init__(self) -> None:
        if not self.endpoint_id:
            raise ValueError("endpoint_id must be non-empty")
        for value, label in (
            (self.as_command_delay_min, "as_command_delay_min"),
            (self.hs_command_delay_min, "hs_command_delay_min"),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
        if not self.interpretation:
            raise ValueError("interpretation must be non-empty")


T0 = P4BBehaviorTimingEndpoint(
    endpoint_id="T0_ZERO_EXTRA_LAG",
    as_command_delay_min=0.0,
    hs_command_delay_min=0.0,
    interpretation=(
        "neutral lower sensitivity endpoint; no additional behavior-layer lag above the "
        "validated plant; not a claim that real OOW response time is zero"
    ),
)

T60 = P4BBehaviorTimingEndpoint(
    endpoint_id="T60_HS_PLUS_60S_STRESS",
    as_command_delay_min=0.0,
    hs_command_delay_min=1.0,
    interpretation=(
        "one-sided conventional-behavior timing stress endpoint using the approximately "
        "60 s AIS-derived action-time-window scale; not an empirical human reaction-time "
        "distribution, mean, or 95% upper bound"
    ),
)

P4B_BEHAVIOR_TIMING_ENDPOINTS = (T0, T60)


def publication_bindings_for_endpoint(
    endpoint: P4BBehaviorTimingEndpoint,
) -> tuple[P4BBehaviorBinding, P4BBehaviorBinding]:
    """Return the AS and HS bindings for one frozen timing endpoint."""
    if not isinstance(endpoint, P4BBehaviorTimingEndpoint):
        raise TypeError("endpoint must be P4BBehaviorTimingEndpoint")
    as_binding = P4BBehaviorBinding(
        BehaviorProfile.AUTONOMOUS_POLICY,
        f"R1_P4B_AS_AUTONOMOUS_POLICY/{endpoint.endpoint_id}",
        endpoint.as_command_delay_min,
    )
    hs_binding = P4BBehaviorBinding(
        BehaviorProfile.COMPLIANT_MERCHANT,
        f"R1_P4B_HS_CONVENTIONAL_TIMING/{endpoint.endpoint_id}",
        endpoint.hs_command_delay_min,
    )
    return as_binding, hs_binding


def timing_endpoint_interpretation_rule() -> str:
    return (
        "Report T0 and T60 separately. Do not average them and do not treat them as a "
        "confidence interval or sampled distribution. If a qualitative discrepancy, regime "
        "classification, or autonomy-gain conclusion changes between endpoints, label that "
        "conclusion BEHAVIOR_TIMING_SENSITIVE."
    )
