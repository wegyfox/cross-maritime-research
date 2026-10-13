"""Passive conflict-boundary observer for R1 P4D.

The observer is deliberately *not* an event source.  It inspects each native realized
motion interval before that interval is committed, evaluates the same frozen trajectory
used by the motion backend, and records first-passage times inside the interval.  It
never schedules a command, advances the world, or introduces a new scheduler boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from vessel_crossing.nextgen.environmental_candidate_transition import (
    build_environmental_candidate_transition,
)
from vessel_crossing.nextgen.environmental_dynamics import WaterRelativeVesselState
from vessel_crossing.nextgen.environmental_network_paired_experiment import _MetricEnvironmentalWorld

_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class P4DConflictGate:
    role: str
    axis: int
    coordinate_nm: float
    direction_sign: int

    def __post_init__(self) -> None:
        if self.role not in {
            "CONFLICT_REQUEST",
            "CONFLICT_ENTRY",
            "CONFLICT_CENTER_PASS",
            "CONFLICT_EXIT",
        }:
            raise ValueError("unsupported P4D gate role")
        if self.axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        if not isfinite(self.coordinate_nm):
            raise ValueError("coordinate_nm must be finite")
        if self.direction_sign not in (-1, 1):
            raise ValueError("direction_sign must be -1 or +1")

    def progress(self, state: WaterRelativeVesselState) -> float:
        return self.direction_sign * (state.position[self.axis] - self.coordinate_nm)


@dataclass(frozen=True, slots=True)
class P4DConflictObservation:
    vessel_id: str
    stream_id: str
    role: str
    observation_time_min: float
    position_nm: tuple[float, float]
    through_water_heading_deg: float
    through_water_speed_nm_per_min: float


_DEFAULT_GATES_BY_STREAM: dict[str, tuple[P4DConflictGate, ...]] = {
    # East -> West: x decreases from +room toward -room.
    "west": (
        P4DConflictGate("CONFLICT_REQUEST", 0, 0.5, -1),
        P4DConflictGate("CONFLICT_ENTRY", 0, 0.2, -1),
        P4DConflictGate("CONFLICT_CENTER_PASS", 0, 0.0, -1),
        P4DConflictGate("CONFLICT_EXIT", 0, -0.2, -1),
    ),
    # South -> North: y increases from -room toward +room.
    "north": (
        P4DConflictGate("CONFLICT_REQUEST", 1, -0.5, +1),
        P4DConflictGate("CONFLICT_ENTRY", 1, -0.2, +1),
        P4DConflictGate("CONFLICT_CENTER_PASS", 1, 0.0, +1),
        P4DConflictGate("CONFLICT_EXIT", 1, 0.2, +1),
    ),
}


def p4d_default_gates_for_stream(stream_id: str) -> tuple[P4DConflictGate, ...]:
    try:
        return _DEFAULT_GATES_BY_STREAM[stream_id]
    except KeyError as exc:
        raise KeyError(f"no P4D conflict gates declared for stream {stream_id!r}") from exc


def _first_passage_time(
    trajectory,
    gate: P4DConflictGate,
    duration_min: float,
    *,
    time_tolerance_min: float = 1e-7,
    max_depth: int = 40,
) -> float | None:
    """Return earliest directed gate crossing in one frozen trajectory interval.

    The recursive search is safe for temporary motion away from the gate.  A Lipschitz
    bound from the trajectory ground-speed bound prunes intervals that provably cannot
    reach the gate; uncertain intervals are subdivided left-first until a crossing is
    bracketed or the time tolerance is reached.
    """
    if duration_min <= _EPS:
        return None
    if not isfinite(duration_min) or duration_min < 0.0:
        raise ValueError("duration_min must be finite and non-negative")
    speed_bound = float(trajectory.ground_speed_bound)
    if not isfinite(speed_bound) or speed_bound < 0.0:
        raise ValueError("trajectory ground_speed_bound must be finite and non-negative")

    cache: dict[float, float] = {}

    def f(t: float) -> float:
        key = round(float(t), 12)
        if key not in cache:
            cache[key] = gate.progress(trajectory.state_at_or_continue(float(t)))
        return cache[key]

    f0 = f(0.0)
    if f0 >= -_EPS:
        # Gate was already reached before this realized segment; not a new passage.
        return None

    def search(a: float, fa: float, b: float, fb: float, depth: int) -> float | None:
        if fa >= -_EPS:
            return a
        width = b - a
        # For an L-Lipschitz scalar trajectory with endpoint values fa/fb, the
        # intersection of the two endpoint cones gives a tight interval upper bound:
        # max f(t) <= (fa + fb + L*width)/2.  If that bound is still negative, a
        # directed crossing is impossible even when the trajectory temporarily moves
        # toward and then away from the gate.
        upper_bound = 0.5 * (fa + fb + speed_bound * width)
        if fb < -_EPS and upper_bound < -_EPS:
            return None
        if width <= time_tolerance_min or depth >= max_depth:
            if fb >= -_EPS:
                lo, hi = a, b
                flo = fa
                for _ in range(64):
                    if hi - lo <= time_tolerance_min:
                        break
                    mid = 0.5 * (lo + hi)
                    fm = f(mid)
                    if fm >= -_EPS:
                        hi = mid
                    else:
                        lo, flo = mid, fm
                return hi
            # At the resolution limit with both endpoints negative, sample the midpoint
            # rather than inventing a crossing.  This is fail-closed measurement.
            mid = 0.5 * (a + b)
            fm = f(mid)
            if fm >= -_EPS:
                left = search(a, fa, mid, fm, depth + 1)
                return mid if left is None else left
            return None

        mid = 0.5 * (a + b)
        fm = f(mid)
        left = search(a, fa, mid, fm, depth + 1)
        if left is not None:
            return left
        return search(mid, fm, b, fb, depth + 1)

    return search(0.0, f0, float(duration_min), f(float(duration_min)), 0)


class P4DObservedMetricEnvironmentalWorld(_MetricEnvironmentalWorld):
    """Metric world with passive, non-scheduling P4D first-passage telemetry."""

    def __init__(
        self,
        *args,
        stream_by_vessel: Mapping[str, str],
        gates_by_stream: Mapping[str, tuple[P4DConflictGate, ...]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._p4d_stream_by_vessel = dict(stream_by_vessel)
        self._p4d_gates_by_stream = dict(gates_by_stream or _DEFAULT_GATES_BY_STREAM)
        self._p4d_observations: list[P4DConflictObservation] = []
        self._p4d_seen: set[tuple[str, str]] = set()

    @property
    def p4d_conflict_observations(self) -> tuple[P4DConflictObservation, ...]:
        return tuple(sorted(self._p4d_observations, key=lambda x: (x.observation_time_min, x.vessel_id, x.role)))

    def _advance_all(self, duration: float):  # noqa: SLF001 - passive telemetry subclass
        if duration > _EPS:
            start_time = self.time
            snapshot = self.snapshot()
            for state in snapshot.vessels:
                vessel_id = state.vessel_id
                stream_id = self._p4d_stream_by_vessel.get(vessel_id)
                if stream_id is None:
                    continue
                gates = self._p4d_gates_by_stream.get(stream_id, ())
                remaining = [g for g in gates if (vessel_id, g.role) not in self._p4d_seen]
                if not remaining:
                    continue
                trajectory = build_environmental_candidate_transition(
                    state,
                    self.drift_for(vessel_id),
                    self.command_for(vessel_id),
                    self.limits_for(vessel_id),
                    motion_backend=self.motion_backend,
                    motion_state=self.motion_state_for(vessel_id),
                    evaluation_horizon_min=duration,
                )
                # Gates are topologically ordered along the declared route.  Search only
                # for the next unseen gate.  If it is crossed inside this same native
                # segment, continue to the following gate; otherwise later gates cannot
                # be valid first-passage observations in this segment.
                for gate in remaining:
                    crossing = _first_passage_time(trajectory, gate, duration)
                    if crossing is None:
                        break
                    cross_state = trajectory.state_at_or_continue(crossing)
                    self._p4d_seen.add((vessel_id, gate.role))
                    self._p4d_observations.append(
                        P4DConflictObservation(
                            vessel_id=vessel_id,
                            stream_id=stream_id,
                            role=gate.role,
                            observation_time_min=start_time + crossing,
                            position_nm=tuple(cross_state.position),
                            through_water_heading_deg=cross_state.through_water_heading_deg,
                            through_water_speed_nm_per_min=cross_state.through_water_speed,
                        )
                    )
        return super()._advance_all(duration)
