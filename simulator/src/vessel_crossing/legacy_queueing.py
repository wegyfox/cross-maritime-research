"""Legacy-faithful single-server queueing primitives.

This module intentionally models the *service abstraction* used in the 2020
thesis separately from the modern pairwise CPA speed-control policies.

For the no-priority two-stream case, the thesis states that two independent
Poisson vessel streams are superposed and served by one crossing in order of
approach.  With equal vessel speed and a fixed CPA limit, service time is
constant, so the mathematically matching queue is M/D/1 (not M/M/1).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

from .traffic import generate_poisson_arrivals


_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class ServiceRequest:
    """One vessel asking for access to a single intersection server."""

    request_id: str
    stream_id: str
    requested_at: float
    service_time: float
    priority_class: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.stream_id:
            raise ValueError("stream_id must be non-empty")
        if not isfinite(self.requested_at) or self.requested_at < 0:
            raise ValueError("requested_at must be finite and non-negative")
        if not isfinite(self.service_time) or self.service_time <= 0:
            raise ValueError("service_time must be finite and positive")
        if not isinstance(self.priority_class, int) or self.priority_class < 0:
            raise ValueError("priority_class must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ServiceInterval:
    """Scheduled occupation of the single intersection server."""

    request_id: str
    stream_id: str
    requested_at: float
    service_started_at: float
    service_completed_at: float
    priority_class: int = 0

    @property
    def waiting_time(self) -> float:
        return self.service_started_at - self.requested_at

    @property
    def service_time(self) -> float:
        return self.service_completed_at - self.service_started_at

    @property
    def system_time(self) -> float:
        return self.service_completed_at - self.requested_at


@dataclass(frozen=True, slots=True)
class SingleServerRun:
    """Immutable FCFS schedule and simple queueing summaries."""

    intervals: tuple[ServiceInterval, ...]

    @property
    def request_count(self) -> int:
        return len(self.intervals)

    @property
    def mean_waiting_time(self) -> float | None:
        if not self.intervals:
            return None
        return sum(item.waiting_time for item in self.intervals) / len(self.intervals)

    @property
    def mean_system_time(self) -> float | None:
        if not self.intervals:
            return None
        return sum(item.system_time for item in self.intervals) / len(self.intervals)

    @property
    def final_completion_time(self) -> float | None:
        if not self.intervals:
            return None
        return self.intervals[-1].service_completed_at


@dataclass(frozen=True, slots=True)
class FCFSIntersectionServer:
    """Stateless one-server scheduler using request time, then request ID.

    The object deliberately owns no mutable run state.  A schedule is a pure
    function of the supplied requests, which prevents state leakage when the
    same server definition is reused across Monte Carlo replications.
    """

    name: str = "legacy_single_server_fcfs"

    def schedule(self, requests: tuple[ServiceRequest, ...] | list[ServiceRequest]) -> SingleServerRun:
        ordered = tuple(sorted(requests, key=lambda item: (item.requested_at, item.request_id)))
        ids = [item.request_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("service request IDs must be unique")

        available_at = 0.0
        intervals: list[ServiceInterval] = []
        for request in ordered:
            started_at = max(request.requested_at, available_at)
            completed_at = started_at + request.service_time
            intervals.append(
                ServiceInterval(
                    request_id=request.request_id,
                    stream_id=request.stream_id,
                    requested_at=request.requested_at,
                    service_started_at=started_at,
                    service_completed_at=completed_at,
                    priority_class=request.priority_class,
                )
            )
            available_at = completed_at

        return SingleServerRun(tuple(intervals))



@dataclass(frozen=True, slots=True)
class PriorityReservationIntersectionServer:
    """Two-class non-preemptive reservation server for the recovered thesis logic.

    Requests with ``priority_class >= priority_threshold`` reserve the server
    first, in their own request-time order.  Ordinary FCFS requests are then
    placed into the earliest remaining gaps, but only when their *entire*
    service interval fits before the next privileged reservation.

    This look-ahead reservation rule encodes the recovered thesis requirement
    that a privileged arrival should not find the crossing occupied by a
    non-priority vessel.  Privileged requests can still queue behind earlier
    privileged requests when priority demand itself exceeds capacity.
    """

    priority_threshold: int = 1
    name: str = "legacy_single_server_priority_reservation"

    def __post_init__(self) -> None:
        if self.priority_threshold < 1:
            raise ValueError("priority_threshold must be at least 1")

    @staticmethod
    def _interval(request: ServiceRequest, start: float) -> ServiceInterval:
        return ServiceInterval(
            request_id=request.request_id,
            stream_id=request.stream_id,
            requested_at=request.requested_at,
            service_started_at=start,
            service_completed_at=start + request.service_time,
            priority_class=request.priority_class,
        )

    def schedule(self, requests: tuple[ServiceRequest, ...] | list[ServiceRequest]) -> SingleServerRun:
        requests = tuple(requests)
        ids = [item.request_id for item in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("service request IDs must be unique")

        privileged = sorted(
            (item for item in requests if item.priority_class >= self.priority_threshold),
            key=lambda item: (item.requested_at, item.request_id),
        )
        ordinary = sorted(
            (item for item in requests if item.priority_class < self.priority_threshold),
            key=lambda item: (item.requested_at, item.request_id),
        )

        reservations: list[ServiceInterval] = []
        privileged_available = 0.0
        for request in privileged:
            start = max(request.requested_at, privileged_available)
            interval = self._interval(request, start)
            reservations.append(interval)
            privileged_available = interval.service_completed_at

        ordinary_intervals: list[ServiceInterval] = []
        ordinary_available = 0.0
        for request in ordinary:
            candidate = max(request.requested_at, ordinary_available)
            for reservation in reservations:
                if reservation.service_completed_at <= candidate + _EPSILON:
                    continue
                if candidate + request.service_time <= reservation.service_started_at + _EPSILON:
                    break
                candidate = max(candidate, reservation.service_completed_at)

            interval = self._interval(request, candidate)
            ordinary_intervals.append(interval)
            ordinary_available = interval.service_completed_at

        combined = tuple(
            sorted(
                (*reservations, *ordinary_intervals),
                key=lambda item: (item.service_started_at, -item.priority_class, item.request_id),
            )
        )
        for previous, current in zip(combined, combined[1:]):
            if current.service_started_at < previous.service_completed_at - _EPSILON:
                raise AssertionError("priority reservation schedule contains overlapping service intervals")
        return SingleServerRun(combined)


def reconfigure_service_requests(
    requests: tuple[ServiceRequest, ...] | list[ServiceRequest],
    *,
    service_time_by_stream: Mapping[str, float],
    priority_class_by_stream: Mapping[str, int] | None = None,
) -> tuple[ServiceRequest, ...]:
    """Reuse one arrival realization under a different service discipline.

    Request IDs and approach timestamps remain identical.  Only service-time
    and priority metadata are replaced, which is useful for paired comparison
    of FCFS and priority-reservation rules on the same stochastic arrivals.
    """

    priorities = {} if priority_class_by_stream is None else dict(priority_class_by_stream)
    result: list[ServiceRequest] = []
    for request in requests:
        try:
            service_time = float(service_time_by_stream[request.stream_id])
        except KeyError as exc:
            raise ValueError(f"missing service time for stream {request.stream_id!r}") from exc
        priority_class = int(priorities.get(request.stream_id, 0))
        result.append(
            ServiceRequest(
                request_id=request.request_id,
                stream_id=request.stream_id,
                requested_at=request.requested_at,
                service_time=service_time,
                priority_class=priority_class,
            )
        )
    return tuple(result)

def crossing_service_time_minutes(
    *,
    cpa_limit_nm: float,
    speed_knots: float,
    occupation_multiplier: float = 3.0,
) -> float:
    """Return thesis-style crossing occupation time in minutes.

    The recovered thesis derives ``D = 1.5 * CPA`` and therefore
    ``t = 2D/v = 3*CPA/v`` for the ordinary no-priority service interval.
    ``occupation_multiplier`` is explicit so the later priority reconstruction
    can hand-check the thesis' ``4.5*CPA/v`` privileged interval without
    embedding another hidden default.
    """

    if not isfinite(cpa_limit_nm) or cpa_limit_nm <= 0:
        raise ValueError("cpa_limit_nm must be finite and positive")
    if not isfinite(speed_knots) or speed_knots <= 0:
        raise ValueError("speed_knots must be finite and positive")
    if not isfinite(occupation_multiplier) or occupation_multiplier <= 0:
        raise ValueError("occupation_multiplier must be finite and positive")
    return 60.0 * occupation_multiplier * cpa_limit_nm / speed_knots


def generate_poisson_service_requests(
    *,
    stream_rates: Mapping[str, float],
    horizon: float,
    service_time: float,
    rng: np.random.Generator,
) -> tuple[ServiceRequest, ...]:
    """Generate independent Poisson request streams and merge by approach time."""

    if horizon < 0 or not isfinite(horizon):
        raise ValueError("horizon must be finite and non-negative")
    if service_time <= 0 or not isfinite(service_time):
        raise ValueError("service_time must be finite and positive")
    if not stream_rates:
        return ()

    requests: list[ServiceRequest] = []
    for stream_id in sorted(stream_rates):
        rate = float(stream_rates[stream_id])
        if not stream_id:
            raise ValueError("stream IDs must be non-empty")
        if rate < 0 or not isfinite(rate):
            raise ValueError("stream rates must be finite and non-negative")
        arrivals = generate_poisson_arrivals(rate, horizon, rng)
        for index, requested_at in enumerate(arrivals):
            requests.append(
                ServiceRequest(
                    request_id=f"{stream_id}-{index:08d}",
                    stream_id=stream_id,
                    requested_at=float(requested_at),
                    service_time=service_time,
                )
            )

    return tuple(sorted(requests, key=lambda item: (item.requested_at, item.request_id)))


def traffic_intensity(*, arrival_rate: float, service_time: float) -> float:
    """Return one-server utilization ``rho = lambda * E[S]``."""

    if arrival_rate < 0 or not isfinite(arrival_rate):
        raise ValueError("arrival_rate must be finite and non-negative")
    if service_time <= 0 or not isfinite(service_time):
        raise ValueError("service_time must be finite and positive")
    return arrival_rate * service_time


def md1_mean_waiting_time(*, arrival_rate: float, service_time: float) -> float:
    """Steady-state M/D/1 mean queue waiting time.

    Pollaczek-Khinchine for deterministic service ``S=s`` gives
    ``Wq = lambda*s^2 / (2*(1-rho)) = rho*s/(2*(1-rho))``.
    """

    rho = traffic_intensity(arrival_rate=arrival_rate, service_time=service_time)
    if rho >= 1.0 - _EPSILON:
        raise ValueError("M/D/1 steady-state mean is undefined for rho >= 1")
    return rho * service_time / (2.0 * (1.0 - rho))


def legacy_mm1_mean_waiting_time(*, arrival_rate: float, service_time: float) -> float:
    """Return the M/M/1-form expression used in the 2020 thesis.

    This is kept only as an explicit historical reference.  It is not the
    matching formula for deterministic service times.
    """

    rho = traffic_intensity(arrival_rate=arrival_rate, service_time=service_time)
    if rho >= 1.0 - _EPSILON:
        raise ValueError("steady-state mean is undefined for rho >= 1")
    return rho * service_time / (1.0 - rho)
