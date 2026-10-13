"""Traffic-arrival models.

The legacy diploma attempted to create a Poisson stream by first choosing a
number of ships for a day and then deriving a rate from that same random
number.  In a homogeneous Poisson process the rate is an input parameter;
the number of arrivals over the horizon is an output of the process.
"""

from __future__ import annotations

from math import isfinite
from typing import Protocol

import numpy as np


class ExponentialRng(Protocol):
    """Minimal RNG interface required by :func:`generate_poisson_arrivals`."""

    def exponential(self, scale: float) -> float: ...


def generate_poisson_arrivals(
    rate: float,
    horizon: float,
    rng: ExponentialRng | np.random.Generator,
    *,
    start_time: float = 0.0,
) -> np.ndarray:
    """Generate arrivals of a homogeneous Poisson process.

    Parameters
    ----------
    rate:
        Mean event rate per unit of time.  For example, ``10 / 1440`` means
        ten ships per day when time is measured in minutes.
    horizon:
        Duration of the simulated interval in the same time unit.
    rng:
        Random-number generator providing exponentially distributed samples.
    start_time:
        Absolute time corresponding to the beginning of the interval.

    Returns
    -------
    numpy.ndarray
        Strictly increasing absolute arrival times in the half-open interval
        ``[start_time, start_time + horizon)``.

    Notes
    -----
    Inter-arrival times of a homogeneous Poisson process are independent
    exponential random variables with mean ``1 / rate``.  Consequently, the
    number of generated arrivals over ``horizon`` is random and has expected
    value ``rate * horizon``.
    """

    if not isfinite(rate) or rate < 0:
        raise ValueError("rate must be finite and non-negative")
    if not isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be finite and non-negative")
    if not isfinite(start_time):
        raise ValueError("start_time must be finite")
    if rate == 0 or horizon == 0:
        return np.empty(0, dtype=float)

    end_time = start_time + horizon
    current = start_time
    arrivals: list[float] = []

    while True:
        interarrival = float(rng.exponential(scale=1.0 / rate))
        if not isfinite(interarrival) or interarrival <= 0:
            raise ValueError("rng must return finite positive exponential inter-arrivals")
        current += interarrival
        if current >= end_time:
            break
        arrivals.append(current)

    return np.asarray(arrivals, dtype=float)
