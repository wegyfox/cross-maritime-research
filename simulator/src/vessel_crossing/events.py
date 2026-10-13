"""Priority-ordered events for the simulation lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
from itertools import count


class EventKind(str, Enum):
    COMPLETION = "completion"
    CONFLICT_ZONE_EXIT = "conflict_zone_exit"
    CONFLICT_ZONE_ENTRY = "conflict_zone_entry"
    RELEASE = "release"
    ARRIVAL = "arrival"
    CONTROL_REVIEW = "control_review"


_EVENT_PRIORITY = {
    EventKind.COMPLETION: 0,
    EventKind.CONFLICT_ZONE_EXIT: 1,
    EventKind.CONFLICT_ZONE_ENTRY: 2,
    EventKind.RELEASE: 3,
    EventKind.ARRIVAL: 4,
    EventKind.CONTROL_REVIEW: 5,
}


@dataclass(order=True, frozen=True, slots=True)
class SimulationEvent:
    """One event ordered by time, semantic priority, then insertion order.

    ``generation`` is a version token owned by ``Simulation``.  When a speed
    command changes an analytical prediction, a new generation is scheduled;
    the old heap item remains physically present but is ignored as stale.
    """

    time: float
    priority: int
    sequence: int
    kind: EventKind = field(compare=False)
    vessel_id: str = field(compare=False)
    generation: int = field(default=0, compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[SimulationEvent] = []
        self._sequence = count()

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def push(
        self,
        time: float,
        kind: EventKind,
        vessel_id: str,
        *,
        generation: int = 0,
    ) -> SimulationEvent:
        if time < 0:
            raise ValueError("event time must be non-negative")
        if generation < 0:
            raise ValueError("event generation must be non-negative")
        event = SimulationEvent(
            time=time,
            priority=_EVENT_PRIORITY[kind],
            sequence=next(self._sequence),
            kind=kind,
            vessel_id=vessel_id,
            generation=generation,
        )
        heapq.heappush(self._heap, event)
        return event

    def pop(self) -> SimulationEvent:
        if not self._heap:
            raise IndexError("pop from empty event queue")
        return heapq.heappop(self._heap)

    def peek(self) -> SimulationEvent:
        if not self._heap:
            raise IndexError("peek from empty event queue")
        return self._heap[0]
