"""Bounded B5 queue-order optimization over the exact qualified A65 B4 policy."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import (
    EnvironmentalNetworkSourceControllerContext,
)
from vessel_crossing.nextgen.environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceDemand,
    EnvironmentalNetworkSourceReservoir,
)

from .b4 import B4EarliestSafeSlotController, B4PolicyError, _EPS

B5_POLICY_ALGORITHM_ID = "cross-b5-bounded-slot-cost-staging-v1"


@dataclass(frozen=True, slots=True)
class B5CandidateEvaluation:
    stream_id: str
    candidate_vessel_id: str
    bypassed_vessel_ids: tuple[str, ...]
    baseline_next_center_min: float
    candidate_next_center_min: float
    classification: str

    @property
    def improvement_min(self) -> float:
        return self.baseline_next_center_min - self.candidate_next_center_min


class B5BoundedSlotCostController(B4EarliestSafeSlotController):
    """Exact cumulative B4→B5 controller with bounded, deterministic staging.

    B5 changes only source service-order evidence. It evaluates a bounded arrived prefix
    of one stream at a time using the same B4 next-slot cost function. A reorder is
    applied only for a strict reduction in the *next global B4 center target*.

    ``max_extra_wait_min`` is a B5-induced bypass deadline: once a bypassed pending
    demand reaches that deadline, B5 promotes it to the front of the pending portion of
    its stream. Canonical physical blocking can still extend real waiting time; B5 does
    not claim control over that physical delay.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._managed_order: dict[str, list[str]] | None = None
        self._emit_managed_order = False
        self._bypass_count: dict[str, int] = {}
        self._bypass_deadline: dict[str, float] = {}
        self._last_candidate_evaluations: tuple[B5CandidateEvaluation, ...] = ()

    @property
    def bypass_count(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._bypass_count))

    @property
    def last_candidate_evaluations(self) -> tuple[B5CandidateEvaluation, ...]:
        return self._last_candidate_evaluations

    def _canonical_order(self, reservoir: EnvironmentalNetworkSourceReservoir) -> dict[str, list[str]]:
        by_stream: dict[str, list[str]] = {}
        for demand in reservoir.demands:
            by_stream.setdefault(demand.stream_id, []).append(demand.vessel_id)
        explicit = dict(reservoir.service_order_by_stream)
        return {stream: list(explicit.get(stream, tuple(ids))) for stream, ids in by_stream.items()}

    def _serialized_order(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        assert self._managed_order is not None
        return tuple((s, tuple(self._managed_order[s])) for s in sorted(self._managed_order))

    def _sync_and_validate(self, reservoir: EnvironmentalNetworkSourceReservoir) -> None:
        if self._managed_order is None:
            self._managed_order = self._canonical_order(reservoir)
            self._emit_managed_order = bool(reservoir.service_order_by_stream)
        elif self._emit_managed_order:
            if reservoir.service_order_by_stream != self._serialized_order():
                raise B4PolicyError("B5 managed service order changed outside the qualified controller")
        elif reservoir.service_order_by_stream:
            raise B4PolicyError("unexpected external service-order mutation before B5 staging")
        self._validate_demand_identity_and_domain(reservoir)

    def _with_managed_order(self, reservoir: EnvironmentalNetworkSourceReservoir) -> EnvironmentalNetworkSourceReservoir:
        if not self._emit_managed_order:
            return reservoir
        return EnvironmentalNetworkSourceReservoir(
            demands=reservoir.demands,
            admissions=reservoir.admissions,
            service_order_by_stream=self._serialized_order(),
        )

    def _pending_by_stream(self, reservoir: EnvironmentalNetworkSourceReservoir) -> dict[str, list[EnvironmentalNetworkSourceDemand]]:
        assert self._managed_order is not None
        pending = {d.vessel_id: d for d in reservoir.pending_demands}
        return {
            stream: [pending[v] for v in order if v in pending]
            for stream, order in self._managed_order.items()
        }

    def _promote_fairness_deadlines(self, reservoir: EnvironmentalNetworkSourceReservoir, now: float) -> None:
        assert self._managed_order is not None
        pending = {d.vessel_id for d in reservoir.pending_demands}
        for vessel_id, deadline in sorted(self._bypass_deadline.items(), key=lambda row: (row[1], row[0])):
            if vessel_id not in pending or deadline > now + _EPS:
                continue
            demand = self._demand_identity[vessel_id]
            order = self._managed_order[demand.stream_id]
            pending_positions = [i for i, vid in enumerate(order) if vid in pending]
            if not pending_positions:
                continue
            first = pending_positions[0]
            current = order.index(vessel_id)
            if current != first:
                order.pop(current)
                order.insert(first, vessel_id)
                self._emit_managed_order = True
                self._record(
                    now,
                    "B5_FAIRNESS_PROMOTION",
                    vessel_id=vessel_id,
                    stream_id=demand.stream_id,
                    detail=f"bypass_deadline_min={deadline}",
                )

    def _global_best(self, heads: Mapping[str, EnvironmentalNetworkSourceDemand], *, now: float):
        anchor = self._effective_anchor()
        candidates = []
        for demand in heads.values():
            if demand.demand_time > now + _EPS:
                continue
            center, certificate = self._slot_cost_for_demand(demand, now=now, anchor=anchor)
            candidates.append((center, demand.demand_time, demand.stream_id, demand.vessel_id, certificate))
        return None if not candidates else min(candidates, key=lambda row: row[:4])

    def _maybe_stage_b5(self, reservoir: EnvironmentalNetworkSourceReservoir, now: float) -> None:
        assert self._managed_order is not None
        self._last_candidate_evaluations = ()
        pending = self._pending_by_stream(reservoir)
        heads = {stream: rows[0] for stream, rows in pending.items() if rows}
        baseline = self._global_best(heads, now=now)
        if baseline is None:
            return
        baseline_center = baseline[0]
        fairness = self.profile.b5_fairness
        evaluations: list[B5CandidateEvaluation] = []
        admissible_help: list[tuple[tuple[float, float, str, str], str, int, EnvironmentalNetworkSourceDemand, tuple[str, ...]]] = []

        for stream_id in sorted(pending):
            rows = pending[stream_id]
            for index, candidate in enumerate(rows[1:fairness.lookahead_depth], start=1):
                if candidate.demand_time > now + _EPS:
                    continue
                bypassed = tuple(row.vessel_id for row in rows[:index])
                if any(self._bypass_count.get(v, 0) + 1 > fairness.max_bypass_per_demand for v in bypassed):
                    continue
                if any(self._bypass_deadline.get(v, float("inf")) <= now + _EPS for v in bypassed):
                    continue
                staged_heads = dict(heads)
                staged_heads[stream_id] = candidate
                candidate_best = self._global_best(staged_heads, now=now)
                if candidate_best is None:
                    continue
                center = candidate_best[0]
                if center < baseline_center - _EPS:
                    classification = "HELP"
                elif center > baseline_center + _EPS:
                    classification = "HURT"
                else:
                    classification = "NULL"
                evaluation = B5CandidateEvaluation(
                    stream_id=stream_id,
                    candidate_vessel_id=candidate.vessel_id,
                    bypassed_vessel_ids=bypassed,
                    baseline_next_center_min=baseline_center,
                    candidate_next_center_min=center,
                    classification=classification,
                )
                evaluations.append(evaluation)
                self._record(
                    now,
                    "B5_CANDIDATE_EVALUATED",
                    vessel_id=candidate.vessel_id,
                    stream_id=stream_id,
                    other_vessel_id=bypassed[0] if bypassed else None,
                    center_target_min=center,
                    detail=f"classification={classification}; baseline={baseline_center}; bypassed={','.join(bypassed)}",
                )
                if classification == "HELP":
                    admissible_help.append((candidate_best[:4], stream_id, index, candidate, bypassed))

        self._last_candidate_evaluations = tuple(evaluations)
        if not admissible_help:
            return
        _, stream_id, _, chosen, bypassed = min(
            admissible_help,
            key=lambda row: (row[0], row[1], row[3].vessel_id),
        )
        order = self._managed_order[stream_id]
        chosen_index = order.index(chosen.vessel_id)
        # Move only within the pending portion. Earlier admitted IDs remain fixed.
        order.pop(chosen_index)
        first_pending = min(i for i, vessel_id in enumerate(order) if vessel_id in {d.vessel_id for d in reservoir.pending_demands})
        order.insert(first_pending, chosen.vessel_id)
        for vessel_id in bypassed:
            self._bypass_count[vessel_id] = self._bypass_count.get(vessel_id, 0) + 1
            self._bypass_deadline.setdefault(vessel_id, now + fairness.max_extra_wait_min)
        self._emit_managed_order = True
        chosen_eval = next(x for x in evaluations if x.candidate_vessel_id == chosen.vessel_id and x.stream_id == stream_id)
        self._record(
            now,
            "B5_SLOT_COST_REORDER",
            vessel_id=chosen.vessel_id,
            stream_id=stream_id,
            other_vessel_id=bypassed[0] if bypassed else None,
            center_target_min=chosen_eval.candidate_next_center_min,
            detail=(
                f"baseline_next_center_min={chosen_eval.baseline_next_center_min}; "
                f"improvement_min={chosen_eval.improvement_min}; bypassed={','.join(bypassed)}"
            ),
        )

    def prepare_reservoir(
        self,
        reservoir: EnvironmentalNetworkSourceReservoir,
        context: EnvironmentalNetworkSourceControllerContext,
    ) -> EnvironmentalNetworkSourceReservoir:
        self._sync_and_validate(reservoir)
        now = self._time(context)
        self._consume_center_observations(now, reservoir)
        pending_ids = {d.vessel_id for d in reservoir.pending_demands}
        if self._reservation is not None and self._reservation.vessel_id not in pending_ids:
            self._reservation = None
        if self._reservation is None:
            self._promote_fairness_deadlines(reservoir, now)
            ordered = self._with_managed_order(reservoir)
            self._maybe_stage_b5(ordered, now)
            ordered = self._with_managed_order(reservoir)
            self._reservation = self._new_reservation(ordered, context)
            return self._apply_gates(ordered, context)
        ordered = self._with_managed_order(reservoir)
        return self._apply_gates(ordered, context)
