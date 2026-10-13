"""Strict-balanced deterministic B6v2 saturated reference.

This module is a post-B6v1 semantic correction frozen before B6v2 outcomes.  It
preserves the qualified R2 physics/safety/pair-context runtime and adds only an
explicit north/west admission-turn constraint so saturated equal demand cannot
collapse into single-stream starvation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Mapping

from vessel_crossing.nextgen import (
    EncounterRegistry,
    EnvironmentalNetworkSourceFutureIngressConfig,
    EnvironmentalNetworkSourceReservoir,
    VesselProfileCatalog,
    start_environmental_network_source_fifo_runtime,
)
from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import EnvironmentalNetworkSourceControllerContext
from vessel_crossing.nextgen.environmental_network_source_reservoir import EnvironmentalNetworkSourceResolution
from vessel_crossing.nextgen.pair_context_following import PairContextFollowingSupervisor
from vessel_crossing.research import r1_robustness_matrix as stage1
from vessel_crossing.research.r1_p4b_engineering_preflight import preflight_classification_config
from vessel_crossing.research.r1_p4b_publication_execution import build_publication_config
from vessel_crossing.research.r1_p4d_conflict_observer import P4DObservedMetricEnvironmentalWorld
from vessel_crossing.research.r1_p4d_lifecycle_audit import capture_terminal_lifecycle_audit
from vessel_crossing.research.r1_p4d_screening import _conflict_summary, _route_completion_records
from vessel_crossing.research.r1_robustness_stage2 import backend_for, first_order_profile
from vessel_crossing.research.r2_b0_b6_authoritative import (
    B6_DEMANDS_PER_STREAM,
    B6_DRAIN_END_MIN,
    B6_MEASUREMENT_END_MIN,
    B6_MEASUREMENT_START_MIN,
    B6_RELEASE_CUTOFF_MIN,
    B6_WARMUP_MIN,
    _EPS,
    _fixed_center_window,
    _safety_doc,
    _saturated_demand,
    _source_doc,
    _terminal_doc,
    canonical_json,
)
from vessel_crossing.research.r2_b0_b6_reference_controller import (
    R2B4B5SourceManagementController,
    R2ManagedReferenceConfig,
    _Reservation,
    _SlotAnchor,
    controller_trace_as_dicts,
)
from vessel_crossing.research.r2_fs2_full_network_publication import (
    AS,
    HS,
    FS2_DEMAND_PER_HOUR,
    FS2_ROOM_NM,
    pair_context_config_for_arm,
)

B6V2_CAMPAIGN_ID = "R2_B6V2_STRICT_BALANCED_SATURATED_REFERENCE_V1_2026-09-30"
B6V2_POINT_FORMAT = "cross-r2-b6v2-strict-balanced-point-v1"
_STREAM_CYCLE = ("north", "west")


class StrictBalancedB6Controller(R2B4B5SourceManagementController):
    """B4/B5 controller with a strict alternating physical-stream turn.

    Only the current turn may receive a new reservation.  The turn advances only
    after an actual canonical admission.  Base B4 competing-head gating then holds
    the other stream out of the same resolution; no unsafe admission is forced.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._turn_index = 0
        self._balanced_trace: list[dict[str, object]] = []

    @property
    def current_turn(self) -> str:
        return _STREAM_CYCLE[self._turn_index]

    @property
    def balanced_trace(self) -> tuple[dict[str, object], ...]:
        return tuple(self._balanced_trace)

    def _maybe_stage_b5(self, reservoir, context) -> None:
        # Preserve the frozen B5 algorithm, but staging is allowed only inside the
        # current-turn stream.  This prevents future-turn order mutation.
        if self.config.level != "B5" or self.config.delta_arm == "ZERO":
            return
        now = float(getattr(context.snapshot, "time"))
        pending = self._pending_by_stream(reservoir)
        stream_id = self.current_turn
        rows = pending.get(stream_id, [])
        if len(rows) < self.config.staging_depth:
            return
        first, second = rows[0], rows[1]
        if first.demand_time > now + _EPS or second.demand_time > now + _EPS:
            return
        if self._bypass_count.get(first.vessel_id, 0) >= self.config.max_bypass_per_demand:
            return
        anchor_type = self._last_admitted_type_by_stream.get(stream_id)
        if anchor_type is None:
            return
        first_type = self._type_by_vessel.get(first.vessel_id)
        second_type = self._type_by_vessel.get(second.vessel_id)
        if first_type is None or second_type is None:
            raise RuntimeError("B6v2 B5 requires a type label for every staged demand")
        if self.config.delta_arm == "NEG050":
            first_preferred = first_type != anchor_type
            second_preferred = second_type != anchor_type
        else:
            first_preferred = first_type == anchor_type
            second_preferred = second_type == anchor_type
        if first_preferred or not second_preferred:
            return
        if not self._canonical_physical_admissible(first, context):
            return
        if not self._canonical_physical_admissible(second, context):
            return
        order = self._service_order_by_stream[stream_id]
        first_index = order.index(first.vessel_id)
        second_index = order.index(second.vessel_id)
        if second_index != first_index + 1:
            raise RuntimeError("B6v2 B5 candidates ceased to be adjacent")
        order[first_index], order[second_index] = order[second_index], order[first_index]
        self._bypass_count[first.vessel_id] = self._bypass_count.get(first.vessel_id, 0) + 1
        self._record(
            now,
            "B5_ADJACENT_SWAP",
            vessel_id=second.vessel_id,
            other_vessel_id=first.vessel_id,
            stream_id=stream_id,
            detail=f"{anchor_type}->{second_type}; bypass_count={self._bypass_count[first.vessel_id]}",
        )

    def _new_reservation(self, reservoir, context):
        now = float(getattr(context.snapshot, "time"))
        pending = self._pending_by_stream(reservoir)
        rows = pending.get(self.current_turn, [])
        heads = [rows[0]] if rows and rows[0].demand_time <= now + _EPS else []
        if not heads:
            return None
        anchor = self._effective_anchor()
        demand = heads[0]
        center_target = now + self.config.source_to_center_free_flow_min
        if anchor is not None:
            center_target = max(center_target, anchor.center_time_min + self._slot_separation(anchor, demand))
        domain = None if demand.profile is None else demand.profile.safety_domain
        if domain is None:
            raise RuntimeError("B6v2 requires explicit safety domain")
        release_target = center_target - self.config.source_to_center_free_flow_min
        reservation = _Reservation(
            anchor=_SlotAnchor(
                vessel_id=demand.vessel_id,
                stream_id=demand.stream_id,
                center_time_min=center_target,
                heading_deg=float(demand.vessel.through_water_heading_deg),
                domain=domain,
            ),
            release_time_min=max(now, release_target),
        )
        self._record(
            now,
            "B4_SLOT_RESERVED",
            vessel_id=demand.vessel_id,
            stream_id=demand.stream_id,
            center_target_min=center_target,
            release_target_min=reservation.release_time_min,
            detail=f"B6V2_BALANCED_TURN={self.current_turn}",
        )
        return reservation

    def observe_resolution(self, resolution, context) -> None:
        turn_before = self.current_turn
        admitted_turn = [x for x in resolution.admissions if x.stream_id == turn_before]
        other = [x for x in resolution.admissions if x.stream_id != turn_before]
        if other:
            raise RuntimeError(f"B6v2 balanced-turn violation: admitted non-turn stream {other[0].stream_id}")
        if len(admitted_turn) > 1:
            raise RuntimeError("B6v2 balanced-turn violation: more than one turn admission in one resolution")
        super().observe_resolution(resolution, context)
        if admitted_turn:
            self._turn_index = (self._turn_index + 1) % len(_STREAM_CYCLE)
            self._balanced_trace.append({
                "time_min": float(resolution.snapshot_time),
                "event": "B6V2_TURN_ADVANCE",
                "admitted_vessel_id": admitted_turn[0].vessel_id,
                "from_stream": turn_before,
                "to_stream": self.current_turn,
            })


class BalancedReleaseCutoffController:
    """Close releases after cutoff without mutating B5 staging post-cutoff."""

    def __init__(self, inner: StrictBalancedB6Controller) -> None:
        self.inner = inner
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def prepare_reservoir(self, reservoir, context):
        if not self.closed:
            return self.inner.prepare_reservoir(reservoir, context)
        pending = {d.vessel_id for d in reservoir.pending_demands}
        changed = tuple(
            replace(d, release_not_before_time=(1.0e9 if d.vessel_id in pending else d.release_not_before_time))
            for d in reservoir.demands
        )
        return EnvironmentalNetworkSourceReservoir(
            demands=changed,
            admissions=reservoir.admissions,
            service_order_by_stream=reservoir.service_order_by_stream,
        )

    def observe_resolution(self, resolution, context) -> None:
        if not self.closed:
            self.inner.observe_resolution(resolution, context)


def _prefix_imbalance(admissions) -> int:
    counts = Counter()
    worst = 0
    for row in admissions:
        counts[row.stream_id] += 1
        worst = max(worst, abs(counts["north"] - counts["west"]))
    return worst


def run_b6v2_point(*, delta_arm: str, plan_sha256: str, protocol_sha256: str,
                    demands_per_stream: int = B6_DEMANDS_PER_STREAM,
                    release_cutoff_min: float = B6_RELEASE_CUTOFF_MIN,
                    drain_end_min: float = B6_DRAIN_END_MIN,
                    measurement_start_min: float = B6_MEASUREMENT_START_MIN,
                    measurement_end_min: float = B6_MEASUREMENT_END_MIN) -> bytes:
    if delta_arm not in {"NEG050", "ZERO", "POS050"}:
        raise ValueError(delta_arm)
    cfg = build_publication_config(room_nm=FS2_ROOM_NM, demand_per_hour=FS2_DEMAND_PER_HOUR, traffic_seed=0)
    cycle = (AS, AS, HS, HS)
    demands = []
    type_map: dict[str, str] = {}
    for stream_id in _STREAM_CYCLE:
        for index in range(1, demands_per_stream + 1):
            vessel_id = f"{stream_id}-b6v2-{index:03d}"
            type_map[vessel_id] = cycle[(index - 1) % len(cycle)]
            demands.append(_saturated_demand(cfg, vessel_id=vessel_id, stream_id=stream_id))
    demands.sort(key=lambda d: (d.demand_time, d.vessel_id))
    reservoir = EnvironmentalNetworkSourceReservoir(tuple(demands))
    level = "B4" if delta_arm == "ZERO" else "B5"
    response = pair_context_config_for_arm(type_map, delta_arm=delta_arm)
    following = PairContextFollowingSupervisor(response)
    profiles = VesselProfileCatalog()
    motion_backend = backend_for(first_order_profile())
    world = P4DObservedMetricEnvironmentalWorld(
        (), limits={}, environmental_drift_velocity={}, start_time=0.0,
        profile_catalog=profiles, safety_config=stage1.stage1_safety_metric_config(),
        motion_backend=motion_backend, retain_safety_intervals=False,
        stream_by_vessel={d.vessel_id:d.stream_id for d in demands},
    )
    registry = EncounterRegistry(discovery=cfg.scenario.discovery, lifecycle=cfg.scenario.lifecycle, profile_catalog=profiles)
    inner = StrictBalancedB6Controller(
        config=R2ManagedReferenceConfig(level=level, delta_arm=delta_arm),
        type_by_vessel=type_map,
        center_observation_provider=lambda: world.p4d_conflict_observations,
    )
    controller = BalancedReleaseCutoffController(inner)
    pair = cfg.scenario.recapture.joint_cycle.replanning.semantic_config.physics.pair_clearance
    started = start_environmental_network_source_fifo_runtime(
        registry, world, {}, reservoir,
        recapture_config=cfg.scenario.recapture,
        runtime_config=cfg.scenario.runtime,
        recheck_config=cfg.recheck,
        classification_config=preflight_classification_config(),
        profile_catalog=profiles,
        same_route_dynamic_ingress=following,
        future_ingress_clearance=EnvironmentalNetworkSourceFutureIngressConfig(pair_clearance=pair),
        persistent_same_route_following=following,
        source_admission_controller=controller,
    )
    pre = started.session.run_until_capturing_terminal_outcome(release_cutoff_min)
    if pre.status.value == "SUCCESS" and pre.terminal_time_min >= release_cutoff_min - _EPS:
        controller.close()
        outcome = started.session.run_until_capturing_terminal_outcome(drain_end_min)
    else:
        outcome = pre
    runtime = started.runtime_start.session
    completions = _route_completion_records(runtime, outcome.reservoir, at=outcome.terminal_time_min)
    conflict = _conflict_summary(world.p4d_conflict_observations, outcome.reservoir.admissions, room_nm=FS2_ROOM_NM)
    lifecycle = capture_terminal_lifecycle_audit(runtime, outcome.reservoir, completions)
    if any(x.admission_time > release_cutoff_min + _EPS for x in outcome.reservoir.admissions):
        raise RuntimeError("B6v2 release cutoff violation")
    admissions_by_stream = Counter(x.stream_id for x in outcome.reservoir.admissions)
    primary = _fixed_center_window(world.p4d_conflict_observations, start_min=measurement_start_min, end_min=measurement_end_min)
    passages_by_stream = Counter(x["stream_id"] for x in primary["passages"])
    document = {
        "format": B6V2_POINT_FORMAT,
        "schema_version": 1,
        "scientific_role": "STRICT_BALANCED_SATURATED_FULL_NETWORK_REFERENCE_NOT_FIELD_CAPACITY",
        "campaign_id": B6V2_CAMPAIGN_ID,
        "level": "B6V2",
        "delta_arm": delta_arm,
        "plan_sha256": plan_sha256,
        "protocol_sha256": protocol_sha256,
        "demand_contract": {
            "streams": list(_STREAM_CYCLE),
            "demands_per_stream": demands_per_stream,
            "all_demand_time_min": 0.0,
            "type_cycle_per_stream": list(cycle),
            "controller_level": level,
            "stream_turn_cycle": list(_STREAM_CYCLE),
            "initial_turn": "north",
            "advance_turn_only_after_actual_canonical_admission": True,
            "no_turn_skip_on_temporary_infeasibility": True,
            "warmup_min": B6_WARMUP_MIN,
            "measurement_window_min": [measurement_start_min, measurement_end_min],
            "release_cutoff_min": release_cutoff_min,
            "drain_end_min": drain_end_min,
        },
        "runtime_semantics": {
            "canonical_physics_runtime_unchanged": True,
            "reference_headway_forcing": False,
            "future_demand_lookahead": False,
            "canonical_source_clearance_final_gate": True,
            "pair_context_config_sha256": following.config_identity_sha256,
        },
        "terminal": _terminal_doc(outcome),
        "source": _source_doc(outcome),
        "lifecycle": lifecycle,
        "conflict": conflict,
        "primary_kpi": primary,
        "fairness": {
            "admission_count_by_stream": dict(admissions_by_stream),
            "measurement_passage_count_by_stream": dict(passages_by_stream),
            "max_prefix_admission_count_imbalance": _prefix_imbalance(outcome.reservoir.admissions),
            "strict_balance_gate_pass": _prefix_imbalance(outcome.reservoir.admissions) <= 1,
        },
        "controller": {
            "trace": list(controller_trace_as_dicts(inner.trace)),
            "trace_count": len(inner.trace),
            "balanced_trace": list(inner.balanced_trace),
            "bypass_count": dict(inner.bypass_count),
            "release_cutoff_applied": controller.closed,
        },
        "safety": _safety_doc(world),
        "claim_scope": {
            "field_calibrated_capacity": False,
            "deployment_safety": False,
            "unconstrained_aggregate_upper_ceiling": False,
            "stochastic_confidence_interval": False,
        },
    }
    return canonical_json(document)
