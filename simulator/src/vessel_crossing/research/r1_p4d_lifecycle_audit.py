"""Research-only terminal lifecycle conservation audit for R1 P4D.

This module does not alter runtime ownership or motion.  It inspects the already-reached
terminal boundary and partitions every source-admitted vessel into one of two expected
states:

* route-completed/disposed; or
* still physically active under exactly one live command owner.

The audit is intentionally stricter than the historical P4B ``terminal=SUCCESS`` label,
which only means the runtime reached the requested horizon without a modeled/resource
terminal.  A SUCCESS run can still contain active vessels at the horizon.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence

from vessel_crossing.nextgen.environmental_mixed_owner_control import EnvironmentalMixedRouteStage


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _cursor_payload(cursor: object | None) -> dict[str, object] | None:
    if cursor is None:
        return None
    follower = getattr(cursor, "follower_state", None)
    projection = getattr(cursor, "projection", None)
    return {
        "mode": _enum_value(getattr(cursor, "mode", None)),
        "next_review_at": getattr(cursor, "next_review_at", None),
        "follower_along_track_nm": None if follower is None else float(follower.along_track),
        "follower_completed": None if follower is None else bool(follower.completed),
        "projection_along_track_nm": None if projection is None else float(projection.along_track),
        "projection_cross_track_nm": None if projection is None else float(projection.cross_track),
        "projection_distance_nm": None if projection is None else float(projection.distance),
    }


def capture_terminal_lifecycle_audit(
    runtime,
    reservoir,
    completion_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Capture a read-only ownership/lifecycle partition at the current runtime boundary."""
    persistent = runtime.persistent
    mixed = persistent.mixed_session
    snapshot = runtime.world.snapshot()

    admitted_ids = {str(x.vessel_id) for x in reservoir.admissions}
    completed_ids = {str(x["vessel_id"]) for x in completion_records}
    active_state = {str(v.vessel_id): v for v in snapshot.vessels}
    active_ids = set(active_state)

    avoidance_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    claims: dict[str, list[dict[str, object]]] = {}

    for started in mixed._avoidance_started():  # noqa: SLF001 - research audit only
        owner_id = str(started.owner.component_id)
        if hasattr(started, "pair_cycle"):
            owner_kind = "PAIR_AVOIDANCE"
            stage = _enum_value(started.pair_cycle.stage)
            next_review = getattr(started.guarded_session, "next_review_time", None)
        else:
            owner_kind = "JOINT_AVOIDANCE"
            stage = _enum_value(started.joint_cycle.stage)
            next_review = getattr(started.guarded_session, "next_review_time", None)
        vessel_ids = tuple(str(v) for v in started.owner.vessel_ids)
        row = {
            "owner_id": owner_id,
            "owner_kind": owner_kind,
            "stage": stage,
            "vessel_ids": list(vessel_ids),
            "next_review_at": next_review,
        }
        avoidance_rows.append(row)
        for vessel_id in vessel_ids:
            claims.setdefault(vessel_id, []).append({
                "owner_id": owner_id, "owner_kind": owner_kind, "stage": stage,
            })

    for owner_id, owner in sorted(mixed._route_owners.items()):  # noqa: SLF001 - research audit only
        if owner.stage is EnvironmentalMixedRouteStage.COMPLETE:
            continue
        vessel_ids = tuple(str(v) for v in owner.vessel_ids)
        cursor_rows: list[dict[str, object]] = []
        if owner.stage is EnvironmentalMixedRouteStage.ROUTE_HANDOFF:
            session = owner.handoff_session
            cursors = getattr(session, "_cursors", {})  # noqa: SLF001
            for vessel_id in vessel_ids:
                cursor_rows.append({"vessel_id": vessel_id, "cursor": _cursor_payload(cursors.get(vessel_id))})
        else:
            session = owner.nominal_session
            cursors = {} if session is None else getattr(session, "_cursors", {})  # noqa: SLF001
            for vessel_id in vessel_ids:
                cursor_rows.append({"vessel_id": vessel_id, "cursor": _cursor_payload(cursors.get(vessel_id))})
        row = {
            "owner_id": str(owner_id),
            "owner_kind": "ROUTE",
            "stage": _enum_value(owner.stage),
            "vessel_ids": list(vessel_ids),
            "next_fleet_review_time": owner.next_fleet_review_time,
            "cursors": cursor_rows,
        }
        route_rows.append(row)
        for vessel_id in vessel_ids:
            claims.setdefault(vessel_id, []).append({
                "owner_id": str(owner_id), "owner_kind": "ROUTE", "stage": _enum_value(owner.stage),
            })

    pending_source_rows: list[dict[str, object]] = []
    for vessel_id, owner_id in sorted(
        getattr(mixed, "_pending_source_control_ids", {}).items()  # noqa: SLF001 - audit seam
    ):
        row = {
            "owner_id": str(owner_id),
            "owner_kind": "SOURCE_PENDING_RECAPTURE",
            "stage": "pending_recapture",
            "vessel_ids": [str(vessel_id)],
        }
        pending_source_rows.append(row)
        claims.setdefault(str(vessel_id), []).append({
            "owner_id": str(owner_id),
            "owner_kind": "SOURCE_PENDING_RECAPTURE",
            "stage": "pending_recapture",
        })

    active_rows: list[dict[str, object]] = []
    for vessel_id in sorted(active_ids):
        state = active_state[vessel_id]
        active_rows.append({
            "vessel_id": vessel_id,
            "position_nm": [float(state.position[0]), float(state.position[1])],
            "through_water_heading_deg": float(state.through_water_heading_deg),
            "through_water_speed_nm_per_min": float(state.through_water_speed),
            "owner_claims": claims.get(vessel_id, []),
        })

    missing_ids = admitted_ids - active_ids - completed_ids
    active_not_admitted = active_ids - admitted_ids
    completed_still_active = completed_ids & active_ids
    active_admitted = admitted_ids & active_ids
    ownerless_active = {v for v in active_admitted if len(claims.get(v, ())) == 0}
    multiply_owned_active = {v for v in active_admitted if len(claims.get(v, ())) > 1}
    claimed_inactive = {v for v in claims if v not in active_ids}

    lifecycle_conserved = not missing_ids and not completed_still_active
    ownership_conserved = not ownerless_active and not multiply_owned_active and not claimed_inactive

    return {
        "format": "cross-r1-p4d-terminal-lifecycle-audit-v1",
        "terminal_time_min": float(runtime.world.time),
        "admitted_count": len(admitted_ids),
        "completed_count": len(completed_ids),
        "active_count": len(active_ids),
        "admitted_ids": sorted(admitted_ids),
        "completed_ids": sorted(completed_ids),
        "active_ids": sorted(active_ids),
        "active_not_admitted_ids": sorted(active_not_admitted),
        "missing_from_active_and_completion_ids": sorted(missing_ids),
        "completed_still_active_ids": sorted(completed_still_active),
        "ownerless_active_admitted_ids": sorted(ownerless_active),
        "multiply_owned_active_admitted_ids": sorted(multiply_owned_active),
        "claimed_but_inactive_ids": sorted(claimed_inactive),
        "lifecycle_conserved": lifecycle_conserved,
        "ownership_conserved": ownership_conserved,
        "all_admitted_accounted_for": lifecycle_conserved and ownership_conserved,
        "avoidance_owners": avoidance_rows,
        "route_owners": route_rows,
        "pending_source_owners": pending_source_rows,
        "active_states": active_rows,
    }
