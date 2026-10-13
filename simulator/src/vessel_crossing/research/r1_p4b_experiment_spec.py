"""Frozen R1 P4B experiment specification and configuration materialization.

Refactor-only module extracted at M0-08.  It owns campaign constants, factor/window
specification, canonical identity helpers, and deterministic microscopic config
construction.  It does not execute scientific points.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from math import isfinite

from vessel_crossing.nextgen import Route2D, RouteNetwork, RouteNetworkEdge, RouteNetworkNode
from vessel_crossing.research import r1_robustness_matrix as stage1
from vessel_crossing.research.r1_p4b_behavior_grounding import P4B_BEHAVIOR_TIMING_ENDPOINTS
from vessel_crossing.research.r1_robustness_stage2 import (
    build_v4_config,
    first_order_profile,
    tabulated_steerage_profile,
)

P4B_CAMPAIGN_ID = "R1_P4B_PUBLICATION_SCALE_V3_2026-09-18"
P4B_PROTOCOL_ID = "R1_P4B_PUBLICATION_PROTOCOL_V3"
P4B_PLAN_FORMAT = "cross-r1-p4b-publication-execution-plan"
P4B_PLAN_SCHEMA_VERSION = 3
P4B_POINT_FORMAT = "cross-r1-p4b-publication-physical-point"
P4B_POINT_SCHEMA_VERSION = 2
P4B_CHECKPOINT_FORMAT = "cross-r1-p4b-publication-checkpoint"
P4B_AUTH_FORMAT = "cross-r1-p4b-scientific-execution-authorization"
P4B_AUTH_SCHEMA_VERSION = 1
P4B_REPLICATION_COUNT = 24
P4B_MAX_PARALLEL_SCIENTIFIC_WORKERS = 2
P4B_TRAFFIC_HORIZON_MIN = 90.0
P4B_PRIMARY_MEASUREMENT_END_MIN = 90.0
P4B_EVALUATION_END_MIN = 150.0
P4B_ROOMS_NM = (1.300, 1.375, 1.600)
P4B_DEMANDS_PER_HOUR = (4.0, 6.0, 8.0)
P4B_PENETRATIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
P4B_P2_PENETRATIONS = (0.0, 0.50, 1.0)
P4B_P2_SENTINEL_ROOM_DEMAND = (
    (1.300, 8.0),
    (1.375, 6.0),
    (1.375, 8.0),
    (1.600, 6.0),
    (1.600, 8.0),
)
P4B_SOURCE_ARCHIVE_SHA_PLACEHOLDER = "0" * 64
P4B_EXECUTION_AUTHORIZATION_STATE = "EXTERNAL_AUTHORIZATION_FILE_REQUIRED_NOT_INCLUDED"
_EPS = 1e-10

def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")

def _sha_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()

def _valid_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def _room_tag(room: float) -> str:
    return f"R{int(round(room * 1000)):04d}"

def _demand_tag(demand: float) -> str:
    return f"D{int(round(demand)):02d}"

def _penetration_tag(penetration: float | None) -> str:
    if penetration is None:
        return "PDEDUP"
    return f"P{int(round(penetration * 100)):03d}"

def _plant_profiles():
    return {
        "P1_FIRST_ORDER_RESPONSE_SYNTHETIC": first_order_profile(),
        "P2_TABULATED_STEERAGE_SYNTHETIC": tabulated_steerage_profile(),
    }

def _timing_endpoint(endpoint_id: str):
    for item in P4B_BEHAVIOR_TIMING_ENDPOINTS:
        if item.endpoint_id == endpoint_id:
            return item
    raise KeyError(endpoint_id)

def _network_for_room(room_nm: float) -> RouteNetwork:
    if not isfinite(room_nm) or room_nm <= 0.0:
        raise ValueError("room_nm must be finite and positive")
    return RouteNetwork(
        (
            RouteNetworkNode("S", (0.0, -room_nm)),
            RouteNetworkNode("N", (0.0, room_nm)),
            RouteNetworkNode("E", (room_nm, 0.0)),
            RouteNetworkNode("W", (-room_nm, 0.0)),
        ),
        (
            RouteNetworkEdge("sn", "S", "N", Route2D(((0.0, -room_nm), (0.0, room_nm)))),
            RouteNetworkEdge("ew", "E", "W", Route2D(((room_nm, 0.0), (-room_nm, 0.0)))),
        ),
    )

def build_publication_config(*, room_nm: float, demand_per_hour: float, traffic_seed: int):
    """Materialize one publication micro config from the qualified V4 Stage-2 runtime."""
    if room_nm not in P4B_ROOMS_NM:
        raise ValueError(f"room_nm must be one of {P4B_ROOMS_NM!r}")
    if demand_per_hour not in P4B_DEMANDS_PER_HOUR:
        raise ValueError(f"demand_per_hour must be one of {P4B_DEMANDS_PER_HOUR!r}")
    if not isinstance(traffic_seed, int) or not 0 <= traffic_seed < 2**63:
        raise ValueError("traffic_seed must be a 63-bit non-negative integer")

    base = build_v4_config(stage1.baseline_point(), stage1.build_stage1_config)
    rate = (demand_per_hour / 2.0) / 60.0
    network = _network_for_room(room_nm)
    streams = tuple(replace(stream, rate_per_minute=rate) for stream in base.streams)
    scenario = replace(
        base.scenario,
        seed=traffic_seed,
        traffic_horizon_min=P4B_TRAFFIC_HORIZON_MIN,
        start_time=0.0,
        end_time=P4B_EVALUATION_END_MIN,
    )
    return replace(
        base,
        network=network,
        streams=streams,
        scenario=scenario,
        label=f"{P4B_PROTOCOL_ID}-{_room_tag(room_nm)}-{_demand_tag(demand_per_hour)}",
    )
