"""Canonical topic-agnostic scenario representation for the CROSS research testbed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Mapping

from .world_objects import CanonicalWorldObject, VesselObjectClass


class ScenarioVesselRole(str, Enum):
    CONTROLLED = "CONTROLLED"
    TRAFFIC = "TRAFFIC"


@dataclass(frozen=True, slots=True)
class ScenarioLineage:
    source_kind: str
    source_version: str | None
    source_uri: str | None
    source_sha256: str
    license_id: str | None
    adapter_id: str
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.source_kind:
            raise ValueError("source_kind must be non-empty")
        if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256.lower()):
            raise ValueError("source_sha256 must be a 64-character hexadecimal digest")
        if not self.adapter_id:
            raise ValueError("adapter_id must be non-empty")
        if tuple(sorted(self.attributes)) != self.attributes:
            raise ValueError("lineage attributes must be sorted for canonical identity")

    def attribute_map(self) -> Mapping[str, str]:
        return dict(self.attributes)


@dataclass(frozen=True, slots=True)
class CanonicalCoordinateSystem:
    frame_id: str
    origin_lat_deg: float | None = None
    origin_lon_deg: float | None = None
    projection: str = "LOCAL_ENU"
    horizontal_unit: str = "nautical_mile"
    heading_convention: str = "marine_deg_clockwise_from_true_north"
    speed_unit: str = "knot"
    time_unit: str = "minute_from_epoch"
    hull_dimension_unit: str = "metre"

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")
        if (self.origin_lat_deg is None) != (self.origin_lon_deg is None):
            raise ValueError("geographic origin requires both latitude and longitude")
        if self.origin_lat_deg is not None and (not isfinite(self.origin_lat_deg) or not -90 <= self.origin_lat_deg <= 90):
            raise ValueError("origin_lat_deg must be within [-90, 90]")
        if self.origin_lon_deg is not None and (not isfinite(self.origin_lon_deg) or not -180 <= self.origin_lon_deg <= 180):
            raise ValueError("origin_lon_deg must be within [-180, 180]")


@dataclass(frozen=True, slots=True)
class CanonicalDimensions:
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    draught_m: float | None = None
    a_m: float | None = None
    b_m: float | None = None
    c_m: float | None = None
    d_m: float | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when provided")


@dataclass(frozen=True, slots=True)
class CanonicalWaypoint:
    x_east_nm: float
    y_north_nm: float
    turn_radius_nm: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.x_east_nm) or not isfinite(self.y_north_nm):
            raise ValueError("waypoint coordinates must be finite")
        if self.turn_radius_nm is not None and (not isfinite(self.turn_radius_nm) or self.turn_radius_nm < 0):
            raise ValueError("turn_radius_nm must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CanonicalSpeedProfile:
    value_knots: float
    interp_start_nm: float | None = None
    interp_end_nm: float | None = None
    interp_method: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.value_knots) or self.value_knots < 0:
            raise ValueError("value_knots must be finite and non-negative")
        for name in ("interp_start_nm", "interp_end_nm"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CanonicalRouteLeg:
    start_index: int
    end_index: int
    portside_xtd_nm: float | None = None
    starboard_xtd_nm: float | None = None
    speed_profile: CanonicalSpeedProfile | None = None

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.end_index != self.start_index + 1:
            raise ValueError("route legs must connect consecutive waypoint indices")
        for name in ("portside_xtd_nm", "starboard_xtd_nm"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CanonicalRoute:
    waypoints: tuple[CanonicalWaypoint, ...]
    legs: tuple[CanonicalRouteLeg, ...]

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError("canonical route requires at least two waypoints")
        if len(self.legs) != len(self.waypoints) - 1:
            raise ValueError("canonical route requires exactly one leg per waypoint pair")
        expected = tuple((i, i + 1) for i in range(len(self.waypoints) - 1))
        actual = tuple((leg.start_index, leg.end_index) for leg in self.legs)
        if actual != expected:
            raise ValueError("canonical route legs must be ordered and contiguous")


@dataclass(frozen=True, slots=True)
class CanonicalInitialState:
    x_east_nm: float
    y_north_nm: float
    sog_knots: float | None = None
    cog_deg: float | None = None
    heading_deg: float | None = None
    nav_status: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.x_east_nm) or not isfinite(self.y_north_nm):
            raise ValueError("initial coordinates must be finite")
        if self.sog_knots is not None and (not isfinite(self.sog_knots) or self.sog_knots < 0):
            raise ValueError("sog_knots must be finite and non-negative")
        for name in ("cog_deg", "heading_deg"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or not 0 <= value <= 360):
                raise ValueError(f"{name} must be within [0, 360]")


@dataclass(frozen=True, slots=True)
class CanonicalVessel:
    vessel_id: str
    role: ScenarioVesselRole
    route: CanonicalRoute
    initial: CanonicalInitialState
    source_numeric_id: int | None = None
    mmsi: int | None = None
    imo: int | None = None
    name: str | None = None
    ship_type: str | None = None
    path_type: str | None = None
    dimensions: CanonicalDimensions | None = None
    init_delay_seconds: float = 0.0
    vessel_class: VesselObjectClass = VesselObjectClass.OTHER

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")
        if not isfinite(self.init_delay_seconds) or self.init_delay_seconds < 0:
            raise ValueError("init_delay_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CanonicalEnvironment:
    wind_speed_mps: float | None = None
    wind_from_deg: float | None = None
    current_speed_mps: float | None = None
    current_towards_deg: float | None = None

    def __post_init__(self) -> None:
        for name in ("wind_speed_mps", "current_speed_mps"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("wind_from_deg", "current_towards_deg"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or not 0 <= value <= 360):
                raise ValueError(f"{name} must be within [0, 360]")


@dataclass(frozen=True, slots=True)
class CanonicalScenario:
    scenario_id: str
    title: str | None
    description: str | None
    start_time_utc: datetime | None
    coordinates: CanonicalCoordinateSystem
    vessels: tuple[CanonicalVessel, ...]
    environment: CanonicalEnvironment | None
    lineage: tuple[ScenarioLineage, ...]
    world_objects: tuple[CanonicalWorldObject, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if self.start_time_utc is not None and self.start_time_utc.tzinfo is None:
            raise ValueError("start_time_utc must be timezone-aware")
        if not self.vessels:
            raise ValueError("canonical scenario requires at least one vessel")
        ids = [vessel.vessel_id for vessel in self.vessels]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate canonical vessel_id")
        if not self.lineage:
            raise ValueError("canonical scenario requires source lineage")
        object_ids = [obj.object_id for obj in self.world_objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("duplicate canonical world object_id")
        overlap = set(ids).intersection(object_ids)
        if overlap:
            raise ValueError(f"vessel/world-object IDs must be distinct: {sorted(overlap)}")

    @property
    def controlled_vessels(self) -> tuple[CanonicalVessel, ...]:
        return tuple(v for v in self.vessels if v.role is ScenarioVesselRole.CONTROLLED)
