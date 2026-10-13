"""C4 GroundTruth/PhysicalWorld versus ObservedWorld separation.

This module introduces a strict observation boundary without changing A162 physics.
Physical truth is first normalized into an immutable testbed snapshot.  Observation
providers then produce a separate immutable snapshot consumed by controllers.  The
perfect provider is the compatibility mode: it reproduces all modeled kinematic/object
values exactly while still crossing the explicit truth -> observation boundary.

C4 deliberately does not implement sensor noise, missed detections, track lifecycle or
environmental belief.  Those are C5/C6 responsibilities built on this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Protocol

from vessel_crossing.kinematics import nm_per_minute_to_knots
from vessel_crossing.nextgen.environmental_world import EnvironmentalWorld2D
from vessel_crossing.nextgen.world import World2D

from .behavior import BehaviorContext, BehaviorDecision, BehaviorModel, BehaviorState, RequestedManeuver
from .scope import TestbedFeature
from .world_objects import (
    CanonicalWorldObject,
    CircleGeometry,
    PointGeometry,
    PolygonGeometry,
    PolylineGeometry,
    WorldGeometry,
    WorldObjectKind,
    WorldObjectRegistry,
)

_EPS = 1e-12


def _optional_heading(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 360.0:
        raise ValueError(f"{name} must be within [0, 360]")
    return value % 360.0


class ObservationTrackStatus(str, Enum):
    MEASURED = "MEASURED"
    COASTED = "COASTED"
    GHOST = "GHOST"


class ObservationSource(str, Enum):
    PERFECT = "PERFECT"
    SENSOR = "SENSOR"
    AIS = "AIS"
    FUSED = "FUSED"
    PREDICTED = "PREDICTED"
    GHOST = "GHOST"


@dataclass(frozen=True, slots=True)
class PhysicalVesselState:
    """Controller-independent vessel truth at one instant.

    Position is local east/north nautical miles.  ``heading_deg`` is physical vessel
    orientation.  COG/SOG are explicit ground kinematics and therefore need not equal
    heading/through-water speed in an environmental world.
    """

    vessel_id: str
    x_east_nm: float
    y_north_nm: float
    heading_deg: float
    course_over_ground_deg: float | None
    speed_over_ground_knots: float

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")
        for name in ("x_east_nm", "y_north_nm", "heading_deg", "speed_over_ground_knots"):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.heading_deg <= 360.0:
            raise ValueError("heading_deg must be within [0, 360]")
        if self.speed_over_ground_knots < 0.0:
            raise ValueError("speed_over_ground_knots must be non-negative")
        object.__setattr__(self, "heading_deg", self.heading_deg % 360.0)
        object.__setattr__(
            self,
            "course_over_ground_deg",
            _optional_heading(self.course_over_ground_deg, name="course_over_ground_deg"),
        )


@dataclass(frozen=True, slots=True)
class PhysicalObjectState:
    object_id: str
    kind: WorldObjectKind
    geometry: WorldGeometry
    source_object: CanonicalWorldObject

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty")
        if self.source_object.object_id != self.object_id:
            raise ValueError("source_object identity mismatch")
        if self.source_object.kind is not self.kind:
            raise ValueError("source_object kind mismatch")


@dataclass(frozen=True, slots=True)
class PhysicalWorldSnapshot:
    time_min: float
    vessels: tuple[PhysicalVesselState, ...]
    objects: tuple[PhysicalObjectState, ...] = ()
    source_kind: str = "TESTBED_PHYSICAL"

    def __post_init__(self) -> None:
        if not isfinite(self.time_min) or self.time_min < 0.0:
            raise ValueError("time_min must be finite and non-negative")
        if not self.source_kind:
            raise ValueError("source_kind must be non-empty")
        vessel_ids = [v.vessel_id for v in self.vessels]
        object_ids = [o.object_id for o in self.objects]
        if vessel_ids != sorted(vessel_ids) or len(vessel_ids) != len(set(vessel_ids)):
            raise ValueError("physical vessels must be unique and sorted by vessel_id")
        if object_ids != sorted(object_ids) or len(object_ids) != len(set(object_ids)):
            raise ValueError("physical objects must be unique and sorted by object_id")

    def by_vessel_id(self, vessel_id: str) -> PhysicalVesselState:
        for vessel in self.vessels:
            if vessel.vessel_id == vessel_id:
                return vessel
        raise KeyError(vessel_id)

    def by_object_id(self, object_id: str) -> PhysicalObjectState:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        raise KeyError(object_id)

    def identity_sha256(self) -> str:
        payload = {
            "time_min": self.time_min,
            "source_kind": self.source_kind,
            "vessels": [
                {
                    "vessel_id": v.vessel_id,
                    "x_east_nm": v.x_east_nm,
                    "y_north_nm": v.y_north_nm,
                    "heading_deg": v.heading_deg,
                    "course_over_ground_deg": v.course_over_ground_deg,
                    "speed_over_ground_knots": v.speed_over_ground_knots,
                }
                for v in self.vessels
            ],
            "objects": [
                {
                    "object_id": o.object_id,
                    "kind": o.kind.value,
                    "geometry": _geometry_payload(o.geometry),
                    "source": _world_object_payload(o.source_object),
                }
                for o in self.objects
            ],
        }
        return _sha256_payload(payload)


@dataclass(frozen=True, slots=True)
class ObservedVesselState:
    vessel_id: str
    x_east_nm: float
    y_north_nm: float
    heading_deg: float | None
    course_over_ground_deg: float | None
    speed_over_ground_knots: float | None
    source_time_min: float
    track_status: ObservationTrackStatus = ObservationTrackStatus.MEASURED
    observation_source: ObservationSource = ObservationSource.PERFECT

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")
        if not isfinite(self.x_east_nm) or not isfinite(self.y_north_nm):
            raise ValueError("observed position must be finite")
        if not isfinite(self.source_time_min) or self.source_time_min < 0.0:
            raise ValueError("source_time_min must be finite and non-negative")
        object.__setattr__(self, "heading_deg", _optional_heading(self.heading_deg, name="heading_deg"))
        object.__setattr__(
            self,
            "course_over_ground_deg",
            _optional_heading(self.course_over_ground_deg, name="course_over_ground_deg"),
        )
        if self.speed_over_ground_knots is not None:
            if not isfinite(self.speed_over_ground_knots) or self.speed_over_ground_knots < 0.0:
                raise ValueError("speed_over_ground_knots must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ObservedObjectState:
    object_id: str
    kind: WorldObjectKind
    geometry: WorldGeometry
    source_time_min: float
    track_status: ObservationTrackStatus = ObservationTrackStatus.MEASURED
    observation_source: ObservationSource = ObservationSource.PERFECT

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty")
        if not isfinite(self.source_time_min) or self.source_time_min < 0.0:
            raise ValueError("source_time_min must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ObservedWorldSnapshot:
    observation_time_min: float
    vessels: tuple[ObservedVesselState, ...]
    objects: tuple[ObservedObjectState, ...]
    provider_id: str
    source_truth_sha256: str

    def __post_init__(self) -> None:
        if not isfinite(self.observation_time_min) or self.observation_time_min < 0.0:
            raise ValueError("observation_time_min must be finite and non-negative")
        if not self.provider_id:
            raise ValueError("provider_id must be non-empty")
        if len(self.source_truth_sha256) != 64:
            raise ValueError("source_truth_sha256 must be a SHA-256 hex digest")
        try:
            int(self.source_truth_sha256, 16)
        except ValueError as exc:
            raise ValueError("source_truth_sha256 must be a SHA-256 hex digest") from exc
        vessel_ids = [v.vessel_id for v in self.vessels]
        object_ids = [o.object_id for o in self.objects]
        if vessel_ids != sorted(vessel_ids) or len(vessel_ids) != len(set(vessel_ids)):
            raise ValueError("observed vessels must be unique and sorted by vessel_id")
        if object_ids != sorted(object_ids) or len(object_ids) != len(set(object_ids)):
            raise ValueError("observed objects must be unique and sorted by object_id")
        for vessel in self.vessels:
            if vessel.source_time_min > self.observation_time_min + _EPS:
                raise ValueError("observed vessel source time cannot be in the future")
        for obj in self.objects:
            if obj.source_time_min > self.observation_time_min + _EPS:
                raise ValueError("observed object source time cannot be in the future")

    def by_vessel_id(self, vessel_id: str) -> ObservedVesselState:
        for vessel in self.vessels:
            if vessel.vessel_id == vessel_id:
                return vessel
        raise KeyError(vessel_id)

    def by_object_id(self, object_id: str) -> ObservedObjectState:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        raise KeyError(object_id)

    def age_of_vessel_min(self, vessel_id: str) -> float:
        return self.observation_time_min - self.by_vessel_id(vessel_id).source_time_min

    def age_of_object_min(self, object_id: str) -> float:
        return self.observation_time_min - self.by_object_id(object_id).source_time_min


class ObservationProvider(Protocol):
    component_id: TestbedFeature
    provider_id: str

    def observe(self, truth: PhysicalWorldSnapshot) -> ObservedWorldSnapshot:
        ...


@dataclass(frozen=True, slots=True)
class PerfectObservationProvider:
    """C4 compatibility mode: exact modeled truth values, but in a separate snapshot."""

    provider_id: str = "cross-c4/perfect-observation/v1"
    component_id = TestbedFeature.C4_OBSERVED_WORLD_BOUNDARY

    def __post_init__(self) -> None:
        if not self.provider_id:
            raise ValueError("provider_id must be non-empty")

    def observe(self, truth: PhysicalWorldSnapshot) -> ObservedWorldSnapshot:
        return ObservedWorldSnapshot(
            observation_time_min=truth.time_min,
            vessels=tuple(
                ObservedVesselState(
                    vessel_id=v.vessel_id,
                    x_east_nm=v.x_east_nm,
                    y_north_nm=v.y_north_nm,
                    heading_deg=v.heading_deg,
                    course_over_ground_deg=v.course_over_ground_deg,
                    speed_over_ground_knots=v.speed_over_ground_knots,
                    source_time_min=truth.time_min,
                )
                for v in truth.vessels
            ),
            objects=tuple(
                ObservedObjectState(
                    object_id=o.object_id,
                    kind=o.kind,
                    geometry=o.geometry,
                    source_time_min=truth.time_min,
                )
                for o in truth.objects
            ),
            provider_id=self.provider_id,
            source_truth_sha256=truth.identity_sha256(),
        )


def physical_snapshot_from_world2d(
    world: World2D,
    *,
    world_objects: WorldObjectRegistry | None = None,
) -> PhysicalWorldSnapshot:
    """Normalize an inherited A162 ``World2D`` snapshot into C4 physical truth."""

    if not isinstance(world, World2D):
        raise TypeError("world must be World2D")
    snapshot = world.snapshot()
    vessels = tuple(
        PhysicalVesselState(
            vessel_id=v.vessel_id,
            x_east_nm=v.position[0],
            y_north_nm=v.position[1],
            heading_deg=v.heading_deg,
            course_over_ground_deg=(v.heading_deg if v.speed > _EPS else None),
            speed_over_ground_knots=nm_per_minute_to_knots(v.speed),
        )
        for v in snapshot.vessels
    )
    return PhysicalWorldSnapshot(
        time_min=snapshot.time,
        vessels=vessels,
        objects=_physical_objects_at(world_objects, snapshot.time),
        source_kind="A162_WORLD2D",
    )


def physical_snapshot_from_environmental_world2d(
    world: EnvironmentalWorld2D,
    *,
    world_objects: WorldObjectRegistry | None = None,
) -> PhysicalWorldSnapshot:
    """Normalize environmental truth using physical ground kinematics."""

    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    snapshot = world.snapshot()
    physical: list[PhysicalVesselState] = []
    for vessel in snapshot.vessels:
        frames = vessel.velocity_frames(world.drift_for(vessel.vessel_id))
        physical.append(
            PhysicalVesselState(
                vessel_id=vessel.vessel_id,
                x_east_nm=vessel.position[0],
                y_north_nm=vessel.position[1],
                heading_deg=vessel.through_water_heading_deg,
                course_over_ground_deg=frames.ground_course_deg,
                speed_over_ground_knots=frames.ground_speed_knots,
            )
        )
    return PhysicalWorldSnapshot(
        time_min=snapshot.time,
        vessels=tuple(physical),
        objects=_physical_objects_at(world_objects, snapshot.time),
        source_kind="A162_ENVIRONMENTAL_WORLD2D",
    )


def behavior_context_from_observed(
    observed: ObservedWorldSnapshot,
    *,
    vessel_id: str,
    route_heading_deg: float | None = None,
    route_speed_knots: float | None = None,
    requested_maneuver: RequestedManeuver | None = None,
) -> BehaviorContext:
    """Build the C3 behavior input strictly from an observed vessel state.

    This is the first controller-facing bridge that makes truth bypass explicit.  If a
    later C5 provider drops or stales a vessel, this function sees only that observed
    result; it never consults a physical world.
    """

    vessel = observed.by_vessel_id(vessel_id)
    if vessel.heading_deg is None:
        raise ValueError("behavior input requires observed heading")
    if vessel.speed_over_ground_knots is None:
        raise ValueError("behavior input requires observed speed over ground")
    return BehaviorContext(
        state=BehaviorState(
            vessel_id=vessel.vessel_id,
            time_min=observed.observation_time_min,
            x_east_nm=vessel.x_east_nm,
            y_north_nm=vessel.y_north_nm,
            heading_deg=vessel.heading_deg,
            speed_knots=vessel.speed_over_ground_knots,
            route_heading_deg=route_heading_deg,
            route_speed_knots=route_speed_knots,
        ),
        requested_maneuver=requested_maneuver,
    )


@dataclass(frozen=True, slots=True)
class BehaviorModelObservedController:
    """Adapter proving the controller decision path consumes ``ObservedWorld`` only."""

    model: BehaviorModel
    component_id = TestbedFeature.C4_OBSERVED_WORLD_BOUNDARY

    def decide(
        self,
        observed: ObservedWorldSnapshot,
        *,
        vessel_id: str,
        route_heading_deg: float | None = None,
        route_speed_knots: float | None = None,
        requested_maneuver: RequestedManeuver | None = None,
    ) -> BehaviorDecision:
        return self.model.decide(
            behavior_context_from_observed(
                observed,
                vessel_id=vessel_id,
                route_heading_deg=route_heading_deg,
                route_speed_knots=route_speed_knots,
                requested_maneuver=requested_maneuver,
            )
        )


def _physical_objects_at(
    world_objects: WorldObjectRegistry | None,
    time_min: float,
) -> tuple[PhysicalObjectState, ...]:
    if world_objects is None:
        return ()
    if not isinstance(world_objects, WorldObjectRegistry):
        raise TypeError("world_objects must be WorldObjectRegistry")
    return tuple(
        PhysicalObjectState(
            object_id=obj.object_id,
            kind=obj.kind,
            geometry=obj.geometry_at(time_min),
            source_object=obj,
        )
        for obj in sorted(world_objects.active_objects(time_min), key=lambda item: item.object_id)
    )



def _world_object_payload(obj: CanonicalWorldObject) -> dict[str, object]:
    return {
        "object_id": obj.object_id,
        "kind": obj.kind.value,
        "rules": [
            {
                "consequence": rule.consequence.value,
                "clearance_nm": rule.clearance_nm,
                "max_safe_draught_m": rule.max_safe_draught_m,
            }
            for rule in obj.rules
        ],
        "lifecycle": {
            "active_from_min": obj.lifecycle.active_from_min,
            "active_until_min": obj.lifecycle.active_until_min,
            "appearance_mode": obj.lifecycle.appearance_mode.value,
        },
        "motion": None if obj.motion is None else {
            "course_deg": obj.motion.course_deg,
            "speed_knots": obj.motion.speed_knots,
        },
        "uncertainty_radius_nm": obj.uncertainty_radius_nm,
        "label": obj.label,
        "attributes": [list(item) for item in obj.attributes],
    }

def _geometry_payload(geometry: WorldGeometry) -> dict[str, object]:
    if isinstance(geometry, PointGeometry):
        return {"type": "POINT", "point": [geometry.point.x_east_nm, geometry.point.y_north_nm]}
    if isinstance(geometry, CircleGeometry):
        return {
            "type": "CIRCLE",
            "center": [geometry.center.x_east_nm, geometry.center.y_north_nm],
            "radius_nm": geometry.radius_nm,
        }
    if isinstance(geometry, PolylineGeometry):
        return {
            "type": "POLYLINE",
            "points": [[p.x_east_nm, p.y_north_nm] for p in geometry.points],
        }
    if isinstance(geometry, PolygonGeometry):
        return {
            "type": "POLYGON",
            "vertices": [[p.x_east_nm, p.y_north_nm] for p in geometry.vertices],
        }
    raise TypeError(f"unsupported geometry {type(geometry)!r}")


def _sha256_payload(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
