"""Edge adapter for DNV maritime-schema traffic situations (0.2.x).

DNV field names and geographic coordinates are intentionally contained here. The
canonical scenario and A162 physical kernel remain schema-agnostic.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from math import atan2, degrees
from typing import Any, Mapping

from pyproj import CRS, Transformer

from ..scenario import (
    CanonicalCoordinateSystem,
    CanonicalDimensions,
    CanonicalEnvironment,
    CanonicalInitialState,
    CanonicalRoute,
    CanonicalRouteLeg,
    CanonicalScenario,
    CanonicalSpeedProfile,
    CanonicalVessel,
    CanonicalWaypoint,
    ScenarioLineage,
    ScenarioVesselRole,
)
from ..scope import TestbedFeature
from ..world_objects import VesselObjectClass


DNV_ADAPTER_ID = "dnv-maritime-schema-traffic-situation-0.2.x/v1"
DNV_SCHEMA_LINE = "0.2"
METRES_PER_NAUTICAL_MILE = 1852.0


class DnvMaritimeSchemaError(ValueError):
    pass


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DnvMaritimeSchemaError(f"{label} must be an object")
    return value


def _schema_version(payload: Mapping[str, Any]) -> str:
    value = payload.get("schemaVersion", payload.get("version"))
    if not isinstance(value, str) or not value.startswith(DNV_SCHEMA_LINE + "."):
        raise DnvMaritimeSchemaError(f"supported DNV maritime-schema line is {DNV_SCHEMA_LINE}.x, got {value!r}")
    return value


def _parse_start_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DnvMaritimeSchemaError("startTime must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DnvMaritimeSchemaError("invalid startTime") from exc
    if parsed.tzinfo is None:
        raise DnvMaritimeSchemaError("startTime must include a timezone")
    return parsed.astimezone(timezone.utc)


class _LocalProjector:
    def __init__(self, *, origin_lat_deg: float, origin_lon_deg: float) -> None:
        local = CRS.from_proj4(
            f"+proj=aeqd +lat_0={origin_lat_deg:.12f} +lon_0={origin_lon_deg:.12f} "
            "+datum=WGS84 +units=m +no_defs"
        )
        self._forward = Transformer.from_crs("EPSG:4326", local, always_xy=True)
        self._inverse = Transformer.from_crs(local, "EPSG:4326", always_xy=True)

    def to_local_nm(self, lon_deg: float, lat_deg: float) -> tuple[float, float]:
        x_m, y_m = self._forward.transform(lon_deg, lat_deg)
        return x_m / METRES_PER_NAUTICAL_MILE, y_m / METRES_PER_NAUTICAL_MILE

    def to_lonlat(self, x_nm: float, y_nm: float) -> tuple[float, float]:
        lon, lat = self._inverse.transform(x_nm * METRES_PER_NAUTICAL_MILE, y_nm * METRES_PER_NAUTICAL_MILE)
        return float(lon), float(lat)


def _position_lonlat(value: Any, label: str) -> tuple[float, float]:
    pos = _require_mapping(value, label)
    try:
        lon = float(pos["lon"])
        lat = float(pos["lat"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DnvMaritimeSchemaError(f"{label} requires numeric lon/lat") from exc
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise DnvMaritimeSchemaError(f"{label} lon/lat out of range")
    return lon, lat


def _extract_speed_profile(leg: Mapping[str, Any]) -> CanonicalSpeedProfile | None:
    data = leg.get("data")
    if isinstance(data, Mapping) and isinstance(data.get("sog"), Mapping):
        sog = data["sog"]
        value = sog.get("value")
        if value is None:
            return None
        return CanonicalSpeedProfile(
            value_knots=float(value),
            interp_start_nm=None if sog.get("interpStart") is None else float(sog["interpStart"]),
            interp_end_nm=None if sog.get("interpEnd") is None else float(sog["interpEnd"]),
            interp_method=None if sog.get("interpMethod") is None else str(sog["interpMethod"]),
        )
    if leg.get("sog") is not None:
        return CanonicalSpeedProfile(value_knots=float(leg["sog"]))
    return None


def _route_from_dnv(waypoints_raw: Any, projector: _LocalProjector) -> CanonicalRoute:
    if not isinstance(waypoints_raw, list) or len(waypoints_raw) < 2:
        raise DnvMaritimeSchemaError("waypoints must contain at least two entries")
    waypoints: list[CanonicalWaypoint] = []
    legs: list[CanonicalRouteLeg] = []
    for index, raw in enumerate(waypoints_raw):
        item = _require_mapping(raw, f"waypoints[{index}]")
        lon, lat = _position_lonlat(item.get("position"), f"waypoints[{index}].position")
        x_nm, y_nm = projector.to_local_nm(lon, lat)
        turn_radius = item.get("turnRadius")
        waypoints.append(
            CanonicalWaypoint(
                x_east_nm=float(x_nm),
                y_north_nm=float(y_nm),
                turn_radius_nm=None if turn_radius is None else float(turn_radius),
            )
        )
        if index > 0:
            leg_raw = item.get("leg", {})
            leg = _require_mapping(leg_raw, f"waypoints[{index}].leg")
            legs.append(
                CanonicalRouteLeg(
                    start_index=index - 1,
                    end_index=index,
                    portside_xtd_nm=None if leg.get("portsideXtd") is None else float(leg["portsideXtd"]),
                    starboard_xtd_nm=None if leg.get("starboardXtd") is None else float(leg["starboardXtd"]),
                    speed_profile=_extract_speed_profile(leg),
                )
            )
    return CanonicalRoute(tuple(waypoints), tuple(legs))


def _first_leg_cog(route: CanonicalRoute) -> float:
    first, second = route.waypoints[:2]
    dx = second.x_east_nm - first.x_east_nm
    dy = second.y_north_nm - first.y_north_nm
    return degrees(atan2(dx, dy)) % 360.0


def _dimensions(raw: Any) -> CanonicalDimensions | None:
    if raw is None:
        return None
    item = _require_mapping(raw, "dimensions")
    if not item:
        return None
    def get(name: str) -> float | None:
        value = item.get(name)
        return None if value is None else float(value)
    return CanonicalDimensions(
        length_m=get("length"), width_m=get("width"), height_m=get("height"), draught_m=get("draught"),
        a_m=get("a"), b_m=get("b"), c_m=get("c"), d_m=get("d"),
    )




def _vessel_class_from_dnv(ship_type: str | None) -> VesselObjectClass:
    if ship_type is None:
        return VesselObjectClass.OTHER
    normalized = ship_type.strip().lower()
    if normalized == "fishing":
        return VesselObjectClass.FISHING_CRAFT
    if normalized in {"pleasure craft", "sailing", "pilot vessel", "tug", "port tender", "search and rescue vessel"}:
        return VesselObjectClass.SMALL_CRAFT
    if normalized.startswith(("cargo", "tanker", "passenger")):
        return VesselObjectClass.POWERED_MERCHANT
    return VesselObjectClass.OTHER

def _vessel_from_dnv(raw: Any, *, role: ScenarioVesselRole, index: int, projector: _LocalProjector) -> CanonicalVessel:
    item = _require_mapping(raw, "ship")
    route = _route_from_dnv(item.get("waypoints"), projector)
    static = _require_mapping(item.get("static", {}), "ship.static")
    initial = _require_mapping(item.get("initial", {}), "ship.initial")

    source_id = static.get("id")
    vessel_id = str(source_id) if source_id is not None else f"{role.value.lower()}-{index}"
    first_wp = route.waypoints[0]
    if initial.get("position") is None:
        initial_x, initial_y = first_wp.x_east_nm, first_wp.y_north_nm
    else:
        lon, lat = _position_lonlat(initial["position"], "ship.initial.position")
        initial_x, initial_y = projector.to_local_nm(lon, lat)
    first_speed = route.legs[0].speed_profile.value_knots if route.legs[0].speed_profile else None
    sog = first_speed if initial.get("sog") is None else float(initial["sog"])
    cog = _first_leg_cog(route) if initial.get("cog") is None else float(initial["cog"])

    numeric_id: int | None = None
    if isinstance(source_id, int) and not isinstance(source_id, bool):
        numeric_id = source_id

    ship_type = None if static.get("shipType") is None else str(static["shipType"])

    return CanonicalVessel(
        vessel_id=vessel_id,
        role=role,
        route=route,
        initial=CanonicalInitialState(
            x_east_nm=float(initial_x), y_north_nm=float(initial_y), sog_knots=sog, cog_deg=cog,
            heading_deg=None if initial.get("heading") is None else float(initial["heading"]),
            nav_status=None if initial.get("navStatus") is None else str(initial["navStatus"]),
        ),
        source_numeric_id=numeric_id,
        mmsi=None if static.get("mmsi") is None else int(static["mmsi"]),
        imo=None if static.get("imo") is None else int(static["imo"]),
        name=None if static.get("name") is None else str(static["name"]),
        ship_type=ship_type,
        vessel_class=_vessel_class_from_dnv(ship_type),
        path_type=None if static.get("pathType") is None else str(static["pathType"]),
        dimensions=_dimensions(static.get("dimensions")),
        init_delay_seconds=float(static.get("initDelay", 0.0)),
    )


def _environment_from_dnv(raw: Any) -> CanonicalEnvironment | None:
    if raw is None:
        return None
    item = _require_mapping(raw, "environment")
    supported = {
        "wind_speed_mps": item.get("windSpeed"),
        "wind_from_deg": item.get("windDirection"),
        "current_speed_mps": item.get("currentSpeed"),
        "current_towards_deg": item.get("currentDirection"),
    }
    if all(value is None for value in supported.values()):
        return None
    return CanonicalEnvironment(**{k: None if v is None else float(v) for k, v in supported.items()})


def import_dnv_traffic_situation(
    payload: Mapping[str, Any],
    *,
    source_uri: str | None = None,
    license_id: str | None = "MIT",
    source_kind: str = "DNV_MARITIME_SCHEMA",
) -> CanonicalScenario:
    version = _schema_version(payload)
    own = _require_mapping(payload.get("ownShip"), "ownShip")
    own_waypoints = own.get("waypoints")
    if not isinstance(own_waypoints, list) or not own_waypoints:
        raise DnvMaritimeSchemaError("ownShip.waypoints is required")
    origin_lon, origin_lat = _position_lonlat(_require_mapping(own_waypoints[0], "ownShip.waypoints[0]").get("position"), "ownShip.waypoints[0].position")
    projector = _LocalProjector(origin_lat_deg=origin_lat, origin_lon_deg=origin_lon)

    targets = payload.get("targetShips", [])
    if not isinstance(targets, list):
        raise DnvMaritimeSchemaError("targetShips must be an array")
    vessels = [_vessel_from_dnv(own, role=ScenarioVesselRole.CONTROLLED, index=1, projector=projector)]
    vessels.extend(
        _vessel_from_dnv(target, role=ScenarioVesselRole.TRAFFIC, index=index, projector=projector)
        for index, target in enumerate(targets, start=1)
    )
    raw_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    attributes: list[tuple[str, str]] = []
    if payload.get("trafficgenVersion") is not None:
        attributes.append(("trafficgenVersion", str(payload["trafficgenVersion"])))
    lineage = ScenarioLineage(
        source_kind=source_kind,
        source_version=version,
        source_uri=source_uri,
        source_sha256=raw_hash,
        license_id=license_id,
        adapter_id=DNV_ADAPTER_ID,
        attributes=tuple(sorted(attributes)),
    )
    scenario_id = str(payload.get("id") or raw_hash[:24])
    return CanonicalScenario(
        scenario_id=scenario_id,
        title=None if payload.get("title") is None else str(payload["title"]),
        description=None if payload.get("description") is None else str(payload["description"]),
        start_time_utc=_parse_start_time(payload.get("startTime")),
        coordinates=CanonicalCoordinateSystem(
            frame_id=f"dnv-aeqd:{origin_lat:.9f},{origin_lon:.9f}",
            origin_lat_deg=origin_lat,
            origin_lon_deg=origin_lon,
            projection="WGS84_AZIMUTHAL_EQUIDISTANT",
        ),
        vessels=tuple(vessels),
        environment=_environment_from_dnv(payload.get("environment")),
        lineage=(lineage,),
    )


def import_dnv_ship_traffic_generator_situation(
    payload: Mapping[str, Any], *, source_uri: str | None = None, license_id: str | None = "MIT"
) -> CanonicalScenario:
    if payload.get("trafficgenVersion") is None:
        raise DnvMaritimeSchemaError("ship-traffic-generator payload requires trafficgenVersion")
    return import_dnv_traffic_situation(
        payload,
        source_uri=source_uri,
        license_id=license_id,
        source_kind="DNV_SHIP_TRAFFIC_GENERATOR",
    )


def _position_dict(projector: _LocalProjector, waypoint: CanonicalWaypoint | CanonicalInitialState) -> dict[str, float]:
    lon, lat = projector.to_lonlat(waypoint.x_east_nm, waypoint.y_north_nm)
    return {"lon": lon, "lat": lat}


def _leg_to_dnv(leg: CanonicalRouteLeg) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if leg.starboard_xtd_nm is not None:
        payload["starboardXtd"] = leg.starboard_xtd_nm
    if leg.portside_xtd_nm is not None:
        payload["portsideXtd"] = leg.portside_xtd_nm
    if leg.speed_profile is not None:
        speed = leg.speed_profile
        data: dict[str, Any] = {"value": speed.value_knots}
        if speed.interp_start_nm is not None:
            data["interpStart"] = speed.interp_start_nm
        if speed.interp_end_nm is not None:
            data["interpEnd"] = speed.interp_end_nm
        if speed.interp_method is not None:
            data["interpMethod"] = speed.interp_method
        payload["data"] = {"sog": data}
    return payload


def _vessel_to_dnv(vessel: CanonicalVessel, projector: _LocalProjector) -> dict[str, Any]:
    waypoints: list[dict[str, Any]] = []
    for index, waypoint in enumerate(vessel.route.waypoints):
        item: dict[str, Any] = {"position": _position_dict(projector, waypoint)}
        if waypoint.turn_radius_nm is not None:
            item["turnRadius"] = waypoint.turn_radius_nm
        if index > 0:
            leg = _leg_to_dnv(vessel.route.legs[index - 1])
            if leg:
                item["leg"] = leg
        waypoints.append(item)

    static: dict[str, Any] = {}
    if vessel.source_numeric_id is not None:
        static["id"] = vessel.source_numeric_id
    if vessel.mmsi is not None:
        static["mmsi"] = vessel.mmsi
    if vessel.imo is not None:
        static["imo"] = vessel.imo
    if vessel.name is not None:
        static["name"] = vessel.name
    if vessel.ship_type is not None:
        static["shipType"] = vessel.ship_type
    if vessel.path_type is not None:
        static["pathType"] = vessel.path_type
    if vessel.init_delay_seconds:
        static["initDelay"] = vessel.init_delay_seconds
    if vessel.dimensions is not None:
        dims = vessel.dimensions
        mapped = {
            "length": dims.length_m, "width": dims.width_m, "height": dims.height_m, "draught": dims.draught_m,
            "a": dims.a_m, "b": dims.b_m, "c": dims.c_m, "d": dims.d_m,
        }
        static["dimensions"] = {key: value for key, value in mapped.items() if value is not None}

    initial: dict[str, Any] = {"position": _position_dict(projector, vessel.initial)}
    if vessel.initial.sog_knots is not None:
        initial["sog"] = vessel.initial.sog_knots
    if vessel.initial.cog_deg is not None:
        initial["cog"] = vessel.initial.cog_deg
    if vessel.initial.heading_deg is not None:
        initial["heading"] = vessel.initial.heading_deg
    if vessel.initial.nav_status is not None:
        initial["navStatus"] = vessel.initial.nav_status
    return {"waypoints": waypoints, "static": static, "initial": initial}


def export_dnv_traffic_situation(scenario: CanonicalScenario, *, schema_version: str | None = None) -> dict[str, Any]:
    if scenario.world_objects:
        raise DnvMaritimeSchemaError("DNV traffic situation export cannot represent C2 world_objects; use a dedicated adapter or strip them explicitly")
    controlled = scenario.controlled_vessels
    if len(controlled) != 1:
        raise DnvMaritimeSchemaError("DNV traffic situation export requires exactly one CONTROLLED vessel")
    version = schema_version
    if version is None:
        versions = [lineage.source_version for lineage in scenario.lineage if lineage.adapter_id == DNV_ADAPTER_ID]
        version = next((item for item in versions if item is not None), "0.2.1")
    if not version.startswith(DNV_SCHEMA_LINE + "."):
        raise DnvMaritimeSchemaError(f"supported DNV maritime-schema line is {DNV_SCHEMA_LINE}.x")
    if scenario.coordinates.origin_lat_deg is None or scenario.coordinates.origin_lon_deg is None:
        raise DnvMaritimeSchemaError("DNV export requires a WGS84 geographic origin")
    projector = _LocalProjector(
        origin_lat_deg=scenario.coordinates.origin_lat_deg,
        origin_lon_deg=scenario.coordinates.origin_lon_deg,
    )
    payload: dict[str, Any] = {"schemaVersion": version}
    if scenario.title is not None:
        payload["title"] = scenario.title
    if scenario.description is not None:
        payload["description"] = scenario.description
    if scenario.start_time_utc is not None:
        payload["startTime"] = scenario.start_time_utc.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    payload["ownShip"] = _vessel_to_dnv(controlled[0], projector)
    payload["targetShips"] = [
        _vessel_to_dnv(vessel, projector)
        for vessel in scenario.vessels
        if vessel.role is ScenarioVesselRole.TRAFFIC
    ]
    if scenario.environment is not None:
        env = scenario.environment
        mapped = {
            "windSpeed": env.wind_speed_mps,
            "windDirection": env.wind_from_deg,
            "currentSpeed": env.current_speed_mps,
            "currentDirection": env.current_towards_deg,
        }
        payload["environment"] = {key: value for key, value in mapped.items() if value is not None}
    return payload


def export_dnv_ship_traffic_generator_situation(
    scenario: CanonicalScenario, *, trafficgen_version: str, schema_version: str | None = None
) -> dict[str, Any]:
    if not trafficgen_version:
        raise DnvMaritimeSchemaError("trafficgen_version must be non-empty")
    payload = export_dnv_traffic_situation(scenario, schema_version=schema_version)
    payload["trafficgenVersion"] = trafficgen_version
    return payload


class DnvMaritimeSchemaAdapter:
    component_id = TestbedFeature.C1_SCENARIO_INTERCHANGE
    adapter_id = DNV_ADAPTER_ID

    def import_situation(self, payload: Mapping[str, Any], *, source_uri: str | None = None) -> CanonicalScenario:
        return import_dnv_traffic_situation(payload, source_uri=source_uri)

    def export_situation(self, scenario: CanonicalScenario, *, schema_version: str | None = None) -> dict[str, Any]:
        return export_dnv_traffic_situation(scenario, schema_version=schema_version)
