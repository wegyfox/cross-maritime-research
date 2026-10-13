"""Strict D0 translation into the existing NextGen environmental-network FIFO config.

The translator is intentionally narrower than the full NextGen object graph.  Complex
scientific/runtime semantics are selected through an explicit named semantics profile;
scenario YAML owns geometry, traffic, vessel/maneuver/safety metadata and time/seed
inputs.  Unknown or omitted supported-surface fields fail closed instead of acquiring
new D0 defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

from ..kinematics import knots_to_nm_per_minute
from ..nextgen.dynamics import ManeuverLimits
from ..nextgen.lifecycle import EncounterLifecycleConfig
from ..nextgen.encounter_registry import EncounterDiscoveryConfig
from ..nextgen.environmental_fleet_multi_owner_recapture import (
    EnvironmentalFleetMultiOwnerRecaptureConfig,
)
from ..nextgen.environmental_fleet_runtime import EnvironmentalFleetRuntimeConfig
from ..nextgen.environmental_network_fifo_experiment import (
    EnvironmentalNetworkFIFOAdmissionPolicy,
    EnvironmentalNetworkFIFOExperimentConfig,
)
from ..nextgen.environmental_network_scenario import (
    EnvironmentalNetworkIngressTimingPolicy,
    EnvironmentalNetworkScenarioConfig,
)
from ..nextgen.environmental_network_source_recheck import EnvironmentalNetworkSourceRecheckConfig
from ..nextgen.network_traffic import PoissonNetworkStreamSpec
from ..nextgen.route import Route2D
from ..nextgen.route_exit import RouteEndAction
from ..nextgen.route_network import (
    RouteNetwork,
    RouteNetworkEdge,
    RouteNetworkNode,
    RouteRequest,
    RouteSelectionPolicy,
)
from ..nextgen.ship_domain import ShipDomain
from ..nextgen.vessel_classes import (
    VesselClassProfile,
    VesselDimensions,
    VesselSpeedEnvelope,
)
from .contract import DiagnosticScenarioSpec
from .nextgen_fifo_bindings import NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_ADAPTER_ID


class NextgenFIFOTranslationError(ValueError):
    """Raised when D0 parameters cannot be translated without inventing semantics."""


@dataclass(frozen=True, slots=True)
class NextgenFIFOSemanticsProfile:
    """Explicit existing-runtime semantics selected by ID from scenario YAML."""

    profile_id: str
    discovery: EncounterDiscoveryConfig
    lifecycle: EncounterLifecycleConfig
    recapture: EnvironmentalFleetMultiOwnerRecaptureConfig
    runtime: EnvironmentalFleetRuntimeConfig
    recheck: EnvironmentalNetworkSourceRecheckConfig
    admission_policy: EnvironmentalNetworkFIFOAdmissionPolicy
    ingress_timing_policy: EnvironmentalNetworkIngressTimingPolicy

    def __post_init__(self) -> None:
        value = self.profile_id.strip() if isinstance(self.profile_id, str) else ""
        if not value:
            raise NextgenFIFOTranslationError("semantics profile_id must be non-empty")
        expected = (
            (self.discovery, EncounterDiscoveryConfig, "discovery"),
            (self.lifecycle, EncounterLifecycleConfig, "lifecycle"),
            (self.recapture, EnvironmentalFleetMultiOwnerRecaptureConfig, "recapture"),
            (self.runtime, EnvironmentalFleetRuntimeConfig, "runtime"),
            (self.recheck, EnvironmentalNetworkSourceRecheckConfig, "recheck"),
            (self.admission_policy, EnvironmentalNetworkFIFOAdmissionPolicy, "admission_policy"),
            (self.ingress_timing_policy, EnvironmentalNetworkIngressTimingPolicy, "ingress_timing_policy"),
        )
        for item, kind, name in expected:
            if not isinstance(item, kind):
                raise TypeError(f"{name} must be {kind.__name__}")
        object.__setattr__(self, "profile_id", value)


class NextgenFIFOSemanticsProfileRegistry:
    def __init__(self, profiles: tuple[NextgenFIFOSemanticsProfile, ...] = ()) -> None:
        ids = [item.profile_id for item in profiles]
        if len(ids) != len(set(ids)):
            raise NextgenFIFOTranslationError("duplicate NextGen FIFO semantics profile_id")
        self._profiles = MappingProxyType({item.profile_id: item for item in profiles})

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def resolve(self, profile_id: str) -> NextgenFIFOSemanticsProfile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise NextgenFIFOTranslationError(
                f"unknown NextGen FIFO semantics_profile {profile_id!r}"
            ) from exc


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NextgenFIFOTranslationError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], *, name: str, keys: set[str]) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        raise NextgenFIFOTranslationError(
            f"{name} fields mismatch; missing={missing}, unknown={unknown}"
        )


def _nonempty(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NextgenFIFOTranslationError(f"{name} must be a non-empty string")
    return value.strip()


def _number(value: Any, *, name: str, minimum: float | None = None, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NextgenFIFOTranslationError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise NextgenFIFOTranslationError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise NextgenFIFOTranslationError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise NextgenFIFOTranslationError(f"{name} must be >= {minimum}")
    return result


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NextgenFIFOTranslationError(f"{name} must be an integer")
    return value


def _vector2(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise NextgenFIFOTranslationError(f"{name} must contain exactly two numeric values")
    return (_number(value[0], name=f"{name}[0]"), _number(value[1], name=f"{name}[1]"))


def _parse_network(payload: Mapping[str, Any]) -> RouteNetwork:
    _exact_keys(payload, name="model.parameters.network", keys={"nodes", "edges"})
    raw_nodes = payload["nodes"]
    raw_edges = payload["edges"]
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise NextgenFIFOTranslationError("network.nodes must be a non-empty array")
    if not isinstance(raw_edges, list) or not raw_edges:
        raise NextgenFIFOTranslationError("network.edges must be a non-empty array")
    nodes: list[RouteNetworkNode] = []
    for index, raw in enumerate(raw_nodes):
        item = _mapping(raw, name=f"network.nodes[{index}]")
        _exact_keys(item, name=f"network.nodes[{index}]", keys={"node_id", "position_nm"})
        nodes.append(
            RouteNetworkNode(
                _nonempty(item["node_id"], name=f"network.nodes[{index}].node_id"),
                _vector2(item["position_nm"], name=f"network.nodes[{index}].position_nm"),
            )
        )
    edges: list[RouteNetworkEdge] = []
    for index, raw in enumerate(raw_edges):
        item = _mapping(raw, name=f"network.edges[{index}]")
        _exact_keys(
            item,
            name=f"network.edges[{index}]",
            keys={"edge_id", "from_node_id", "to_node_id", "waypoints_nm"},
        )
        raw_waypoints = item["waypoints_nm"]
        if not isinstance(raw_waypoints, list) or len(raw_waypoints) < 2:
            raise NextgenFIFOTranslationError(
                f"network.edges[{index}].waypoints_nm must contain at least two points"
            )
        waypoints = tuple(
            _vector2(point, name=f"network.edges[{index}].waypoints_nm[{point_index}]")
            for point_index, point in enumerate(raw_waypoints)
        )
        edges.append(
            RouteNetworkEdge(
                _nonempty(item["edge_id"], name=f"network.edges[{index}].edge_id"),
                _nonempty(item["from_node_id"], name=f"network.edges[{index}].from_node_id"),
                _nonempty(item["to_node_id"], name=f"network.edges[{index}].to_node_id"),
                Route2D(waypoints),
            )
        )
    try:
        return RouteNetwork(nodes, edges)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise NextgenFIFOTranslationError(f"invalid route network: {exc}") from exc


def _parse_maneuver(payload: Mapping[str, Any], *, name: str) -> ManeuverLimits:
    _exact_keys(
        payload,
        name=name,
        keys={
            "max_acceleration_knots_per_min",
            "max_deceleration_knots_per_min",
            "max_turn_rate_deg_per_min",
        },
    )
    return ManeuverLimits.from_knots_per_minute(
        max_acceleration_knots_per_min=_number(
            payload["max_acceleration_knots_per_min"],
            name=f"{name}.max_acceleration_knots_per_min",
            minimum=0.0,
        ),
        max_deceleration_knots_per_min=_number(
            payload["max_deceleration_knots_per_min"],
            name=f"{name}.max_deceleration_knots_per_min",
            minimum=0.0,
        ),
        max_turn_rate_deg_per_min=_number(
            payload["max_turn_rate_deg_per_min"],
            name=f"{name}.max_turn_rate_deg_per_min",
            minimum=0.0,
        ),
    )


def _parse_domain(payload: Mapping[str, Any], *, name: str) -> ShipDomain:
    kind = payload.get("type")
    if kind == "circular_nm":
        _exact_keys(payload, name=name, keys={"type", "radius_nm"})
        return ShipDomain.circular(_number(payload["radius_nm"], name=f"{name}.radius_nm", positive=True))
    if kind == "four_quadrant_m":
        _exact_keys(
            payload,
            name=name,
            keys={"type", "ahead_m", "astern_m", "port_m", "starboard_m"},
        )
        return ShipDomain.from_meters(
            ahead_m=_number(payload["ahead_m"], name=f"{name}.ahead_m", positive=True),
            astern_m=_number(payload["astern_m"], name=f"{name}.astern_m", positive=True),
            port_m=_number(payload["port_m"], name=f"{name}.port_m", positive=True),
            starboard_m=_number(payload["starboard_m"], name=f"{name}.starboard_m", positive=True),
        )
    raise NextgenFIFOTranslationError(
        f"{name}.type must be 'circular_nm' or 'four_quadrant_m'"
    )


def _parse_profiles(payload: Mapping[str, Any]) -> Mapping[str, VesselClassProfile]:
    if not payload:
        raise NextgenFIFOTranslationError("vessel_classes must contain at least one profile")
    profiles: dict[str, VesselClassProfile] = {}
    for class_id, raw in payload.items():
        class_name = _nonempty(class_id, name="vessel_classes key")
        item = _mapping(raw, name=f"vessel_classes[{class_name!r}]")
        _exact_keys(
            item,
            name=f"vessel_classes[{class_name!r}]",
            keys={"dimensions_m", "maneuver_limits", "safety_domain", "maximum_speed_knots"},
        )
        dimensions = _mapping(item["dimensions_m"], name=f"vessel_classes[{class_name!r}].dimensions_m")
        _exact_keys(
            dimensions,
            name=f"vessel_classes[{class_name!r}].dimensions_m",
            keys={"length_m", "beam_m"},
        )
        maximum = item["maximum_speed_knots"]
        speed_envelope = (
            None
            if maximum is None
            else VesselSpeedEnvelope.from_knots(
                maximum_speed_knots=_number(
                    maximum,
                    name=f"vessel_classes[{class_name!r}].maximum_speed_knots",
                    positive=True,
                )
            )
        )
        profiles[class_name] = VesselClassProfile(
            class_name,
            VesselDimensions.from_meters(
                length_m=_number(
                    dimensions["length_m"],
                    name=f"vessel_classes[{class_name!r}].dimensions_m.length_m",
                    positive=True,
                ),
                beam_m=_number(
                    dimensions["beam_m"],
                    name=f"vessel_classes[{class_name!r}].dimensions_m.beam_m",
                    positive=True,
                ),
            ),
            _parse_maneuver(
                _mapping(item["maneuver_limits"], name=f"vessel_classes[{class_name!r}].maneuver_limits"),
                name=f"vessel_classes[{class_name!r}].maneuver_limits",
            ),
            _parse_domain(
                _mapping(item["safety_domain"], name=f"vessel_classes[{class_name!r}].safety_domain"),
                name=f"vessel_classes[{class_name!r}].safety_domain",
            ),
            speed_envelope,
        )
    return MappingProxyType(profiles)


def _parse_streams(
    payload: Any,
    *,
    profiles: Mapping[str, VesselClassProfile],
) -> tuple[PoissonNetworkStreamSpec, ...]:
    if not isinstance(payload, list) or not payload:
        raise NextgenFIFOTranslationError("streams must be a non-empty array")
    streams: list[PoissonNetworkStreamSpec] = []
    for index, raw in enumerate(payload):
        item = _mapping(raw, name=f"streams[{index}]")
        _exact_keys(
            item,
            name=f"streams[{index}]",
            keys={
                "stream_id",
                "origin_node_id",
                "destination_node_id",
                "route_policy",
                "rate_per_hour",
                "cruise_speed_knots",
                "end_action",
                "recovery_max_intercept_angle_deg",
                "vessel_class_id",
            },
        )
        class_id = _nonempty(item["vessel_class_id"], name=f"streams[{index}].vessel_class_id")
        try:
            profile = profiles[class_id]
        except KeyError as exc:
            raise NextgenFIFOTranslationError(
                f"streams[{index}] references unknown vessel_class_id {class_id!r}"
            ) from exc
        try:
            policy = RouteSelectionPolicy(item["route_policy"])
        except (TypeError, ValueError) as exc:
            raise NextgenFIFOTranslationError(
                f"streams[{index}].route_policy is unsupported"
            ) from exc
        try:
            end_action = RouteEndAction(item["end_action"])
        except (TypeError, ValueError) as exc:
            raise NextgenFIFOTranslationError(
                f"streams[{index}].end_action is unsupported"
            ) from exc
        streams.append(
            PoissonNetworkStreamSpec(
                stream_id=_nonempty(item["stream_id"], name=f"streams[{index}].stream_id"),
                request=RouteRequest(
                    _nonempty(item["origin_node_id"], name=f"streams[{index}].origin_node_id"),
                    _nonempty(item["destination_node_id"], name=f"streams[{index}].destination_node_id"),
                    policy,
                ),
                rate_per_minute=_number(
                    item["rate_per_hour"], name=f"streams[{index}].rate_per_hour", minimum=0.0
                )
                / 60.0,
                cruise_speed=knots_to_nm_per_minute(
                    _number(
                        item["cruise_speed_knots"],
                        name=f"streams[{index}].cruise_speed_knots",
                        positive=True,
                    )
                ),
                end_action=end_action,
                recovery_max_intercept_angle_deg=_number(
                    item["recovery_max_intercept_angle_deg"],
                    name=f"streams[{index}].recovery_max_intercept_angle_deg",
                    minimum=0.0,
                ),
                limits=profile.maneuver_limits,
                vessel_class=profile,
            )
        )
    return tuple(streams)


def translate_nextgen_fifo_config(
    scenario: DiagnosticScenarioSpec,
    *,
    profiles: NextgenFIFOSemanticsProfileRegistry,
) -> EnvironmentalNetworkFIFOExperimentConfig:
    """Translate the supported D0 parameter subset into the canonical A152 config."""

    if not isinstance(scenario, DiagnosticScenarioSpec):
        raise TypeError("scenario must be DiagnosticScenarioSpec")
    if scenario.model_adapters.get("runtime") != NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_ADAPTER_ID:
        raise NextgenFIFOTranslationError(
            f"scenario runtime adapter must be {NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_ADAPTER_ID!r}"
        )
    parameters = _mapping(scenario.model_parameters, name="model.parameters")
    _exact_keys(
        parameters,
        name="model.parameters",
        keys={"semantics_profile", "network", "vessel_classes", "streams", "scenario", "label"},
    )
    profile = profiles.resolve(
        _nonempty(parameters["semantics_profile"], name="model.parameters.semantics_profile")
    )
    network = _parse_network(_mapping(parameters["network"], name="model.parameters.network"))
    vessel_profiles = _parse_profiles(
        _mapping(parameters["vessel_classes"], name="model.parameters.vessel_classes")
    )
    streams = _parse_streams(parameters["streams"], profiles=vessel_profiles)
    scenario_parameters = _mapping(parameters["scenario"], name="model.parameters.scenario")
    _exact_keys(
        scenario_parameters,
        name="model.parameters.scenario",
        keys={
            "seed",
            "traffic_horizon_min",
            "start_time",
            "end_time",
            "default_environmental_drift_nm_per_min",
            "stream_environmental_drift_nm_per_min",
        },
    )
    stream_drifts_raw = _mapping(
        scenario_parameters["stream_environmental_drift_nm_per_min"],
        name="model.parameters.scenario.stream_environmental_drift_nm_per_min",
    )
    stream_ids = {item.stream_id for item in streams}
    unknown_drift_streams = set(stream_drifts_raw) - stream_ids
    if unknown_drift_streams:
        raise NextgenFIFOTranslationError(
            f"environmental drift configured for unknown streams: {sorted(unknown_drift_streams)}"
        )
    stream_drifts = {
        stream_id: _vector2(
            drift,
            name=f"model.parameters.scenario.stream_environmental_drift_nm_per_min[{stream_id!r}]",
        )
        for stream_id, drift in stream_drifts_raw.items()
    }
    canonical_scenario = EnvironmentalNetworkScenarioConfig(
        seed=_integer(scenario_parameters["seed"], name="model.parameters.scenario.seed"),
        traffic_horizon_min=_number(
            scenario_parameters["traffic_horizon_min"],
            name="model.parameters.scenario.traffic_horizon_min",
            minimum=0.0,
        ),
        discovery=profile.discovery,
        lifecycle=profile.lifecycle,
        recapture=profile.recapture,
        start_time=_number(
            scenario_parameters["start_time"],
            name="model.parameters.scenario.start_time",
            minimum=0.0,
        ),
        end_time=_number(scenario_parameters["end_time"], name="model.parameters.scenario.end_time"),
        default_environmental_drift_velocity=_vector2(
            scenario_parameters["default_environmental_drift_nm_per_min"],
            name="model.parameters.scenario.default_environmental_drift_nm_per_min",
        ),
        stream_environmental_drift_velocity=stream_drifts,
        runtime=profile.runtime,
        ingress_timing_policy=profile.ingress_timing_policy,
    )
    label = parameters["label"]
    if label is not None:
        label = _nonempty(label, name="model.parameters.label")
    try:
        return EnvironmentalNetworkFIFOExperimentConfig(
            network=network,
            streams=streams,
            scenario=canonical_scenario,
            recheck=profile.recheck,
            admission_policy=profile.admission_policy,
            label=label,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise NextgenFIFOTranslationError(f"canonical FIFO config rejected translated input: {exc}") from exc
