"""Engineering-only C16 maneuvering-realism sensitivity diagnostics.

This is not an R1 research protocol.  It compares the immutable A162 kinematic plant
against explicit low-order actuator/inertia mechanics on small deterministic cases so
model-form differences are visible before richer physics is introduced into campaign
execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from typing import Any

from vessel_crossing.nextgen.dynamics import ManeuverLimits, Vessel2DState
from vessel_crossing.testbed.behavior import RequestedManeuver
from vessel_crossing.testbed.plant_fidelity import (
    A162KinematicPlantAdapter,
    LowOrderActuatedPlantAdapter,
    LowOrderActuatedPlantParameters,
    PlantActuatorCommand,
    PlantActuatorState,
    PlantActuatorTransientState,
    PlantDynamicState,
    PlantStepRequest,
    PropulsionMapPoint,
    SteerageAuthorityPoint,
)

DIAGNOSTIC_ROLE = "ENGINEERING_DIAGNOSTIC_ONLY_NOT_RESEARCH_RESULT"


def _sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ManeuveringRealismDiagnosticConfig:
    vessel_length_m: float = 120.0
    vessel_beam_m: float = 20.0
    turn_initial_speed_knots: float = 6.0
    low_speed_knots: float = 3.0
    turn_target_heading_deg: float = 90.0
    turn_duration_min: float = 4.0
    stop_duration_min: float = 4.0
    reverse_duration_min: float = 8.0

    def __post_init__(self) -> None:
        for name in ("vessel_length_m", "vessel_beam_m", "turn_initial_speed_knots", "low_speed_knots",
                     "turn_duration_min", "stop_duration_min", "reverse_duration_min"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.turn_target_heading_deg):
            raise ValueError("turn_target_heading_deg must be finite")

    def payload(self) -> dict[str, float | str]:
        return {
            "diagnostic_role": DIAGNOSTIC_ROLE,
            "vessel_length_m": self.vessel_length_m,
            "vessel_beam_m": self.vessel_beam_m,
            "turn_initial_speed_knots": self.turn_initial_speed_knots,
            "low_speed_knots": self.low_speed_knots,
            "turn_target_heading_deg": self.turn_target_heading_deg,
            "turn_duration_min": self.turn_duration_min,
            "stop_duration_min": self.stop_duration_min,
            "reverse_duration_min": self.reverse_duration_min,
        }

    @property
    def sha256(self) -> str:
        return _sha(self.payload())


def diagnostic_limits() -> ManeuverLimits:
    return ManeuverLimits.from_knots_per_minute(
        max_acceleration_knots_per_min=1.0,
        max_deceleration_knots_per_min=1.5,
        max_turn_rate_deg_per_min=12.0,
    )


def diagnostic_actuated_parameters() -> LowOrderActuatedPlantParameters:
    """Synthetic profile used only to exercise MR mechanics, not vessel claims."""
    return LowOrderActuatedPlantParameters(
        steerage_curve=(
            SteerageAuthorityPoint(0.0, 0.0),
            SteerageAuthorityPoint(3.5, 0.0),
            SteerageAuthorityPoint(6.0, 0.5),
            SteerageAuthorityPoint(10.0, 1.0),
        ),
        propulsion_map=(
            PropulsionMapPoint(-1.0, -4.0),
            PropulsionMapPoint(0.0, 0.0),
            PropulsionMapPoint(0.5, 6.0),
            PropulsionMapPoint(1.0, 12.0),
        ),
        max_rudder_angle_deg=35.0,
        rudder_rate_deg_per_min=18.0,
        rudder_response_delay_min=0.20,
        propeller_rate_slew_rps_per_min=0.5,
        propulsion_response_delay_min=0.20,
        reversal_delay_min=0.50,
        yaw_time_constant_min=0.50,
        surge_time_constant_min=1.20,
        max_astern_acceleration_knots_per_min=0.8,
        autopilot_full_rudder_error_deg=30.0,
        max_step_min=0.01,
    )


def _base_state(speed_knots: float, propeller_rate_rps: float) -> PlantDynamicState:
    speed = speed_knots / 60.0
    return PlantDynamicState(
        vessel=Vessel2DState.from_knots("diagnostic-vessel", (0.0, 0.0), 0.0, speed_knots),
        surge_velocity_nm_per_min=speed,
        sway_velocity_nm_per_min=0.0,
        actuator_state=PlantActuatorState(0.0, propeller_rate_rps),
        actuator_transient_state=PlantActuatorTransientState(
            last_nonzero_propeller_sign=1 if propeller_rate_rps > 0 else 0,
        ),
    )


def _state_payload(state: PlantDynamicState) -> dict[str, Any]:
    return {
        "position_nm": list(state.vessel.position),
        "heading_deg": state.vessel.heading_deg,
        "speed_knots": state.vessel.speed * 60.0,
        "surge_speed_knots": None if state.surge_velocity_nm_per_min is None else state.surge_velocity_nm_per_min * 60.0,
        "sway_speed_knots": None if state.sway_velocity_nm_per_min is None else state.sway_velocity_nm_per_min * 60.0,
        "cog_deg": state.course_over_ground_deg,
        "yaw_rate_deg_per_min": state.yaw_rate_deg_per_min,
        "actuator": None if state.actuator_state is None else dict(state.actuator_state.canonical_payload()),
    }


def run_maneuvering_realism_diagnostic(
    config: ManeuveringRealismDiagnosticConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ManeuveringRealismDiagnosticConfig()
    limits = diagnostic_limits()
    parity = A162KinematicPlantAdapter()
    actuated = LowOrderActuatedPlantAdapter(diagnostic_actuated_parameters())

    turn_maneuver = RequestedManeuver(0.0, cfg.turn_target_heading_deg, cfg.turn_initial_speed_knots, "mr-diagnostic-turn")
    turn_base = _base_state(cfg.turn_initial_speed_knots, 0.5)
    turn_request = PlantStepRequest(0.0, cfg.turn_duration_min, turn_base, turn_maneuver, limits)
    parity_turn = parity.step(turn_request)
    actuated_turn = actuated.step(turn_request)

    low_maneuver = RequestedManeuver(0.0, cfg.turn_target_heading_deg, cfg.low_speed_knots, "mr-diagnostic-low-speed")
    # Map 3 kn to the ahead propulsion map (0.25 rps for the synthetic linear map).
    low_base = _base_state(cfg.low_speed_knots, 0.25)
    low_request = PlantStepRequest(0.0, cfg.turn_duration_min, low_base, low_maneuver, limits)
    parity_low = parity.step(low_request)
    actuated_low = actuated.step(low_request)

    stop_maneuver = RequestedManeuver(0.0, 0.0, 0.0, "mr-diagnostic-stop")
    stop_base = _base_state(cfg.turn_initial_speed_knots, 0.5)
    stop_request = PlantStepRequest(0.0, cfg.stop_duration_min, stop_base, stop_maneuver, limits)
    parity_stop = parity.step(stop_request)
    actuated_stop = actuated.step(stop_request)

    reverse_request = PlantStepRequest(
        0.0,
        cfg.reverse_duration_min,
        stop_base,
        stop_maneuver,
        limits,
        actuator_command=PlantActuatorCommand(0.0, 0.0, -1.0, "mr-diagnostic-full-astern"),
    )
    actuated_reverse = actuated.step(reverse_request)

    cases = {
        "turn_6kn": {
            "parity": _state_payload(parity_turn.state),
            "actuated": _state_payload(actuated_turn.state),
            "heading_lag_deg": parity_turn.state.vessel.heading_deg - actuated_turn.state.vessel.heading_deg,
        },
        "turn_low_speed": {
            "parity": _state_payload(parity_low.state),
            "actuated": _state_payload(actuated_low.state),
            "heading_lag_deg": parity_low.state.vessel.heading_deg - actuated_low.state.vessel.heading_deg,
        },
        "stop_from_6kn": {
            "parity": _state_payload(parity_stop.state),
            "actuated": _state_payload(actuated_stop.state),
            "parity_distance_nm": parity_stop.distance_nm,
            "actuated_distance_nm": actuated_stop.distance_nm,
        },
        "full_astern_from_6kn": {
            "actuated": _state_payload(actuated_reverse.state),
            "distance_nm": actuated_reverse.distance_nm,
        },
    }
    result = {
        "evidence_role": DIAGNOSTIC_ROLE,
        "config": cfg.payload(),
        "config_sha256": cfg.sha256,
        "parity_plant_sha256": parity.spec.identity_sha256(),
        "actuated_plant_sha256": actuated.spec.identity_sha256(),
        "profile_note": "synthetic mechanics fixture; not calibrated to a named vessel",
        "cases": cases,
    }
    result["result_sha256"] = _sha(result)
    return result
