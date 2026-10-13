"""CROSS <-> independent MMG-reference adapter for fidelity diagnostics.

The MMG kernel consumes physical rudder angle / shaft rate and uses the
ShipMMG/Yasukawa-Yoshimura body convention.  CROSS uses east/north positions,
marine heading (0=north, clockwise positive), NM/min and clockwise-positive yaw.
This module is the explicit conversion boundary between those conventions.

It is intentionally diagnostic-only.  It does not infer rudder/shaft commands
from desired heading/speed and it does not claim experimental KVLCC2 validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import degrees, hypot, pi, radians

import numpy as np
from scipy.integrate import solve_ivp

from vessel_crossing.kinematics import nm_per_minute_to_knots
from vessel_crossing.nextgen.dynamics import normalize_heading

from .mmg_reference import MmgBenchmarkProfile, kvlcc2_l7_shipmmg_profile, mmg_derivatives
from .plant_fidelity import (
    ExternalPlantOutcome,
    ExternalCoSimulationPlantAdapter,
    PlantModelKind,
    PlantSpec,
    PlantStepRequest,
)

_METRES_PER_NM = 1852.0
_SECONDS_PER_MIN = 60.0


def nm_per_min_to_m_per_s(value: float) -> float:
    return float(value) * _METRES_PER_NM / _SECONDS_PER_MIN


def m_per_s_to_nm_per_min(value: float) -> float:
    return float(value) * _SECONDS_PER_MIN / _METRES_PER_NM


def cross_heading_to_mmg_psi_rad(heading_deg: float) -> float:
    """Marine true heading -> mathematical CCW yaw from +x/east."""
    return radians(90.0 - float(heading_deg))


def mmg_psi_rad_to_cross_heading_deg(psi_rad: float) -> float:
    return normalize_heading(90.0 - degrees(float(psi_rad)))


def cross_yaw_deg_per_min_to_mmg_r_rad_s(yaw_deg_per_min: float) -> float:
    # CROSS positive = clockwise/starboard; MMG mathematical positive = CCW.
    return -float(yaw_deg_per_min) * pi / 180.0 / _SECONDS_PER_MIN


def mmg_r_rad_s_to_cross_yaw_deg_per_min(r_rad_s: float) -> float:
    return -float(r_rad_s) * 180.0 / pi * _SECONDS_PER_MIN


@dataclass(frozen=True, slots=True)
class MmgReferenceTransportConfig:
    max_step_s: float = 0.10
    rtol: float = 1e-9
    atol: float = 1e-11

    def __post_init__(self) -> None:
        if self.max_step_s <= 0.0:
            raise ValueError("max_step_s must be positive")
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError("rtol/atol must be positive")


class MmgReferenceTransport:
    """In-process transport using the independent MMG reference kernel.

    ``PlantActuatorCommand`` uses CROSS sign convention: positive rudder is a
    starboard order.  ShipMMG/MMG positive delta produces mathematical-positive
    yaw, so the adapter negates rudder sign at the boundary.

    This reference implementation is ahead-only because the source MMG formula
    used here is not validated for negative shaft rate.  Astern/reversal remains
    covered by CROSS's low-order signed-surge plant until a provenance-clean
    astern-capable hydrodynamic backend is added.
    """

    transport_id = "cross-independent-mmg-reference-v1"

    def __init__(
        self,
        *,
        profile: MmgBenchmarkProfile | None = None,
        config: MmgReferenceTransportConfig = MmgReferenceTransportConfig(),
    ) -> None:
        self.profile = profile or kvlcc2_l7_shipmmg_profile()
        self.config = config

    def exchange(self, request: PlantStepRequest) -> ExternalPlantOutcome:
        command = request.actuator_command
        if command is None:
            return ExternalPlantOutcome(None, "mmg_reference_requires_actuator_command")
        if command.target_propeller_rate_rps < 0.0:
            return ExternalPlantOutcome(None, "mmg_reference_astern_not_supported")

        state = request.state
        vessel = state.vessel
        if state.has_body_velocity:
            assert state.surge_velocity_nm_per_min is not None
            assert state.sway_velocity_nm_per_min is not None
            u0 = nm_per_min_to_m_per_s(state.surge_velocity_nm_per_min)
            # Both CROSS rich state and ShipMMG convention use +v to port.
            v0 = nm_per_min_to_m_per_s(state.sway_velocity_nm_per_min)
        else:
            u0 = nm_per_min_to_m_per_s(vessel.speed)
            v0 = 0.0
        r0 = cross_yaw_deg_per_min_to_mmg_r_rad_s(state.yaw_rate_deg_per_min)
        x0 = vessel.position[0] * _METRES_PER_NM
        y0 = vessel.position[1] * _METRES_PER_NM
        psi0 = cross_heading_to_mmg_psi_rad(vessel.heading_deg)

        duration_s = request.duration_min * _SECONDS_PER_MIN
        if duration_s <= 0.0:
            return ExternalPlantOutcome(
                {
                    "vessel_id": vessel.vessel_id,
                    "position_nm": [vessel.position[0], vessel.position[1]],
                    "heading_deg": vessel.heading_deg,
                    "speed_knots": nm_per_minute_to_knots(vessel.speed),
                    "surge_velocity_knots": nm_per_minute_to_knots(
                        state.surge_velocity_nm_per_min if state.has_body_velocity else vessel.speed
                    ),
                    "sway_velocity_knots": nm_per_minute_to_knots(
                        state.sway_velocity_nm_per_min if state.has_body_velocity else 0.0
                    ),
                    "yaw_rate_deg_per_min": state.yaw_rate_deg_per_min,
                    "distance_nm": 0.0,
                    "integration_steps": 0,
                    "actuator_state": {
                        "rudder_angle_deg": command.target_rudder_angle_deg,
                        "propeller_rate_rps": command.target_propeller_rate_rps,
                    },
                    "opaque_state": {
                        "mmg_profile_sha256": self.profile.identity_sha256(),
                        "mmg_transport": self.transport_id,
                    },
                }
            )

        delta_mmg_rad = -radians(command.target_rudder_angle_deg)
        nps = command.target_propeller_rate_rps
        y_init = np.asarray([u0, v0, r0, x0, y0, psi0], dtype=float)
        sol = solve_ivp(
            lambda _t, y: mmg_derivatives(
                y,
                basic=self.profile.basic,
                maneuvering=self.profile.maneuvering,
                rudder_rad=delta_mmg_rad,
                nps_rps=nps,
            ),
            (0.0, duration_s),
            y_init,
            rtol=self.config.rtol,
            atol=self.config.atol,
            max_step=min(self.config.max_step_s, duration_s),
            dense_output=False,
        )
        if not sol.success:
            return ExternalPlantOutcome(None, f"mmg_integration_failed:{sol.message}")

        u1, v1, r1, x1, y1, psi1 = map(float, sol.y[:, -1])
        speed1_m_s = hypot(u1, v1)
        # Integrate speed along the returned adaptive trajectory for distance.
        speed_hist = np.hypot(sol.y[0], sol.y[1])
        distance_m = float(np.trapezoid(speed_hist, sol.t))
        return ExternalPlantOutcome(
            {
                "vessel_id": vessel.vessel_id,
                "position_nm": [x1 / _METRES_PER_NM, y1 / _METRES_PER_NM],
                "heading_deg": mmg_psi_rad_to_cross_heading_deg(psi1),
                "speed_knots": speed1_m_s * 3600.0 / _METRES_PER_NM,
                "surge_velocity_knots": u1 * 3600.0 / _METRES_PER_NM,
                "sway_velocity_knots": v1 * 3600.0 / _METRES_PER_NM,
                "yaw_rate_deg_per_min": mmg_r_rad_s_to_cross_yaw_deg_per_min(r1),
                "distance_nm": distance_m / _METRES_PER_NM,
                "integration_steps": max(0, len(sol.t) - 1),
                "actuator_state": {
                    "rudder_angle_deg": command.target_rudder_angle_deg,
                    "propeller_rate_rps": command.target_propeller_rate_rps,
                },
                "opaque_state": {
                    "mmg_profile_sha256": self.profile.identity_sha256(),
                    "mmg_transport": self.transport_id,
                },
            }
        )


def make_kvlcc2_l7_mmg_reference_adapter(
    *, config: MmgReferenceTransportConfig = MmgReferenceTransportConfig()
) -> ExternalCoSimulationPlantAdapter:
    profile = kvlcc2_l7_shipmmg_profile()
    spec = PlantSpec(
        PlantModelKind.EXTERNAL_COSIM,
        "kvlcc2-l7-independent-mmg-reference",
        "1",
        "cross-shipmmg-coordinate-adapter-v1",
        tuple(sorted({
            "profile_sha256": profile.identity_sha256(),
            "shipmmg_reference_version": "0.0.11",
            "transport_id": MmgReferenceTransport.transport_id,
        }.items())),
    )
    return ExternalCoSimulationPlantAdapter(
        spec=spec,
        transport=MmgReferenceTransport(profile=profile, config=config),
    )
