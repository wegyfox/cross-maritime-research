"""Independent MMG 3-DOF reference kernel for maneuvering-fidelity diagnostics.

This module is deliberately *not* the default CROSS plant.  It exists as a
reproducible benchmark implementation of the MMG standard equations used by
ShipMMG's KVLCC2 L7 example.  Keeping it in-tree lets engineering diagnostics
run without a network-installed third-party package while preserving an
explicit implementation/provenance boundary.

Primary provenance:
- Yasukawa & Yoshimura (2015), Introduction of MMG standard method for ship
  maneuvering predictions, J. Marine Science and Technology 20, 37-52.
- ShipMMG 0.0.11, commit b004dab9cbfcfd9543ad7e74b95570c6e8aa3c0e,
  tests/test_mmg_3dof.py KVLCC2 L7 fixture and shipmmg/mmg_3dof.py equations.

Coordinate convention here follows the cited ShipMMG implementation exactly:
``u`` is body-forward, ``v`` is body-lateral, ``r`` is yaw rate [rad/s], and
``psi`` is mathematical yaw in the earth x/y frame.  CROSS north-clockwise
heading conversion belongs to an adapter, not this benchmark kernel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from math import pi
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp


SHIPMMG_REPOSITORY = "https://github.com/ShipMMG/shipmmg"
SHIPMMG_VERSION = "0.0.11"
SHIPMMG_COMMIT = "b004dab9cbfcfd9543ad7e74b95570c6e8aa3c0e"
SHIPMMG_MMG_BLOB = "f27683c3a2d21a566ee27a912fcec3952d007f76"
SHIPMMG_TEST_BLOB = "f17ac11d40674ef20c090c6e9d4f43ed0128411b"
MMG_STANDARD_DOI = "10.1007/s00773-014-0293-y"


class MmgReferenceTier(str, Enum):
    """Provenance tier for MMG fixtures and validation targets.

    ``IMPLEMENTATION_FIXTURE`` means the inputs are frozen to a software
    example/smoke fixture and are useful for equation/adapter parity only.
    ``EXPERIMENTAL_TARGET`` means the values describe a published free-running
    validation protocol or its reported target metrics.  The two tiers must not
    be compared as though they were the same experiment.
    """

    IMPLEMENTATION_FIXTURE = "IMPLEMENTATION_FIXTURE"
    EXPERIMENTAL_TARGET = "EXPERIMENTAL_TARGET"


@dataclass(frozen=True, slots=True)
class MmgExperimentalTarget:
    target_id: str
    source_label: str
    L_pp_m: float
    initial_speed_m_s: float
    target_rudder_deg: float
    rudder_rate_deg_s_model: float | None
    propeller_rate_rps: float | None
    calculated_advance_L: float | None
    experimental_advance_L: float | None
    calculated_tactical_diameter_L: float | None
    experimental_tactical_diameter_L: float | None
    notes: tuple[str, ...] = ()

    def identity_sha256(self) -> str:
        return _sha({
            "tier": MmgReferenceTier.EXPERIMENTAL_TARGET.value,
            **asdict(self),
        })


def kvlcc2_l7_yasukawa_2015_validation_target() -> MmgExperimentalTarget:
    """Published +35° L7 validation target from Yasukawa & Yoshimura (2015).

    This target intentionally does *not* pretend that ShipMMG's unit-test
    fixture reproduces the paper's validation protocol.  The paper uses a
    finite steering rate and a separate three-dimensional extrapolation for
    straight-ahead hull resistance; those details are not represented by the
    simple ShipMMG fixture's constant ``R_0_dash`` setup.
    """

    scale = 45.7
    return MmgExperimentalTarget(
        target_id="KVLCC2_L7_YASUKAWA_2015_PLUS35",
        source_label=(
            "Yasukawa & Yoshimura (2015), Table 4 / section 5; "
            f"DOI {MMG_STANDARD_DOI}"
        ),
        L_pp_m=7.0,
        initial_speed_m_s=15.5 * 0.514444 / (scale ** 0.5),
        target_rudder_deg=35.0,
        rudder_rate_deg_s_model=1.76 * (scale ** 0.5),
        propeller_rate_rps=None,
        calculated_advance_L=3.31,
        experimental_advance_L=3.25,
        calculated_tactical_diameter_L=3.36,
        experimental_tactical_diameter_L=3.34,
        notes=(
            "Paper states propeller revolution is held at the value for U0 but the exact value is not frozen here.",
            "Paper uses 3-D extrapolated straight-ahead hull resistance rather than equating validation resistance to the captive-test R_0_dash coefficient.",
        ),
    )


def kvlcc2_l7_simman_deep_water_protocol_target() -> MmgExperimentalTarget:
    """SIMMAN-2020 Case 1.2 public protocol shell, kept separate from MARIN-2011.

    The public Case-1.2 page specifies that propeller RPM must correspond to the
    longitudinal self-propulsion equilibrium at the approach speed, but does not
    publish the exact numerical RPM on that page.  Therefore this target keeps
    ``propeller_rate_rps=None`` rather than borrowing the 620 rpm MARIN-2011 value
    or the 614 rpm value associated with a different historical dataset.
    """

    return MmgExperimentalTarget(
        target_id="KVLCC2_L7_SIMMAN2020_CASE12_TC35",
        source_label="SIMMAN-2020 KVLCC2 Case 1.2 public protocol",
        L_pp_m=7.0,
        initial_speed_m_s=1.18,
        target_rudder_deg=35.0,
        rudder_rate_deg_s_model=15.68,
        propeller_rate_rps=None,
        calculated_advance_L=None,
        experimental_advance_L=None,
        calculated_tactical_diameter_L=None,
        experimental_tactical_diameter_L=None,
        notes=(
            "Protocol requires longitudinal equilibrium before the manoeuvre.",
            "Exact Case-1.2 RPM must be source-bound before publication-grade validation.",
            "Do not substitute MARIN-2011 620 rpm or another workshop's 614 rpm by convenience.",
        ),
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class MmgBasicParams:
    rho_kg_m3: float
    L_pp_m: float
    B_m: float
    d_m: float
    x_G_m: float
    D_p_m: float
    mass_kg: float
    I_zG_kg_m2: float
    A_R_m2: float
    eta: float
    m_x_kg: float
    m_y_kg: float
    J_z_kg_m2: float
    f_alpha: float
    epsilon: float
    t_R: float
    x_R_m: float
    a_H: float
    x_H_m: float
    gamma_R_minus: float
    gamma_R_plus: float
    l_R: float
    kappa: float
    t_P: float
    w_P0: float
    x_P: float


@dataclass(frozen=True, slots=True)
class MmgManeuveringParams:
    k_0: float
    k_1: float
    k_2: float
    R_0_dash: float
    X_vv_dash: float
    X_vr_dash: float
    X_rr_dash: float
    X_vvvv_dash: float
    Y_v_dash: float
    Y_r_dash: float
    Y_vvv_dash: float
    Y_vvr_dash: float
    Y_vrr_dash: float
    Y_rrr_dash: float
    N_v_dash: float
    N_r_dash: float
    N_vvv_dash: float
    N_vvr_dash: float
    N_vrr_dash: float
    N_rrr_dash: float


@dataclass(frozen=True, slots=True)
class MmgBenchmarkProfile:
    profile_id: str
    source_label: str
    basic: MmgBasicParams
    maneuvering: MmgManeuveringParams
    initial_u_m_s: float
    constant_rudder_deg: float
    constant_nps_rps: float

    def identity_sha256(self) -> str:
        return _sha(
            {
                "profile_id": self.profile_id,
                "source_label": self.source_label,
                "basic": asdict(self.basic),
                "maneuvering": asdict(self.maneuvering),
                "initial_u_m_s": self.initial_u_m_s,
                "constant_rudder_deg": self.constant_rudder_deg,
                "constant_nps_rps": self.constant_nps_rps,
                "shipmmg_version": SHIPMMG_VERSION,
                "shipmmg_commit": SHIPMMG_COMMIT,
                "shipmmg_mmg_blob": SHIPMMG_MMG_BLOB,
                "shipmmg_test_blob": SHIPMMG_TEST_BLOB,
                "mmg_standard_doi": MMG_STANDARD_DOI,
            }
        )


@dataclass(frozen=True, slots=True)
class MmgTrajectory:
    time_s: np.ndarray
    u_m_s: np.ndarray
    v_m_s: np.ndarray
    r_rad_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    psi_rad: np.ndarray

    def final_payload(self) -> dict[str, float]:
        return {
            "time_s": float(self.time_s[-1]),
            "u_m_s": float(self.u_m_s[-1]),
            "v_m_s": float(self.v_m_s[-1]),
            "r_rad_s": float(self.r_rad_s[-1]),
            "x_m": float(self.x_m[-1]),
            "y_m": float(self.y_m[-1]),
            "psi_rad": float(self.psi_rad[-1]),
        }


def kvlcc2_l7_shipmmg_profile() -> MmgBenchmarkProfile:
    """Return ShipMMG's exact KVLCC2 L7 *implementation fixture*.

    This is not a free-running experimental validation protocol: the ShipMMG
    test applies 35° immediately and uses 17.95 rps.  Use it for equation and
    adapter parity only.
    """

    rho = 1025.0
    L = 7.00
    B = 1.27
    d = 0.46
    nabla = 3.27
    x_G = 0.25
    D_p = 0.216
    H_R = 0.345
    A_R = 0.0539
    basic = MmgBasicParams(
        rho_kg_m3=rho,
        L_pp_m=L,
        B_m=B,
        d_m=d,
        x_G_m=x_G,
        D_p_m=D_p,
        mass_kg=rho * nabla,
        I_zG_kg_m2=rho * nabla * ((0.25 * L) ** 2),
        A_R_m2=A_R,
        eta=D_p / H_R,
        m_x_kg=(0.5 * rho * L**2 * d) * 0.022,
        m_y_kg=(0.5 * rho * L**2 * d) * 0.223,
        J_z_kg_m2=(0.5 * rho * L**4 * d) * 0.011,
        f_alpha=2.747,
        epsilon=1.09,
        t_R=0.387,
        x_R_m=-0.500 * L,
        a_H=0.312,
        x_H_m=-0.464 * L,
        gamma_R_minus=0.395,
        gamma_R_plus=0.640,
        l_R=-0.710,
        kappa=0.50,
        t_P=0.220,
        w_P0=0.40,
        x_P=-0.690,
    )
    maneuvering = MmgManeuveringParams(
        k_0=0.2931,
        k_1=-0.2753,
        k_2=-0.1385,
        R_0_dash=0.022,
        X_vv_dash=-0.040,
        X_vr_dash=0.002,
        X_rr_dash=0.011,
        X_vvvv_dash=0.771,
        Y_v_dash=-0.315,
        Y_r_dash=0.083,
        Y_vvv_dash=-1.607,
        Y_vvr_dash=0.379,
        Y_vrr_dash=-0.391,
        Y_rrr_dash=0.008,
        N_v_dash=-0.137,
        N_r_dash=-0.049,
        N_vvv_dash=-0.030,
        N_vvr_dash=-0.294,
        N_vrr_dash=0.055,
        N_rrr_dash=-0.013,
    )
    return MmgBenchmarkProfile(
        profile_id="KVLCC2_L7_SHIPMMG_0_0_11",
        source_label=(
            "ShipMMG 0.0.11 implementation fixture tests/test_mmg_3dof.py at " + SHIPMMG_COMMIT
        ),
        basic=basic,
        maneuvering=maneuvering,
        initial_u_m_s=2.29 * 0.512,
        constant_rudder_deg=35.0,
        constant_nps_rps=17.95,
    )


def mmg_derivatives(
    state: np.ndarray,
    *,
    basic: MmgBasicParams,
    maneuvering: MmgManeuveringParams,
    rudder_rad: float,
    nps_rps: float,
) -> np.ndarray:
    """Evaluate the ShipMMG/Yasukawa-Yoshimura 3-DOF equations."""

    u, v, r, _x, _y, psi = map(float, state)
    p = basic
    q = maneuvering
    U = np.sqrt(u**2 + (v - r * p.x_G_m) ** 2)
    beta = 0.0 if U == 0.0 else np.arcsin(-(v - r * p.x_G_m) / U)
    v_dash = 0.0 if U == 0.0 else v / U
    r_dash = 0.0 if U == 0.0 else r * p.L_pp_m / U

    w_P = p.w_P0 * np.exp(-4.0 * (beta - p.x_P * r_dash) ** 2)
    J = 0.0 if nps_rps == 0.0 else (1.0 - w_P) * u / (nps_rps * p.D_p_m)
    K_T = q.k_0 + q.k_1 * J + q.k_2 * J**2
    beta_R = beta - p.l_R * r_dash
    gamma_R = p.gamma_R_minus if beta_R < 0.0 else p.gamma_R_plus
    v_R = U * gamma_R * beta_R
    if J == 0.0:
        u_R = np.sqrt(
            p.eta
            * (
                p.kappa
                * p.epsilon
                * 8.0
                * q.k_0
                * nps_rps**2
                * p.D_p_m**4
                / np.pi
            )
            ** 2
        )
    else:
        u_R = (
            u
            * (1.0 - w_P)
            * p.epsilon
            * np.sqrt(
                p.eta
                * (
                    1.0
                    + p.kappa
                    * (np.sqrt(1.0 + 8.0 * K_T / (np.pi * J**2)) - 1.0)
                )
                ** 2
                + (1.0 - p.eta)
            )
        )
    U_R = np.sqrt(u_R**2 + v_R**2)
    alpha_R = rudder_rad - np.arctan2(v_R, u_R)
    F_N = 0.5 * p.A_R_m2 * p.rho_kg_m3 * p.f_alpha * U_R**2 * np.sin(alpha_R)

    X_H = (
        0.5
        * p.rho_kg_m3
        * p.L_pp_m
        * p.d_m
        * U**2
        * (
            -q.R_0_dash
            + q.X_vv_dash * v_dash**2
            + q.X_vr_dash * v_dash * r_dash
            + q.X_rr_dash * r_dash**2
            + q.X_vvvv_dash * v_dash**4
        )
    )
    X_R = -(1.0 - p.t_R) * F_N * np.sin(rudder_rad)
    X_P = (1.0 - p.t_P) * p.rho_kg_m3 * K_T * nps_rps**2 * p.D_p_m**4
    Y_H = (
        0.5
        * p.rho_kg_m3
        * p.L_pp_m
        * p.d_m
        * U**2
        * (
            q.Y_v_dash * v_dash
            + q.Y_r_dash * r_dash
            + q.Y_vvv_dash * v_dash**3
            + q.Y_vvr_dash * v_dash**2 * r_dash
            + q.Y_vrr_dash * v_dash * r_dash**2
            + q.Y_rrr_dash * r_dash**3
        )
    )
    Y_R = -(1.0 + p.a_H) * F_N * np.cos(rudder_rad)
    N_H = (
        0.5
        * p.rho_kg_m3
        * p.L_pp_m**2
        * p.d_m
        * U**2
        * (
            q.N_v_dash * v_dash
            + q.N_r_dash * r_dash
            + q.N_vvv_dash * v_dash**3
            + q.N_vvr_dash * v_dash**2 * r_dash
            + q.N_vrr_dash * v_dash * r_dash**2
            + q.N_rrr_dash * r_dash**3
        )
    )
    N_R = -(p.x_R_m + p.a_H * p.x_H_m) * F_N * np.cos(rudder_rad)

    m = p.mass_kg
    d_u = (X_H + X_R + X_P + (m + p.m_y_kg) * v * r + p.x_G_m * m * r**2) / (
        m + p.m_x_kg
    )
    inertia = p.I_zG_kg_m2 + p.J_z_kg_m2 + p.x_G_m**2 * m
    denom = inertia * (m + p.m_y_kg) - p.x_G_m**2 * m**2
    d_v = (
        p.x_G_m**2 * m**2 * u * r
        - (N_H + N_R) * p.x_G_m * m
        + (Y_H + Y_R - (m + p.m_x_kg) * u * r) * inertia
    ) / denom
    d_r = (N_H + N_R - p.x_G_m * m * (d_v + u * r)) / inertia
    d_x = u * np.cos(psi) - v * np.sin(psi)
    d_y = u * np.sin(psi) + v * np.cos(psi)
    d_psi = r
    return np.asarray([d_u, d_v, d_r, d_x, d_y, d_psi], dtype=float)


def simulate_control_functions(
    profile: MmgBenchmarkProfile,
    *,
    duration_s: float,
    rudder_deg_at: Callable[[float], float],
    nps_rps_at: Callable[[float], float],
    initial_u_m_s: float | None = None,
    initial_v_m_s: float = 0.0,
    initial_r_rad_s: float = 0.0,
    initial_x_m: float = 0.0,
    initial_y_m: float = 0.0,
    initial_psi_rad: float = 0.0,
    samples: int = 2001,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> MmgTrajectory:
    """Integrate the MMG reference kernel with explicit physical controls.

    This function deliberately accepts rudder angle and shaft rate, not a
    navigation-level desired heading/speed.  Autopilot/actuator dynamics belong
    outside the MMG hydrodynamic kernel.
    """

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if samples < 2:
        raise ValueError("samples must be >= 2")
    u0 = profile.initial_u_m_s if initial_u_m_s is None else float(initial_u_m_s)
    t = np.linspace(0.0, duration_s, samples)
    y0 = np.asarray(
        [u0, initial_v_m_s, initial_r_rad_s, initial_x_m, initial_y_m, initial_psi_rad],
        dtype=float,
    )

    def rhs(now_s: float, state: np.ndarray) -> np.ndarray:
        rudder_rad = float(rudder_deg_at(now_s)) * pi / 180.0
        nps = float(nps_rps_at(now_s))
        if not np.isfinite(rudder_rad) or not np.isfinite(nps):
            raise ValueError("MMG control functions must return finite values")
        return mmg_derivatives(
            state,
            basic=profile.basic,
            maneuvering=profile.maneuvering,
            rudder_rad=rudder_rad,
            nps_rps=nps,
        )

    sol = solve_ivp(
        rhs,
        (0.0, duration_s),
        y0,
        t_eval=t,
        dense_output=False,
        rtol=rtol,
        atol=atol,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(f"MMG integration failed: {sol.message}")
    return MmgTrajectory(t, *sol.y)


def straight_ahead_acceleration_m_s2(
    profile: MmgBenchmarkProfile,
    *,
    speed_m_s: float | None = None,
    nps_rps: float | None = None,
) -> float:
    """Return initial du/dt at zero rudder as an equilibrium diagnostic."""

    u = profile.initial_u_m_s if speed_m_s is None else float(speed_m_s)
    nps = profile.constant_nps_rps if nps_rps is None else float(nps_rps)
    d = mmg_derivatives(
        np.asarray([u, 0.0, 0.0, 0.0, 0.0, 0.0]),
        basic=profile.basic,
        maneuvering=profile.maneuvering,
        rudder_rad=0.0,
        nps_rps=nps,
    )
    return float(d[0])


def simulate_rudder_ramp(
    profile: MmgBenchmarkProfile,
    *,
    duration_s: float,
    target_rudder_deg: float,
    rudder_rate_deg_s: float,
    nps_rps: float,
    initial_u_m_s: float | None = None,
    samples: int = 2001,
) -> MmgTrajectory:
    """Convenience diagnostic with a finite-rate rudder ramp and fixed shaft rate."""

    if rudder_rate_deg_s <= 0.0:
        raise ValueError("rudder_rate_deg_s must be positive")
    sign = 1.0 if target_rudder_deg >= 0.0 else -1.0
    magnitude = abs(target_rudder_deg)
    return simulate_control_functions(
        profile,
        duration_s=duration_s,
        rudder_deg_at=lambda t: sign * min(magnitude, rudder_rate_deg_s * t),
        nps_rps_at=lambda _t: nps_rps,
        initial_u_m_s=initial_u_m_s,
        samples=samples,
    )


def simulate_constant_controls(
    profile: MmgBenchmarkProfile,
    *,
    duration_s: float = 200.0,
    samples: int = 2001,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> MmgTrajectory:
    return simulate_control_functions(
        profile,
        duration_s=duration_s,
        rudder_deg_at=lambda _t: profile.constant_rudder_deg,
        nps_rps_at=lambda _t: profile.constant_nps_rps,
        samples=samples,
        rtol=rtol,
        atol=atol,
    )


def turning_metrics(trajectory: MmgTrajectory, *, L_pp_m: float) -> dict[str, float | None]:
    """Return simple dimensionless turning diagnostics without overclaiming validation."""

    psi = np.unwrap(trajectory.psi_rad)
    x = trajectory.x_m
    y = trajectory.y_m

    def at_heading(deg: float, values: np.ndarray) -> float | None:
        target = deg * pi / 180.0
        idx = np.flatnonzero(np.abs(psi) >= target)
        if len(idx) == 0:
            return None
        i = int(idx[0])
        if i == 0:
            return float(values[0])
        a0, a1 = abs(psi[i - 1]), abs(psi[i])
        if a1 == a0:
            return float(values[i])
        w = (target - a0) / (a1 - a0)
        return float(values[i - 1] + w * (values[i] - values[i - 1]))

    advance = at_heading(90.0, x)
    tactical = at_heading(180.0, y)
    return {
        "advance_m_at_90deg": advance,
        "advance_L": None if advance is None else advance / L_pp_m,
        "tactical_y_m_at_180deg": tactical,
        "tactical_y_L": None if tactical is None else tactical / L_pp_m,
        "max_abs_y_m": float(np.max(np.abs(y))),
        "final_heading_deg": float(psi[-1] * 180.0 / pi),
    }
