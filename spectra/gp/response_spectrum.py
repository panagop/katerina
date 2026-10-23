"""Computes elastic response spectra (Sa, Sv, Sd) via OpenSeesPy SDOF analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openseespy.opensees as ops


G = 9.81  # m/s²


@dataclass(frozen=True)
class ResponseSpectrum:
    periods: np.ndarray       # T  [s]
    sa_pseudo: np.ndarray     # pseudo-spectral acceleration  [g]
    sv_pseudo: np.ndarray     # pseudo-spectral velocity      [cm/s]
    sa_real: np.ndarray       # true spectral acceleration    [g]
    sv_real: np.ndarray       # true spectral velocity        [cm/s]
    sd: np.ndarray            # spectral displacement         [cm]


def _sdof_max_response(
    ag_ms2: np.ndarray,
    dt: float,
    period: float,
    xi: float,
) -> tuple[float, float, float]:
    """
    Run a single SDOF and return (max_disp_m, max_rel_vel_ms, max_abs_accel_ms2).

    Real Sa uses the equation of motion identity:
        ü_total = -2·xi·omega·u̇ - omega²·u
    which avoids the need to track ground acceleration at each step.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    m = 1.0
    omega = 2.0 * np.pi / period
    k = omega**2 * m
    c = 2.0 * m * omega * xi

    ops.node(1, 0.0)
    ops.node(2, 0.0)
    ops.fix(1, 1)
    ops.mass(2, m)

    # Spring
    ops.uniaxialMaterial("Elastic", 1, k)
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)

    # Dashpot (explicit viscous damper — exact for SDOF, no Rayleigh approximation)
    ops.uniaxialMaterial("Viscous", 2, c, 1.0)
    ops.element("zeroLength", 2, 1, 2, "-mat", 2, "-dir", 1)

    # Ground motion (already in m/s²; factor=1.0)
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *ag_ms2.tolist(), "-factor", 1.0)
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 25)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    max_disp = 0.0
    max_rel_vel = 0.0
    max_abs_accel = 0.0

    for _ in range(len(ag_ms2)):
        ok = ops.analyze(1, dt)
        if ok != 0:
            break
        u = ops.nodeDisp(2, 1)
        udot = ops.nodeVel(2, 1)

        d = abs(u)
        v = abs(udot)
        # Total acceleration from equation of motion: ü_total = -2ξω·u̇ - ω²·u
        a = abs(-2.0 * xi * omega * udot - omega**2 * u)

        if d > max_disp:
            max_disp = d
        if v > max_rel_vel:
            max_rel_vel = v
        if a > max_abs_accel:
            max_abs_accel = a

    return max_disp, max_rel_vel, max_abs_accel


def compute(
    ag_g: np.ndarray,
    dt: float,
    xi: float = 0.05,
    periods: np.ndarray | None = None,
) -> ResponseSpectrum:
    """
    Compute elastic response spectra for a ground motion record.

    Parameters
    ----------
    ag_g   : acceleration time series in g
    dt     : time step in seconds
    xi     : damping ratio (default 5 %)
    periods: array of periods to evaluate (default: 200 log-spaced from 0.01–4 s)
    """
    if periods is None:
        periods = np.logspace(np.log10(0.01), np.log10(4.0), 200)

    ag_ms2 = ag_g * G  # convert to m/s²

    sd_list: list[float] = []
    sv_pseudo_list: list[float] = []
    sa_pseudo_list: list[float] = []
    sv_real_list: list[float] = []
    sa_real_list: list[float] = []

    n = len(periods)
    for i, T in enumerate(periods):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Computing period {i + 1}/{n}  (T = {T:.3f} s) ...")

        omega = 2.0 * np.pi / T
        sd, sv_r, sa_r = _sdof_max_response(ag_ms2, dt, T, xi)

        sd_list.append(sd * 100.0)                    # cm
        sv_pseudo_list.append(omega * sd * 100.0)     # cm/s
        sa_pseudo_list.append(omega**2 * sd / G)      # g
        sv_real_list.append(sv_r * 100.0)             # cm/s
        sa_real_list.append(sa_r / G)                 # g

    ops.wipe()
    return ResponseSpectrum(
        periods=periods,
        sa_pseudo=np.array(sa_pseudo_list),
        sv_pseudo=np.array(sv_pseudo_list),
        sa_real=np.array(sa_real_list),
        sv_real=np.array(sv_real_list),
        sd=np.array(sd_list),
    )
