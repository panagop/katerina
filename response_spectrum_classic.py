"""Compute elastic response spectra without OpenSeesPy.

This module uses classic SDOF structural dynamics and Newmark-beta integration
to compute elastic spectral displacement, pseudo-velocity, pseudo-acceleration,
and true peak spectral acceleration/velocity for a ground motion record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from elcentro_loader import load_elcentro

G = 9.81  # m/s²


@dataclass(frozen=True)
class ResponseSpectrum:
    periods: np.ndarray
    sa_pseudo: np.ndarray
    sv_pseudo: np.ndarray
    sa_real: np.ndarray
    sv_real: np.ndarray
    sd: np.ndarray


def _newmark_beta_sdof_response(
    ag_ms2: np.ndarray,
    dt: float,
    omega: float,
    xi: float,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> tuple[float, float, float, float]:
    """Integrate a single SDOF oscillator and return peak responses."""
    m = 1.0
    c = 2.0 * m * xi * omega
    k = m * omega**2

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    keff = k + a1 * c + a0 * m

    u = 0.0
    udot = 0.0
    uddot = 0.0

    max_u = 0.0
    max_udot = 0.0
    max_abs_acc = 0.0

    for ag in ag_ms2:
        p = -m * ag

        p_eff = (
            p
            + m * (a0 * u + a2 * udot + a3 * uddot)
            + c * (a1 * u + a4 * udot + a5 * uddot)
        )

        u_new = p_eff / keff
        uddot_new = a0 * (u_new - u) - a2 * udot - a3 * uddot
        udot_new = udot + dt * ((1.0 - gamma) * uddot + gamma * uddot_new)

        abs_acc = abs(uddot_new + ag)

        max_u = max(max_u, abs(u_new))
        max_udot = max(max_udot, abs(udot_new))
        max_abs_acc = max(max_abs_acc, abs_acc)

        u, udot, uddot = u_new, udot_new, uddot_new

    return max_u, max_udot, max_abs_acc, omega


def compute_response_spectrum(
    ag_g: np.ndarray,
    dt: float,
    xi: float = 0.05,
    periods: np.ndarray | None = None,
) -> ResponseSpectrum:
    """Compute elastic response spectra for a ground motion record.

    Parameters
    ----------
    ag_g : np.ndarray
        Ground acceleration time series in g.
    dt : float
        Time step in seconds.
    xi : float
        Damping ratio (default 0.05).
    periods : np.ndarray | None
        Array of oscillator periods. Defaults to 200 values from 0.01 to 4 s.
    """
    if periods is None:
        periods = np.linspace(0.01, 4.0, 200)

    ag_ms2 = ag_g * G

    sd = np.zeros_like(periods)
    sv_pseudo = np.zeros_like(periods)
    sa_pseudo = np.zeros_like(periods)
    sv_real = np.zeros_like(periods)
    sa_real = np.zeros_like(periods)

    for i, T in enumerate(periods):
        omega = 2.0 * np.pi / T
        max_u, max_udot, max_abs_acc, _ = _newmark_beta_sdof_response(
            ag_ms2=ag_ms2,
            dt=dt,
            omega=omega,
            xi=xi,
        )

        sd[i] = max_u * 100.0
        sv_pseudo[i] = omega * max_u * 100.0
        sa_pseudo[i] = omega**2 * max_u / G
        sv_real[i] = max_udot * 100.0
        sa_real[i] = max_abs_acc / G

    return ResponseSpectrum(
        periods=periods,
        sa_pseudo=sa_pseudo,
        sv_pseudo=sv_pseudo,
        sa_real=sa_real,
        sv_real=sv_real,
        sd=sd,
    )


def save_spectrum_plot(spectrum: ResponseSpectrum, filename: Path) -> None:
    """Plot and save the response spectra to a PNG file."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)

    axes[0].plot(spectrum.periods, spectrum.sa_pseudo, label="Sa pseudo", color="tab:red")
    axes[0].plot(spectrum.periods, spectrum.sa_real, label="Sa true", color="tab:blue")
    axes[0].set_ylabel("Acceleration [g]")
    axes[0].set_title("Spectral Acceleration")
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle="--", alpha=0.3)

    axes[1].plot(spectrum.periods, spectrum.sv_pseudo, label="Sv pseudo", color="tab:green")
    axes[1].plot(spectrum.periods, spectrum.sv_real, label="Sv true", color="tab:purple")
    axes[1].set_ylabel("Velocity [cm/s]")
    axes[1].set_title("Spectral Velocity")
    axes[1].legend()
    axes[1].grid(True, which="both", linestyle="--", alpha=0.3)

    axes[2].plot(spectrum.periods, spectrum.sd, label="Sd", color="tab:orange")
    axes[2].set_ylabel("Displacement [cm]")
    axes[2].set_xlabel("Period [s]")
    axes[2].set_title("Spectral Displacement")
    axes[2].legend()
    axes[2].grid(True, which="both", linestyle="--", alpha=0.3)

    fig.suptitle("Elastic Response Spectra", fontsize=16)
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def _print_summary(spectrum: ResponseSpectrum) -> None:
    print("Computed elastic response spectrum")
    print(f"  periods: {spectrum.periods.size} values")
    print(f"  SD range: {spectrum.sd.min():.4f} cm to {spectrum.sd.max():.4f} cm")
    print(f"  Sa (pseudo) max: {spectrum.sa_pseudo.max():.4f} g")
    print(f"  Sa (true) max:   {spectrum.sa_real.max():.4f} g")


def main() -> None:
    ag_g, dt = load_elcentro()
    spectrum = compute_response_spectrum(ag_g, dt)
    _print_summary(spectrum)

    sample_idx = np.argsort(spectrum.periods)[:5]
    print("\nSample outputs for first 5 periods:")
    for idx in sample_idx:
        print(
            f"T={spectrum.periods[idx]:.3f} s: "
            f"Sd={spectrum.sd[idx]:.4f} cm, "
            f"Sa_pseudo={spectrum.sa_pseudo[idx]:.4f} g, "
            f"Sa_real={spectrum.sa_real[idx]:.4f} g"
        )

    output_path = Path(__file__).parent / "response_spectrum_classic.png"
    save_spectrum_plot(spectrum, output_path)
    print(f"Saved response spectra plot to: {output_path}")


if __name__ == "__main__":
    main()
