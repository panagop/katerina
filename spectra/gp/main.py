"""Response spectra for the El Centro 1940 NS accelerogram."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from elcentro_loader import load_elcentro
from response_spectrum import compute, ResponseSpectrum

DAMPING = 0.05  # 5 % critical damping


def plot_accelerogram(ag_g: np.ndarray, dt: float, ax: plt.Axes) -> None:
    time = np.arange(len(ag_g)) * dt
    ax.plot(time, ag_g, color="#2c7bb6", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [g]")
    ax.set_title("El Centro 1940 NS — Ground Acceleration")
    ax.set_xlim(time[0], time[-1])
    peak = float(np.max(np.abs(ag_g)))
    peak_idx = int(np.argmax(np.abs(ag_g)))
    ax.annotate(
        f"PGA = {peak:.3f} g",
        xy=(time[peak_idx], ag_g[peak_idx]),
        xytext=(time[-1] * 0.65, peak * 1.05),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontsize=9,
    )


def plot_spectra(rs: ResponseSpectrum, xi: float, fig: plt.Figure) -> None:
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    xi_label = f"xi = {xi * 100:.0f} %"

    # --- Acceleration ---
    ax_a = fig.add_subplot(gs[0])
    ax_a.plot(rs.periods, rs.sa_real,   color="#d7191c", linewidth=1.6, label="Real $S_a$")
    ax_a.plot(rs.periods, rs.sa_pseudo, color="#d7191c", linewidth=1.2,
              linestyle="--", alpha=0.7, label="Pseudo $S_a$")
    ax_a.set_xlabel("Period  $T$  [s]")
    ax_a.set_ylabel("Spectral Acceleration  [g]")
    ax_a.set_title(f"Acceleration spectra  ({xi_label})")
    ax_a.legend(fontsize=8)

    # --- Velocity ---
    ax_v = fig.add_subplot(gs[1])
    ax_v.plot(rs.periods, rs.sv_real,   color="#1a9641", linewidth=1.6, label="Real $S_v$")
    ax_v.plot(rs.periods, rs.sv_pseudo, color="#1a9641", linewidth=1.2,
              linestyle="--", alpha=0.7, label="Pseudo $S_v$")
    ax_v.set_xlabel("Period  $T$  [s]")
    ax_v.set_ylabel("Spectral Velocity  [cm/s]")
    ax_v.set_title(f"Velocity spectra  ({xi_label})")
    ax_v.legend(fontsize=8)

    # --- Displacement ---
    ax_d = fig.add_subplot(gs[2])
    ax_d.plot(rs.periods, rs.sd, color="#2c7bb6", linewidth=1.6)
    ax_d.set_xlabel("Period  $T$  [s]")
    ax_d.set_ylabel("Spectral Displacement  $S_d$  [cm]")
    ax_d.set_title(f"Displacement spectrum  ({xi_label})")

    for ax in (ax_a, ax_v, ax_d):
        ax.set_xlim(0, rs.periods[-1])
        ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(0, rs.periods[-1] + 0.5, 0.5))
        ax.grid(True, alpha=0.3)


def main() -> None:
    print("Loading El Centro accelerogram ...")
    ag_g, dt = load_elcentro()
    duration = len(ag_g) * dt
    pga = float(np.max(np.abs(ag_g)))
    print(f"  {len(ag_g)} points  |  dt = {dt} s  |  duration = {duration:.2f} s  |  PGA = {pga:.3f} g")

    print(f"\nComputing response spectra (xi = {DAMPING * 100:.0f} %) ...")
    periods = np.linspace(0.01, 4.0, 400)
    rs = compute(ag_g, dt, xi=DAMPING, periods=periods)
    print("Done.\n")

    # --- Figure 1: accelerogram ---
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    fig1.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.18)
    plot_accelerogram(ag_g, dt, ax1)
    fig1.savefig("elcentro_accelerogram.png", dpi=150)

    # --- Figure 2: response spectra ---
    fig2 = plt.figure(figsize=(14, 4.5))
    fig2.suptitle(
        f"Elastic Response Spectra — El Centro 1940 NS  (xi = {DAMPING * 100:.0f} %)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plot_spectra(rs, DAMPING, fig2)
    fig2.savefig("response_spectra.png", dpi=150, bbox_inches="tight")

    plt.show()
    print("Plots saved: elcentro_accelerogram.png, response_spectra.png")


if __name__ == "__main__":
    main()
