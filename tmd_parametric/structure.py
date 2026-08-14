"""
Παράμετροι κατασκευής και σχεδιασμός του TMD.

Οι σχέσεις είναι των Tsai & Lin (1993) για βέλτιστο TMD σε σύστημα ενός βαθμού
ελευθερίας με απόσβεση, υπό διέγερση στήριξης (base excitation).

Μονάδες: μάζα σε tn, δυσκαμψία σε kN/m, μετακίνηση σε m, χρόνος σε s.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Structure:
    """Ο κύριος φορέας (SDOF)."""

    m: float = 1.0      # Μάζα κατασκευής (tn)
    T: float = 0.49     # Ιδιοπερίοδος (s)
    ξ: float = 0.02     # Λόγος απόσβεσης

    @property
    def ω(self) -> float:
        """Κυκλική ιδιοσυχνότητα (rad/s)."""
        return 2.0 * np.pi / self.T

    @property
    def k(self) -> float:
        """Δυσκαμψία (kN/m)."""
        return self.ω**2 * self.m

    @property
    def c(self) -> float:
        """Συντελεστής ιξώδους απόσβεσης (kN·s/m)."""
        return 2.0 * self.ξ * self.ω * self.m


@dataclass(frozen=True)
class TMD:
    """Ο αποσβεστήρας συντονισμένης μάζας, όπως προκύπτει από τον σχεδιασμό."""

    m_hat: float    # Λόγος μαζών md / m
    md: float       # Μάζα TMD (tn)
    f: float        # Λόγος συχνοτήτων ωd / ω
    ξd: float       # Λόγος απόσβεσης του TMD
    ωd: float       # Κυκλική ιδιοσυχνότητα TMD (rad/s)
    kd: float       # Δυσκαμψία ελατηρίου (kN/m)
    cd: float       # Συντελεστής απόσβεσης (kN·s/m)


def tsai_lin_f(m_hat, ξ):
    """Βέλτιστος λόγος συχνοτήτων f = ωd/ω (Tsai & Lin, 1993)."""
    return (
        ((np.sqrt(1 - 0.5 * m_hat) / (1 + m_hat)) + np.sqrt(1 - 2 * ξ**2) - 1)
        - ((2.375 - 1.034 * np.sqrt(m_hat) - 0.426 * m_hat) * ξ * np.sqrt(m_hat))
        - ((3.730 - 16.903 * np.sqrt(m_hat) + 20.496 * m_hat) * (ξ**2) * np.sqrt(m_hat))
    )


def tsai_lin_xi(m_hat, ξ):
    """Βέλτιστος λόγος απόσβεσης ξd του TMD (Tsai & Lin, 1993)."""
    return (
        np.sqrt((3 * m_hat) / (8 * (1 + m_hat) * (1 - 0.5 * m_hat)))
        + (0.151 * ξ - 0.170 * ξ**2)
        + (0.163 * ξ + 4.980 * ξ**2) * m_hat
    )


def design_tmd(structure: Structure, m_hat: float, f=None, ξd=None) -> TMD:
    """
    Σχεδιασμός TMD για δεδομένο λόγο μαζών.

    Αν δεν δοθούν τα f και ξd, χρησιμοποιούνται οι βέλτιστες τιμές των
    Tsai & Lin. Δίνοντάς τα ρητά μπορούμε να μελετήσουμε την ευαισθησία
    σε αποσυντονισμό (detuning) ή σε μη βέλτιστη απόσβεση.
    """
    if f is None:
        f = float(tsai_lin_f(m_hat, structure.ξ))
    if ξd is None:
        ξd = float(tsai_lin_xi(m_hat, structure.ξ))

    md = m_hat * structure.m
    ωd = f * structure.ω

    return TMD(
        m_hat=m_hat,
        md=md,
        f=f,
        ξd=ξd,
        ωd=ωd,
        kd=md * ωd**2,
        cd=2.0 * ξd * ωd * md,
    )


def describe(structure: Structure, tmd: TMD | None = None) -> str:
    """Σύντομη περιγραφή των παραμέτρων, για εκτύπωση."""
    lines = [
        (
            f"Κατασκευή : m = {structure.m:.3f} tn, T = {structure.T:.3f} s, "
            f"ω = {structure.ω:.2f} rad/s, ξ = {structure.ξ:.3f}, "
            f"k = {structure.k:.2f} kN/m, c = {structure.c:.3f} kN·s/m"
        )
    ]
    if tmd is not None:
        lines.append(
            f"TMD       : m_hat = {tmd.m_hat:.4f}, md = {tmd.md:.4f} tn, "
            f"f = {tmd.f:.3f}, ξd = {tmd.ξd:.3f}, "
            f"kd = {tmd.kd:.3f} kN/m, cd = {tmd.cd:.4f} kN·s/m"
        )
    return "\n".join(lines)
