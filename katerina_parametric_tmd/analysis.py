"""
Στήσιμο και εκτέλεση του μοντέλου στο OpenSees.

Το μοντέλο είναι 1-D (ndm=1, ndf=1):
    κόμβος 1: πακτωμένη βάση
    κόμβος 2: μάζα κατασκευής, συνδεδεμένη στη βάση με zeroLength
              (Elastic k παράλληλα με Viscous c)
    κόμβος 3: μάζα TMD (μόνο όταν υπάρχει TMD), συνδεδεμένη στον κόμβο 2
              με zeroLength (Elastic kd παράλληλα με Viscous cd)

Η απόσβεση δίνεται απευθείας από τα υλικά Viscous, οπότε ΔΕΝ χρησιμοποιείται
απόσβεση Rayleigh.
"""

from dataclasses import dataclass, field

import numpy as np
import openseespy.opensees as ops

from excitation import Excitation
from structure import Structure, TMD


@dataclass(frozen=True)
class Response:
    """Το αποτέλεσμα μίας ανάλυσης χρονοϊστορίας."""

    label: str
    t: np.ndarray           # Χρόνος (s)
    u: np.ndarray           # Μετακίνηση κατασκευής, κόμβος 2 (m)
    stroke: np.ndarray      # Σχετική διαδρομή TMD, κόμβος 3 - κόμβος 2 (m)
    structure: Structure
    excitation: Excitation
    tmd: TMD | None = field(default=None)

    @property
    def has_tmd(self) -> bool:
        return self.tmd is not None


def build_model(structure: Structure, excitation: Excitation, tmd: TMD | None = None) -> None:
    """Καθαρίζει το OpenSees και χτίζει το μοντέλο με τη διέγερση."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    # Βάση και μάζα κατασκευής
    ops.node(1, 0.0)
    ops.node(2, 0.0)
    ops.fix(1, 1)
    ops.mass(2, structure.m)

    # Ελατήριο + αποσβεστήρας κατασκευής σε παράλληλη σύνδεση
    ops.uniaxialMaterial("Elastic", 1, structure.k)
    ops.uniaxialMaterial("Viscous", 2, structure.c, 1)
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, 2, "-dir", 1, 1)

    # Ο αποσβεστήρας συντονισμένης μάζας
    if tmd is not None:
        ops.node(3, 0.0)
        ops.mass(3, tmd.md)
        ops.uniaxialMaterial("Elastic", 3, tmd.kd)
        ops.uniaxialMaterial("Viscous", 4, tmd.cd, 1)
        ops.element("zeroLength", 2, 2, 3, "-mat", 3, 4, "-dir", 1, 1)

    # Διέγερση βάσης: οι τιμές είναι ήδη σε m/s^2 (βλ. excitation.py)
    ops.timeSeries("Path", 1, "-dt", excitation.dt, "-values", *excitation.a.tolist())
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

    # Ρυθμίσεις ανάλυσης - Newmark μέσης επιτάχυνσης (άνευ όρων ευσταθής)
    ops.system("ProfileSPD")
    ops.numberer("Plain")
    ops.constraints("Transformation")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.algorithm("Linear")
    ops.analysis("Transient")


def run_analysis(
    structure: Structure,
    excitation: Excitation,
    tmd: TMD | None = None,
    label: str | None = None,
) -> Response:
    """
    Εκτελεί την ανάλυση χρονοϊστορίας και επιστρέφει ένα Response.

    Το πλήθος των βημάτων είναι σταθερό (excitation.n_steps), ώστε ο χρόνος να
    μην συσσωρεύει σφάλμα στρογγύλευσης και η ανάλυση να σταματά ακριβώς στο
    τέλος της διέγερσης.
    """
    build_model(structure, excitation, tmd)

    # Ξεκινάμε από t = 0 με μηδενικές αρχικές συνθήκες
    disp = [0.0]
    stroke = [0.0]

    for i in range(excitation.n_steps):
        if ops.analyze(1, excitation.dt) != 0:
            raise RuntimeError(
                f"Η ανάλυση απέτυχε στο βήμα {i + 1} "
                f"(t = {(i + 1) * excitation.dt:.3f} s)"
            )
        disp.append(ops.nodeDisp(2, 1))
        stroke.append(ops.nodeDisp(3, 1) - ops.nodeDisp(2, 1) if tmd is not None else 0.0)

    if label is None:
        label = "με TMD" if tmd is not None else "χωρίς TMD"

    return Response(
        label=label,
        t=np.arange(len(disp)) * excitation.dt,
        u=np.array(disp),
        stroke=np.array(stroke),
        structure=structure,
        excitation=excitation,
        tmd=tmd,
    )


def run_pair(structure: Structure, excitation: Excitation, tmd: TMD) -> tuple[Response, Response]:
    """Τρέχει το ίδιο πρόβλημα χωρίς και με TMD - η τυπική σύγκριση."""
    baseline = run_analysis(structure, excitation, tmd=None)
    controlled = run_analysis(structure, excitation, tmd=tmd)
    return baseline, controlled
