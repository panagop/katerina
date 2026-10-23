"""
Παραμετρικές αναλύσεις.

Τρεις σαρώσεις:
  1. sweep_mass_ratio : λόγος μαζών m_hat, για σεισμό και αρμονική διέγερση
                        -> πόσο μεγάλο πρέπει να είναι το TMD;
  2. sweep_detuning   : αποσυντονισμός f / f_opt σε σταθερό m_hat
                        -> πόσο ευαίσθητο είναι το TMD σε λάθος ρύθμιση;
  3. sweep_frequency  : λόγος συχνοτήτων διέγερσης ρ = Ω/ω
                        -> η καμπύλη απόκρισης συχνότητας, με και χωρίς TMD.

Όλα τα αποτελέσματα αποθηκεύονται σε csv (φάκελος results) και σε png
(φάκελος figures).

    python run_parametric.py            # και οι τρεις σαρώσεις
    python run_parametric.py --sweep mass
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Χωρίς --show δεν ανοίγουν παράθυρα: επιλογή backend ΠΡΙΝ φορτωθεί το pyplot
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import excitation as exc
import plots
import results
from analysis import run_analysis, run_pair
from structure import Structure, design_tmd, tsai_lin_f

HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"


def sweep_mass_ratio(structure, m_hats, harmonic_duration=60.0, show=False):
    """Μείωση απόκρισης και απαιτούμενη διαδρομή TMD, συναρτήσει του m_hat."""
    print("\n=== ΣΑΡΩΣΗ 1: λόγος μαζών m_hat ===")

    grounds = [
        exc.elcentro(),
        exc.harmonic(structure, rho=1.0, duration=harmonic_duration),
    ]

    rows = []
    for ground in grounds:
        # Η ανάλυση χωρίς TMD δεν εξαρτάται από το m_hat - τρέχει μία φορά
        baseline = run_analysis(structure, ground)
        for m_hat in m_hats:
            tmd = design_tmd(structure, m_hat)
            controlled = run_analysis(structure, ground, tmd)
            row = results.compare(baseline, controlled)
            rows.append(row)
            print(f"  {ground.name:>16} | m_hat = {m_hat:.3f} | "
                  f"μείωση u_max = {row['μείωση_u_max_%']:6.2f} % | "
                  f"διαδρομή = {row['διαδρομή_TMD']:.3f} m")

    df = results.to_dataframe(rows)
    results.save_table(df, RESULT_DIR / "sweep_mass_ratio.csv")

    plots.plot_sweep(
        df, x="m_hat", ys=["μείωση_u_max_%"], group="διέγερση",
        xlabel="Λόγος μαζών m_hat", ylabel="Μείωση μέγιστης μετακίνησης (%)",
        title="Επίδραση του λόγου μαζών στην αποτελεσματικότητα του TMD",
        save="sweep_mass_reduction.png", show=show,
    )
    plots.plot_sweep(
        df, x="m_hat", ys=["διαδρομή_TMD"], group="διέγερση",
        xlabel="Λόγος μαζών m_hat", ylabel="Μέγιστη σχετική διαδρομή TMD (m)",
        title="Απαιτούμενη διαδρομή της μάζας του TMD",
        save="sweep_mass_stroke.png", show=show,
    )
    return df


def sweep_detuning(structure, m_hat=0.05, ratios=None, harmonic_duration=60.0, show=False):
    """Ευαισθησία σε αποσυντονισμό: f = ratio · f_opt."""
    print(f"\n=== ΣΑΡΩΣΗ 2: αποσυντονισμός (m_hat = {m_hat}) ===")
    if ratios is None:
        ratios = np.arange(0.80, 1.201, 0.02)

    f_opt = float(tsai_lin_f(m_hat, structure.ξ))
    grounds = [
        exc.elcentro(),
        exc.harmonic(structure, rho=1.0, duration=harmonic_duration),
    ]

    rows = []
    for ground in grounds:
        baseline = run_analysis(structure, ground)
        for ratio in ratios:
            tmd = design_tmd(structure, m_hat, f=ratio * f_opt)
            controlled = run_analysis(structure, ground, tmd)
            row = results.compare(baseline, controlled)
            row["f/f_opt"] = float(ratio)
            rows.append(row)
        print(f"  {ground.name}: ολοκληρώθηκαν {len(ratios)} αναλύσεις")

    df = results.to_dataframe(rows)
    results.save_table(df, RESULT_DIR / "sweep_detuning.csv")

    plots.plot_sweep(
        df, x="f/f_opt", ys=["μείωση_u_max_%"], group="διέγερση",
        xlabel="f / f_opt (αποσυντονισμός)",
        ylabel="Μείωση μέγιστης μετακίνησης (%)",
        title=f"Ευαισθησία του TMD σε αποσυντονισμό (m_hat = {m_hat})",
        save="sweep_detuning.png", show=show,
    )
    return df


def sweep_frequency(structure, m_hat=0.05, rhos=None, duration=60.0, show=False):
    """
    Καμπύλη απόκρισης συχνότητας: πλάτος μόνιμης κατάστασης για αρμονική
    διέγερση σταθερού πλάτους, καθώς μεταβάλλεται ο λόγος ρ = Ω/ω.
    """
    print(f"\n=== ΣΑΡΩΣΗ 3: συχνότητα διέγερσης (m_hat = {m_hat}) ===")
    if rhos is None:
        rhos = np.arange(0.70, 1.301, 0.01)

    tmd = design_tmd(structure, m_hat)
    # Σταθερό πλάτος διέγερσης (ανεξάρτητο του ρ), ώστε οι καμπύλες να συγκρίνονται
    amplitude = 2.0 * structure.ξ * structure.k / structure.m

    rows = []
    for rho in rhos:
        ground = exc.harmonic(structure, rho=rho, amplitude=amplitude, duration=duration)
        baseline, controlled = run_pair(structure, ground, tmd)
        m_base = results.compute_metrics(baseline)
        m_ctrl = results.compute_metrics(controlled)
        rows.append({
            "ρ": float(rho),
            "u_ss_χωρίς": m_base.u_ss,
            "u_ss_με": m_ctrl.u_ss,
            "μείωση_u_ss_%": results.reduction(m_base.u_ss, m_ctrl.u_ss),
            "διαδρομή_TMD": m_ctrl.stroke_max,
        })
    print(f"  ολοκληρώθηκαν {len(rows)} αναλύσεις")

    df = results.to_dataframe(rows)
    results.save_table(df, RESULT_DIR / "sweep_frequency.csv")

    plots.plot_sweep(
        df, x="ρ", ys=["u_ss_χωρίς", "u_ss_με"],
        xlabel="Λόγος συχνοτήτων ρ = Ω / ω",
        ylabel="Πλάτος μόνιμης κατάστασης (m)",
        title=f"Καμπύλη απόκρισης συχνότητας (m_hat = {m_hat})",
        save="sweep_frequency.png", show=show,
    )
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Παραμετρικές αναλύσεις TMD")
    p.add_argument("--sweep", choices=["mass", "detuning", "frequency", "all"], default="all")
    p.add_argument("--m-hat", type=float, default=0.05,
                   help="Λόγος μαζών για τις σαρώσεις 2 και 3")
    p.add_argument("--T", type=float, default=0.49)
    p.add_argument("--xi", type=float, default=0.02)
    p.add_argument("--duration", type=float, default=60.0,
                   help="Διάρκεια αρμονικής διέγερσης (s)")
    p.add_argument("--show", action="store_true",
                   help="Εμφάνιση των διαγραμμάτων (εξ ορισμού μόνο αποθήκευση)")
    return p.parse_args()


def main():
    args = parse_args()
    structure = Structure(m=1.0, T=args.T, ξ=args.xi)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if args.sweep in ("mass", "all"):
        sweep_mass_ratio(
            structure,
            m_hats=np.arange(0.005, 0.1001, 0.005),
            harmonic_duration=args.duration,
            show=args.show,
        )
    if args.sweep in ("detuning", "all"):
        sweep_detuning(structure, m_hat=args.m_hat,
                       harmonic_duration=args.duration, show=args.show)
    if args.sweep in ("frequency", "all"):
        sweep_frequency(structure, m_hat=args.m_hat,
                        duration=args.duration, show=args.show)


if __name__ == "__main__":
    main()
