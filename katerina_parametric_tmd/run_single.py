"""
Μία μεμονωμένη ανάλυση: κατασκευή με και χωρίς TMD, για σεισμική ή αρμονική
διέγερση. Αντιστοιχεί στα αρχικά scripts TMD_gp.py / TMD_harm_excit_gp.py,
γραμμένα όμως πάνω στις συναρτήσεις του φακέλου.

Παραδείγματα:
    python run_single.py
    python run_single.py --excitation harmonic --m-hat 0.05
    python run_single.py --excitation seismic --m-hat 0.03 --no-show
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import excitation as exc
import plots
import results
from analysis import run_pair
from structure import Structure, design_tmd, describe


def parse_args():
    p = argparse.ArgumentParser(description="Ανάλυση SDOF με TMD")
    p.add_argument("--excitation", choices=["seismic", "harmonic"], default="seismic",
                   help="Είδος διέγερσης")
    p.add_argument("--m-hat", type=float, default=0.01, help="Λόγος μαζών md/m")
    p.add_argument("--T", type=float, default=0.49, help="Ιδιοπερίοδος κατασκευής (s)")
    p.add_argument("--xi", type=float, default=0.02, help="Απόσβεση κατασκευής")
    p.add_argument("--rho", type=float, default=1.0,
                   help="Λόγος συχνοτήτων Ω/ω της αρμονικής διέγερσης")
    p.add_argument("--duration", type=float, default=60.0,
                   help="Διάρκεια αρμονικής διέγερσης (s)")
    p.add_argument("--no-show", action="store_true", help="Μόνο αποθήκευση, χωρίς εμφάνιση")
    return p.parse_args()


def main():
    args = parse_args()
    show = not args.no_show

    structure = Structure(m=1.0, T=args.T, ξ=args.xi)

    if args.excitation == "seismic":
        ground = exc.elcentro()
        tag = "elcentro"
        zoom = None
    else:
        ground = exc.harmonic(structure, rho=args.rho, duration=args.duration)
        tag = f"harmonic_rho{args.rho:.2f}"
        zoom = (0, 5)  # ζουμ ώστε να φαίνεται το ημίτονο

    tmd = design_tmd(structure, args.m_hat)

    print(describe(structure, tmd))
    print(f"Διέγερση  : {ground.name}, dt = {ground.dt} s, "
          f"διάρκεια = {ground.t_final} s ({ground.n_steps} βήματα)")

    plots.plot_excitation(ground, xlim=zoom, save=f"{tag}_excitation.png", show=show)

    baseline, controlled = run_pair(structure, ground, tmd)
    results.print_report(baseline, controlled)

    plots.plot_responses(baseline, controlled,
                         save=f"{tag}_response.png", show=show)
    plots.plot_responses(baseline, controlled, normalize=True,
                         title=f"Κανονικοποιημένη απόκριση - {ground.name}",
                         save=f"{tag}_response_norm.png", show=show)
    plots.plot_stroke(controlled, save=f"{tag}_stroke.png", show=show)


if __name__ == "__main__":
    main()
