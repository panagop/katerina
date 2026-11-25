"""
Αναζήτηση βέλτιστου TMD (πλέγμα m̂ × f × ξd) για πολλές διεγέρσεις, με αποθήκευση.

Τα αποτελέσματα γράφονται στη βάση `results_db/` (δες store.py) και τα βλέπεις μετά
στην εφαρμογή, καρτέλα «Εύρεση Βέλτιστων» → «Αποθηκευμένες αναζητήσεις», ή με pandas.
Αν το διακόψεις (Ctrl+C) και το ξανατρέξεις, συνεχίζει από εκεί που έμεινε.

Παραδείγματα (από τον φάκελο tmd_parametric):

    uv run --project .. python -X utf8 run_optimum_batch.py
        όλα τα .at2 του aug26/GM_X, χωρίς κλιμάκωση, πλέγμα όπως στην εφαρμογή

    uv run --project .. python -X utf8 run_optimum_batch.py ../aug26/GM_X/RIO270.at2 --pga 0.36
        ένα αρχείο, κλιμακωμένο σε PGA 0.36 g

    uv run --project .. python -X utf8 run_optimum_batch.py elcentro --xi 0.05 --histories
        El Centro, ξ = 0.05, με αποθήκευση και των χρονοϊστοριών

Αρχεία: .at2 (PEER, τιμές σε g) ή δύο στήλες [χρόνος, επιτάχυνση σε g], όπως το
aug26/elCentro.txt. Η λέξη «elcentro» δίνει το El Centro της εφαρμογής.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import excitation as exc
from optimum import Eta, grid_search
from store import DEFAULT_DIR, ResultStore, grid_values
from structure import Structure

GM_DIR = exc.DATA_DIR / "GM_X"


def find_records(items):
    """Λίστα (όνομα, path ή None) από αρχεία, φακέλους ή τη λέξη «elcentro»."""
    records = []
    for item in items or [str(GM_DIR)]:
        if item.lower() == "elcentro":
            records.append(("El Centro", None))
            continue
        p = Path(item)
        if p.is_dir():
            records += [(f.stem, f) for f in sorted(p.glob("*.at2"))]
        elif p.is_file():
            records.append((p.stem, p))
        else:
            sys.exit(f"Δεν βρέθηκε: {item}")
    return records


def load(name, path, pga):
    """Η διέγερση σε m/s², κλιμακωμένη σε PGA (g) αν δοθεί."""
    if path is None:
        e = exc.elcentro()
    elif path.suffix.lower() == ".at2":
        e = exc.load_peer_at2(path)
    else:
        e = exc.load_from_file(path, name=name, factor=exc.G)
    scale = 1.0
    if pga:
        scale = pga * exc.G / float(abs(e.a).max())
        e = exc.Excitation(name=f"{e.name} (PGA {pga:g} g)", dt=e.dt, a=e.a * scale)
    return e, scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="*", help="αρχεία, φάκελοι ή «elcentro» (προεπιλογή: aug26/GM_X)")
    ap.add_argument("--pga", type=float, help="κλιμάκωση κάθε καταγραφής σε αυτό το PGA (g)")
    ap.add_argument("--m", type=float, default=1.0, help="μάζα κατασκευής (tn)")
    ap.add_argument("--T", type=float, default=0.49, help="ιδιοπερίοδος (s)")
    ap.add_argument("--xi", type=float, default=0.02, help="λόγος απόσβεσης κατασκευής")
    ap.add_argument("--m-range", type=float, nargs=3, default=[0.01, 0.10, 10],
                    metavar=("ΑΠΟ", "ΕΩΣ", "ΠΛΗΘΟΣ"), help="λόγος μαζών m̂")
    ap.add_argument("--f-range", type=float, nargs=3, default=[0.70, 1.20, 20],
                    metavar=("ΑΠΟ", "ΕΩΣ", "ΠΛΗΘΟΣ"), help="λόγος συχνοτήτων f")
    ap.add_argument("--xd-range", type=float, nargs=3, default=[0.01, 0.25, 30],
                    metavar=("ΑΠΟ", "ΕΩΣ", "ΠΛΗΘΟΣ"), help="απόσβεση TMD ξd")
    ap.add_argument("--histories", action="store_true",
                    help="αποθήκευση και των χρονοϊστοριών (πολύς χώρος για μεγάλες καταγραφές)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="παράλληλες διεργασίες (προεπιλογή: πυρήνες - 1)")
    ap.add_argument("--db", type=Path, default=DEFAULT_DIR, help="φάκελος της βάσης")
    args = ap.parse_args()

    structure = Structure(m=args.m, T=args.T, ξ=args.xi)
    m_vals, f_vals, x_vals = (grid_values(*r) for r in (args.m_range, args.f_range, args.xd_range))
    records = find_records(args.records)
    store = ResultStore(args.db)

    n_grid = len(m_vals) * len(f_vals) * len(x_vals)
    print(f"Κατασκευή: m = {args.m:g} tn, T = {args.T:g} s, ξ = {args.xi:g}")
    print(f"Πλέγμα: {len(m_vals)} × {len(f_vals)} × {len(x_vals)} = {n_grid} αναλύσεις ανά διέγερση")
    print(f"Διεγέρσεις: {len(records)}, διεργασίες: {args.workers}, "
          f"χρονοϊστορίες: {'ναι' if args.histories else 'όχι'}")
    print(f"Βάση: {store.path}\n")

    t_start = time.time()
    for i, (name, path) in enumerate(records, 1):
        e, scale = load(name, path, args.pga)
        eta = Eta()

        def progress(done, total):
            print(f"\r  {done}/{total} νέες αναλύσεις, υπόλοιπο ≈ {eta(done, total)}   ",
                  end="", flush=True)

        print(f"[{i}/{len(records)}] {e.name}: {len(e.a)} σημεία, dt = {e.dt:g} s")
        df = grid_search(structure, e, m_vals, f_vals, x_vals, store=store,
                         histories=args.histories, progress=progress, workers=args.workers,
                         source="batch", source_file=path, scale=scale)
        reused = df.attrs["n_total"] - df.attrs["n_computed"]
        print(f"\r  έτοιμο: {df.attrs['n_computed']} νέες, {reused} από τη βάση"
              f"{' ' * 30}")

    print(f"\nΣυνολικός χρόνος: {(time.time() - t_start) / 60:.1f} min")
    store.close()


if __name__ == "__main__":
    main()
