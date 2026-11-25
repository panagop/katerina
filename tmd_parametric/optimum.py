"""
Αναζήτηση πλέγματος m̂ × f × ξd για βέλτιστο TMD, με αποθήκευση των αποτελεσμάτων.

Τρέχουν μόνο τα σημεία του πλέγματος που δεν υπάρχουν ήδη στη βάση (store.py).
Επιστρέφεται ΟΛΟ το πλέγμα (όχι μόνο τα βέλτιστα), ώστε το κριτήριο (RMS, u_max,
διαδρομή), το εύρος του m̂ και η προσαρμογή εξισώσεων να γίνονται χωρίς νέες αναλύσεις.
"""

import itertools
import multiprocessing as mp
import time

import pandas as pd

from analysis import run_analysis
from excitation import Excitation
from results import Metrics, compare_metrics, compute_metrics
from store import ResultStore, excitation_id, key
from structure import Structure, design_tmd


def _run_point(structure, excitation, histories, point, lock=None):
    """Μία ανάλυση με TMD. Επιστρέφει (m_hat, f, ξd, Metrics, (u, stroke) ή None)."""
    m_hat, f, xd = point
    tmd = design_tmd(structure, m_hat, f=f, ξd=xd)
    r = _locked(lock, run_analysis, structure, excitation, tmd)
    return m_hat, f, xd, compute_metrics(r), ((r.u, r.stroke) if histories else None)


def _locked(lock, fn, *args):
    if lock is None:
        return fn(*args)
    with lock:
        return fn(*args)


# Κάθε διεργασία του Pool κρατά τη δική της κατασκευή/διέγερση (και το δικό της OpenSees)
_WORKER = {}


def _init_worker(structure, excitation, histories):
    _WORKER.update(structure=structure, excitation=excitation, histories=histories)


def _run_chunk(points):
    w = _WORKER
    return [_run_point(w["structure"], w["excitation"], w["histories"], p) for p in points]


def grid_search(structure: Structure, excitation: Excitation, m_vals, f_vals, x_vals,
                store: ResultStore | None = None, histories: bool = False, progress=None,
                workers: int = 1, chunk: int = 50, source: str = "app",
                source_file=None, scale: float = 1.0, log: bool = True,
                lock=None) -> pd.DataFrame:
    """
    Όλο το πλέγμα m_vals × f_vals × x_vals (σειρά: m̂, μετά f, μετά ξd).

    store      : αν δοθεί, ξαναχρησιμοποιούνται τα αποθηκευμένα σημεία και αποθηκεύονται
                 τα νέα (ανά `chunk` αναλύσεις, ώστε μια διακοπή να μη χάνει τη δουλειά).
    histories  : αποθήκευση και των χρονοϊστοριών των νέων αναλύσεων (θέλει store).
                 Σημεία που υπάρχουν ήδη χωρίς χρονοϊστορία δεν ξανατρέχουν.
    progress   : συνάρτηση progress(έγιναν, σύνολο_νέων) για μπάρα προόδου.
    workers    : >1 για παράλληλες αναλύσεις (μόνο σε script, όχι μέσα στο Streamlit).
    log        : καταγραφή της αναζήτησης στον πίνακα searches (False όταν απλώς
                 ξαναφορτώνεται μια αποθηκευμένη αναζήτηση).
    lock       : κλείδωμα γύρω από κάθε ανάλυση (το OpenSees έχει ένα μοντέλο ανά
                 διεργασία· η εφαρμογή Streamlit δίνει το δικό της).

    Στήλες όπως του results.compare (u_max_με, RMS_με, μείωση_u_max_% κ.λπ.).
    df.attrs: n_total, n_computed, exc_id.
    """
    m_vals = [key(v) for v in m_vals]
    f_vals = [key(v) for v in f_vals]
    x_vals = [key(v) for v in x_vals]
    points = list(itertools.product(m_vals, f_vals, x_vals))
    histories = bool(histories and store is not None)

    # --- Απόκριση χωρίς TMD
    exc_id = (store.register_excitation(excitation, source_file, scale) if store
              else excitation_id(excitation))
    base = store.baseline(exc_id, structure) if store else None
    if base is None:
        r = _locked(lock, run_analysis, structure, excitation)
        base = compute_metrics(r)
        if store:
            store.save_baseline(exc_id, structure, base,
                                (r.u, r.stroke) if histories else None, excitation.dt)

    # --- Τι υπάρχει ήδη
    found: dict[tuple, Metrics] = {}
    if store:
        stored = store.runs(exc_id, structure)
        for row in stored.itertuples(index=False):
            found[(row.m_hat, row.f, row.xd)] = Metrics(row.u_max, row.u_rms, row.u_ss,
                                                        row.stroke_max)
    missing = [p for p in points if p not in found]

    # --- Νέες αναλύσεις
    done = 0

    def collect(results):
        nonlocal done
        for mh, f, xd, mt, _ in results:
            found[(mh, f, xd)] = mt
        if store:
            hist = [h for *_, h in results] if histories else None
            store.save_runs(exc_id, structure, [r[:4] for r in results], hist, excitation.dt)
        done += len(results)
        if progress:
            progress(done, len(missing))

    chunks = [missing[i:i + chunk] for i in range(0, len(missing), chunk)]
    if workers > 1 and len(chunks) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, _init_worker, (structure, excitation, histories)) as pool:
            for results in pool.imap_unordered(_run_chunk, chunks):
                collect(results)
    else:
        for c in chunks:
            collect([_run_point(structure, excitation, histories, p, lock) for p in c])

    if store and log:
        store.log_search(exc_id, structure, m_vals, f_vals, x_vals, histories,
                         len(points), len(missing), source)

    df = pd.DataFrame([compare_metrics(excitation.name, mh, f, xd, base, found[(mh, f, xd)])
                       for mh, f, xd in points])
    df.attrs.update(n_total=len(points), n_computed=len(missing), exc_id=exc_id)
    return df


def count_stored(store: ResultStore, structure, excitation, m_vals, f_vals, x_vals) -> int:
    """Πόσα σημεία του πλέγματος υπάρχουν ήδη (χωρίς να καταχωρηθεί η διέγερση)."""
    stored = store.runs(excitation_id(excitation), structure)
    have = set(zip(stored.m_hat, stored.f, stored.xd))
    return sum(p in have for p in itertools.product([key(v) for v in m_vals],
                                                    [key(v) for v in f_vals],
                                                    [key(v) for v in x_vals]))


class Eta:
    """Εκτίμηση υπολοίπου χρόνου για το batch script."""

    def __init__(self):
        self.t0 = time.time()

    def __call__(self, done, total) -> str:
        if done == 0:
            return "?"
        left = (time.time() - self.t0) / done * (total - done)
        return f"{left / 60:.1f} min" if left >= 60 else f"{left:.0f} s"
