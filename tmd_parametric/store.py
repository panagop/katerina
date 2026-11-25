"""
Αποθήκευση αποτελεσμάτων αναλύσεων, ώστε να μην ξανατρέχουν.

Δομή (στον φάκελο `results_db/` της ρίζας του repo, εκτός git):

    tmd_results.sqlite      κατάλογος + όλα τα βαθμωτά μεγέθη (u_max, RMS, u_ss, διαδρομή)
    histories/<exc_id>.h5   προαιρετικά: χρονοϊστορίες u και διαδρομής TMD, ένα αρχείο
                            ανά διέγερση (float32, συμπιεσμένες)

Πίνακες της βάσης:

    excitations  μία γραμμή ανά διέγερση. Το exc_id είναι hash των dt και των τιμών της
                 επιτάχυνσης (m/s²): ίδιο όνομα με άλλη κλιμάκωση = άλλη διέγερση.
                 Κρατά και τις τιμές, ώστε μια αποθηκευμένη αναζήτηση να ξαναφορτώνεται.
    baselines    απόκριση χωρίς TMD, ανά (διέγερση, κατασκευή)
    runs         μία γραμμή ανά ανάλυση με TMD (διέγερση, κατασκευή, m̂, f, ξd)
    searches     μία γραμμή ανά αναζήτηση πλέγματος: τι ζητήθηκε και πότε

Όλες οι παράμετροι αποθηκεύονται στρογγυλεμένες σε 6 δεκαδικά (`key()`), ώστε το ίδιο
σημείο του πλέγματος να βρίσκεται πάντα. Το MODEL_VERSION αυξάνεται αν αλλάξει ποτέ το
μοντέλο ή η ολοκλήρωση, ώστε να μην ανακατεύονται παλιά και νέα αποτελέσματα.
"""

import hashlib
import sqlite3
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from excitation import G, Excitation
from results import Metrics
from structure import Structure

MODEL_VERSION = 1
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "results_db"
_DECIMALS = 6

_SCHEMA = """
CREATE TABLE IF NOT EXISTS excitations (
    exc_id   TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    source   TEXT,
    dt       REAL NOT NULL,
    npts     INTEGER NOT NULL,
    pga_g    REAL NOT NULL,
    scale    REAL NOT NULL DEFAULT 1.0,
    a        BLOB NOT NULL,          -- επιταχύνσεις σε m/s², float64
    created  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS baselines (
    exc_id   TEXT NOT NULL REFERENCES excitations(exc_id),
    m REAL NOT NULL, T REAL NOT NULL, xi REAL NOT NULL,
    model_version INTEGER NOT NULL,
    u_max REAL, u_rms REAL, u_ss REAL,
    hist_row INTEGER,
    PRIMARY KEY (exc_id, m, T, xi, model_version)
);
CREATE TABLE IF NOT EXISTS runs (
    exc_id   TEXT NOT NULL REFERENCES excitations(exc_id),
    m REAL NOT NULL, T REAL NOT NULL, xi REAL NOT NULL,
    m_hat REAL NOT NULL, f REAL NOT NULL, xd REAL NOT NULL,
    model_version INTEGER NOT NULL,
    u_max REAL, u_rms REAL, u_ss REAL, stroke_max REAL,
    hist_row INTEGER,
    PRIMARY KEY (exc_id, m, T, xi, m_hat, f, xd, model_version)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS searches (
    search_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created  TEXT NOT NULL,
    source   TEXT,                   -- 'app' ή 'batch'
    exc_id   TEXT NOT NULL REFERENCES excitations(exc_id),
    m REAL NOT NULL, T REAL NOT NULL, xi REAL NOT NULL,
    m_lo REAL, m_hi REAL, m_n INTEGER,
    f_lo REAL, f_hi REAL, f_n INTEGER,
    xd_lo REAL, xd_hi REAL, xd_n INTEGER,
    histories INTEGER NOT NULL,
    n_total INTEGER, n_computed INTEGER,
    model_version INTEGER NOT NULL,
    git_commit TEXT
);
"""


def key(x) -> float:
    """Στρογγύλευση μιας παραμέτρου για χρήση ως κλειδί."""
    return round(float(x), _DECIMALS)


def grid_values(lo, hi, n) -> np.ndarray:
    """np.linspace με τιμές στρογγυλεμένες όπως τα κλειδιά της βάσης."""
    return np.round(np.linspace(float(lo), float(hi), int(n)), _DECIMALS)


def excitation_id(excitation: Excitation) -> str:
    """Hash των dt και των τιμών (m/s²): ίδια διέγερση -> ίδιο exc_id."""
    h = hashlib.sha1()
    h.update(f"{excitation.dt:.9g}".encode())
    h.update(np.ascontiguousarray(excitation.a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def histories_available() -> bool:
    """Οι χρονοϊστορίες θέλουν το h5py (στο Streamlit Cloud δεν είναι εγκατεστημένο)."""
    try:
        import h5py  # noqa: F401
    except ImportError:
        return False
    return True


class ResultStore:
    """Η βάση αποτελεσμάτων. Ένας εγγραφέας τη φορά (κλείδωμα για τα threads του Streamlit)."""

    def __init__(self, folder=DEFAULT_DIR):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "tmd_results.sqlite"
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------------
    # Διεγέρσεις
    # ------------------------------------------------------------------

    def register_excitation(self, excitation: Excitation, source=None, scale=1.0) -> str:
        """Καταχωρεί τη διέγερση (αν δεν υπάρχει) και επιστρέφει το exc_id."""
        exc_id = excitation_id(excitation)
        a = np.ascontiguousarray(excitation.a, dtype=np.float64)
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO excitations VALUES (?,?,?,?,?,?,?,?,?)",
                (exc_id, excitation.name, None if source is None else str(source),
                 float(excitation.dt), len(a), float(np.max(np.abs(a)) / G), float(scale),
                 a.tobytes(), datetime.now().isoformat(timespec="seconds")),
            )
        return exc_id

    def excitation(self, exc_id: str) -> Excitation:
        """Ξαναφτιάχνει μια αποθηκευμένη διέγερση."""
        row = self._db.execute("SELECT name, dt, a FROM excitations WHERE exc_id = ?",
                               (exc_id,)).fetchone()
        if row is None:
            raise KeyError(f"Δεν υπάρχει διέγερση με exc_id = {exc_id}")
        name, dt, blob = row
        return Excitation(name=name, dt=dt, a=np.frombuffer(blob, dtype=np.float64).copy())

    def excitations(self) -> pd.DataFrame:
        return pd.read_sql("SELECT exc_id, name, source, dt, npts, pga_g, scale, created "
                           "FROM excitations ORDER BY created", self._db)

    # ------------------------------------------------------------------
    # Απόκριση χωρίς TMD
    # ------------------------------------------------------------------

    def baseline(self, exc_id: str, structure: Structure) -> Metrics | None:
        row = self._db.execute(
            "SELECT u_max, u_rms, u_ss FROM baselines WHERE exc_id=? AND m=? AND T=? AND xi=? "
            "AND model_version=?", (exc_id, *_skey(structure), MODEL_VERSION)).fetchone()
        return None if row is None else Metrics(u_max=row[0], u_rms=row[1], u_ss=row[2],
                                                stroke_max=0.0)

    def save_baseline(self, exc_id, structure, metrics: Metrics, history=None, dt=None) -> None:
        """`history` = (u, stroke) για αποθήκευση και της χρονοϊστορίας."""
        with self._lock:
            hist_row = None
            if history is not None:
                hist_row = self._append_histories(exc_id, [history], dt)[0]
            with self._db:
                self._db.execute(
                    "INSERT OR REPLACE INTO baselines VALUES (?,?,?,?,?,?,?,?,?)",
                    (exc_id, *_skey(structure), MODEL_VERSION,
                     metrics.u_max, metrics.u_rms, metrics.u_ss, hist_row))

    # ------------------------------------------------------------------
    # Αναλύσεις με TMD
    # ------------------------------------------------------------------

    def runs(self, exc_id: str, structure: Structure) -> pd.DataFrame:
        """Όλες οι αποθηκευμένες αναλύσεις για μια διέγερση και κατασκευή."""
        return pd.read_sql(
            "SELECT m_hat, f, xd, u_max, u_rms, u_ss, stroke_max, hist_row FROM runs "
            "WHERE exc_id=? AND m=? AND T=? AND xi=? AND model_version=?",
            self._db, params=(exc_id, *_skey(structure), MODEL_VERSION))

    def save_runs(self, exc_id, structure, rows, histories=None, dt=None) -> None:
        """
        rows: λίστα από (m_hat, f, ξd, Metrics).
        histories: None ή λίστα από (u, stroke), στην ίδια σειρά με τα rows.
        Η χρονοϊστορία γράφεται πρώτα: αν διακοπεί η εγγραφή, στη βάση δεν μένει
        γραμμή που δείχνει σε ανύπαρκτη χρονοϊστορία.
        """
        if not rows:
            return
        with self._lock:
            hist_rows = (self._append_histories(exc_id, histories, dt) if histories
                         else [None] * len(rows))
            s = _skey(structure)
            with self._db:
                self._db.executemany(
                    "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(exc_id, *s, key(mh), key(f), key(xd), MODEL_VERSION,
                      mt.u_max, mt.u_rms, mt.u_ss, mt.stroke_max, hr)
                     for (mh, f, xd, mt), hr in zip(rows, hist_rows)])

    # ------------------------------------------------------------------
    # Χρονοϊστορίες (HDF5)
    # ------------------------------------------------------------------

    def _h5_path(self, exc_id: str) -> Path:
        return self.folder / "histories" / f"{exc_id}.h5"

    def _append_histories(self, exc_id, histories, dt) -> list[int]:
        import h5py

        path = self._h5_path(exc_id)
        npts = len(histories[0][0])
        if not path.exists():
            # Δημιουργία σε προσωρινό όνομα και μετονομασία: ποτέ μισό αρχείο
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".h5.tmp")
            with h5py.File(tmp, "w") as h5:
                h5.attrs.update(exc_id=exc_id, dt=float(dt), model_version=MODEL_VERSION)
                for name in ("u", "stroke"):
                    h5.create_dataset(name, shape=(0, npts), maxshape=(None, npts),
                                      dtype="float32", chunks=(1, npts),
                                      compression="gzip", compression_opts=4, shuffle=True)
            tmp.replace(path)
        with h5py.File(path, "a") as h5:
            start = h5["u"].shape[0]
            n = len(histories)
            for i, name in enumerate(("u", "stroke")):
                ds = h5[name]
                ds.resize(start + n, axis=0)
                ds[start:start + n] = np.array([h[i] for h in histories], dtype=np.float32)
        return list(range(start, start + n))

    def history(self, exc_id: str, hist_row: int):
        """(t, u, stroke) μίας αποθηκευμένης χρονοϊστορίας."""
        import h5py

        with h5py.File(self._h5_path(exc_id), "r") as h5:
            u = h5["u"][hist_row].astype(float)
            stroke = h5["stroke"][hist_row].astype(float)
            t = np.arange(len(u)) * float(h5.attrs["dt"])
        return t, u, stroke

    def find_history(self, exc_id, structure, m_hat=None, f=None, ξd=None):
        """(t, u, stroke) αν η χρονοϊστορία είναι αποθηκευμένη, αλλιώς None.
        Χωρίς m_hat: η απόκριση χωρίς TMD."""
        if m_hat is None:
            row = self._db.execute(
                "SELECT hist_row FROM baselines WHERE exc_id=? AND m=? AND T=? AND xi=? "
                "AND model_version=?", (exc_id, *_skey(structure), MODEL_VERSION)).fetchone()
        else:
            row = self._db.execute(
                "SELECT hist_row FROM runs WHERE exc_id=? AND m=? AND T=? AND xi=? AND m_hat=? "
                "AND f=? AND xd=? AND model_version=?",
                (exc_id, *_skey(structure), key(m_hat), key(f), key(ξd),
                 MODEL_VERSION)).fetchone()
        if row is None or row[0] is None or not histories_available():
            return None
        return self.history(exc_id, row[0])

    # ------------------------------------------------------------------
    # Αναζητήσεις
    # ------------------------------------------------------------------

    def log_search(self, exc_id, structure, m_vals, f_vals, x_vals, histories,
                   n_total, n_computed, source) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO searches (created, source, exc_id, m, T, xi, m_lo, m_hi, m_n, "
                "f_lo, f_hi, f_n, xd_lo, xd_hi, xd_n, histories, n_total, n_computed, "
                "model_version, git_commit) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), source, exc_id,
                 *_skey(structure), *_range(m_vals), *_range(f_vals), *_range(x_vals),
                 int(bool(histories)), int(n_total), int(n_computed), MODEL_VERSION,
                 _git_commit()))

    def searches(self) -> pd.DataFrame:
        """Οι αναζητήσεις που έχουν γίνει, οι πιο πρόσφατες πρώτα, με το όνομα της διέγερσης."""
        return pd.read_sql(
            "SELECT s.*, e.name AS excitation, e.scale, e.pga_g FROM searches s "
            "JOIN excitations e USING (exc_id) ORDER BY s.search_id DESC", self._db)


def _skey(structure: Structure) -> tuple[float, float, float]:
    return key(structure.m), key(structure.T), key(structure.ξ)


def _range(vals) -> tuple[float, float, int]:
    vals = np.asarray(vals)
    return float(vals.min()), float(vals.max()), int(len(vals))
