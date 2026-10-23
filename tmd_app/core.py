"""
Κοινές βοηθητικές συναρτήσεις της εφαρμογής Streamlit.

Η φυσική (μοντέλο OpenSees, σχέσεις Tsai & Lin, μεγέθη απόκρισης) ΔΕΝ
ξαναγράφεται εδώ: χρησιμοποιούνται αυτούσια τα modules του φακέλου
`tmd_parametric/`. Εδώ υπάρχουν μόνο η κατάσταση της συνεδρίας, η ανάγνωση
αρχείων που ανεβάζει ο χρήστης και οι cached κλήσεις της ανάλυσης.
"""

import io
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PARAM_DIR = Path(__file__).resolve().parent.parent / "tmd_parametric"
sys.path.insert(0, str(PARAM_DIR))

import excitation as exc  # noqa: E402
import results  # noqa: E402
from analysis import run_analysis  # noqa: E402
from structure import Structure, design_tmd, tsai_lin_f, tsai_lin_xi  # noqa: E402

# Συντελεστές μετατροπής σε m/s^2
UNITS = {
    "g": exc.G,
    "m/s²": 1.0,
    "cm/s² (gal)": 0.01,
}

# Το OpenSees κρατά ένα καθολικό μοντέλο ανά διεργασία: δύο ταυτόχρονες
# αναλύσεις (π.χ. δύο καρτέλες του browser) θα το χαλούσαν.
_OPS_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Κατάσταση συνεδρίας
# --------------------------------------------------------------------------

DEFAULTS = {
    # Κατασκευή
    "m": 1.0,
    "T": 0.49,
    "xi": 0.02,
    # TMD
    "m_hat": 0.01,
    "tmd_optimum": True,
    "f_manual": 0.98,
    "xid_manual": 0.06,
}


def init_state() -> None:
    """Αρχικές τιμές - καλείται από το streamlit_app.py πριν από κάθε σελίδα."""
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    # Η διέγερση αποθηκεύεται ως «συνταγή», ώστε η αρμονική να ξαναπαράγεται
    # με το τρέχον ω της κατασκευής.
    st.session_state.setdefault("excitation_spec", {"kind": "elcentro"})


def current_structure() -> Structure:
    s = st.session_state
    return Structure(m=float(s.m), T=float(s.T), ξ=float(s.xi))


def current_tmd(structure: Structure, m_hat: float | None = None):
    """TMD με βέλτιστες τιμές Tsai & Lin ή με τις τιμές που έδωσε ο χρήστης."""
    s = st.session_state
    m_hat = float(s.m_hat) if m_hat is None else m_hat
    if s.tmd_optimum:
        return design_tmd(structure, m_hat)
    return design_tmd(structure, m_hat, f=float(s.f_manual), ξd=float(s.xid_manual))


def current_excitation(structure: Structure) -> exc.Excitation:
    spec = st.session_state.excitation_spec
    if spec["kind"] == "elcentro":
        return exc.elcentro()
    if spec["kind"] == "harmonic":
        return exc.harmonic(structure, rho=spec["rho"], dt=spec["dt"], duration=spec["duration"])
    return exc.Excitation(name=spec["name"], dt=spec["dt"], a=np.asarray(spec["a"]))


# --------------------------------------------------------------------------
# Ανάγνωση αρχείου επιταχυνσιογραφήματος
# --------------------------------------------------------------------------

def parse_accelerogram(raw: bytes, skiprows: int, layout: str, dt: float | None):
    """
    Ανάγνωση αρχείου κειμένου με επιταχύνσεις.

    layout:
      "t, a"    : δύο στήλες [χρόνος, επιτάχυνση], το dt βγαίνει από τα δεδομένα
      "a"       : μία στήλη επιταχύνσεων, απαιτείται dt
      "πολλές"  : πολλές τιμές ανά γραμμή (π.χ. μορφή PEER .AT2), διαβάζονται
                  κατά γραμμή, απαιτείται dt

    Επιστρέφει (dt, a) με το a στις μονάδες του αρχείου (χωρίς μετατροπή).
    """
    text = raw.decode("utf-8", errors="replace").replace(",", " ")
    lines = text.splitlines()[skiprows:]

    if layout == "πολλές":
        values = []
        for line in lines:
            values.extend(float(v) for v in line.split())
        a = np.array(values)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # κενό αρχείο: ελέγχεται παρακάτω
            data = np.loadtxt(io.StringIO("\n".join(lines)), ndmin=2)
        if data.shape[0] < 2:
            raise ValueError("Το αρχείο έχει λιγότερα από δύο σημεία.")
        if layout == "t, a":
            if data.shape[1] < 2:
                raise ValueError("Το αρχείο έχει μία στήλη - επιλέξτε τη μορφή «a».")
            t = data[:, 0]
            steps = np.diff(t)
            dt = round(float(steps[0]), 6)
            if not np.allclose(steps, dt, rtol=1e-3, atol=1e-9):
                raise ValueError("Το χρονικό βήμα δεν είναι σταθερό.")
            a = data[:, 1]
        else:
            a = data[:, 0]

    if dt is None or dt <= 0:
        raise ValueError("Δώστε θετικό χρονικό βήμα dt.")
    if len(a) < 2:
        raise ValueError("Το αρχείο έχει λιγότερα από δύο σημεία.")
    return dt, a


# --------------------------------------------------------------------------
# Ανάλυση (cached)
# --------------------------------------------------------------------------

def _key(structure, excitation, tmd):
    """Απλοί, hashable τύποι για το st.cache_data."""
    s = (structure.m, structure.T, structure.ξ)
    t = None if tmd is None else (tmd.m_hat, tmd.f, tmd.ξd)
    return s, excitation.name, excitation.dt, excitation.a, t


@st.cache_data(max_entries=500, show_spinner=False)
def _run_cached(s, name, dt, a, t):
    structure = Structure(m=s[0], T=s[1], ξ=s[2])
    excitation = exc.Excitation(name=name, dt=dt, a=a)
    tmd = None if t is None else design_tmd(structure, t[0], f=t[1], ξd=t[2])
    with _OPS_LOCK:
        r = run_analysis(structure, excitation, tmd)
    return r


def analyse(structure, excitation, tmd=None):
    """Μία ανάλυση χρονοϊστορίας (cached ανά σύνολο παραμέτρων)."""
    return _run_cached(*_key(structure, excitation, tmd))


def history_frame(baseline, controlled) -> pd.DataFrame:
    """Χρονοϊστορίες σε πίνακα, για διαγράμματα και λήψη csv."""
    return pd.DataFrame({
        "t (s)": baseline.t,
        "a_g (m/s²)": baseline.excitation.a,
        "u χωρίς TMD (m)": baseline.u,
        "u με TMD (m)": controlled.u,
        "διαδρομή TMD (m)": controlled.stroke,
    })


def to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, float_format="%.6g").encode("utf-8-sig")


__all__ = [
    "UNITS", "init_state", "current_structure", "current_tmd", "current_excitation",
    "parse_accelerogram", "analyse", "history_frame", "to_csv",
    "results", "exc", "design_tmd", "tsai_lin_f", "tsai_lin_xi", "Structure",
]
