"""
Κοινές βοηθητικές συναρτήσεις της εφαρμογής Streamlit.

Η φυσική (μοντέλο OpenSees, σχέσεις Tsai & Lin, μεγέθη απόκρισης) ΔΕΝ
ξαναγράφεται εδώ: χρησιμοποιούνται αυτούσια τα modules του φακέλου
`tmd_parametric/`. Εδώ υπάρχουν μόνο η κατάσταση της συνεδρίας, η ανάγνωση
αρχείων που ανεβάζει ο χρήστης και οι cached κλήσεις της ανάλυσης.
"""

import io
import re
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PARAM_DIR = Path(__file__).resolve().parent.parent / "tmd_parametric"
sys.path.insert(0, str(PARAM_DIR))

# The modules are added to sys.path at runtime; static analyzers don't resolve
# that dynamic import path unless we silence the missing-import warning.
import excitation as exc  # type: ignore[reportMissingImports]  # noqa: E402
import results  # type: ignore[reportMissingImports]  # noqa: E402
from analysis import run_analysis  # type: ignore[reportMissingImports]  # noqa: E402
from structure import Structure, design_tmd, tsai_lin_f, tsai_lin_xi  # type: ignore[reportMissingImports]  # noqa: E402

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

# Γραμμή NPTS/DT των αρχείων PEER NGA .AT2, στις δύο μορφές που κυκλοφορούν:
#   "NPTS=   2000, DT=   .0050 SEC"   (NGA-West2, επικεφαλίδα στην αρχή)
#   "2000    0.0050    NPTS, DT"      (παλαιότερα NGA, επικεφαλίδα στην αρχή ή στο τέλος)
# Ίδια λογική με το Accelerogram.ParsePeerAt2 της βιβλιοθήκης STREX (C#).
_PEER_NPTS_DT = re.compile(r"NPTS\s*=\s*(\d+)\s*,\s*DT\s*=\s*([\d.Ee+\-]+)", re.IGNORECASE)
_PEER_NPTS_DT_POS = re.compile(r"^(\d+)\s+([\d.Ee+\-]+)\s+NPTS\s*,\s*DT\b", re.IGNORECASE)


def _is_numeric_line(line: str) -> bool:
    parts = line.split()
    if not parts:
        return False
    try:
        for p in parts:
            float(p)
    except ValueError:
        return False
    return True


def parse_peer_at2(text: str):
    """
    Αρχείο PEER NGA .AT2 (τιμές σε g).

    Η επικεφαλίδα 4 γραμμών (τίτλος βάσης· σεισμός, σταθμός, συνιστώσα· μονάδες·
    NPTS και DT) είναι συνήθως στην αρχή, αλλά σε κάποιες συλλογές (π.χ. aug26/GM_X)
    είναι στο τέλος. Διαβάζονται και οι δύο μορφές· τα dt και NPTS από την επικεφαλίδα.

    Επιστρέφει (dt, a).
    """
    lines = [line.strip() for line in text.splitlines()]

    npts_line, npts, dt = -1, 0, 0.0
    for i, line in enumerate(lines):
        m = _PEER_NPTS_DT.search(line) or _PEER_NPTS_DT_POS.match(line)
        if m:
            npts_line, npts, dt = i, int(m.group(1)), float(m.group(2))
            break
    if npts_line < 0:
        raise ValueError("Αρχείο PEER .AT2: δεν βρέθηκε η γραμμή «NPTS=n, DT=t SEC» "
                         "(ή «n t NPTS, DT») της επικεφαλίδας.")
    if dt <= 0:
        raise ValueError("Αρχείο PEER .AT2: το DT της επικεφαλίδας δεν είναι θετικός αριθμός.")

    # Η επικεφαλίδα είναι οι γραμμές κειμένου που τελειώνουν στη γραμμή NPTS
    header_start = npts_line
    while (header_start > 0 and lines[header_start - 1]
           and not _is_numeric_line(lines[header_start - 1])):
        header_start -= 1

    # Δεδομένα μετά την επικεφαλίδα (συνήθως) ή, αν δεν υπάρχουν, πριν από αυτήν (στο τέλος)
    data = [line for line in lines[npts_line + 1:] if _is_numeric_line(line)]
    if not data:
        data = [line for line in lines[:header_start] if _is_numeric_line(line)]

    a = np.array([float(v) for line in data for v in line.split()])
    if len(a) < npts:
        raise ValueError(f"Αρχείο PEER .AT2: η επικεφαλίδα δίνει NPTS = {npts}, "
                         f"αλλά το αρχείο έχει μόνο {len(a)} τιμές.")
    return dt, a[:npts]  # τυχόν συμπληρωματικά μηδενικά μετά το τελευταίο σημείο


def parse_accelerogram(raw: bytes, skiprows: int, layout: str, dt: float | None):
    """
    Ανάγνωση αρχείου κειμένου με επιταχύνσεις.

    layout:
      "PEER .at2": αρχείο PEER NGA, dt και NPTS από την επικεφαλίδα (αγνοούνται
                   τα skiprows και dt)
      "t, a"    : δύο στήλες [χρόνος, επιτάχυνση], το dt βγαίνει από τα δεδομένα
      "a"       : μία στήλη επιταχύνσεων, απαιτείται dt
      "πολλές"  : πολλές τιμές ανά γραμμή, διαβάζονται κατά γραμμή, απαιτείται dt

    Επιστρέφει (dt, a) με το a στις μονάδες του αρχείου (χωρίς μετατροπή).
    """
    if layout == "PEER .at2":
        return parse_peer_at2(raw.decode("utf-8", errors="replace"))

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
