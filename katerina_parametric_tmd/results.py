"""
Επεξεργασία αποτελεσμάτων: μεγέθη απόκρισης, ποσοστά μείωσης, πίνακες.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from analysis import Response


@dataclass(frozen=True)
class Metrics:
    """Τα χαρακτηριστικά μεγέθη μίας απόκρισης."""

    u_max: float        # Μέγιστη απόλυτη μετακίνηση (m)
    u_rms: float        # Ενεργός τιμή RMS της μετακίνησης (m)
    u_ss: float         # Πλάτος μόνιμης κατάστασης (m)
    stroke_max: float   # Μέγιστη σχετική διαδρομή TMD (m)


def compute_metrics(response: Response, ss_fraction: float = 0.2) -> Metrics:
    """
    Υπολογισμός των μεγεθών απόκρισης.

    Το πλάτος μόνιμης κατάστασης λαμβάνεται από το τελευταίο `ss_fraction` της
    χρονοϊστορίας, όπου η μεταβατική απόκριση έχει σβήσει. Έχει νόημα κυρίως
    στην αρμονική διέγερση - στον σεισμό είναι απλώς πληροφοριακό.
    """
    i_ss = int((1.0 - ss_fraction) * len(response.u))
    return Metrics(
        u_max=float(np.max(np.abs(response.u))),
        u_rms=float(np.sqrt(np.mean(response.u**2))),
        u_ss=float(np.max(np.abs(response.u[i_ss:]))),
        stroke_max=float(np.max(np.abs(response.stroke))),
    )


def reduction(base: float, controlled: float) -> float:
    """Ποσοστιαία μείωση (%) του `controlled` ως προς το `base`."""
    return (base - controlled) / base * 100.0 if base != 0 else np.nan


def compare(baseline: Response, controlled: Response, ss_fraction: float = 0.2) -> dict:
    """
    Συγκρίνει την απόκριση χωρίς και με TMD και επιστρέφει ένα dict με όλα τα
    μεγέθη - κατάλληλο για γραμμή πίνακα σε παραμετρική ανάλυση.
    """
    base = compute_metrics(baseline, ss_fraction)
    ctrl = compute_metrics(controlled, ss_fraction)
    tmd = controlled.tmd

    row = {
        "διέγερση": controlled.excitation.name,
        "m_hat": tmd.m_hat if tmd else 0.0,
        "f": tmd.f if tmd else np.nan,
        "ξd": tmd.ξd if tmd else np.nan,
        "u_max_χωρίς": base.u_max,
        "u_max_με": ctrl.u_max,
        "μείωση_u_max_%": reduction(base.u_max, ctrl.u_max),
        "RMS_χωρίς": base.u_rms,
        "RMS_με": ctrl.u_rms,
        "μείωση_RMS_%": reduction(base.u_rms, ctrl.u_rms),
        "u_ss_χωρίς": base.u_ss,
        "u_ss_με": ctrl.u_ss,
        "μείωση_u_ss_%": reduction(base.u_ss, ctrl.u_ss),
        "διαδρομή_TMD": ctrl.stroke_max,
        "λόγος_διαδρομής": ctrl.stroke_max / ctrl.u_max if ctrl.u_max else np.nan,
    }
    return row


def print_report(baseline: Response, controlled: Response, ss_fraction: float = 0.2) -> None:
    """Εκτύπωση των αποτελεσμάτων μίας σύγκρισης, στη μορφή των αρχικών scripts."""
    r = compare(baseline, controlled, ss_fraction)

    print("\n ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑΛΥΣΗΣ - " + str(r["διέγερση"]))
    print(f"Μέγιστη Μετακίνηση χωρίς TMD    : {r['u_max_χωρίς']:.4f} m")
    print(f"Μέγιστη Μετακίνηση με TMD       : {r['u_max_με']:.4f} m")
    print(f"Ποσοστό Μείωσης Μεγίστου        : {r['μείωση_u_max_%']:.2f} %")
    print(f"Μείωση Συνολικής Ενέργειας (RMS): {r['μείωση_RMS_%']:.2f} %")
    print(f"\nΠλάτος μόνιμης κατάστασης χωρίς TMD : {r['u_ss_χωρίς']:.4f} m")
    print(f"Πλάτος μόνιμης κατάστασης με TMD    : {r['u_ss_με']:.4f} m")
    print(f"Ποσοστό Μείωσης μόνιμης κατάστασης  : {r['μείωση_u_ss_%']:.2f} %")
    print(
        f"\nΜέγιστη σχετική διαδρομή TMD    : {r['διαδρομή_TMD']:.4f} m "
        f"({r['λόγος_διαδρομής']:.1f} x η μετακίνηση της κατασκευής)"
    )


def to_dataframe(rows) -> pd.DataFrame:
    """Μετατροπή λίστας από dict (μία ανά ανάλυση) σε DataFrame."""
    return pd.DataFrame(list(rows))


def save_table(df: pd.DataFrame, path, float_format: str = "%.4f") -> None:
    """Αποθήκευση πίνακα σε csv (με UTF-8 BOM ώστε να ανοίγει σωστά στο Excel)."""
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format=float_format)
    print(f"Αποθηκεύτηκε: {path}")


def metrics_dict(metrics: Metrics) -> dict:
    """Βοηθητικό: τα Metrics ως dict."""
    return asdict(metrics)
