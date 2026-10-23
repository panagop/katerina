"""
Σχεδίαση και αποθήκευση διαγραμμάτων.

Κάθε συνάρτηση φτιάχνει ένα σχήμα και επιστρέφει το Figure. Αν δοθεί `save`,
το αποθηκεύει· αν `show=True`, το εμφανίζει. Έτσι τα ίδια διαγράμματα
χρησιμοποιούνται και σε διαδραστική εκτέλεση και σε μαζικές αναλύσεις.
"""

from pathlib import Path

import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent / "figures"


def _finish(fig, save=None, show=False):
    """Κοινό τελείωμα: tight_layout, αποθήκευση, εμφάνιση ή κλείσιμο."""
    fig.tight_layout()
    if save is not None:
        save = Path(save)
        if not save.is_absolute():
            save = FIG_DIR / save
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150)
        print(f"Αποθηκεύτηκε: {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_excitation(excitation, xlim=None, save=None, show=False):
    """Η χρονοϊστορία της διέγερσης βάσης."""
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(excitation.t, excitation.a, color="k", linewidth=0.8)
    ax.set_title(f"Διέγερση βάσης - {excitation.name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Χρόνος (s)")
    ax.set_ylabel(excitation.unit_label)
    ax.set_xlim(xlim if xlim is not None else (0, excitation.t_final))
    ax.grid(True, linestyle=":", alpha=0.7)
    return _finish(fig, save, show)


def plot_responses(baseline, controlled, title=None, normalize=False, save=None, show=False):
    """Σύγκριση μετακίνησης κατασκευής χωρίς και με TMD."""
    scale = max(abs(baseline.u.max()), abs(baseline.u.min())) if normalize else 1.0
    ylabel = "u / u_max" if normalize else "Μετακίνηση (m)"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(baseline.t, baseline.u / scale, label="Χωρίς TMD", color="blue", linewidth=1.5)
    ax.plot(
        controlled.t, controlled.u / scale,
        label="Με TMD", color="red", linestyle="--", linewidth=1.5,
    )
    ax.set_title(
        title or f"Απόκριση συστήματος - {baseline.excitation.name}",
        fontweight="bold",
    )
    ax.set_xlabel("Χρόνος (s)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, baseline.excitation.t_final)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper right")
    return _finish(fig, save, show)


def plot_stroke(controlled, save=None, show=False):
    """Η σχετική διαδρομή του TMD - ο απαιτούμενος χώρος κίνησης της μάζας."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(controlled.t, controlled.stroke, color="green", linewidth=1.5)
    ax.set_title(
        f"Σχετική διαδρομή TMD ως προς την κατασκευή - {controlled.excitation.name}",
        fontweight="bold",
    )
    ax.set_xlabel("Χρόνος (s)")
    ax.set_ylabel("u_TMD - u_κατασκευής (m)")
    ax.set_xlim(0, controlled.excitation.t_final)
    ax.grid(True, linestyle=":", alpha=0.7)
    return _finish(fig, save, show)


def plot_sweep(df, x, ys, xlabel, ylabel, title, group=None, save=None, show=False):
    """
    Γενικό διάγραμμα παραμετρικής ανάλυσης.

    `ys` : λίστα από στήλες του DataFrame που θα σχεδιαστούν.
    `group` : προαιρετική στήλη ομαδοποίησης (π.χ. 'διέγερση'), ώστε να
              σχεδιαστεί μία καμπύλη ανά ομάδα.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if group is None:
        for y in ys:
            ax.plot(df[x], df[y], marker="o", markersize=3, linewidth=1.8, label=y)
    else:
        for name, sub in df.groupby(group, sort=False):
            for y in ys:
                lbl = f"{name} - {y}" if len(ys) > 1 else str(name)
                ax.plot(sub[x], sub[y], marker="o", markersize=3, linewidth=1.8, label=lbl)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.legend()
    return _finish(fig, save, show)
