"""
Διεγέρσεις: ανάγνωση από αρχείο (σεισμός) ή παραγωγή αρμονικής διέγερσης.

Σύμβαση: το πεδίο `a` της Excitation είναι ΠΑΝΤΑ σε m/s^2 και εφαρμόζεται ως
επιτάχυνση βάσης (UniformExcitation). Τυχόν συντελεστής μετατροπής (π.χ. g -> m/s^2)
εφαρμόζεται εδώ, κατά την ανάγνωση, και όχι μέσα στο μοντέλο.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Τα αρχεία δεδομένων βρίσκονται στον φάκελο aug26, δίπλα σε αυτόν τον φάκελο.
DATA_DIR = Path(__file__).resolve().parent.parent / "aug26"

G = 9.81  # m/s^2


@dataclass(frozen=True)
class Excitation:
    """Χρονοϊστορία επιτάχυνσης βάσης."""

    name: str           # Όνομα για τίτλους/αρχεία
    dt: float           # Χρονικό βήμα (s)
    a: np.ndarray       # Επιταχύνσεις σε m/s^2
    unit_label: str = "Επιτάχυνση (m/s²)"

    @property
    def n_steps(self) -> int:
        """Πλήθος χρονικών ΒΗΜΑΤΩΝ (= σημεία - 1)."""
        return len(self.a) - 1

    @property
    def t_final(self) -> float:
        return self.n_steps * self.dt

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.a)) * self.dt


def load_from_file(path, name=None, factor=1.0, unit_label="Επιτάχυνση (m/s²)") -> Excitation:
    """
    Ανάγνωση αρχείου δύο στηλών [χρόνος, επιτάχυνση].

    Το dt υπολογίζεται από τα δεδομένα. Ο `factor` πολλαπλασιάζει τις τιμές
    ώστε το αποτέλεσμα να είναι σε m/s^2 (π.χ. 9.81 για τιμές σε g).
    """
    path = Path(path)
    data = np.loadtxt(path)
    t = data[:, 0]
    dt = round(float(t[1] - t[0]), 5)
    return Excitation(
        name=name or path.stem,
        dt=dt,
        a=data[:, 1] * factor,
        unit_label=unit_label,
    )


def elcentro(path=None) -> Excitation:
    """Το επιταχυνσιογράφημα El Centro (τιμές σε g -> μετατροπή σε m/s^2)."""
    return load_from_file(
        path or DATA_DIR / "elCentro.txt",
        name="El Centro",
        factor=G,
    )


def harmonic(structure, rho=1.0, amplitude=None, dt=0.01, duration=60.0) -> Excitation:
    """
    Αρμονική διέγερση βάσης a(t) = A·sin(Ω t), με Ω = ρ·ω.

    Παράγεται στη μνήμη (δεν χρειάζεται αρχείο), ώστε να μπορούμε να σαρώσουμε
    τον λόγο συχνοτήτων ρ σε παραμετρικές αναλύσεις.

    Το προεπιλεγμένο πλάτος A = 2ξk/m είναι αυτό του αρχείου
    'harmonic excitation.txt': δίνει πλάτος μόνιμης κατάστασης ακριβώς 1.00 m
    για την ακυβέρνητη κατασκευή σε συντονισμό (ρ = 1).
    """
    if amplitude is None:
        amplitude = 2.0 * structure.ξ * structure.k / structure.m

    Ω = rho * structure.ω
    t = np.arange(0.0, duration + dt / 2, dt)
    return Excitation(
        name=f"Αρμονική ρ={rho:.2f}",
        dt=dt,
        a=amplitude * np.sin(Ω * t),
    )


def harmonic_from_file(path=None) -> Excitation:
    """
    Η αρμονική διέγερση όπως την είχε αποθηκεύσει το 'harmonic excitation.py'.

    Οι τιμές εκεί είναι ήδη P/m, δηλαδή σε m/s^2, οπότε factor = 1.
    """
    return load_from_file(
        path or DATA_DIR / "harmonic excitation.txt",
        name="Αρμονική (αρχείο)",
        factor=1.0,
    )
