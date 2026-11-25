"""
Διεγέρσεις: ανάγνωση από αρχείο (σεισμός) ή παραγωγή αρμονικής διέγερσης.

Σύμβαση: το πεδίο `a` της Excitation είναι ΠΑΝΤΑ σε m/s^2 και εφαρμόζεται ως
επιτάχυνση βάσης (UniformExcitation). Τυχόν συντελεστής μετατροπής (π.χ. g -> m/s^2)
εφαρμόζεται εδώ, κατά την ανάγνωση, και όχι μέσα στο μοντέλο.
"""

import re
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


def load_peer_at2(path, name=None, scale=1.0) -> Excitation:
    """Αρχείο PEER .AT2 (τιμές σε g) -> Excitation σε m/s^2, με προαιρετική κλιμάκωση."""
    path = Path(path)
    dt, a = parse_peer_at2(path.read_text(encoding="utf-8", errors="replace"))
    return Excitation(name=name or path.stem, dt=dt, a=a * G * scale)


def elcentro(path=None) -> Excitation:
    """Το επιταχυνσιογράφημα El Centro (τιμές σε g -> μετατροπή σε m/s^2)."""
    return load_from_file(
        path or DATA_DIR / "elCentro.txt",
        name="El Centro",
        factor=G,
    )


def Athens(path=None) -> Excitation:
    """Το επιταχυνσιογράφημα Athens 1999 (τιμές σε m/s^2, χωρίς μετατροπή).

    ΠΡΟΣΟΧΗ: το αρχείο έχει μέγιστη τιμή 3.01. Σε m/s^2 αυτό είναι PGA ≈ 0.31 g, λογικό
    για τον σεισμό της Αθήνας· σε g θα ήταν 3 g. Γι' αυτό factor = 1 (όχι 9.81).
    Να επιβεβαιωθεί από την πηγή του αρχείου.
    """
    return load_from_file(
        path or DATA_DIR / "Athens.txt",
        name="Athens",
        factor=1.0,
    )


# Τα αρχεία των δύο παρακάτω ΔΕΝ υπάρχουν ακόμα στο aug26/: πρέπει να προστεθούν
# (δύο στήλες [χρόνος, επιτάχυνση σε g]) πριν χρησιμοποιηθούν οι συναρτήσεις.

def thessaloniki78(path=None) -> Excitation:
    """Το επιταχυνσιογράφημα Thessaloniki '78 (τιμές σε g -> μετατροπή σε m/s^2)."""
    return load_from_file(
        path or DATA_DIR / "thessaloniki78.txt",
        name="Thessaloniki '78",
        factor=G,
    )


def mexico95(path=None) -> Excitation:
    """Το επιταχυνσιογράφημα Mexico '95 (τιμές σε g -> μετατροπή σε m/s^2)."""
    return load_from_file(
        path or DATA_DIR / "mexico95.txt",
        name="Mexico '95",
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
