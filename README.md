# Δυναμική των κατασκευών: φάσματα απόκρισης και TMD

| Φάκελος | Περιεχόμενο |
| --- | --- |
| `spectra/` | Φάσματα απόκρισης El Centro (`part1.py`, `part2.py`, `mePSa.py` κ.ά.), επιταχυνσιογραφήματα `elCentro.txt`, `Athens.txt` και τα αποτελέσματα `.xlsx` |
| `spectra/gp/` | Ίδια ανάλυση σε μορφή modules (`main.py`, `response_spectrum.py`, `response_spectrum_classic.py`) |
| `aug26/` | SDOF με TMD στο OpenSeesPy: `TMD.py` (El Centro), `TMD_harm_excit.py` (αρμονική), διορθωμένα `_gp`, `Tsai&Lin.py` |
| `tmd_parametric/` | Οι αναλύσεις του `aug26/` ως συναρτήσεις, για παραμετρικές αναλύσεις (βλ. `README.md` του φακέλου) |
| `tmd_app/` | Εφαρμογή Streamlit πάνω στο `tmd_parametric/` (βλ. `README.md` του φακέλου) |

## Εκτέλεση

```bash
uv sync
cd spectra
uv run --project .. python -X utf8 part1.py
```

Τα scripts διαβάζουν τα δεδομένα με σχετική διαδρομή, άρα τρέχουν **από τον
φάκελό τους**. Στο VS Code αυτό το ρυθμίζει το `.vscode/settings.json`
(`python.terminal.executeInFileDir`).
