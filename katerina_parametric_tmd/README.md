# Παραμετρικές αναλύσεις TMD

Αναδιάρθρωση των `aug26/TMD_gp.py` και `aug26/TMD_harm_excit_gp.py` σε
συναρτήσεις, ώστε να τρέχουν πολλές αναλύσεις με διαφορετικές παραμέτρους.

## Αρχεία

| Αρχείο | Περιεχόμενο |
|---|---|
| `structure.py` | `Structure`, `TMD`, σχέσεις Tsai & Lin, `design_tmd()` |
| `excitation.py` | `Excitation`, `elcentro()`, `harmonic()`, `load_from_file()` |
| `analysis.py` | `build_model()`, `run_analysis()`, `run_pair()`, `Response` |
| `results.py` | `compute_metrics()`, `compare()`, `print_report()`, `save_table()` |
| `plots.py` | `plot_excitation()`, `plot_responses()`, `plot_stroke()`, `plot_sweep()` |
| `run_single.py` | Μία ανάλυση (σεισμική ή αρμονική) - ό,τι έκαναν τα αρχικά scripts |
| `run_parametric.py` | Τρεις σαρώσεις παραμέτρων, με csv και png |

Τα δεδομένα διαβάζονται από τον φάκελο `../aug26/` (El Centro). Η αρμονική
διέγερση παράγεται στη μνήμη, ώστε να μπορεί να αλλάζει η συχνότητά της.

## Εκτέλεση

```bash
uv run --project .. python run_single.py
uv run --project .. python run_single.py --excitation harmonic --m-hat 0.05
uv run --project .. python run_parametric.py            # και οι τρεις σαρώσεις
uv run --project .. python run_parametric.py --sweep frequency --m-hat 0.03
```

Στα Windows, για σωστή εμφάνιση των ελληνικών στην κονσόλα:
`python -X utf8 run_single.py`.

Τα διαγράμματα αποθηκεύονται στο `figures/`, οι πίνακες στο `results/`.

## Παραδοχές

- Μονάδες: tn, kN/m, m, s. Οι τιμές του El Centro είναι σε g και
  μετατρέπονται σε m/s² κατά την ανάγνωση (`factor = 9.81`).
- Η διέγερση εφαρμόζεται πάντα ως επιτάχυνση βάσης (`UniformExcitation`),
  δηλαδή φορτίζει και τη μάζα του TMD. Αυτό είναι συνεπές με τις σχέσεις
  Tsai & Lin, που έχουν διατυπωθεί για διέγερση στήριξης.
- Η απόσβεση δίνεται από υλικά `Viscous` παράλληλα με τα `Elastic`, οπότε
  δεν χρησιμοποιείται απόσβεση Rayleigh.
- Το πλήθος των βημάτων είναι σταθερό (`n_steps = σημεία - 1`), ώστε η
  ανάλυση να τελειώνει ακριβώς στο τέλος της διέγερσης.

## Παραδοτέα μεγέθη

Για κάθε ανάλυση καταγράφονται: μέγιστη μετακίνηση, RMS, πλάτος μόνιμης
κατάστασης (τελευταίο 20% της χρονοϊστορίας) και **μέγιστη σχετική διαδρομή
του TMD** - το μέγεθος που καθορίζει αν η συσκευή είναι κατασκευάσιμη.
