# Open issues — katerina

One entry per issue, short; details live where the link says. Add an entry when an issue comes
up; **remove it in the commit that settles it**. Katerina's Greek copy is the «Ανοιχτά θέματα»
list in `REVIEW_NOTES.md`: keep the two in step. Rules and the list of all projects: the global
`CLAUDE.md`, "Open issues".

- **2026-11-25 · Curve fit in the «Εύρεση Βέλτιστων» tab — discuss with Katerina.** Grid optima
  are quantized (f step ≈ 0.026) and, for u_max under El Centro, noisy; a 3-coefficient fit to
  ~10 points gives high R² but meaningless coefficients. R² is 0/0 when all optima are equal;
  optima on the grid edge are just the edge; the form omits Tsai & Lin's ξ terms. Code:
  `tmd_app/app_pages/parametric.py`, tab `tab_opt`. Details: `REVIEW_NOTES.md`, «Ανοιχτά θέματα».
- **2026-11-25 · `aug26/TMD.py` — waiting on Katerina.** Two questions (which El Centro record
  the thesis uses, `elcentro_NS.txt` 53.7 s or `elCentro.txt` 31.2 s; whether the sign flip
  `/ -u_max_SPOF` is deliberate) and five fixes she makes herself. When she pushes: compare with
  `TMD_gp.py` and the reference values in `CLAUDE.md`. Details: `REVIEW_NOTES.md`, section
  «`aug26/TMD.py`».
- **2026-11-25 · Units of `aug26/Athens.txt` — Katerina to confirm.** Read as m/s² by
  `tmd_parametric/excitation.py` `Athens()` (factor 1; peak 3.01 → PGA 0.31 g; as g it would be
  3 g). Her original code multiplied by 9.81. Details: `REVIEW_NOTES.md`, item 7.
