# CLAUDE.md

Coursework/thesis code for a structural dynamics student (Katerina), maintained
with help from her supervisor (the user, a civil engineer). Structural dynamics
in Python: response spectra, and SDOF systems with a Tuned Mass Damper (TMD)
modelled in OpenSeesPy.

## Working conventions

- **Comments, printed output and plot labels are in Greek. Keep them Greek.**
  Greek identifiers (`ω`, `ξ`, `ξd`, `ρ`) are used deliberately — they match the
  notation in the thesis. Don't transliterate them.
- **Never modify the student's original files.** When fixing one of her scripts,
  save a corrected copy with a `_gp` suffix (e.g. `TMD.py` → `TMD_gp.py`).
  This has been an explicit instruction; assume it still holds.
- Fixes should preserve her structure and physics, not rewrite the code. She
  needs to recognise it as hers.
- Console output on Windows needs `python -X utf8` or the Greek comes out as
  mojibake.

## Environment

`uv` project at the repo root (`pyproject.toml`, requires-python >= 3.12).
Deps already installed: numpy, scipy, matplotlib, pandas, openpyxl, openseespy,
jupyter.

```bash
cd <subfolder>
uv run --project .. python -X utf8 script.py
```

Bare `python script.py` fails with `ModuleNotFoundError: matplotlib` — always go
through `uv run`. For headless verification: `matplotlib.use('Agg')` and
`plt.show = lambda *a, **k: None`.

## Layout

Restructured in Oct 2026 (agreed with Katerina): no scripts in the repo root.
Scripts load data by bare relative filename, so they must run from their own
folder; `.vscode/settings.json` sets `python.terminal.executeInFileDir`.

- **`spectra/`** — her response-spectrum scripts (`part1.py`, `part2.py`,
  `mePSa.py`, `part1_copy.py`, `part1panag.py`), the accelerograms
  `elCentro.txt` and `Athens.txt` (Athens 1999), and their `.xlsx` outputs.
- **`spectra/gp/`** — the supervisor's modular version (`main.py`,
  `response_spectrum.py`, `response_spectrum_classic.py`, `elcentro_loader.py`
  reading `data/ELCENTRO`).
- **`aug26/`** — the TMD work. Her originals `TMD.py` (El Centro) and
  `TMD_harm_excit.py` (harmonic), the reviewed copies `TMD_gp.py` /
  `TMD_harm_excit_gp.py`, `Tsai&Lin.py` (plots of the optimum-tuning formulas),
  `harmonic excitation.py` (generates `harmonic excitation.txt`), the data
  files `elCentro.txt`, `harmonic excitation.txt`, and older experiments
  `TMD_sin.py`, `TMD_dokimi.py` (reads `disp_building.txt`, `disp_tmd.txt`).
- **`tmd_parametric/`** — refactor of the two `_gp` scripts into functions for
  parametric studies. See its own `README.md`. Modules: `structure.py`,
  `excitation.py`, `analysis.py`, `results.py`, `plots.py`, drivers
  `run_single.py` and `run_parametric.py`. Outputs go to `figures/` and
  `results/`.

## The model (shared by everything TMD-related)

1-D OpenSees model, `ndm=1, ndf=1`:

- node 1 fixed base; node 2 structure mass; node 3 TMD mass (when present).
- Each connection is a `zeroLength` element with an `Elastic` **and** a
  `Viscous` material in parallel (`-mat k c -dir 1 1`). Damping comes from the
  materials, so **no Rayleigh damping** anywhere.
- Excitation is always applied as base acceleration via
  `timeSeries('Path', ...)` + `pattern('UniformExcitation', ...)`. This loads
  the TMD mass too, which is correct: the Tsai & Lin optimum formulas are
  derived for **support excitation**. The harmonic case is applied the same way
  even though `harmonic excitation.py` frames it as a force `P` — the file
  stores `P/m`, i.e. m/s², so it is already an acceleration.
- Newmark (0.5, 0.25), `ProfileSPD`, `Plain`, `Transformation`, `Linear`,
  `Transient`.
- **Units: tn, kN/m, m, s.** El Centro values are in g and need `× 9.81`.

Baseline parameters: m = 1 tn, T = 0.49 s (ω ≈ 12.82 rad/s), ξ = 0.02,
k ≈ 164.4 kN/m, m_hat = 0.01. El Centro: dt = 0.02 s, 1559 points, 31.16 s.
Harmonic: dt = 0.01 s, 6001 points, 60 s, amplitude `2ξk/m` chosen so the
resonant steady-state amplitude of the bare SDOF is exactly 1.00 m.

## Recurring bugs in this code

- `while t < t_final: t += dt` overshoots by one step from float accumulation.
  Use `n_steps = len(values) - 1` and `for i in range(n_steps)`, then derive
  time as `np.arange(len(disp)) * dt`.
- The time history must be seeded with `t = 0, u = 0` before the loop.
- `ops.analyze()` return value goes unchecked.
- Hard-coded axis limits copied between scripts that clip the actual data
  (`plt.xlim(0, 60)` on a 31 s record; `plt.ylim(-1, 1)` sitting exactly on a
  0.996 m peak).
- Titles copied from the seismic script into the harmonic one.

## Reference results (regressions — a change that moves these needs explaining)

| Case | u_max no TMD | u_max with TMD | peak red. | RMS red. | TMD stroke |
|---|---|---|---|---|---|
| El Centro, m_hat=0.01 | 0.0641 m | 0.0612 m | 4.48 % | 23.54 % | 0.2840 m |
| Harmonic ρ=1, m_hat=0.01 | 0.9963 m | 0.3806 m | 61.80 % | 63.39 % | 2.7925 m |

The contrast is the physically interesting point: a μ = 1 % TMD is a
narrow-band device that needs many cycles at its tuned frequency to build up.
A resonant sine gives it those cycles; an earthquake's first pulse does not.
Stroke is always reported — it's what decides whether the device is buildable,
and it *decreases* as m_hat grows.
