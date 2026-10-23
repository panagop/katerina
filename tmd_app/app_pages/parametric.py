"""Παραμετρικές αναλύσεις για την επιλεγμένη διέγερση (βλ. tmd_parametric/run_parametric.py)."""

import numpy as np
import pandas as pd
import streamlit as st

import core

SWEEPS = {
    "m̂": "Λόγος μαζών m̂",
    "f/f_opt": "Αποσυντονισμός f/f_opt",
    "ξd/ξd,opt": "Απόσβεση TMD ξd/ξd,opt",
    "ρ": "Συχνότητα αρμονικής ρ = Ω/ω",
}

structure = core.current_structure()
ground = core.current_excitation(structure)
m_hat0 = float(st.session_state.m_hat)

sweep = st.segmented_control("Παράμετρος σάρωσης", list(SWEEPS), format_func=SWEEPS.get,
                             default="m̂", required=True, key="sweep_kind",
                             persist_state="session")

if sweep == "ρ":
    st.caption("Καμπύλη απόκρισης συχνότητας: αρμονικές διεγέρσεις σταθερού πλάτους A = 2ξk/m. "
               "Δεν χρησιμοποιεί την επιλεγμένη διέγερση. Το TMD είναι το βέλτιστο για το τρέχον m̂.")
elif sweep == "m̂":
    st.caption(f"Διέγερση: **{ground.name}**. Για κάθε m̂ το TMD σχεδιάζεται με τις βέλτιστες "
               "τιμές Tsai & Lin.")
else:
    st.caption(f"Διέγερση: **{ground.name}**, m̂ = {m_hat0:g}. Η άλλη παράμετρος του TMD "
               "μένει στη βέλτιστη τιμή.")

FILE_TAG = {"m̂": "mass_ratio", "f/f_opt": "detuning", "ξd/ξd,opt": "damping", "ρ": "frequency"}
DEFAULT_RANGE = {"m̂": (0.005, 0.10, 20), "f/f_opt": (0.80, 1.20, 21),
                 "ξd/ξd,opt": (0.25, 3.0, 23), "ρ": (0.70, 1.30, 61)}
lo0, hi0, n0 = DEFAULT_RANGE[sweep]

with st.form(f"sweep_form_{sweep}"):
    c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
    lo = c1.number_input("Από", value=lo0, step=0.005, format="%.3f")
    hi = c2.number_input("Έως", value=hi0, step=0.005, format="%.3f")
    n = c3.number_input("Πλήθος τιμών", 2, 200, n0, step=1)
    duration = 60.0
    if sweep == "ρ":
        duration = c4.number_input("Διάρκεια (s)", 5.0, 300.0, 60.0, step=10.0)
    run = st.form_submit_button("Εκτέλεση", type="primary", icon=":material/play_arrow:")

# Χωρίς st.stop(): θα σταματούσε και τη σύνοψη της πλαϊνής στήλης
if run and hi <= lo:
    st.error("Η τιμή «Έως» πρέπει να είναι μεγαλύτερη από την «Από».")
elif run:
    values = np.linspace(lo, hi, int(n))
    rows = []
    bar = st.progress(0.0, text="Αναλύσεις...")

    if sweep == "ρ":
        tmd = core.design_tmd(structure, m_hat0)
        amplitude = 2.0 * structure.ξ * structure.k / structure.m
        for i, rho in enumerate(values):
            g = core.exc.harmonic(structure, rho=rho, amplitude=amplitude, duration=duration)
            m_b = core.results.compute_metrics(core.analyse(structure, g))
            m_c = core.results.compute_metrics(core.analyse(structure, g, tmd))
            rows.append({"ρ": rho, "u_ss_χωρίς": m_b.u_ss, "u_ss_με": m_c.u_ss,
                         "μείωση_u_ss_%": core.results.reduction(m_b.u_ss, m_c.u_ss),
                         "διαδρομή_TMD": m_c.stroke_max})
            bar.progress((i + 1) / len(values), text=f"ρ = {rho:.3f}")
    else:
        # Η ανάλυση χωρίς TMD δεν εξαρτάται από την παράμετρο - τρέχει μία φορά
        baseline = core.analyse(structure, ground)
        f_opt = float(core.tsai_lin_f(m_hat0, structure.ξ))
        xid_opt = float(core.tsai_lin_xi(m_hat0, structure.ξ))
        for i, v in enumerate(values):
            if sweep == "m̂":
                tmd = core.design_tmd(structure, v)
            elif sweep == "f/f_opt":
                tmd = core.design_tmd(structure, m_hat0, f=v * f_opt)
            else:
                tmd = core.design_tmd(structure, m_hat0, ξd=v * xid_opt)
            row = core.results.compare(baseline, core.analyse(structure, ground, tmd))
            rows.append({sweep: v, **row})
            bar.progress((i + 1) / len(values), text=f"{sweep} = {v:.3f}")

    bar.empty()
    label = "αρμονικές" if sweep == "ρ" else ground.name
    st.session_state[f"sweep_result_{sweep}"] = (pd.DataFrame(rows), label)

stored = st.session_state.get(f"sweep_result_{sweep}")
if stored is None:
    st.info("Ορίστε το εύρος και πατήστε «Εκτέλεση».")
else:
    df, label = stored
    st.caption(f"Αποτελέσματα της τελευταίας εκτέλεσης (διέγερση: {label}).")

    # --------------------------------------------------------------------------
    # Διαγράμματα
    # --------------------------------------------------------------------------

    x_label = SWEEPS[sweep]
    if sweep == "ρ":
        st.line_chart(df, x="ρ", y=["u_ss_χωρίς", "u_ss_με"], x_label=x_label,
                      y_label="Πλάτος μόνιμης κατάστασης (m)", color=["#9aa0a6", "#d62728"],
                      alt="Καμπύλη απόκρισης συχνότητας με και χωρίς TMD")
        st.line_chart(df, x="ρ", y="διαδρομή_TMD", x_label=x_label,
                      y_label="Μέγιστη διαδρομή TMD (m)", alt="Διαδρομή TMD συναρτήσει του ρ")
    else:
        c1, c2 = st.columns(2)
        c1.line_chart(df, x=sweep, y=["μείωση_u_max_%", "μείωση_RMS_%"], x_label=x_label,
                      y_label="Μείωση (%)", alt="Μείωση μέγιστης και RMS μετακίνησης")
        c2.line_chart(df, x=sweep, y="διαδρομή_TMD", x_label=x_label,
                      y_label="Μέγιστη διαδρομή TMD (m)", alt="Μέγιστη διαδρομή TMD")

    st.dataframe(df, hide_index=True, alt="Αποτελέσματα σάρωσης",
                 column_config={c: st.column_config.NumberColumn(format="%.4f")
                                for c in df.columns if df[c].dtype.kind == "f"})
    st.download_button("Λήψη πίνακα (csv)", core.to_csv(df), file_name=f"sweep_{FILE_TAG[sweep]}.csv",
                       mime="text/csv", icon=":material/download:")
