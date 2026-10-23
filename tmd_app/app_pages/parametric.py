"""Παραμετρικές αναλύσεις για την επιλεγμένη διέγερση (βλ. tmd_parametric/run_parametric.py)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

tab_1d, tab_3d = st.tabs(["Μία παράμετρος", "m̂ × f/f_opt (3Δ)"])

# ==========================================================================
# Σάρωση μίας παραμέτρου
# ==========================================================================

with tab_1d:
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

        # ----------------------------------------------------------------------
        # Διαγράμματα
        # ----------------------------------------------------------------------

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

# ==========================================================================
# Σάρωση δύο παραμέτρων: m̂ × f/f_opt
# ==========================================================================

# Μεγέθη που μπορούν να απεικονιστούν στον άξονα z
Z_OPTIONS = {
    "μείωση_u_max_%": "Μείωση u_max (%)",
    "μείωση_RMS_%": "Μείωση RMS (%)",
    "u_max_με": "u_max με TMD (m)",
    "διαδρομή_TMD": "Μέγιστη διαδρομή TMD (m)",
}

with tab_3d:
    st.caption(f"Διέγερση: **{ground.name}**. Για κάθε m̂ το TMD έχει f = (f/f_opt)·f_opt(m̂) "
               "και τη βέλτιστη ξd,opt(m̂) κατά Tsai & Lin. "
               "Η γραμμή f/f_opt = 1 είναι ο βέλτιστος συντονισμός.")

    with st.form("grid_form"):
        st.markdown("**Λόγος μαζών m̂**")
        c1, c2, c3 = st.columns(3)
        m_lo = c1.number_input("Από", value=0.005, step=0.005, format="%.3f", key="grid_m_lo")
        m_hi = c2.number_input("Έως", value=0.10, step=0.005, format="%.3f", key="grid_m_hi")
        m_n = c3.number_input("Πλήθος τιμών", 2, 50, 12, step=1, key="grid_m_n")
        st.markdown("**Αποσυντονισμός f/f_opt**")
        c1, c2, c3 = st.columns(3)
        r_lo = c1.number_input("Από", value=0.80, step=0.05, format="%.3f", key="grid_r_lo")
        r_hi = c2.number_input("Έως", value=1.20, step=0.05, format="%.3f", key="grid_r_hi")
        r_n = c3.number_input("Πλήθος τιμών", 2, 50, 17, step=1, key="grid_r_n")
        run_grid = st.form_submit_button("Εκτέλεση", type="primary", icon=":material/play_arrow:")

    if run_grid and (m_hi <= m_lo or r_hi <= r_lo):
        st.error("Η τιμή «Έως» πρέπει να είναι μεγαλύτερη από την «Από».")
    elif run_grid:
        m_hats = np.linspace(m_lo, m_hi, int(m_n))
        ratios = np.linspace(r_lo, r_hi, int(r_n))
        total = len(m_hats) * len(ratios)
        rows = []
        bar = st.progress(0.0, text="Αναλύσεις...")

        # Η ανάλυση χωρίς TMD δεν εξαρτάται από τις παραμέτρους - τρέχει μία φορά
        baseline = core.analyse(structure, ground)
        for mh in m_hats:
            f_opt = float(core.tsai_lin_f(mh, structure.ξ))
            for r in ratios:
                tmd = core.design_tmd(structure, mh, f=r * f_opt)
                row = core.results.compare(baseline, core.analyse(structure, ground, tmd))
                rows.append({"m̂": mh, "f/f_opt": r, **row})
                bar.progress(len(rows) / total, text=f"m̂ = {mh:.3f}, f/f_opt = {r:.3f}")

        bar.empty()
        st.session_state["grid_result"] = (pd.DataFrame(rows), ground.name)

    stored = st.session_state.get("grid_result")
    if stored is None:
        st.info(f"Ορίστε τα εύρη και πατήστε «Εκτέλεση». Το πλέγμα των προεπιλογών είναι "
                f"12 × 17 = 204 αναλύσεις.")
    else:
        df, label = stored
        st.caption(f"Αποτελέσματα της τελευταίας εκτέλεσης (διέγερση: {label}, "
                   f"{len(df)} αναλύσεις).")

        z_col = st.segmented_control("Μέγεθος στον άξονα z", list(Z_OPTIONS),
                                     format_func=Z_OPTIONS.get, default="μείωση_u_max_%",
                                     required=True, key="grid_z", persist_state="session")
        z_label = Z_OPTIONS[z_col]

        # Πίνακας z: γραμμές f/f_opt, στήλες m̂
        grid = df.pivot(index="f/f_opt", columns="m̂", values=z_col)
        fig = go.Figure(go.Surface(
            x=grid.columns.values, y=grid.index.values, z=grid.values,
            colorscale="Viridis", colorbar=dict(title=dict(text=z_label, side="right")),
            contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))),
            hovertemplate="m̂ = %{x:.3f}<br>f/f_opt = %{y:.3f}<br>" + z_label
                          + " = %{z:.4f}<extra></extra>",
        ))

        # Γραμμή βέλτιστου συντονισμού (αν το 1.0 ανήκει στο πλέγμα)
        at_opt = np.isclose(grid.index.values, 1.0)
        if at_opt.any():
            fig.add_trace(go.Scatter3d(
                x=grid.columns.values, y=np.ones(grid.shape[1]), z=grid.values[at_opt][0],
                mode="lines", line=dict(color="red", width=6), name="f/f_opt = 1",
                hovertemplate="m̂ = %{x:.3f}<br>" + z_label + " = %{z:.4f}<extra>f/f_opt = 1</extra>",
            ))

        # Το καλύτερο σημείο του πλέγματος για τις μειώσεις
        if z_col.startswith("μείωση"):
            best = df.loc[df[z_col].idxmax()]
            fig.add_trace(go.Scatter3d(
                x=[best["m̂"]], y=[best["f/f_opt"]], z=[best[z_col]],
                mode="markers", marker=dict(color="red", size=6, symbol="diamond"),
                name="μέγιστη μείωση",
                hovertemplate="m̂ = %{x:.3f}<br>f/f_opt = %{y:.3f}<br>" + z_label
                              + " = %{z:.2f}<extra>μέγιστο</extra>",
            ))

        fig.update_layout(
            scene=dict(xaxis_title="Λόγος μαζών m̂", yaxis_title="f/f_opt", zaxis_title=z_label,
                       camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))),
            height=650, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=0.0, x=0.0),
        )
        st.plotly_chart(fig, alt=f"Επιφάνεια: {z_label} συναρτήσει του m̂ και του f/f_opt")

        if z_col.startswith("μείωση"):
            st.caption(f"Μέγιστη μείωση στο πλέγμα: {best[z_col]:.2f} % για m̂ = {best['m̂']:.3f}, "
                       f"f/f_opt = {best['f/f_opt']:.3f}.")

        with st.expander("Πίνακας αποτελεσμάτων"):
            st.dataframe(df, hide_index=True, alt="Αποτελέσματα σάρωσης m̂ × f/f_opt",
                         column_config={c: st.column_config.NumberColumn(format="%.4f")
                                        for c in df.columns if df[c].dtype.kind == "f"})
        st.download_button("Λήψη πίνακα (csv)", core.to_csv(df),
                           file_name="sweep_mass_ratio_detuning.csv",
                           mime="text/csv", icon=":material/download:", key="grid_download")
