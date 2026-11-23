"""Παραμετρικές αναλύσεις για την επιλεγμένη διέγερση (βλ. tmd_parametric/run_parametric.py)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import scipy.optimize as opt

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

tab_1d, tab_3d, tab_opt = st.tabs(["Μία παράμετρος", "m̂ × f/f_opt (3Δ)", "Εύρεση Βέλτιστων"])
#δυο tabs (καρτελες), το πρώτο για σάρωση μίας παραμέτρου, το δεύτερο για σάρωση δύο παραμέτρων (3Δ)
#Καρτελα 1: "Τι θα γίνει αν κρατήσω τα πάντα σταθερά και αλλάξω μόνο ένα πράγμα;" (μπορεί να είναι λόγος μαζών, αποσυντονισμός, απόσβεση TMD ή συχνότητα αρμονικής) το αλλο σταθερο 
#καρτελα 2 Αντί να αλλάζει ένα πράγμα, αλλάζει δύο πράγματα ταυτόχρονα: και τη μάζα m̂ και το κούρδισμα f/fopt ...3D Επιφάνεια


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
    DEFAULT_RANGE = {"m̂": (0.01, 0.10, 20), "f/f_opt": (0.80, 1.20, 21),
                     "ξd/ξd,opt": (0.1, 5.0, 23), "ρ": (0.70, 1.30, 61)}
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
        m_lo = c1.number_input("Από", value=0.01, step=0.005, format="%.3f", key="grid_m_lo")
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



# ==========================================================================
# Πλήρης Αναζήτηση (Grid Search) Βέλτιστων Παραμέτρων (f, ξd) ανά m̂
# ==========================================================================

# ==========================================================================
# Πλήρης Αναζήτηση (Grid Search) Βέλτιστων Παραμέτρων (f, ξd) ανά m̂
# ==========================================================================

with tab_opt:
    st.markdown("### Αριθμητική εύρεση βέλτιστων παραμέτρων TMD")
    st.caption(f"Διέγερση: **{ground.name}**. Επιλέξτε την απόσβεση του κτιρίου και σαρώστε "
               f"f και ξd για να βρείτε το ιδανικό κούρδισμα.")

    with st.form("opt_search_form"):
        st.markdown("**Γενική Παράμετρος Κτιρίου**")
        user_xi = st.number_input("Απόσβεση Κτιρίου ξ (π.χ. 0.05 για 5%)", value=0.05, min_value=0.00, max_value=0.20, step=0.01, format="%.2f")
        
        st.divider() 
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Λόγος μαζών m̂**")
            opt_m_lo = st.number_input("Από (m)", value=0.01, step=0.01, format="%.3f")
            opt_m_hi = st.number_input("Έως (m)", value=0.10, step=0.01, format="%.3f")
            opt_m_n = st.number_input("Πλήθος (m)", 2, 20, 10, step=1)
        with c2:
            st.markdown("**Λόγος f**")
            opt_f_lo = st.number_input("Από (f)", value=0.70, step=0.05, format="%.3f")
            opt_f_hi = st.number_input("Έως (f)", value=1.20, step=0.05, format="%.3f")
            opt_f_n = st.number_input("Πλήθος (f)", 2, 200, 20, step=1)
        with c3:
            st.markdown("**Απόσβεση TMD ξd**")
            opt_x_lo = st.number_input("Από (ξd)", value=0.01, step=0.01, format="%.3f")
            opt_x_hi = st.number_input("Έως (ξd)", value=0.25, step=0.01, format="%.3f")
            opt_x_n = st.number_input("Πλήθος (ξd)", 2, 200, 30, step=1)
        
        run_opt = st.form_submit_button("Έναρξη Αναζήτησης", type="primary", icon=":material/search:")

    if run_opt:
        import dataclasses
        temp_structure = dataclasses.replace(structure, ξ=user_xi)
        
        m_vals = np.linspace(opt_m_lo, opt_m_hi, int(opt_m_n))
        f_vals = np.linspace(opt_f_lo, opt_f_hi, int(opt_f_n))
        x_vals = np.linspace(opt_x_lo, opt_x_hi, int(opt_x_n))
        
        total_runs = len(m_vals) * len(f_vals) * len(x_vals)
        st.info(f"Εκτέλεση {total_runs} αναλύσεων με απόσβεση κτιρίου ξ = {user_xi:.2f}...")
        bar = st.progress(0.0, text="Αναζήτηση βέλτιστων...")
        
        baseline = core.analyse(temp_structure, ground)
        best_results = []
        counter = 0

        # --- ΝΕΟ: Εύρεση Βέλτιστων ΚΑΙ για RMS ΚΑΙ για u_max ταυτόχρονα! ---
        for mh in m_vals:
            best_rms = float('inf')
            best_umax = float('inf')
            f_rms, x_rms, row_rms = 0, 0, None
            f_umax, x_umax, row_umax = 0, 0, None
            
            for f_val in f_vals:
                for x_val in x_vals:
                    tmd = core.design_tmd(temp_structure, mh, f=f_val, ξd=x_val)
                    row = core.results.compare(baseline, core.analyse(temp_structure, ground, tmd))
                    
                    if row["RMS_με"] < best_rms:
                        best_rms = row["RMS_με"]
                        f_rms, x_rms, row_rms = f_val, x_val, row
                        
                    if row["u_max_με"] < best_umax:
                        best_umax = row["u_max_με"]
                        f_umax, x_umax, row_umax = f_val, x_val, row
                    
                    counter += 1
                    if counter % 100 == 0: 
                        bar.progress(counter / total_runs, text=f"m̂ = {mh:.3f} | Αναλύθηκαν {counter}/{total_runs}")
            
            # Αποθηκεύουμε τα βέλτιστα ΚΑΙ των 2 κριτηρίων στον ίδιο πίνακα
            best_results.append({
                "m̂": mh,
                "f_opt_RMS": f_rms, "ξd_opt_RMS": x_rms, 
                "Μείωση_RMS_%": row_rms["μείωση_RMS_%"], "RMS_με": row_rms["RMS_με"],
                "f_opt_umax": f_umax, "ξd_opt_umax": x_umax, 
                "Μείωση_umax_%": row_umax["μείωση_u_max_%"], "u_max_με": row_umax["u_max_με"],
                "u_max_χωρίς": row_umax["u_max_χωρίς"] # Το χρειαζόμαστε για τα metrics μετά
            })

        bar.empty()
        st.session_state["opt_result"] = (pd.DataFrame(best_results), ground.name, user_xi)

    stored_opt = st.session_state.get("opt_result")
    if stored_opt is not None:
        df_opt, label, saved_xi = stored_opt
        st.success(f"Η αναζήτηση ολοκληρώθηκε για τη διέγερση: {label} (με ξ = {saved_xi:.2f})")
        
        # --- 1. ΥΠΟΛΟΓΙΣΜΟΣ ΕΞΙΣΩΣΕΩΝ (CURVE FITTING) ΓΙΑ RMS & U_MAX ---
        m_arr = df_opt["m̂"].values
        
        def eq_template_f(m, a, b, c):
            return a * (np.sqrt(1 - 0.5 * m) / (1 + m)) + b * np.sqrt(m) + c
            
        def eq_template_x(m, a, b, c):
            return a * np.sqrt((3 * m) / (8 * (1 + m) * (1 - 0.5 * m))) + b * m + c
            
        import scipy.optimize as opt
        try:
            # Fit για RMS
            popt_f_rms, _ = opt.curve_fit(eq_template_f, m_arr, df_opt["f_opt_RMS"].values)
            popt_x_rms, _ = opt.curve_fit(eq_template_x, m_arr, df_opt["ξd_opt_RMS"].values)
            df_opt["Προσαρμοσμένο_f_opt_RMS"] = eq_template_f(m_arr, *popt_f_rms)
            df_opt["Προσαρμοσμένο_ξd_opt_RMS"] = eq_template_x(m_arr, *popt_x_rms)
            
            # Fit για u_max
            popt_f_umax, _ = opt.curve_fit(eq_template_f, m_arr, df_opt["f_opt_umax"].values)
            popt_x_umax, _ = opt.curve_fit(eq_template_x, m_arr, df_opt["ξd_opt_umax"].values)
            df_opt["Προσαρμοσμένο_f_opt_umax"] = eq_template_f(m_arr, *popt_f_umax)
            df_opt["Προσαρμοσμένο_ξd_opt_umax"] = eq_template_x(m_arr, *popt_x_umax)
            
            fit_success = True
        except Exception as e:
            st.error(f"Το Curve Fitting απέτυχε: {e}")
            fit_success = False

        # --- 2. ΕΜΦΑΝΙΣΗ ΕΞΙΣΩΣΕΩΝ LATEX ---
        st.divider()
        st.markdown("### 📈 Εμπειρικές Εξισώσεις (Προσαρμογή Curve Fit)")
        
        if fit_success:
            def format_f_eq(p):
                sign_b = "+" if p[1] >= 0 else "-"
                sign_c = "+" if p[2] >= 0 else "-"
                return f"f_{{opt}} = {p[0]:.4f} \\left( \\frac{{\\sqrt{{1 - 0.5\\bar{{m}}}}}}{{1 + \\bar{{m}}}} \\right) {sign_b} {abs(p[1]):.4f} \\sqrt{{\\bar{{m}}}} {sign_c} {abs(p[2]):.4f}"

            def format_x_eq(p):
                sign_b = "+" if p[1] >= 0 else "-"
                sign_c = "+" if p[2] >= 0 else "-"
                return f"\\xi_{{d,opt}} = {p[0]:.4f} \\sqrt{{ \\frac{{3\\bar{{m}}}}{{8(1+\\bar{{m}})(1-0.5\\bar{{m}})}} }} {sign_b} {abs(p[1]):.4f} \\bar{{m}} {sign_c} {abs(p[2]):.4f}"

            # R^2 για RMS
            r2_f_rms = 1 - np.sum((df_opt["f_opt_RMS"].values - df_opt["Προσαρμοσμένο_f_opt_RMS"])**2) / np.sum((df_opt["f_opt_RMS"].values - np.mean(df_opt["f_opt_RMS"].values))**2)
            r2_x_rms = 1 - np.sum((df_opt["ξd_opt_RMS"].values - df_opt["Προσαρμοσμένο_ξd_opt_RMS"])**2) / np.sum((df_opt["ξd_opt_RMS"].values - np.mean(df_opt["ξd_opt_RMS"].values))**2)
            
            # R^2 για u_max
            r2_f_umax = 1 - np.sum((df_opt["f_opt_umax"].values - df_opt["Προσαρμοσμένο_f_opt_umax"])**2) / np.sum((df_opt["f_opt_umax"].values - np.mean(df_opt["f_opt_umax"].values))**2)
            r2_x_umax = 1 - np.sum((df_opt["ξd_opt_umax"].values - df_opt["Προσαρμοσμένο_ξd_opt_umax"])**2) / np.sum((df_opt["ξd_opt_umax"].values - np.mean(df_opt["ξd_opt_umax"].values))**2)

            colA, colB = st.columns(2)
            with colA:
                st.markdown("**🎯 Βέλτιστα για Ελάχιστο RMS (Συνολική Ενέργεια)**")
                st.latex(format_f_eq(popt_f_rms) + f" \\quad (R^2 = {r2_f_rms:.4f})")
                st.latex(format_x_eq(popt_x_rms) + f" \\quad (R^2 = {r2_x_rms:.4f})")
            with colB:
                st.markdown("**🎯 Βέλτιστα για Ελάχιστο u_max (Μέγιστη Μετακίνηση)**")
                st.latex(format_f_eq(popt_f_umax) + f" \\quad (R^2 = {r2_f_umax:.4f})")
                st.latex(format_x_eq(popt_x_umax) + f" \\quad (R^2 = {r2_x_umax:.4f})")


        # --- 3. ΓΡΑΦΗΜΑΤΑ (ΑΡΙΘΜΗΤΙΚΟ vs ΠΡΟΣΑΡΜΟΣΜΕΝΟ) ---
        st.divider()
        st.markdown("##### Σύγκριση Αριθμητικής Αναζήτησης και Προσαρμοσμένης Εξίσωσης")
        tab_rms, tab_umax = st.tabs(["Γραφήματα βάσει RMS", "Γραφήματα βάσει u_max"])
        
        with tab_rms:
            c1, c2 = st.columns(2)
            c1.line_chart(df_opt, x="m̂", y=["f_opt_RMS", "Προσαρμοσμένο_f_opt_RMS"], x_label="m̂", y_label="Βέλτιστο f", color=["#ff7f0e", "#9467bd"])
            c2.line_chart(df_opt, x="m̂", y=["ξd_opt_RMS", "Προσαρμοσμένο_ξd_opt_RMS"], x_label="m̂", y_label="Βέλτιστο ξd", color=["#2ca02c", "#9467bd"])
        
        with tab_umax:
            c1, c2 = st.columns(2)
            c1.line_chart(df_opt, x="m̂", y=["f_opt_umax", "Προσαρμοσμένο_f_opt_umax"], x_label="m̂", y_label="Βέλτιστο f", color=["#ff7f0e", "#9467bd"])
            c2.line_chart(df_opt, x="m̂", y=["ξd_opt_umax", "Προσαρμοσμένο_ξd_opt_umax"], x_label="m̂", y_label="Βέλτιστο ξd", color=["#2ca02c", "#9467bd"])
            
        st.dataframe(df_opt, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.4f") for c in df_opt.columns if df_opt[c].dtype.kind == "f"})
        st.download_button("Λήψη Βέλτιστων Τιμών (csv)", core.to_csv(df_opt), file_name=f"optimal_parameters_{ground.name}.csv", mime="text/csv", icon=":material/download:")


     # --- 4. ΙΣΤΟΡΙΚΟ ΧΡΟΝΟΥ (ΓΡΑΦΗΜΑΤΑ) ---
        st.divider()
        st.markdown("### ⏱️ Έλεγχος Σεισμικής Απόκρισης στο Χρόνο")
        
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            # Επιλογή Κριτηρίου
            target_choice = st.radio("Στόχος Βελτιστοποίησης για το Γράφημα:", ["Ελάχιστο RMS", "Ελάχιστο u_max"], horizontal=True)
        
        # Εύρεση της καλύτερης μάζας ανάλογα με την επιλογή του χρήστη
        if target_choice == "Ελάχιστο RMS":
            best_idx = int(df_opt["Μείωση_RMS_%"].idxmax())
        else:
            best_idx = int(df_opt["Μείωση_umax_%"].idxmax())
            
        best_overall_m = df_opt.loc[best_idx, "m̂"]
        default_idx = list(m_arr).index(best_overall_m)
        
        with c_sel2:
            selected_m = st.selectbox("Επιλέξτε Λόγο Μαζών (m̂):", m_arr, index=default_idx, format_func=lambda x: f"{x:.3f}")
        
        idx = list(m_arr).index(selected_m)
        if target_choice == "Ελάχιστο RMS":
            f_num = df_opt.loc[idx, "f_opt_RMS"]
            x_num = df_opt.loc[idx, "ξd_opt_RMS"]
        else:
            f_num = df_opt.loc[idx, "f_opt_umax"]
            x_num = df_opt.loc[idx, "ξd_opt_umax"]
            
        if selected_m == best_overall_m:
            st.success(f"🏆 Αυτή η μάζα (m̂ = {selected_m:.3f}) έδωσε τη μέγιστη μείωση για το κριτήριο {target_choice}! Εφαρμόζονται: **f = {f_num:.4f}** και **ξd = {x_num:.4f}**")
        else:
            st.info(f"Για m̂ = {selected_m:.3f} εφαρμόζονται τα βέλτιστα ({target_choice}): **f = {f_num:.4f}** και **ξd = {x_num:.4f}**")
        
        # --- Ανάλυση με τα νέα δεδομένα ---
        import dataclasses
        temp_structure2 = dataclasses.replace(structure, ξ=saved_xi)
        tmd_num = core.design_tmd(temp_structure2, selected_m, f=f_num, ξd=x_num)
        
        with st.spinner("Ανάλυση στο OpenSees..."):
            res_baseline = core.analyse(temp_structure2, ground)
            res_tmd = core.analyse(temp_structure2, ground, tmd_num)
        
        r_opt = core.results.compare(res_baseline, res_tmd)
        
        with st.container(horizontal=True):
            st.metric("u_max χωρίς TMD", f"{r_opt['u_max_χωρίς']:.4f} m", border=True)
            st.metric("u_max με TMD", f"{r_opt['u_max_με']:.4f} m", delta=f"{-r_opt['μείωση_u_max_%']:.2f} %", delta_color="inverse", border=True)
            st.metric("Ενεργός Τιμή RMS", f"{r_opt['RMS_με']:.4f} m", delta=f"{-r_opt['μείωση_RMS_%']:.2f} %", delta_color="inverse", border=True)
        
        # --- ΧΕΙΡΟΚΙΝΗΤΗ ΚΑΤΑΣΚΕΥΗ ΤΩΝ DATAFRAME ΓΙΑ ΤΑ ΓΡΑΦΗΜΑΤΑ ---
        try:
            # Τραβάμε τον χρόνο από το ground
            t = ground.t
            
            # Τραβάμε τις μετακινήσεις (u) από τα αποτελέσματα
            u_base = res_baseline.u if res_baseline.u.ndim == 1 else res_baseline.u[:, 0]
            u_tmd_bldg = res_tmd.u[:, 0] if res_tmd.u.ndim > 1 else res_tmd.u
            
            df_hist = pd.DataFrame({
                "t (s)": t,
                "u χωρίς TMD (m)": u_base,
                "u με TMD (m)": u_tmd_bldg
            })
            
            st.subheader("Μετακίνηση κατασκευής", anchor=False)
            st.line_chart(df_hist, x="t (s)", y=["u χωρίς TMD (m)", "u με TMD (m)"], 
                          x_label="Χρόνος (s)", y_label="Μετακίνηση u (m)", 
                          color=["#9aa0a6", "#d62728"])
            
            # Υπολογισμός διαδρομής TMD (Σχετική μετακίνηση: TMD - Κτίριο)
            if res_tmd.u.ndim > 1 and res_tmd.u.shape[1] > 1:
                stroke = res_tmd.u[:, 1] - res_tmd.u[:, 0]
                df_stroke = pd.DataFrame({
                    "t (s)": t,
                    "διαδρομή TMD (m)": stroke
                })
                st.subheader("Σχετική διαδρομή TMD", anchor=False)
                st.line_chart(df_stroke, x="t (s)", y="διαδρομή TMD (m)", 
                              x_label="Χρόνος (s)", y_label="u₃ − u₂ (m)")
                              
        except AttributeError as e:
            st.error(f"Δεν μπόρεσα να βρω τις μεταβλητές της μετακίνησης. Σφάλμα: {e}")
            st.info("Πιθανόν στο `core.py` τα αποτελέσματα να μην αποθηκεύονται ως `.u`. Έλεγξε πώς ονομάζεται η μετακίνηση (π.χ. `.disp`) και αντικατάστησέ το στον κώδικα.")