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
        # Προεπιλογή: το ξ της σελίδας «Κατασκευή & TMD», ώστε τα αποτελέσματα να
        # συμφωνούν με τις άλλες σελίδες. Μπορεί να αλλάξει εδώ μόνο για την αναζήτηση.
        user_xi = st.number_input("Απόσβεση Κτιρίου ξ (π.χ. 0.05 για 5%)", value=float(structure.ξ),
                                  min_value=0.00, max_value=0.20, step=0.005, format="%.3f",
                                  help=f"Προεπιλογή το ξ = {structure.ξ:g} της σελίδας «Κατασκευή & TMD».")
        
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
        
        result_store = core.result_store()
        can_store_hist = result_store is not None and core.rstore.histories_available()
        save_hist = st.checkbox(
            "Αποθήκευση χρονοϊστοριών", value=can_store_hist, disabled=not can_store_hist,
            help="Τα βαθμωτά αποτελέσματα (u_max, RMS, διαδρομή) αποθηκεύονται πάντα στο "
                 "results_db/. Με αυτή την επιλογή αποθηκεύονται και οι χρονοϊστορίες "
                 "(≈ 10 kB ανά ανάλυση για το El Centro, πολύ περισσότερο για μεγάλες καταγραφές).")

        run_opt = st.form_submit_button("Έναρξη Αναζήτησης", type="primary", icon=":material/search:")

    if result_store is None:
        st.caption("Η βάση αποτελεσμάτων δεν είναι διαθέσιμη εδώ: οι αναζητήσεις δεν αποθηκεύονται.")

    def best_per_mass(grid):
        """Τα βέλτιστα (ελάχιστο RMS και ελάχιστο u_max) για κάθε m̂, από όλο το πλέγμα."""
        best_results = []
        for mh, g in grid.groupby("m_hat", sort=False):
            row_rms = g.loc[g["RMS_με"].idxmin()]
            row_umax = g.loc[g["u_max_με"].idxmin()]
            # Αποθηκεύουμε τα βέλτιστα ΚΑΙ των 2 κριτηρίων στον ίδιο πίνακα
            best_results.append({
                "m̂": mh,
                "f_opt_RMS": row_rms["f"], "ξd_opt_RMS": row_rms["ξd"],
                "Μείωση_RMS_%": row_rms["μείωση_RMS_%"], "RMS_με": row_rms["RMS_με"],
                "f_opt_umax": row_umax["f"], "ξd_opt_umax": row_umax["ξd"],
                "Μείωση_umax_%": row_umax["μείωση_u_max_%"], "u_max_με": row_umax["u_max_με"],
                "u_max_χωρίς": row_umax["u_max_χωρίς"]  # Το χρειαζόμαστε για τα metrics μετά
            })
        return pd.DataFrame(best_results)

    # --- Αποθηκευμένες αναζητήσεις: φόρτωση χωρίς νέες αναλύσεις ---
    if result_store is not None:
        saved = result_store.searches()
        with st.expander(f"Αποθηκευμένες αναζητήσεις ({len(saved)})", icon=":material/database:"):
            if saved.empty:
                st.caption("Δεν υπάρχουν ακόμα. Κάθε αναζήτηση από εδώ ή από το "
                           "`tmd_parametric/run_optimum_batch.py` αποθηκεύεται στο `results_db/`.")
            else:
                view = saved.rename(columns={
                    "search_id": "α/α", "created": "ημερομηνία", "source": "από",
                    "excitation": "διέγερση", "xi": "ξ", "histories": "χρονοϊστορίες",
                    "n_total": "αναλύσεις", "n_computed": "νέες"})
                st.dataframe(view[["α/α", "ημερομηνία", "από", "διέγερση", "m", "T", "ξ",
                                   "m_lo", "m_hi", "m_n", "f_lo", "f_hi", "f_n",
                                   "xd_lo", "xd_hi", "xd_n", "χρονοϊστορίες", "αναλύσεις", "νέες"]],
                             hide_index=True)
                c_load1, c_load2 = st.columns([3, 1], vertical_alignment="bottom")
                # (στήλες με r["T"]: το r.T σε pandas είναι ο ανάστροφος πίνακας)
                rows = {int(r["search_id"]): r for _, r in saved.iterrows()}
                labels = {i: f'{i}: {r["excitation"]}, T = {r["T"]:g} s, ξ = {r["xi"]:g}, '
                             f'{r["n_total"]} αναλύσεις' for i, r in rows.items()}
                load_id = c_load1.selectbox("Αναζήτηση", list(rows), key="opt_load_id",
                                            format_func=labels.get)
                if c_load2.button("Φόρτωση", icon=":material/download:", key="opt_load"):
                    r = rows[load_id]
                    s_load = core.Structure(m=float(r["m"]), T=float(r["T"]), ξ=float(r["xi"]))
                    g_load = result_store.excitation(r["exc_id"])
                    grid = core.optimum.grid_search(
                        s_load, g_load, core.rstore.grid_values(r["m_lo"], r["m_hi"], r["m_n"]),
                        core.rstore.grid_values(r["f_lo"], r["f_hi"], r["f_n"]),
                        core.rstore.grid_values(r["xd_lo"], r["xd_hi"], r["xd_n"]),
                        store=result_store, log=False, lock=core._OPS_LOCK)
                    st.session_state["opt_result"] = (best_per_mass(grid), g_load.name, s_load.ξ,
                                                      s_load, g_load)

    if run_opt:
        import dataclasses
        temp_structure = dataclasses.replace(structure, ξ=user_xi)
        
        m_vals = core.rstore.grid_values(opt_m_lo, opt_m_hi, opt_m_n)
        f_vals = core.rstore.grid_values(opt_f_lo, opt_f_hi, opt_f_n)
        x_vals = core.rstore.grid_values(opt_x_lo, opt_x_hi, opt_x_n)
        
        total_runs = len(m_vals) * len(f_vals) * len(x_vals)
        n_stored = (core.optimum.count_stored(result_store, temp_structure, ground, m_vals, f_vals, x_vals)
                    if result_store is not None else 0)
        st.info(f"Εκτέλεση {total_runs - n_stored} αναλύσεων με απόσβεση κτιρίου ξ = {user_xi:g} "
                f"({n_stored} από {total_runs} ήδη αποθηκευμένες)...")
        if user_xi != structure.ξ and st.session_state.excitation_spec["kind"] == "harmonic":
            # Το πλάτος της αρμονικής (2ξk/m) υπολογίστηκε με το ξ της κατασκευής
            st.warning(f"Η αρμονική διέγερση έχει πλάτος 2ξk/m με ξ = {structure.ξ:g} (σελίδα "
                       f"«Κατασκευή & TMD»), ενώ η αναζήτηση γίνεται με ξ = {user_xi:g}: η "
                       "απόκριση χωρίς TMD στον συντονισμό δεν θα είναι 1.00 m.")
        bar = st.progress(0.0, text="Αναζήτηση βέλτιστων...")

        def show_progress(done, total):
            bar.progress(done / total, text=f"Αναλύθηκαν {done}/{total}")

        # Όλο το πλέγμα m̂ × f × ξd: τα αποθηκευμένα σημεία διαβάζονται, τα υπόλοιπα
        # τρέχουν και αποθηκεύονται (tmd_parametric/optimum.py)
        grid = core.optimum.grid_search(temp_structure, ground, m_vals, f_vals, x_vals,
                                        store=result_store, histories=save_hist,
                                        progress=show_progress, lock=core._OPS_LOCK)
        best_results = best_per_mass(grid)

        bar.empty()
        # Κρατάμε και την κατασκευή και τη διέγερση της αναζήτησης, ώστε ο έλεγχος
        # στο χρόνο παρακάτω να γίνεται με τα ΙΔΙΑ δεδομένα (και όχι με τα τρέχοντα).
        st.session_state["opt_result"] = (best_results, ground.name, user_xi,
                                          temp_structure, ground)

    stored_opt = st.session_state.get("opt_result")
    if stored_opt is not None:
        df_opt, label, saved_xi, opt_structure, opt_ground = stored_opt
        st.success(f"Η αναζήτηση ολοκληρώθηκε για τη διέγερση: {label} (με ξ = {saved_xi:g})")
        # Αν άλλαξε η κατασκευή ή η διέγερση μετά την αναζήτηση, προειδοποίηση
        same_structure = (opt_structure.m, opt_structure.T) == (structure.m, structure.T)
        same_ground = (opt_ground.name == ground.name and opt_ground.dt == ground.dt
                       and np.array_equal(opt_ground.a, ground.a))
        if not (same_structure and same_ground):
            st.warning("Η κατασκευή ή η διέγερση άλλαξαν μετά την αναζήτηση. Τα αποτελέσματα "
                       f"παρακάτω αφορούν τα δεδομένα της αναζήτησης ({label}, T = "
                       f"{opt_structure.T:g} s, m = {opt_structure.m:g} tn). "
                       "Ξανατρέξτε την αναζήτηση για τα τρέχοντα δεδομένα.")
        
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
        st.caption(f"Ισχύουν για απόσβεση κτιρίου ξ = {saved_xi:g}, διέγερση {label} και "
                   f"m̂ από {df_opt['m̂'].min():g} έως {df_opt['m̂'].max():g}.")
        
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
        st.download_button("Λήψη Βέλτιστων Τιμών (csv)", core.to_csv(df_opt), file_name=f"optimal_parameters_{label}.csv", mime="text/csv", icon=":material/download:")


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
        
        # --- Ανάλυση με τα δεδομένα της αναζήτησης (κατασκευή, ξ και διέγερση) ---
        temp_structure2 = opt_structure
        tmd_num = core.design_tmd(temp_structure2, selected_m, f=f_num, ξd=x_num)

        with st.spinner("Ανάλυση στο OpenSees..."):
            # Από τις αποθηκευμένες χρονοϊστορίες αν υπάρχουν, αλλιώς νέα ανάλυση
            res_baseline = core.response(temp_structure2, opt_ground)
            res_tmd = core.response(temp_structure2, opt_ground, tmd_num)
        
        r_opt = core.results.compare(res_baseline, res_tmd)
        
        with st.container(horizontal=True):
            st.metric("u_max χωρίς TMD", f"{r_opt['u_max_χωρίς']:.4f} m", border=True)
            st.metric("u_max με TMD", f"{r_opt['u_max_με']:.4f} m", delta=f"{-r_opt['μείωση_u_max_%']:.2f} %", delta_color="inverse", border=True)
            st.metric("Ενεργός Τιμή RMS", f"{r_opt['RMS_με']:.4f} m", delta=f"{-r_opt['μείωση_RMS_%']:.2f} %", delta_color="inverse", border=True)
        
        # --- ΓΡΑΦΗΜΑΤΑ ΧΡΟΝΟΪΣΤΟΡΙΑΣ ---
        # Το Response έχει .u (μετακίνηση κόμβου 2, πίνακας 1-Δ) και .stroke
        # (διαδρομή TMD, κόμβος 3 - κόμβος 2). Το core.history_frame τα βάζει σε πίνακα.
        df_hist = core.history_frame(res_baseline, res_tmd)

        st.subheader("Μετακίνηση κατασκευής", anchor=False)
        st.line_chart(df_hist, x="t (s)", y=["u χωρίς TMD (m)", "u με TMD (m)"],
                      x_label="Χρόνος (s)", y_label="Μετακίνηση u (m)",
                      color=["#9aa0a6", "#d62728"])

        st.subheader("Σχετική διαδρομή TMD", anchor=False)
        st.line_chart(df_hist, x="t (s)", y="διαδρομή TMD (m)",
                      x_label="Χρόνος (s)", y_label="u₃ − u₂ (m)")