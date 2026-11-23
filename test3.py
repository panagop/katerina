# ==========================================================================
# Πλήρης Αναζήτηση (Grid Search) Βέλτιστων Παραμέτρων (f, ξd) ανά m̂
# ==========================================================================

with tab_opt:
    st.markdown("### Αριθμητική εύρεση βέλτιστων παραμέτρων TMD")
    st.caption(f"Διέγερση: **{ground.name}**. Επιλέξτε την απόσβεση του κτιρίου και σαρώστε "
               f"f και ξd για να βρείτε το ιδανικό κούρδισμα.")

    with st.form("opt_search_form"):
        # Το ξ μπαίνει πάνω από τις στήλες
        st.markdown("**Γενική Παράμετρος Κτιρίου**")
        user_xi = st.number_input("Απόσβεση Κτιρίου ξ (π.χ. 0.05 για 5%)", value=0.05, min_value=0.00, max_value=0.20, step=0.01, format="%.2f")
        
        st.divider() # Μια διακριτική γραμμή διαχωρισμού
        
        # Χωρίζουμε τα υπόλοιπα 3 πεδία αναζήτησης σε 3 στήλες
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("**Λόγος μαζών m̂**")
            opt_m_lo = st.number_input("Από (m)", value=0.01, step=0.01, format="%.3f")
            opt_m_hi = st.number_input("Έως (m)", value=0.10, step=0.01, format="%.3f")
            opt_m_n = st.number_input("Πλήθος (m)", 2, 50, 10, step=1)
        with c2:
            st.markdown("**Λόγος f**")
            opt_f_lo = st.number_input("Από (f)", value=0.70, step=0.05, format="%.3f")
            opt_f_hi = st.number_input("Έως (f)", value=1.20, step=0.05, format="%.3f")
            opt_f_n = st.number_input("Πλήθος (f)", 2, 100, 20, step=1)
        with c3:
            st.markdown("**Απόσβεση TMD ξd**")
            opt_x_lo = st.number_input("Από (ξd)", value=0.01, step=0.01, format="%.3f")
            opt_x_hi = st.number_input("Έως (ξd)", value=0.25, step=0.01, format="%.3f")
            opt_x_n = st.number_input("Πλήθος (ξd)", 2, 100, 20, step=1)
        
        run_opt = st.form_submit_button("Έναρξη Αναζήτησης", type="primary", icon=":material/search:")

    if run_opt:
        import dataclasses
        
        # Δημιουργούμε ένα προσωρινό αντίγραφο του κτιρίου με το νέο ξ του χρήστη
        temp_structure = dataclasses.replace(structure, ξ=user_xi)
        
        m_vals = np.linspace(opt_m_lo, opt_m_hi, int(opt_m_n))
        f_vals = np.linspace(opt_f_lo, opt_f_hi, int(opt_f_n))
        x_vals = np.linspace(opt_x_lo, opt_x_hi, int(opt_x_n))
        
        total_runs = len(m_vals) * len(f_vals) * len(x_vals)
        st.info(f"Εκτέλεση {total_runs} αναλύσεων με απόσβεση κτιρίου ξ = {user_xi:.2f}...")
        bar = st.progress(0.0, text="Αναζήτηση βέλτιστων...")
        
        # Υπολογισμός του baseline με το νέο temp_structure
        baseline = core.analyse(temp_structure, ground)
        best_results = []
        counter = 0

        # Τριπλός βρόχος
        for mh in m_vals:
            best_u = float('inf')
            best_row = None
            
            for f_val in f_vals:
                for x_val in x_vals:
                    # Χρησιμοποιούμε το temp_structure παντού εδώ
                    tmd = core.design_tmd(temp_structure, mh, f=f_val, ξd=x_val)
                    row = core.results.compare(baseline, core.analyse(temp_structure, ground, tmd))
                    
                    if row["u_max_με"] < best_u:
                        best_u = row["u_max_με"]
                        best_row = {"m̂": mh, "Αριθμητικό_f_opt": f_val, "Αριθμητικό_ξd_opt": x_val, **row}
                    
                    counter += 1
                    if counter % 100 == 0: 
                        bar.progress(counter / total_runs, text=f"m̂ = {mh:.3f} | Αναλύθηκαν {counter}/{total_runs}")
            
            best_results.append(best_row)

        bar.empty()
        st.session_state["opt_result"] = (pd.DataFrame(best_results), ground.name, user_xi)

    stored_opt = st.session_state.get("opt_result")
    if stored_opt is not None:
        df_opt, label, saved_xi = stored_opt
        st.success(f"Η αναζήτηση ολοκληρώθηκε για τη διέγερση: {label} (με ξ = {saved_xi:.2f})")
        
        # Γραφήματα Βέλτιστων
        c1, c2 = st.columns(2)
        c1.line_chart(df_opt, x="m̂", y="Αριθμητικό_f_opt", x_label="Λόγος μαζών m̂", y_label="Βέλτιστο f", color="#ff7f0e")
        c2.line_chart(df_opt, x="m̂", y="Αριθμητικό_ξd_opt", x_label="Λόγος μαζών m̂", y_label="Βέλτιστο ξd", color="#2ca02c")
        
        st.dataframe(df_opt, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.4f") for c in df_opt.columns if df_opt[c].dtype.kind == "f"})
        st.download_button("Λήψη Βέλτιστων Τιμών (csv)", core.to_csv(df_opt), file_name=f"optimal_parameters_{ground.name}.csv", mime="text/csv", icon=":material/download:")

        # --- 1. ΕΞΑΓΩΓΗ ΕΞΙΣΩΣΕΩΝ ---
        st.divider()
        st.markdown("### 📈 Εξαγωγή Εμπειρικών Εξισώσεων (Μορφή Tsai & Lin)")
        
        m_arr = df_opt["m̂"].values
        f_arr = df_opt["Αριθμητικό_f_opt"].values
        x_arr = df_opt["Αριθμητικό_ξd_opt"].values
        
        def eq_template_f(m, a, b, c):
            return a * (np.sqrt(1 - 0.5 * m) / (1 + m)) + b * np.sqrt(m) + c
            
        def eq_template_x(m, a, b, c):
            return a * np.sqrt((3 * m) / (8 * (1 + m) * (1 - 0.5 * m))) + b * m + c
        
        try:
            import scipy.optimize as opt
            popt_f, _ = opt.curve_fit(eq_template_f, m_arr, f_arr)
            popt_x, _ = opt.curve_fit(eq_template_x, m_arr, x_arr)
            
            def format_f_eq(p):
                sign_b = "+" if p[1] >= 0 else "-"
                sign_c = "+" if p[2] >= 0 else "-"
                return f"f_{{opt}} = {p[0]:.4f} \\left( \\frac{{\\sqrt{{1 - 0.5\\bar{{m}}}}}}{{1 + \\bar{{m}}}} \\right) {sign_b} {abs(p[1]):.4f} \\sqrt{{\\bar{{m}}}} {sign_c} {abs(p[2]):.4f}"
                
            def format_x_eq(p):
                sign_b = "+" if p[1] >= 0 else "-"
                sign_c = "+" if p[2] >= 0 else "-"
                return f"\\xi_{{d,opt}} = {p[0]:.4f} \\sqrt{{ \\frac{{3\\bar{{m}}}}{{8(1+\\bar{{m}})(1-0.5\\bar{{m}})}} }} {sign_b} {abs(p[1]):.4f} \\bar{{m}} {sign_c} {abs(p[2]):.4f}"
            
            st.latex(format_f_eq(popt_f))
            st.latex(format_x_eq(popt_x))
            st.info(f"💡 Οι εξισώσεις υπολογίστηκαν για σταθερή απόσβεση κτιρίου ξ = {saved_xi:.2f}.")
        except Exception as e:
            st.error(f"Το Curve Fitting δεν ολοκληρώθηκε: {e}")

      
      # --- 2. ΙΣΤΟΡΙΚΟ ΧΡΟΝΟΥ (ΓΡΑΦΗΜΑ ΜΕ ΤΑ ΑΡΙΘΜΗΤΙΚΑ ΒΕΛΤΙΣΤΑ) ---
        st.divider()
        st.markdown("### ⏱️ Έλεγχος Σεισμικής Απόκρισης στο Χρόνο")
        st.caption("Το πρόγραμμα έχει προεπιλέξει αυτόματα τον λόγο μαζών που πέτυχε τη **μέγιστη μείωση** μετακίνησης. Χρησιμοποιούνται τα αριθμητικά βέλτιστα.")
        
        # Εύρεση της "καλύτερης" μάζας για προεπιλογή
        best_overall_idx = int(df_opt["u_max_με"].idxmin())
        best_overall_m = df_opt.loc[best_overall_idx, "m̂"]
        default_dropdown_idx = list(m_arr).index(best_overall_m)
        
        selected_m = st.selectbox(
            "Επιλέξτε Λόγο Μαζών (m̂):", 
            m_arr, 
            index=default_dropdown_idx, 
            format_func=lambda x: f"{x:.3f}"
        )
        
        # Ανάκτηση των βέλτιστων για την επιλεγμένη μάζα
        idx = list(m_arr).index(selected_m)
        f_num = f_arr[idx]
        x_num = x_arr[idx]
        
        if selected_m == best_overall_m:
            st.success(f"🏆 Αυτή η μάζα (m̂ = {selected_m:.3f}) έδωσε το καλύτερο αποτέλεσμα! Εφαρμόζονται: **f = {f_num:.4f}** και **ξd = {x_num:.4f}**")
        else:
            st.write(f"Εφαρμόζονται τα βέλτιστα: **f = {f_num:.4f}** και **ξd = {x_num:.4f}**")
        
        # --- Ανάλυση με τα νέα δεδομένα ---
        import dataclasses
        temp_structure2 = dataclasses.replace(structure, ξ=saved_xi)
        
        # Φτιάχνουμε το TMD αυστηρά με τα δικά σου νούμερα
        tmd_num = core.design_tmd(temp_structure2, selected_m, f=f_num, ξd=x_num)
        
        with st.spinner("Ανάλυση στο OpenSees με τα αριθμητικά βέλτιστα..."):
            res_baseline = core.analyse(temp_structure2, ground)
            res_tmd = core.analyse(temp_structure2, ground, tmd_num)
        
        # Χρήση της δικής σου ρουτίνας για τα γραφήματα
        df_hist = core.history_frame(res_baseline, res_tmd)
        
        st.subheader("Μετακίνηση κατασκευής", anchor=False)
        st.line_chart(df_hist, x="t (s)", y=["u χωρίς TMD (m)", "u με TMD (m)"],
                      x_label="Χρόνος (s)", y_label="Μετακίνηση u (m)",
                      color=["#9aa0a6", "#d62728"],
                      alt="Μετακίνηση της κατασκευής χωρίς και με TMD (Αριθμητικά βέλτιστα)")
        
        st.subheader("Σχετική διαδρομή TMD", anchor=False)
        st.line_chart(df_hist, x="t (s)", y="διαδρομή TMD (m)",
                      x_label="Χρόνος (s)", y_label="u₃ − u₂ (m)",
                      alt="Σχετική μετακίνηση της μάζας του TMD")