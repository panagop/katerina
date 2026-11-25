# ==========================================================================
# Πλήρης Αναζήτηση (Grid Search) Βέλτιστων Παραμέτρων (f, ξd) ανά m̂
# ==========================================================================

with tab_opt:
    st.markdown("### Αριθμητική εύρεση βέλτιστων παραμέτρων TMD")
    st.caption(f"Διέγερση: **{ground.name}**. Για κάθε λόγο μάζας m̂, το πρόγραμμα σαρώνει όλους τους συνδυασμούς "
               f"f και ξd για να βρει το μοναδικό ζευγάρι που ελαχιστοποιεί τη μέγιστη μετακίνηση του κτιρίου.")

    with st.form("opt_search_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Λόγος μαζών m̂**")
            opt_m_lo = st.number_input("Από", value=0.01, step=0.01, format="%.3f", key="opt_m_lo")
            opt_m_hi = st.number_input("Έως", value=0.10, step=0.01, format="%.3f", key="opt_m_hi")
            opt_m_n = st.number_input("Πλήθος", 2, 50, 10, step=1, key="opt_m_n")
        with c2:
            st.markdown("**Λόγος συχνοτήτων f**")
            opt_f_lo = st.number_input("Από", value=0.70, step=0.05, format="%.3f", key="opt_f_lo")
            opt_f_hi = st.number_input("Έως", value=1.20, step=0.05, format="%.3f", key="opt_f_hi")
            opt_f_n = st.number_input("Πλήθος", 2, 100, 25, step=1, key="opt_f_n")
        with c3:
            st.markdown("**Απόσβεση TMD ξd**")
            opt_x_lo = st.number_input("Από", value=0.01, step=0.01, format="%.3f", key="opt_x_lo")
            opt_x_hi = st.number_input("Έως", value=0.25, step=0.01, format="%.3f", key="opt_x_hi")
            opt_x_n = st.number_input("Πλήθος", 2, 100, 25, step=1, key="opt_x_n")
        
        run_opt = st.form_submit_button("Έναρξη Αναζήτησης", type="primary", icon=":material/search:")

    if run_opt:
        m_vals = np.linspace(opt_m_lo, opt_m_hi, int(opt_m_n))
        f_vals = np.linspace(opt_f_lo, opt_f_hi, int(opt_f_n))
        x_vals = np.linspace(opt_x_lo, opt_x_hi, int(opt_x_n))
        
        total_runs = len(m_vals) * len(f_vals) * len(x_vals)
        st.info(f"Εκτέλεση {total_runs} αναλύσεων...")
        bar = st.progress(0.0, text="Αναζήτηση βέλτιστων...")
        
        baseline = core.analyse(structure, ground)
        best_results = []
        counter = 0

        # Τριπλός βρόχος (Triple Loop)
        for mh in m_vals:
            best_u = float('inf')
            best_row = None
            
            for f_val in f_vals:
                for x_val in x_vals:
                    # Σχεδιασμός TMD με το f και το ξd της τρέχουσας λούπας
                    tmd = core.design_tmd(structure, mh, f=f_val, ξd=x_val)
                    row = core.results.compare(baseline, core.analyse(structure, ground, tmd))
                    
                    # Αν βρήκαμε μικρότερη μετακίνηση, την αποθηκεύουμε ως το νέο "βέλτιστο"
                    if row["u_max_με"] < best_u:
                        best_u = row["u_max_με"]
                        # Αποθηκεύουμε τα f και ξd που βρήκαμε, μαζί με τα αποτελέσματα
                        best_row = {"m̂": mh, "Αριθμητικό_f_opt": f_val, "Αριθμητικό_ξd_opt": x_val, **row}
                    
                    counter += 1
                    if counter % 100 == 0:  # Ανανέωση της μπάρας κάθε 100 αναλύσεις
                        bar.progress(counter / total_runs, text=f"m̂ = {mh:.3f} | Αναλύθηκαν {counter}/{total_runs}")
            
            # Τέλος εσωτερικών λούπων: Προσθήκη του απόλυτου βέλτιστου για το συγκεκριμένο m̂
            best_results.append(best_row)

        bar.empty()
        st.session_state["opt_result"] = (pd.DataFrame(best_results), ground.name)

    stored_opt = st.session_state.get("opt_result")
    if stored_opt is not None:
        df_opt, label = stored_opt
        st.success(f"Η αναζήτηση ολοκληρώθηκε για τη διέγερση: {label}")
        
        # Γραφήματα σύγκρισης (m̂ vs f_opt και m̂ vs ξd_opt)
        c1, c2 = st.columns(2)
        c1.line_chart(df_opt, x="m̂", y="Αριθμητικό_f_opt", x_label="Λόγος μαζών m̂", y_label="Βέλτιστο f", color="#ff7f0e")
        c2.line_chart(df_opt, x="m̂", y="Αριθμητικό_ξd_opt", x_label="Λόγος μαζών m̂", y_label="Βέλτιστο ξd", color="#2ca02c")
        
        st.dataframe(df_opt, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.4f") for c in df_opt.columns if df_opt[c].dtype.kind == "f"})
        st.download_button("Λήψη Βέλτιστων Τιμών (csv)", core.to_csv(df_opt), file_name=f"optimal_parameters_{ground.name}.csv", mime="text/csv", icon=":material/download:")


        #-----------------------------------------------------------------------
# --- ΝΕΟ ΚΟΜΜΑΤΙ: ADVANCED CURVE FITTING (ΜΟΡΦΗ TSAI & LIN) ---
        st.divider()
        st.markdown("### 📈 Εξαγωγή Εμπειρικών Εξισώσεων (Μορφή Tsai & Lin)")
        
        m_arr = df_opt["m̂"].values
        f_arr = df_opt["Αριθμητικό_f_opt"].values
        x_arr = df_opt["Αριθμητικό_ξd_opt"].values
        
        # 1. Καλούπι για το f (βασισμένο στο πρώτο κλάσμα της εξίσωσης Tsai & Lin)
        def eq_template_f(m, a, b, c):
            term1 = np.sqrt(1 - 0.5 * m) / (1 + m)
            return a * term1 + b * np.sqrt(m) + c
            
        # 2. Καλούπι για το ξd (βασισμένο στη μεγάλη ρίζα της εξίσωσης Tsai & Lin)
        def eq_template_x(m, a, b, c):
            root_term = np.sqrt((3 * m) / (8 * (1 + m) * (1 - 0.5 * m)))
            return a * root_term + b * m + c
        
        try:
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
            
            st.info("💡 Οι εξισώσεις χρησιμοποιούν πλέον τον ακριβή μαθηματικό σκελετό των Tsai & Lin, προσαρμόζοντας τους συντελεστές στα δικά σας αριθμητικά δεδομένα!")
            
        except Exception as e:
            st.warning("Δεν υπήρχαν αρκετά δεδομένα για εξαγωγή εξίσωσης. Εκτελέστε πρώτα την αναζήτηση.")