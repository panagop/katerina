"""Ανάλυση χρονοϊστορίας: η κατασκευή χωρίς και με TMD για την επιλεγμένη διέγερση."""

import streamlit as st

import core

structure = core.current_structure()
tmd = core.current_tmd(structure)
ground = core.current_excitation(structure)

with st.spinner("Ανάλυση στο OpenSees..."):
    baseline = core.analyse(structure, ground)
    controlled = core.analyse(structure, ground, tmd)

r = core.results.compare(baseline, controlled)

# --------------------------------------------------------------------------
# Αποτελέσματα
# --------------------------------------------------------------------------

st.caption(f"Διέγερση: **{ground.name}**, {len(ground.a)} σημεία, dt = {ground.dt:g} s. "
           f"TMD: m̂ = {tmd.m_hat:g}, f = {tmd.f:.4f}, ξd = {tmd.ξd:.4f}.")

with st.container(horizontal=True):
    st.metric("u_max χωρίς TMD", f"{r['u_max_χωρίς']:.4f} m", border=True)
    st.metric("u_max με TMD", f"{r['u_max_με']:.4f} m",
              delta=f"{-r['μείωση_u_max_%']:.2f} %", delta_color="inverse", border=True)
    st.metric("RMS με TMD", f"{r['RMS_με']:.4f} m",
              delta=f"{-r['μείωση_RMS_%']:.2f} %", delta_color="inverse", border=True)
    st.metric("Μέγιστη διαδρομή TMD", f"{r['διαδρομή_TMD']:.4f} m",
              help="Μέγιστη σχετική μετακίνηση της μάζας του TMD ως προς την κατασκευή: "
                   "καθορίζει αν η συσκευή είναι κατασκευάσιμη.",
              border=True)

with st.expander("Όλα τα μεγέθη"):
    st.table(
        {
            "Μέγεθος": ["Μέγιστη μετακίνηση (m)", "RMS μετακίνησης (m)",
                        "Πλάτος τελευταίου 20 % (m)"],
            "Χωρίς TMD": [f"{r['u_max_χωρίς']:.4f}", f"{r['RMS_χωρίς']:.4f}", f"{r['u_ss_χωρίς']:.4f}"],
            "Με TMD": [f"{r['u_max_με']:.4f}", f"{r['RMS_με']:.4f}", f"{r['u_ss_με']:.4f}"],
            "Μείωση (%)": [f"{r['μείωση_u_max_%']:.2f}", f"{r['μείωση_RMS_%']:.2f}",
                           f"{r['μείωση_u_ss_%']:.2f}"],
        },
        alt="Πίνακας αποτελεσμάτων",
    )
    st.caption(f"Η διαδρομή του TMD είναι {r['λόγος_διαδρομής']:.1f} φορές η μέγιστη μετακίνηση "
               "της κατασκευής. Το πλάτος του τελευταίου 20 % είναι το πλάτος μόνιμης κατάστασης "
               "στην αρμονική διέγερση· στον σεισμό είναι μόνο πληροφοριακό.")

# --------------------------------------------------------------------------
# Χρονοϊστορίες
# --------------------------------------------------------------------------

df = core.history_frame(baseline, controlled)

st.subheader("Μετακίνηση κατασκευής", anchor=False)
st.line_chart(df, x="t (s)", y=["u χωρίς TMD (m)", "u με TMD (m)"],
              x_label="Χρόνος (s)", y_label="Μετακίνηση u (m)",
              color=["#9aa0a6", "#d62728"],
              alt="Μετακίνηση της κατασκευής χωρίς και με TMD")

st.subheader("Σχετική διαδρομή TMD", anchor=False)
st.line_chart(df, x="t (s)", y="διαδρομή TMD (m)",
              x_label="Χρόνος (s)", y_label="u₃ − u₂ (m)",
              alt="Σχετική μετακίνηση της μάζας του TMD")

st.download_button("Λήψη χρονοϊστοριών (csv)", core.to_csv(df),
                   file_name=f"tmd_{ground.name}_mhat{tmd.m_hat:g}.csv",
                   mime="text/csv", icon=":material/download:")
