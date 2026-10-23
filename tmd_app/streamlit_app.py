"""
Εφαρμογή Streamlit για SDOF με αποσβεστήρα συντονισμένης μάζας (TMD).

    cd tmd_app
    uv run --project .. streamlit run streamlit_app.py
"""

import streamlit as st

import core

st.set_page_config(page_title="TMD σε SDOF", page_icon=":material/vibration:", layout="wide")

core.init_state()

page = st.navigation(
    [
        st.Page("app_pages/excitation.py", title="Διέγερση", icon=":material/earthquake:"),
        st.Page("app_pages/model.py", title="Κατασκευή & TMD", icon=":material/tune:"),
        st.Page("app_pages/analysis.py", title="Ανάλυση", icon=":material/show_chart:"),
        st.Page("app_pages/parametric.py", title="Παραμετρική", icon=":material/stacked_line_chart:"),
    ],
    position="top",
)

st.title(page.title, anchor=False)
page.run()

# Σύνοψη των τρεχουσών επιλογών, ορατή από κάθε σελίδα.
# Γράφεται μετά τη σελίδα, ώστε να δείχνει τις τιμές που μόλις άλλαξαν.
with st.sidebar:
    structure = core.current_structure()
    tmd = core.current_tmd(structure)
    st.subheader("Τρέχουσες επιλογές")
    st.markdown(
        f"**Διέγερση:** {core.current_excitation(structure).name}  \n"
        f"**Κατασκευή:** m = {structure.m:g} tn, T = {structure.T:g} s, ξ = {structure.ξ:g}  \n"
        f"**TMD:** m̂ = {tmd.m_hat:g}, f = {tmd.f:.4f}, ξd = {tmd.ξd:.4f}"
        + ("" if st.session_state.tmd_optimum else "  \n:orange[μη βέλτιστο TMD]")
    )
    st.caption("Μονάδες: tn, kN/m, m, s. Μοντέλο OpenSeesPy, Newmark μέσης επιτάχυνσης.")
