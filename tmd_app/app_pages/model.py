"""Παράμετροι της κατασκευής και σχεδιασμός του TMD (Tsai & Lin, 1993)."""

import numpy as np
import pandas as pd
import streamlit as st

import core

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("Κατασκευή (SDOF)", anchor=False)
        st.number_input("Μάζα m (tn)", 0.001, 1e6, key="m", step=0.1, format="%.3f",
                        persist_state="session")
        st.number_input("Ιδιοπερίοδος T (s)", 0.02, 10.0, key="T", step=0.01, format="%.3f",
                        persist_state="session")
        st.number_input("Λόγος απόσβεσης ξ", 0.0, 0.3, key="xi", step=0.005, format="%.3f",
                        persist_state="session")

with right:
    with st.container(border=True):
        st.subheader("TMD", anchor=False)
        st.number_input("Λόγος μαζών m̂ = md/m", 0.001, 0.5, key="m_hat", step=0.005,
                        format="%.3f", persist_state="session")
        st.toggle("Βέλτιστες τιμές Tsai & Lin", key="tmd_optimum", persist_state="session")
        c1, c2 = st.columns(2)
        manual = not st.session_state.tmd_optimum
        c1.number_input("f = ωd/ω", 0.5, 1.5, key="f_manual", step=0.01, format="%.4f",
                        disabled=not manual, persist_state="session")
        c2.number_input("ξd", 0.0, 0.5, key="xid_manual", step=0.005, format="%.4f",
                        disabled=not manual, persist_state="session")

structure = core.current_structure()
tmd = core.current_tmd(structure)
f_opt = float(core.tsai_lin_f(tmd.m_hat, structure.ξ))
xid_opt = float(core.tsai_lin_xi(tmd.m_hat, structure.ξ))

st.subheader("Παράμετροι που προκύπτουν", anchor=False)
c1, c2 = st.columns(2)
c1.table(pd.DataFrame(
    {
        "Μέγεθος": ["ω (rad/s)", "k (kN/m)", "c (kN·s/m)"],
        "Τιμή": [f"{structure.ω:.4f}", f"{structure.k:.3f}", f"{structure.c:.4f}"],
    }
).set_index("Μέγεθος"), alt="Παράμετροι κατασκευής")
c2.table(pd.DataFrame(
    {
        "Μέγεθος": ["md (tn)", "f = ωd/ω", "ξd", "ωd (rad/s)", "Td (s)", "kd (kN/m)", "cd (kN·s/m)"],
        "Τιμή": [f"{tmd.md:.4f}", f"{tmd.f:.4f}", f"{tmd.ξd:.4f}", f"{tmd.ωd:.4f}",
                 f"{2 * np.pi / tmd.ωd:.4f}", f"{tmd.kd:.4f}", f"{tmd.cd:.5f}"],
    }
).set_index("Μέγεθος"), alt="Παράμετροι TMD")

if not st.session_state.tmd_optimum:
    st.caption(f"Βέλτιστες τιμές για σύγκριση: f_opt = {f_opt:.4f}, ξd,opt = {xid_opt:.4f} "
               f"(f/f_opt = {tmd.f / f_opt:.3f}).")

# Οι σχέσεις Tsai & Lin συναρτήσει του m̂, για το τρέχον ξ (όπως στο Tsai&Lin.py)
st.subheader("Βέλτιστες παράμετροι Tsai & Lin", anchor=False)
m_hats = np.linspace(0.001, 0.2, 200)
curves = pd.DataFrame({
    "m̂": m_hats,
    "f_opt": core.tsai_lin_f(m_hats, structure.ξ),
    "ξd,opt": core.tsai_lin_xi(m_hats, structure.ξ),
})
c1, c2 = st.columns(2)
c1.line_chart(curves, x="m̂", y="f_opt", x_label="Λόγος μαζών m̂", y_label="f_opt = ωd/ω",
              alt="Βέλτιστος λόγος συχνοτήτων")
c2.line_chart(curves, x="m̂", y="ξd,opt", x_label="Λόγος μαζών m̂", y_label="ξd,opt",
              alt="Βέλτιστος λόγος απόσβεσης TMD")
st.caption(f"Για ξ = {structure.ξ:g}. Οι σχέσεις έχουν διατυπωθεί για διέγερση στήριξης, "
           "όπως εφαρμόζεται η διέγερση στο μοντέλο.")
