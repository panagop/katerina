"""Επιλογή της διέγερσης βάσης: El Centro, αρχείο του χρήστη ή αρμονική."""

import numpy as np
import pandas as pd
import streamlit as st

import core


SOURCES = ["El Centro", "Αρχείο", "Αρμονική"]
KIND_TO_SOURCE = {"elcentro": "El Centro", "file": "Αρχείο", "harmonic": "Αρμονική"}

spec = st.session_state.excitation_spec
structure = core.current_structure()

#Εδώ δημιουργείται ένα οριζόντιο μενού (σαν διακόπτης με 3 κουμπιά
source = st.segmented_control(
    "Πηγή διέγερσης",
    SOURCES,
    default=KIND_TO_SOURCE[spec["kind"]], #δες πανω το kind_to_source
    required=True,
    key="exc_source",
)

#αν πατησεις το κουμπί "El Centro" εμφανειζεται στην οθονη
if source == "El Centro":
    st.caption("Επιταχυνσιογράφημα El Centro (1940, N-S) από το `aug26/elCentro.txt`, τιμές σε g.")
    st.session_state.excitation_spec = {"kind": "elcentro"}

# αν πατησεις το κουμπί "Αρχείο" εμφανίζεται στην οθονη
elif source == "Αρχείο":
    with st.container(border=True):
        upload = st.file_uploader("Αρχείο επιταχυνσιογραφήματος (.txt, .dat, .csv, .at2)",
                                  type=["txt", "dat", "csv", "at2", "acc"])
        c1, c2, c3, c4 = st.columns(4)
        # Για αρχεία .at2 προεπιλέγεται η μορφή PEER (dt και NPTS από την επικεφαλίδα)
        LAYOUTS = ["t, a", "a", "πολλές", "PEER .at2"]
        is_at2 = upload is not None and upload.name.lower().endswith(".at2")
        layout = c1.selectbox("Μορφή", LAYOUTS, index=3 if is_at2 else 0,
                              help="«t, a»: δύο στήλες χρόνος-επιτάχυνση. "
                                   "«a»: μία στήλη. «πολλές»: πολλές τιμές ανά γραμμή. "
                                   "«PEER .at2»: αρχείο PEER NGA, το dt διαβάζεται από "
                                   "την επικεφαλίδα (στην αρχή ή στο τέλος του αρχείου).")
        peer = layout == "PEER .at2"
        unit = c2.selectbox("Μονάδες αρχείου", list(core.UNITS), disabled=peer,
                            help="Τα αρχεία PEER είναι πάντα σε g.")
        if peer:
            unit = "g"
        skiprows = c3.number_input("Γραμμές επικεφαλίδας", 0, 200, 0, step=1, disabled=peer)
        dt_in = c4.number_input("dt (s)", 0.0001, 1.0, 0.01, step=0.005, format="%.4f",
                                disabled=layout in ("t, a", "PEER .at2"),
                                help="Χρειάζεται μόνο όταν το αρχείο δεν έχει στήλη χρόνου.")
        scale = st.number_input("Συντελεστής κλίμακας", 0.0, 100.0, 1.0, step=0.1,
                                help="Πολλαπλασιάζει την καταγραφή (π.χ. για κλιμάκωση σε PGA στόχο).")

    if upload is not None:
        try:
            dt, a_raw = core.parse_accelerogram(
                upload.getvalue(), int(skiprows), layout,
                None if layout in ("t, a", "PEER .at2") else dt_in
            )
        except ValueError as e:
            # Χωρίς st.stop(): θα σταματούσε και τη σύνοψη της πλαϊνής στήλης
            st.error(f"{e} Η ανάλυση χρησιμοποιεί την προηγούμενη διέγερση.",
                     title="Το αρχείο δεν διαβάστηκε")
        else:
            st.session_state.excitation_spec = {
                "kind": "file",
                "name": upload.name.rsplit(".", 1)[0],
                "dt": dt,
                "a": a_raw * core.UNITS[unit] * scale,
            }
    elif spec["kind"] == "file":
        st.info(f"Χρησιμοποιείται το αρχείο που φορτώθηκε προηγουμένως: **{spec['name']}**.")
    else:
        st.info("Ανεβάστε αρχείο. Μέχρι τότε η ανάλυση χρησιμοποιεί την προηγούμενη διέγερση.")

else:  # Αρμονική
    st.caption(
        "a(t) = A·sin(ρ·ω·t), με A = 2ξk/m: για ρ = 1 η μόνιμη απόκριση της "
        "κατασκευής χωρίς TMD έχει πλάτος 1.00 m. Παράγεται με το τρέχον ω της κατασκευής."
    )
    old = spec if spec["kind"] == "harmonic" else {"rho": 1.0, "duration": 60.0, "dt": 0.01}
    c1, c2, c3 = st.columns(3)
    rho = c1.number_input("ρ = Ω/ω", 0.1, 5.0, float(old["rho"]), step=0.05, format="%.3f")
    duration = c2.number_input("Διάρκεια (s)", 1.0, 300.0, float(old["duration"]), step=10.0)
    dt = c3.number_input("dt (s)", 0.001, 0.05, float(old["dt"]), step=0.005, format="%.3f")
    st.session_state.excitation_spec = {"kind": "harmonic", "rho": rho, "duration": duration, "dt": dt}

# --------------------------------------------------------------------------
# Η διέγερση που θα χρησιμοποιηθεί
# --------------------------------------------------------------------------

ground = core.current_excitation(structure)

# Βρίσκουμε το μέγιστο PGA στο αρχικό σύστημα (συνήθως m/s²) και το μετατρέπουμε σε g
pga_raw = float(np.max(np.abs(ground.a)))
pga_g = pga_raw / core.exc.G

with st.container(horizontal=True):
    # Τώρα τα metrics δείχνουν κατευθείαν g 
    st.metric("PGA (g)", f"{pga_g:.3f} g", border=True)
    st.metric("PGA (m/s²)", f"{pga_raw:.3f} m/s²", border=True) # Το κρατάμε βοηθητικά αν θες
    st.metric("dt", f"{ground.dt:g} s", border=True)
    st.metric("Διάρκεια", f"{ground.t_final:.2f} s", border=True)
    st.metric("Σημεία", f"{len(ground.a)}", border=True)

# Δείκτης για αρμονική με πολύ αραιό βήμα σε σχέση με την περίοδο
spec = st.session_state.excitation_spec
T_exc_min = structure.T / spec["rho"] if spec["kind"] == "harmonic" else structure.T
if ground.dt > T_exc_min / 10:
    st.warning(f"Το dt = {ground.dt:g} s είναι μεγαλύτερο από T/10 = {T_exc_min / 10:.3f} s: "
               "η ολοκλήρωση Newmark θα έχει αισθητό σφάλμα περιόδου.")

# Μετατροπή ολόκληρου του array της επιτάχυνσης σε g για το γράφημα
a_in_g = ground.a / core.exc.G

# Φτιάχνουμε το DataFrame με τα δεδομένα σε g
df = pd.DataFrame({"t (s)": ground.t, "Επιτάχυνση (g)": a_in_g})

# Σχεδιασμός γραφήματος με τους νέους άξονες
st.line_chart(
    df, 
    x="t (s)", 
    y="Επιτάχυνση (g)", 
    x_label="Χρόνος (s)", 
    y_label="Επιτάχυνση (g)",
    alt=f"Επιταχυνσιογράφημα {ground.name} (σε g)"
)