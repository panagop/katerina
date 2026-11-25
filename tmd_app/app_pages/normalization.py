import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import math

import core

# --- 1. ΑΝΑΓΝΩΣΗ ΑΡΧΕΙΩΝ PEER (.at2) ---
def read_peer_at2(filepath):
    """Διαβάζει αρχείο .at2 με το core.parse_peer_at2 (ίδιο με τη σελίδα «Διέγερση»).

    Τα dt και NPTS διαβάζονται από την επικεφαλίδα, στην αρχή ή στο τέλος του αρχείου.
    Αρχείο χωρίς επικεφαλίδα δίνει ValueError (δεν μαντεύουμε dt).
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        dt, a_g = core.parse_peer_at2(f.read())

    # Φτιάχνουμε τον άξονα του χρόνου: [0, dt, 2*dt, 3*dt, ...]
    t = np.arange(len(a_g)) * dt

    return t, a_g

# --- 2. ΣΤΗΣΙΜΟ ΕΦΑΡΜΟΓΗΣ (UI) ---
st.set_page_config(page_title="Κλιμάκωση Σεισμών", layout="wide")
st.title("Ταυτόχρονη προβολή σεισμών με κοινό PGA")


# --- 3. ΑΝΑΓΝΩΣΗ ΦΑΚΕΛΟΥ & ΕΠΙΛΟΓΗ ---
# Ορίζεις ΤΟΝ ΦΑΚΕΛΟ που έχει μέσα όλα τα .at2
# Σχετικά με τη θέση αυτού του αρχείου (tmd_app/app_pages/), ώστε να δουλεύει σε κάθε υπολογιστή
folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "aug26", "GM_X")

# Βρίσκει αυτόματα ΟΛΑ τα αρχεία .at2 μέσα στον φάκελο
all_at2_files = glob.glob(os.path.join(folder_path, "*.at2"))

if not all_at2_files:
    st.warning(f"Δεν βρέθηκαν αρχεία .at2 στον φάκελο: {folder_path}. Ελέγξτε τη διαδρομή.")
else:
    # Φτιάχνει ένα λεξικό { "Ονομα αρχείου χωρίς .at2" : "Πλήρες path" }
    available_seismoi = {}
    for filepath in all_at2_files:
        name = os.path.basename(filepath).replace(".at2", "")
        available_seismoi[name] = filepath

    st.markdown("### 🗂️ Επιλογή Επιταχυνσιογραφημάτων")
    selected_names = st.multiselect(
        "Διάλεξε ποιους σεισμούς θέλεις να φορτώσεις από τον φάκελο:",
        options=list(available_seismoi.keys()),
        default=list(available_seismoi.keys())[:3] if len(available_seismoi) >= 3 else list(available_seismoi.keys())
    )

    # Κρατάμε μόνο αυτούς που επέλεξες στο μενού
    seismoi_to_plot = {name: available_seismoi[name] for name in selected_names}


    # --- 4. ΣΤΟΧΟΣ PGA ---
    # --- 4. ΣΤΟΧΟΣ PGA, ΦΙΛΤΡΑ & ΧΡΟΝΟΣ ---
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_pga = st.number_input("Στόχος PGA (g)", value=0.36, step=0.01)
        
    with col2:
        min_sf, max_sf = st.slider(
            "Επιτρεπτά όρια Scale Factor:",
            min_value=0.1, max_value=4.0, value=(0.10, 4.0), step=0.1,
            help="Αποκλεισμός σεισμών που απαιτούν ακραία κλιμάκωση."
        )
        
    with col3:
        # Προσθήκη slider για τον άξονα του χρόνου (π.χ. από 0 έως 100 sec)
        time_range = st.slider(
            "Χρονικό Παράθυρο Προβολής (sec):",
            min_value=0.0, max_value=200.0, value=(0.0, 50.0), step=1.0,
            help="Εστίασε στο τμήμα του σεισμού που σε ενδιαφέρει (π.χ. ισχυρή κίνηση)."
        )

   # --- 5. ΕΠΕΞΕΡΓΑΣΙΑ (ΓΡΑΦΗΜΑ & ΠΙΝΑΚΑΣ) ---
    table_data = []
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted_count = 0  # Μετρητής για το πόσοι σχεδιάστηκαν

    # ΠΡΟΣΟΧΗ: Πλέον η λούπα τρέχει σε ΟΛΟΥΣ τους σεισμούς του φακέλου (available_seismoi)
    for onoma, arxeio in available_seismoi.items():
        try:
            t, a_g = read_peer_at2(arxeio)
            current_pga = np.max(np.abs(a_g))
            scale_factor = target_pga / current_pga
            
            # ΕΛΕΓΧΟΣ 1 (Για τον Πίνακα): Περνάει το φίλτρο με τις μπάρες;
            if min_sf <= scale_factor <= max_sf:
                
                # Μπαίνει στον πίνακα!
                table_data.append({
                    "Σεισμός": onoma,
                    "Αρχικό PGA (g)": current_pga,
                    "Πολλαπλασιαστής (Scale Factor)": scale_factor,
                    "Τελικό PGA (g)": target_pga
                })
                
                # ΕΛΕΓΧΟΣ 2 (Για το Γράφημα): Τον έχεις επιλέξει από το μενού;
                if onoma in selected_names:
                    a_scaled = a_g * scale_factor              
                    ax.plot(t, a_scaled, label=f"{onoma} (Scale: {scale_factor:.2f})", alpha=0.75, linewidth=1.2)
                    plotted_count += 1

        except Exception as e:
            pass # Αγνοούμε σιωπηλά όσα αρχεία δεν διαβάζονται

    # --- 6. ΕΠΕΞΕΡΓΑΣΙΑ (ΓΡΑΦΗΜΑ & ΠΙΝΑΚΑΣ) ---
    table_data = []
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted_count = 0 

# --- ΠΡΟ-ΣΑΡΩΣΗ ΔΕΔΟΜΕΝΩΝ ΚΑΙ ΑΠΟΘΗΚΕΥΣΗ ΣΤΗ ΜΝΗΜΗ ---
    parsed_data = {}
    global_max_sf = 0.0
    global_max_dur = 0.0

    for onoma, arxeio in available_seismoi.items():
        try:
            t, a_g = read_peer_at2(arxeio)
            parsed_data[onoma] = (t, a_g) # Η κρίσιμη αποθήκευση στη μνήμη!
            
            # Εύρεση μέγιστης διάρκειας (για να ρυθμιστεί το slider του χρόνου)
            dur = t[-1]
            if dur > global_max_dur:
                global_max_dur = dur
                
            # Εύρεση μέγιστου scale factor (για να ρυθμιστεί το slider κλίμακας)
            current_pga = np.max(np.abs(a_g))
            sf = target_pga / current_pga
            if sf > global_max_sf:
                global_max_sf = sf
        except:
            pass # Αγνοεί τυχόν προβληματικά αρχεία χωρίς να "σκάει"
        
    for onoma, (t, a_g) in parsed_data.items():
        
        current_pga = np.max(np.abs(a_g))
        scale_factor = target_pga / current_pga
        
        # --- Υπολογισμός Διάρκειας και Βήματος (dt) ---
        duration = t[-1]
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        
        # ΕΛΕΓΧΟΣ: Περνάει το φίλτρο του Scale Factor;
        if min_sf <= scale_factor <= max_sf:
            
            # --- ΑΠΟΘΗΚΕΥΣΗ ΣΤΟΝ ΠΙΝΑΚΑ (Εδώ μπαίνουν η Διάρκεια και το dt) ---
            table_data.append({
                "Σεισμός": onoma,
                "Διάρκεια (sec)": duration,
                "Βήμα dt (sec)": dt,
                "Αρχικό PGA (g)": current_pga,
                "Πολλαπλασιαστής (Scale Factor)": scale_factor,
                "Τελικό PGA (g)": target_pga
            })
            
            # ΕΛΕΓΧΟΣ: Τον έχεις επιλέξει στο μενού για να ζωγραφιστεί;
            if onoma in selected_names:
                a_scaled = a_g * scale_factor              
                ax.plot(t, a_scaled, label=f"{onoma} (Scale: {scale_factor:.2f})", alpha=0.75, linewidth=1.2)
                plotted_count += 1


    # --- 7. ΕΜΦΑΝΙΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ---
    
    # Εμφάνιση Γραφήματος
    if plotted_count > 0:
        ax.set_xlabel("Χρόνος (s)")
        ax.set_ylabel("Επιτάχυνση (g)")
        ax.axhline(target_pga, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label="Στόχος PGA")
        ax.axhline(-target_pga, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        
        # Εφαρμογή του χρονικού παραθύρου (zoom)
        ax.set_xlim(time_range[0], time_range[1])
        
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout() 
        st.pyplot(fig) 
    elif selected_names:
        st.warning("Κανένας από τους επιλεγμένους σεισμούς δεν πέρασε το φίλτρο της κλίμακας (Scale Factor).")

    # Εμφάνιση Πίνακα
    st.divider()
    st.markdown(f"### 📊 Έγκυροι Σεισμοί ({len(table_data)} συνολικά)")
    st.caption("Ο πίνακας δείχνει όλους τους σεισμούς του φακέλου που ικανοποιούν τα κριτήρια της κλίμακας, ανεξάρτητα από το αν εμφανίζονται στο γράφημα.")

    if table_data:
        df = pd.DataFrame(table_data)
        
        # Μορφοποίηση και εμφάνιση ΟΛΩΝ των στηλών
        st.dataframe(
            df.style.format({
                "Διάρκεια (sec)": "{:.2f}",
                "Βήμα dt (sec)": "{:.4f}",
                "Αρχικό PGA (g)": "{:.4f}",
                "Πολλαπλασιαστής (Scale Factor)": "{:.4f}",
                "Τελικό PGA (g)": "{:.2f}"
            }),
            use_container_width=True
        )