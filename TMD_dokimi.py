import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Δεδομένα Κατασκευής
m = 1000            # Μάζα κατασκευής
ω = 2 * np.pi / 0.49   # Κυκλική ιδιοσυχνότητα κατασκευής (T=0.49s)
ξ = 0.02            # Απόσβεση κατασκευής
k = (ω**2) * m
c = 2 * ξ * ω * m

# Δεδομένα Αποσβεστήρα (TMD)
m_hat = 0.01        # Αναλογία μαζών (md / m) αρα md=10 
md = m_hat * m      # Μάζα TMD

# ΕΞΙΣΩΣΕΙΣ TSAI & LIN 
ξd = np.sqrt((3 * m_hat) / (8 * (1 + m_hat) * (1 - 0.5 * m_hat))) + (0.151 * ξ - 0.170 * ξ**2) + (0.163 * ξ + 4.980 * ξ**2) * m_hat
f_opt = ((np.sqrt(1 - 0.5 * m_hat) / (1 + m_hat)) + np.sqrt(1 - 2 * ξ**2) - 1) - ((2.375 - 1.034 * np.sqrt(m_hat) - 0.426 * m_hat) * ξ * np.sqrt(m_hat)) - ((3.730 - 16.903 * np.sqrt(m_hat) + 20.496 * m_hat) * (ξ**2) * np.sqrt(m_hat))

ωd = f_opt * ω            # Η πραγματική ιδιοσυχνότητα του TMD
kd = md * (ωd**2)         # Η πραγματική δυσκαμψία του ελατηρίου 
cd = 2 * ξd * ωd * md     # Ο πραγματικός συντελεστής απόσβεσης 


print(f"Βέλτιστος Λόγος f: {f_opt:.3f}")
print(f"Βέλτιστη Απόσβεση ξd: {ξd:.3f}")
print(f"Δυσκαμψία kd: {kd:.2f}, Απόσβεση cd: {cd:.2f}")

# 2. ΣΤΗΣΙΜΟ ΜΟΝΤΕΛΟΥ ΣΤΟ OPENSEES
ops.wipe()
ops.model('basic', '-ndm', 1, '-ndf', 1)  # Μοντέλο 1 Διάστασης, 1 Βαθμός Ελευθερίας ανά κόμβο (μετακίνηση στον Χ)

# Δημιουργία Κόμβων (Όλοι στην ίδια συντεταγμένη X=0, δεν παίζει ρόλο η γεωμετρία στα ελατήρια)
ops.node(1, 0.0)  # Έδαφος
ops.node(2, 0.0)  # Κατασκευή
ops.node(3, 0.0)  # TMD

# Στηρίξεις
ops.fix(1, 1)   # Πάκτωση στο έδαφος
# Εφαρμογή Μαζών
ops.mass(2, m)
ops.mass(3, md)

# 3. ΟΡΙΣΜΟΣ ΕΛΑΤΗΡΙΩΝ & ΑΠΟΣΒΕΣΤΗΡΩΝ
# Υλικά της Κατασκευής ( 1 για k,  2 για c)
ops.uniaxialMaterial('Elastic', 1, k)
ops.uniaxialMaterial('Viscous', 2, c, 1)  # Το 1 στο τέλος σημαίνει Γραμμική Απόσβεση (δύναμη ανάλογη της ταχύτητας)

# Υλικά του TMD ( 3 για k_d,  4 για c_d)
ops.uniaxialMaterial('Elastic', 3, kd)
ops.uniaxialMaterial('Viscous', 4, cd, 1)

# Σύνδεση κομβων-υλικων ,κανω στοιχεια zeroLength
# Έδαφος-Κατασκευή και παίρνει παράλληλα το υλικό 1 και 2
ops.element('zeroLength', 1, 1, 2, '-mat', 1, 2, '-dir', 1, 1)

# Συνδέει Κατασκευή με TMD και παίρνει παράλληλα το υλικό 3 και 4
ops.element('zeroLength', 2, 2, 3, '-mat', 3, 4, '-dir', 1, 1)

# 4. ΗΜΙΤΟΝΟΕΙΔΗΣ ΔΙΕΓΕΡΣΗ P = P_hat*sin(Ωt)
Ω = 1 * ω          # Συχνότητα της δύναμης  (έστω ρ = 1 ) Ω = ω*ρ
period_force = 2 * np.pi / Ω   # Ω = 2π / Τforce
p_hat = 1000                   # Πλάτος της δύναμης p^

# Το timeSeries 'Trig' δημιουργεί το ημίτονο: sin(Ωt)
ops.timeSeries('Trig', 1, 0.0, 999, period_force)
ops.pattern('Plain', 1, 1)

ops.load(2, p_hat) # Εφαρμόζουμε τη δύναμη στον κόμβο 2 (Κατασκευή)


#++++++++++++++++
# 5. RECORDERS (Καταγραφή Απotελεσμάτων)
# Το '-time' αποθηκεύει την 1η στήλη ως χρόνο. 
# Η 2η στήλη θα είναι η μετακίνηση του κόμβου στον 1ο βαθμό ελευθερίας (άξονας Χ).
ops.recorder('Node', '-file', 'disp_building.txt', '-time', '-node', 2, '-dof', 1, 'disp')
ops.recorder('Node', '-file', 'disp_tmd.txt', '-time', '-node', 3, '-dof', 1, 'disp')


# 6. ΔΥΝΑΜΙΚΗ ΑΝΑΛΥΣΗ
ops.system('ProfileSPD')             # Επίλυση μητρώων
ops.numberer('Plain')                # Αρίθμηση κομβων/στοιχείων
ops.constraints('Transformation')    # Χειρισμός συνοριακών συνθηκών (πακτώσεις)
ops.integrator('Newmark', 0.5, 0.25) # Κλασική μέθοδος ολοκλήρωσης Newmark (β=0.25, γ=0.5) για σταθερή επιτάχυνση
ops.algorithm('Linear')              # Γραμμικη ανάλυση
ops.analysis('Transient')            # Δυναμικη αναλυση λαμβανοντας υπ'οψιν την μαζα και την αποσβεση

# 7. ΕΚΤΕΛΕΣΗ ΤΗΣ ΑΝΑΛΥΣΗΣ
dt = 0.01             # Χρονικό βήμα ανάλυσης (sec)
t_final = 60.0        # Συνολικός χρόνος προσομοίωσης (sec)
num_steps = ops.analyze(int(t_final/dt), dt)  # Εκτέλεση ανάλυσης για t_final/dt βήματα
ops.analyze(num_steps, dt)

ops.wipe()# Η εντολή wipe "κλείνει" το OpenSees και ώστε να μπορέσουμε να διαβάσουμε τα αρχεία txt


#ΣΧΕΔΙΑΣΗ
data_building = np.loadtxt('disp_building.txt')
data_tmd = np.loadtxt('disp_tmd.txt')

time = data_building[:, 0]
u_building = data_building[:, 1]
u_tmd_absolute = data_tmd[:, 1]

# Ο κόμβος 3 (TMD) καταγράφει την απόλυτη μετακίνηση. 
# Για να βρούμε πόσο παραμορφώθηκε το ελατήριο του TMD, βρίσκουμε τη σχετική μετακίνηση (u_d):
u_d = u_tmd_absolute - u_building

# Σχεδιασμός
plt.figure(figsize=(12, 6))

# Σχεδιάζουμε τη μετακίνηση της κατασκευής και τη σχετική μετακίνηση του TMD
plt.plot(time, u_building, label='Μετακίνηση Κτιρίου χωρίς TMD (u)', color='blue', linewidth=1.5)
plt.plot(time, u_d, label='Σχετική Μετακίνηση με TMD (ud)', color='red', linestyle='--', linewidth=1.5)

plt.title('Απόκριση Συστήματος σε Ημιτονοειδή Διέγερση', fontweight='bold')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Μετακίνηση (m)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right')
#plt.xlim(0, t_final)
#plt.ylim(-1, 1)
plt.tight_layout()
plt.show()