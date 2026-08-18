import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Φόρτωση δεδομένων 
# Υποθέτουμε 2 στήλες: [Χρόνος, Επιτάχυνση]
data = np.loadtxt('harmonic excitation.txt')  #για να παρουμε την επιταχυνση σε m/s^2 διαιρω τη δυναμη με τη μαζα ( Ν / kg = m/s^2)
ground_sin = data[:, 1] 


# Εκτύπωση επιταχυνσιογραφήματος σεισμού
plt.figure(figsize=(15,3))
plt.plot(data[:,0], data[:,1], color='k')
plt.title('Επιταχυνσιογράφημα Σεισμού', fontsize=14, fontweight='bold')
plt.grid()
plt.show()

# Το dt του σεισμού(πρέπει να το ξέρω) το υπολογιζω απο τα δεδομενα του σεισμου 
t = data[:,0]
dt = t[1] - t[0]
dt = round(dt, 5)
print(f"Χρονικό βήμα (dt) του σεισμού: {dt} sec")
t_final = t[-1]

# Δεδομένα Κατασκευής
m = 1               # Μάζα κατασκευής σε tn δλδ 1000 kg
T = 0.49           # Περίοδος κατασκευής 
ω = 2 * np.pi / T   # Κυκλική ιδιοσυχνότητα κατασκευής 
ξ = 0.02            # Απόσβεση κατασκευής
k = (ω**2) * m
c = 2 * ξ * ω * m

# Δεδομένα Αποσβεστήρα (TMD)
m_hat = 0.01        # Αναλογία μαζών (md / m) αρα md=10κg=0,01tn 
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
def TMD_model(with_TMD):
    ops.wipe()
    #   ΧΩΡΙΣ TMD
    ops.model('basic', '-ndm', 1, '-ndf', 1) 

    # Κόμβοι & Στηρίξεις
    ops.node(1, 0.0) 
    ops.node(2, 0.0) 
    ops.fix(1, 1) 
    ops.mass(2, m)

    # Υλικά & Σύνδεση
    ops.uniaxialMaterial('Elastic', 1, k)
    ops.uniaxialMaterial('Viscous', 2, c, 1) 
    ops.element('zeroLength', 1, 1, 2, '-mat', 1, 2, '-dir', 1, 1)

    # ΜΕ TMD
    if with_TMD:
        ops.node(3, 0.0)
        ops.mass(3, md)
        ops.uniaxialMaterial('Elastic', 3, kd)
        ops.uniaxialMaterial('Viscous', 4, cd, 1)
        ops.element('zeroLength', 2, 2, 3, '-mat', 3, 4, '-dir', 1, 1)

    # Φόρτιση δημιουργεί το ημίτονο: sin(Ωt)
    ops.timeSeries('Path', 1, '-dt', dt, '-values', *ground_sin.tolist()) 
    ops.pattern('UniformExcitation', 1, 1, '-accel', 1)
    #κούνα όλη τη βάση του κτιρίου ομοιόμορφα 1 δλδ κατευθυνση κατα χ.....«φορτώνει» τον σεισμό στο σύστημα
    
    
    # Ρυθμίσεις Ανάλυσης
    ops.system('ProfileSPD')           
    ops.numberer('Plain')                
    ops.constraints('Transformation')    
    ops.integrator('Newmark', 0.5, 0.25) 
    ops.algorithm('Linear')              
    ops.analysis('Transient')  

    # Εκτέλεση και αποθήκευση αποτελεσμάτων στη μνήμη (όχι σε txt)
    time_data = []
    disp_data = []
    
    t = 0.0
    while  t < t_final:
        ops.analyze(1, dt)
        t = t + dt
        time_data.append(t)
        disp_data.append(ops.nodeDisp(2, 1)) 
   
    return np.array(time_data), np.array(disp_data)


t_SPOF , u_SPOF = TMD_model(with_TMD=False)  # Χωρίς TMD
t_TMD  , u_TMD  = TMD_model(with_TMD=True)   # Με TMD

#ΣΧΕΔΙΑΣΗ
plt.figure(figsize=(12, 6))
plt.plot(t_SPOF, u_SPOF, label='Μετακίνηση SDOF (χωρίς TMD) ', color='blue', linewidth=1.5)
plt.plot(t_TMD, u_TMD, label='Μετακίνηση με TMD ', color='red', linestyle='--', linewidth=1.5)

plt.title('Απόκριση Συστήματος σε Ημιτονοειδή Διέγερση', fontweight='bold')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Μετακίνηση (m)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right')
plt.xlim(0, t_final)
plt.ylim(-1, 1)
plt.tight_layout()
plt.show()