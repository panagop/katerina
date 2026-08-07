import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Δεδομένα Κατασκευής
m = 1000            # Μάζα κατασκευής
T = 0.49            # Περίοδος κατασκευής 
ω = 2 * np.pi / T   # Κυκλική ιδιοσυχνότητα κατασκευής 
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

# 4. ΗΜΙΤΟΝΟΕΙΔΗΣ ΔΙΕΓΕΡΣΗ P = P_hat*sin(Ωt)
Ω = 1 * ω                         # Συχνότητα της δύναμης  (έστω ρ = 1 ) Ω = ω*ρ
period_force = 2 * np.pi / Ω      # Ω = 2π / Τforce
p_hat = 2 * ξ * k * 1.0           # Πλάτος της δύναμης p^

# 5. ΧΡΟΝΙΚΑ ΒΗΜΑΤΑ ΑΝΑΛΥΣΗΣ
dt = 0.01            # Χρονικό βήμα ανάλυσης (sec)
t_final = 60.0        # Συνολικός χρόνος προσομοίωσης (sec)

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

    # Υλικά Κατασκευής & Σύνδεση
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
    ops.timeSeries('Trig', 1, 0.0, 999.0, period_force)
    ops.pattern('Plain', 1, 1)
    ops.load(2, p_hat)
    
    
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
    while t < t_final:
        ops.analyze(1, dt)
        t = t + dt
        time_data.append(t)
        disp_data.append(ops.nodeDisp(2, 1)) # Καταγραφή μετακίνησης του κόμβου 2 (Κτίριο)
        
    return np.array(time_data), np.array(disp_data)


t_bild , u_bild = TMD_model(with_TMD=False)  # Χωρίς TMD
t_tmd  , u_tmd = TMD_model(with_TMD=True)  # Με TMD

P_all = p_hat * np.sin(Ω * t_bild)

#ΣΧΕΔΙΑΣΗ
plt.figure(figsize=(12, 6))
plt.plot(t_bild, u_bild, label='Μετακίνηση Κτιρίου χωρίς TMD ', color='blue', linewidth=1.5)
plt.plot(t_tmd, u_tmd, label='Μετακίνηση Κτιρίου με TMD ', color='red', linestyle='--', linewidth=1.5)

plt.title('Απόκριση Συστήματος σε Ημιτονοειδή Διέγερση', fontweight='bold')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Μετακίνηση (m)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right')
#plt.xlim(0, t_final)
#plt.ylim(-1, 1)
plt.tight_layout()
plt.show()

