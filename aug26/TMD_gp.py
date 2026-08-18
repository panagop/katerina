import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt

## Φόρτωση δεδομένων
# Υποθέτουμε 2 στήλες: [Χρόνος, Επιτάχυνση]. Οι επιταχύνσεις είναι σε g,
# οπότε η μετατροπή σε m/s^2 γίνεται με το '-factor', 9.81 στο timeSeries.
data = np.loadtxt('elCentro.txt')
ground_accel = data[:, 1]

# Εκτύπωση επιταχυνσιογραφήματος σεισμού
plt.figure(figsize=(15,3))
plt.plot( data[:,0] , data[:,1] , color='k')
plt.title('Επιταχυνσιογράφημα Σεισμού', fontsize=14, fontweight='bold')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Επιτάχυνση (g)')
plt.xlim(0, data[-1,0])
plt.grid()
plt.show()

# Το dt του σεισμού(πρέπει να το ξέρω) το υπολογιζω απο τα δεδομενα του σεισμου
t = data[:,0]
dt = t[1] - t[0]
dt = round(dt, 5)
n_steps = len(ground_accel) - 1   # πλήθος χρονικών βημάτων (όχι σημείων)
t_final = n_steps * dt
print(f"Χρονικό βήμα (dt) του σεισμού: {dt} sec")
print(f"Διάρκεια σεισμού: {t_final} sec ({n_steps} βήματα)")

# Δεδομένα Κατασκευής
m = 1              # Μάζα κατασκευής σε tn δλδ 1000 kg
T = 0.49            # Περίοδος κατασκευής
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

    # Φόρτιση: το επιταχυνσιογράφημα (σε g) επί 9.81 -> m/s^2
    ops.timeSeries('Path', 1, '-dt', dt, '-values', *ground_accel.tolist(), '-factor', 9.81)
    ops.pattern('UniformExcitation', 1, 1, '-accel', 1)
    #κούνα όλη τη βάση του κτιρίου ομοιόμορφα 1 δλδ κατευθυνση κατα χ... «φορτώνει» τον σεισμό στο σύστημα


    # Ρυθμίσεις Ανάλυσης
    ops.system('ProfileSPD')
    ops.numberer('Plain')
    ops.constraints('Transformation')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.algorithm('Linear')
    ops.analysis('Transient')

    # Εκτέλεση και αποθήκευση αποτελεσμάτων στη μνήμη (όχι σε txt)
    # Ξεκινάμε από t=0 με μηδενικές αρχικές συνθήκες
    disp_data = [0.0]     # μετακίνηση κατασκευής (κόμβος 2)
    stroke_data = [0.0]   # σχετική διαδρομή του TMD (κόμβος 3 - κόμβος 2)

    # Σταθερό πλήθος βημάτων -> χωρίς σφάλμα συσσώρευσης στο t
    for i in range(n_steps):
        if ops.analyze(1, dt) != 0:
            print(f"Η ανάλυση απέτυχε στο βήμα {i+1} (t = {(i+1)*dt:.2f} s)")
            break
        disp_data.append(ops.nodeDisp(2, 1))
        if with_TMD:
            stroke_data.append(ops.nodeDisp(3, 1) - ops.nodeDisp(2, 1))
        else:
            stroke_data.append(0.0)

    # Ο χρόνος προκύπτει από το πλήθος των βημάτων που όντως έτρεξαν
    time_data = np.arange(len(disp_data)) * dt

    return time_data, np.array(disp_data), np.array(stroke_data)


t_SPOF , u_SPOF , _        = TMD_model(with_TMD=False)  # Χωρίς TMD
t_TMD  , u_TMD  , stroke   = TMD_model(with_TMD=True)   # Με TMD


# ΥΠΟΛΟΓΙΣΜΟΣ ΜΕΓΙΣΤΩΝ ΜΕΤΑΚΙΝΗΣΕΩΝ ΚΑΙ ΠΟΣΟΣΤΟΥ ΑΠΟΔΟΣΗΣ TMD
u_max_SPOF = np.max(np.abs(u_SPOF))
u_max_TMD = np.max(np.abs(u_TMD))
reduction = (u_max_SPOF - u_max_TMD) / u_max_SPOF * 100

#me τη τετραγ τιμη της ριζας RMS = Root Mean Square (Ενεργός Τιμή)
RMS_SPOF = np.sqrt(np.mean(u_SPOF**2))
RMS_TMD = np.sqrt(np.mean(u_TMD**2))
diff_RMS = (RMS_SPOF - RMS_TMD) / RMS_SPOF * 100

# Η μέγιστη σχετική διαδρομή του TMD - κρίσιμο μέγεθος για τον σχεδιασμό της συσκευής
stroke_max = np.max(np.abs(stroke))

print("\n ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑΛΥΣΗΣ")
print(f"Μέγιστη Μετακίνηση χωρίς TMD : {u_max_SPOF:.4f} m")
print(f"Μέγιστη Μετακίνηση με TMD    : {u_max_TMD:.4f} m")
print(f"Ποσοστό Μείωσης Μεγιστου        : {reduction:.2f} %")
print(f"Μείωση Συνολικής Ενέργειας (RMS): {diff_RMS:.2f} %")
print(f"Μέγιστη σχετική διαδρομή TMD    : {stroke_max:.4f} m ({stroke_max/u_max_TMD:.1f} x η μετακίνηση της κατασκευής)")

u_SPOFnorm = u_SPOF / u_max_SPOF
u_TMDnorm = u_TMD / u_max_SPOF


# ΣΧΕΔΙΑΣΗ (Τώρα σχεδιάζουμε τα κανονικοποιημένα δεδομένα)
plt.figure(figsize=(12, 6))
# Χρησιμοποιούμε μαύρο χρώμα και διακεκομμένη/συνεχή γραμμή για να μοιάζει με το βιβλίο
plt.plot(t_SPOF, u_SPOFnorm, label='Μετακίνηση SDOF (χωρίς TMD) ', color='blue', linewidth=1.5)
plt.plot(t_TMD, u_TMDnorm, label='Μετακίνηση με TMD ', color='red', linestyle='--', linewidth=1.5)

plt.title('Κανονικοποιημενη Απόκριση Συστήματος με ElCentro ', fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('u / u_max') # αδιάστατο μέγεθος
plt.xlim(0, t_final)
plt.ylim(-1, 1) # Βάζουμε τα όρια του άξονα Υ όπως στην εικόνα σου
plt.legend(loc='lower left') # Τοποθέτηση του υπομνήματος κάτω αριστερά
plt.tight_layout()
plt.show()


#ΣΧΕΔΙΑΣΗ
plt.figure(figsize=(12, 6))
plt.plot(t_SPOF, u_SPOF, label='Μετακίνηση SDOF (χωρίς TMD) ', color='blue', linewidth=1.5)
plt.plot(t_TMD, u_TMD, label='Μετακίνηση με TMD ', color='red', linestyle='--', linewidth=1.5)

plt.title('Απόκριση Συστήματος σε σεισμική διέγερση elCentro ', fontweight='bold')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Μετακίνηση (m)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right')
plt.xlim(0, t_final)
plt.tight_layout()
plt.show()


# ΣΧΕΤΙΚΗ ΔΙΑΔΡΟΜΗ ΤΟΥ TMD (απαιτούμενος χώρος κίνησης της μάζας)
plt.figure(figsize=(12, 4))
plt.plot(t_TMD, stroke, color='green', linewidth=1.5)
plt.title('Σχετική διαδρομή TMD ως προς την κατασκευή', fontweight='bold')
plt.xlabel('Χρόνος (s)')
plt.ylabel('u_TMD - u_κατασκευής (m)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.xlim(0, t_final)
plt.tight_layout()
plt.show()
