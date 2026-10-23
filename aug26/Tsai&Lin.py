import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt

# f=f1/f2*
#m_bar = 0.05  # Αναλογία μάζας TMD προς μάζα κτιρίου  m_bar=m2/m1
#ξ = 0.05     # Απόσβεση m1*

m_hat = np.arange(0.001, 0.101, 0.001) # m_hat=0.001 εως m_hat=0.1 με βήμα 0.001
ξ = np.array([0.000001, 0.01, 0.02, 0.05, 0.1]) 

def f(m_hat, ξ):
    # Όλη η εξίσωση σε μία γραμμή
    f = ((np.sqrt(1 - 0.5 * m_hat) / (1 + m_hat)) + np.sqrt(1 - 2 * ξ**2) - 1) - ((2.375 - 1.034 * np.sqrt(m_hat) - 0.426 * m_hat) * ξ * np.sqrt(m_hat)) - ((3.730 - 16.903 * np.sqrt(m_hat) + 20.496 * m_hat) * (ξ**2) * np.sqrt(m_hat))
    
    return f

plt.figure(figsize=(10, 7)) 

# Λούπα: Για κάθε τιμή του ξ που έφτιαξες στο np.arange, ζωγράφισε μια καμπύλη
for j in ξ:
    # Υπολογισμός των τιμών f για όλο το array του m_hat, για το τρέχον ξ
    f_values = f(m_hat, j)
    # Σχεδιασμός της γραμμής. Το .3f στρογγυλοποιεί το όνομα στο υπόμνημα σε 3 δεκαδικά
    plt.plot(m_hat, f_values, lw=2, label=f'ξ = {j:.3f}')


# Διαγράμματα (Τίτλοι και Άξονες)
plt.title('Συντελεστής βέλτιστης συχνότητας συντονισμού για το TMD', fontsize=14, fontweight='bold')
plt.xlabel('Λόγος Μαζών (m_hat)', fontsize=12)
plt.ylabel('Παράμετρος (f)', fontsize=12)

plt.grid(True, which='major', linestyle='-', linewidth=0.8)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', alpha=0.6)

plt.xlim(0, 0.1)   
plt.ylim(0.8, 1)

#Προσθήκη υπομνήματος 
plt.legend(title='Απόσβεση (ξ)', loc='upper right', shadow=True)
plt.tight_layout()
plt.show()

# 2η ΕΞΙΣΩΣΗ TSAI & LIN

# Η εξίσωση σε μία γραμμή
def ξd(m_hat, ξ):
    return np.sqrt((3*m_hat)/(8*(1+m_hat)*(1-0.5*m_hat))) + (0.151*ξ - 0.170*ξ**2) + (0.163*ξ + 4.980*ξ**2)*m_hat

# Δημιουργία 2ου γραφήματος
plt.figure(figsize=(10, 7))

# Λούπα σχεδίασης
for xi in ξ:
    plt.plot(m_hat, ξd(m_hat, xi), lw=2, label=f'ξ = {xi:.3f}')

# Μορφοποίηση με τις ελάχιστες δυνατές εντολές
plt.title('Βέλτιστος λόγος απόσβεσης για το TMD', fontweight='bold')
plt.xlabel('Λόγος Μαζών (m_hat)', fontsize=12)
plt.ylabel('ξd', fontsize=12)
plt.xlim(0, 0.1)
plt.ylim(0, 0.2)  # Τα όρια του Υ προσαρμοσμένα για το ξ_d
plt.grid(True, which='major'); plt.minorticks_on(); plt.grid(True, which='minor', linestyle=':')
plt.legend()
plt.show()