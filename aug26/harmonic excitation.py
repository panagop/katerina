import numpy as np

m = 1              # Μάζα κατασκευής σε tn
T = 0.49              
ω = 2 * np.pi / T    
ξ = 0.02
k = (ω**2) * m       

dt = 0.01             # Βήμα χρόνου
t_final = 60.0        # Συνολικός χρόνος 

# 4. ΗΜΙΤΟΝΟΕΙΔΗΣ ΔΙΕΓΕΡΣΗ P = P_hat*sin(Ωt)
Ω = 1 * ω                         # Συχνότητα της δύναμης  (έστω ρ = 1 ) Ω = ω*ρ
#period_force = 2 * np.pi / Ω      # Ω = 2π / Τforce
p_hat = 2 * ξ * k * 1.0           # Πλάτος της δύναμης p^    
           
#Δημιουργία διανύσματος χρόνου
time = np.arange(0, t_final + dt, dt)

#Υπολογισμός της δύναμης για κάθε χρονική στιγμή P = P_hat*sin(Ωt)
P = p_hat * np.sin(Ω * time)/m #για να τη παρουμε ως επιταχυνση σε m/s^2 διαιρω τη δυναμη με τη μαζα ( Ν / kg = m/s^2)


# Σε πίνακα 
export_data = np.column_stack((time, P))  # Η 1η στήλη θα είναι ο χρόνος και η 2η η δύναμη

#Εξαγωγή του πίνακα σε αρχείο .txt
np.savetxt('harmonic excitation.txt', export_data, fmt='%.6f')
