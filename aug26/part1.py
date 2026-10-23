import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

## Φόρτωση δεδομένων 
# Υποθέτουμε 2 στήλες: [Χρόνος, Επιτάχυνση]
data = np.loadtxt('elcentro.txt') #se g einai
ground_accel = data[:, 1] 
#ground_accel = ground_accel / 9.81 # αν οι επιταχ  ειναι σε m/s^2 (αν ειναι πολλ επι g δεν βαζω αυτη την γραμμή) 

## Εκτύπωση επιταχυνσιογραφήματος σεισμού
plt.figure(figsize=(15,3))
plt.plot(data[:,0], data[:,1], color='k')
plt.title('Επιταχυνσιογράφημα Σεισμού', fontsize=14, fontweight='bold')
plt.grid()
plt.show()

## 1. Παράμετροι
m = 1.0           # Σταθερή μάζα
ξ = 0.05         # Απόσβεση 5%

# Το dt του σεισμού(πρέπει να το ξέρω) το υπολογιζω απο τα δεδομενα του σεισμου 
t = data[:,0]
dt = t[1] - t[0]
dt = round(dt, 5)
print(f"Χρονικό βήμα (dt) του σεισμού: {dt} sec")

#Πινακας περιοδων για να κανω το φασμα αποκρισης, απο 0.00001 sec μεχρι 3 sec με βημα 0.01 sec
periods = np.arange(0.00001, 3.00001, 0.01) # Εύρος περιόδων το 0,01 ειναι το βημα των περιόδων (το βημα που σαρωνω τα διαφορετικα κτιρια), 0.00001 για να αποφύγουμε το μηδέν που θα μας δώσει άπειρη συχνότητα και δυσκαμψία
max_accelerations = []  


## 2. Κύριος Βρόχος
for T in periods:
    ops.wipe() #"Σβήσε" το προηγούμενο μοντέλο
    ops.model('basic', '-ndm', 1, '-ndf', 1)
    #ndm Αριθμός Διαστάσεων , ndf Αριθμός Βαθμών Ελευθερίας ανά κόμβο 1 για SDOF (Single Degree of Freedom)
    
    # Φυσική: Υπολογισμός K και C
    ω = 2 * np.pi / T
    k = m * (ω**2)
    c = 2 * ξ * ω * m
    
    ## Μοντέλο-ΓΕΩΜΕΤΡΙΑ
    ops.node(1, 0.0)
    ops.node(2, 0.0)
    #Φτιάχνουμε δύο σημεία. Το 1 είναι η βάση (στο έδαφος) και το 2 είναι η οροφή
    ops.fix(1, 1) #κομβος 1 πακτωμένος στο έδαφος
    ops.mass(2, m) #στον κόμβο 2 υπάρχει η μάζα
    
    # Στοιχεία: Ελατήριο και Αποσβεστήρας
    ops.uniaxialMaterial('Elastic', 1, k) #Φτιάχνουμε ένα "υλικό" που λειτουργεί σαν ελατήριο με δυσκαμψία k
    ops.uniaxialMaterial('Viscous', 2, c, 1.0) # με συντελεστή c
    ops.element('zeroLength', 1, 1, 2, '-mat', 1, '-dir', 1) 
    ops.element('zeroLength', 2, 1, 2, '-mat', 2, '-dir', 1)
  #"μηδενικό μήκος" (δεν προσθέτει βάρος ή διαστάσεις η ογκο) ενωνει τον 1και2 με υλικο mat σε dir 1 δλδ οριζοντιο αξονα  

    ## Φόρτωση Σεισμού
    ops.timeSeries('Path', 1, '-dt', dt, '-values', *ground_accel.tolist(), '-factor', 9.81) #αν ο σεισμος ειναι σε g για να γινει m/s^2
    ops.pattern('UniformExcitation', 1, 1, '-accel', 1)
    #κούνα όλη τη βάση του κτιρίου ομοιόμορφα 1 δλδ κατευθυνση κατα χ.....«φορτώνει» τον σεισμό στο σύστημα


    # Ρύθμιση Ανάλυσης
    ops.system('ProfileSPD') #λυτης της εξισωσης κινησης [Μ]{u''} + [C]{u'} + [K]{u} = -[M]{ug''} βρισκει Δu 
    ops.numberer('Plain') #Κανει αριθμηση κόμβων και στοιχείων...β.ε.
    ops.constraints('Transformation') #φροντίζει για τις συνθήκες ορίων (π.χ. πακτωμένος κόμβος)
    ops.integrator('Newmark', 0.5, 0.25) #Με τη μέθοδο αριθμητικής ολοκλήρωσης Newmark με γ=0.5 και β=0.25),μετατρεπει την δε σε [Κ]{Δu} = ΔF και λύνει για Δu
    ops.algorithm('Linear') #ειναι γραμμικη η ανάλυση, δεν χρειάζεται επαναληπτικός αλγόριθμος
    ops.analysis('Transient') #δυναμικη αναλυση λαμβανοντας υπ'οψιν την μαζα και την αποσβεση

    
    # Εύρεση μέγιστου..επιταχυνσης Sa
    max_acc = 0
    for i in range(len(ground_accel)):  #Για κάθε χρονικό βήμα του σεισμού (από το 0 δευτερόλεπτο μέχρι να τελειώσει
        ops.analyze(1, dt)  #προχώρα την προσομοίωση μπροστά κατά ένα βήμα dt (π.χ. 0.02 sec) και λύσε τις εξισώσεις κίνησης

        a =ops.nodeAccel(2,1) # σχετική επιτάχυνση
        a_tot = a + (ground_accel[i] *9.81) #ολικη επιταχυνση = Σχετική + Επιτάχυνση Εδάφους

       #βρες τη μεγαλυτερη
        if abs(a_tot) > max_acc:   #Αν η απόλυτη τιμή της επιτάχυνσης είναι h μεγαλύτερη  που έχω δει μέχρι τώρα, τότε κράτα αυτήν
            max_acc = abs(a_tot)
              
    max_accelerations.append(max_acc) #γεμιζω την αδεια λιστα με τις μεγιστες επιταχυνσεις για καθε περιοδο T

## 3. Σχεδιαση
plt.figure(figsize=(14,5))
plt.plot(periods, np.array(max_accelerations)/9.81) # κανει τις τιμες σε *g
plt.title('Φάσμα Απόκρισης Επιταχύνσεων', fontsize=14, fontweight='bold')
plt.xlabel('Περίοδος (sec)')
plt.ylabel('Φασματική Επιτάχυνση (g)') # δινει τις τιμες με *g
plt.grid()
plt.axis(xmin=0, ymin=0)
plt.show()

#ΕΛΕΓΧΟΣ
pga = np.max(np.abs(ground_accel))
print(f"Το PGA του σεισμού είναι: {pga:.4f} g") #υο μαχ του σεισμου

PGA = max_accelerations[0] / 9.81
print(f"Το PGA του σεισμού για Τ=0 είναι: {PGA:.4f} g") #apo fasma gia T=O


## Export max_accelerations to Excel (using pandas)
# 1. Βάζουμε τα δεδομένα σε ένα "πλαίσιο" (DataFrame)
df = pd.DataFrame({
    'Περίοδος T (sec)': periods,
    'Επιτάχυνση (g)': np.array(max_accelerations) / 9.81,
    'Επιτάχυνση (m/s²)': max_accelerations
})

# 2. Εξάγουμε το DataFrame σε αρχείο Excel
df.to_excel('my_response_spectrum.xlsx', index=False , float_format="%.5f") #Τα δεδομένα αποθηκεύτηκαν στο my_response_spectrum.xlsx!