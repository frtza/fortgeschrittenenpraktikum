import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds, uarray as uar
from uncertainties import ufloat
import csv

with open('data/daten.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

# covert data to float
data = np.array(data, dtype=float)

print(data)

#teil 1:Magnetfeld der Erde
#funktion zur Berechnung des Magnetfeldes des Helmholzspulenpaares
def B(I, n, R):
    return const.mu_0 * (8 * I * n) / (np.sqrt(125) * R)

R_sweep = 0.1639
N_sweep = 11
R_hori = 0.1579
N_hori = 154

#Umrechnung der Stromstärke in Ampere

def ohm(U,o):
    return U/o

#widerstände definieren
o_sweep = 1
o_hori = 0.5


#Stromstärken berechnen

# get values from column 2 of data numpy array
U_sweep_1 = data[:,2]
U_hori_1 = data[:,1]
U_hori_1 = U_hori_1 - 13.8
U_sweep_2 = data[:,4]
U_hori_2 = data[:,3]
U_hori_2 = U_hori_2 - 13.8

I_sweep_1 = ohm(U_sweep_1, o_sweep)
I_hori_1 = ohm(U_hori_1, o_hori)
I_sweep_2 = ohm(U_sweep_2, o_sweep)
I_hori_2 = ohm(U_hori_2, o_hori)

#ermittlung b feld


B_sweep_1 = B(I_sweep_1, N_sweep, R_sweep)
B_hori_1 = B(I_hori_1, N_hori, R_hori)
B_sweep_2 = B(I_sweep_2, N_sweep, R_sweep)
B_hori_2 = B(I_hori_2, N_hori, R_hori)

#Addieren der beiden Magnetfelder

B_1 = B_sweep_1 + B_hori_1
B_2 = B_sweep_2 + B_hori_2
print('B_1:', B_1)
print('B_2:', B_2)

#Teil 2:Lande Faktor
#Definition der Ausgleichsfunktion
def f(a,b,x):
    return a * x + b

#Einlesen der Daten der Frequenzen
F_res = data[:,0]

#Plotte resonanzfrequenzen gegen die Magnetfelder
plt.plot(F_res, B_1 * 10**3, 'gx', label='Messwerte 1')
plt.plot(F_res, B_2* 10**3, 'bx', label='Messwerte 2')

#Lineare Regression der Messwerte
from scipy.optimize import curve_fit
params_1, covariance_1 = curve_fit(f, F_res, B_1)
params_2, covariance_2 = curve_fit(f, F_res, B_2)

errors_1 = np.sqrt(np.diag(covariance_1))
errors_2 = np.sqrt(np.diag(covariance_2))


x_plot = np.linspace(0, 1000, 1000)
plt.plot(x_plot, f(x_plot, *params_1) * 10**3, 'g-', label='Lineare Regression 1')
plt.plot(x_plot, f(x_plot, *params_2) * 10**3, 'b-', label='Lineare Regression 2')

#plt.xlabel(r'$B \:/\: \si{\tesla}$')
#plt.ylabel(r'$f \:/\: \si{\hertz}$')
plt.legend()
plt.grid()
plt.savefig('build/plot.pdf')

#Ausgabe der Parameter

a_1 = ufloat(params_1[0], errors_1[0])
b_1 = ufloat(params_1[1], errors_1[1])
a_2 = ufloat(params_2[0], errors_2[0])
b_2 = ufloat(params_2[1], errors_2[1])

print('a_1:', a_1)
print('b_1:', b_1)
print('a_2:', a_2)
print('b_2:', b_2)

#Teil 3: Lande Faktor
#Berechnung des Lande-Faktors

mu_b = const.value('Bohr magneton') # J/T
h = const.h #J s
def g(a):
    return h / (mu_b * a * 10**(-3))

g_1 = g(a_1)
g_2 = g(a_2)

#verhältnis der Lande-Faktoren
g_verh = g_1 / g_2

#Teil 4: Kernspins
#Berechnung der Kernspins

S = 1/2
L = 0
J = S + L

