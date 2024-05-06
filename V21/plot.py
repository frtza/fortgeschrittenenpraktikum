import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds, uarray as uar
from uncertainties import ufloat
import csv

with open('data/messung.csv', newline='') as csvfile:
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
U_sweep_1 = data[:,1]
U_hori_1 = data[:,2]
U_sweep_2 = data[:,3]
U_hori_2 = data[:,4]

I_sweep_1 = U_sweep_1 * 0.1  # I in ampere
I_hori_1 = 2 * U_hori_1 * 10 ** (-3)
I_sweep_2 = U_sweep_2 * 0.1  # I in ampere
I_hori_2 = 2 * U_hori_2 * 10 ** (-3)
#ausgabe der stromstärken
print('I_sweep_1:', I_sweep_1)
print('I_hori_1:', I_hori_1)
print('I_sweep_2:', I_sweep_2)
print('I_hori_2:', I_hori_2 )

#ermittlung b feld


B_sweep_1 = B(I_sweep_1, N_sweep, R_sweep)
B_hori_1 = B(I_hori_1, N_hori, R_hori)
B_sweep_2 = B(I_sweep_2, N_sweep, R_sweep)
B_hori_2 = B(I_hori_2, N_hori, R_hori)

#Addieren der beiden Magnetfelder

B_1 = B_sweep_1[:-1] + B_hori_1[:-1]
B_2 = B_sweep_2 + B_hori_2
print('B_1:', B_1)
print('B_2:', B_2)

#Teil 2:Lande Faktor
#Definition der Ausgleichsfunktion
def f(x, a, b):
    return a * x + b

#Einlesen der Daten der Frequenzen
F_res = data[:,0]

print(len(F_res))

#Plotte resonanzfrequenzen gegen die Magnetfelder
plt.plot(F_res[:-1], B_1 * 10**3, 'x', c = 'steelblue' , label='Messwerte 1')
plt.plot(F_res, B_2 * 10**3, 'x', c = 'seagreen' , label='Messwerte 2')

print(B_1)

#Lineare Regression der Messwerte
from scipy.optimize import curve_fit
params_1, covariance_1 = curve_fit(f, F_res[:-1], B_1)
params_2, covariance_2 = curve_fit(f, F_res, B_2)

errors_1 = np.sqrt(np.diag(covariance_1))
errors_2 = np.sqrt(np.diag(covariance_2))

x_plot = np.linspace(0, 1000, 1000)
plt.plot(x_plot, f(x_plot, *params_1) * 10**3, '-', c = 'steelblue', alpha = 0.5, label='Lineare Regression 1')
plt.plot(x_plot, f(x_plot, *params_2) * 10**3, '-',c = 'seagreen', alpha = 0.5 , label='Lineare Regression 2')

#plt.xlabel(r'$B \:/\: \si{\tesla}$')
#plt.ylabel(r'$f \:/\: \si{\hertz}$')
plt.xlabel(r'$f \, / \, \mathrm{kHz}$')
plt.ylabel(r'$B \, / \, \mathrm{mT}$')
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

#bilde mittelwert von b_1 und b_2
#mittelwert bilden

mw = (b_1 + b_2)/ 2
print('mw:', mw)

I_V = 0.23
R_V = 0.11735
N_V= 20.
B_V = B(I_V,N_V, R_V)

B_Erde = unp.sqrt(mw ** 2 + B_V ** 2)
print('Erde:', B_Erde)
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

