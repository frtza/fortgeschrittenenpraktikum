import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds, uarray as uar
from uncertainties import ufloat
import csv

#Data import

with open('data/daten_philipp.csv') as csvfile:
    data = list(csv.reader(csvfile))
with open('data/alpha_ph.csv') as csvfile:
    alpha = list(csv.reader(csvfile))
with open('data/debye_philipp.csv') as csvfile:
    debye = list(csv.reader(csvfile))

#coverting data to float

data = np.array(data, dtype=float)
print(data)
alpha = np.array(alpha, dtype=float)
alpha = alpha[:,0] * 10 ** (-6)
debye = np.array(debye, dtype=float)
debye = debye[:,0]

# constants definition

V_m = 7.11 * 10 ** (-6)
rho = 8920
m = 0.324
M = rho * V_m #molare masse
k = 137 * 10 ** 9
kb = const.Boltzmann
v_t = 2260 # transversalgeschwindigkeit
v_l = 4700 # longitudinalgeschwindigkeit
N_a = const.Avogadro
N = (m / M) * N_a #anzahl teilchen probe
V = m / rho
h_quer = const.hbar

#funtion to calculate cp

def cp(E, dT):
    return (M * E) / (m * dT)
def Energie(U, I, dt):
    return U * I * dt
def T(R):
    return 0.00134 * R ** 2 + 2.296 * R - 243.02 + 273.15 #in kelvin
def cv(cp, a, T):
    return cp - (9 * (a ** 2) * T * k * V_m)

#theoretische debye temperature

omega_D =  ((3 * 6 * (np.pi ** 2) * (N/V) ) ** (1/3)) * (((v_l ** 3) * (v_t ** 3))/(2 * (v_l ** 3) + (v_t ** 3))) ** (1/3)
theta_D = h_quer * omega_D / kb
print('theta_D:', theta_D)
print('Molare Masse:', M)
print('Anzahl der Teilchen:', N)
print('Volumen:', V)

#Resistnce to temperature
R = data[:,0]
#create a u array with resistance
R = uar(R, 0.1)
print('R:', R)

T = T(R) #array an temperaturen
print('T:', T)

#create array with temperature differences
dT = T[1:] - T[:-1]

print('dT:', dT)

#create array with time
t = data[:,3]
t = uar(t, 5)

#time difference
dt = t[1:] - t[:-1]
dt = np.insert(dt, 0, 0)
print('dt:', dt)

#calculate energy
U = data[:,1]
U = uar(U, 0.01)
I = data[:,2] #in ampere
I = uar(I, 0.1)

E = Energie(U, I, dt)
print('E:', E)

#calculate cp

Cp = cp(E[:-1], dT)
print('Cp:', Cp)

#caluclate cv
Cv = cv(Cp, alpha[:-1], T[:-1])
print('alpha:', alpha[:-1])
print('T:', T[:-1])
print('Cv:', Cv)

Cv_mean = np.mean(Cv)
print('Cv_mean:', Cv_mean)

#molare wärmekapazität

Cv_m = Cv_mean * M
print('Cv_m:', Cv_m)


# plot Cp against T
plt.plot(noms(T[:-1]), noms(Cp), 'x',c = 'seagreen', label='Messwerte')
plt.xlabel('T in K')
plt.ylabel('Cp in J/K')
plt.legend()
plt.grid()
plt.savefig('build/Cp.pdf')
plt.clf()

#plot Cv against T

plt.plot(noms(T[:-1]), noms(Cv), 'x',c = 'seagreen', label='Messwerte')
plt.xlabel('T in K')
plt.ylabel('Cv in J/K')
plt.legend()
plt.grid()
plt.savefig('build/Cv.pdf')
plt.clf()

#debye temperature

D = debye * T
print('D:', D)



