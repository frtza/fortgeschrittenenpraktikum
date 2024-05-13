import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds, uarray as uar
from uncertainties import ufloat
import csv

#Data import

with open('data/messung.csv') as csvfile:
    data = list(csv.reader(csvfile))
with open('data/alpha_ph.csv') as csvfile:
    alpha = list(csv.reader(csvfile))
with open('data/debye.csv') as csvfile:
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
rho = 8930 # kg/m^3
m = 0.324
M = rho * V_m #molare masse
k = 137 * 10 ** 9
kb = const.Boltzmann
v_t = 2260 # transversalgeschwindigkeit
v_l = 4700 # longitudinalgeschwindigkeit
N_a = const.Avogadro
N = (m / M) * N_a #anzahl teilchen probe
n = m / M #anzahl mol
V = m / rho
h_quer = const.hbar

print('n:', n)
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
I = data[:,2] * 10 ** (-3)#in ampere
I = uar(I, 0.1)

E = Energie(U, I, dt)
print('E:', E)

#calculate cp

Cp = cp(E[:-1], dT)
print('Cp:', Cp)

# plot Cp against T
plt.plot(noms(T[:-1]), noms(Cp), 'x',c = 'darkorchid', label='Messwerte')
plt.xlabel('T in K')
plt.ylabel('Cp in J/K')
plt.legend()
plt.grid()
plt.savefig('build/Cp.pdf')
plt.clf()

#caluclate cv
Cv = cv(Cp, alpha[:-1], T[:-1])
print('alpha:', alpha[:-1])
print('T:', T[:-1])
print('Cv:', Cv)


#plot Cv against T

plt.plot(noms(T[:-1]), noms(Cv), 'x',c = 'darkorchid', label='Messwerte')
plt.xlabel('T in K')
plt.ylabel('$c_V$ in J/mol K')
plt.legend()
plt.grid()
plt.savefig('build/Cv.pdf')
plt.clf()

#debye temperature

D = debye * T
print('D:', D)

#print('dT:')
#for i in range(len(dT)):
    #print(np.round(noms(dT[i]), 2), "+/-", np.round(stds(dT[i]), 2))

#print('dt:')
#for i in range(len(dt)):
 #   print(np.round(noms(dt[i]), 2), "+/-", np.round(stds(dt[i]), 2))

#print('I:')
#for i in range(len(I)):
 #   print(np.round(noms(I[i]*10**3), 1), "+/-", np.round(stds(I[i]*10**3), 1))

#print('U:')
#for i in range(len(U)):
 #   print(np.round(noms(U[i]), 2), "+/-", np.round(stds(U[i]), 2))

#print('Cp:')
#for i in range(len(Cp)):
 #   print(np.round(noms(Cp[i]), 2), "+/-", np.round(stds(Cp[i]), 2))

print('T:')
for i in range(10):
    print(np.round(noms(T[i]),2), '+/-', np.round(stds(T[i]),2))


#print('Cv:')
#for i in range(len(Cv)):
    #print(np.round(noms(Cv[i]), 2), "+/-", np.round(stds(Cv[i]), 2))

#print('alpha:')
#for i in range(len(alpha)):
    #print(alpha[i])

print('D:')
for i in range(10):
    print(D[i])

#calculate mean value of debye temperature for the first 10 values

D_mean = np.mean(D[:10])
print('D_mean:', D_mean)

def abw(emp, theo):
    return abs(emp - theo)/theo * 100

print('Abweichung:', abw(D_mean, theta_D))

#theorethische wärmekapazität
C_exp= Cv[-1]
print('C_exp:', C_exp)
C_theo = 390 * M
print('C_theo:', C_theo)

print('Abweichung:', abw(C_exp, C_theo))