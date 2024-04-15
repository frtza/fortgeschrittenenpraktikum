import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds, uarray as uar

def degmin(ang : np.ndarray):
    return ang.astype(int) + (ang - ang.astype(int)) / 0.6

def rad(ang : np.ndarray):
    return ang * np.pi / 180

B, z = np.genfromtxt('data/field.txt', unpack=True)
B, z = B * 1e-3, z * 1e-3

plt.plot(z, B, 'kx', ms=3.21)

plt.xlabel(r'$z$ / m')
plt.ylabel(r'$B$ / T')

plt.gca().ticklabel_format(scilimits=(0,0))

plt.savefig('build/field.pdf')
plt.close()

L = 1.36e-3

l, th1, th2 = np.genfromtxt('data/doped-1.txt', unpack=True)
th1_1, th2_1 = degmin(th1), degmin(th2)

l = l * 1e-6
theta_1 = rad((th1_1 - th2_1) / 2) / L

plt.plot(l**2, theta_1, 'kx', ms=3.21)

plt.xlabel(r'$\lambda^2$ / m$^2$')
plt.ylabel(r'$\theta$ / rad m$^{-1}$')

plt.gca().ticklabel_format(scilimits=(0,0))

plt.savefig('build/doped-1.pdf')
plt.close()

L = 1.296e-3

l, th1, th2 = np.genfromtxt('data/doped-2.txt', unpack=True)
th1_2, th2_2 = degmin(th1), degmin(th2)

l = l * 1e-6
theta_2 = rad((th1_2 - th2_2) / 2) / L

plt.plot(l**2, theta_2, 'kx', ms=3.21)

plt.xlabel(r'$\lambda^2$ / m$^2$')
plt.ylabel(r'$\theta$ / rad m$^{-1}$')

plt.gca().ticklabel_format(scilimits=(0,0))

plt.savefig('build/doped-2.pdf')
plt.close()

L = 5.11e-3

l, th1, th2 = np.genfromtxt('data/pure.txt', unpack=True)
th1_3, th2_3 = degmin(th1), degmin(th2)

l = l * 1e-6
theta = rad((th1_3 - th2_3) / 2) / L

plt.plot(l**2, theta, 'kx', ms=3.21)

plt.xlabel(r'$\lambda^2$ / m$^2$')
plt.ylabel(r'$\theta$ / rad m$^{-1}$')

plt.gca().ticklabel_format(scilimits=(0,0))

plt.savefig('build/pure.pdf')
plt.close()

thetadiff_1 = np.abs(theta - theta_1)
thetadiff_2 = np.abs(theta - theta_2)

mask = np.array([False, False, True, True, True, True, True, False, True])

b = np.max(B)
n = 3.397

N_1 = 1.2e24
N_2 = 2.8e24

e = const.e
c = const.c
eps = const.epsilon_0

K = e**3 / (8 * np.pi**2 * eps * c**3)

par_1, cov_1 = np.polyfit(l[mask]**2, theta_1[mask], deg=1, cov=True)
err_1 = np.sqrt(np.diag(cov_1))
par_2, cov_2 = np.polyfit(l[mask]**2, theta_2[mask], deg=1, cov=True)
err_2 = np.sqrt(np.diag(cov_2))

a_1, b_1 = par_1[0], par_1[1] 
ae_1, be_1 = err_1[0], err_1[1]
a_2, b_2 = par_2[0], par_2[1] 
ae_2, be_2 = err_2[0], err_2[1]

ll = np.array([-1, 1])

plt.plot(l[mask]**2, theta_1[mask], 'kx', ms=4, zorder=10, label=r'$\theta_1$ Daten')
plt.plot(l[mask]**2, theta_2[mask], 'k+', ms=5.5, zorder=10, label=r'$\theta_2$ Daten')
plt.plot(l[~mask]**2, theta_1[~mask], 'x', ms=4, c='firebrick', zorder=10)
plt.plot(l[~mask]**2, theta_2[~mask], '+', ms=5.5, c='firebrick', zorder=10)

plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

plt.plot(ll, a_1 * ll + b_1, c='olivedrab', zorder=0, label=r'$\theta_1$ Fit')
plt.plot(ll, a_2 * ll + b_2, c='steelblue', zorder=0, label=r'$\theta_2$ Fit')

plt.xlabel(r'$\lambda^2$ / m$^2$')
plt.ylabel(r'$\theta$ / rad m$^{-1}$')

plt.legend(frameon=False)

plt.gca().ticklabel_format(scilimits=(0,0))

plt.savefig('build/mass.pdf')
plt.close()

me = const.electron_mass
m = 0.067 * me

m1 = np.sqrt((K * N_1 * b / n) / a_1)
m2 = np.sqrt((K * N_2 * b / n) / a_2)

print()
print('Fit:')
print()
print(f'(1)   a = {a_1:.3} +- {ae_1:.3}   b = {b_1:.3} +- {be_1:.3}')
print(f'(2)   a = {a_2:.3} +- {ae_2:.3}   b = {b_2:.3} +- {be_2:.3}')
print()
print()
print('Literatur:')
print()
print(f'      me = {me:.2} kg')
print()
print(f'      m* = {m:.2} kg = {m / me:.3f} me')
print()
print('Ergebnis:')
print()
print(f'(1)   m* = {m1:.2} kg = {m1 / me:.3f} me')
print(f'(2)   m* = {m2:.2} kg = {m2 / me:.3f} me')
print()

with open('build/a-1.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{a_1*1e-12:.2f}({ae_1*1e-12:.2f})e12')
	f.write(r'}{\radian\per\meter\cubed}')

with open('build/a-2.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{a_2*1e-12:.2f}({ae_2*1e-12:.2f})e12')
	f.write(r'}{\radian\per\meter\cubed}')

with open('build/b-1.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{b_1:.2f}({be_1:.2f})')
	f.write(r'}{\radian\per\meter}')

with open('build/b-2.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{b_2:.2f}({be_2:.2f})')
	f.write(r'}{\radian\per\meter}')

table_footer = r'''		\bottomrule
	\end{tabular}
'''
table_header = r'''	\begin{tabular}{S[table-format=1.3] S[table-format=3.2] S[table-format=3.2] S[table-format=3.2] S[table-format=3.2] S[table-format=3.2] S[table-format=3.2]}
		\toprule
		& \multicolumn{2}{c}{n-GaAs (1)} & \multicolumn{2}{c}{n-GaAs (2)} & \multicolumn{2}{c}{GaAs} \\
		\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule{6-7}
		{$\lambda \mathbin{/} \unit{\micro\meter}$} &
        {$\theta_1 \mathbin{/} \unit{\degree}$} & {$\theta_2 \mathbin{/} \unit{\degree}$} &
        {$\theta_1 \mathbin{/} \unit{\degree}$} & {$\theta_2 \mathbin{/} \unit{\degree}$} &
        {$\theta_1 \mathbin{/} \unit{\degree}$} & {$\theta_2 \mathbin{/} \unit{\degree}$} \\
		\midrule
'''
row_template = r'		{0:1.3f} & {1:3.2f} & {2:3.2f} & {3:3.2f} & {4:3.2f} & {5:3.2f} & {6:3.2f} \\'
with open('build/table_samples.tex', 'w') as f:
    f.write(table_header)
    for row in zip(l * 1e6, th1_1, th2_1, th1_2, th2_2, th1_3, th2_3):
        f.write(row_template.format(*row))
        f.write('\n')
    f.write(table_footer)
table_header = r'''	\begin{tabular}{S[table-format=2.0] S[table-format=3.0] S[table-format=2.0] S[table-format=3.0] S[table-format=2.0] S[table-format=3.0]}
		\toprule
		{$z \mathbin{/} \unit{\milli\meter}$} & {$B \mathbin{/} \unit{\milli\tesla}$} &
		{$z \mathbin{/} \unit{\milli\meter}$} & {$B \mathbin{/} \unit{\milli\tesla}$} &
		{$z \mathbin{/} \unit{\milli\meter}$} & {$B \mathbin{/} \unit{\milli\tesla}$} \\
		\cmidrule(lr){1-2}\cmidrule(lr){3-4}\cmidrule(lr){5-6}
'''
row_template = r'		{0:2.0f} & {1:3.0f} & {2:2.0f} & {3:3.0f} & {4:2.0f} & {5:3.0f} \\'
with open('build/table_field.tex', 'w') as f:
    f.write(table_header)
    for row in zip(z[0:7] * 1e3, B[0:7] * 1e3, z[7:14] * 1e3, B[7:14] * 1e3, z[14:21] * 1e3, B[14:21] * 1e3):
        f.write(row_template.format(*row))
        f.write('\n')
    f.write(table_footer)
