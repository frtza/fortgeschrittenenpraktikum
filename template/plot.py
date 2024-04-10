# essential libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

# read columns of data from txt file
a_1, b_1, c_1 = np.genfromtxt('data/allgemein.txt', unpack=True)
a_2, c_2 = np.genfromtxt('data/speziell.txt', unpack=True)

# parameter and error arrays, covariance matrix for linear regression or polynomials
par_2, cov_2 = np.polyfit(a_2**2, c_2**2, deg=1, cov=True)
err_2 = np.sqrt(np.diag(cov_2))

# graphical representation of correlation
a = np.linspace(-0.1, 5.1, 2)
ax = plt.subplot(111)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlabel(r'$a^2 \mathbin{/} \unit{\milli\meter\squared}$')
ax.set_ylabel(r'$c^2 \mathbin{/} \unit{\milli\meter\squared}$')
plt.plot(a**2, 2*a**2, '-', color='olivedrab', label='Theorieverlauf')
plt.plot(a_2**2, c_2**2, 'kx', markersize=3.21, label='Messdaten')
leg = ax.legend(edgecolor='k', facecolor='none')
leg.get_frame().set_linewidth(0.25)
plt.savefig('build/plot.pdf')
plt.close()

# format and export latex table
table_header = r'''	\begin{tabular}
		{S[table-format=1.0]
		 S[table-format=1.0]
		 S[table-format=1.2]}
		\toprule
		{$a \mathbin{/} \unit{\milli\meter}$} & 
		{$b \mathbin{/} \unit{\milli\meter}$} & 
		{$c \mathbin{/} \unit{\milli\meter}$} \\
		\midrule
'''
table_footer = r''' 	\bottomrule
	\end{tabular}
'''
row_template = r'		{0:1.0f} & {1:1.0f} & {2:1.2f} \\'
with open('build/table_allgemein.tex', 'w') as f:
	f.write(table_header)
	for row in zip(a_1, b_1, c_1):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(table_footer)
table_header = r'''	\begin{tabular}
		{S[table-format=1.0]
		 S[table-format=1.1]}
		\toprule
		{$a \mathbin{/} \unit{\milli\meter}$} & 
		{$c \mathbin{/} \unit{\milli\meter}$} \\
		\midrule
'''
table_footer = r''' 	\bottomrule
	\end{tabular}
'''
row_template = r'		{0:1.0f} & {1:1.1f} \\'
with open('build/table_speziell.tex', 'w') as f:
	f.write(table_header)
	for row in zip(a_2, c_2):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(table_footer)

# format and export calculated values to build directory
with open('build/m.tex', 'w') as f: 
	f.write(r'\num{')
	f.write(f'{par_2[0]:.2f}({err_2[0]:.2f})')
	f.write(r'}')
with open('build/n.tex', 'w') as f: 
	f.write(r'\qty{')
	f.write(f'{par_2[1]:.2f}({err_2[1]:.2f})')
	f.write(r'}{\milli\meter\squared}')
