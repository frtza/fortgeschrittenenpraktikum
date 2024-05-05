import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds, uarray as uar
from uncertainties import ufloat

#Magnetfeld der Erde
#funktion zur Berechnung des Magnetfeldes
def B(I, n, R):
    return const.mu_0 * (8 * I * n) / (np.sqrt(125) * R)
