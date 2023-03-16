import numpy as np
from numpy.core.numeric import ones_like
import os
##############################################################################

print(os.getcwd())

file = "SrTiO3_RT.dat"
# what are the current units of the data?
data_units = "kHz"

## Units dictionnary
units = {}
units["Hz"] = 1e0
units["kHz"] = 1e3
units["MHz"] = 1e6

## Load
data = np.genfromtxt(file, dtype="float", comments="#", skip_header=0, usecols=0)
data = data * units[data_units]

## Save
Data = np.vstack((data /  1e6, np.ones_like(data))).transpose()
np.savetxt(file[:-4] + "_for_fit.dat", Data, fmt='%.8f\t%i', header="res(MHz)\tweight", delimiter="")
