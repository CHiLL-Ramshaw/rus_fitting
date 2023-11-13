import numpy as np
from numpy.core.numeric import ones_like
import os
##############################################################################
# to run a fit you need a file of resonances which has two columns:
#   - first column: resonance frequencies in MHz
#   - second column: weight - number between 0 and 1
#       - 1: res is fully taken into account
#       - 0: res is not used for fit; it is recognized that there is a resonance at this spot, but it's value is not used in RMS
# this script converts a file which is only a list of resonances in units "data_units" (Hz, kHz, or MHz) to the required format with 1 as every weight
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
