from numpy.lib.function_base import angle
from rus_fitting.rus_comsol import RUSComsol
from rus_fitting.rus_fitting_scipy_leastsq import RUSSCIPYLEASTSQ
from scipy.spatial.transform import Rotation as R
import sys
from time import time
import os
import numpy as np


# elastic constants init in GPa
elastic_dict = {
    'c11': 320.934,
    'c12': 102.462,
    'c44': 124.991
    }

# bounds for fit; el const in GPa, angles in degrees
bounds_dict = {
    'c11': [290, 350],
    'c12': [70, 130],
    'c44': [95, 155]
    }

density    = 5110
nb_freq    = 50
nb_missing = 0
mesh       = 3
nb_workers = 25


report_name = f'fit_report.txt'
freqs_file  = "ResonanceList.dat"
mph_file    = 'comsol_file.mph'

rus_object = RUSComsol(cij_dict=elastic_dict, symmetry="cubic",
                       density =density,
                       mph_file=mph_file,
                       mesh=mesh,nb_freq=nb_freq)
rus_object.start_comsol()


fit = RUSSCIPYLEASTSQ(rus_object, bounds_dict,
                        freqs_file=freqs_file, nb_freqs=nb_freq, nb_max_missing=nb_missing,
                        use_Jacobian=False, tolerance=1e-10, report_name=report_name)

fit.run_fit(print_derivatives=False)

rus_object.stop_comsol()
