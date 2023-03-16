#%%
from numpy.lib.function_base import angle
from rus_fitting.rus_fitting_lmfit import RUSLMFIT
from rus_fitting.rus_xyz import RUSXYZ
from rus_fitting.rpr_matrices import RPRMatrices
import sys
from time import time
import os
import numpy as np


# elastic constants init in GPa
elastic_dict = {
    'c11': 316.5660020863,
    'c12': 102.8230023146,
    'c44': 121.9140006323
    }

# bounds for fit; el const in GPa, angles in degrees
bounds_dict = {
    'c11': [290, 350],
    'c12': [70, 130],
    'c44': [95, 155]
    }

density    = 5110
dimensions = np.array([1.49e-3, 2.035e-3, 3.02e-3])
nb_freq    = 50
nb_missing = 5
order      = 14

nb_workers = 6

E_path        = f'Emat_order_{order}.npy'
I_path        = f'Itens_order_{order}.npy'
integral_path = f'integrals_order_{order}.npy'

stl_path      = 'surface_mesh.stl'
freqs_file    = "ResonanceList.dat"
report_name   = 'fit_report.txt'


if __name__ == '__main__':
    print('create RPR matrices')
    stokes_object = RPRMatrices(order=order, dimensions=dimensions, Emat_path=E_path, Itens_path=I_path)
    stokes_object.create_G_E_matrices()
    
    
    
    print('create RUSXYZ object')
    rus_object = RUSXYZ(cij_dict=elastic_dict, symmetry='cubic', order=order,
                    Emat_path=E_path, Itens_path=I_path,
                    nb_freq=nb_freq, density=density,
                    angle_x=0, angle_y=0, angle_z=0,
                    init=True, use_quadrants=True)
    
    print('start fitting')
    lmfit = RUSLMFIT(rus_object, bounds_dict,
                    freqs_file=freqs_file, nb_freqs=nb_freq,
                    nb_max_missing=nb_missing,
                    report_name=report_name,
                    method='differential_evolution', tolerance=1e-3,
                    population=15, mutation=0.7, N_generation=1000, 
                    crossing=0.9, polish=True, updating='immediate')

    lmfit.run_fit(print_derivatives=True)