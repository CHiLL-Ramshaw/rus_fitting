from numpy.lib.function_base import angle
from time import time
from rus_fitting.rus_fitting_ray import RUSRAY
from rus_fitting.rus_xyz import RUSXYZ
from rus_fitting.smi_matrices import SMIMatrices
from scipy.spatial.transform import Rotation as R
import sys
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
    print('create SMI matrices')
    stokes_object = SMIMatrices(basis_order=order, surface_file_type='stl', surface_path=stl_path,
                       scale='mm', parallel=True, nb_processes=nb_workers,
                       E_path=E_path, I_path=I_path, integral_path=integral_path,
                       shift_com=True, find_good_rotation=False)
    stokes_object.create_G_E_matrices()



    print('create RUSXYZ object')
    rus_object = RUSXYZ(cij_dict=elastic_dict, symmetry='cubic', order=order,
                    Emat_path=E_path, Itens_path=I_path,
                    nb_freq=nb_freq, density=density,
                    angle_x=0, angle_y=0, angle_z=0,
                    init=True, use_quadrants=True)

    print('start fitting')
    fit = RUSRAY(rus_object, bounds_dict,
                    freqs_file=freqs_file, nb_freqs=nb_freq,
                    nb_max_missing=nb_missing, nb_workers=nb_workers, polish=True,
                    population=15, N_generation=1000, mutation=0.7, crossing=0.9,
                    tolerance=1e-3, updating='deferred',
                    report_name=report_name)
    
    fit.run_fit(print_derivatives=True)