#%%
# import packages
from numbers import Integral
from numpy.lib.function_base import angle
from rus_comsol.rus_fitting import RUSFitting
from rus_comsol.rus_lmfit import RUSLMFIT
from rus_comsol.rus_comsol import RUSComsol
from rus_comsol.rus_xyz import RUSXYZ
from rus_comsol.rpr_matrices import RPRMatrices
from rus_comsol.stokes_matrices import StokesMatrices
import sys
from time import time
import os
import numpy as np


if __name__ == '__main__':
    # print (os.getcwd())
    #%%
    # intilaize RUS object

    # elastic constants init in GPa
    elastic_dict = {"c11": 84.9,
                "c12": 26.6,
                "c13": 38.2,
                "c22": 139.6,
                "c23": 31.8,
                "c33": 91.1,
                "c44": 29.93,
                "c55": 52.20,
                "c66": 29.43
                }


    density    = 9190 # kg / m^3
    nb_freq    = 150
    order      = 16
    nb_workers = 6

    
    stl_path      = 'test_rot.stl'

    E_path        = f'rot_Emat.npy'
    I_path        = f'rot_Itnes.npy'
    integral_path = f'rot_int.npy'


    # stokes_object = StokesMatrices(order=order, surface_file_type='stl',
    #                         scale='mm', parallel=True, nb_processes=nb_workers,
    #                         surface_file_path=stl_path, move_com=True, rotate_mesh=True,
    #                         Emat_path=E_path, Itens_path =I_path,
    #                         integral_path=integral_path)
    # stokes_object.create_G_E_matrices()


    rus_object = RUSXYZ(cij_dict=elastic_dict, symmetry='orthorhombic', order=order,
                    load_matrices=True, #matrix_object=rpr_object,
                    Emat_path=E_path, Itens_path=I_path,
                    nb_freq=nb_freq, density=density,
                    angle_x=-20.9018242, angle_y=-11.009141853, angle_z=-13.9455566844,
                    init=True, use_quadrants=False)
    f1 = rus_object.compute_resonances()


    E_path        = f'no_rot_Emat.npy'
    I_path        = f'no_rot_Itnes.npy'
    integral_path = f'no_rot_int.npy'

    rus_object = RUSXYZ(cij_dict=elastic_dict, symmetry='orthorhombic', order=order,
                    load_matrices=True, #matrix_object=rpr_object,
                    Emat_path=E_path, Itens_path=I_path,
                    nb_freq=nb_freq, density=density,
                    angle_x=0, angle_y=0, angle_z=0,
                    init=True, use_quadrants=False)
    f2 = rus_object.compute_resonances()

    print(f1-f2)

  

# %%
