from rus_comsol.rus_fitting import RUSFitting
from rus_comsol.rus_comsol import RUSComsol
from rus_comsol.rus_rpr import RUSRPR
import sys
import os
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# get current directory
# print (os.getcwd())

## Initial values
elastic_dict = {"c11": 138.4,
                "c12": 42.2,
                "c13": 14.5,
                "c33": 194.4,
                "c44": 45.1,
                }

## Bounds
bounds_dict  = {"c11": [135, 142],
               "c12": [39, 45],
               "c13": [12, 17],
               "c33": [192, 197],
               "c44": [43, 47]
               }

# rus_object = RUSComsol(cij_dict=elastic_dict, symmetry="hexagonal",
#                        mph_file="mn3ge\\rus_mn3ge_2001B.mph",
#                         nb_freq=15)

rus_object = RUSRPR(cij_dict=elastic_dict, symmetry="hexagonal",
                        dimensions=[0.911e-3,1.02e-3,1.305e-3],
                        mass = 8.9e-6,
                        order = 12,
                        nb_freq=84)

# rus_object.start_comsol()
# print(rus_object.compute_freqs())
fitObject = RUSFitting(rus_object=rus_object, bounds_dict=bounds_dict,
                        freqs_file='examples\\mn3ge\\Mn3Ge_2001B_frequency_list.dat',
                        # freqsz_file='mn3ge\\Mn3Ge_2001B_frequency_list.dat',
                        nb_freqs='all',
                        nb_workers=30, nb_max_missing=5, polish=False)

fitObject.run_fit()
# fitObject.print_logarithmic_derivative()

# rus_object.initialize()
# print(rus_object.log_derivatives_analytical())

# rus_object.start_comsol()
# rus_object.log_derivatives()