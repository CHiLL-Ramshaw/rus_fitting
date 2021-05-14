from rus_comsol.rus_fitting import RUSFitting
from rus_comsol.rus_comsol import RUSComsol
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## Initial values
elastic_dict = {"c11": 321.49167,
                "c12": 103.52989,
                "c44": 124.91915,
                }

## Bounds
bounds_dict  = {"c11": [300, 350],
               "c12": [70, 130],
               "c44": [100, 150],
               "angle_x": [-10, 10],
               "angle_y": [-10, 10],
               "angle_z": [-10, 10],
               }

rus_object = RUSComsol(cij_dict=elastic_dict, symmetry="cubic",
                       mph_file="srtio3_2104A/rus_srtio3_2104A.mph",
                       nb_freq=51)

# rus_object.start_comsol()
# print(rus_object.compute_freqs())
fitObject = RUSFitting(rus_object=rus_object, bounds_dict=bounds_dict,
                        freqs_file="srtio3_2104A/SrTiO3_2104A_frequency_list.dat",
                        nb_freqs=51,
                        nb_workers=30, nb_max_missing=5)
fitObject.run_fit()
