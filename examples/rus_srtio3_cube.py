from rus_comsol.rus_fitting import RUSFitting
from rus_comsol.rus_comsol import RUSComsol
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## Initial values
# init_member = {"c11": [321.49167, "GPa"],
#                 "c23": [103.52989, "GPa"],
#                 "c44": [124.91915, "GPa"]}
init_member = {"c11": [350, "GPa"],
               "c23": [71,  "GPa"],
               "c44": [124, "GPa"]}

## Ranges
# ranges_dict  = {"c11": [20, 1000],
#                 "c23": [30, 1000],
#                 "c44": [40, 1000]}
ranges_dict  = {"c11": [300, 350],
                "c23": [70, 130],
                "c44": [100, 150]
                }


rus_object = RUSComsol(pars=init_member,
                       mph_file="srtio3/rus_srtio3_cube.mph",
                       nb_freq=10)

fitObject = RUSFitting(rus_object=rus_object, ranges=ranges_dict,
                       freqs_file="srtio3/SrTiO3_RT_frequencies.dat",
                       nb_freq_data = 42, nb_freq_sim = 50,
                       nb_workers=2)
fitObject.run_fit()



