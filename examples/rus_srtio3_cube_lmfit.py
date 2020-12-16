from rus_comsol.fitting_rus_lmfit import FittingRUS
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## Initial values
init_member = {"c11": 321.49167,
                "c23": 103.52989,
                "c44": 124.91915}

## Ranges
ranges_dict  = {"c11": [300, 350],
                "c23": [70, 130],
                "c44": [100, 150]}

## Create
fitObject = FittingRUS(init_member=init_member, ranges_dict=ranges_dict,
                        freqs_file = "data/srtio3/SrTiO3_RT_frequencies.dat",
                        mph_file="mph/srtio3/rus_srtio3_cube.mph",
                        nb_freq = 42)
fitObject.run_fit()