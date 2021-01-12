from rus_comsol.fitting_rus_lmfit import FittingRUS
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## Initial values
# init_member = {"c11": 321.49167,
#                 "c23": 103.52989,
#                 "c44": 124.91915}
init_member = {"c11": 350,
                "c23": 71,
                "c44": 149}

## Ranges
ranges_dict  = {"c11": [20, 1000],
                "c23": [30, 1000],
                "c44": [40, 1000]}
# ranges_dict  = {"c11": [300, 350],
#                 "c23": [70, 130],
#                 "c44": [100, 150]}


## Create
if __name__ == '__main__':
    fitObject = FittingRUS(init_member=init_member, ranges_dict=ranges_dict,
                            freqs_file = "data/srtio3/SrTiO3_RT_frequencies_missing1.dat",
                            mph_file="mph/srtio3/rus_srtio3_cube.mph",
                            nb_freq_data = 6, nb_freq_sim = 10, missing=True,
                            method="differential_evolution_scipy", parallel=True)
    fitObject.run_fit()