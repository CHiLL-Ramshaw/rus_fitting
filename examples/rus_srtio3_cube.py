from rus_comsol.fitting_rus import FittingRUS
import time
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## Initial values
# init_member = {"c11": 321.49167,
#                 "c23": 103.52989,
#                 "c44": 124.91915}
init_member = {"c11": [350, "GPa"],
               "c23": [71,  "GPa"],
               "c44": [124, "GPa"]}

## Ranges
# ranges_dict  = {"c11": [20, 1000],
#                 "c23": [30, 1000],
#                 "c44": [40, 1000]}
ranges_dict  = {"c11": [300, 350],
                "c23": [70, 130],
                # "c44": [100, 150]
                }


# # Create
# fitObject = FittingRUS(init_member=init_member, ranges_dict=ranges_dict,
#                         freqs_file = "data/srtio3/SrTiO3_RT_frequencies.dat",
#                         mph_file="mph/srtio3/rus_srtio3_cube.mph",
#                         nb_freq_data = 42, nb_freq_sim = 50, missing=False,
#                         method="differential_evolution_scipy")
# fitObject.run_fit()


if __name__ == '__main__':
    fitObject = FittingRUS(init_member=init_member, ranges_dict=ranges_dict,
                            freqs_file = "srtio3/SrTiO3_RT_frequencies.dat",
                            mph_file="srtio3/rus_srtio3_cube.mph",
                            nb_freq_data = 42, nb_freq_sim = 50,
                            missing=False,
                            parallel=True, nb_workers=2)
    fitObject.run_fit()


# from scipy.optimize import differential_evolution
# from multiprocessing import cpu_count, Pool
# if __name__ == '__main__':
#     fitObject = FittingRUS(init_member=init_member, ranges_dict=ranges_dict,
#                             freqs_file = "data/srtio3/SrTiO3_RT_frequencies.dat",
#                             mph_file="mph/srtio3/rus_srtio3_cube.mph",
#                             nb_freq_data = 42, nb_freq_sim = 50,
#                             missing=False,
#                             parallel=True,
#                             method="differential_evolution_scipy")
#     start_total_time = time.time()
#     # fitObject.initiate_fit()
#     num_cpu = cpu_count()
#     num_workers = 10
#     pool = Pool(processes=num_workers, initializer=fitObject.initiate_fit)
#     workers = pool.map
#     print("--- Pool initialized with ", num_workers, " workers ---")

#     out = differential_evolution(fitObject.compute_chi2, bounds=fitObject.pars_bounds,
#                                          workers=workers, updating='deferred',
#                                          polish=False, maxiter=10000)
#     print("Done within %.6s seconds ----" % (time.time() - start_total_time))
#     print(out.x)
#     print(out.message)