import numpy as np
import time
from scipy.optimize import differential_evolution
from multiprocessing import cpu_count
from multiprocessing import Pool

class FitClass:
    def __init__(self, data, bounds):
        ## Initialize
        self.data = data
        self.bounds  = bounds

    def comput_chi2(self, pars_array):
        return np.sum((pars_array - self.data)**2)

    def __call__(self, pars_array):
        return self.comput_chi2(pars_array)



if __name__ == '__main__':
    data = np.array([340, 82, 149])
    bounds = ((-20000,20000), (-20000,20000), (-20000,20000))
    fitObject = FitClass(data, bounds)
    start_total_time = time.time()

    num_workers = 8
    pool = Pool(processes=num_workers)
    workers = pool.map
    print("--- Pool initialized with ", num_workers, " workers ---")

    out = differential_evolution(fitObject.comput_chi2, bounds=fitObject.bounds, workers=workers, updating='deferred')
    print("Done within %.6s seconds ----" % (time.time() - start_total_time))

################################################################################################