import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import differential_evolution, linear_sum_assignment
import time
from copy import deepcopy
import sys
import ray
from psutil import cpu_count
from rus_comsol.rus_comsol import RUSComsol
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class RUSFitting:
    def __init__(self, rus_object, ranges,
                 freqs_file,
                 nb_freq_data, nb_freq_sim,
                 missing=True,
                 nb_workers=4,
                 population=15, N_generation=10000, mutation=0.7, crossing=0.9,
                 polish=False, updating='deferred', tolerance=0.01):
        """
        If nb_workers = 1, the calculation is in series
        """
        self.rus_object = rus_object
        self.init_pars  = deepcopy(self.rus_object.pars)
        self.best_pars  = deepcopy(self.rus_object.pars)
        self.last_gen   = None
        self.ranges     = ranges
        try:
            assert nb_freq_sim >= nb_freq_data
        except AssertionError:
            print("You need --- nb_freq_sim > nb_freq_data")
            sys.exit(1)
        self.nb_freq_data     = nb_freq_data
        self.nb_freq_sim      = nb_freq_sim
        self.freqs_file       = freqs_file
        self.missing          = missing
        self.freepars_name    = sorted(self.ranges.keys())
        self.fixedpars_name   = np.setdiff1d(sorted(self.init_pars.keys()), self.freepars_name)

        ## Create list of bounds
        self.bounds  = []
        for free_name in self.freepars_name:
            self.bounds.append((self.ranges[free_name][0],
                                         self.ranges[free_name][-1]))
        self.bounds = tuple(self.bounds)

        ## Differential evolution
        if nb_workers > cpu_count(logical=False):
            nb_workers = cpu_count(logical=False)
            print("!! Changed #workers to "
                  + str(nb_workers)
                  + " the #max of available cores !!")
        self._nb_workers   = nb_workers
        self.workers      = []
        self.pool         = None
        self.population   = population # (popsize = population * len(x))
        self.N_generation = N_generation
        self.mutation     = mutation
        self.crossing     = crossing
        self.polish       = polish
        self.updating     = updating
        self.tolerance    = tolerance

        ## Empty spaces
        self.best_chi2  = None
        self.nb_gens    = 0
        self.freqs_data = None # in MHz

        ## Load data
        self.load_data()

    ## Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def _get_nb_workers(self):
        return self._nb_workers
    def _set_nb_workers(self, nb_workers):
        if nb_workers > cpu_count(logical=False):
            nb_workers = cpu_count(logical=False)
            print("!! Changed #workers to "
                  + str(nb_workers)
                  + " the #max of available cores !!")
        self._nb_workers = nb_workers
    nb_workers = property(_get_nb_workers, _set_nb_workers)

    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def load_data(self):
        """
        Frequencies should be in MHz
        """
        ## Load the resonance data in MHz
        freqs_data = np.loadtxt(self.freqs_file, dtype="float", comments="#")
        ## Only select the first number of "freq to compare"
        self.freqs_data = freqs_data[:self.nb_freq_data]


    def assignement(self, freqs_data, freqs_sim):
        """
        Linear assigment of the simulated frequencies to the data frequencies
        in case there is one or more missing frequencies in the data
        """
        cost_matrix = distance_matrix(freqs_data[:, None], freqs_sim[:, None])**2
        index_sim = linear_sum_assignment(cost_matrix)[1]
        ## sim_index is the indices for freqs_sim to match freqs_data
        return index_sim, freqs_sim[index_sim]


    def sort_freqs(self, freqs_sim_calc):
        if self.missing == True:
            ## Linear assignement of the simulated frequencies to the data
            index_sim, freqs_sim = self.assignement(self.freqs_data, freqs_sim_calc)
            ## Give the missing frequencies in the data -------------------------
            # Let's remove the extra simulated frequencies that are beyond
            # the list of data frequencies
            bool_missing = np.ones(freqs_sim_calc.size, dtype=bool)
            bool_missing[index_sim] = False
            index_missing = np.arange(0, freqs_sim_calc.size, 1)[bool_missing]
            index_missing = index_missing[index_missing < self.freqs_data.size]
            freqs_missing = freqs_sim_calc[index_missing]
        else:
            ## Only select the first number of "freq to compare"
            freqs_sim = freqs_sim_calc[:self.nb_freq_data]
            freqs_missing = []
        return freqs_sim, freqs_missing


    def compute_chi2(self, freqs_calc_list):
        ## Remove the useless small frequencies
        chi2 = np.empty(len(freqs_calc_list), dtype=np.float64)
        freqs_missing_list = []
        for i, freqs_calc in enumerate(freqs_calc_list):
            ## Remove the first 6 bad frequencies
            # freqs_sim = freqs_sim[6:]
            freqs_calc = freqs_calc[freqs_calc > 1e-4]
            freqs_sim, freqs_missing = self.sort_freqs(freqs_calc)
            freqs_missing_list.append(freqs_missing)
            chi2[i] = np.sum((freqs_sim - self.freqs_data)**2)
        ## Best parameters for lowest chi2
        index_best = np.argmin(chi2)
        if self.best_chi2 == None or self.best_chi2 > chi2[index_best]:
            self.best_chi2 = chi2[index_best]
            for i, free_name in enumerate(self.freepars_name):
                self.best_pars[free_name][0] = self.last_gen[index_best][i]
            self.best_freqs_missing = freqs_missing_list[index_best]
        return chi2


    def generate_workers(self):
        for _ in range(self._nb_workers):
            worker = ray.remote(RUSComsol).remote(pars=self.init_pars,
                                        mph_file=self.rus_object.mph_file,
                                        nb_freq=self.nb_freq_sim,
                                        study_name=self.rus_object.study_name,
                                        init=True)
            self.workers.append(worker)
        self.pool = ray.util.ActorPool(self.workers)


    def update_worker(self, worker, value):
        ## Update pars with fit parameters
        pars = deepcopy(self.init_pars)
        for i, free_name in enumerate(self.freepars_name):
            pars[free_name][0] = value[i]
        return worker.compute_freqs.remote(pars) #, worker._get_pars.remote()


    def close_workers(self):
        for worker in self.workers:
            worker.stop_comsol.remote()


    def map(self, func, iterable):
        """
        It is because Pool.map from multiprocessing has the shape
        map(fun, iterable) that the arguments of this function are the same
        but func is not being used
        """
        self.last_gen = deepcopy(iterable)
        start_total_time = time.time()
        freqs_calc_list = self.pool.map(self.update_worker, iterable)
        chi2 = self.compute_chi2(list(freqs_calc_list))
        ## Print End Generation
        print("Gen "
              + str(self.nb_gens) + ":: Pop "
              + str(self.population * len(self.freepars_name))
              + " :: %.6s s" % (time.time() - start_total_time))
        self.nb_gens += 1
        ## Print Best member of the population
        for free_name in self.freepars_name:
            print(free_name
                  + " : "
                  + r"{0:.3f}".format(self.best_pars[free_name][0])
                  + " "
                  + self.best_pars[free_name][1])
        print("Missing frequencies --- ", self.best_freqs_missing, " MHz\n")
        return chi2


    def polish_func(self, single_value):
        freqs_calc_list = self.pool.map(self.update_worker, single_value)
        chi2 = self.compute_chi2(list(freqs_calc_list))
        return chi2[0]


    def run_fit(self):
        ## Start Ray
        ray.init(num_cpus=cpu_count(logical=False),
                 include_dashboard=False,
                 log_to_driver=False)

        self.generate_workers()
        print("--- Pool of "
              + str(self._nb_workers)
              + " workers for "
              + str(cpu_count(logical=False))
              + " cores ---")

        ## Start Stopwatch
        t0 = time.time()
        ## Run fit algorithm
        out = differential_evolution(self.polish_func, bounds=self.bounds,
                                    workers=self.map, updating=self.updating,
                                    polish=self.polish,
                                    maxiter=self.N_generation,
                                    popsize=self.population,
                                    mutation=self.mutation,
                                    recombination=self.crossing,
                                    tol=self.tolerance
                                    )

        ## Export final parameters from the fit
        for i, free_name in enumerate(self.freepars_name):
            self.best_pars[free_name][0] = out.x[i]
            self.rus_object.pars[free_name][0] = out.x[i]

        ## Fit report
        duration    = np.round(time.time() - t0, 2)
        N_points    = self.nb_freq_data
        N_variables = len(out.x)
        chi2 = out.fun
        reduced_chi2 = chi2 / (N_points - N_variables)

        print("\n")
        report = "#[[Fit Statistics]]" + "\n"
        report+= "\t# fit success        \t= " + str(out.success) + "\n"
        report+= "\t# fitting method     \t= " + "differential evolutin" + "\n"
        report+= "\t# generations        \t= " + str(out.nit) + " + 1" + "\n"
        report+= "\t# function evals     \t= " + str(out.nfev) + "\n"
        report+= "\t# data points        \t= " + str(N_points) + "\n"
        report+= "\t# variables          \t= " + str(N_variables) + "\n"
        report+= "\t# fit duration       \t= " + str(duration) + " seconds" + "\n"
        report+= "\t# chi-square         \t= " + r"{0:.8f}".format(chi2) + "\n"
        report+= "\t# reduced chi-square \t= " + r"{0:.8f}".format(reduced_chi2) + "\n"
        report+= "#[[Variables]]" + "\n"
        for i, free_name in enumerate(self.freepars_name):
            report+= "\t# " + free_name + " : " + r"{0:.3f}".format(out.x[i]) + " " + \
                     self.best_pars[free_name][1] + \
                     " (init = [" + str(self.ranges[free_name][0]) + \
                     ", " +         str(self.ranges[free_name][1]) + "])" + "\n"
        report+= "#[[Fixed values]]" + "\n"
        for fixedpars_name in self.fixedpars_name:
            report+= "\t# " + fixedpars_name + " : " + \
                     r"{0:.8f}".format(self.best_pars[fixedpars_name][0]) + " " + \
                     self.best_pars[fixedpars_name][1] + "\n"

        report += "#[[Missing frequencies]]\n"
        for freqs_missing in self.best_freqs_missing:
            report += r"{0:.4f}".format(freqs_missing) + " MHz\n"

        print(report)


        report_file = open(self.rus_object.mph_file[:-4] + "_fit_report.txt", "w")
        report_file.write(report)
        report_file.close()

        ## Close COMSOL for each workers
        self.close_workers()

        ## Stop Ray
        ray.shutdown()

        return self.rus_object



