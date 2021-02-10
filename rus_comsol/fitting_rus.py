import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import differential_evolution, linear_sum_assignment
from multiprocessing import cpu_count, Pool
# from pathos.multiprocessing import ProcessingPool as Pool
import mph
import time
from copy import deepcopy
import sys
import json

##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class FittingRUS:
    def __init__(self, init_member, ranges_dict,
                 freqs_file, mph_file,
                 nb_freq_data, nb_freq_sim,
                 missing=True,
                 study_name="resonances",
                 method="differential_evolution", parallel=False, nb_workers=1,
                 population=15, N_generation=10000, mutation=0.1, crossing=0.9,
                 polish=False, updating='deferred', tolerance = 0.01,
                 **trash):
        ## Initialize
        self.init_member  = deepcopy(init_member)
        self.member       = deepcopy(init_member)
        self.ranges_dict  = ranges_dict
        try:
            assert nb_freq_sim >= nb_freq_data
        except AssertionError:
            print("You need --- nb_freq_sim > nb_freq_data")
            sys.exit(1)
        self.nb_freq_data = nb_freq_data
        self.nb_freq_sim  = nb_freq_sim
        self.freqs_file   = freqs_file
        self.mph_file     = mph_file
        self.missing      = missing
        self.study_name   = study_name
        self.method       = method # "shgo", "differential_evolution", "leastsq"
        self.pars_name    = sorted(self.ranges_dict.keys())
        self.fixed_name   = np.setdiff1d(sorted(self.init_member.keys()), self.pars_name)

        ## Create list of bounds
        self.pars_bounds  = []
        for param_name in self.pars_name:
            self.pars_bounds.append((ranges_dict[param_name][0], ranges_dict[param_name][-1]))
        self.pars_bounds = tuple(self.pars_bounds)

        ## Differential evolution
        self.parallel     = parallel
        self.nb_workers   = nb_workers
        self.population   = population # (popsize = population * len(x))
        self.N_generation = N_generation
        self.mutation     = mutation
        self.crossing     = crossing
        self.polish       = polish
        self.updating     = updating
        self.tolerance    = tolerance

        ## Empty spaces
        self.nb_calls   = 0
        self.json_name  = None
        self.freqs_data = None # in MHz
        # self.client     = None
        # self.model      = None

        ## Load data
        self.load_data()


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


    def objective_func(self):
        start_total_time = time.time()

        ## Update elastic constants --------------------------------------------------
        for param_name in self.pars_name:
            model.parameter(param_name, str(self.member[param_name][0]) +  "[" + self.member[param_name][1] + "]")

        ## Compute resonances --------------------------------------------------------
        model.solve(self.study_name)
        freqs_sim_calc = model.evaluate('abs(freq)', 'MHz')
        ## Remove the useless small frequencies
        freqs_sim_calc = freqs_sim_calc[freqs_sim_calc > 1e-4]
        model.clear()
        model.reset()

        if self.missing == True:
            ## Linear assignement of the simulated frequencies to the data
            index_sim, freqs_sim = self.assignement(self.freqs_data, freqs_sim_calc)

            ## Give the missing frequencies in the data -----------------------------
            # Let's remove the extra simulated frequencies that are beyond
            # the list of data frequencies
            bool_missing = np.ones(freqs_sim_calc.size, dtype=bool)
            bool_missing[index_sim] = False
            index_missing = np.arange(0, freqs_sim_calc.size, 1)[bool_missing]
            index_missing = index_missing[index_missing < self.freqs_data.size]
            freqs_missing = freqs_sim_calc[index_missing]

            print("Missing frequencies ---", freqs_missing, " MHz")

        else:
            ## Remove the first 6 bad frequencies
            # freqs_sim = freqs_sim[6:]
            freqs_sim = freqs_sim_calc[freqs_sim_calc > 1e-4]

            ## Only select the first number of "freq to compare"
            freqs_sim = freqs_sim[:self.nb_freq_data]

        self.nb_calls += 1
        print("---- call #" + str(self.nb_calls) + " in %.6s seconds ----" % (time.time() - start_total_time))
        sys.stdout.flush()

        return freqs_sim


    def compute_chi2(self, pars_array):
        ## Update member with fit parameters
        for i, param_name in enumerate(self.pars_name):
            self.member[param_name][0] = pars_array[i]
            print(param_name + " : " + "{0:g}".format(self.member[param_name][0]) + " " + self.member[param_name][1])
        freqs_sim = self.objective_func()
        # freqs_sim = pars_array
        # self.freqs_data = np.array([321, 103])
        return np.sum((freqs_sim - self.freqs_data)**2)


    def initiate_fit(self):
        ## Initialize the COMSOL file
        global client
        global model
        client = mph.Client()
        model = client.load(self.mph_file)
        for param_name in self.pars_name:
            model.parameter(param_name, str(self.member[param_name][0]) +  "[" + self.member[param_name][1] + "]")

        ## Modifying COMSOL parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        model.parameter('nb_freq', str(self.nb_freq_sim + 6))

        if self.parallel == True:
            print(">>>>  Worker Init <<<<")
            sys.stdout.flush()


    def run_fit(self):

        ## Initialize client
        if self.parallel == True:
            print("# of available cores: ", cpu_count())
            pool = Pool(processes=self.nb_workers, initializer=self.initiate_fit)
            workers = pool.map
            print("--- Pool initialized with ", self.nb_workers, " workers ---")
        else:
            self.initiate_fit()
            workers = 1

        ## Initialize number of calls
        self.nb_calls = 0

        ## Start Stopwatch
        t0 = time.time()

        ## Run fit algorithm
        out = differential_evolution(self.compute_chi2, bounds=self.pars_bounds,
                                     workers=workers, updating=self.updating,
                                     polish=self.polish,
                                     maxiter=self.N_generation,
                                     popsize=self.population,
                                     mutation=self.mutation,
                                     recombination=self.crossing,
                                     tol=self.tolerance
                                     )

        ## Export final parameters from the fit
        for i, param_name in enumerate(self.pars_name):
            self.member[param_name][0] = out.x[i]

        ## Fit report
        duration    = np.round(time.time() - t0, 2)
        N_points    = self.nb_freq_data
        N_variables = len(out.x)
        chi2 = out.fun
        reduced_chi2 = chi2 / (N_points - N_variables)

        print("\n")
        report = "#[[Fit Statistics]]" + "\n"
        report+= "\t# fit success        \t= " + str(out.success) + "\n"
        report+= "\t# fitting method     \t= " + self.method + "\n"
        report+= "\t# generations        \t= " + str(out.nit) + " + 1" + "\n"
        report+= "\t# function evals     \t= " + str(out.nfev) + "\n"
        report+= "\t# data points        \t= " + str(N_points) + "\n"
        report+= "\t# variables          \t= " + str(N_variables) + "\n"
        report+= "\t# fit duration       \t= " + str(duration) + " seconds" + "\n"
        report+= "\t# chi-square         \t= " + r"{0:.8f}".format(chi2) + "\n"
        report+= "\t# reduced chi-square \t= " + r"{0:.8f}".format(reduced_chi2) + "\n"
        report+= "#[[Variables]]" + "\n"
        for i, param_name in enumerate(self.pars_name):
            report+= "\t# " + param_name + " : " + r"{0:.8f}".format(out.x[i]) + " " + \
                     self.member[param_name][1] + \
                     " (init = [" + str(self.ranges_dict[param_name][0]) + \
                     ", " +         str(self.ranges_dict[param_name][1]) + "])" + "\n"
        report+= "#[[Fixed values]]" + "\n"
        for fixed_name in self.fixed_name:
            report+= "\t# " + fixed_name + " : " + \
                     r"{0:.8f}".format(self.member[fixed_name][0]) + " " + \
                     self.member[fixed_name][1] + "\n"

        print(report)

        report_file = open(self.mph_file[:-4] + "_fit_report.txt", "w")
        report_file.write(report)
        report_file.close()

        # ## Close COMSOL file without saving solutions in the file
        # if self.parallel != True:
        #     client.clear()

        ## Close pool
        if self.parallel == True:
            pool.terminate()





