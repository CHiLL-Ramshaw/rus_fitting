import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import differential_evolution, linear_sum_assignment
import time
from copy import deepcopy
import sys
import ray
from psutil import cpu_count
from rus_comsol.rus_comsol import RUSComsol
from rus_comsol.rus_rpr import RUSRPR
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class RUSFitting:
    def __init__(self, rus_object, bounds_dict,
                 freqs_file,
                 nb_freqs,
                 nb_max_missing=0,
                 nb_workers=4,
                 report_name="",
                 population=15, N_generation=10000, mutation=0.7, crossing=0.9,
                 polish=False, updating='deferred', tolerance=0.01):
        self.rus_object  = rus_object
        self.init_pars   = deepcopy(self.rus_object.cij_dict)
        self.init_pars["angle_x"] = self.rus_object.angle_x
        self.init_pars["angle_y"] = self.rus_object.angle_y
        self.init_pars["angle_z"] = self.rus_object.angle_z
        self.best_pars   = deepcopy(self.init_pars)
        self.last_gen    = None
        self.bounds_dict = bounds_dict

        ## Load data
        self.nb_freqs       = nb_freqs
        self.nb_max_missing = nb_max_missing
        self.freqs_file     = freqs_file
        self.col_freqs      = 0
        self.freqs_data     = self.load_data()
        self.free_pars_name  = sorted(self.bounds_dict.keys())
        self.fixed_pars_name = np.setdiff1d(sorted(self.init_pars.keys()),
                                             self.free_pars_name)

        ## Create tuple of bounds for scipy
        self.bounds  = []
        for free_name in self.free_pars_name:
            self.bounds.append((self.bounds_dict[free_name][0],
                                self.bounds_dict[free_name][1]))
        self.bounds = tuple(self.bounds)

        ## Differential evolution
        if nb_workers > cpu_count(logical=False):
            nb_workers = cpu_count(logical=False)
            print("!! Changed #workers to "
                  + str(nb_workers)
                  + " the #max of available cores !!")
        self._nb_workers   = nb_workers
        self.workers       = []
        self.pool          = None
        self.ray_init_auto = True
        self.population    = population # (popsize = population * len(x))
        self.N_generation  = N_generation
        self.mutation      = mutation
        self.crossing      = crossing
        self.polish        = polish
        self.updating      = updating
        self.tolerance     = tolerance

        self.report_name = report_name

        ## Empty spaces
        self.best_chi2 = None
        self.nb_gens   = 0
        self.best_freqs_calc = []
        self.best_index_missing = []
        self.best_freqs_missing = []


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
    def load_data(self, nb_freq=None):
        """
        Frequencies should be in MHz
        """
        if nb_freq is None:
            nb_freq = self.nb_freqs
        ## Load the resonance data in MHz
        freqs_data = np.loadtxt(self.freqs_file, dtype="float", comments="#")
        if freqs_data.size is tuple:
            freqs_data = freqs_data[:,self.col_freqs]
        ## Only select the first number of "freq to compare"
        if nb_freq == 'all':
            self.nb_freqs = nb_freq = len(freqs_data)
        try:
            assert nb_freq <= freqs_data.size
        except AssertionError:
            print("You need --- nb calculated freqs <= nb data freqs")
            sys.exit(1)
        return freqs_data[:nb_freq]


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
        if self.nb_max_missing != 0:
            ## Linear assignement of the simulated frequencies to the data
            index_sim, freqs_sim = self.assignement(self.freqs_data,
                                                    freqs_sim_calc)
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
            freqs_sim = freqs_sim_calc[:self.nb_freqs]
            freqs_missing = []
            index_missing = []
        return freqs_sim, freqs_missing, index_missing


    def compute_chi2(self, freqs_calc_list):
        ## Remove the useless small frequencies
        chi2 = np.empty(len(freqs_calc_list), dtype=np.float64)
        freqs_missing_list = []
        index_missing_list = []
        for i, freqs_calc in enumerate(freqs_calc_list):
            freqs_sim, freqs_missing, index_missing = self.sort_freqs(freqs_calc)
            freqs_missing_list.append(freqs_missing)
            index_missing_list.append(index_missing)
            chi2[i] = np.sum((freqs_sim - self.freqs_data)**2)
        ## Best parameters for lowest chi2
        index_best = np.argmin(chi2)
        if self.best_chi2 == None or self.best_chi2 > chi2[index_best]:
            self.best_chi2 = chi2[index_best]
            for i, free_name in enumerate(self.free_pars_name):
                self.best_pars[free_name] = self.last_gen[index_best][i]
            self.best_freqs_missing = freqs_missing_list[index_best]
            self.best_freqs_calc = freqs_calc_list[index_best]
            self.best_index_missing = index_missing_list[index_best]
        return chi2


    def generate_workers(self):
        if isinstance(self.rus_object, RUSRPR):
            self.rus_object.initialize()
        for _ in range(self._nb_workers):
            if isinstance(self.rus_object, RUSComsol):
                worker = ray.remote(RUSComsol).remote(cij_dict=self.rus_object.cij_dict,
                                        symmetry=self.rus_object.symmetry,
                                        mph_file=self.rus_object.mph_file,
                                        nb_freq=self.nb_freqs+self.nb_max_missing,
                                        angle_x=self.rus_object.angle_x,
                                        angle_y=self.rus_object.angle_y,
                                        angle_z=self.rus_object.angle_z,
                                        study_name=self.rus_object.study_name,
                                        init=True)

            if isinstance(self.rus_object, RUSRPR):
                worker = ray.remote(RUSRPR).remote(cij_dict=self.rus_object.cij_dict,
                                        symmetry=self.rus_object.symmetry,
                                        mass=self.rus_object.mass,
                                        dimensions=self.rus_object.dimensions,
                                        order=self.rus_object.order,
                                        nb_freq=self.nb_freqs+self.nb_max_missing,
                                        angle_x=self.rus_object.angle_x,
                                        angle_y=self.rus_object.angle_y,
                                        angle_z=self.rus_object.angle_z,
                                        init=False)
                worker.copy_object.remote(self.rus_object)
            self.workers.append(worker)
        self.pool = ray.util.ActorPool(self.workers)


    def update_worker(self, worker, value):
        ## Update cij with fit parameters
        for i, free_name in enumerate(self.free_pars_name):
            if free_name not in ["angle_x", "angle_y", "angle_z"]:
                worker.set_cij_value.remote(free_name, value[i])
        ## Update angles
        for i, free_name in enumerate(self.free_pars_name):
            if   free_name=="angle_x":
                worker._set_angle_x.remote(value[i])
            elif free_name=="angle_y":
                worker._set_angle_y.remote(value[i])
            elif free_name=="angle_z":
                worker._set_angle_z.remote(value[i])
        return worker.compute_resonances.remote()


    def close_workers(self):
        if isinstance(self.rus_object, RUSComsol):
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
              + str(self.population * len(self.free_pars_name))
              + " :: %.6s s" % (time.time() - start_total_time))
        self.nb_gens += 1
        ## Print Best member of the population
        for free_name in self.free_pars_name:
            print(free_name
                  + " : "
                  + r"{0:.3f}".format(self.best_pars[free_name])
                  + " "
                  + " ")
        print("Missing frequencies --- ", self.best_freqs_missing, " MHz\n")
        return chi2


    def polish_func(self, single_value):
        freqs_calc_list = self.pool.map(self.update_worker, single_value)
        chi2 = self.compute_chi2(list(freqs_calc_list))
        return chi2[0]


    def ray_init(self, num_cpus=None):
        if num_cpus==None:
            num_cpus = cpu_count(logical=False)
        ray.init(num_cpus=num_cpus,
                 include_dashboard=False,
                 log_to_driver=False)

    def print_fit_report(self, fit_out, fit_start_time=0):
        duration    = np.round(time.time() - fit_start_time, 2)
        N_points    = self.nb_freqs
        N_variables = len(fit_out.x)
        chi2 = fit_out.fun
        reduced_chi2 = chi2 / (N_points - N_variables)
        # print("\n")
        report = "\n#[[Fit Statistics]]" + "\n"
        report+= "\t# fit success        \t= " + str(fit_out.success) + "\n"
        report+= "\t# fitting method     \t= " + "differential evolution" + "\n"
        report+= "\t# generations        \t= " + str(fit_out.nit) + " + 1" + "\n"
        report+= "\t# function evals     \t= " + str(fit_out.nfev) + "\n"
        report+= "\t# data points        \t= " + str(N_points) + "\n"
        report+= "\t# variables          \t= " + str(N_variables) + "\n"
        report+= "\t# fit duration       \t= " + str(duration) + " seconds" + "\n"
        report+= "\t# chi-square         \t= " + r"{0:.8f}".format(chi2) + "\n"
        report+= "\t# reduced chi-square \t= " + r"{0:.8f}".format(reduced_chi2) + "\n"
        report+= "#[[Variables]]" + "\n"
        for i, free_name in enumerate(self.free_pars_name):
            report+= "\t# " + free_name + " : " + r"{0:.3f}".format(fit_out.x[i]) + " " + \
                     " unit " + \
                     " (init = [" + str(self.bounds_dict[free_name]) + \
                     ", " +         "unit" + "])" + "\n"
        report+= "#[[Fixed values]]" + "\n"
        for fixed_pars_name in self.fixed_pars_name:
            report+= "\t# " + fixed_pars_name + " : " + \
                     r"{0:.8f}".format(self.best_pars[fixed_pars_name]) + " " + \
                     " unit " + "\n"

        report += "#[[Missing frequencies]]\n"
        for freqs_missing in self.best_freqs_missing:
            report += r"{0:.4f}".format(freqs_missing) + " MHz\n"

        # print(report)
        return (report)


    def print_best_frequencies (self, freqs_calc=None, nb_additional_freqs=10, comsol_start=True):
        freqs_data = np.array(self.load_data())

        if freqs_calc is None:
            if isinstance(self.rus_object, RUSComsol):
                if comsol_start == True:
                    self.rus_object.start_comsol()
                    freqs_calc = np.array(self.rus_object.compute_resonances(nb_freq=self.nb_freqs+self.nb_max_missing+nb_additional_freqs))
                    self.rus_object.stop_comsol()
                else:
                    freqs_calc = np.array(self.rus_object.compute_resonances(nb_freq=self.nb_freqs+self.nb_max_missing+nb_additional_freqs))
            if isinstance(self.rus_object, RUSRPR):
                if self.rus_object.Emat is None:
                    self.rus_object.initialize()
                freqs_calc = np.array(self.rus_object.compute_resonances(nb_freq=self.nb_freqs+self.nb_max_missing+nb_additional_freqs))

        index_missing = self.best_index_missing
        for idx in index_missing:
            freqs_data = np.insert(freqs_data, idx, 0)

        diff = np.zeros(len(freqs_data))
        for idx in np.arange(len(freqs_data)):
            if freqs_data[idx] != 0:
                diff[idx] = abs(freqs_data[idx]-freqs_calc[idx]) / freqs_data[idx] * 100
        rms = sum(diff[diff!=0]**2) / len(diff[diff!=0])

        template = "{0:<8}{1:<13}{2:<13}{3:<13}"
        compare_text = template.format(*['index', 'freq exp', 'freq calc', 'diff (%)']) + '\n'
        compare_text = compare_text + template.format(*['', '(MHz)', '(MHz)', '']) + '\n'
        compare_text = compare_text + '-'*(8+13+13+13) + '\n'
        for ii in np.arange(len(freqs_calc)):
            if ii < len(freqs_data):
                compare_text+= template.format(*[ii, round(freqs_data[ii],6), round(freqs_calc[ii],6), round(diff[ii], 3)]) + '\n'
            else:
                compare_text+= template.format(*[ii, '', round(freqs_calc[ii],6)], '') + '\n'
        compare_text = compare_text + '-'*(6+13+13+10) + '\n'
        compare_text = compare_text + 'RMS = ' + str(round(rms,3)) + ' %\n'
        compare_text = compare_text + '-'*(6+13+13+10) + '\n'

        # if print_bool==True:
        #     print (compare_text)
        return (compare_text)


    # def print_logarithmic_derivative (self, nb_additional_freqs=10, comsol_start=True):
    #     print ('start taking derivatives ...')
    #     if isinstance(self.rus_object, RUSComsol):
    #         if comsol_start == True:
    #             self.rus_object.start_comsol()
    #             log_der, freqs_calc = self.rus_object.log_derivatives_numerical(nb_freq=self.nb_freqs+self.nb_max_missing+nb_additional_freqs, return_freqs=True)
    #             self.rus_object.stop_comsol()
    #         else:
    #             log_der, freqs_calc = self.rus_object.log_derivatives_numerical(nb_freq=self.nb_freqs+self.nb_max_missing+nb_additional_freqs, return_freqs=True)
    #     if isinstance(self.rus_object, RUSRPR):
    #         if self.rus_object.Emat is None:
    #             self.rus_object.initialize()
    #         log_der, freqs_calc = self.rus_object.log_derivatives_analytical(nb_freq=self.nb_freqs+self.nb_max_missing+nb_additional_freqs, return_freqs=True)

    #     freq_text = self.print_best_frequencies (freqs_calc=freqs_calc, nb_additional_freqs=nb_additional_freqs, comsol_start=False)

    #     cij = deepcopy(sorted(self.rus_object.cij_dict))
    #     template = ""
    #     for i, _ in enumerate(cij):
    #         template += "{" + str(i) + ":<13}"
    #     header = ['2 x logarithmic derivative (2 x dlnf / dlnc)']+(len(cij)-1)*['']
    #     der_text = template.format(*header) + '\n'
    #     der_text = der_text + template.format(*cij) + '\n'
    #     der_text = der_text + '-'*13*len(cij) + '\n'
    #     for ii in np.arange(len(freq_text.split('\n'))):
    #         if ii < self.nb_freqs+len(self.best_freqs_missing):
    #             text = [str(round(log_der[ii,j], 6)) for j in np.arange(len(cij))]
    #             der_text = der_text + template.format(*text) + '\n'
    #         else:
    #             text = ['']*len(cij)
    #             der_text = der_text + template.format(*text) + '\n'

    #     total_text = ''
    #     for ii in np.arange(len(freq_text.split('\n'))):
    #         total_text = total_text + freq_text.split('\n')[ii] + der_text.split('\n')[ii] + '\n'
    #     # print(total_text)

    #     return total_text



    def save_report (self, report):
        if self.report_name == "":
            self.report_name = "fit_report.txt"
        report_file = open(self.report_name, "w")
        report_file.write(report)
        report_file.close()


    def run_fit(self):
        ## Start Ray
        if self.ray_init_auto == True:
            self.ray_init()

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
        for i, free_name in enumerate(self.free_pars_name):
            self.best_pars[free_name] = out.x[i]
            self.rus_object.cij_dict[free_name] = out.x[i]

        ## Close COMSOL for each workers
        self.close_workers()

        ## Stop Ray
        ray.shutdown()

        ## Fit report
        report = self.print_fit_report(out, fit_start_time=t0)
        self.save_report(report)
        print (report)
        logder = self.print_logarithmic_derivative()
        print(logder)

        return self.rus_object