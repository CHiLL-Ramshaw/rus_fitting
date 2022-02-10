import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import differential_evolution, linear_sum_assignment
import time
from copy import deepcopy
import os
import sys
from IPython.display import clear_output
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
        """
        freqs_files should contain experimental resonance frequencies in MHz and a weight
        """
        self.rus_object  = rus_object
        self.init_pars   = deepcopy(self.rus_object.cij_dict)
        self.init_pars["angle_x"] = self.rus_object.angle_x
        self.init_pars["angle_y"] = self.rus_object.angle_y
        self.init_pars["angle_z"] = self.rus_object.angle_z
        self.best_pars   = deepcopy(self.init_pars)
        self.last_gen    = None
        self.bounds_dict = bounds_dict

        ## Load data
        self.nb_freqs        = nb_freqs
        self.nb_max_missing  = nb_max_missing
        self.freqs_file      = freqs_file
        self.col_freqs       = 0
        self.col_weight      = 1
        self.freqs_data = None
        self.weight = None
        self.load_data()
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
        self.best_freqs_found   = []
        self.best_index_found   = []
        self.best_freqs_missing = []
        self.best_index_missing = []

        ## empty spaces for fit properties
        self.fit_output = None
        self.fit_duration = 0


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
        data = np.loadtxt(self.freqs_file, dtype="float", comments="#")
        if len(data.shape) > 1:
            freqs_data = data[:,self.col_freqs]
            # weight     = np.ones_like(freqs_data)
            weight     = data[:,self.col_weight]
        else:
            freqs_data = data
            weight     = np.ones_like(freqs_data)
        ## Only select the first number of "freq to compare"
        if self.nb_freqs == 'all':
            self.nb_freqs = len(freqs_data)
        try:
            assert self.nb_freqs <= freqs_data.size
        except AssertionError:
            print("You need --- nb calculated freqs <= nb data freqs")
            sys.exit(1)
        self.freqs_data = freqs_data[:self.nb_freqs]
        self.weight = weight[:self.nb_freqs]


    def assignement(self, freqs_data, freqs_sim):
        """
        Linear assigment of the simulated frequencies to the data frequencies
        in case there is one or more missing frequencies in the data
        """
        cost_matrix = distance_matrix(freqs_data[:, None], freqs_sim[:, None])**2
        index_found = linear_sum_assignment(cost_matrix)[1]
        ## sim_index is the indices for freqs_sim to match freqs_data
        return index_found, freqs_sim[index_found]


    def sort_freqs(self, freqs_sim):
        if self.nb_max_missing != 0:
            ## Linear assignement of the simulated frequencies to the data
            index_found, freqs_found = self.assignement(self.freqs_data, freqs_sim)
            ## Give the missing frequencies in the data -------------------------
            # Let's remove the extra simulated frequencies that are beyond
            # the list of data frequencies
            bool_missing = np.ones(freqs_sim.size, dtype=bool)
            bool_missing[index_found] = False
            index_missing = np.arange(0, freqs_sim.size, 1)[bool_missing]
            # index_missing = index_missing[index_missing < self.freqs_data.size]
            freqs_missing = freqs_sim[index_missing]
        else:
            ## Only select the first number of "freq to compare"
            freqs_found = freqs_sim[:self.nb_freqs]
            index_found = np.arange(len(freqs_found))
            freqs_missing = []
            index_missing = []
        return freqs_found, index_found, freqs_missing, index_missing


    def compute_chi2(self, freqs_sim_list):
        ## Remove the useless small frequencies
        chi2 = np.empty(len(freqs_sim_list), dtype=np.float64)
        freqs_found_list   = []
        index_found_list   = []
        freqs_missing_list = []
        index_missing_list = []
        for i, freqs_sim in enumerate(freqs_sim_list):
            freqs_found, index_found, freqs_missing, index_missing = self.sort_freqs(freqs_sim)
            freqs_found_list.append(freqs_found)
            index_found_list.append(index_found)
            freqs_missing_list.append(freqs_missing)
            index_missing_list.append(index_missing)
            chi2[i] = np.sum(((freqs_found - self.freqs_data)/freqs_found)**2 * self.weight)
        ## Best parameters for lowest chi2
        index_best = np.argmin(chi2)
        if self.best_chi2 == None or self.best_chi2 > chi2[index_best]:
            self.best_chi2 = chi2[index_best]
            for i, free_name in enumerate(self.free_pars_name):
                self.best_pars[free_name] = self.last_gen[index_best][i]
            self.best_freqs_found   = freqs_found_list[index_best]
            self.best_index_found   = index_found_list[index_best]
            self.best_freqs_missing = freqs_missing_list[index_best]
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
                                        density=self.rus_object.density,
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
        freqs_sim_list = self.pool.map(self.update_worker, iterable)
        chi2 = self.compute_chi2(list(freqs_sim_list))
        ## Print End Generation
        clear_output(wait=True)
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
        ## Save the report of the best parameters
        v_spacing = '#' + '-'*(79) + '\n'
        report  = self.report_best_pars()
        report += v_spacing
        report += self.report_best_freqs()
        self.save_report(report)
        return chi2


    def polish_func(self, single_value):
        if type(single_value[0])==list:
            freqs_sim_list = self.pool.map(self.update_worker, single_value)
        else:
            freqs_sim_list = self.pool.map(self.update_worker, [single_value])
        chi2 = self.compute_chi2(list(freqs_sim_list))
        index_best = np.argmin(chi2)
        return chi2[index_best]


    def ray_init(self, num_cpus=None):
        if num_cpus==None:
            num_cpus = cpu_count(logical=False)
        ray.init(num_cpus=num_cpus,
                 include_dashboard=False,
                 log_to_driver=False)


    def run_fit(self, print_derivatives=False):
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
        fit_output = differential_evolution(self.polish_func, bounds=self.bounds,
                                    workers=self.map, updating=self.updating,
                                    polish=self.polish,
                                    maxiter=self.N_generation,
                                    popsize=self.population,
                                    mutation=self.mutation,
                                    recombination=self.crossing,
                                    tol=self.tolerance
                                    )
        self.fit_output = fit_output
        self.fit_duration = time.time() - t0
        ## Export final parameters from the fit
        for i, free_name in enumerate(self.free_pars_name):
            self.best_pars[free_name] = fit_output.x[i]
            self.rus_object.cij_dict[free_name] = fit_output.x[i]
        ## Close COMSOL for each workers
        self.close_workers()
        ## Stop Ray
        ray.shutdown()
        ## Fit report
        clear_output(wait=True)
        if print_derivatives == False:
            v_spacing = '\n' + 79*'#' + '\n' + 79*'#' + '\n' + '\n'
            report  = v_spacing
            report += self.report_sample_text()
            report += v_spacing
            report += self.report_fit()
            report += v_spacing
            report += self.report_best_pars()
            report += v_spacing
            report += self.report_best_freqs()            
            print(report)
        else:
            report = self.report_total()
            print(report)
        self.save_report(report)
        return self.rus_object


    def report_best_pars(self):
        report = "#Variables" + '-'*(70) + '\n'
        for free_name in self.free_pars_name:
            if free_name[0] == "c": unit = "GPa"
            else: unit = "deg"
            report+= "\t# " + free_name + " : " + r"{0:.3f}".format(self.best_pars[free_name]) + " " + \
                     unit + \
                     " (init = [" + str(self.bounds_dict[free_name]) + \
                     ", " +         unit + "])" + "\n"
        report+= "#Fixed values" + '-'*(67) + '\n'
        if len(self.fixed_pars_name) == 0:
            report += "\t# " + "None" + "\n"
        else:
            for fixed_name in self.fixed_pars_name:
                if free_name[0] == "c": unit = "GPa"
                else: unit = "deg"
                report+= "\t# " + fixed_name + " : " + \
                        r"{0:.8f}".format(self.best_pars[fixed_name]) + " " + \
                        unit + "\n"
        # report += "#Missing frequencies" + '-'*(60) + '\n'
        # for freqs_missing in self.best_freqs_missing:
        #     report += "\t# " + r"{0:.4f}".format(freqs_missing) + " MHz\n"
        return report


    def report_fit(self):
        fit_output  = self.fit_output
        duration    = np.round(self.fit_duration, 2)
        N_points    = self.nb_freqs
        N_variables = len(self.bounds)
        chi2 = fit_output.fun
        reduced_chi2 = chi2 / (N_points - N_variables)
        report = "#Fit Statistics" + '-'*(65) + '\n'
        report+= "\t# fitting method     \t= " + "differential evolution" + "\n"
        report+= "\t# data points        \t= " + str(N_points) + "\n"
        report+= "\t# variables          \t= " + str(N_variables) + "\n"
        report+= "\t# fit success        \t= " + str(fit_output.success) + "\n"
        report+= "\t# generations        \t= " + str(fit_output.nit) + " + 1" + "\n"
        report+= "\t# function evals     \t= " + str(fit_output.nfev) + "\n"
        report+= "\t# fit duration       \t= " + str(duration) + " seconds" + "\n"
        report+= "\t# chi-square         \t= " + r"{0:.8f}".format(chi2) + "\n"
        report+= "\t# reduced chi-square \t= " + r"{0:.8f}".format(reduced_chi2) + "\n"
        return report


    def report_best_freqs(self, nb_additional_freqs=10):
        if (nb_additional_freqs != 0) or (self.best_freqs_found == []):
            if isinstance(self.rus_object, RUSComsol) and (self.rus_object.client is None):
                self.rus_object.start_comsol()
            if isinstance(self.rus_object, RUSRPR) and (self.rus_object.Emat is None):
                self.rus_object.initialize()
            self.rus_object.nb_freq = self.nb_freqs + len(self.best_index_missing) + nb_additional_freqs
            freqs_sim = self.rus_object.compute_resonances()
            freqs_found, index_found, freqs_missing, index_missing = self.sort_freqs(freqs_sim)
        else:
            freqs_found   = self.best_freqs_found
            index_found   = self.best_index_found
            freqs_missing = self.best_freqs_missing
            index_missing = self.best_index_missing
            # print(len(freqs_found) + len(freqs_missing))
            freqs_sim = np.empty(len(freqs_found) + len(freqs_missing))
            # print(freqs_sim.size)
            freqs_sim[index_found]   = freqs_found
            freqs_sim[index_missing] = freqs_missing

        freqs_data = np.empty(len(freqs_found) + len(freqs_missing))
        freqs_data[index_found] = self.freqs_data
        freqs_data[index_missing] = 0

        weight = np.empty(len(freqs_found) + len(freqs_missing))
        weight[index_found] = self.weight
        weight[index_missing] = 0

        diff = np.zeros_like(freqs_data)
        for i in range(len(freqs_data)):
            if freqs_data[i] != 0:
                diff[i] = np.abs(freqs_data[i]-freqs_sim[i]) / freqs_data[i] * 100 * weight[i]
        rms = np.sqrt(np.sum((diff[diff!=0])**2) / len(diff[diff!=0]))

        template = "{0:<8}{1:<13}{2:<13}{3:<13}{4:<8}"
        report  = template.format(*['#index', 'f exp(MHz)', 'f calc(MHz)', 'diff (%)', 'weight']) + '\n'
        report += '#' + '-'*(79) + '\n'
        for j in range(len(freqs_sim)):
            if j < len(freqs_data):
                report+= template.format(*[j, round(freqs_data[j],6), round(freqs_sim[j],6), round(diff[j], 3), round(weight[j], 0)]) + '\n'
            else:
                report+= template.format(*[j, '', round(freqs_sim[j],6)], '') + '\n'
        report += '#' + '-'*(79) + '\n'
        report += '# RMS = ' + str(round(rms,3)) + ' %\n'
        report += '#' + '-'*(79) + '\n'

        return report

    def report_sample_text(self):
        sample_template = "{0:<40}{1:<20}"
        sample_text = '# [[Sample Characteristics]] \n'
        sample_text += '# ' + sample_template.format(*['crystal symmetry:', self.rus_object.symmetry]) + '\n'
        if isinstance(self.rus_object, RUSRPR):
            sample_text += '# ' + sample_template.format(*['sample dimensions (mm) (x,y,z):', str(self.rus_object.dimensions*1e3)]) + '\n'
            sample_text += '# ' + sample_template.format(*['mass (mg):', self.rus_object.mass*1e6]) + '\n'
            sample_text += '# ' + sample_template.format(*['highest order basis polynomial:', self.rus_object.order]) + '\n'
            sample_text += '# ' + sample_template.format(*['resonance frequencies calculated with:', 'RUS_RPR']) + '\n'
        if isinstance(self.rus_object, RUSComsol):
            sample_text += '# ' + sample_template.format(*['Comsol file:', self.rus_object.mph_file]) + '\n'
            sample_text += '# ' + sample_template.format(*['resonance frequencies calculated with:', 'Comsol']) + '\n'
        return sample_text


    def report_total(self, comsol_start=True):
        report_fit = self.report_fit()
        report_fit += self.report_best_pars()
        if isinstance(self.rus_object, RUSRPR):
            freq_text  = self.report_best_freqs(nb_additional_freqs=10)
            der_text = self.rus_object.print_logarithmic_derivative(print_frequencies=False)
        if isinstance(self.rus_object, RUSComsol):
            freq_text  = self.report_best_freqs(nb_additional_freqs=10)
            der_text = self.rus_object.print_logarithmic_derivative(print_frequencies=False, comsol_start=False)
            self.rus_object.stop_comsol()

        sample_text = self.report_sample_text()

        data_text = ''
        freq_text_split = freq_text.split('\n')
        freq_text_prepend = [' '*len(freq_text_split[0])] + freq_text_split
        for j in np.arange(len(freq_text.split('\n'))):
            if j == 2 or j==len(freq_text.split('\n'))-3:
                data_text += '#' + '-'*119 +'\n'
            elif j < len(der_text.split('\n')):
                data_text += freq_text_prepend[j] + der_text.split('\n')[j] + '\n'
            else:
                if j==len(freq_text.split('\n'))-1:
                    data_text += '#' + '-'*119 +'\n'
                else:
                    data_text += freq_text_prepend[j] + '\n'

        v_spacing = '\n' + 120*'#' + '\n' + 120*'#' + '\n' + '\n'
        report_total = v_spacing + sample_text + v_spacing +\
                 report_fit + v_spacing + data_text

        return report_total


    def save_report(self, report):
        if self.report_name == "":
            self.report_name = "fit_report.txt"
        report_file = open(self.report_name, "w")
        report_file.write(report)
        report_file.close()


