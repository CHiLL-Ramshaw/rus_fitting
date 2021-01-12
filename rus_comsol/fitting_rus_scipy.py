import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import differential_evolution, linear_sum_assignment
import mph
import time
from copy import deepcopy
from lmfit import minimize, Parameters, report_fit
import sys

##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class FittingRUS:
    def __init__(self, init_member, ranges_dict,
                 freqs_file, mph_file,
                 nb_freq_data, nb_freq_sim,
                 missing=True,
                 study_name="resonances",
                 method="differential_evolution",
                 population=100, N_generation=20, mutation_s=0.1, crossing_p=0.9,
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
        self.pars         = Parameters()
        for param_name, param_range in self.ranges_dict.items():
            self.pars.add(param_name, value = self.init_member[param_name], min = param_range[0], max = param_range[-1])

        ## Differential evolution
        self.population   = population
        self.N_generation = N_generation
        self.mutation_s   = mutation_s
        self.crossing_p   = crossing_p

        ## Empty spaces
        self.nb_calls   = 0
        self.json_name  = None
        self.freqs_data = None # in MHz

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


    def compute_diff(self, pars):
        """
        Compute diff = freqs_sim - freqs_data
        """
        self.pars = pars

        start_total_time = time.time()

        ## Update member with fit parameters
        for param_name in self.ranges_dict.keys():
            self.member[param_name] = self.pars[param_name].value
            print(param_name + " : " + "{0:g}".format(self.pars[param_name].value) + " GPa")

        ## Update elastic constants --------------------------------------------------
        for param_name in self.ranges_dict.keys():
            self.model.parameter(param_name, str(self.pars[param_name].value)+"[GPa]")

        ## Compute resonances --------------------------------------------------------
        self.model.solve(self.study_name)
        freqs_sim_calc = self.model.evaluate('abs(freq)', 'MHz')
        ## Remove the useless small frequencies
        freqs_sim_calc = freqs_sim_calc[freqs_sim_calc > 1e-4]
        self.model.clear()
        self.model.reset()

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


        return freqs_sim - self.freqs_data


    def initiate_fit(self):
        ## Initialize the COMSOL file
        self.client = mph.Client()
        self.model = self.client.load(self.mph_file)
        for param_name, param_value in self.init_member.items():
            self.model.parameter(param_name, str(param_value)+"[GPa]")
        ## Modifying COMSOL parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.model.parameter('nb_freq', str(self.nb_freq_sim + 6))


    def run_fit(self):

        ## Initialize client
        self.initiate_fit()

        ## Initialize number of calls
        self.nb_calls = 0

        ## Run fit algorithm
        if self.method=="least_square":
            out = minimize(self.compute_diff, self.pars)
        if self.method=="shgo":
            out = minimize(self.compute_diff, self.pars,
                           method='shgo') # , sampling_method='sobol', options={"f_tol": 1e-16}, n = 100, iters=20)
        if self.method=="differential_evolution":
            out = minimize(self.compute_diff, self.pars,
                           method='differential_evolution')
        if self.method=="ampgo":
            out = minimize(self.compute_diff, self.pars,
                           method='ampgo')
        else:
            print("This method does not exist in the class")

        ## Display fit report
        report_fit(out)

        ## Export final parameters from the fit
        for param_name in self.ranges_dict.keys():
            self.member[param_name] = out.params[param_name].value

        ## Close COMSOL file without saving solutions in the file
        self.client.clear()




if __name__ == '__main__':
    init_member = {"c11": 321.49167,
                   "c23": 103.52989,
                   "c44": 124.91915,
                   }
    ranges_dict  = {"c11": [300, 350],
                    "c23": [70, 130],
                    "c44": [100, 150]
                    }

    fitObject = FittingRUS(init_member=init_member, ranges_dict=ranges_dict,
                           freqs_file = "../examples/data/srtio3/SrTiO3_RT_frequencies.dat",
                           mph_file="../examples/srtio3/mph/rus_srtio3_cube.mph",
                           nb_freq_data = 10, nb_freq_sim = 12)
    fitObject.run_fit()

