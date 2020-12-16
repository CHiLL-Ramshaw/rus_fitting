import numpy as np
import mph
import time
import random
import os
from copy import deepcopy
from lmfit import minimize, Parameters, report_fit

##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class FittingRUS:
    def __init__(self, init_member, ranges_dict,
                 freqs_file, mph_file,
                 nb_freq,
                 study_name="resonances",
                 method="differential_evolution",
                 population=100, N_generation=20, mutation_s=0.1, crossing_p=0.9,
                 **trash):
        ## Initialize
        self.init_member = deepcopy(init_member)
        self.member      = deepcopy(init_member)
        self.best_member = deepcopy(init_member)
        self.ranges_dict = ranges_dict
        self.nb_freq     = nb_freq
        self.freqs_file  = freqs_file
        self.mph_file    = mph_file
        self.study_name  = study_name
        self.method      = method # "shgo", "differential_evolution", "leastsq"
        self.pars        = Parameters()
        for param_name, param_range in self.ranges_dict.items():
            self.pars.add(param_name, value = self.init_member[param_name], min = param_range[0], max = param_range[-1])

        ## Differential evolution
        self.population   = population
        self.N_generation = N_generation
        self.mutation_s   = mutation_s
        self.crossing_p   = crossing_p
        self.seed         = 1

        ## Empty spaces
        self.nb_calls   = 0
        self.n_better   = 0
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
        self.freqs_data = freqs_data[:self.nb_freq]

    # def compute_diff(self, pars):
    #     """Compute diff = freqs_sim - freqs_data"""

    #     self.pars = pars

    #     start_total_time = time.time()

    #     ## Update member with fit parameters
    #     for param_name in self.ranges_dict.keys():
    #         self.member[param_name] = self.pars[param_name].value
    #         print(param_name + " : " + "{0:g}".format(self.pars[param_name].value) + " GPa")

    #     ## Update elastic constants --------------------------------------------------
    #     for param_name in self.ranges_dict.keys():
    #         self.model.parameter(param_name, str(self.pars[param_name].value)+"[GPa]")

    #     ## Compute resonances --------------------------------------------------------
    #     self.model.solve(self.study_name)
    #     freqs_sim = self.model.evaluate('abs(freq)', 'MHz')
    #     self.model.clear()
    #     self.model.reset()

    #     self.nb_calls += 1
    #     print("---- call #" + str(self.nb_calls) + " in %.6s seconds ----" % (time.time() - start_total_time))

    #     ## Remove the first 6 bad frequencies
    #     freqs_sim = freqs_sim[6:]

    #     ## Only select the first number of "freq to compare"
    #     freqs_sim = freqs_sim[:self.nb_freq]

    #     return freqs_sim - self.freqs_data



    def compute_chi2(self, member):
        """Compute chi^2 = sum((freqs_sim[i] - freqs_data[i])^2)"""

        start_total_time = time.time()

        ## Update elastic constants ----------------------------------------------
        for param_name in self.ranges_dict.keys():
            self.model.parameter(param_name, str(member[param_name])+"[GPa]")

        ## Compute resonances ---------------------------------------------------
        self.model.solve(self.study_name)
        freqs_sim = self.model.evaluate('abs(freq)', 'MHz')
        self.model.clear()
        self.model.reset()

        ## Remove the first 6 bad frequencies
        freqs_sim = freqs_sim[6:]

        ## Only select the first number of "freq to compare"
        freqs_sim = freqs_sim[:self.nb_freq]

        ## Compute Chi^2
        chi2 = np.sum(np.square(freqs_sim - self.freqs_data))
        member['chi2'] = float(chi2)

        self.nb_calls += 1
        print("---- call #" + str(self.nb_calls) + " in %.6s seconds ----" % (time.time() - start_total_time)
            + "---- chi2 = " + "{0:.3e}".format(chi2))

        return member



    def genetic_algorithm(self):
        """init_member is the inital set where we want the algorthim to start from"""

        ## Randomize the radom generator at first
        np.random.seed(self.seed)

        ## INITIALIAZE ----------------------------------------------------------
        print('\n## Initial Member ##')
        member = self.compute_chi2(self.init_member)
        self.best_member = deepcopy(member)
        # best_member_path = utils.save_member_to_json(best_member, folder=folder)


        ## GENERATION 0 ----------------------------------------------------------
        ## Create GENERATION_0 from the initial set of parameters
        ## and mutating the ones that should variate in the proposed search range
        print('\n## Generation 0 ##')
        generation_0_list = []

        ## Loop over the MEMBERs of GENERATION_0
        for i_pop in range(self.population):
            print('\n## Generation 0 MEMBER '+str(i_pop + 1))
            this_member = deepcopy(member)

            ## Loop over the genes of this MEMBER picked within the range
            for gene_name, gene_range in ranges_dict.items():
                this_member[gene_name] = random.uniform(gene_range[0], gene_range[-1])

            ## Compute the fitness of this MEMBER
            this_member = self.compute_chi2(this_member)

            ## Just display if this member has a better Chi^2 than
            ## the best member so far, and also erase best member
            ## if this new member is better
            generation_0_list.append(this_member)
            if this_member["chi2"] < self.best_member["chi2"]:
                self.best_member = deepcopy(this_member)
                self.n_better += 1
                print("!!!" + str(self.n_better)+' IMPROVEMENTS SO FAR!!!')
                ## Save BEST member to JSON
                # os.remove(best_member_path) # remove previous json
                # best_member_path = utils.save_member_to_json(best_member, folder=folder)


        ## NEXT GENERATIONS -----------------------------------------------------
        generations_list = []
        last_generation = generation_0_list # is the list of all the MEMBERs previous GENERATION

        ## Loop over ALL next GENERATIONs
        for n in range(self.N_generation):
            print("\n\n## GENERATION " + str(n + 1) + "##")
            next_gen = []

            ## Loop over the MEMBERs of this GENERATION
            for i_pop in range(self.population):

                print("\n##Generation " + str(n +1 ) + " Member " + str(i_pop + 1))
                parent = last_generation[i_pop] # take i^th member of the last computed generation
                child = deepcopy(parent)

                ## Loop over the different genes that will either cross or mutate
                for gene_name, gene_range in ranges_dict.items():
                    # crossing
                    if random.uniform(0,1) > self.crossing_p:
                        # within the probability of crossing "p", keep the gene of the parent
                        # crossing : keep the same gene (gene_name) value for the child as the parent without mutation.
                        child[gene_name] = parent[gene_name]
                    # mutation
                    else:
                        parent_1 = random.choice(last_generation) # choose a random member in the last generation
                        parent_2 = random.choice(last_generation)
                        parent_3 = random.choice(last_generation)
                        new_gene = parent_1[gene_name] + mutation_s*(parent_2[gene_name]-parent_3[gene_name])
                        ## Is the mutated gene within the range of the gene?
                        child[gene_name] = np.clip(new_gene, gene_range[0], gene_range[-1])

                ## Compute the fitness of the CHILD
                child = utils.compute_chi2(child, data_dict, normalized_data)[0]
                print('this chi2 = ' + "{0:.3e}".format(child["chi2"]))

                # If the CHILD has a better fitness than the PARENT, then keep CHILD
                if child['chi2']<last_generation[i_pop]['chi2']:
                    next_gen.append(child)
                # If the CHILD is not better than the PARENT, then keep the PARENT
                else:
                    next_gen.append(last_generation[i_pop])

                # Erase the best chi2 member if found better
                if child['chi2'] < best_member['chi2']:
                    best_member = deepcopy(child)
                    self.n_better += 1
                    print(str(self.n_better)+' IMPROVEMENTS SO FAR!')
                    ## Save BEST member to JSON
                    os.remove(best_member_path) # remove previous json
                    os.remove(best_member_path[:-4] + "pdf") # remove previous figure
                    best_member_path = utils.save_member_to_json(best_member, folder=folder)
                    utils.fig_compare(best_member, data_dict, folder=folder, fig_show=False, normalized_data=normalized_data)
                print('BEST CHI2 = ' + "{0:.3e}".format(best_member["chi2"]))

            generations_list.append(deepcopy(last_generation))
            last_generation = next_gen


        ## The End of time ---------------------------------------------------------

        print('THE END OF TIME')

        ## Save BEST member to JSON
        os.remove(best_member_path)
        utils.save_member_to_json(best_member, folder=folder)

        print('BEST CHI2 = ' + "{0:.3e}".format(best_member["chi2"]))



    def run_fit(self):

        ## Initialize the COMSOL file
        client = mph.Client()
        self.model = client.load(self.mph_file)
        for param_name, param_value in self.init_member.items():
            self.model.parameter(param_name, str(param_value)+"[GPa]")

        ## Modifying COMSOL parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.model.parameter('nb_freq', str(self.nb_freq + 6))

        ## Initialize number of calls
        self.nb_calls = 0

        ## Run fit algorithm
        if self.method=="least_square":
            out = minimize(self.compute_diff, self.pars)
        if self.method=="shgo":
            out = minimize(self.compute_diff, self.pars,
                           method='shgo',sampling_method='sobol', options={"f_tol": 1e-16}, n = 100, iters=20)
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
        client.clear()




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
                           nb_freq = 42)
    fitObject.run_fit()

