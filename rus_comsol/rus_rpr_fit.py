import numpy as np
from scipy import linalg
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
from lmfit import minimize, Parameters, report_fit
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool

class ElasticSolid:
    def __init__(self, cij_dict, bounds_dict, cij_vary,
                 mass, dimensions,
                 order, nb_freq=0,
                 method='differential_evolution', freqs_file=None,
                 nb_missing_res=0, maxiter=1000,
                 init=False):
        """
        cij_dict: a dictionary of elastic constants in Pa
        mass: a number in kg
        dimensions: numpy array of x, y, z lengths in m
        order: integer - highest order polynomial used to express basis functions
        nb_freq: number of frequencies to display
        method: fitting method
        """
        self.crystal_structure = None

        self.mass       = mass # mass of the sample
        self.rho        = mass / np.prod(dimensions)
        self.dimensions = dimensions

        self.order      = order # order of the highest polynomial used to calculate the resonacne frequencies
        self.N          = int((order+1)*(order+2)*(order+3)/6) # this is the number of basis functions

        self.Vol         = dimensions/2 # sample dimensions divided by 2

        self.init_cij_dict = deepcopy(cij_dict)
        self.bounds_dict = deepcopy(bounds_dict)
        self.cij_vary = cij_vary

        # imports the measured resonance frequencies
        self.freqs_file = freqs_file
        self.freqs_data = self.load_data()

        if nb_freq == 0:
            self.nb_freq = len(self.freqs_data)
        else:
            self.nb_freq = nb_freq
        self.nb_missing_res = nb_missing_res
        self.matching_idx = np.array([])
        self.missing_idx = np.array([])

        # maximum number of iterations in fit
        self.maxiter = maxiter

        ## fit algorithm
        self.method = method # "shgo", "differential_evolution", "leastsq"
        ## Initialize fit parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.params = Parameters()
        for c in self.init_cij_dict:
            self.params.add(c, value=self.init_cij_dict[c], vary=self.cij_vary[c], min=self.bounds_dict[c][0], max=self.bounds_dict[c][-1])

        # keeps track of how often the residual function is called
        self.call = 1
        # initialize a variable which will contain the fit results for the elastic constants
        self.fit_results = {}

        self.basis  = np.zeros((self.N, 3))
        self.idx    =  0
        self.block  = [[],[],[],[],[],[],[],[]]

        self.Emat  = None
        self.Itens = None

        if init == True:
            self.initialize()


    def initialize(self):
        # create basis and sort it based on its parity;
        # for details see Arkady's paper;
        # this is done here in __init__ because we only need to is once and it is the "long" part of the calculation
        lookUp = {(1, 1, 1) : 0, (1, 1, -1) : 1, (1, -1, 1) : 2, (-1, 1, 1) : 3, (1, -1, -1): 4, (-1, 1, -1) : 5, (-1, -1, 1) : 6, (-1, -1, -1) : 7}
        for k in range(self.order+1):
            for l in range(self.order+1):
                for m in range(self.order+1):
                    if k+l+m <= self.order:
                        self.basis[self.idx] = np.array([k,l,m])
                        for ii in range(3):
                            self.block[lookUp[tuple((-1,-1,-1)**(self.basis[self.idx] + np.roll([1,0,0], ii)))]].append(ii*self.N + self.idx)
                        self.idx += 1
        self.Emat = self.E_mat()
        self.Itens = self.I_tens()


    def load_data(self):
        """
        Frequencies should be in Hz
        """
        ## Load the resonance data in MHz
        freqs_data = np.loadtxt(self.freqs_file, dtype="float", comments="#")
        return freqs_data


    def elastic_tensor(self, pars):
        """
        returns the elastic tensor from given elastic constants in pars
        (a dictionary of elastic constants)
        based on the length of pars it decides what crystal structure we the sample has
        """
        ctens = np.zeros([3,3,3,3])

        if len(pars) == 3:                      # cubic
            self.crystal_structure = 'cubic'
            c11 = c22 = c33 = pars['c11']
            c12 = c13 = c23 = pars['c12']
            c44 = c55 = c66 = pars['c44']

        elif len(pars) == 5:                    # hexagonal
            self.crystal_structure = 'hexagonal'
            indicator = np.any( np.array([i=='c11' for i in pars]) )
            if indicator == True:
                c11 = c22       = pars['c11']
                c33             = pars['c33']
                c12             = pars['c12']
                c13 = c23       = pars['c13']
                c44 = c55       = pars['c44']
                c66             = (pars['c11']-pars['c12'])/2
            else:
                c11 = c22       = 2*pars['c66'] + pars['c12']
                c33             = pars['c33']
                c12             = pars['c12']
                c13 = c23       = pars['c13']
                c44 = c55       = pars['c44']
                c66             = pars['c66']

        elif len(pars) == 6:                    # tetragonal
            self.crystal_structure = 'tetragonal'
            c11 = c22       = pars['c11']
            c33             = pars['c33']
            c12             = pars['c12']
            c13 = c23       = pars['c13']
            c44 = c55       = pars['c44']
            c66             = pars['c66']

        elif len(pars) == 9:                    # orthorhombic
            self.crystal_structure = 'orthorhombic'
            c11             = pars['c11']
            c22             = pars['c22']
            c33             = pars['c33']
            c12             = pars['c12']
            c13             = pars['c13']
            c23             = pars['c23']
            c44             = pars['c44']
            c55             = pars['c55']
            c66             = pars['c66']

        else:
            print ('You have not given a valid Crystal Structure')

        ctens[0,0,0,0] = c11
        ctens[1,1,1,1] = c22
        ctens[2,2,2,2] = c33
        ctens[0,0,1,1] = ctens[1,1,0,0] = c12
        ctens[2,2,0,0] = ctens[0,0,2,2] = c13
        ctens[1,1,2,2] = ctens[2,2,1,1] = c23
        ctens[0,1,0,1] = ctens[1,0,0,1] = ctens[0,1,1,0] = ctens[1,0,1,0] = c66
        ctens[0,2,0,2] = ctens[2,0,0,2] = ctens[0,2,2,0] = ctens[2,0,2,0] = c55
        ctens[1,2,1,2] = ctens[2,1,2,1] = ctens[2,1,1,2] = ctens[1,2,2,1] = c44

        return ctens


    def E_int(self, i, j):
        """
        calculates integral for kinetic energy matrix, i.e. the product of two basis functions
        """
        ps = self.basis[i] + self.basis[j] + 1.
        if np.any(ps%2==0): return 0.
        return 8*np.prod(self.Vol**ps / ps)


    def G_int(self, i, j, k, l):
        """
        calculates the integral for potential energy matrix, i.e. the product of the derivatives of two basis functions
        """
        M = np.array([[[2.,0.,0.],[1.,1.,0.],[1.,0.,1.]],[[1.,1.,0.],[0.,2.,0.],[0.,1.,1.]],[[1.,0.,1.],[0.,1.,1.],[0.,0.,2.]]])
        if not self.basis[i][k]*self.basis[j][l]: return 0
        ps = self.basis[i] + self.basis[j] + 1. - M[k,l]
        if np.any(ps%2==0): return 0.
        return 8*self.basis[i][k]*self.basis[j][l]*np.prod(self.Vol**ps / ps)


    def E_mat(self):
        """
        put the integrals from E_int in a matrix
        Emat is the kinetic energy matrix from Arkady's paper
        """
        Etens = np.zeros((3,self.idx,3,self.idx), dtype= np.double)
        for x in range(3*self.idx):
            i, k = int(x/self.idx), x%self.idx
            for y in range(x, 3*self.idx):
                j, l = int(y/self.idx), y%self.idx
                if i==j: Etens[i,k,j,l]=Etens[j,l,i,k]=self.E_int(k,l)*self.rho
        Emat = Etens.reshape(3*self.idx,3*self.idx)
        return Emat


    def I_tens(self):
        """
        put the integrals from G_int in a tensor;
        it is the tensor in the potential energy matrix in Arkady's paper;
        i.e. it is the the potential energy matrix without the elastic tensor;
        """
        Itens = np.zeros((3,self.idx,3,self.idx), dtype= np.double)
        for x in range(3*self.idx):
            i, k = int(x/self.idx), x%self.idx
            for y in range(x, 3*self.idx):
                j, l = int(y/self.idx), y%self.idx
                Itens[i,k,j,l]=Itens[j,l,i,k]=self.G_int(k,l,i,j)
        return Itens


    def G_mat(self, pars):
        """
        get potential energy matrix;
        this is a separate step because I_tens is independent of elastic constants, but only dependent on geometry;
        it is also the slow part of the calculation but only has to be done once this way
        """
        C = self.elastic_tensor(pars)
        Gtens = np.tensordot(C, self.Itens, axes= ([1,3],[0,2]))
        Gmat = np.swapaxes(Gtens, 2, 1).reshape(3*self.idx, 3*self.idx)
        return Gmat


    def compute_resonances(self, pars, nb_freq, eigvals_only=True):
        """
        calculates resonance frequencies in Hz;
        pars: dictionary of elastic constants
        nb_freq: number of elastic constants to be displayed
        eigvals_only (True/False): gets only eigenvalues (i.e. resonance frequencies) or also gives eigenvectors (the latter is important when we want to calculate derivatives)
        """
        # if nb_freq==None:
        #     nb_freq = self.nb_freq
        # if pars is None:
        #     pars = self.init_cij_dict
        Gmat = self.G_mat(pars)
        if eigvals_only==True:
            w = np.array([])
            for ii in range(8):
                w = np.concatenate((w, linalg.eigh(Gmat[np.ix_(self.block[ii], self.block[ii])], self.Emat[np.ix_(self.block[ii], self.block[ii])], eigvals_only=True)))
            f = np.sqrt(np.absolute(np.sort(w))[6:nb_freq+6])/(2*np.pi) # resonance frequencies in Hz
            return f
        else:
            w, a = linalg.eigh(Gmat, self.Emat)
            a = a.transpose()[np.argsort(w)][6:nb_freq+6]
            f = np.sqrt(np.absolute(np.sort(w))[6:nb_freq+6])/(2*np.pi)
            return f, a



    # def log_derivatives_analytical(self, pars, nb_freq):
    #     """
    #     calculating logarithmic derivatives of the resonance frequencies with respect to elastic constants,
    #     i.e. (df/dc)*(c/f), following Arkady's paper
    #     """
    #     f, a = self.compute_resonances(pars, nb_freq, eigvals_only=False)
    #     derivative_matrix = np.zeros((nb_freq, len(pars)))
    #     ii = 0


    #     for direction in sorted(pars):
    #         value = pars[direction]
    #         Cderivative_dict = {key: 0 for key in pars}
    #         # Cderivative_dict = {'c11': 0,'c22': 0, 'c33': 0, 'c13': 0, 'c23': 0, 'c12': 0, 'c44': 0, 'c55': 0, 'c66': 0}
    #         Cderivative_dict[direction] = 1
    #         Gmat_derivative = self.G_mat(Cderivative_dict)
    #         for idx, res in enumerate(f):
    #             derivative_matrix[idx, ii] = np.matmul(a[idx].T, np.matmul(Gmat_derivative, a[idx]) ) / (res**2) * value
    #         ii += 1
    #     log_derivative = np.zeros((nb_freq, len(pars)))
    #     for idx, der in enumerate(derivative_matrix):
    #         log_derivative[idx] = der / sum(der)


    #     # print the logarithmic derivatives of each frequency
    #     # formats = "{0:<15}{1:<15}"
    #     # k = 2
    #     # for _ in log_derivative[0]:
    #     #     formats = formats + '{' + str(k) + ':<15}'
    #     #     k+=1
    #     # print ('-----------------------------------------------------------------------')
    #     # print ('-----------------------------------------------------------------------')
    #     # print ('2 x LOGARITHMIC DERIVATIVES')
    #     # print ('-----------------------------------------------------------------------')
    #     # print (formats.format('f [MHz]','dlnf/dlnc11','dlnf/dlnc12','dlnf/dlnc44','SUM') )
    #     # for idx, line in enumerate(log_derivative):
    #     #     text = [str(round(f[idx]/1e6,6))] + [str(round(d, 6)) for d in line] + [str(round(sum(line),7))]
    #     #     print ( formats.format(*text) )
    #     # print ('-----------------------------------------------------------------------')
    #     # print ('-----------------------------------------------------------------------')

    #     return log_derivative




    # def log_derivatives_numerical(self, pars, nb_freq, dc=1e5, N=5, Rsquared_threshold=1e-5, parallel=False, nb_workers=None ):
    #     """
    #     calculating logarithmic derivatives of the resonance frequencies with respect to elastic constants,
    #     i.e. (df/dc)*(c/f), by calculating the resonance frequencies for slowly varying elastic constants
    #     variables: pars (dictionary of elastic constants), dc, N
    #     The derivative is calculated by computing resonance frequencies for N different elastic cosntants centered around the value given in pars and spaced by dc.
    #     A line is then fitted through these points and the slope is extracted as the derivative.
    #     """
    #     if nb_workers is None:
    #         nb_workers = min( [int(cpu_count()/2), N] )


    #     # calculate the resonance frequencies for the "true" elastic constants
    #     freq_result = self.compute_resonances(pars=pars, nb_freq=nb_freq)


    #     if parallel == True:
    #             # print("# of available cores: ", cpu_count())
    #             pool = Pool(processes=nb_workers)
    #             # print("--- Pool initialized with ", nb_workers, " workers ---")


    #     fit_results_dict = {}
    #     Rsquared_matrix = np.zeros([len(freq_result), len(pars)])
    #     log_derivative_matrix = np.zeros([len(freq_result), len(pars)])
    #     # take derivatives with respect to all elastic constants
    #     ii = 0
    #     for elastic_constant in sorted(pars):
    #         # print ('start taking derivative with respect to ', elastic_constant)
    #         # t1 = time()
    #         # create an array of elastic constants centered around the "true" value
    #         c_result = pars[elastic_constant]
    #         c_derivative_array = np.linspace(c_result-N/2*dc, c_result+N/2*dc, N)
    #         elasticConstants_derivative_dict = deepcopy(pars)
    #         # calculate the resonance frequencies for all elastic constants in c_test and store them in Ctest

    #         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #         # # this calculates all the necessary sets of resonance frequencies for the derivative in parallel
    #         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #         elasticConstants_derivative_array = []
    #         # here we are creating an array where each element is a dictionary of a full set
    #         # of elastic constants. Hopefully, we can give this array to a pool.map
    #         for c in c_derivative_array:
    #             elasticConstants_derivative_dict[elastic_constant] = c
    #             # deepcopy actually makes a copy of the dictionary instead of just creating a new pointer to the same location
    #             elasticConstants_derivative_array.append(deepcopy(elasticConstants_derivative_dict))

    #         if parallel == True:
    #             # print("# of available cores: ", cpu_count())
    #             # pool = Pool(processes=nb_workers)
    #             elasticConstants_derivative_array = [(c, nb_freq) for c in elasticConstants_derivative_array]
    #             freq_derivative_matrix = pool.starmap(self.compute_resonances, elasticConstants_derivative_array) - np.array([freq_result for _ in np.arange(N)])
    #             freq_derivative_matrix = np.transpose( np.array( freq_derivative_matrix ) )
    #         else:
    #             freq_derivative_matrix = np.zeros([len(freq_result), N])
    #             for idx, parameter_set in enumerate(elasticConstants_derivative_array):
    #                 freq_derivative_matrix[:, idx] = self.compute_resonances(pars=parameter_set, nb_freq=nb_freq) - freq_result


    #         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #         # # this calculates all the necessary sets of resonance frequencies for the derivative in series
    #         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #         # for idx, c in enumerate(c_derivative_array):
    #         #     elasticConstants_derivative_dict[elastic_constant] = c
    #         #     # note we don't actually save the resonance frequencies, but we shift them by the values at the "true" elastic constants;
    #         #     # this is done because within the elastic constants in c_test the frequencies change only very little compared to their absolute value,
    #         #     # thus this shift is important to get a good fit later
    #         #     freq_derivative_matrix[:,idx] = self.compute_resonances(pars=elasticConstants_derivative_dict)-freq_result

    #         # shift array of elastic constants to be centered around zero, for similar argument made for the shift of resonance frequencies
    #         c_derivative_array = c_derivative_array - c_result

    #         fit_matrix = np.zeros([len(freq_result), N])
    #         # here we fit a straight line to the resonance frequency vs elastic costants for all resonances
    #         for idx, freq_derivative_array in enumerate(freq_derivative_matrix):
    #             slope, y_intercept = np.polyfit(c_derivative_array, freq_derivative_array, 1)
    #             log_derivative_matrix[idx, ii] = 2 * slope * pars[elastic_constant]/freq_result[idx]

    #             ## check if data really lies on a line
    #             # offset.append(popt[1])
    #             current_fit = slope*c_derivative_array + y_intercept
    #             fit_matrix[idx,:] = current_fit
    #             # calculate R^2;
    #             # this is a value judging how well the data is described by a straight line
    #             SStot = sum( (freq_derivative_array - np.mean(freq_derivative_array))**2 )
    #             SSres = sum( (freq_derivative_array - current_fit)**2 )
    #             Rsquared = 1 - SSres/SStot
    #             Rsquared_matrix[idx, ii] = Rsquared
    #             # we want a really good fit!
    #             # R^2 = 1 would be perfect
    #             if abs(1-Rsquared) > Rsquared_threshold:
    #                 # if these two fits differ by too much, just print the below line and plot that particular data
    #                 print ('not sure if data is a straight line ', elastic_constant, ' ', freq_result[idx]/1e6, ' MHz')
    #                 plt.figure()
    #                 plt.plot(c_derivative_array/1e3, freq_derivative_array, 'o')
    #                 plt.plot(c_derivative_array/1e3, current_fit)
    #                 plt.title(elastic_constant +'; f = ' + str(round(freq_result[idx]/1e6, 3)) + ' MHz; $R^2$ = ' + str(round(Rsquared, 7)))
    #                 plt.xlabel('$\\Delta c$ [kPa]')
    #                 plt.ylabel('$\\Delta f$ [Hz]')
    #                 plt.show()

    #         # store all fit results in this dictionary, just in case you need to look at this at some point later
    #         fit_results_dict[elastic_constant] = {
    #             'freq_test': freq_derivative_matrix,
    #             'c_test': c_derivative_array,
    #             'fit': fit_matrix,
    #             'Rsquared': Rsquared_matrix
    #         }

    #         # print ('derivative with respect to ', elastic_constant, ' done in ', round(time()-t1, 4), ' s')
    #         ii += 1

    #     if parallel == True:
    #         pool.terminate()

    #     # print the logarithmic derivatives of each frequency
    #     # formats = "{0:<15}{1:<15}"
    #     # k = 2
    #     # for _ in log_derivative_matrix[0]:
    #     #     formats = formats + '{' + str(k) + ':<15}'
    #     #     k+=1
    #     # print ('-----------------------------------------------------------------------')
    #     # print ('-----------------------------------------------------------------------')
    #     # print ('2 x LOGARITHMIC DERIVATIVES')
    #     # print ('-----------------------------------------------------------------------')
    #     # print (formats.format('f [MHz]','dlnf/dlnc11','dlnf/dlnc12','dlnf/dlnc44','SUM') )
    #     # for idx, line in enumerate(log_derivative_matrix):
    #     #     text = [str(round(freq_result[idx]/1e6,6))] + [str(round(d, 6)) for d in line] + [str(round(sum(line),7))]
    #     #     print ( formats.format(*text) )
    #     # print ('-----------------------------------------------------------------------')
    #     # print ('-----------------------------------------------------------------------')

    #     return (log_derivative_matrix)




    # ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
    # ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
    # everything above was code to perform the actual calculations
    # everything below will be fit algorithms
    # ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
    # ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
    def assignement(self, freqs_data, freqs_sim):
        """
        Linear assigment of the simulated frequencies to the data frequencies
        in case there is one or more missing frequencies in the data
        """
        cost_matrix = distance_matrix(freqs_data[:, None], freqs_sim[:, None])**2
        ## sim_index is the indices for freqs_sim to match freqs_data
        index_sim = linear_sum_assignment(cost_matrix)[1]

        ## Give indices of the missing frequencies in the data
        bool_missing = np.ones(freqs_sim.size, dtype=bool)
        bool_missing[index_sim] = False
        index_missing = np.arange(0, freqs_sim.size, 1)[bool_missing]
        # index_missing = index_missing[index_missing < self.freqs_data.size]

        return index_sim, index_missing, freqs_sim[index_sim]


    def residual_function(self, pars):
        """
        define the residual function used in lmfit;
        i.e. (simulated resonance frequencies - data)
        """
        freqs_sim = self.compute_resonances(pars=pars, nb_freq=self.nb_freq+self.nb_missing_res)
        freqs_exp = self.freqs_data[:self.nb_freq]

        if self.nb_missing_res > 0:
            self.matching_idx, self.missing_idx, freqs_sim_matched = self.assignement(freqs_exp, freqs_sim)
            delta = freqs_sim_matched - freqs_exp
        else:
            delta = freqs_sim - freqs_exp

        print ('call number ', self.call)
        pars.pretty_print(columns=['value', 'min', 'max', 'vary'])
        self.call += 1
        return delta


    def fit(self):
        self.call = 0
        if self.method == 'differential_evolution':
            out = minimize(self.residual_function, self.params, method=self.method, polish=True, maxiter=self.maxiter)
        elif self.method == 'leastsq':
            out = minimize(self.residual_function, self.params, method=self.method)
        else:
            print ('your fit method is not a valid method')

        ## Display fit report
        report_fit(out)
        result_text = self.print_results(out)
        self.save_results(result_text)
        return 1


    def print_results(self, lmfit_out):
        """
        create a nice printable output of the fit results and derivatives
        """
        total_text = ''
        divider = '#->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->'
        print ()
        print (divider)
        total_text = total_text + divider + '\n'
        print (divider)
        total_text = total_text + divider + '\n'
        print ()
        total_text = total_text + '#' + '\n'
        formats = "{0:<33}{1:<10}"
        print ( formats.format(*['Crystal Structure:', self.crystal_structure]) )
        total_text = total_text + '# ' + formats.format(*['Crystal Structure:', self.crystal_structure]) + '\n'
        print ( formats.format(*['Mass (mg):', self.mass*1e6]) )
        total_text = total_text + '# ' + formats.format(*['Mass (mg):', self.mass*1e6]) + '\n'
        print ( formats.format(*['Sample Dimensions (mm):',str(np.array(self.dimensions)*1e3)]) )
        total_text = total_text + '# ' + formats.format(*['Sample Dimensions (mm):',str(np.array(self.dimensions)*1e3)]) + '\n'
        print ( formats.format(*['Highest Order Basis Polynomial:', self.order]) )
        total_text = total_text + '# ' + formats.format(*['Highest Order Basis Polynomial:', self.order]) + '\n'
        print ( formats.format(*['Number Of Calls:', self.call]) )
        total_text = total_text + '# ' + formats.format(*['Number Of Calls:', self.call]) + '\n'
        print ()
        total_text = total_text + '#' + '\n'
        print (divider)
        total_text = total_text + divider + '\n'
        print (divider)
        total_text = total_text + divider + '\n'
        print ()
        total_text = total_text + '#' + '\n'
        formats = "{0:<7}{1:<10}{2:<5}{3:<12}{4:<12}{5:<30}{6:<20}"
        for name in sorted(lmfit_out.params):
            param = lmfit_out.params[name]
            self.fit_results[name] = param.value
            if param.stderr == None:
                param.stderr = 0
            text = [name+' =', '('+ str(round(param.value/1e9,3)), '+/-', str(round(param.stderr/1e9, 3))+') GPa', '('+str(round(param.stderr/param.value*100,2))+'%);', 'bounds (GPa): '+str(np.array(self.bounds_dict[name])/1e9)+';', 'init value = '+str(round(self.init_cij_dict[name]/1e9,3))+' GPa']
            text = formats.format(*text)
            total_text = total_text + '# ' + text + '\n'
            print ( text )
        print ()
        total_text = total_text + '#' + '\n'
        print (divider)
        total_text = total_text + divider + '\n'
        print (divider)
        total_text = total_text + divider + '\n'
        print ()
        total_text = total_text + '#' + '\n'
        fsim = self.compute_resonances(pars=self.fit_results, nb_freq=self.nb_freq+self.nb_missing_res+10)
        log_der = self.log_derivatives_analytical(self.fit_results, self.nb_freq+self.nb_missing_res+10)
        # log_der = self.log_derivatives_numerical(self.fit_results, self.nb_freq+self.nb_missing_res+10, parallel=True)
        formats = "{0:<9}{1:<15}{2:<15}{3:<23}"
        header_text = ['index', 'f exp (MHz)', 'f calc (MHz)', 'difference (%)']
        nb = 4
        for c in self.fit_results:
            formats = formats + '{' + str(nb) + ':<15}'
            header_text = header_text + ['2*df/dln'+c]
            nb +=1
        print(formats.format(*header_text))
        total_text = total_text + '# ' + formats.format(*header_text) + '\n'
        print ()
        total_text = total_text + '' + '\n'
        idx_exp = 0
        difference = []
        for idx_sim in np.arange(self.nb_freq+self.nb_missing_res+10):
            if idx_sim in self.missing_idx:
                text_f = [idx_sim, 0, round(fsim[idx_sim]/1e6,5), 0]
                derivatives = list(log_der[idx_sim]*0)
                text = '#' + formats.format(*(text_f + derivatives))
            elif idx_sim < self.nb_freq+self.nb_missing_res:
                text_f = [idx_sim, round(self.freqs_data[idx_exp]/1e6, 5), round(fsim[idx_sim]/1e6,5), round((self.freqs_data[idx_exp]-fsim[idx_sim])/self.freqs_data[idx_exp]*100,5)]
                derivatives = list(np.round(log_der[idx_sim],6))
                text = formats.format(*(text_f + derivatives))
                difference.append((self.freqs_data[idx_exp]-fsim[idx_sim])/self.freqs_data[idx_exp])
                idx_exp += 1
            else:
                text_f = [idx_sim, '', round(fsim[idx_sim]/1e6,5), '']
                derivatives = [''] * len(log_der[idx_sim])
                text = '#' + formats.format(*(text_f + derivatives))

            total_text = total_text + text + '\n'
            print ( text )
        print ()
        total_text = total_text + '#' + '\n'
        print (divider)
        total_text = total_text + divider + '\n'
        print()
        total_text = total_text + '#' + '\n'

        difference = np.array(difference)
        rms = np.sqrt(sum(difference**2)) / len(difference) * 100
        print (' RMS = ', round(rms, 3), ' %' )
        total_text = total_text + "# RMS = " + str( round( rms, 3 ) ) + ' %\n'

        print()
        total_text = total_text + '#' + '\n'
        print (divider)
        total_text = total_text + divider + '\n'
        print (divider)
        total_text = total_text + divider + '\n'

        return total_text


    def save_results(self, text):
        save_path = self.freqs_file[:-4] + '_out.txt'
        with open(save_path, 'w') as g:
            g.write(text)