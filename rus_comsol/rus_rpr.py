import numpy as np
from scipy import linalg
from copy import deepcopy
from rus_comsol.elastic_constants import ElasticConstants
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class RUSRPR(ElasticConstants):
    def __init__(self, cij_dict, symmetry,
                 mass, dimensions,
                 order, nb_freq,
                 angle_x=0, angle_y=0, angle_z=0,
                 init=False):
        """
        cij_dict: a dictionary of elastic constants in GPa
        mass: a number in kg
        dimensions: numpy array of x, y, z lengths in m
        order: integer - highest order polynomial used to express basis functions
        nb_freq: number of frequencies to display
        method: fitting method
        """
        super().__init__(cij_dict,
                         symmetry=symmetry,
                         angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)

        self.mass       = mass # mass of the sample
        self.density    = mass / np.prod(np.array(dimensions))
        self.dimensions = np.array(dimensions) # in meters

        self.order      = order # order of the highest polynomial used to calculate the resonacne frequencies
        self.N          = int((order+1)*(order+2)*(order+3)/6) # this is the number of basis functions

        self.cij_dict = deepcopy(cij_dict)
        self._nb_freq = nb_freq
        self.freqs    = None

        self.basis  = np.zeros((self.N, 3))
        self.idx    =  0
        self.block  = [[],[],[],[],[],[],[],[]]
        self.Emat  = None
        self.Itens = None

        if init == True:
            self.initialize()

    ## Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def _get_nb_freq(self):
        return self._nb_freq
    def _set_nb_freq(self, nb_freq):
        self._nb_freq = nb_freq
    nb_freq = property(_get_nb_freq, _set_nb_freq)

    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def initialize(self):
        # create basis and sort it based on its parity;
        # for details see Arkady's paper;
        # this is done here in __init__ because we only need to is once and it is the "long" part of the calculation
        self.idx    =  0
        self.block  = [[],[],[],[],[],[],[],[]]
        self.Emat  = None
        self.Itens = None

        lookUp = {(1, 1, 1) : 0,
                   (1, 1, -1) : 1,
                    (1, -1, 1) : 2,
                     (-1, 1, 1) : 3,
                      (1, -1, -1): 4,
                       (-1, 1, -1) : 5,
                        (-1, -1, 1) : 6,
                         (-1, -1, -1) : 7}
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

    def copy_object(self, rpr_object):
        self.block = rpr_object.block
        self.idx   = rpr_object.idx
        self.Emat  = rpr_object.Emat
        self.Itens = rpr_object.Itens

    def E_int(self, i, j):
        """
        calculates integral for kinetic energy matrix, i.e. the product of two basis functions
        """
        ps = self.basis[i] + self.basis[j] + 1.
        if np.any(ps%2==0): return 0.
        return 8*np.prod((self.dimensions/2)**ps / ps)


    def G_int(self, i, j, k, l):
        """
        calculates the integral for potential energy matrix, i.e. the product of the derivatives of two basis functions
        """
        M = np.array([[[2.,0.,0.],[1.,1.,0.],[1.,0.,1.]],[[1.,1.,0.],[0.,2.,0.],[0.,1.,1.]],[[1.,0.,1.],[0.,1.,1.],[0.,0.,2.]]])
        if not self.basis[i][k]*self.basis[j][l]: return 0
        ps = self.basis[i] + self.basis[j] + 1. - M[k,l]
        if np.any(ps%2==0): return 0.
        return 8*self.basis[i][k]*self.basis[j][l]*np.prod((self.dimensions/2)**ps / ps)


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
                if i==j: Etens[i,k,j,l]=Etens[j,l,i,k]=self.E_int(k,l)*self.density
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


    def G_mat(self):
        """
        get potential energy matrix;
        this is a separate step because I_tens is independent of elastic constants, but only dependent on geometry;
        it is also the slow part of the calculation but only has to be done once this way
        """
        Gtens = np.tensordot(self.cijkl*1e9, self.Itens, axes= ([1,3],[0,2]))
        Gmat = np.swapaxes(Gtens, 2, 1).reshape(3*self.idx, 3*self.idx)
        return Gmat



    def compute_resonances(self, eigvals_only=True):
        """
        calculates resonance frequencies in Hz;
        pars: dictionary of elastic constants
        nb_freq: number of elastic constants to be displayed
        eigvals_only (True/False): gets only eigenvalues (i.e. resonance frequencies) or also gives eigenvectors (the latter is important when we want to calculate derivatives)
        """
        Gmat = self.G_mat()
        if eigvals_only==True:
            w = np.array([])
            for ii in range(8):
                w = np.concatenate((w, linalg.eigh(Gmat[np.ix_(self.block[ii], self.block[ii])], self.Emat[np.ix_(self.block[ii], self.block[ii])], eigvals_only=True)))
            self.freqs = np.sqrt(np.absolute(np.sort(w))[6:self.nb_freq+6])/(2*np.pi) * 1e-6 # resonance frequencies in MHz
            return self.freqs
        else:
            w, a = linalg.eigh(Gmat, self.Emat)
            a = a.transpose()[np.argsort(w)][6:self.nb_freq+6]
            self.freqs = np.sqrt(np.absolute(np.sort(w))[6:self.nb_freq+6])/(2*np.pi) * 1e-6 # resonance frequencies in MHz
            return self.freqs, a




    def log_derivatives_analytical(self, return_freqs=False):
        """
        calculating logarithmic derivatives of the resonance frequencies with respect to elastic constants,
        i.e. (df/dc)*(c/f), following Arkady's paper
        """

        f, a = self.compute_resonances(eigvals_only=False)
        derivative_matrix = np.zeros((self.nb_freq, len(self.cij_dict)))
        ii = 0
        cij_dict_original = deepcopy(self.cij_dict)

        for direction in sorted(cij_dict_original):
            value = cij_dict_original[direction]
            Cderivative_dict = {key: 0 for key in cij_dict_original}
            # Cderivative_dict = {'c11': 0,'c22': 0, 'c33': 0, 'c13': 0, 'c23': 0, 'c12': 0, 'c44': 0, 'c55': 0, 'c66': 0}
            Cderivative_dict[direction] = 1
            self.cij_dict = Cderivative_dict

            Gmat_derivative = self.G_mat()
            for idx, res in enumerate(f):
                derivative_matrix[idx, ii] = np.matmul(a[idx].T, np.matmul(Gmat_derivative, a[idx]) ) / (res**2) * value
            ii += 1
        log_derivative = np.zeros((self.nb_freq, len(self.cij_dict)))
        for idx, der in enumerate(derivative_matrix):
            log_derivative[idx] = der / sum(der)

        self.cij_dict = cij_dict_original

        if return_freqs == True:
            return log_derivative, f
        elif return_freqs == False:
            return log_derivative


    
    def print_logarithmic_derivative (self, print_frequencies=True):
        print ('start taking derivatives ...')
        if self.Emat is None:
            self.initialize()
        
        log_der, freqs_calc = self.log_derivatives_analytical(return_freqs=True)

        cij = deepcopy(sorted(self.cij_dict))
        template = ""
        for i, _ in enumerate(cij):
            template += "{" + str(i) + ":<13}"
        header = ['2 x logarithmic derivative (2 x dlnf / dlnc)']+(len(cij)-1)*['']
        der_text = template.format(*header) + '\n'
        der_text = der_text + template.format(*cij) + '\n'
        der_text = der_text + '-'*13*len(cij) + '\n'

        for ii in np.arange(self.nb_freq):
            text = [str(round(log_der[ii,j], 6)) for j in np.arange(len(cij))]
            der_text = der_text + template.format(*text) + '\n'

        if print_frequencies == True:
            freq_text = {}
            freq_template = "{0:<10}{1:<13}"
            freq_text += freq_template.format(*['idx', 'freq calc']) + '\n'
            freq_text += freq_template.format(*['', '(MHz)']) + '\n'
            freq_text += '-'*23 + '\n'
            for ii, f in enumerate(freqs_calc):
                freq_text += freq_template.format(*[int(ii), round(f, 4)]) + '\n'

            total_text = ''
            for ii in np.arange(len(der_text.split('\n'))):
                total_text = total_text + freq_text.split('\n')[ii] + der_text.split('\n')[ii] + '\n'
        else:
            total_text = der_text
        
        return total_text


if __name__ == "__main__":
    import time
    order = 12              # highest order basis polynomial
    nb_freq = 10             # number of frequencies included in fit (if 0, all resonances in file are used)

    mass = 0.002132e-3   # mass in kg
    dimensions = np.array([1560e-6, 1033e-6, 210e-6]) # dimensions of sample in m
                # first value is length along [100], second along [010], and third is along [001]

    # initial elastic constants in Pa
    cij_dict = {"c11": 231e9,
                "c12": 132e9,
                "c13": 71e9,
                "c33": 186e9,
                "c44": 49e9,
                "c66": 95e9,
                }

    rus = RUSRPR(cij_dict, "tetragonal", mass, dimensions, order, nb_freq)
    t0 = time.time()
    print ('initialize the class ...')
    rus.initialize()
    print ('class initialized in ', round(time.time()-t0, 4), ' s')
    print(rus.compute_resonances()*1e-6)






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


    # def print_results(self, lmfit_out):
    #     """
    #     create a nice printable output of the fit results and derivatives
    #     """
    #     total_text = ''
    #     divider = '#->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->'
    #     print ()
    #     print (divider)
    #     total_text = total_text + divider + '\n'
    #     print (divider)
    #     total_text = total_text + divider + '\n'
    #     print ()
    #     total_text = total_text + '#' + '\n'
    #     formats = "{0:<33}{1:<10}"
    #     print ( formats.format(*['Crystal Structure:', self.crystal_structure]) )
    #     total_text = total_text + '# ' + formats.format(*['Crystal Structure:', self.crystal_structure]) + '\n'
    #     print ( formats.format(*['Mass (mg):', self.mass*1e6]) )
    #     total_text = total_text + '# ' + formats.format(*['Mass (mg):', self.mass*1e6]) + '\n'
    #     print ( formats.format(*['Sample Dimensions (mm):',str(np.array(self.dimensions)*1e3)]) )
    #     total_text = total_text + '# ' + formats.format(*['Sample Dimensions (mm):',str(np.array(self.dimensions)*1e3)]) + '\n'
    #     print ( formats.format(*['Highest Order Basis Polynomial:', self.order]) )
    #     total_text = total_text + '# ' + formats.format(*['Highest Order Basis Polynomial:', self.order]) + '\n'
    #     print ( formats.format(*['Number Of Calls:', self.call]) )
    #     total_text = total_text + '# ' + formats.format(*['Number Of Calls:', self.call]) + '\n'
    #     print ()
    #     total_text = total_text + '#' + '\n'
    #     print (divider)
    #     total_text = total_text + divider + '\n'
    #     print (divider)
    #     total_text = total_text + divider + '\n'
    #     print ()
    #     total_text = total_text + '#' + '\n'
    #     formats = "{0:<7}{1:<10}{2:<5}{3:<12}{4:<12}{5:<30}{6:<20}"
    #     for name in sorted(lmfit_out.params):
    #         param = lmfit_out.params[name]
    #         self.fit_results[name] = param.value
    #         if param.stderr == None:
    #             param.stderr = 0
    #         text = [name+' =', '('+ str(round(param.value/1e9,3)), '+/-', str(round(param.stderr/1e9, 3))+') GPa', '('+str(round(param.stderr/param.value*100,2))+'%);', 'bounds (GPa): '+str(np.array(self.bounds_dict[name])/1e9)+';', 'init value = '+str(round(self.cij_dict[name]/1e9,3))+' GPa']
    #         text = formats.format(*text)
    #         total_text = total_text + '# ' + text + '\n'
    #         print ( text )
    #     print ()
    #     total_text = total_text + '#' + '\n'
    #     print (divider)
    #     total_text = total_text + divider + '\n'
    #     print (divider)
    #     total_text = total_text + divider + '\n'
    #     print ()
    #     total_text = total_text + '#' + '\n'
    #     fsim = self.compute_resonances(pars=self.fit_results, nb_freq=self.nb_freq+self.nb_missing_res+10)
    #     log_der = self.log_derivatives_analytical(self.fit_results, self.nb_freq+self.nb_missing_res+10)
    #     # log_der = self.log_derivatives_numerical(self.fit_results, self.nb_freq+self.nb_missing_res+10, parallel=True)
    #     formats = "{0:<9}{1:<15}{2:<15}{3:<23}"
    #     header_text = ['index', 'f exp (MHz)', 'f calc (MHz)', 'difference (%)']
    #     nb = 4
    #     for c in self.fit_results:
    #         formats = formats + '{' + str(nb) + ':<15}'
    #         header_text = header_text + ['2*df/dln'+c]
    #         nb +=1
    #     print(formats.format(*header_text))
    #     total_text = total_text + '# ' + formats.format(*header_text) + '\n'
    #     print ()
    #     total_text = total_text + '' + '\n'
    #     idx_exp = 0
    #     difference = []
    #     for idx_sim in np.arange(self.nb_freq+self.nb_missing_res+10):
    #         if idx_sim in self.missing_idx:
    #             text_f = [idx_sim, 0, round(fsim[idx_sim]/1e6,5), 0]
    #             derivatives = list(log_der[idx_sim]*0)
    #             text = '#' + formats.format(*(text_f + derivatives))
    #         elif idx_sim < self.nb_freq+self.nb_missing_res:
    #             text_f = [idx_sim, round(self.freqs_data[idx_exp]/1e6, 5), round(fsim[idx_sim]/1e6,5), round((self.freqs_data[idx_exp]-fsim[idx_sim])/self.freqs_data[idx_exp]*100,5)]
    #             derivatives = list(np.round(log_der[idx_sim],6))
    #             text = formats.format(*(text_f + derivatives))
    #             difference.append((self.freqs_data[idx_exp]-fsim[idx_sim])/self.freqs_data[idx_exp])
    #             idx_exp += 1
    #         else:
    #             text_f = [idx_sim, '', round(fsim[idx_sim]/1e6,5), '']
    #             derivatives = [''] * len(log_der[idx_sim])
    #             text = '#' + formats.format(*(text_f + derivatives))

    #         total_text = total_text + text + '\n'
    #         print ( text )
    #     print ()
    #     total_text = total_text + '#' + '\n'
    #     print (divider)
    #     total_text = total_text + divider + '\n'
    #     print()
    #     total_text = total_text + '#' + '\n'

    #     difference = np.array(difference)
    #     rms = np.sqrt(sum(difference**2)) / len(difference) * 100
    #     print (' RMS = ', round(rms, 3), ' %' )
    #     total_text = total_text + "# RMS = " + str( round( rms, 3 ) ) + ' %\n'

    #     print()
    #     total_text = total_text + '#' + '\n'
    #     print (divider)
    #     total_text = total_text + divider + '\n'
    #     print (divider)
    #     total_text = total_text + divider + '\n'

    #     return total_text


    # def save_results(self, text):
    #     save_path = self.freqs_file[:-4] + '_out.txt'
    #     with open(save_path, 'w') as g:
    #         g.write(text)