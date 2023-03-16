from unicodedata import decimal
import numpy as np
from scipy import linalg
import sys
from copy import deepcopy
from rus_fitting.elastic_constants import ElasticConstants
from time import time, sleep
from rus_fitting.smi_matrices import SMIMatrices
from rus_fitting.rpr_matrices import RPRMatrices
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class RUSXYZ(ElasticConstants):
    def __init__(self, cij_dict, symmetry, order, density,
                 Emat_path = None, Itens_path = None,
                 nb_freq=1,
                 angle_x=0, angle_y=0, angle_z=0,
                 init=False, use_quadrants=False):
        """
        cij_dict: a dictionary of elastic constants in GPa
        mass: a number in kg
        dimensions: numpy array of x, y, z lengths in m
        order: integer - highest order polynomial used to express basis functions
        nb_freq: number of frequencies to display
        method: fitting method
        use_quadrants: if True, uses symmetry arguments of the elastic tensor to simplify and speed up eigenvalue solver;
                        only gives correct result if crystal symmetry is orthorhombic or higher;
                        if symmetry is e.g. rhombohedral, use use_quadrants=False
                        (use_quadrants=True ignores all terms c14, c15, c16, c24, c25, ... in the elastic tensor)
        """
        super().__init__(cij_dict,
                         symmetry=symmetry,
                         angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)

        self.Emat_path     = Emat_path
        self.Itens_path    = Itens_path


        self.order   = order # order of the highest polynomial used to calculate the resonacne frequencies
        self.N       = int((order+1)*(order+2)*(order+3)/6) # this is the number of basis functions
        self.density = density

        self.cij_dict = deepcopy(cij_dict)
        self._nb_freq = nb_freq
        self.freqs    = None

        self.basis  = np.zeros((self.N, 3))
        self.idx    =  0
        self.block  = [[],[],[],[],[],[],[],[]]
        self.Emat  = None
        self.Itens = None

        self.use_quadrants = use_quadrants

        if init == True:
            self.initialize()

        if use_quadrants:
            print('BE CAREFUL!')
            print('YOU CAN ONLY "USE QUADRANTS" IF:')
            print('      - THE SYMMETRY IS ORTHORHOMBIC OR HIGHER, AND')
            print('      - THE SAMPLE HAS X=-X, Y=-Y, Z=-Z SYMMETRY')

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

        self.Emat  = np.load(self.Emat_path)
        self.Emat = self.Emat*self.density
        self.Itens = np.load(self.Itens_path)

        idx = int((self.order+1)*(self.order+2)*(self.order+3)/6)
        if abs(len(self.Emat)-3*idx) > 0.01:
            print ('the order in the "RUSXYZ" class and the imported matrices need to be the same')
            idx = len(self.Emat)/3
            order_emat = int(np.round(-2 + (27*idx+np.sqrt(-3+729*idx**2))**(1/3)/3**(2/3) + 1/(81*idx+3*np.sqrt(-3+729*idx**2))**(1/3), decimals=0))
            print ('the order in the RUSXYZ class is:      ', self.order)
            print ('the order in the imported matrices is: ', order_emat)
            sys.exit()



    def copy_object(self, xyz_object):
        self.block = xyz_object.block
        self.idx   = xyz_object.idx
        self.Emat  = xyz_object.Emat
        self.Itens = xyz_object.Itens


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
        calculates resonance frequencies in MHz;
        pars: dictionary of elastic constants
        nb_freq: number of elastic constants to be displayed
        eigvals_only (True/False): gets only eigenvalues (i.e. resonance frequencies) or also gives eigenvectors (the latter is important when we want to calculate derivatives)
        """
        Gmat = self.G_mat()
        if eigvals_only==True:
            if self.use_quadrants==True:
                w = np.array([])
                for ii in range(8):
                    w = np.concatenate((w, linalg.eigh(Gmat[np.ix_(self.block[ii], self.block[ii])], self.Emat[np.ix_(self.block[ii], self.block[ii])], eigvals_only=True)))
                self.freqs = np.sqrt(np.absolute(np.sort(w))[6:self.nb_freq+6])/(2*np.pi) * 1e-6 # resonance frequencies in MHz
            else:
                # w = linalg.eigh(Gmat, self.Emat, eigvals_only=True)
                # self.freqs = np.sqrt(np.absolute(np.sort(w))[6:self.nb_freq+6])/(2*np.pi) * 1e-6 # resonance frequencies in MHz
                w = linalg.eigh(Gmat, self.Emat, eigvals_only=True, eigvals=(6,self.nb_freq+5))
                self.freqs = np.sqrt(np.absolute(np.sort(w)))/(2*np.pi) * 1e-6 # resonance frequencies in MHz
            return self.freqs
        else:
            # w, a = linalg.eigh(Gmat, self.Emat)
            # a = a.transpose()[np.argsort(w)][6:self.nb_freq+6]
            # self.freqs = np.sqrt(np.absolute(np.sort(w))[6:self.nb_freq+6])/(2*np.pi) * 1e-6 # resonance frequencies in MHz
            w, a = linalg.eigh(Gmat, self.Emat, eigvals=(6,self.nb_freq+5))
            a = a.transpose()[np.argsort(w)]
            self.freqs = np.sqrt(np.absolute(np.sort(w)))/(2*np.pi) * 1e-6 # resonance frequencies in MHz
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
                derivative_matrix[idx, ii] = np.matmul(a[idx].T, np.matmul(Gmat_derivative, a[idx]) ) / ((res*2*np.pi*1e6)**2) * (value)
            ii += 1
        log_derivative = np.zeros((self.nb_freq, len(self.cij_dict)))
        for idx, der in enumerate(derivative_matrix):
            log_derivative[idx] = der #/ sum(der)

        self.cij_dict = cij_dict_original

        if return_freqs == True:
            return log_derivative, f
        elif return_freqs == False:
            return log_derivative



    def print_logarithmic_derivative(self, print_frequencies=True):
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
            freq_text = ''
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



    def log_derivatives_numerical (self, dc=1e-5, N=10, Rsquared_threshold=1e-5, return_freqs=False):
        """
        calculating logarithmic derivatives of the resonance frequencies with respect to elastic constants,
        i.e. (df/dc)*(c/f);
        variables: pars (dictionary of elastic constants), dc, N
        The derivative is calculated by computing resonance frequencies for N different elastic cosntants centered around the value given in pars and spaced by dc.
        A line is then fitted through these points and the slope is extracted as the derivative.
        """
        print ('start taking derivatives ...')

        cij_dict_original = deepcopy(self.cij_dict)
        freq_result = self.compute_resonances()

        fit_results_dict = {}
        Rsquared_matrix = np.zeros([len(freq_result), len(cij_dict_original)])
        log_derivative_matrix = np.zeros([len(freq_result), len(cij_dict_original)])
        # take derivatives with respect to all elastic constants
        print ('These are the \"true\" elastic constnats:')
        print(cij_dict_original)
        ii = 0
        for elastic_constant in sorted(cij_dict_original):
            print ('start taking derivative with respect to ', elastic_constant)
            print ('these are the elastic constants around the true values used for the derivative:')
            t1 = time()
            # create an array of elastic constants centered around the "true" value
            c_result = cij_dict_original[elastic_constant]
            c_derivative_array = np.linspace(c_result-N/2*dc, c_result+N/2*dc, N)
            elasticConstants_derivative_dict = deepcopy(cij_dict_original)

            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # this calculates all the necessary sets of resonance frequencies for the derivative in series
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            freq_derivative_matrix = np.zeros([len(freq_result), N])
            for idx, c in enumerate(c_derivative_array):
                elasticConstants_derivative_dict[elastic_constant] = c
                print (elasticConstants_derivative_dict)
                self.cij_dict = elasticConstants_derivative_dict
                # note we don't actually save the resonance frequencies, but we shift them by the values at the "true" elastic constants;
                # this is done because within the elastic constants in c_test the frequencies change only very little compared to their absolute value,
                # thus this shift is important to get a good fit later
                freq_derivative_matrix[:,idx] = self.compute_resonances()-freq_result

            # shift array of elastic constants to be centered around zero, for similar argument made for the shift of resonance frequencies
            c_derivative_array = c_derivative_array - c_result

            fit_matrix = np.zeros([len(freq_result), N])
            # here we fit a straight line to the resonance frequency vs elastic costants for all resonances
            for idx, freq_derivative_array in enumerate(freq_derivative_matrix):
                # popt, pcov = curve_fit(line, Ctest, freq, p0=[1e-7, 0])
                # slope, y_intercept = np.polyfit(c_derivative_array, freq_derivative_array, 1)
                fit_results_temp = polynomial.polyfit(c_derivative_array, freq_derivative_array, deg=1)
                y_intercept, slope = fit_results_temp
                log_derivative_matrix[idx, ii] = 2 * slope * cij_dict_original[elastic_constant]/freq_result[idx]

                ## check if data really lies on a line
                # offset.append(popt[1])
                # current_fit = slope*c_derivative_array + y_intercept
                current_fit = polynomial.polyval(c_derivative_array, fit_results_temp)
                fit_matrix[idx,:] = current_fit
                # calculate R^2;
                # this is a value judging how well the data is described by a straight line
                SStot = sum( (freq_derivative_array - np.mean(freq_derivative_array))**2 )
                SSres = sum( (freq_derivative_array - current_fit)**2 )
                Rsquared = 1 - SSres/SStot
                Rsquared_matrix[idx, ii] = Rsquared
                # we want a really good fit!
                # R^2 = 1 would be perfect
                if abs(1-Rsquared) > Rsquared_threshold:
                    # if these two fits differ by too much, just print the below line and plot that particular data
                    print ('not sure if data is a straight line for ', elastic_constant, ' at f = ', freq_result[idx], ' MHz')
                    plt.figure()
                    plt.plot(c_derivative_array*1e3, freq_derivative_array*1e6, 'o')
                    plt.plot(c_derivative_array*1e3, current_fit*1e6)
                    plt.title(elastic_constant +'; f = ' + str(round(freq_result[idx], 3)) + ' MHz; $R^2$ = ' + str(round(Rsquared, 7)))
                    plt.xlabel('$\\Delta c$ [kPa]')
                    plt.ylabel('$\\Delta f$ [Hz]')
                    plt.show()
                # else:
                #     print ('looks like a straight line ', elastic_constant, ' ', freq_result[idx]/1e6, ' MHz')
                #     plt.figure()
                #     plt.plot(c_derivative_array/1e3, freq_derivative_array, 'o')
                #     plt.plot(c_derivative_array/1e3, current_fit)
                #     plt.title(elastic_constant +'; f = ' + str(round(freq_result[idx]/1e6, 3)) + ' MHz; $R^2$ = ' + str(round(Rsquared, 10)))
                #     plt.xlabel('$\\Delta c$ [kPa]')
                #     plt.ylabel('$\\Delta f$ [Hz]')
                #     plt.show()

            # store all fit results in this dictionary, just in case you need to look at this at some point later
            fit_results_dict[elastic_constant] = {
                'freq_test': freq_derivative_matrix,
                'c_test': c_derivative_array,
                'fit': fit_matrix,
                'Rsquared': Rsquared_matrix
            }

            # calculate the logarithmic derivative from the derivative
            # log_der = 2 * np.array(derivative) * pars[elastic_constant]/freq_results
            # store it in a dictionary
            # log_derivatives[elastic_constant] = log_der
            print ('derivative with respect to ', elastic_constant, ' done in ', round(time()-t1, 4), ' s')
            ii += 1

        # set the elastic constants back to their original value
        self.cij_dict = cij_dict_original

        if return_freqs == True:
            return (log_derivative_matrix, freq_result)
        else:
            return (log_derivative_matrix)#, fit_results_dict)


if __name__ == "__main__":

    # elastic_dict = {
    #                 'c11': 82.299,
    #                 'c12': 24.767,
    #                 'c13': 12.806,
    #                 'c22': 122.585,
    #                 'c23': 41.304,
    #                 'c33': 78.880,
    #                 'c44': 44.075,
    #                 'c55': 31.502,
    #                 'c66': 26.291
    #                 }

    # N          = 10
    # Emat_path  = f'UTe2_stokes_mesh3_Etens_basis_{N}.npy'
    # Itens_path = f'UTe2_stokes_mesh3_Itens_basis_{N}.npy'

    # rusxyz = RUSXYZ(cij_dict=elastic_dict, symmetry='orthorhombic', order=N,
    #                 load_matrices=True, xyz_object = None,
    #                 Emat_path=Emat_path, Itens_path=Itens_path,
    #                 nb_freq=40,
    #                 angle_x=0, angle_y=0, angle_z=0,
    #                 init=False, use_quadrants=True)
    # rusxyz.initialize()
    # print(rusxyz.compute_resonances())

    # elastic_dict = {
    #                 'c11': 141.362,
    #                 'c12': 57.929,
    #                 'c13': 11.218,
    #                 'c33': 190.680,
    #                 'c44': 49.005
    #                 }

    # N          = 16
    # Emat_path  = f'Mn3Ge_2105A_stokes_mesh3_Etens_basis_{N}.npy'
    # Itens_path = f'Mn3Ge_2105A_stokes_mesh3_Itens_basis_{N}.npy'

    # rusxyz = RUSXYZ(cij_dict=elastic_dict, symmetry='hexagonal', order=N,
    #                 load_matrices=True, xyz_object=None,
    #                 Emat_path=Emat_path, Itens_path=Itens_path,
    #                 nb_freq=50,
    #                 angle_x=0, angle_y=0, angle_z=0,
    #                 init=False, use_quadrants=True)
    # rusxyz.initialize()
    # print(rusxyz.compute_resonances())


    elastic_dict = {
                    'c11': 133.523,
                    'c12': 47.905,
                    'c13': 8.318,
                    'c33': 207.140,
                    'c44': 48.152
                    }

    N          = 16
    Emat_path  = f'Mn3Ge_2104C_stokes_mesh3_Etens_basis_{N}.npy'
    Itens_path = f'Mn3Ge_2104C_stokes_mesh3_Itens_basis_{N}.npy'

    rusxyz = RUSXYZ(cij_dict=elastic_dict, symmetry='hexagonal', order=N,
                    load_matrices=True, matrix_object=None,
                    Emat_path=Emat_path, Itens_path=Itens_path,
                    nb_freq=55,
                    angle_x=0, angle_y=0, angle_z=0,
                    init=False, use_quadrants=False)
    rusxyz.initialize()
    print(rusxyz.compute_resonances())
