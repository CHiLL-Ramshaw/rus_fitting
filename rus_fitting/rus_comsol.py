import mph
from rus_fitting.elastic_constants import ElasticConstants
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from time import time
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class RUSComsol(ElasticConstants):
    def __init__(self, cij_dict, symmetry,
                 density, mph_file,
                 nb_freq=1, mesh=5,
                 angle_x=0, angle_y=0, angle_z=0,
                 study_name="resonances",
                 study_tag="std1",
                 init=False):
        """
        density (float): expressed in kg/m^3
        mesh (int): goes from 1 for Extremely fine to 9 for Extremely coarse
        """
        super().__init__(cij_dict,
                         symmetry=symmetry,
                         angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
        self.mph_file   = mph_file
        self.study_name = study_name
        self.study_tag  = study_tag
        self.nb_freq    = nb_freq
        self.client     = None
        self.model      = None
        self.freqs      = None
        self._density   = density # in kg/m^3
        self._mesh       = mesh
        if init == True:
            self.start_comsol()

    ## Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def _get_density(self):
        return self._density
    def _set_density(self, density):
        self._density = density
        self.model.parameter('rho', str(self._density) + " [kg/m^3]")
    density = property(_get_density, _set_density)

    def _get_mesh(self):
        return self._mesh
    def _set_mesh(self, mesh):
        self._mesh = mesh
        self.model.java.component('comp1').mesh("mesh1").autoMeshSize(self._mesh)
    mesh = property(_get_mesh, _set_mesh)


    def compute_resonances(self):
        ## Set number of frequencies --------------------------------------------
        self.model.parameter('nb_freq', str(self.nb_freq + 6))
        ## Set parameters  ------------------------------------------------------
        for c_name in sorted(self.voigt_dict.keys()):
            self.model.parameter(c_name, str(self.voigt_dict[c_name]) + " [GPa]")
        ## Compute resonances ---------------------------------------------------
        self.model.solve(self.study_name)
        self.freqs = self.model.evaluate('abs(freq)', 'MHz')[6:]
        self.model.clear()
        self.model.reset()
        return self.freqs


    def start_comsol(self):
        """Initialize the COMSOL file"""
        self.client = mph.Client()
        self.model = self.client.load(self.mph_file)
        ## Forces to get all the resonances from 0 MHz
        self.model.java.study(self.study_tag).feature("eig").set('shiftactive', 'on')
        self.model.java.study(self.study_tag).feature("eig").set('shift', '0')
        ## Set density ----------------------------------------------------------
        self.model.parameter('rho', str(self._density) + " [kg/m^3]")
        ## Set Mesh -------------------------------------------------------------
        self.model.java.component('comp1').mesh("mesh1").automatic(True)
        self.model.java.component('comp1').mesh("mesh1").autoMeshSize(self._mesh)


    def stop_comsol(self):
        self.client.clear()
        self.client = None
        self.model  = None


    def log_derivatives_numerical (self, dc=1e-4, N=5, Rsquared_threshold=1e-5, return_freqs=False):
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
                slope, y_intercept = np.polyfit(c_derivative_array, freq_derivative_array, 1)
                log_derivative_matrix[idx, ii] = 2 * slope * cij_dict_original[elastic_constant]/freq_result[idx]

                ## check if data really lies on a line
                # offset.append(popt[1])
                current_fit = slope*c_derivative_array + y_intercept
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
                    plt.plot(c_derivative_array, current_fit*1e6)
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




    def print_logarithmic_derivative (self, comsol_start=False, print_frequencies=True):
        if comsol_start == False:
            log_der, freqs_calc = self.log_derivatives_numerical(return_freqs=True)
        else:
            self.start_comsol()
            log_der, freqs_calc = self.log_derivatives_numerical(return_freqs=True)

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