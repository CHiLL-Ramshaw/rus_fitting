import mph
from rus_comsol.elastic_constants import ElasticConstants
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from time import time
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class RUSComsol(ElasticConstants):
    def __init__(self, cij_dict, symmetry,
                 mph_file,
                 nb_freq,
                 angle_x=0, angle_y=0, angle_z=0,
                 study_name="resonances",
                 study_tag="std1",
                 init=False):
        super().__init__(cij_dict,
                         symmetry=symmetry,
                         angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
        self.mph_file   = mph_file
        self.study_name = study_name
        self.study_tag  = study_tag
        self._nb_freq   = nb_freq
        self.client     = None
        self.model      = None
        self.freqs      = None
        if init == True:
            self.start_comsol()

    ## Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def _get_nb_freq(self):
        return self._nb_freq
    def _set_nb_freq(self, nb_freq):
        self._nb_freq = nb_freq
    nb_freq = property(_get_nb_freq, _set_nb_freq)


    def compute_resonances(self, nb_freq=None, voigt_dict=None):
        if nb_freq == None:
            nb_freq = self._nb_freq
        if voigt_dict is None:
            voigt_dict = self.voigt_dict
        ## Set number of frequencies --------------------------------------------
        self.model.parameter('nb_freq', str(nb_freq + 6))
        ## Set parameters  ------------------------------------------------------
        for c_name in sorted(voigt_dict.keys()):
            self.model.parameter(c_name, str(voigt_dict[c_name]) + " [GPa]")
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


    def stop_comsol(self):
        self.client.clear()



    def log_derivatives_numerical (self, cij_dict=None, dc=1e-4, N=5, Rsquared_threshold=1e-5, nb_freq=None, return_freqs=False):
        """
        calculating logarithmic derivatives of the resonance frequencies with respect to elastic constants,
        i.e. (df/dc)*(c/f);
        variables: pars (dictionary of elastic constants), dc, N
        The derivative is calculated by computing resonance frequencies for N different elastic cosntants centered around the value given in pars and spaced by dc.
        A line is then fitted through these points and the slope is extracted as the derivative.
        """
        print ('start taking derivatives ...')
        if cij_dict is None:
            cij_dict = self.cij_dict
        if nb_freq is None:
            nb_freq = self.nb_freq

        voigt_dict = self.cij_dict_to_voigt_dict(cij_dict=cij_dict)
        freq_result = self.compute_resonances(nb_freq=nb_freq, voigt_dict=voigt_dict)

        fit_results_dict = {}
        Rsquared_matrix = np.zeros([len(freq_result), len(cij_dict)])
        log_derivative_matrix = np.zeros([len(freq_result), len(cij_dict)])
        # take derivatives with respect to all elastic constants
        print ('These are the \"true\" elastic constnats:')
        print(cij_dict)
        ii = 0
        for elastic_constant in sorted(cij_dict):
            print ('start taking derivative with respect to ', elastic_constant)
            print ('these are the elastic constants around the true values used for the derivative:')
            t1 = time()
            # create an array of elastic constants centered around the "true" value
            c_result = cij_dict[elastic_constant]
            c_derivative_array = np.linspace(c_result-N/2*dc, c_result+N/2*dc, N)
            elasticConstants_derivative_dict = deepcopy(cij_dict)

            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # this calculates all the necessary sets of resonance frequencies for the derivative in series
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            freq_derivative_matrix = np.zeros([len(freq_result), N])
            for idx, c in enumerate(c_derivative_array):
                elasticConstants_derivative_dict[elastic_constant] = c
                print (elasticConstants_derivative_dict)
                parameter_set_voigt = self.cij_dict_to_voigt_dict(cij_dict=elasticConstants_derivative_dict)
                # note we don't actually save the resonance frequencies, but we shift them by the values at the "true" elastic constants;
                # this is done because within the elastic constants in c_test the frequencies change only very little compared to their absolute value,
                # thus this shift is important to get a good fit later
                freq_derivative_matrix[:,idx] = self.compute_resonances(voigt_dict=parameter_set_voigt, nb_freq=nb_freq)-freq_result

            # shift array of elastic constants to be centered around zero, for similar argument made for the shift of resonance frequencies
            c_derivative_array = c_derivative_array - c_result

            fit_matrix = np.zeros([len(freq_result), N])
            # here we fit a straight line to the resonance frequency vs elastic costants for all resonances
            for idx, freq_derivative_array in enumerate(freq_derivative_matrix):
                # popt, pcov = curve_fit(line, Ctest, freq, p0=[1e-7, 0])
                slope, y_intercept = np.polyfit(c_derivative_array, freq_derivative_array, 1)
                log_derivative_matrix[idx, ii] = 2 * slope * cij_dict[elastic_constant]/freq_result[idx]

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
                    plt.plot(c_derivative_array/1e3, freq_derivative_array, 'o')
                    plt.plot(c_derivative_array/1e3, current_fit)
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

        if return_freqs == True:
            return (log_derivative_matrix, freq_result)
        else:
            return (log_derivative_matrix)#, fit_results_dict)