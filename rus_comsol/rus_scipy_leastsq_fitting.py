import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import differential_evolution, linear_sum_assignment, leastsq
import time
from copy import deepcopy
import os
import sys
from IPython.display import clear_output
from psutil import cpu_count
from rus_comsol.rus_comsol import RUSComsol
from rus_comsol.rus_xyz import RUSXYZ
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class RUSSCIPYLEASTSQ:
    def __init__(self, rus_object, bounds_dict,
                 freqs_file, nb_freqs, nb_max_missing=0, report_name="",
                 tolerance=0.01, xtol=1e-5, epsfcn=None, use_Jacobian=False):
        """
        freqs_files should contain experimental resonance frequencies in MHz and a weight
        """
        # get initial guess from parameters given to rus_object
        self.rus_object  = rus_object
        self.init_pars   = deepcopy(self.rus_object.cij_dict)
        self.init_pars["angle_x"] = self.rus_object.angle_x
        self.init_pars["angle_y"] = self.rus_object.angle_y
        self.init_pars["angle_z"] = self.rus_object.angle_z
        self.best_pars   = deepcopy(self.init_pars)
        self.best_cij_dict   = deepcopy(self.rus_object.cij_dict)
        # bounds_dict are the bounds given to a genetic algorithm
        self.bounds_dict = bounds_dict
        # the parameters given in bounds_dict are "free" parameters, i.e. they are varied in the fit
        self.free_pars_name  = sorted(self.bounds_dict.keys())
        # fixed parameters are given as parameters which are in init_pars, but NOT in bounds_dict
        # they are fixed parameters, not varied in the fit

        self.fixed_pars_name = np.setdiff1d(sorted(self.init_pars.keys()),
                                             self.free_pars_name)


        ## Load data
        if (use_Jacobian==True) & (nb_max_missing!=0):
            print()
            print('You can only use the Jacobian if you have zero missing frequencies!')
            print('nb_max_missing has been set to zero in the following fit')
            print()
            nb_max_missing = 0
        self.nb_freqs           = nb_freqs
        self.nb_missing         = nb_max_missing
        self.rus_object.nb_freq = nb_freqs + nb_max_missing
        self.freqs_file      = freqs_file
        self.col_freqs       = 0
        self.col_weight      = 1
        self.freqs_data      = None
        self.weight          = None
        self.load_data()


        ## fit algorith
        self.errorbars = False # True if uncertainties have been calculated, False if not


        ## scipy leastsq parameters
        self.tolerance     = tolerance
        self.epsfcn        = epsfcn
        self.use_Jacobian  = use_Jacobian
        self.xtol          = xtol


        self.report_name = report_name

        ## Empty spaces
        self.rms                = None
        self.chi2               = None
        self.rms_list           = []
        self.nb_gens            = 0
        self.best_freqs_found   = []
        self.best_index_found   = []
        self.best_freqs_missing = []
        self.best_index_missing = []

        ## empty spaces for fit properties
        self.fit_output = None
        self.fit_duration = 0

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


    # def assignement(self, freqs_data, freqs_sim):
    #     """
    #     Linear assigment of the simulated frequencies to the data frequencies
    #     in case there is one or more missing frequencies in the data
    #     """
    #     cost_matrix = distance_matrix(freqs_data[:, None], freqs_sim[:, None])**2
    #     index_found = linear_sum_assignment(cost_matrix)[1]
    #     ## index_found is the indices for freqs_sim to match freqs_data
    #     return index_found, freqs_sim[index_found]


    def sort_freqs(self, freqs_sim):
        if self.nb_missing != 0:
            ## Linear assignement of the simulated frequencies to the data
            cost_matrix = distance_matrix(self.freqs_data[:, None], freqs_sim[:, None])**2
            index_found = linear_sum_assignment(cost_matrix)[1]
            freqs_found = freqs_sim[index_found]
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


    def residual_function (self, pars):
        """
        define the residual function used in scipy.optimize.leastsq;
        i.e. (simulated resonance frequencies - data)
        """
        
        for ii, free_name in enumerate(self.free_pars_name):
            if free_name not in ["angle_x", "angle_y", "angle_z"]:
                self.best_pars[free_name]     = pars[ii]
                self.best_cij_dict[free_name] = pars[ii]
            elif free_name=='angle_x':
                self.best_pars[free_name] = pars[ii]
                self.rus_object.angle_x   = pars[ii]
            elif free_name=='angle_y':
                self.best_pars[free_name] = pars[ii]
                self.rus_object.angle_y   = pars[ii]
            elif free_name=='angle_z':
                self.best_pars[free_name] = pars[ii]
                self.rus_object.angle_z   = pars[ii]
        self.rus_object.cij_dict = self.best_cij_dict

        # calculate resonances with new parameters
        freqs_sim = self.rus_object.compute_resonances()
        # find missing resonances
        freqs_found, index_found, freqs_missing, index_missing = self.sort_freqs(freqs_sim)

        # update attributes
        self.nb_gens += 1
        self.best_freqs_found   = freqs_found
        self.best_index_found   = index_found
        self.best_freqs_missing = freqs_missing
        self.best_index_missing = index_missing

        # this is what we want to be minimized
        diff = (self.best_freqs_found - self.freqs_data) / self.best_freqs_found * self.weight
        self.rms = np.sqrt(np.sum((diff[diff!=0])**2) / len(diff[diff!=0])) * 100
        self.rms_list.append(self.rms)
        self.chi2 = np.sum(diff**2)

        print ('NUMBER OF GENERATIONS: ', self.nb_gens)
        print ('BEST PARAMETERS:')
        for key, item in self.best_pars.items():
            print ('\t',key, ': ', round(item, 5))
        print ('MISSING FREQUENCIES: ', np.round(np.array(freqs_missing)[np.array(index_missing)<len(self.freqs_data)], 3), ' MHz')
        print ('RMS: ', round(self.rms, 5), ' %')
        print ('')
        print ('#', 50*'-')
        print ('')

        return diff



    def jacobian_residual_function (self):
        """
        define the jacobian of the residual function used in lmfit;
        i.e. (simulated resonance frequencies - data)
        """
        print ('calculate jacobian ...')
        # alpha is logarithmic derivative, i.e. (df/dc)*(c/f)
        if isinstance(self.rus_object, RUSXYZ):
            alpha, f = self.rus_object.log_derivatives_analytical(return_freqs=True)
        else:
            print("we can only calculate the jacobian for the RUS_xyz class")
            sys.exit()

        c_matrix      = np.zeros_like(alpha)
        f_calc_matrix = np.zeros_like(alpha)
        f_data_matrix = np.zeros_like(alpha)
        ii = 0      
        for el in sorted(self.rus_object.cij_dict):
            c_matrix[:,ii]      = np.ones(len(f)) * self.rus_object.cij_dict[el]
            f_calc_matrix[:,ii] = f
            f_data_matrix[:,ii] = self.freqs_data
            ii+=1
        
        # now we can calculate the derivatives
        dfdc = alpha * f_calc_matrix / c_matrix

        # residual function is (f-f_data)/f = 1 - f_data/f
        # the derivative of that is therefore (f_data/f^2) * df/dc
        d_residual_function = f_data_matrix / f_calc_matrix**2 * dfdc
  
        print ('calculated jacobian!')
        return d_residual_function




    def run_fit (self, print_derivatives=False):
        if isinstance(self.rus_object, RUSComsol) and (self.rus_object.client is None):
            print ("the rus_comsol object was not started!")
            print ("it is being initialized right now ...")
            self.rus_object.start_comsol()
        if isinstance(self.rus_object, RUSXYZ) and (self.rus_object.Emat is None):
            print ("the rus_rpr object was not initialized!")
            print ("it is being initialized right now ...")
            self.rus_object.initialize()

        # start timer
        t0 = time.time()

        # define jacobian function
        if self.use_Jacobian==True and len(self.free_pars_name)==len(self.rus_object.cij_dict):
            # here I define a function calculating the jacobian to be called in "minimize"
            # however, I don't want it to recalculate the jacobian every time, so I just define my function such that it
            # calculated it once at the beginning at returns the same one all the time
            jacobian = self.jacobian_residual_function()
            def dfunc (params):
                return jacobian
        elif self.use_Jacobian==False and len(self.free_pars_name)==len(self.rus_object.cij_dict):
            dfunc = None
        else:
            print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
            print()
            print('As of now you can only use the Jacobian if the only free parameters are elastic moduli!')
            print('If you want to use the Jacobian you need to fix angles and other parameters, and not let them vary during the fit!')
            print('The following fit will NOT be using the Jacobian')
            print()
            print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->')
            dfunc = None


        # create list of initial parameters
        x0 = []
        for free_name in self.free_pars_name:
            if free_name not in ["angle_x", "angle_y", "angle_z"]:
                x0.append(self.rus_object.cij_dict[free_name])                    
            elif free_name=='angle_x':
                x0.append(self.rus_object.angle_x)
            elif free_name=='angle_y':
                x0.append(self.rus_object.angle_y)
            elif free_name=='angle_z':
                x0.append(self.rus_object.angle_z)
        # run fit
        fit_output = leastsq(self.residual_function, x0=x0, ftol=self.tolerance, xtol=self.xtol, epsfcn=self.epsfcn, full_output=1, Dfun=dfunc)

        # fit out put
        self.fit_output = fit_output
        # ask if error bars have been calculated
        self.errorbars = ( fit_output[1] is not None )

        
        # stop timer
        self.fit_duration = time.time() - t0


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
        return self.rms_list

    # the following methods are just to display the fit report and data in a nice way

    def report_best_pars(self):
        report = "#Variables" + '-'*(70) + '\n'
        for ii, free_name in enumerate(self.free_pars_name):
            if free_name[0] == "c": unit = "GPa"
            else: unit = "deg"

            if self.errorbars:
                cov_x        = self.fit_output[1]
                N_points     = self.nb_freqs
                N_variables  = len(self.bounds_dict)
                reduced_chi2 = self.chi2 / (N_points - N_variables)
                err          = np.sqrt( np.diag( cov_x*reduced_chi2 ) )[ii]

                report+= "\t# " + free_name + " : (" + r"{0:.16f}".format(self.best_pars[free_name]) + " +- " + \
                         r"{0:.16f}".format(err) + ') ' + unit + \
                         " (init = [" + str(self.bounds_dict[free_name]) + \
                         ", " +         unit + "])" + "\n"
            else:
                report+= "\t# " + free_name + " : " + r"{0:.16f}".format(self.best_pars[free_name]) + " " + \
                         unit + \
                         " (init = [" + str(self.bounds_dict[free_name]) + \
                         ", " +         unit + "])" + "\n"
        report+= "#Fixed values" + '-'*(67) + '\n'
        if len(self.fixed_pars_name) == 0:
            report += "\t# " + "None" + "\n"
        else:
            for fixed_name in self.fixed_pars_name:
                if fixed_name[0] == "c": unit = "GPa"
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
        _, _, infodict, _, ier = fit_output
        success = ( ier in np.array([1,2,3,4], dtype=int) )
        nfev    = infodict['nfev']
        chi2    = self.chi2


        duration    = np.round(self.fit_duration, 2)
        N_points    = self.nb_freqs
        N_variables = len(self.bounds_dict)
        
        reduced_chi2 = chi2 / (N_points - N_variables)
        report = "#Fit Statistics" + '-'*(65) + '\n'
        report+= "\t# Fitting Class      \t= rus_lmfit\n"
        report+= "\t# fitting method     \t= " + 'scipy.optmimize.leastsq' + "\n"
        report+= "\t# data points        \t= " + str(N_points) + "\n"
        report+= "\t# variables          \t= " + str(N_variables) + "\n"
        report+= "\t# fit success        \t= " + str(success) + "\n"
        # report+= "\t# generations        \t= " + str(fit_output.nit) + " + 1" + "\n"
        report+= "\t# function evals     \t= " + str(nfev) + "\n"
        report+= "\t# fit duration       \t= " + str(duration) + " seconds" + "\n"
        report+= "\t# chi-square         \t= " + r"{0:.8f}".format(chi2) + "\n"
        report+= "\t# reduced chi-square \t= " + r"{0:.8f}".format(reduced_chi2) + "\n"
        return report


    def report_best_freqs(self, nb_additional_freqs=10):
        if (nb_additional_freqs != 0) or (self.best_freqs_found == []):
            if isinstance(self.rus_object, RUSComsol) and (self.rus_object.client is None):
                self.rus_object.start_comsol()
            if isinstance(self.rus_object, RUSXYZ) and (self.rus_object.Emat is None):
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
        # for i in range(len(freqs_data)):
        #     if freqs_data[i] != 0:
        #         diff[i] = np.abs(freqs_data[i]-freqs_sim[i]) / freqs_data[i] * 100 * weight[i]
        diff = np.abs(freqs_data-freqs_sim[:len(freqs_found) + len(freqs_missing)]) / freqs_sim[:len(freqs_found) + len(freqs_missing)] * 100 * weight
        rms = np.sqrt( np.sum(diff[diff!=0]**2) / len(diff[diff!=0]) )

        template = "{0:<8}{1:<23}{2:<23}{3:<13}{4:<8}"
        report  = template.format(*['#index', 'f exp(MHz)', 'f calc(MHz)', 'diff (%)', 'weight']) + '\n'
        report += '#' + '-'*(79) + '\n'
        for j in range(len(freqs_sim)):
            if j < len(freqs_data):
                report+= template.format(*[j, np.round(freqs_data[j],16), np.round(freqs_sim[j],16), np.round(diff[j], 3), np.round(weight[j], 0)]) + '\n'
            else:
                report+= template.format(*[j, 0,                         np.round(freqs_sim[j],16), 0,                    0])                      + '\n'
        report += '#' + '-'*(79) + '\n'
        report += '# RMS = ' + str(np.round(rms,16)) + ' %\n'
        report += '#' + '-'*(79) + '\n'

        return report

    def report_sample_text(self):
        sample_template = "{0:<40}{1:<20}"
        sample_text = '# [[Sample Characteristics]] \n'
        sample_text += '# ' + sample_template.format(*['crystal symmetry:', self.rus_object.symmetry]) + '\n'
        if isinstance(self.rus_object, RUSXYZ):
            # sample_text += '# ' + sample_template.format(*['sample dimensions (mm) (x,y,z):', str(self.rus_object.dimensions*1e3)]) + '\n'
            sample_text += '# ' + sample_template.format(*['density (kg/m^3):', self.rus_object.density]) + '\n'
            sample_text += '# ' + sample_template.format(*['highest order basis polynomial:', self.rus_object.order]) + '\n'
            sample_text += '# ' + sample_template.format(*['resonance frequencies calculated with:', 'RUS_XYZ']) + '\n'
        if isinstance(self.rus_object, RUSComsol):
            sample_text += '# ' + sample_template.format(*['Comsol file:', self.rus_object.mph_file]) + '\n'
            sample_text += '# ' + sample_template.format(*['density (kg/m^3):', self.rus_object.density]) + '\n'
            sample_text += '# ' + sample_template.format(*['resonance frequencies calculated with:', 'Comsol']) + '\n'
        return sample_text


    def report_total(self, comsol_start=True):
        report_fit = self.report_fit()
        report_fit += self.report_best_pars()
        if isinstance(self.rus_object, RUSXYZ):
            freq_text  = self.report_best_freqs(nb_additional_freqs=10)
            der_text = self.rus_object.print_logarithmic_derivative(print_frequencies=False)
        if isinstance(self.rus_object, RUSComsol):
            freq_text  = self.report_best_freqs(nb_additional_freqs=10)
            der_text = self.rus_object.print_logarithmic_derivative(print_frequencies=False, comsol_start=False)
            self.rus_object.stop_comsol()

        sample_text = self.report_sample_text()

        data_text = ''
        freq_text_split = freq_text.split('\n')
        freq_text_prepend = ['#'+' '*(len(freq_text_split[0])-1)] + freq_text_split
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
            index = self.freqs_file[::-1].find('/')
            if index == -1:
                index = self.freqs_file[::-1].find('\\')
            self.report_name = self.freqs_file[:-index] + "fit_report.txt"

            try:
                report_file = open(self.report_name, "w")
                report_file.write(report)
                report_file.close()
            except:
                print ('there was a problem with the filename for the fit report')
                print ('it will be saved in the current working directory')
                self.report_name = "fit_report.txt"
                report_file = open(self.report_name, "w")
                report_file.write(report)
                report_file.close()
        else:
            report_file = open(self.report_name, "w")
            report_file.write(report)
            report_file.close()


