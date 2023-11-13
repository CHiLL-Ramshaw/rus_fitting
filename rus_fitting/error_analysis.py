import numpy as np
import time
from copy import deepcopy
import os
import sys
import scipy





class ErrorAnalysis:

    def __init__(self, rus_object, fit_report_path, percent=2, scan_width=15, save_path=None):
        """
        - calculates an uncertainty of elastic constants from an RUS fit:
        - the uncertainty is defined as: how much do we have to change a given elastic constant such that the RMS increases by "percent" percent?
            - define a residual function, which gives the relative change in RMS with respect to the original fit minus "percent"
            - use scipy.optimize.brentq to find the root of the residual fct within a range of elastic moduli given by "scan_width"
            - the root is the uncertainty
            - this uncertainty can be different for values above/below the original fit result; the maximum uncertainty is picked here
        - rus_object: an RUS object used to calculate resonance frequencies from set of elastic constants (can be RUSComsol or RUSXYZ)
            - rus_object needs to be initialized with "correct" elastic constants from original fit
        - fit_report_path: location of the fit report: mostly used to extract experimental resonances used in fit to get accurate RMS
        - percent (2 %): how much do you want your RMS to increase for your definition of the uncertainty
        - scan width (15 GPa): how far away from the original elastic constants in fit_report_path are you looking for a "percent" increase of the RMS
        - save_path (None): where do you want to save the result of the error analysis (won't be saved if None)
        """
        self.rus_object = rus_object
        self.fit_report = fit_report_path
        self.save_path  = save_path

        self.freqs_data, self.weight = self.load_data(self.fit_report)

        # set nb_freq so that in the future we will calculate the correct amount of resonances
        self.rus_object.nb_freq = len(self.freqs_data)

        self.original_cij_dict = deepcopy(self.rus_object.cij_dict)     # dictionary of elastic constants given from previous fit
        self.cij_error_dict    = {}                                     # initialize dictionary to store errors of elastic constants

        self.rms_min = self.find_rms(self.original_cij_dict)

        self.percent    = percent     # by how much do we want the rms to increase
        self.scan_width = scan_width  # how far away from the correct elastic constants are we going to search


    def load_data (self, fit_report):
        """
        import experimental resonances from fit report
        """
        data   = np.loadtxt(fit_report).T
        index  = data[0].astype(int)
        fexp   = data[1]
        fcalc  = data[2]
        diff   = data[3]
        weight = data[4].astype(int)

        # these arrays are populated with calculated resonances after the last measured resonace
        # here we are trying to get rid of those
        idx_ones = index[weight>0]
        last_idx = max(idx_ones)
        # last_idx = idx_ones[-1]

        # so this is the relevant data
        weight = weight[:last_idx+1]
        fexp   = fexp[:last_idx+1]

        return fexp, weight


    def find_rms(self, el_dict):
        """
        calculate the RMS of experimental resonances minus resonances of self.rus_object calculated with elastic constants in el_dict
        """
        self.rus_object.cij_dict = el_dict
        freqs_sim = self.rus_object.compute_resonances()

        diff    = (freqs_sim - self.freqs_data)/freqs_sim * self.weight
        chi2    = np.sum(diff[diff!=0]**2)
        rms     = np.sqrt(chi2/len(diff[diff!=0])) * 100

        return rms


    def residual_function(self, el_const_value, el_const_name, percent):
        """
        define residual function as relative change of rms with respect to fit result minus "percent"
        """
        el_dict                = deepcopy(self.original_cij_dict)
        el_dict[el_const_name] = el_const_value

        rms        = self.find_rms(el_dict)
        rms_bisect = 100*(rms-self.rms_min)/self.rms_min - percent

        return rms_bisect


    def find_single_root (self, el_const_name, percent, scan_width=15):
        """
        - find uncertainty of one elastic constant defined by "el_const_name" by finding root of self.residual_function
        - find root above and below original value; use maximum difference as uncertainty
        """
        el_const_initial = self.original_cij_dict[el_const_name]

        # find root above the initial value
        el_const_final = el_const_initial + scan_width
        root_above     = scipy.optimize.brentq(self.residual_function, el_const_initial, el_const_final, args=(el_const_name, percent))
        diff_above     = root_above - el_const_initial

        # find root above the initial value
        el_const_final = el_const_initial - scan_width
        root_below     = scipy.optimize.brentq(self.residual_function, el_const_final, el_const_initial, args=(el_const_name, percent))
        diff_below     = el_const_initial - root_below

        if diff_above > diff_below:
            root_max = root_above
            diff_max = diff_above
        else:
            root_max = root_below
            diff_max = diff_below

        return diff_max, root_max


    def find_all_errors (self):
        """
        find uncertainties of all elastic constants by repeating self.find_single_root
        """
        error_text = f'the rms changes by {self.percent} % if we change the elastic moduli by:'
        print(error_text)
        for cij in self.original_cij_dict:
            diff_max, root_max       = self.find_single_root(cij, self.percent, self.scan_width)
            diff_max                 = np.round(diff_max, 5)
            self.cij_error_dict[cij] = diff_max

            temp_text = f'{cij} = ( {self.original_cij_dict[cij]} +- {diff_max} ) GPa'
            print( temp_text )
            error_text += '\n'
            error_text += temp_text

        if self.save_path is not None:
            report_file = open(self.save_path, "w")
            report_file.write(error_text)
            report_file.close()

        return error_text