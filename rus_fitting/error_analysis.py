import numpy as np
import time
from copy import deepcopy
import os
import sys
import scipy





class ErrorAnalysis:

    def __init__(self, rus_object, fit_report_path, percent=2, scan_width=15, save_path=None):
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
        self.rus_object.cij_dict = el_dict
        freqs_sim = self.rus_object.compute_resonances()

        diff    = (freqs_sim - self.freqs_data)/freqs_sim * self.weight
        chi2    = np.sum(diff[diff!=0]**2)
        rms     = np.sqrt(chi2/len(diff[diff!=0])) * 100

        return rms


    def residual_function(self, el_const_value, el_const_name, percent):
        el_dict                = deepcopy(self.original_cij_dict)
        el_dict[el_const_name] = el_const_value

        rms        = self.find_rms(el_dict)
        rms_bisect = 100*(rms-self.rms_min)/self.rms_min - percent

        return rms_bisect


    def find_single_root (self, el_const_name, percent, scan_width=15):
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