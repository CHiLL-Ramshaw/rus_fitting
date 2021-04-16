import mph
from copy import deepcopy
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class RUSComsol:
    def __init__(self, pars,
                 mph_file,
                 nb_freq,
                 study_name="Study 1",
                 study_tag="std1",
                 init=False):
        self._pars      = deepcopy(pars)
        self.pars_name  = sorted(self._pars.keys())
        self._nb_freq   = nb_freq
        self.mph_file   = mph_file
        self.study_name = study_name
        self.study_tag  = study_tag
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

    def _get_pars(self):
        return self._pars
    def _set_pars(self, pars):
        self._pars = deepcopy(pars)
    pars = property(_get_pars, _set_pars)


    def compute_freqs(self, pars=None):
        ## Set number of frequencies --------------------------------------------
        self.model.parameter('nb_freq', str(self._nb_freq + 6))
        ## Set parameters  ------------------------------------------------------
        if pars != None:
            self.pars = pars
        for pars_name in self.pars_name:
            self.model.parameter(pars_name, str(self._pars[pars_name][0]) +
                                 "[" + self._pars[pars_name][1] + "]")
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
