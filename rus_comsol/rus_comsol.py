import mph
from rus_comsol.elastic_constants import ElasticConstants
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


    def compute_resonances(self):
        ## Set number of frequencies --------------------------------------------
        self.model.parameter('nb_freq', str(self._nb_freq + 6))
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


    def stop_comsol(self):
        self.client.clear()



