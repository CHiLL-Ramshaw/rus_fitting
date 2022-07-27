from unicodedata import decimal
import numpy as np
from scipy import linalg
import sys
from copy import deepcopy
from rus_comsol.elastic_constants import ElasticConstants
from time import time, sleep
from rus_comsol.stokes_matrices import StokesMatrices
from rus_comsol.rpr_matrices import RPRMatrices
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class RUSXYZ(ElasticConstants):
    def __init__(self, cij_dict, symmetry, order, density,
                 Emat_path = None, Itens_path = None,
                 nb_freq=1,
                 angle_x=0, angle_y=0, angle_z=0,
                 init=False, use_quadrants=True):
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
                w = linalg.eigh(Gmat, self.Emat, eigvals_only=True)
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
