import numpy as np
from numpy import cos, sin
from copy import deepcopy
import sys

class ElasticConstants:
    def __init__(self, cij_dict, symmetry,
                angle_x=0, angle_y=0, angle_z=0):
        """
        - class deals with the elastic tensor: main objective is to rotate elastic moduli
        - cij_dict: dictionary of elastic constants in Voigt notation (units are GPa)
        - symmetry: symmetry of the crystal lattice; options are: "cubic", "tetragonal", "orthorhombic", "hexagonal", "rhombohedral"
        - angle_i: rotation around i axis in degrees
        """
        self._cij_dict = cij_dict
        self.symmetry  = symmetry
        self._angle_x  = angle_x
        self._angle_y  = angle_y
        self._angle_z  = angle_z

        ## Build Voigt
        self.voigt_matrix = self.cij_dict_to_voigt_matrix()
        self.R = self.rotation_matrix()
        self.rotation_voigt()
        self.voigt_dict = self.voigt_matrix_to_voigt_dict()
        self.cijkl = self.voigt_matrix_to_tensor()


    ## Special Method >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def __setitem__(self, key, value):
        ## Add security not to add keys later
        if key not in self._cij_dict.keys():
            print(key + " was not added (new band parameters are only allowed within object initialization)")
        else:
            self._cij_dict[key] = value
            self._update()

    def __getitem__(self, key):
        try:
            assert self._cij_dict[key]
        except KeyError:
            print(key + " is not a defined in cij_dict")
        else:
            return self._cij_dict[key]

    def get_cij_value(self, key):
        return self[key]

    def set_cij_value(self, key, val):
        self[key] = val


    ## Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def _get_cij_dict(self):
        return self._cij_dict
    def _set_cij_dict(self, cij_dict):
        self._cij_dict = deepcopy(cij_dict)
        self._update()
    cij_dict = property(_get_cij_dict, _set_cij_dict)

    def _get_angle_x(self):
        return self._angle_x
    def _set_angle_x(self, angle_x):
        self._angle_x = angle_x
        self._update()
    angle_x = property(_get_angle_x, _set_angle_x)

    def _get_angle_y(self):
        return self._angle_y
    def _set_angle_y(self, angle_y):
        self._angle_y = angle_y
        self._update()
    angle_y = property(_get_angle_y, _set_angle_y)

    def _get_angle_z(self):
        return self._angle_z
    def _set_angle_z(self, angle_z):
        self._angle_z = angle_z
        self._update()
    angle_z = property(_get_angle_z, _set_angle_z)


    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def _update(self):
        self.rotation_matrix()
        self.cij_dict_to_voigt_matrix()
        self.rotation_voigt()
        self.voigt_matrix_to_voigt_dict()
        self.voigt_matrix_to_tensor()


    def cij_dict_to_voigt_matrix(self):
        """
        converts dictionary of elastic constants to 6x6 matrix in Voigt notation
        """
        voigt_matrix = np.zeros((6,6))

        if self.symmetry not in ['cubic', 'tetragonal', 'orthorhombic', 'rhombohedral', 'hexagonal']:
            print ('your provided symmetry is not in ', "['cubic', 'tetragonal', 'orthorhombic', 'rhombohedral', 'hexagonal']")
            sys.exit()

        if self.symmetry=="cubic":
            voigt_matrix[0,0] = voigt_matrix[1,1] = voigt_matrix[2,2] = self.cij_dict['c11']
            voigt_matrix[0,1] = voigt_matrix[0,2] = voigt_matrix[1,2] = self.cij_dict['c12']
            voigt_matrix[3,3] = voigt_matrix[4,4] = voigt_matrix[5,5] = self.cij_dict['c44']

        elif self.symmetry=="tetragonal":
            voigt_matrix[0,0] = voigt_matrix[1,1] = self.cij_dict['c11']
            voigt_matrix[2,2]                     = self.cij_dict['c33']
            voigt_matrix[0,1]                     = self.cij_dict['c12']
            voigt_matrix[0,2] = voigt_matrix[1,2] = self.cij_dict['c13']
            voigt_matrix[3,3] = voigt_matrix[4,4] = self.cij_dict['c44']
            voigt_matrix[5,5]                     = self.cij_dict['c66']

        elif self.symmetry=="orthorhombic":
            voigt_matrix[0,0] = self.cij_dict['c11']
            voigt_matrix[1,1] = self.cij_dict['c22']
            voigt_matrix[2,2] = self.cij_dict['c33']
            voigt_matrix[0,1] = self.cij_dict['c12']
            voigt_matrix[0,2] = self.cij_dict['c13']
            voigt_matrix[1,2] = self.cij_dict['c23']
            voigt_matrix[3,3] = self.cij_dict['c44']
            voigt_matrix[4,4] = self.cij_dict['c55']
            voigt_matrix[5,5] = self.cij_dict['c66']

        elif self.symmetry=="hexagonal":                    # hexagonal
            indicator = np.any( np.array([i=='c11' for i in self.cij_dict]) )
            if indicator == True:
                voigt_matrix[0,0] = self.cij_dict['c11']
                voigt_matrix[1,1] = self.cij_dict['c11']
                voigt_matrix[2,2] = self.cij_dict['c33']
                voigt_matrix[0,1] = self.cij_dict['c12']
                voigt_matrix[0,2] = self.cij_dict['c13']
                voigt_matrix[1,2] = self.cij_dict['c13']
                voigt_matrix[3,3] = self.cij_dict['c44']
                voigt_matrix[4,4] = self.cij_dict['c44']
                voigt_matrix[5,5] = (self.cij_dict['c11']-self.cij_dict['c12'])/2
            else:
                voigt_matrix[0,0] = 2*self.cij_dict['c66'] + self.cij_dict['c12']
                voigt_matrix[1,1] = 2*self.cij_dict['c66'] + self.cij_dict['c12']
                voigt_matrix[2,2] = self.cij_dict['c33']
                voigt_matrix[0,1] = self.cij_dict['c12']
                voigt_matrix[0,2] = self.cij_dict['c13']
                voigt_matrix[1,2] = self.cij_dict['c13']
                voigt_matrix[3,3] = self.cij_dict['c44']
                voigt_matrix[4,4] = self.cij_dict['c44']
                voigt_matrix[5,5] = self.cij_dict['c66']

        elif self.symmetry=="rhombohedral":
            # rhomnohedral symmetry is special, as it contains a non-zero c14!
            indicator = np.any( np.array([i=='c11' for i in self.cij_dict]) )
            if indicator == True:
                voigt_matrix[0,0] = self.cij_dict['c11']
                voigt_matrix[1,1] = self.cij_dict['c11']
                voigt_matrix[2,2] = self.cij_dict['c33']
                voigt_matrix[0,1] = self.cij_dict['c12']
                voigt_matrix[0,2] = self.cij_dict['c13']
                voigt_matrix[1,2] = self.cij_dict['c13']
                voigt_matrix[3,3] = self.cij_dict['c44']
                voigt_matrix[4,4] = self.cij_dict['c44']
                voigt_matrix[5,5] = (self.cij_dict['c11']-self.cij_dict['c12'])/2
                voigt_matrix[0,3] = self.cij_dict['c14']
                voigt_matrix[1,3] = -self.cij_dict['c14']
                voigt_matrix[4,5] = self.cij_dict['c14']
            else:
                voigt_matrix[0,0] = 2*self.cij_dict['c66'] + self.cij_dict['c12']
                voigt_matrix[1,1] = 2*self.cij_dict['c66'] + self.cij_dict['c12']
                voigt_matrix[2,2] = self.cij_dict['c33']
                voigt_matrix[0,1] = self.cij_dict['c12']
                voigt_matrix[0,2] = self.cij_dict['c13']
                voigt_matrix[1,2] = self.cij_dict['c13']
                voigt_matrix[3,3] = self.cij_dict['c44']
                voigt_matrix[4,4] = self.cij_dict['c44']
                voigt_matrix[5,5] = self.cij_dict['c66']
                voigt_matrix[0,3] = self.cij_dict['c14']
                voigt_matrix[1,3] = -self.cij_dict['c14']
                voigt_matrix[4,5] = self.cij_dict['c14']

        self.voigt_matrix = (voigt_matrix + voigt_matrix.T
                             - np.diag(voigt_matrix.diagonal()))
        return self.voigt_matrix


    def voigt_matrix_to_voigt_dict(self):
        """
        convert 6x6 matrix of elastic contants in Voigt notation to dictionary of elastic constants
        """
        self.voigt_dict = {}
        voigt_matrix = np.triu(self.voigt_matrix) # get only upper part of the matrix
        for i in range(6):
            for j in range(i, 6):
                self.voigt_dict["c"+str(i+1)+str(j+1)] = voigt_matrix[i,j]
        return self.voigt_dict


    def rotation_matrix(self):
        """
        - define general 3D rotation matrix with rotation angles angle_x, angle_y, angle_z about x, y, z axes respectively
        - angles are given in degrees
        - order of rotation is (from first to last): x, y, z
        """
        angle_x = np.deg2rad(self.angle_x)
        angle_y = np.deg2rad(self.angle_y)
        angle_z = np.deg2rad(self.angle_z)
        
        Rx = np.array([[1, 0, 0],
                    [0, cos(angle_x), -sin(angle_x)],
                    [0, sin(angle_x), cos(angle_x)]])
        Ry = np.array([[cos(angle_y), 0, sin(angle_y)],
                    [0, 1, 0],
                    [-sin(angle_y), 0, cos(angle_y)]])
        Rz = np.array([[cos(angle_z), -sin(angle_z), 0],
                    [sin(angle_z), cos(angle_z), 0],
                    [0, 0, 1]])
        self.R = np.matmul(Rz, np.matmul(Ry, Rx))
        return self.R


    def rotation_voigt(self):
        """
        rotate 6x6 matrix of elastic constants in Voigt notation
        """
        R = self.R
        M = np.array([
        [R[0,0]**2,
        R[0,1]**2,
        R[0,2]**2,
        2*R[0,1]*R[0,2],
        2*R[0,2]*R[0,0],
        2*R[0,0]*R[0,1]],
        [R[1,0]**2,
        R[1,1]**2,
        R[1,2]**2,
        2*R[1,1]*R[1,2],
        2*R[1,2]*R[1,0],
        2*R[1,0]*R[1,1]],
        [R[2,0]**2,
        R[2,1]**2,
        R[2,2]**2,
        2*R[2,1]*R[2,2],
        2*R[2,2]*R[2,0],
        2*R[2,0]*R[2,1]],
        [R[1,0]*R[2,0],
        R[1,1]*R[2,1],
        R[1,2]*R[2,2],
        R[1,1]*R[2,2]+R[1,2]*R[2,1],
        R[1,0]*R[2,2]+R[1,2]*R[2,0],
        R[1,1]*R[2,0]+R[1,0]*R[2,1]],
        [R[2,0]*R[0,0],
        R[2,1]*R[0,1],
        R[2,2]*R[0,2],
        R[0,1]*R[2,2]+R[0,2]*R[2,1],
        R[0,2]*R[2,0]+R[0,0]*R[2,2],
        R[0,0]*R[2,1]+R[0,1]*R[2,0]],
        [R[0,0]*R[1,0],
        R[0,1]*R[1,1],
        R[0,2]*R[1,2],
        R[0,1]*R[1,2]+R[0,2]*R[1,1],
        R[0,2]*R[1,0]+R[0,0]*R[1,2],
        R[0,0]*R[1,1]+R[0,1]*R[1,0]]
        ])

        self.voigt_matrix = np.matmul(M, np.matmul(self.voigt_matrix, M.T))
        return self.voigt_matrix


    # def voigt_matrix_to_tensor(self):
    #     """
    #     returns the elastic tensor from given elastic constants in pars
    #     (a dictionary of elastic constants)
    #     based on the length of pars it decides what crystal structure we the sample has
    #     """
    #     cijkl = np.zeros([3,3,3,3])

    #     cijkl[0,0,0,0] = self.voigt_matrix[0,0]
    #     cijkl[1,1,1,1] = self.voigt_matrix[1,1]
    #     cijkl[2,2,2,2] = self.voigt_matrix[2,2]
    #     cijkl[0,0,1,1] = cijkl[1,1,0,0] = self.voigt_matrix[0,1]
    #     cijkl[2,2,0,0] = cijkl[0,0,2,2] = self.voigt_matrix[0,2]
    #     cijkl[1,1,2,2] = cijkl[2,2,1,1] = self.voigt_matrix[1,2]
    #     cijkl[0,1,0,1] = cijkl[1,0,0,1] = cijkl[0,1,1,0] = cijkl[1,0,1,0] = self.voigt_matrix[5,5]
    #     cijkl[0,2,0,2] = cijkl[2,0,0,2] = cijkl[0,2,2,0] = cijkl[2,0,2,0] = self.voigt_matrix[4,4]
    #     cijkl[1,2,1,2] = cijkl[2,1,2,1] = cijkl[2,1,1,2] = cijkl[1,2,2,1] = self.voigt_matrix[3,3]

    #     self.cijkl = cijkl
    #     return cijkl


    def voigt_matrix_to_tensor(self):
        """
        returns the elastic tensor from 6x6 matrix of elastic constants in Voigt notation
        """
        cijkl = np.zeros([3,3,3,3])
        lookup = {(0,0):0, (1,1):1, (2,2):2, (1,2):3, (2,1):3, (0,2):4, (2,0):4, (0,1):5, (1,0):5}

        for i in np.arange(3):
            for j in np.arange(3):
                for k in np.arange(3):
                    for l in np.arange(3):
                        cijkl[i,j,k,l] = self.voigt_matrix[lookup[(i,j)], lookup[(k,l)]]

        self.cijkl = cijkl
        return cijkl

    
    def check_cholesky (self):
        try:
            np.linalg.cholesky(self.voigt_matrix)
            print("good cholesky!")
        except:
            print("bad cholseky!")


    # def cij_dict_to_tensor (self, angle_x=None, angle_y=None, angle_z=None, cij_dict=None):
    #     if angle_x is None:
    #         angle_x = self.angle_x
    #     if angle_y is None:
    #         angle_y = self.angle_y
    #     if angle_z is None:
    #         angle_z = self.angle_z
    #     if cij_dict is None:
    #         cij_dict = self.cij_dict

    #     voigt_matrix = self.cij_dict_to_voigt_matrix(cij_dict=cij_dict)
    #     R = self.rotation_matrix(angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
    #     voigt_matrix_R = self.rotation_voigt(R=R, voigt_matrix=voigt_matrix)
    #     cijkl = self.voigt_matrix_to_tensor(voigt_matrix=voigt_matrix_R)

    #     return cijkl


    # def cij_dict_to_voigt_dict (self, angle_x=None, angle_y=None, angle_z=None, cij_dict=None):
    #     if angle_x is None:
    #         angle_x = self.angle_x
    #     if angle_y is None:
    #         angle_y = self.angle_y
    #     if angle_z is None:
    #         angle_z = self.angle_z
    #     if cij_dict is None:
    #         cij_dict = self.cij_dict

    #     voigt_matrix = self.cij_dict_to_voigt_matrix(cij_dict=cij_dict)
    #     R = self.rotation_matrix(angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
    #     voigt_matrix_R = self.rotation_voigt(R=R, voigt_matrix=voigt_matrix)
    #     voigt_dict = self.voigt_matrix_to_voigt_dict(voigt_matrix=voigt_matrix_R)

    #     return voigt_dict





# def to_Voigt(ctens):
#     """
#     takes an elastic tensor and returns a dictionary of elastic constants in Voigt notation
#     """
#     c_Voigt = {}
#     c_Voigt['c11'] = ctens[0,0,0,0]
#     c_Voigt['c22'] = ctens[1,1,1,1]
#     c_Voigt['c33'] = ctens[2,2,2,2]
#     c_Voigt['c44'] = ctens[1,2,1,2]
#     c_Voigt['c55'] = ctens[0,2,0,2]
#     c_Voigt['c66'] = ctens[0,1,0,1]
#     c_Voigt['c12'] = ctens[0,0,1,1]
#     c_Voigt['c13'] = ctens[0,0,2,2]
#     c_Voigt['c14'] = ctens[0,0,1,2]
#     c_Voigt['c15'] = ctens[0,0,0,2]
#     c_Voigt['c16'] = ctens[0,0,0,1]
#     c_Voigt['c23'] = ctens[1,1,2,2]
#     c_Voigt['c24'] = ctens[1,1,1,2]
#     c_Voigt['c25'] = ctens[1,1,0,2]
#     c_Voigt['c26'] = ctens[1,1,0,1]
#     c_Voigt['c34'] = ctens[2,2,1,2]
#     c_Voigt['c35'] = ctens[2,2,0,2]
#     c_Voigt['c36'] = ctens[2,2,0,1]
#     c_Voigt['c45'] = ctens[1,2,0,2]
#     c_Voigt['c46'] = ctens[1,2,0,1]
#     c_Voigt['c56'] = ctens[0,2,0,1]
#     return (c_Voigt)


# def rotate_ctens (angle_x, angle_y, angle_z, ctens):
#     """
#     takes angles angle_x, angle_y, angle_z and an elastic tensor and returns the rotated elastic tensor
#     """
#     crot =  np.zeros([3,3,3,3])
#     R = rotatation_matrix(angle_x, angle_y, angle_z)
#     for i in np.arange(3):
#         for j in np.arange(3):
#             for k in np.arange(3):
#                 for l in np.arange(3):
#                     ctemp = 0
#                     for a in np.arange(3):
#                         for b in np.arange(3):
#                             for c in np.arange(3):
#                                 for d in np.arange(3):
#                                     ctemp += R[i,a]*R[j,b]*R[k,c]*R[l,d]*ctens[a,b,c,d]
#                     crot[i,j,k,l] = ctemp
#     return crot


if __name__=="__main__":
    cij_dict = {'c11':110, 'c33':30, 'c13':40, 'c12':40, 'c44':30}
    e = ElasticConstants(cij_dict, symmetry="hexagonal")
    print(e.voigt_matrix)
    e.check_cholesky()



