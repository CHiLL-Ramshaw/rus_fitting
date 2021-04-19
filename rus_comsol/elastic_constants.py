import numpy as np
from numpy import cos, sin
from copy import deepcopy

class ElasticConstants:
    def __init__(self, cij_dict, symmetry,
                angle_x=0, angle_y=0, angle_z=0):
        """
        - The elastic constants in elasti_dict must be in GPa
        - symmetry can be "cubic", "tetragonal", "orthorhombic"
        - angles should be in degrees, angle_x corresponds to rotation around x
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
        self.voigt_dict   = self.voigt_matrix_to_voigt_dict()


    ## Special Method >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def __setitem__(self, key, value):
        ## Add security not to add keys later
        if key not in self._cij_dict.keys():
            print(key + " was not added (new band parameters are only allowed within object initialization)")
        else:
            self._cij_dict[key] = value
            self.cij_dict_to_voigt_matrix()
            self.rotation_voigt()
            self.voigt_matrix_to_voigt_dict()

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
        self.cij_dict_to_voigt_matrix()
        self.rotation_voigt()
        self.voigt_matrix_to_voigt_dict()
    cij_dict = property(_get_cij_dict, _set_cij_dict)

    def _get_angle_x(self):
        return self._angle_x
    def _set_angle_x(self, angle_x):
        self._angle_x = angle_x
        self.rotation_matrix()
        self.cij_dict_to_voigt_matrix()
        self.rotation_voigt()
        self.voigt_matrix_to_voigt_dict()
    angle_x = property(_get_angle_x, _set_angle_x)

    def _get_angle_y(self):
        return self._angle_y
    def _set_angle_y(self, angle_y):
        self._angle_y = angle_y
        self.rotation_matrix()
        self.cij_dict_to_voigt_matrix()
        self.rotation_voigt()
        self.voigt_matrix_to_voigt_dict()
    angle_y = property(_get_angle_y, _set_angle_y)

    def _get_angle_z(self):
        return self._angle_z
    def _set_angle_z(self, angle_z):
        self._angle_z = angle_z
        self.rotation_matrix()
        self.cij_dict_to_voigt_matrix()
        self.rotation_voigt()
        self.voigt_matrix_to_voigt_dict()
    angle_z = property(_get_angle_z, _set_angle_z)


    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def cij_dict_to_voigt_matrix(self):
        """
        returns the elastic tensor from given elastic constants in pars
        (a dictionary of elastic constants)
        based on the length of pars it decides what crystal structure we the sample has
        """
        cij_dict = deepcopy(self.cij_dict)
        voigt_matrix = np.zeros((6,6))

        if self.symmetry=="cubic":
            voigt_matrix[0,0] = voigt_matrix[1,1] = voigt_matrix[2,2] = cij_dict['c11']
            voigt_matrix[0,1] = voigt_matrix[0,2] = voigt_matrix[1,2] = cij_dict['c12']
            voigt_matrix[3,3] = voigt_matrix[4,4] = voigt_matrix[5,5] = cij_dict['c44']

        elif self.symmetry=="tetragonal":
            voigt_matrix[0,0] = voigt_matrix[1,1] = cij_dict['c11']
            voigt_matrix[2,2]                     = cij_dict['c33']
            voigt_matrix[0,1]                     = cij_dict['c12']
            voigt_matrix[0,2] = voigt_matrix[1,2] = cij_dict['c13']
            voigt_matrix[3,3] = voigt_matrix[4,4] = cij_dict['c44']
            voigt_matrix[5,5]                     = cij_dict['c66']

        elif self.symmetry=="orthorhombic":
            voigt_matrix[0,0] = cij_dict['c11']
            voigt_matrix[1,1] = cij_dict['c22']
            voigt_matrix[2,2] = cij_dict['c33']
            voigt_matrix[0,1] = cij_dict['c12']
            voigt_matrix[0,2] = cij_dict['c13']
            voigt_matrix[1,2] = cij_dict['c23']
            voigt_matrix[3,3] = cij_dict['c44']
            voigt_matrix[4,4] = cij_dict['c55']
            voigt_matrix[5,5] = cij_dict['c66']

        # elif len(cij_dict) == 5:                    # hexagonal
        #     indicator = np.any( np.array([i=='c11' for i in cij_dict]) )
        #     if indicator == True:
        #         c11 = c22       = cij_dict['c11']
        #         c33             = cij_dict['c33']
        #         c12             = cij_dict['c12']
        #         c13 = c23       = cij_dict['c13']
        #         c44 = c55       = cij_dict['c44']
        #         c66             = (cij_dict['c11']-cij_dict['c12'])/2
        #     else:
        #         c11 = c22       = 2*cij_dict['c66'] + cij_dict['c12']
        #         c33             = cij_dict['c33']
        #         c12             = cij_dict['c12']
        #         c13 = c23       = cij_dict['c13']
        #         c44 = c55       = cij_dict['c44']
        #         c66             = cij_dict['c66']

        self.voigt_matrix = (voigt_matrix + voigt_matrix.T
                             - np.diag(voigt_matrix.diagonal()))
        return self.voigt_matrix


    def voigt_matrix_to_voigt_dict(self):
        self.voigt_dict = {}
        voigt_matrix = np.triu(self.voigt_matrix) # get only upper part of the matrix
        for i in range(6):
            for j in range(i, 6):
                self.voigt_dict["c"+str(i+1)+str(j+1)] = voigt_matrix[i,j]
        return self.voigt_dict


    def rotation_matrix(self):
        """
        define general 3D rotation matrix with rotation angles angle_x, angle_y, angle_z about x, y, z
        axes respectively;
        angles are given in degrees
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




# def elastic_tensor(pars):
#     """
#     returns the elastic tensor from given elastic constants in pars
#     (a dictionary of elastic constants)
#     based on the length of pars it decides what crystal structure we the sample has
#     """
#     ctens = np.zeros([3,3,3,3])

#     if len(pars) == 3:                      # cubic
#         c11 = c22 = c33 = pars['c11']
#         c12 = c13 = c23 = pars['c12']
#         c44 = c55 = c66 = pars['c44']

#     elif len(pars) == 5:                    # hexagonal
#         indicator = np.any( np.array([i=='c11' for i in pars]) )
#         if indicator == True:
#             c11 = c22       = pars['c11']
#             c33             = pars['c33']
#             c12             = pars['c12']
#             c13 = c23       = pars['c13']
#             c44 = c55       = pars['c44']
#             c66             = (pars['c11']-pars['c12'])/2
#         else:
#             c11 = c22       = 2*pars['c66'] + pars['c12']
#             c33             = pars['c33']
#             c12             = pars['c12']
#             c13 = c23       = pars['c13']
#             c44 = c55       = pars['c44']
#             c66             = pars['c66']

#     elif len(pars) == 6:                    # tetragonal
#         c11 = c22       = pars['c11']
#         c33             = pars['c33']
#         c12             = pars['c12']
#         c13 = c23       = pars['c13']
#         c44 = c55       = pars['c44']
#         c66             = pars['c66']

#     elif len(pars) == 9:                    # orthorhombic
#         c11             = pars['c11']
#         c22             = pars['c22']
#         c33             = pars['c33']
#         c12             = pars['c12']
#         c13             = pars['c13']
#         c23             = pars['c23']
#         c44             = pars['c44']
#         c55             = pars['c55']
#         c66             = pars['c66']

#     else:
#         print ('You have not given a valid Crystal Structure')

#     ctens[0,0,0,0] = c11
#     ctens[1,1,1,1] = c22
#     ctens[2,2,2,2] = c33
#     ctens[0,0,1,1] = ctens[1,1,0,0] = c12
#     ctens[2,2,0,0] = ctens[0,0,2,2] = c13
#     ctens[1,1,2,2] = ctens[2,2,1,1] = c23
#     ctens[0,1,0,1] = ctens[1,0,0,1] = ctens[0,1,1,0] = ctens[1,0,1,0] = c66
#     ctens[0,2,0,2] = ctens[2,0,0,2] = ctens[0,2,2,0] = ctens[2,0,2,0] = c55
#     ctens[1,2,1,2] = ctens[2,1,2,1] = ctens[2,1,1,2] = ctens[1,2,2,1] = c44

#     return ctens


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
    cij_dict = {'c11':110, 'c33':90, 'c13':70, 'c12':50, 'c44':30, 'c66':10}
    e1 = ElasticConstants(cij_dict, symmetry="tetragonal")
    print(e1.voigt_matrix)
    for key in sorted(e1.voigt_dict.keys()):
        print(key, e1.voigt_dict[key])
    e1.angle_x = 5
    e1.angle_z = 15
    print(np.round(e1.voigt_matrix, 0))
    for key in sorted(e1.voigt_dict.keys()):
        print(key, e1.voigt_dict[key])



