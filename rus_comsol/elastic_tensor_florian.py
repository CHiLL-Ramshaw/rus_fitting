import numpy as np
from numpy import cos, sin

def elastic_tensor(pars):
    """
    returns the elastic tensor from given elastic constants in pars
    (a dictionary of elastic constants)
    based on the length of pars it decides what crystal structure we the sample has
    """
    ctens = np.zeros([3,3,3,3])

    if len(pars) == 3:                      # cubic
        c11 = c22 = c33 = pars['c11']
        c12 = c13 = c23 = pars['c12']
        c44 = c55 = c66 = pars['c44']

    elif len(pars) == 5:                    # hexagonal
        indicator = np.any( np.array([i=='c11' for i in pars]) )
        if indicator == True:
            c11 = c22       = pars['c11']
            c33             = pars['c33']
            c12             = pars['c12']
            c13 = c23       = pars['c13']
            c44 = c55       = pars['c44']
            c66             = (pars['c11']-pars['c12'])/2
        else:
            c11 = c22       = 2*pars['c66'] + pars['c12']
            c33             = pars['c33']
            c12             = pars['c12']
            c13 = c23       = pars['c13']
            c44 = c55       = pars['c44']
            c66             = pars['c66']

    elif len(pars) == 6:                    # tetragonal
        c11 = c22       = pars['c11']
        c33             = pars['c33']
        c12             = pars['c12']
        c13 = c23       = pars['c13']
        c44 = c55       = pars['c44']
        c66             = pars['c66']

    elif len(pars) == 9:                    # orthorhombic
        c11             = pars['c11']
        c22             = pars['c22']
        c33             = pars['c33']
        c12             = pars['c12']
        c13             = pars['c13']
        c23             = pars['c23']
        c44             = pars['c44']
        c55             = pars['c55']
        c66             = pars['c66']

    else:
        print ('You have not given a valid Crystal Structure')

    ctens[0,0,0,0] = c11
    ctens[1,1,1,1] = c22
    ctens[2,2,2,2] = c33
    ctens[0,0,1,1] = ctens[1,1,0,0] = c12
    ctens[2,2,0,0] = ctens[0,0,2,2] = c13
    ctens[1,1,2,2] = ctens[2,2,1,1] = c23
    ctens[0,1,0,1] = ctens[1,0,0,1] = ctens[0,1,1,0] = ctens[1,0,1,0] = c66
    ctens[0,2,0,2] = ctens[2,0,0,2] = ctens[0,2,2,0] = ctens[2,0,2,0] = c55
    ctens[1,2,1,2] = ctens[2,1,2,1] = ctens[2,1,1,2] = ctens[1,2,2,1] = c44

    return ctens


def to_Voigt(ctens):
    """
    takes an elastic tensor and returns a dictionary of elastic constants in Voigt notation
    """
    c_Voigt = {}
    c_Voigt['c11'] = ctens[0,0,0,0]
    c_Voigt['c22'] = ctens[1,1,1,1]
    c_Voigt['c33'] = ctens[2,2,2,2]
    c_Voigt['c44'] = ctens[1,2,1,2]
    c_Voigt['c55'] = ctens[0,2,0,2]
    c_Voigt['c66'] = ctens[0,1,0,1]
    c_Voigt['c12'] = ctens[0,0,1,1]
    c_Voigt['c13'] = ctens[0,0,2,2]
    c_Voigt['c14'] = ctens[0,0,1,2]
    c_Voigt['c15'] = ctens[0,0,0,2]
    c_Voigt['c16'] = ctens[0,0,0,1]
    c_Voigt['c23'] = ctens[1,1,2,2]
    c_Voigt['c24'] = ctens[1,1,1,2]
    c_Voigt['c25'] = ctens[1,1,0,2]
    c_Voigt['c26'] = ctens[1,1,0,1]
    c_Voigt['c34'] = ctens[2,2,1,2]
    c_Voigt['c35'] = ctens[2,2,0,2]
    c_Voigt['c36'] = ctens[2,2,0,1]
    c_Voigt['c45'] = ctens[1,2,0,2]
    c_Voigt['c46'] = ctens[1,2,0,1]
    c_Voigt['c56'] = ctens[0,2,0,1]
    return (c_Voigt)


def rotatation_matrix(angle_x, angle_y, angle_z):
    """
    define general 3D rotation matrix with rotation angles angle_x, angle_y, angle_z about x, y, z
    axes respectively;
    angles are given in degrees
    """
    angle_x = np.deg2rad(angle_x)
    angle_y = np.deg2rad(angle_y)
    angle_z = np.deg2rad(angle_z)
    Rx = np.array([[1, 0, 0],
                   [0, cos(angle_x), -sin(angle_x)],
                   [0, sin(angle_x), cos(angle_x)]])
    Ry = np.array([[cos(angle_y), 0, sin(angle_y)],
                   [0, 1, 0],
                   [-sin(angle_y), 0, cos(angle_y)]])
    Rz = np.array([[cos(angle_z), -sin(angle_z), 0],
                   [sin(angle_z), cos(angle_z), 0],
                   [0, 0, 1]])
    return np.matmul(Rz, np.matmul(Ry, Rx))



def voigt_dict_to_voigt_matrix(voigt_dict, symmetry="tetragonal"):
    """
    returns the elastic tensor from given elastic constants in pars
    (a dictionary of elastic constants)
    based on the length of pars it decides what crystal structure we the sample has
    """
    voigt_matrix = np.zeros((6,6))

    if symmetry=="cubic":
        voigt_matrix[0,0] = voigt_matrix[1,1] = voigt_matrix[2,2] = voigt_dict['c11']
        voigt_matrix[0,1] = voigt_matrix[0,2] = voigt_matrix[1,2] = voigt_dict['c12']
        voigt_matrix[3,3] = voigt_matrix[4,4] = voigt_matrix[5,5] = voigt_dict['c44']

    # elif len(voigt_dict) == 5:                    # hexagonal
    #     indicator = np.any( np.array([i=='c11' for i in voigt_dict]) )
    #     if indicator == True:
    #         c11 = c22       = voigt_dict['c11']
    #         c33             = voigt_dict['c33']
    #         c12             = voigt_dict['c12']
    #         c13 = c23       = voigt_dict['c13']
    #         c44 = c55       = voigt_dict['c44']
    #         c66             = (voigt_dict['c11']-voigt_dict['c12'])/2
    #     else:
    #         c11 = c22       = 2*voigt_dict['c66'] + voigt_dict['c12']
    #         c33             = voigt_dict['c33']
    #         c12             = voigt_dict['c12']
    #         c13 = c23       = voigt_dict['c13']
    #         c44 = c55       = voigt_dict['c44']
    #         c66             = voigt_dict['c66']

    elif symmetry=="tetragonal":                    # tetragonal
        voigt_matrix[0,0] = voigt_matrix[1,1] = voigt_dict['c11']
        voigt_matrix[2,2]                     = voigt_dict['c33']
        voigt_matrix[0,1]                     = voigt_dict['c12']
        voigt_matrix[0,2] = voigt_matrix[1,2] = voigt_dict['c13']
        voigt_matrix[3,3] = voigt_matrix[4,4] = voigt_dict['c44']
        voigt_matrix[5,5]                     = voigt_dict['c66']

    # elif len(voigt_dict) == 9:                    # orthorhombic
    #     c11             = voigt_dict['c11']
    #     c22             = voigt_dict['c22']
    #     c33             = voigt_dict['c33']
    #     c12             = voigt_dict['c12']
    #     c13             = voigt_dict['c13']
    #     c23             = voigt_dict['c23']
    #     c44             = voigt_dict['c44']
    #     c55             = voigt_dict['c55']
    #     c66             = voigt_dict['c66']


    return voigt_matrix + voigt_matrix.T - np.diag(voigt_matrix.diagonal())


def Bond_method(angle_x, angle_y, angle_z, voigt_matrix):
    R = rotatation_matrix(angle_x, angle_y, angle_z)
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

    return np.matmul(M, np.matmul(voigt_matrix, M.T))


def voigt_matrix_to_voigt_dict(voigt_matrix):
    voigt_dict = {}
    voigt_matrix = np.triu(voigt_matrix) # get only upper part of the matrix
    for i in range(6):
        for j in range(i, 6):
            voigt_dict["c"+str(i+1)+str(j+1)] = voigt_matrix[i,j]
    return voigt_dict


def rotate_ctens (angle_x, angle_y, angle_z, ctens):
    """
    takes angles angle_x, angle_y, angle_z and an elastic tensor and returns the rotated elastic tensor
    """
    crot =  np.zeros([3,3,3,3])
    R = rotatation_matrix(angle_x, angle_y, angle_z)
    for i in np.arange(3):
        for j in np.arange(3):
            for k in np.arange(3):
                for l in np.arange(3):
                    ctemp = 0
                    for a in np.arange(3):
                        for b in np.arange(3):
                            for c in np.arange(3):
                                for d in np.arange(3):
                                    ctemp += R[i,a]*R[j,b]*R[k,c]*R[l,d]*ctens[a,b,c,d]
                    crot[i,j,k,l] = ctemp
    return crot


if __name__=="__main__":
    voigt_dict = {'c11':115, 'c33':90, 'c13':70, 'c12':50, 'c44':30, 'c66':10}
    angle_x = 5
    angle_y = 0
    angle_z = 15

    voigt_matrix = voigt_dict_to_voigt_matrix(voigt_dict=voigt_dict,
                                              symmetry="tetragonal")
    # print(voigt_matrix)
    voigt_matrix_to_voigt_dict(voigt_matrix)
    voigt_matrix_rot = Bond_method(angle_x, angle_y, angle_z, voigt_matrix)
    print(np.round(voigt_matrix_rot, 5))
    crot = to_Voigt(rotate_ctens(angle_x, angle_y, angle_z, elastic_tensor(voigt_dict)))
    for key in sorted(crot.keys()):
        print(key, crot[key])


