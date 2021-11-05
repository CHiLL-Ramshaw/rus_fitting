import numpy as np
from rus_comsol.rus_rpr import RUSRPR
from rus_comsol.elastic_constants import ElasticConstants


print ('---------------------------------------------------------------------')
print ('test rhoomohedral crystal symmetry')
cij = {
    'c11':150,
    # 'c66':20,
    'c12':100,
    'c44':50,
    'c33':90,
    'c13':70,
    'c14':200
    }
rhombohedral = ElasticConstants(cij_dict=cij, symmetry='rhombohedral')
print(rhombohedral.voigt_matrix)

rhombohedral_rot = ElasticConstants(cij_dict=cij, symmetry='rhombohedral', angle_z=90)
print (np.round(rhombohedral_rot.voigt_matrix, 1))

print ('---------------------------------------------------------------------')