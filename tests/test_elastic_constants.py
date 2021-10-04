import numpy as np
from rus_comsol.rus_rpr import RUSRPR
from rus_comsol.elastic_constants import ElasticConstants



#%%
# Test cubic
print ('---------------------------------------------------------------------')
print ('test cubic crystal symmetry')
cij = {
    'c11':150,
    'c12':100,
    'c44':50
    }
cubic = ElasticConstants(cij_dict=cij, symmetry='cubic')

difference = cubic.voigt_matrix_to_tensor() - cubic.voigt_matrix_to_tensor_general()
counter = 0
for i in np.arange(3):
            for j in np.arange(3):
                for k in np.arange(3):
                    for l in np.arange(3):
                        if difference[i,j,k,l] != 0:
                            print (i,j,k,l, '    difference: ', difference[i,j,k,l])
                            counter+=1

if counter == 0:
    print ('Yay, no difference found between the two functions!')
else:
    print ('there are some discrepancies found')
print ('---------------------------------------------------------------------')



#%%
# Test tetragonal
print ('---------------------------------------------------------------------')
print ('test tetragonal crystal symmetry')
cij = {
    'c11':150,
    'c12':100,
    'c44':50,
    'c33':90,
    'c13':70,
    'c66':20
    }
cubic = ElasticConstants(cij_dict=cij, symmetry='tetragonal')

difference = cubic.voigt_matrix_to_tensor() - cubic.voigt_matrix_to_tensor_general()
counter = 0
for i in np.arange(3):
            for j in np.arange(3):
                for k in np.arange(3):
                    for l in np.arange(3):
                        if difference[i,j,k,l] != 0:
                            print (i,j,k,l, '    difference: ', difference[i,j,k,l])
                            counter+=1

if counter == 0:
    print ('Yay, no difference found between the two functions!')
else:
    print ('there are some discrepancies found')
print ('---------------------------------------------------------------------')



#%%
# Test hexagonal
print ('---------------------------------------------------------------------')
print ('test hexagonal crystal symmetry')
cij = {
    'c11':150,
    # 'c66':20,
    'c12':100,
    'c44':50,
    'c33':90,
    'c13':70
    }
cubic = ElasticConstants(cij_dict=cij, symmetry='hexagonal')

difference = cubic.voigt_matrix_to_tensor() - cubic.voigt_matrix_to_tensor_general()
counter = 0
for i in np.arange(3):
            for j in np.arange(3):
                for k in np.arange(3):
                    for l in np.arange(3):
                        if difference[i,j,k,l] != 0:
                            print (i,j,k,l, '    difference: ', difference[i,j,k,l])
                            counter+=1

if counter == 0:
    print ('Yay, no difference found between the two functions!')
else:
    print ('there are some discrepancies found')
print ('---------------------------------------------------------------------')



#%%
# Test hexagonal
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
cubic = ElasticConstants(cij_dict=cij, symmetry='rhombohedral')

difference = cubic.voigt_matrix_to_tensor() - cubic.voigt_matrix_to_tensor_general()
counter = 0
for i in np.arange(3):
            for j in np.arange(3):
                for k in np.arange(3):
                    for l in np.arange(3):
                        if difference[i,j,k,l] != 0:
                            print (i,j,k,l, '    difference: ', difference[i,j,k,l])
                            counter+=1

if counter == 0:
    print ('Yay, no difference found between the two functions!')
else:
    print ('there were ', counter, ' discrepancies found')
print ('---------------------------------------------------------------------')