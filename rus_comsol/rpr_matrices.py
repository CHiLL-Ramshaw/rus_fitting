import sys
import pyvista as pv
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan2, pi
from scipy.spatial.transform import Rotation as R
from time import time
import quadpy
from copy import deepcopy
import mph
import os



class RPRMatrices:

    def __init__(self, order, dimensions,
                 Emat_path, Itens_path):

        self.order              = order
        self.dimensions         = np.array(dimensions)
        self.basis, self.lookUp = self.makeBasis(self.order)

        self.Emat_path  = Emat_path
        self.Itens_path = Itens_path

        self.Emat    = None
        self.Etens   = None
        self.Itens   = None

    def makeBasis(self, N):
        '''
        constructs a basis of polynomials of degree <= N
        returns the basis and a lookup table
        '''
        NT = int((N+1)*(N+2)*(N+3)/6)
        basis = np.zeros([NT, 3], dtype= int)
        idx = 0
        lookUp = {(1, 1, 1) : 0,
                   (1, 1, -1) : 1,
                    (1, -1, 1) : 2,
                     (-1, 1, 1) : 3,
                      (1, -1, -1): 4,
                       (-1, 1, -1) : 5,
                        (-1, -1, 1) : 6,
                         (-1, -1, -1) : 7}
        for k in range(N+1):
            for l in range(N+1):
                for m in range(N+1):
                    if k+l+m <= N:
                        basis[idx] = np.array([k,l,m])
                        idx += 1       
        return basis, lookUp

    
    def E_int(self, i, j):
        """
        calculates integral for kinetic energy matrix, i.e. the product of two basis functions
        """
        ps = self.basis[i] + self.basis[j] + 1.
        if np.any(ps%2==0): return 0.
        return 8*np.prod((self.dimensions/2)**ps / ps)


    def G_int(self, i, j, k, l):
        """
        calculates the integral for potential energy matrix, i.e. the product of the derivatives of two basis functions
        """
        M = np.array([[[2.,0.,0.],[1.,1.,0.],[1.,0.,1.]],[[1.,1.,0.],[0.,2.,0.],[0.,1.,1.]],[[1.,0.,1.],[0.,1.,1.],[0.,0.,2.]]])
        if not self.basis[i][k]*self.basis[j][l]: return 0
        ps = self.basis[i] + self.basis[j] + 1. - M[k,l]
        if np.any(ps%2==0): return 0.
        return 8*self.basis[i][k]*self.basis[j][l]*np.prod((self.dimensions/2)**ps / ps)


    def E_mat(self):
        """
        put the integrals from E_int in a matrix
        Emat is the kinetic energy matrix from Arkady's paper
        """
        Etens = np.zeros((3,self.idx,3,self.idx), dtype= np.double)
        for x in range(3*self.idx):
            i, k = int(x/self.idx), x%self.idx
            for y in range(x, 3*self.idx):
                j, l = int(y/self.idx), y%self.idx
                if i==j: Etens[i,k,j,l]=Etens[j,l,i,k]=self.E_int(k,l)
        Emat = Etens.reshape(3*self.idx,3*self.idx)
        return Emat


    def I_tens(self):
        """
        put the integrals from G_int in a tensor;
        it is the tensor in the potential energy matrix in Arkady's paper;
        i.e. it is the the potential energy matrix without the elastic tensor;
        """
        Itens = np.zeros((3,self.idx,3,self.idx), dtype= np.double)
        for x in range(3*self.idx):
            i, k = int(x/self.idx), x%self.idx
            for y in range(x, 3*self.idx):
                j, l = int(y/self.idx), y%self.idx
                Itens[i,k,j,l]=Itens[j,l,i,k]=self.G_int(k,l,i,j)
        return Itens


    def create_G_E_matrices (self):
        idx = int((self.order+1)*(self.order+2)*(self.order+3)/6)
        self.Etens, self.Itens = np.zeros((3,idx,3,idx), dtype= np.float64), np.zeros((3,idx,3,idx), dtype= np.float64)  
    
        for x in range(3*idx):
            i, k = int(x/idx), x%idx
            for y in range(x, 3*idx):
                j, l = int(y/idx), y%idx
                if i==j: self.Etens[i,k,j,l]=self.Etens[j,l,i,k]=self.E_int(k,l)
                self.Itens[i,k,j,l]=self.Itens[j,l,i,k]=self.G_int(k,l,i,j)
    
        self.Emat = self.Etens.reshape(3*idx,3*idx)

        try:
            np.linalg.cholesky(self.Emat)
            print("good cholesky")
        except Exception as e:
            print("no cholesky")
            print(e)

        np.save(self.Emat_path, self.Emat)
        np.save(self.Itens_path, self.Itens)

        return self.Emat, self.Itens

if __name__ == '__main__':
    root_path    = sys.path[0]
    N            = 16
    E_path       = f'Mn3Ge_2105A_RPR_mesh5_Etens_basis_{N}.npy'
    I_path       = f'Mn3Ge_2105A_RPR_mesh5_Itens_basis_{N}.npy'
    dimensions   = np.array([0.869e-3, 1.01e-3,  1.193e-3])
    mass         = 7.684976435144674e-6
    density      = mass/np.prod(dimensions) # kg / m^3

    stokes = RPRMatrices(order=N, dimensions=dimensions,
                         Emat_path = E_path,
                         Itens_path = I_path)
    stokes.create_G_E_matrices()
    