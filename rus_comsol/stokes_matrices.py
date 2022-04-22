import sys
import pyvista as pv
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan2, pi
from scipy.spatial.transform import Rotation as R
from time import time
import quadpy
from copy import deepcopy
import multiprocessing as mp
from itertools import repeat
import mph
import os



class StokesMatrices:

    def __init__(self, order,
                 surface_file_path, surface_file_type, 
                 integral_path, Emat_path, Itens_path,
                 parallel=True, nb_processes=1, scale='m', move_com=True):

        self.order   = order
        scale_lookup = {'m': 1, 'cm':1e-2, 'mm':1e-3}
        self.scale   = scale_lookup[scale]
        self.move_com = move_com
        self.basis, self.lookUp     = self.makeBasis(self.order)
        self.basisx2, self.lookUpx2 = self.makeBasis(self.order*2)

        self.parallel = parallel
        self.nb_processes = nb_processes
        nb_cores = os.cpu_count()
        if self.nb_processes > nb_cores:
            self.nb_processes = nb_cores
            print ('the entered number of processes exceeds the number of cores;')
            print ('the number of processes has been set to the maximum of ', self.nb_processes)
        else:
            print (self.nb_processes, ' out of ', nb_cores, ' processes used', )

        self.integral_path = integral_path
        self.Emat_path     = Emat_path
        self.Itens_path    = Itens_path

        self.surface_path = surface_file_path
        self.surface_type = surface_file_type
        if self.surface_type not in ['stl', 'STL', 'Stl', 'comsol', 'Comsol', 'COMSOL']:
            print('your surface file type must be either "comsol" or "stl"')
            sys.exit()

        self.intVals = None
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
        lookUp = {}
        idx = 0
        for i in range(N+1):
            for j in range(N+1):
                for k in range(N+1):
                    if i+j+k<=N:
                        basis[idx] = np.array([i,j,k])
                        lookUp[(i,j,k)] = idx
                        idx += 1
        return basis, lookUp

    def intBasisFunc(self, exp, mesh_pts, mesh_faces, scheme):
        '''
        give a set of exponents (N,M,L) and a mesh points/faces
        integrates x**N * y**M * z**L over the interior of the mesh
        using the given in quadpy scheme
        '''
        vol = 0.
        numz= 0
        N, M, L = exp[0], exp[1], exp[2]+1
        for idx, f in enumerate(mesh_faces):
            t = [mesh_pts[f[0]], mesh_pts[f[1]], mesh_pts[f[2]]]
            triangle = np.array([[t[0][0], t[0][1]], [t[1][0], t[1][1]], [t[2][0],t[2][1]]])
            tmp = scheme.integrate(lambda x: self.intInPlace(x, [N,M,L], t), triangle)
            if tmp==0: numz += 1
            vol += tmp
        if (N, M, L) == (0,0,1):
            print("STOKES VOLUME: ", vol/L)
        return vol/L

    def intInPlace(self, X, exp, t):
        n, m, l = exp
        p0, p1, p2 = t
        p1 = p1 - p0
        p2 = p2 - p0
        n0 = np.cross(p1, p2)

        if n0[2] == 0.:
            return np.tile(0., (X.shape[-1], 1)).T
        else:
            c0 = np.dot(p0, n0)/n0[2]
            c0 = np.tile(c0, (X.shape[-1], 1)).T
            cx = -n0[0]/n0[2]
            cy = -n0[1]/n0[2]
        res = np.sign(n0[2])*(X[0]**n * X[1]**m * (c0 + cx*X[0] + cy*X[1])**l)

        return res

    def intVecBasisFunc(self, exp, mesh, triangles):
        '''
        give a set of exponents (N,M,L) and a mesh points/faces
        integrates x**N * y**M * z**L over the interior of the mesh
        vecotrized to do the integrals over all the triangles
        '''
        scheme = quadpy.t2.get_good_scheme(40)
        N, M, L = exp[0], exp[1], exp[2]+1
        vol = scheme.integrate(lambda x: self.intVecInPlace(x, [N,M,L], mesh), triangles)
        vol = np.sum(vol)
        if (N, M, L) == (0,0,1):
            print("STOKES VOLUME: ", vol/L)
        return vol/L

    def intVecInPlace(self, X, exp, t):
        '''
        vectorized quadrature integration of a polynomial
        over an array of triangles
        '''
        n, m, l = exp
        p0, p1, p2 = t

        P1 = p1 - p0
        P2 = p2 - p0
        n0 = np.cross(P1, P2)

        a = np.sum(p0*n0, axis= 1)

        n0 = n0.T
        c0 = np.divide(a, n0[2], out= np.zeros_like(a), where=n0[2]!=0.)
        c0 = np.tile(c0, (X.shape[-1], 1)).T
        cx = -np.divide(n0[0], n0[2], out= np.zeros_like(n0[0]), where=n0[2]!=0.)
        cx = np.tile(cx, (X.shape[-1], 1)).T
        cy = -np.divide(n0[1], n0[2], out= np.zeros_like(n0[1]), where=n0[2]!=0.)
        cy = np.tile(cy, (X.shape[-1], 1)).T

        s = np.tile(np.sign(n0[2]), (X.shape[-1], 1)).T
        res = s*(X[0]**n * X[1]**m * (c0 + cx*X[0] + cy*X[1])**l)

        return res

    def getMeshFromComsol(self, save_path=None, component= 'comp1', mesh= 'mesh1'):
        '''
        extracts the vertices and faces of a surface mesh
        from comsol file model_path

        surface mesh is assumed to be free triangular over the
        entire boundary of a closed 3d shape

        saves the vertices, faces as npz file save_path
        for use in xyz-matrix building, and returns the points, faces
        for immediate use
        '''
        client = mph.Client()
        model = client.load(self.surface_path)
        vtx = model.java.component(component).mesh(mesh).getVertex()
        elem = model.java.component(component).mesh(mesh).getVertex('tri')

        vtx = np.array(vtx)
        vtx = vtx.T
        #reformat faces for use with pyvista
        elem = np.array(elem)
        elem = np.stack((np.full((1, elem.shape[1]), 3), elem))
        faces = np.hstack(elem.T)

        polydata = {'faces' : faces, 'points' : vtx}
        if save_path is None:
            return polydata
        else:
            np.savez(save_path, **polydata)
            return polydata

    def intNpzMesh(self, polydata, load_polydata=False):
        if load_polydata:
            npz_stl = np.load(polydata)
        else:
            npz_stl = polydata
        stl = pv.PolyData(npz_stl['points'], npz_stl['faces'])
        stl = stl.scale(self.scale)


        points, faces = stl.points, stl.faces.reshape(-1, 4)[:, 1:]
        if self.move_com:
            offset = np.tile(stl.center_of_mass(), (points.shape[0], 1))
            points -= offset

        MESH = np.zeros([faces.shape[0], 3, 3])
        TRIANGLES = np.zeros([faces.shape[0], 3, 2])
        for idx, f in enumerate(faces):
            t = np.array([points[f[0]], points[f[1]], points[f[2]]])
            MESH[idx] = t
            TRIANGLES[idx] = np.array([[t[0][0], t[0][1]], [t[1][0], t[1][1]], [t[2][0],t[2][1]]])
        MESH = np.stack(MESH, axis= -2)
        MESH0 = deepcopy(MESH)
        TRIANGLES = np.stack(TRIANGLES, axis= -2)

        print('STL VOLUME: ', stl.volume)
        print('STL COM: ', stl.center_of_mass())
        print("NUM POINTS: ", points.shape[0])
        #stl.plot()

        # intBasis, basisLookup = makeBasis(2*basis_N)
        print("NUM BASIS FUNCTIONS: ", len(self.basisx2))
        if self.parallel:
            pool = mp.Pool(processes = self.nb_processes)
            t0 = time()
            output = pool.starmap(self.intVecBasisFunc, zip(self.basisx2, repeat(MESH), repeat(TRIANGLES)))
            self.intVals = np.array(output)

        else:
            self.intVals = np.zeros(len(self.basisx2))
            for i, b in enumerate(self.basisx2):
                tmp = self.intVecBasisFunc(b, MESH, TRIANGLES)
                self.intVals[i] = tmp
                if i%10==0:
                    print("{}/{}".format(i, len(self.basisx2)))

        print("Static Mesh: ", np.allclose(MESH, MESH0))
        print("INTEGRALS: ", time() - t0)

        np.save(self.integral_path, self.intVals)


    def intSTL(self):
        scheme = quadpy.t2.get_good_scheme(40)
        stl = pv.PolyData(self.surface_path)
        stl = stl.scale(self.scale)

        points, faces = stl.points, stl.faces.reshape(-1, 4)[:, 1:]
        if self.move_com:
            offset = np.tile(stl.center_of_mass(), (points.shape[0], 1))
            points -= offset

        MESH = np.zeros([faces.shape[0], 3, 3])
        TRIANGLES = np.zeros([faces.shape[0], 3, 2])
        for idx, f in enumerate(faces):
            t = np.array([points[f[0]], points[f[1]], points[f[2]]])

            MESH[idx] = t
            TRIANGLES[idx] = np.array([[t[0][0], t[0][1]], [t[1][0], t[1][1]], [t[2][0],t[2][1]]])
        MESH = np.stack(MESH, axis= -2)
        MESH0 = deepcopy(MESH)
        TRIANGLES = np.stack(TRIANGLES, axis= -2)

        print('STL VOLUME: ', stl.volume)
        print('STL COM: ', stl.center_of_mass())
        print("NUM POINTS: ", points.shape[0])
        #stl.plot()

        
        print("NUM BASIS FUNCTIONS: ", len(self.basisx2))
        if self.parallel:
            pool = mp.Pool(processes = self.nb_processes)
            t0 = time()
            output = pool.starmap(self.intVecBasisFunc, zip(self.basisx2, repeat(MESH), repeat(TRIANGLES)))
            intVals = np.array(output)
        else:
            intVals = np.zeros(len(self.basisx2))
            for i, b in enumerate(self.basisx2):
                tmp = self.intVecBasisFunc(b, MESH, TRIANGLES)
                intVals[i] = tmp
                if i%10==0:
                    print("{}/{}".format(i, len(self.basisx2)))

        self.intVals = intVals

        print("Static Mesh: ", np.allclose(MESH, MESH0))
        print("INTEGRALS: ", np.round(time() - t0,2), ' s')
        np.save(self.integral_path, intVals)


    def E_int(self, i, j):
        # note that self.intVals is initialized as None
        # self.intVals is filled in self.intSTL or in intNpzMesh,
        # so one of these to methods needs to be executed before running self.E_int
        ps = self.basis[i] + self.basis[j]
        return self.intVals[self.lookUpx2[tuple(ps)]]

    def G_int(self, i, j, k, l):
        # note that self.intVals is initialized as None
        # self.intVals is filled in self.intSTL or in intNpzMesh,
        # so one of these to methods needs to be executed before running self.G_int
        M = np.array([[[2,0,0],[1,1,0],[1,0,1]],[[1,1,0],[0,2,0],[0,1,1]],[[1,0,1],[0,1,1],[0,0,2]]])
        if not self.basis[i][k]*self.basis[j][l]: return 0.
        ps = self.basis[i] + self.basis[j] - M[k,l]
        return self.basis[i][k]*self.basis[j][l]*self.intVals[self.lookUpx2[tuple(ps)]]

    def create_G_E_matrices (self, load_intVals=False):
        # basis, bLookup = makeBasis(N)
        # print(len(basis))
        idx = int((self.order+1)*(self.order+2)*(self.order+3)/6)
        self.Etens, self.Itens = np.zeros((3,idx,3,idx), dtype= np.float64), np.zeros((3,idx,3,idx), dtype= np.float64)
        
        if load_intVals:
            self.intVals = np.load(self.integral_path)
        else:
            if self.surface_type in ['stl', 'STL', 'Stl']:
                self.intSTL()
            elif self.surface_type in ['comsol', 'Comsol', 'COMSOL']:
                polydata = self.getMeshFromComsol()
                self.intNpzMesh(polydata, load_polydata=False)


        print(len(self.basisx2))
    
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
    # root_path    = sys.path[0]
    # density      = 9190 # kg / m^3
    # N            = 10
    # E_path       = f'UTe2_stokes_mesh3_Etens_basis_{N}.npy'
    # I_path       = f'UTe2_stokes_mesh3_Itens_basis_{N}.npy'
    # int_path     = f'Ute2_stokes_mesh3_integral_basis_numb_{N}.npy'
    # surface_path = 'UTe2_no_inclusions_mesh3.stl'

    # stokes = StokesMatrices(order=N, density = density, 
    #                         surface_file_path = surface_path,
    #                         surface_file_type = 'stl',
    #                         scale = 'mm',
    #                         integral_path = int_path,
    #                         Emat_path = E_path,
    #                         Itens_path = I_path, 
    #                         parallel=True, nb_processes=6)
    # stokes.create_G_E_matrices()



    root_path    = sys.path[0]
    dimensions   = np.array([0.869e-3, 1.01e-3,  1.193e-3])
    mass         = 7.684976435144674e-6
    density      = mass/np.prod(dimensions) # kg / m^3
    N            = 16
    E_path       = f'Mn3Ge_2104C_stokes_mesh3_Etens_basis_{N}.npy'
    I_path       = f'Mn3Ge_2104C_stokes_mesh3_Itens_basis_{N}.npy'
    int_path     = f'Mn3Ge_2104C_stokes_mesh3_integral_basis_{N}.npy'
    surface_path = 'Mn3Ge_2104C_no_inclusions_mesh_3.stl'

    stokes = StokesMatrices(order=N,
                            surface_file_path = surface_path,
                            surface_file_type = 'stl',
                            scale = 'mm',
                            integral_path = int_path,
                            Emat_path = E_path,
                            Itens_path = I_path, 
                            parallel=True, nb_processes=6)
    stokes.create_G_E_matrices()