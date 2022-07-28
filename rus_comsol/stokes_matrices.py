import sys
import pyvista as pv
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan2, pi
from scipy.spatial.transform import Rotation
from time import time
import quadpy
from copy import deepcopy
import multiprocessing as mp
from itertools import repeat
import mph
from scipy import linalg as LA
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class StokesMatrices:

    def __init__ (self, basis_order, surface_file_type,
                surface_path, integral_path, E_path, I_path,
                scale='m', parallel=True, nb_processes=1, shift_com=False,
                rotation_matrix=None, find_good_rotation=False):

        self.basis_order = basis_order
        scale_lookup     = {'m': 1, 'cm':1e-2, 'mm':1e-3}
        self.scale       = scale_lookup[scale]

        self.surface_file_type = surface_file_type
        self.surface_path      = surface_path

        self.integral_path = integral_path
        self.E_path        = E_path
        self.I_path        = I_path

        self.parallel     = parallel
        self.nb_processes = nb_processes

        self.shift_com          = shift_com
        self.rotation_matrix    = rotation_matrix
        self.find_good_rotation = find_good_rotation

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
        # if (N, M, L) == (0,0,1):
            # print("STOKES VOLUME: ", vol/L)
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
        vol = self.kahanSum(vol)
        # if (N, M, L) == (0,0,1):
            # print("STOKES VOLUME: ", vol/L)
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
    
    def getMeshFromComsol(self, model_path, save_path, component= 'comp1', mesh= 'mesh1'):
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
        model = client.load(model_path)
        vtx = model.java.component(component).mesh(mesh).getVertex()
        elem = model.java.component(component).mesh(mesh).getElem('tri')
    
        vtx = np.array(vtx)
        vtx = vtx.T
        #reformat faces for use with pyvista
        elem = np.array(elem)
        print(elem.shape)
        elem = np.vstack((np.full((1, elem.shape[1]), 3), elem))
        faces = np.hstack(elem.T)
    
        polydata = {'faces' : faces, 'points' : vtx}
        np.savez(save_path, **polydata)
    
        return polydata
    
    def NpzToPolyData(self, npz_file, scale):
        npz_stl  = np.load(npz_file)
        polydata = pv.PolyData(npz_stl['points'], npz_stl['faces'])
        polydata = polydata.scale(scale)
        return polydata
    
    def StlToPolyData(self, stl_file, scale):
        scheme   = quadpy.t2.get_good_scheme(40)
        polydata = pv.PolyData(stl_file)
        polydata = polydata.scale(scale)
        return polydata
    
    
    
    def intPolyData(self, polydata, basis_order, save_path, parallel=True, nb_processes=1,
                    shift_com=False, chunk_size=20000, rotation_matrix=None):

        print('STL VOLUME:  ', polydata.volume)    
        print("INITIAL COM: ", polydata.center_of_mass())
        points, faces = polydata.points, polydata.faces.reshape(-1, 4)[:, 1:]

        if shift_com == True:
            offset = np.tile(polydata.center_of_mass(), (points.shape[0], 1))
            points -= offset
        
        print("SHIFTED COM: ", polydata.center_of_mass())
        print('')

        if rotation_matrix is not None:
            for i in range(points.shape[0]):
                p = points[i]
                pp = np.matmul(rotation_matrix, p)
                points[i] = pp
    
    
        current_chunk = chunk_size
        chunk_num = 0
        if parallel:
            print("Starting pool...")
            pool = mp.Pool(processes=nb_processes)
        while current_chunk-chunk_size<=faces.shape[0]:
    
            t0 = time()
            print(f"CHUNK {chunk_num} ...")
    
            if current_chunk <= faces.shape[0]: mesh_chunk = chunk_size
            else: mesh_chunk = faces.shape[0]-current_chunk+chunk_size
            MESH = np.zeros([mesh_chunk, 3, 3])
            TRIANGLES = np.zeros([mesh_chunk, 3, 2])
            for idx, f in enumerate(faces[current_chunk-chunk_size:min(faces.shape[0], current_chunk)]):
                t = np.array([points[f[0]], points[f[1]], points[f[2]]])
                MESH[idx] = t
                TRIANGLES[idx] = np.array([[t[0][0], t[0][1]], [t[1][0], t[1][1]], [t[2][0],t[2][1]]])
            MESH = np.stack(MESH, axis= -2)
            MESH0 = deepcopy(MESH)
            TRIANGLES = np.stack(TRIANGLES, axis= -2)
    
            print("NUM POINTS: ", points.shape[0])
            #stl.plot()
    
            intBasis, basisLookup = self.makeBasis(2*basis_order)
            print("NUM BASIS FUNCTIONS: ", len(intBasis))
            if parallel:
                print(MESH.shape, TRIANGLES.shape)
    
                t0 = time()
                output = pool.starmap(self.intVecBasisFunc, zip(intBasis, repeat(MESH), repeat(TRIANGLES)))
                intVals = np.array(output)
    
            else:
                intVals = np.zeros(len(intBasis))
                for i, b in enumerate(intBasis):
                    tmp = self.intVecBasisFunc(b, MESH, TRIANGLES)
                    intVals[i] = tmp
                    if i%10==0:
                        print("{}/{}".format(i, len(intBasis)))

            print("CHUNK VOLUME: ", intVals[0])
    
            print("Static Mesh: ", np.allclose(MESH, MESH0))
            tmp = save_path.split("/")
            end_tmp = tmp[-1].split(".")
            end_tmp[0] += f"_chunk{chunk_num}"
            end_tmp = ".".join(end_tmp)
            tmp[-1] = end_tmp
    
            chunk_save = "/".join(tmp)
            np.save(chunk_save, intVals)
            current_chunk += chunk_size
            chunk_num += 1
    
            print('COMPLETED IN ', np.round(time()-t0,3), ' SECONDS')
            print('')
    
    
        tmp = save_path.split("/")
        end_tmp = tmp[-1].split(".")
        end_tmp[0] += f"_chunk0"
        end_tmp = ".".join(end_tmp)
        tmp[-1] = end_tmp
    
        path0 = "/".join(tmp)
        dat0 = np.load(path0)
        fint = np.zeros(dat0.shape)
        fnum = np.arange(chunk_num)
        for c in fnum:
            tmp = save_path.split("/")
            end_tmp = tmp[-1].split(".")
            end_tmp[0] += f"_chunk{c}"
            end_tmp = ".".join(end_tmp)
            tmp[-1] = end_tmp
            dat_c = "/".join(tmp)
            dat_chunk = np.load(dat_c)
    
            fint += dat_chunk

        print('STL VOLUME:       ', polydata.volume)
        print('STOKES TOTAL VOL: ', fint[0])

        if fint[0]<0:
            fint = -fint
            print('the stokes volume is negative!')
            print('possible reason is wrong sign of normal vectors;')
            print('will multiply all integral values with a minus sign before saving!')
        np.save(save_path, fint)
    
    
    def E_int(self, i, j, intVals, basis, blookup_calc):
        ps = basis[i] + basis[j]
        return intVals[blookup_calc[tuple(ps)]]
    def G_int(self, i, j, k, l, intVals, basis, blookup_calc):
        M = np.array([[[2,0,0],[1,1,0],[1,0,1]],[[1,1,0],[0,2,0],[0,1,1]],[[1,0,1],[0,1,1],[0,0,2]]])
        if not basis[i][k]*basis[j][l]: return 0.
        ps = basis[i] + basis[j] - M[k,l]
        return basis[i][k]*basis[j][l]*intVals[blookup_calc[tuple(ps)]]

    
    def get_moment_of_inertia_rotation_matrix(self, polydata, save_path, parallel=True, nb_processes=1,
                                            shift_com=False, chunk_size=20000):

        N = 2
        self.intPolyData(polydata=polydata, basis_order=N, save_path=save_path, 
                    parallel=parallel, nb_processes=nb_processes,
                    shift_com=shift_com, chunk_size=chunk_size, rotation_matrix=None)

        intVals = np.load(save_path)
        bbasis, blookup = self.makeBasis(2*N)

        Ixx, Iyy, Izz = intVals[blookup[(2,0,0)]], intVals[blookup[(0,2,0)]], intVals[blookup[(0,0,2)]]
        Ixy, Ixz, Iyz = intVals[blookup[(1,1,0)]], intVals[blookup[(1,0,1)]], intVals[blookup[(0,1,1)]]
        IM = np.array([[Ixx, Ixy, Ixz],[Ixy, Iyy, Iyz],[Ixz, Iyz, Izz]])

        rotation_matrix = LA.eigh(IM)[1]
        rotation_matrix = rotation_matrix.T

        tmp = save_path.split("/")
        tmp[-1] = 'rotation_matrix.npy'
        path = "/".join(tmp)
        np.save(path, rotation_matrix)

        ''' this the following analysis is just to get some nice print of what the angles are '''
        rotation = Rotation.from_matrix(rotation_matrix)
        Euler_angles = rotation.as_euler('xyz', degrees=True)
        print('we rotate the surface file:')
        print('       ', Euler_angles[0], ' degrees around the x-axis')
        print('       ', Euler_angles[1], ' degrees around the y-axis')
        print('       ', Euler_angles[2], ' degrees around the z-axis')

        return rotation_matrix
    
    
    def create_G_E_matrices(self):
        if self.surface_file_type in ['stl', 'STL', 'Stl']:
            polydata = self.StlToPolyData(self.surface_path, self.scale)
        elif self.surface_file_type in ['Comsol', 'comsol', 'COMSOL']:
            polydata = self.NpzToPolyData(self.surface_path, self.scale)
        else:
            print('the specified surface_file_type is not valid!')
            print('it must be "stl" or "comsol"')

        if self.find_good_rotation==True:
            tmp = self.integral_path.split("/")
            # end_tmp = tmp[-1].split(".")
            # end_tmp[0] += f"_tensor_for_moment_of_inertia"
            # end_tmp = ".".join(end_tmp)
            tmp[-1] = 'integrals_for_moment_of_inertia_tensor.npy'
            path = "/".join(tmp)
            self.rotation_matrix = self.get_moment_of_inertia_rotation_matrix(polydata, path, shift_com=self.shift_com,
                                                                            parallel=self.parallel, nb_processes=self.nb_processes)
           
    
        self.intPolyData(polydata, self.basis_order, self.integral_path,
                        parallel=self.parallel, nb_processes=self.nb_processes,
                        shift_com=self.shift_com, rotation_matrix=self.rotation_matrix)
    
        basis, bLookup = self.makeBasis(self.basis_order)
        print(len(basis))
        idx = int((self.basis_order+1)*(self.basis_order+2)*(self.basis_order+3)/6)
        E, I = np.zeros((3,idx,3,idx), dtype= np.float64), np.zeros((3,idx,3,idx), dtype= np.float64)
        # M = np.array([[[2,0,0],[1,1,0],[1,0,1]],[[1,1,0],[0,2,0],[0,1,1]],[[1,0,1],[0,1,1],[0,0,2]]])
    
        b_calc, blookup_calc = self.makeBasis(2*self.basis_order)
        intVals = np.load(self.integral_path)
        
    
        for x in range(3*idx):
            i, k = int(x/idx), x%idx
            for y in range(x, 3*idx):
                j, l = int(y/idx), y%idx
                if i==j: E[i,k,j,l]=E[j,l,i,k]=self.E_int(k,l, intVals, basis, blookup_calc)
                I[i,k,j,l]=I[j,l,i,k]=self.G_int(k,l,i,j, intVals, basis, blookup_calc)
    
        E = E.reshape(3*idx,3*idx)
        try:
            np.linalg.cholesky(E)
            print("good cholesky!")
        except:
            print("still ffffff")
    
        np.save(self.E_path, E)
        np.save(self.I_path, I)
    
    
    def kahanSum(self, fa):
        sum = 0.0
    
        # Variable to store the error
        c = 0.0
    
        # Loop to iterate over the array
        for f in fa:
            y = f - c
            t = sum + y
            c = (t - sum) - y
            sum = t
    
        return sum


if __name__ == '__main__':
    root_path = sys.path[0]
    from rus_comsol.rus_xyz import RUSXYZ


    basis_order  = 18
    nb_processes = 6

    # surface_path      = f"{root_path}/SrTiO3_2104B_comsol_super_fine_mesh.stl"
    surface_path      = f"{root_path}/SrTiO3_2104B_comsol_mesh_3.stl"
    surface_file_type = 'stl'
    scale             = 'mm'
    
    integral_path = f"{root_path}/integrals/integrals_basis_{basis_order}.npy"
    E_path        = f"{root_path}\\integrals\\Etens_basis{basis_order}.npy"
    I_path        = f"{root_path}\\integrals\\Itens_basis{basis_order}.npy"


    mats = StokesMatrices(basis_order, surface_file_type, surface_path, integral_path,
                        E_path, I_path, scale='mm', parallel=True, nb_processes=nb_processes,
                        shift_com=True, find_good_rotation=True)

    mats.create_G_E_matrices()


    # elastic constants init in GPa
    elastic_dict = {
        'c11': 315.328,
        'c12': 102.192,
        'c44': 122.201
        }
    
    density    = 5110

    rot_path = f"{root_path}/integrals/rotation_matrix.npy"
    rot_mat  = np.load(rot_path)
    rot      = Rotation.from_matrix(rot_mat)
    Euler    = rot.as_euler('xyz', degrees=True)
    print(Euler)

    print('create rus object')
    print('and calculate matrices')
    rus_object = RUSXYZ(cij_dict=elastic_dict, symmetry='cubic', order=basis_order,
                    Emat_path=E_path, Itens_path=I_path,
                    nb_freq=105, density=density,
                    angle_x=Euler[0], angle_y=Euler[1], angle_z=Euler[2],
                    init=True, use_quadrants=False)
    res = rus_object.compute_resonances()

    res0 = np.loadtxt(f"{root_path}/fit_report_SrTiO3_2104B_RT_xyz_fit_stokes_basis_16_90_resonances.txt")
    res0 = (res0.T)[2]

    print((res-res0)/res0)
