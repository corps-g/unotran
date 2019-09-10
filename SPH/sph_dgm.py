import pydgm
import numpy as np
import sys

class XS():

    # Hold the cross section values with routines for outputting to txt file
    def __init__(self, nFG, nCellPerPin, sig_t, sig_f, chi, sig_s, mu=None):

        order, _, G, nMat = sig_t.shape

        self.sig_t = np.zeros((order, G, nMat, order), order='F')
        self.sig_f = np.zeros((order, G, nMat), order='F')
        self.sig_s = np.zeros((order, 1, G, G, nMat, order), order='F')
        self.chi = np.zeros((G, nCellPerPin * nMat, order), order='F')
        self.mu = np.ones((G, nMat, order), order='F')
        self.nFG = nFG
        self.nCellPerPin = nCellPerPin

        self.sig_f[:,:,:] = sig_f[:,:,:]
        for o in range(order):
            self.sig_t[:,:,:,o] = sig_t[:,o,:,:]
            self.sig_s[:,0,:,:,:,o] = sig_s[:,o,:,:,:]
            for i in range(nCellPerPin * nMat):
                self.chi[:,i,o] = chi[o,:,i // nCellPerPin]
            if mu is not None:
                self.mu[:,:,o] = mu[o,:,:]

    def write_homogenized_XS(self, mu=None):
        if mu is not None:
            order, _, G, nMat = self.sig_t.shape
            for o in range(order):
                self.mu[:,:,o] = mu[o,:,:]
        sig_t = self.sig_t[:,:,:,:] * self.mu[np.newaxis,:,:,:]
        vsig_f = self.sig_f * self.mu[:,:,0]
        sig_s = self.sig_s * self.mu[np.newaxis,np.newaxis,:,np.newaxis,:,:]

        return self.nFG, sig_t, vsig_f, sig_s, self.chi

    def __add__(self, newXS):
        sig_t = np.concatenate([self.sig_t, newXS.sig_t], axis=-1)
        sig_f = np.concatenate([self.sig_f, newXS.sig_f], axis=-1)
        sig_s = np.concatenate([self.sig_s, newXS.sig_s], axis=-1)
        chi = np.concatenate([self.chi, newXS.chi], axis=-1)
        mu = np.concatenate([self.mu, newXS.mu], axis=-1)

        return XS(self.nFG, self.nCellPerPin, sig_t, sig_f, chi, sig_s, mu)


class DGMSOLVER():

    # Solve the problem using unotran
    def __init__(self, G, fname, fm, cm, mm, nPin, dgmstructure, norm=None, mapping=None, XS=None):
        '''
        Inputs:
            G     - Number of energy groups
            fname - Name of the cross section file
            fm    - Fine mesh
            cm    - Coarse mesh
            mm    - Material map
            nPin  - Number of pincells
            dgmstructure - class holding the coarse group structure and basis info
            norm  - norm of the flux to keep constant (match phi shape)
            mapping - structure class that holds fine -> coarse mapping
        '''

        self.G = G
        self.fname = fname
        self.fm = fm
        self.cm = cm
        self.mm = mm
        self.npin = nPin
        self.norm = norm
        self.dgmstructure = dgmstructure
        self.computenorm = self.norm is None
        self.XS = XS

        self.mapping = mapping
        assert self.mapping is not None
        # Pass on the options to unotran
        self.setOptions()
        # Solve using unotran
        self.solve()
        # Homogenize the cross sections over each spatial region
        self.homogenize_space()
        # Homogenize the cross sections over each energy range
        #self.homogenize_energy()

    def setOptions(self):
        '''
        Set the options for the Unotran solve
        '''
        pydgm.control.spatial_dimension = 1
        pydgm.control.fine_mesh_x = self.fm
        pydgm.control.coarse_mesh_x = self.cm
        pydgm.control.material_map = self.mm
        pydgm.control.xs_name = self.fname.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_west = 1.0
        pydgm.control.boundary_east = 1.0
        pydgm.control.allow_fission = True
        pydgm.control.recon_print = 1
        pydgm.control.eigen_print = 0
        pydgm.control.outer_print = 0
        pydgm.control.recon_tolerance = 1e-14
        pydgm.control.eigen_tolerance = 1e-14
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.max_recon_iters = 10000
        pydgm.control.max_eigen_iters = 10000
        pydgm.control.max_outer_iters = 1
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.scatter_leg_order = 0
        pydgm.control.use_dgm = True
        pydgm.control.ignore_warnings = True
        pydgm.control.dgm_basis_name = 'basis/{}.basis'.format(self.dgmstructure.fname).ljust(256)
        pydgm.control.energy_group_map = self.dgmstructure.structure + 1

    def solve(self):
        '''
        Solve the problem using Unotran
        '''

        # Initialize the problem
        if self.XS is None:
            pydgm.dgmsolver.initialize_dgmsolver()
        else:
            pydgm.control.lamb = 0.02
            pydgm.dgmsolver.initialize_dgmsolver_with_moments(*self.XS)

        print('initialized')

        # Call the solver
        pydgm.dgmsolver.dgmsolve()

        # Copy any information from Unotran
        self.extractInfo()

        # Clean up the solver
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()

    def extractInfo(self):
        '''
        Copy information from Unotran before the solver is deallocated
        '''
        self.phi = np.copy(pydgm.dgm.phi_m[:,0,:,:])
        self.dx = np.copy(pydgm.mesh.dx)
        self.mat_map = np.copy(pydgm.mesh.mmap)

        order, nCG, nC = self.phi.shape
        nMat = pydgm.dgm.expanded_sig_t.shape[-2]

        self.sig_t = np.zeros((order, order, nCG, nC))
        self.sig_s = np.zeros((order, order, nCG, nCG, nC))
        self.vsig_f = np.zeros((order, nCG, nC))
        self.chi = np.zeros((order, nCG, nC))
        for c in range(nC):
            for o in range(order):
                self.sig_t[:,o,:,c] = pydgm.dgm.expanded_sig_t[:,:,self.mat_map[c]-1,o]
                self.sig_s[:,o,:,:,c] = pydgm.dgm.expanded_sig_s[:,0,:,:,self.mat_map[c]-1,o]
            self.vsig_f[:,:,c] = pydgm.dgm.expanded_nu_sig_f[:,:,self.mat_map[c]-1]
            self.chi[:,:,c] = pydgm.dgm.chi_m[:,c,:].T

    def homogenize_space(self):
        '''
        Homogenize the cross sections over the spatial region
        '''

        def homo_space(array):
            '''Convenience function to do the integration'''
            # sum over region
            shape = array.shape
            return np.sum(array.reshape(*shape[:-1], self.npin, nCellPerPin, order='F'), axis=-1) / V


        # Check that everything is the right shape of arrays
        shape = self.phi.shape
        assert (shape[2] / self.npin) == (shape[2] // self.npin)

        # Compute the number of pins
        nCellPerPin = shape[2] // self.npin

        # Compute the \sum_{g\in G} \sum_{c\in r} V_c dE_g
        V = np.sum(self.dx.reshape(self.npin, -1), axis=1)

        # \forall g\in G, \forall c\in r compute \phi_{g,c} V_c dE_g
        # Homogenize the flux
        phi_dx = self.phi * self.dx
        self.phi_homo = homo_space(phi_dx)

        # Either find the norm of the flux or normalize the flux to self.norm
        if self.computenorm:
            self.norm = np.sum(phi_dx, axis=-1)
        else:
            print('compute norm')
            norm = self.norm / self.phi_homo.dot(V)
            self.phi_homo *= norm[:,:,np.newaxis]
            phi_dx *= norm[:,:,np.newaxis]

        # Homogenize the cross sections
        self.sig_t_homo = np.nan_to_num(homo_space(self.sig_t * phi_dx[:,np.newaxis,:,:]) / self.phi_homo[:,np.newaxis,:,:])
        self.sig_f_homo = np.nan_to_num(homo_space(self.vsig_f * phi_dx) / self.phi_homo)
        self.chi_homo = homo_space(self.chi * self.dx)
        self.sig_s_homo = np.nan_to_num(homo_space(self.sig_s * phi_dx[:,np.newaxis,:,np.newaxis,:]) / self.phi_homo[:,np.newaxis,:,np.newaxis,:])

    def homogenize_energy(self):
        '''
        Homogenize the cross sections over the energy range
        '''

        def homo_energy(array1, array2=None):
            '''
            convinence function to do the integration

            return \frac{\sum_i array1[i] * array2[i]}{\sum_i array2[i]} for each coarse group
            '''
            if array2 is not None:
                y = np.zeros((nCG, len(array1[0])))
                z = np.zeros((nCG, len(array1[0])))
                for g, cg in enumerate(grouping):
                    z[cg - 1] += array1[g] * array2[g]
                    y[cg - 1] += array2[g]

                return z / y
            else:
                z = np.zeros((nCG, len(array1[0])))
                for g, cg in enumerate(grouping):
                    z[cg - 1] += array1[g]
                return z

        nCG = self.mapping.nCG
        nFG = self.mapping.nFG
        grouping = np.array(self.mapping.grouping)

        dE_coarse = np.array(self.mapping.dE_coarse)
        dE_fine = np.array(self.mapping.dE_fine)

        phi_homo = homo_energy(self.phi_homo, dE_fine[:,np.newaxis])

        if self.computenorm:
            norm = np.zeros(nCG)
            for g, cg in enumerate(grouping):
                norm[cg - 1] += self.norm[g]
            self.norm = norm

        self.sig_t_homo = homo_energy(self.sig_t_homo, self.phi_homo)
        self.sig_f_homo = homo_energy(self.sig_f_homo, self.phi_homo)
        self.chi_homo = homo_energy(self.chi_homo)
        sig_s_homo = np.zeros((nCG, nCG, self.npin))
        for gp, g in enumerate(grouping):
            sig_s_homo[g - 1] += homo_energy(self.sig_s_homo[gp], self.phi_homo)
        self.sig_s_homo = sig_s_homo
        self.phi_homo = phi_homo

