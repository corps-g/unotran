import pydgm
import numpy as np
import sys
import matplotlib.pyplot as plt
import time


class XS():

    # Hold the cross section values with routines for outputting to txt file
    def __init__(self, nFG, nCellPerPin, sig_t, sig_f, chi, sig_s, mu=None):

        order, _, G, nMat = sig_t.shape

        self.sig_t = sig_t
        self.sig_f = sig_f
        self.sig_s = sig_s
        self.chi = chi
        o, _, G, nMat = self.sig_t.shape
        self.mu = mu if mu is not None else np.ones((G, nMat))
        self.nFG = nFG
        self.nCellPerPin = nCellPerPin

    def write_homogenized_XS(self, mu=None, mmap=None, order=None, method='rxn_t_mu_all'):

        o, _, G, nMat = self.sig_t.shape
        order = o if order is None else order + 1

        self.mu = mu if mu is not None else self.mu

        na = np.newaxis

        sig_t = np.zeros((order, G, nMat, order), order='F')
        sig_f = np.zeros((order, G, nMat), order='F')
        sig_s = np.zeros((order, 1, G, G, nMat, order), order='F')
        chi = np.zeros((G, self.nCellPerPin * nMat, order), order='F')

        if 'mu_all' in method:
            for o in range(order):
                sig_f[o, :, :] = self.sig_f[o, :, :] * self.mu[:, :]
                for oo in range(order):
                    sig_t[oo, :, :, o] = self.sig_t[oo, o, :, :] * self.mu[:, :]
                    sig_s[oo, 0, :, :, :, o] = self.sig_s[oo, o, :, :, :] * self.mu[:, na, :]
                for i in range(self.nCellPerPin * nMat):
                    chi[:, i, o] = self.chi[o, :, i // self.nCellPerPin]
        elif 'mu_zeroth' in method:
            for o in range(order):
                sig_f[o, :, :] = self.sig_f[o, :, :] * self.mu[:, :]
                for oo in range(order):
                    sig_t[oo, :, :, o] = self.sig_t[oo, o, :, :] * (1.0 if o > 0 else self.mu[:, :])
                    sig_s[oo, 0, :, :, :, o] = self.sig_s[oo, o, :, :, :] * (1.0 if o > 0 else self.mu[:, na, :])
                for i in range(self.nCellPerPin * nMat):
                    chi[:, i, o] = self.chi[o, :, i // self.nCellPerPin]
        elif 'fine_mu' in method:
            for o in range(order):
                sig_f[o, :, :] = self.sig_f[o, :, :]
                for oo in range(order):
                    sig_t[oo, :, :, o] = self.sig_t[oo, o, :, :]
                    sig_s[oo, 0, :, :, :, o] = self.sig_s[oo, o, :, :, :]
                for i in range(self.nCellPerPin * nMat):
                    chi[:, i, o] = self.chi[o, :, i // self.nCellPerPin]
        else:
            raise(NotImplementedError)

        if mmap is not None:
            X = chi.reshape((G, self.nCellPerPin, -1, order))
            chi = np.concatenate([X[:, :, m - 1, :] for m in mmap], axis=1)

        return self.nFG, sig_t, sig_f, sig_s, chi

    def __add__(self, newXS):
        sig_t = np.concatenate([self.sig_t, newXS.sig_t], axis=-1)
        sig_f = np.concatenate([self.sig_f, newXS.sig_f], axis=-1)
        sig_s = np.concatenate([self.sig_s, newXS.sig_s], axis=-1)
        chi = np.concatenate([self.chi, newXS.chi], axis=-1)
        mu = np.concatenate([self.mu, newXS.mu], axis=-1)

        return XS(self.nFG, self.nCellPerPin, sig_t, sig_f, chi, sig_s, mu)


class DGMSOLVER():

    # Solve the problem using unotran
    def __init__(self, G, fname, fm, cm, mm, nPin, dgmstructure, norm=None, mapping=None, XS=None, vacuum=False, order=None, homogOption=None, solveFlag=True, k=None, phi=None, psi=None):
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
        self.vacuum = vacuum
        self.order = order
        self.homogOption = homogOption

        self.mapping = mapping
        # Pass on the options to unotran
        self.setOptions()
        if not solveFlag: return
        # Solve using unotran
        self.solve(k, phi, psi)
        # Homogenize the cross sections over each spatial region
        self.homogenize_space()
        # Homogenize the cross sections over each energy range
        if self.mapping is not None:
            self.homogenize_energy()

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
        pydgm.control.boundary_west = 0.0 if self.vacuum else 1.0
        pydgm.control.boundary_east = 0.0 if self.vacuum else 1.0
        pydgm.control.allow_fission = True
        pydgm.control.recon_print = 1 if self.dgmstructure.G == 44 else 1
        pydgm.control.eigen_print = 0
        pydgm.control.outer_print = 0
        pydgm.control.recon_tolerance = 1e-7
        pydgm.control.eigen_tolerance = 1e-12
        pydgm.control.outer_tolerance = 1e-12
        pydgm.control.max_recon_iters = 10000
        pydgm.control.max_eigen_iters = 10000
        pydgm.control.max_outer_iters = 1
        pydgm.control.lamb = 0.01 if self.dgmstructure.G == 44 else 0.2
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.scatter_leg_order = 0
        pydgm.control.use_dgm = True
        pydgm.control.ignore_warnings = True
        pydgm.control.dgm_basis_name = 'basis/{}.basis'.format(self.dgmstructure.fname).ljust(256)
        pydgm.control.energy_group_map = self.dgmstructure.structure + 1
        if self.order is not None:
            pydgm.control.truncation_map = [min(d - 1, self.order) for d in self.dgmstructure.counts.values()]

        if self.homogOption == 0 or self.homogOption is None:
            # No homogenization
            pydgm.control.truncate_delta = False
        elif self.homogOption == 1:
            # Approximate delta with flat function
            pydgm.control.truncate_delta = True
            pydgm.control.delta_leg_order = 0
        elif self.homogOption == 2:
            # Approximate delta with linear function
            pydgm.control.truncate_delta = True
            pydgm.control.delta_leg_order = 1
        elif self.homogOption == 3:
            # Homogenize over coarse mesh region
            pydgm.control.truncate_delta = False
            pydgm.control.homogenization_map = [i + 1 for i, fx in enumerate(self.fm) for ii in range(fx)]
        elif self.homogOption == 4:
            # Homogenize over all cells with same material
            matID = {}
            baseID = 1
            for mat in range(min(self.mm), max(self.mm) + 1):
                if mat in self.mm:
                    matID[mat] = baseID
                    baseID += 1
            pydgm.control.truncate_delta = False
            pydgm.control.homogenization_map = [matID[self.mm[i]] for i, fx in enumerate(self.fm) for ii in range(fx)]

    def solve(self, k, phi, psi):
        '''
        Solve the problem using Unotran
        '''

        # Initialize the problem
        if self.XS is None:
            pydgm.dgmsolver.initialize_dgmsolver()
        else:
            pydgm.dgmsolver.initialize_dgmsolver_with_moments(*self.XS)

        if k is not None:
            pydgm.state.keff = k
        if phi is not None:
            pydgm.state.phi = phi
        if psi is not None:
            pydgm.state.psi = psi

        # Call the solver
        pydgm.dgmsolver.dgmsolve()

        # Copy any information from Unotran
        self.extractInfo()

        self.iter_k = np.copy(pydgm.state.keff)
        self.iter_phi = np.copy(pydgm.state.phi)
        self.iter_psi = np.copy(pydgm.state.psi)

        # Clean up the solver
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()

    def extractInfo(self):
        '''
        Copy information from Unotran before the solver is deallocated
        '''
        self.phi = np.copy(pydgm.dgm.phi_m[:, 0, :, :])
        self.dx = np.copy(pydgm.mesh.dx)
        self.mat_map = np.copy(pydgm.mesh.mmap)

        order, nCG, nC = self.phi.shape
        nMat = pydgm.dgm.expanded_sig_t.shape[-2]

        self.sig_t = np.zeros((order, order, nCG, nC))
        self.sig_s = np.zeros((order, order, nCG, nCG, nC))
        self.vsig_f = np.zeros((order, nCG, nC))
        self.chi = np.zeros((order, nCG, nC))
        self.basis = np.copy(pydgm.dgm.basis).T  # i,g
        for c in range(nC):
            for o in range(order):
                self.sig_t[:, o, :, c] = pydgm.dgm.expanded_sig_t[:, :, self.mat_map[c] - 1, o]
                self.sig_s[:, o, :, :, c] = pydgm.dgm.expanded_sig_s[:, 0, :, :, self.mat_map[c] - 1, o]
            self.vsig_f[:, :, c] = pydgm.dgm.expanded_nu_sig_f[:, :, self.mat_map[c] - 1]
        for c, r in enumerate(pydgm.state.mg_mmap):
            self.chi[:, :, c] = pydgm.dgm.chi_m[:, r - 1, :].T

    def homogenize_space(self):
        '''
        Homogenize the cross sections over the spatial region
        '''

        na = np.newaxis

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
        space_map = [r for r in range(self.npin) for _ in range(nCellPerPin)]

        # Compute the \sum_{g\in G} \sum_{c\in r} V_c dE_g
        V = np.sum(self.dx.reshape(self.npin, -1), axis=1)

        # \forall c\in r compute \phi_{g,c} V_c
        # Homogenize the flux
        phi_dx = self.phi[:, :, :] * self.dx[na, na, :]

        self.phi_homo = np.nan_to_num(homo_space(phi_dx))

        # Either find the norm of the flux or normalize the flux to self.norm
        if self.computenorm:
            self.norm = np.sum(self.phi_homo, axis=-1)
        else:
            norm = np.nan_to_num(self.norm / np.sum(self.phi_homo, axis=-1))
            norm[norm == 0.0] = 1.0
            self.phi_homo *= norm[0][na, :, na]
            phi_dx *= norm[0][na, :, na]

        if self.order is not None:
            MO = min(self.dgmstructure.maxOrder, self.order + 1)
        else:
            MO = self.dgmstructure.maxOrder
        nCG = max(self.dgmstructure.structure) + 1

        basis_phi = np.zeros((self.npin, MO, MO, MO, nCG), order='f')

        self.phi_homo_fine = np.zeros((self.dgmstructure.G, self.npin))
        for g, cg in enumerate(self.dgmstructure.structure):
            self.phi_homo_fine[g, :] += self.phi_homo[:, cg, :].T.dot(self.basis[:, g])

        b3 = np.nan_to_num(self.basis[na, :, na, na, :] * self.basis[na, na, :, na, :] * self.basis[na, na, na, :, :] / self.phi_homo_fine.T[:, na, na, na, :])
        basis_phi = np.zeros((self.npin, MO, MO, MO, nCG), order='f')
        for g, cg in enumerate(self.dgmstructure.structure):
            basis_phi[:, :, :, :, cg] += b3[:, :, :, :, g]

        # Homogenize the cross sections

        # total
        self.sig_t_homo = np.zeros((MO, MO, nCG, self.npin), order='f')
        for c, r in enumerate(space_map):
            for G in range(nCG):
                self.sig_t_homo[:, :, G, r] += self.sig_t[:, :, G, c].dot(basis_phi[r, :, :, :, G]).dot(phi_dx[:, G, c]) / V[r]

        # scatter
        self.sig_s_homo = np.zeros((MO, MO, nCG, nCG, self.npin), order='f')
        for c, r in enumerate(space_map):
            for G in range(nCG):
                for GP in range(nCG):
                    for j in range(MO):
                        self.sig_s_homo[j, :, GP, G, r] += phi_dx[:, GP, c].dot(basis_phi[r, :, :, j, GP].T).dot(self.sig_s[:, :, GP, G, c]) / V[r]

        # fission
        self.sig_f_homo = np.zeros((MO, nCG, self.npin), order='f')
        for c, r in enumerate(space_map):
            for G in range(nCG):
                self.sig_f_homo[:, G, r] += self.vsig_f[:, G, c].dot(basis_phi[r, :, :, :, G]).dot(phi_dx[:, G, c]) / V[r]

        self.chi_homo = homo_space(self.chi * self.dx)

        self.sig_t_homo = np.nan_to_num(self.sig_t_homo)
        self.sig_s_homo = np.nan_to_num(self.sig_s_homo)
        self.sig_f_homo = np.nan_to_num(self.sig_f_homo)

        return

