import pydgm
import numpy as np
import sys


class XS():

    # Hold the cross section values with routines for outputting to txt file
    def __init__(self, sig_t, sig_f, chi, sig_s, mu=None):
        self.sig_t = sig_t
        self.sig_f = sig_f
        self.chi = chi
        self.sig_s = sig_s
        self.mu = mu if mu is None else np.ones(self.sig_t.shape)

    def write_homogenized_XS(self, fname, mu=None):
        if mu is not None:
            assert mu.shape == self.sig_t.shape
            self.mu = mu

        G, npin = self.sig_t.shape

        sig_t = self.sig_t * self.mu
        vsig_f = self.sig_f * self.mu
        sig_s = self.sig_s * self.mu

        # Write the cross sections to file
        s = '{} {} 0\n'.format(npin, G)
        s += '{}\n'.format(' '.join([str(g) for g in range(G + 1)]))
        s += '{}\n'.format(' '.join([str(g) for g in range(G)]))
        for mat in range(npin):
            s += 'pin {}\n'.format(mat + 1)
            s += '1 1 1.0 0.0 0.602214179\n'

            for g in range(G):
                s += '{:<12.9f} {:<12.9f} {:<12.9f} {:<12.9f}\n'.format(sig_t[g, mat], vsig_f[g, mat], vsig_f[g, mat], self.chi[g, mat])
            for g in range(G):
                s += '{}\n'.format(' '.join(['{:<12.9f}'.format(s) for s in sig_s[:, g, mat]]))

        with open(fname, 'w') as f:
            f.write(s[:-1])

    def __add__(self, newXS):
        sig_t = np.concatenate([self.sig_t, newXS.sig_t], axis=-1)
        sig_f = np.concatenate([self.sig_f, newXS.sig_f], axis=-1)
        sig_s = np.concatenate([self.sig_s, newXS.sig_s], axis=-1)
        chi = np.concatenate([self.chi, newXS.chi], axis=-1)
        mu = np.concatenate([self.mu, newXS.mu], axis=-1)

        return XS(sig_t, sig_f, chi, sig_s, mu)


class DGMSOLVER():

    # Solve the problem using unotran
    def __init__(self, G, fname, fm, cm, mm, nPin, norm=None, mapping=None, vacuum=False, k=None, phi=None, psi=None):
        '''
        Inputs:
            G     - Number of energy groups
            fname - Name of the cross section file
            fm    - Fine mesh
            cm    - Coarse mesh
            mm    - Material map
            nPin  - Number of pincells
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
        self.computenorm = self.norm is None
        self.vacuum = vacuum

        self.mapping = mapping
        # Pass on the options to unotran
        self.setOptions()
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
        pydgm.control.eigen_print = 0
        pydgm.control.outer_print = 0
        pydgm.control.eigen_tolerance = 1e-14
        pydgm.control.outer_tolerance = 1e-12
        pydgm.control.max_eigen_iters = 10000
        pydgm.control.max_outer_iters = 1
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.scatter_leg_order = 0
        pydgm.control.ignore_warnings = True

    def solve(self, k, phi, psi):
        '''
        Solve the problem using Unotran
        '''

        # Initialize the problem
        pydgm.solver.initialize_solver()

        if k is not None:
            pydgm.state.keff = k
        if phi is not None:
            pydgm.state.phi = phi
        if psi is not None:
            pydgm.state.psi = psi

        # Call the solver
        pydgm.solver.solve()

        # Copy any information from Unotran
        self.extractInfo()

        self.iter_k = np.copy(pydgm.state.keff)
        self.iter_phi = np.copy(pydgm.state.phi)
        self.iter_psi = np.copy(pydgm.state.psi)

        # Clean up the solver
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()

    def extractInfo(self):
        '''
        Copy information from Unotran before the solver is deallocated
        '''
        self.phi = np.copy(pydgm.state.mg_phi[0])
        self.dx = np.copy(pydgm.mesh.dx)
        self.mat_map = np.copy(pydgm.state.mg_mmap)
        self.sig_t = np.array([pydgm.state.mg_sig_t[:, self.mat_map[c] - 1] for c in range(len(self.mat_map))]).T
        self.sig_s = np.array([pydgm.state.mg_sig_s[0, :, :, self.mat_map[c] - 1] for c in range(len(self.mat_map))]).T
        self.vsig_f = np.array([pydgm.state.mg_nu_sig_f[:, self.mat_map[c] - 1] for c in range(len(self.mat_map))]).T
        self.chi = np.array([pydgm.state.mg_chi[:, self.mat_map[c] - 1] for c in range(len(self.mat_map))]).T

    def homogenize_space(self):
        '''
        Homogenize the cross sections over the spatial region
        '''

        def homo_space(array):
            '''Convenience function to do the integration'''
            # sum over region
            return np.sum(array.reshape(-1, self.npin, nCellPerPin), axis=2) / V

        # Check that everything is the right shape of arrays
        shape = self.phi.shape
        #assert shape[0] == self.G
        #assert (shape[1] / self.npin) == (shape[1] // self.npin)

        # Compute the number of pins
        nCellPerPin = shape[1] // self.npin

        # Compute the \sum_{g\in G} \sum_{c\in r} V_c dE_g
        V = np.sum(self.dx.reshape(self.npin, -1), axis=1)

        # \forall g\in G, \forall c\in r compute \phi_{g,c} V_c dE_g
        # Homogenize the flux
        phi_dx = self.phi[:, :] * self.dx[:]
        self.phi_homo = homo_space(phi_dx)

        # Either find the norm of the flux or normalize the flux to self.norm
        if self.computenorm:
            self.norm = np.sum(self.phi_homo, axis=-1)
        else:
            print('compute norm')
            norm = self.norm / np.sum(self.phi_homo, axis=-1)
            self.phi_homo *= norm[:, np.newaxis]
            phi_dx *= norm[:, np.newaxis]

        # Homogenize the cross sections
        self.sig_t_homo = homo_space(self.sig_t * phi_dx) / self.phi_homo
        self.sig_f_homo = homo_space(self.vsig_f * phi_dx) / self.phi_homo
        self.chi_homo = homo_space(self.chi * self.dx)
        self.sig_s_homo = np.zeros((self.G, self.G, self.npin))
        for gp in range(self.G):
            self.sig_s_homo[gp] = homo_space(self.sig_s[gp] * phi_dx) / self.phi_homo

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

        phi_homo = homo_energy(self.phi_homo, dE_fine[:, np.newaxis])

        if self.computenorm:
            norm = np.zeros(nCG)
            for g, cg in enumerate(grouping):
                norm[cg - 1] += self.norm[g]
            self.norm = norm

        print(self.mapping.fine_bounds)
        import matplotlib.pyplot as plt

        def barchart(x, y):
            X = np.zeros(2 * len(y))
            Y = np.zeros(2 * len(y))
            for i in range(0, len(y)):
                X[2 * i] = x[i]
                X[2 * i + 1] = x[i + 1]
                Y[2 * i] = y[i]
                Y[2 * i + 1] = y[i]
            return X, Y

        plt.loglog(*barchart(self.mapping.fine_bounds, self.sig_t_homo[:,0]), 'g-', label='fine group')


        self.sig_t_homo = homo_energy(self.sig_t_homo, self.phi_homo)
        self.sig_f_homo = homo_energy(self.sig_f_homo, self.phi_homo)
        self.chi_homo = homo_energy(self.chi_homo)
        sig_s_homo = np.zeros((nCG, nCG, self.npin))
        for gp, g in enumerate(grouping):
            sig_s_homo[g - 1] += homo_energy(self.sig_s_homo[gp], self.phi_homo)
        self.sig_s_homo = sig_s_homo
        self.phi_homo = phi_homo

        plt.loglog(*barchart(self.mapping.coarse_bounds, self.sig_t_homo[:,0]), 'k-', label='coarse group')
        plt.legend(loc=0)
        plt.xlabel('Energy [MeV]')
        plt.ylabel('$\Sigma_t$ [cm$^{-1}$]')
        plt.savefig('test.pdf', transparent=True)
        exit()

