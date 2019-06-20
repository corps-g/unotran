import pydgm
import numpy as np


class XS():

    # Hold the cross section values with routines for outputting to txt file
    def __init__(self, sig_t, sig_f, chi, sig_s, mu=None):
        self.sig_t = sig_t
        self.D = 1.0 / (3 * self.sig_t)
        self.sig_f = sig_f
        self.chi = chi
        self.sig_s = sig_s
        if mu is None:
            self.mu = np.ones(self.sig_t.shape)
        else:
            self.mu = mu

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
    def __init__(self, G, fname, fm, cm, mm, nPin, norm=None):
        '''
        Inputs:
            G     - Number of energy groups
            fname - Name of the cross section file 
            fm    - Fine mesh
            cm    - Coarse mesh
            mm    - Material map 
            nPin  - Number of pincells 
        '''
        self.G = G
        self.fname = fname
        self.fm = fm
        self.cm = cm
        self.mm = mm
        self.npin = nPin
        self.norm = norm

        # Pass on the options to unotran
        self.setOptions()

        # Solve using unotran
        self.solve()

        # Homogenize the cross sections over each pin cell
        self.homogenize()

    def setOptions(self):
        '''
        Set the options for the Unotran solve
        '''

        pydgm.control.spatial_dimension = 1
        pydgm.control.fine_mesh_x = self.fm
        pydgm.control.coarse_mesh_x = self.cm
        pydgm.control.material_map = self.mm
        pydgm.control.xs_name = self.fname.ljust(256)
        pydgm.control.angle_order = 1
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_west = 1.0
        pydgm.control.boundary_east = 1.0
        pydgm.control.allow_fission = False
        pydgm.control.eigen_print = 0
        pydgm.control.outer_print = 0
        pydgm.control.inner_print = 0
        pydgm.control.eigen_tolerance = 1e-16
        pydgm.control.outer_tolerance = 1e-16
        pydgm.control.inner_tolerance = 1e-16
        pydgm.control.max_eigen_iters = 10000
        pydgm.control.max_outer_iters = 1
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.scatter_leg_order = 0
        pydgm.control.ignore_warnings = True

    def solve(self):
        '''
        Solve the problem using Unotran
        '''

        # Initialize the problem
        pydgm.solver.initialize_solver()

        # Call the solver
        pydgm.solver.solve()

        # Copy any information from Unotran
        self.extractInfo()

        # Clean up the solver
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()

    def extractInfo(self):
        '''
        Copy information from Unotran before the solver is deallocated
        '''

        self.phi = np.copy(pydgm.state.phi[0])
        self.dx = np.copy(pydgm.mesh.dx)
        self.mat_map = np.copy(pydgm.state.mg_mmap)
        self.sig_t = np.array([pydgm.material.sig_t[:, self.mat_map[c] - 1] for c in range(len(self.mat_map))]).T
        self.D = 1 / (3 * self.sig_t)
        self.sig_s = np.array([pydgm.material.sig_s[0, :, :, self.mat_map[c] - 1] for c in range(len(self.mat_map))]).T
        self.vsig_f = np.array([pydgm.material.nu_sig_f[:, self.mat_map[c] - 1] for c in range(len(self.mat_map))]).T
        self.chi = np.array([pydgm.material.chi[:, self.mat_map[c] - 1] for c in range(len(self.mat_map))]).T

    def homogenize(self):
        '''
        Homogenize the cross sections over the pincell
        '''

        def homosum(array):
            '''Convenience function to make things shorter'''
            return np.sum((array).reshape(-1, self.npin, nCellPerPin), axis=2) / l

        # Check that everything is the right shape of arrays
        shape = self.phi.shape
        assert shape[0] == self.G
        assert (shape[1] / self.npin) == (shape[1] // self.npin)

        # Compute the number of pins
        nCellPerPin = shape[1] // self.npin

        # Homogenize the flux
        l = np.sum(self.dx.reshape(self.npin, -1), axis=1)
        d_phi = self.phi[:, :] * self.dx[:]
        self.phi_homo = homosum(d_phi)

        # Either find the norm of the flux or normalize the flux to self.norm
        if self.norm is None:
            self.norm = np.sum(d_phi)
        else:
            norm = self.phi_homo.dot(l)
            self.phi_homo *= self.norm / norm
            d_phi *= self.norm / norm

        # Homogenize the cross sections
        self.sig_t_homo = homosum(d_phi * self.sig_t) / self.phi_homo
        self.sig_f_homo = homosum(d_phi * self.vsig_f) / self.phi_homo
        self.chi_homo = homosum(self.chi * self.dx)
        self.sig_s_homo = np.zeros((self.G, self.G, self.npin))
        for gp in range(self.G):
            self.sig_s_homo[gp] = homosum(d_phi * self.sig_s[gp]) / self.phi_homo
        self.sig_a_homo = self.sig_t_homo - self.sig_s_homo
        self.D_homo = homosum(d_phi * self.D) / self.phi_homo
        self.sig_t_homo = 1 / (3 * self.D_homo)
        self.sig_s_homo = self.sig_t_homo - self.sig_a_homo

