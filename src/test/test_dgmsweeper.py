import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestDGMSWEEPER(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = True
        pydgm.control.use_dgm = True
        pydgm.control.energy_group_map = [4]
        # pydgm.control.max_inner_iters = 5000
        pydgm.control.ignore_warnings = True
        pydgm.control.use_recondensation = False
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.dgm_basis_name = 'test/7gbasis'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 0

        pydgm.dgmsolver.initialize_dgmsolver()

    def test_dgmsweeper_innersolver(self):
        ''' 
        Test the inner solver in the dgm sweeper routine
        '''

        keff_test = 1.06748687099
        phi_test = np.array([0.19893353556856153, 2.7231683533646662, 1.398660040999877, 1.0103619034299416, 0.8149441787223114, 0.85106974186841, 0.0028622460462300096])
        phi_m = np.reshape(np.zeros(2), (1, 2, 1), 'F')
        psi_m = np.reshape(np.zeros(8), (2, 4, 1), 'F')

        # Provide the values for the scalar and angular flux
        for c in range(pydgm.mesh.number_cells):
            pydgm.state.phi[0, :, c] = phi_test
            for a in range(pydgm.angle.number_angles * 2):
                pydgm.state.psi[:, a, c] = phi_test / 2

        # Solve the problem
        pydgm.dgm.compute_flux_moments()
        phi_m_test = np.concatenate((np.loadtxt('test/7gbasis').T.dot(phi_test), [0])).reshape((2, -1))

        print phi_m_test

        for i in range(4):
            pydgm.dgm.compute_incoming_flux(order=i)
            pydgm.dgm.compute_xs_moments(order=i)
            pydgm.dgmsweeper.inner_solve(i, phi_m, psi_m)
            p1 = phi_m[0, :, 0]
            p2 = phi_m_test[:, i]
            f1 = p1[1] / p1[0]
            f2 = p2[1] / p2[0]
            np.testing.assert_almost_equal(f1, f2, 12, 'order {} failed'.format(i))

    def test_dgmsweeper_dgmsweep(self):
        '''
        Test the dgm iteration function
        '''
        pydgm.control.inner_print = False
        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        X = np.outer(pydgm.material.chi[:, 0], pydgm.material.nu_sig_f[:, 0])

        keff, phi = np.linalg.eig(np.linalg.inv(T - S).dot(X))

        i = np.argmax(keff)
        keff_test = keff[i]
        phi_test = np.abs(phi[:, i])
        # Normalize the flux
        phi_test = phi_test / sum(phi_test) * 7

        # Set the values for the scalar and angular flux
        for c in range(pydgm.mesh.number_cells):
            pydgm.state.phi[0, :, c] = phi_test
            for a in range(pydgm.angle.number_angles * 2):
                pydgm.state.psi[:, a, c] = phi_test / 2

        # Set the eigenvalue
        pydgm.state.d_keff = keff_test

        # Make some empty containers
        phi_new = pydgm.state.phi * 0.0
        psi_new = pydgm.state.psi * 0.0

         # Perform one DGM iteration
        pydgm.dgmsweeper.dgmsweep(phi_new, psi_new)

        # Normalize the output
        pydgm.state.normalize_flux(7, phi_new, psi_new)

        np.testing.assert_array_almost_equal(pydgm.state.phi.flatten(), phi_new.flatten(), 12)

    def test_dgmsweeper_inner_solve(self):
        '''
        No good way to test this function...
        '''

    def test_dgmsweeper_unfold_flux_moments(self):
        '''
        Test unfolding flux moments into the scalar and angular fluxes
        '''
        order = 0
        psi_moments = np.array([[0.428191466212384, -0.07050405620345324, -0.21879413525733196, -0.11751528764617786],
                                [0.0004299727459136011, 0.0004201812076152387, 0.000410413659723487, 0.0]])
        psi = np.reshape(np.zeros(8), (2, 4, 1), 'F')
        phi_new = np.reshape(np.zeros(7), (1, 7, 1), 'F')
        psi_new = np.reshape(np.zeros(28), (7, 4, 1), 'F')

        for order in range(4):
            for a in range(4):
                psi[:, a, 0] = psi_moments[:, order]
            
            pydgm.dgmsweeper.unfold_flux_moments(order, psi, phi_new, psi_new)

        phi_test = np.array([ 0.021377987105421 , 0.7984597778757521, 0.5999743700269914, 0.0450954611897237, 0.0014555781016859, 0.0000276607249577, 0.000000019588085 ])
        np.testing.assert_array_almost_equal(phi_new.flatten(), phi_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_new[:, a].flatten(), phi_test * 0.5)

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()

