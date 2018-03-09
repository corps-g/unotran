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

    def test_dgmsweeper_dgmsweep(self):
        '''
        Test the dgm iteration function
        '''
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
        np.set_printoptions(16)

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

    def test_dgmsweeper_inner_solve_order0(self):
        '''
        Test order 0 returns the same value when given the converged input
        '''
        order = 0
        # Set the converged fluxes
        phi = np.array([0.198933535568562, 2.7231683533646702, 1.3986600409998782, 1.010361903429942, 0.8149441787223116, 0.8510697418684054, 0.00286224604623])
        pydgm.state.phi[0, :, 0] = phi
        for a in range(4):
            pydgm.state.psi[:, a, 0] = phi / 2.0
        pydgm.state.d_keff = 1.0674868709852505 
            
        # Get the moments from the fluxes
        pydgm.dgm.compute_flux_moments()
        
        phi_m_test = np.array([2.6655619166815265, 0.9635261040519922])
        
        pydgm.dgm.compute_incoming_flux(order)
        pydgm.dgm.compute_xs_moments(order)

        phi_m = np.reshape(np.zeros(2), (1, 2, 1), 'F')
        psi_m = np.reshape(np.zeros(8), (2, 4, 1), 'F')

        pydgm.dgmsweeper.inner_solve(order, phi_m, psi_m)

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[:,a,0].flatten(), 0.5 * phi_m_test, 12)
        
    def test_dgmsweeper_inner_solve_order1(self):
        '''
        Test order 1 returns the same value when given the converged input
        '''
        order = 1
        # Set the converged fluxes
        phi = np.array([0.198933535568562, 2.7231683533646702, 1.3986600409998782, 1.010361903429942, 0.8149441787223116, 0.8510697418684054, 0.00286224604623])
        pydgm.state.phi[0, :, 0] = phi
        for a in range(4):
            pydgm.state.psi[:, a, 0] = phi / 2.0
        pydgm.state.d_keff = 1.0674868709852505 
            
        # Get the moments from the fluxes
        pydgm.dgm.compute_flux_moments()
        
        phi_m_test = np.array([-0.2481536345018054, 0.5742286414743346])
        
        pydgm.dgm.compute_incoming_flux(order)
        pydgm.dgm.compute_xs_moments(order)

        phi_m = np.reshape(np.zeros(2), (1, 2, 1), 'F')
        psi_m = np.reshape(np.zeros(8), (2, 4, 1), 'F')

        pydgm.dgmsweeper.inner_solve(order, phi_m, psi_m)

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[:,a,0].flatten(), 0.5 * phi_m_test, 12)

    def test_dgmsweeper_inner_solve_order2(self):
        '''
        Test order 2 returns the same value when given the converged input
        '''
        order = 2
        # Set the converged fluxes
        phi = np.array([0.198933535568562, 2.7231683533646702, 1.3986600409998782, 1.010361903429942, 0.8149441787223116, 0.8510697418684054, 0.00286224604623])
        pydgm.state.phi[0, :, 0] = phi
        for a in range(4):
            pydgm.state.psi[:, a, 0] = phi / 2.0
        pydgm.state.d_keff = 1.0674868709852505 
            
        # Get the moments from the fluxes
        pydgm.dgm.compute_flux_moments()
        
        phi_m_test = np.array([-1.4562664776830221, -0.3610274595244746])
        
        pydgm.dgm.compute_incoming_flux(order)
        pydgm.dgm.compute_xs_moments(order)

        phi_m = np.reshape(np.zeros(2), (1, 2, 1), 'F')
        psi_m = np.reshape(np.zeros(8), (2, 4, 1), 'F')

        pydgm.dgmsweeper.inner_solve(order, phi_m, psi_m)

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[:,a,0].flatten(), 0.5 * phi_m_test, 12)

    def test_dgmsweeper_inner_solve_order3(self):
        '''
        Test order 3 returns the same value when given the converged input
        '''
        order = 3
        # Set the converged fluxes
        phi = np.array([0.198933535568562, 2.7231683533646702, 1.3986600409998782, 1.010361903429942, 0.8149441787223116, 0.8510697418684054, 0.00286224604623])
        pydgm.state.phi[0, :, 0] = phi
        for a in range(4):
            pydgm.state.psi[:, a, 0] = phi / 2.0
        pydgm.state.d_keff = 1.0674868709852505 
            
        # Get the moments from the fluxes
        pydgm.dgm.compute_flux_moments()
        
        phi_m_test = np.array([-1.0699480859043353, 0.0])
        
        pydgm.dgm.compute_incoming_flux(order)
        pydgm.dgm.compute_xs_moments(order)

        phi_m = np.reshape(np.zeros(2), (1, 2, 1), 'F')
        psi_m = np.reshape(np.zeros(8), (2, 4, 1), 'F')

        pydgm.dgmsweeper.inner_solve(order, phi_m, psi_m)

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[:,a,0].flatten(), 0.5 * phi_m_test, 12)

    def test_dgmsweeper_unfold_flux_moments(self):
        '''
        Test unfolding flux moments into the scalar and angular fluxes
        '''
        order = 0
        psi_moments = np.array([[1.3327809583407633, -0.1240768172509027, -0.728133238841511, -0.5349740429521677],
                                [0.4817630520259961, 0.2871143207371673, -0.1805137297622373, 0.0]])
        psi = np.reshape(np.zeros(8), (2, 4, 1), 'F')
        phi_new = np.reshape(np.zeros(7), (1, 7, 1), 'F')
        psi_new = np.reshape(np.zeros(28), (7, 4, 1), 'F')

        for order in range(4):
            for a in range(4):
                psi[:, a, 0] = psi_moments[:, order]

            pydgm.dgmsweeper.unfold_flux_moments(order, psi, phi_new, psi_new)

        phi_test = np.array([0.198933535568562, 2.7231683533646702, 1.3986600409998782, 1.010361903429942, 0.8149441787223116, 0.8510697418684054, 0.00286224604623])
        np.testing.assert_array_almost_equal(phi_new.flatten(), phi_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_new[:, a].flatten(), phi_test * 0.5)

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()

