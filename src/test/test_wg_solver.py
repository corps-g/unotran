import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestWG_SOLVER(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/3gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 1.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 1.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.legendre_order = 0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

    def test_wg_solver_wg_solve_1_loop(self):
        ''' 
        Test convergence for the within group solver
        '''
        pydgm.control.max_inner_iters = 1

        phi = np.array([4.019852179505758, 2.4109633616022452, 1.9034006148970093]).reshape((1, 1, 3), order='F')
        incident = np.array([[2.3294137999432, 3.9914477887555],
                             [1.4634869354588, 2.0497936227788],
                             [1.1433645683104, 1.6937410699770]]).reshape((2, 3), order='F')
        psi = np.array([[1.84248286151801, 3.40773029740676, 1.41200640302902, 0.677775961546391],
                        [1.20378375094623, 1.91399209431270, 0.88909528292332, 0.472040283216812],
                        [0.92928367052770, 1.53956056710344, 0.692690032114949, 0.357601386372497]]).reshape((1, 4, 3), order='F')

        pydgm.state.mg_phi = phi
        pydgm.state.mg_psi = psi

        phi_test = np.array([4.019852179505758, 2.4109633616022452, 1.9034006148970093])
        incident_test = np.array([[2.3294137999432, 3.9914477887555],
                                  [1.4634869354588, 2.0497936227788],
                                  [1.1433645683104, 1.6937410699770]])
        psi_test = np.array([[0.677775961546391, 1.41200640302902, 3.40773029740676, 1.84248286151801],
                             [0.472040283216812, 0.88909528292332, 1.91399209431270, 1.20378375094623],
                             [0.357601386372497, 0.692690032114949, 1.53956056710344, 0.92928367052770]])

        for g in range(3):

            pydgm.wg_solver.wg_solve(g + 1, phi[:, :, g], psi[:, :, g], incident[:, g])

            np.testing.assert_array_almost_equal(phi[:, :, g].flatten(), phi_test[g], 12, 'group {} is wrong'.format(g))
            np.testing.assert_array_almost_equal(psi[:, :, g].flatten(), psi_test[g], 12, 'group {} is wrong'.format(g))
            np.testing.assert_array_almost_equal(incident[:, g].flatten(), incident_test[g], 12, 'group {} is wrong'.format(g))

    def test_wg_solver_wg_solve_10_loop(self):
        ''' 
        Test convergence for the within group solver
        '''
        pydgm.control.max_inner_iters = 10

        phi = np.array([4.019852179505758, 2.4109633616022452, 1.9034006148970093]).reshape((1, 1, 3), order='F')
        incident = np.array([[2.3294137999432, 3.9914477887555],
                             [1.4634869354588, 2.0497936227788],
                             [1.1433645683104, 1.6937410699770]]).reshape((2, 3), order='F')
        psi = np.array([[1.84248286151801, 3.40773029740676, 1.41200640302902, 0.677775961546391],
                        [1.20378375094623, 1.91399209431270, 0.88909528292332, 0.472040283216812],
                        [0.92928367052770, 1.53956056710344, 0.692690032114949, 0.357601386372497]]).reshape((1, 4, 3), order='F')

        pydgm.state.mg_phi = phi
        pydgm.state.mg_psi = psi

        phi_test = np.array([4.019852179505758, 2.4109633616022452, 1.9034006148970093])
        incident_test = np.array([[2.3294137999432, 3.9914477887555],
                                  [1.4634869354588, 2.0497936227788],
                                  [1.1433645683104, 1.6937410699770]])
        psi_test = np.array([[0.677775961546391, 1.41200640302902, 3.40773029740676, 1.84248286151801],
                             [0.472040283216812, 0.88909528292332, 1.91399209431270, 1.20378375094623],
                             [0.357601386372497, 0.692690032114949, 1.53956056710344, 0.92928367052770]])

        for g in range(3):

            pydgm.wg_solver.wg_solve(g + 1, phi[:, :, g], psi[:, :, g], incident[:, g])

            np.testing.assert_array_almost_equal(phi[:, :, g].flatten(), phi_test[g], 12, 'group {} is wrong'.format(g))
            np.testing.assert_array_almost_equal(psi[:, :, g].flatten(), psi_test[g], 12, 'group {} is wrong'.format(g))
            np.testing.assert_array_almost_equal(incident[:, g].flatten(), incident_test[g], 12, 'group {} is wrong'.format(g))

    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()
