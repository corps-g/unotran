import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestSTATE(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = True
        pydgm.control.eigen_print = False
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.eigen_tolerance = 1e-15
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = False
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 0
        pydgm.control.use_DGM = False

    def test_state_initialize(self):
        ''' 
        Test initializing of state arrays, no psi
        '''
        pydgm.state.initialize_state()

        phi_test = np.ones((1, 7))

        np.testing.assert_array_almost_equal(pydgm.state.phi[0], phi_test, 12)

    def test_state_initialize2(self):
        ''' 
        Test initializing of state arrays, with psi
        '''
        pydgm.control.store_psi = True
        pydgm.state.initialize_state()

        phi_test = np.ones((1, 7))
        psi_test = np.ones((1, 4, 7)) / 2

        np.testing.assert_array_almost_equal(pydgm.state.phi[0], phi_test, 12)
        np.testing.assert_array_almost_equal(pydgm.state.psi, psi_test, 12)

    def test_state_update_fission_density(self):
        pydgm.control.fine_mesh = [3, 10, 3]
        pydgm.control.coarse_mesh = [0.0, 1.0, 2.0, 3.0]
        pydgm.control.material_map = [1, 2, 6]

        pydgm.state.initialize_state()

        density_test = [0.85157342, 0.85157342, 0.85157342, 1.08926612, 1.08926612,
                        1.08926612, 1.08926612, 1.08926612, 1.08926612, 1.08926612,
                        1.08926612, 1.08926612, 1.08926612, 0.0, 0.0, 0.0]

        np.testing.assert_array_almost_equal(pydgm.state.density, density_test, 12)

    def tearDown(self):
        # Finalize the dependancies
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()
