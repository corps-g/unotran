import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestSWEEPER_2D(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.spatial_dimension = 2
        pydgm.control.fine_mesh_x = [1]
        pydgm.control.fine_mesh_y = [1]
        pydgm.control.coarse_mesh_x = [0.0, 1.0]
        pydgm.control.coarse_mesh_y = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 6
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.allow_fission = False
        pydgm.control.outer_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.equation_type = 'DD'
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = True
        pydgm.control.scatter_leg_order = 1
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.delta_leg_order = 1
        pydgm.control.use_dgm = False

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

    def test_sweeper_computeEQ_dd(self):
        ''' 
        Test diamond difference equation
        '''

        N = 8
        Ps = np.zeros(N)
        inc_x = np.zeros(N)
        inc_y = np.zeros(N)
        delta = np.ones(N)
        S = np.ones(N)
        sig = np.ones(N) * 0.5
        mu = eta = 0.577350269189625764509149

        pydgm.sweeper_2d.computeeq(S, sig, delta, delta, mu, eta, inc_x, inc_y, Ps)

        np.testing.assert_array_almost_equal(inc_x, np.ones(N) * 0.711895505609929, 12)
        np.testing.assert_array_almost_equal(inc_y, np.ones(N) * 0.711895505609929, 12)
        np.testing.assert_array_almost_equal(Ps, np.ones(N) * 0.3559477528049645, 12)

    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()
