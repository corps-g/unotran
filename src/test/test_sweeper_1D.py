import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestSWEEPER_1D(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.spatial_dimension = 1
        pydgm.control.fine_mesh_x = [1]
        pydgm.control.coarse_mesh_x = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/3gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.equation_type = 'DD'
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = False
        pydgm.control.scatter_leg_order = 0
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.legendre_order = 0
        pydgm.control.use_dgm = False

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

    def test_sweeper_computeEQ_dd(self):
        ''' 
        Test diamond difference equation
        '''
        pydgm.control.equation_type = 'DD'

        N = 8
        Ps = np.zeros(N)
        inc = np.zeros(N)
        S = np.ones(N)
        sig = np.ones(N)

        pydgm.sweeper_1d.computeeq(S, sig, 1.0, 0.8611363115940526, inc, Ps)
        np.testing.assert_array_almost_equal(inc, np.ones(N) * 0.734680275209795978, 12)
        np.testing.assert_array_almost_equal(Ps, np.ones(N) * 0.367340137604897989, 12)

    def test_sweeper_computeEQ_sc(self):
        ''' 
        Test the step characteristics equation
        '''
        pydgm.control.equation_type = 'SC'

        N = 8
        Ps = np.zeros(N)
        inc = np.zeros(N)
        S = np.ones(N)
        sig = np.ones(N)

        pydgm.sweeper_1d.computeeq(S, sig, 1.0, 0.8611363115940526, inc, Ps)
        np.testing.assert_array_almost_equal(inc, np.ones(N) * 0.686907416523104323, 12)
        np.testing.assert_array_almost_equal(Ps, np.ones(N) * 0.408479080928661801, 12)

    def test_sweeper_computeEQ_sd(self):
        ''' 
        Test the step difference equation
        '''
        pydgm.control.equation_type = 'SD'

        N = 8
        Ps = np.zeros(N)
        inc = np.zeros(N)
        S = np.ones(N)
        sig = np.ones(N)

        pydgm.sweeper_1d.computeeq(S, sig, 1.0, 0.8611363115940526, inc, Ps)
        np.testing.assert_array_almost_equal(inc, np.ones(N) * 0.5373061574106336, 12)
        np.testing.assert_array_almost_equal(Ps, np.ones(N) * 0.5373061574106336, 12)

    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()
