import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np

class TestDGMSWEEPER(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
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
        pydgm.control.max_inner_iters = 5000
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
        phi_m = np.reshape(np.zeros(20), (1, 2, 10), 'F')
        psi_m = np.reshape(np.zeros(80), (2, 4, 10), 'F')

        # Provide the values for the scalar and angular flux
        for c in range(pydgm.mesh.number_cells):
            pydgm.state.phi[0, :, c] = phi_test
            for a in range(pydgm.angle.number_angles * 2):
                pydgm.state.psi[:, a, c] = phi_test

        # Solve the problem
        pydgm.dgm.compute_flux_moments()
        phi_m_test = np.concatenate((np.loadtxt('test/7gbasis').T.dot(phi_test), [0])).reshape((2, -1))

        for i in range(4):
            pydgm.dgm.compute_incoming_flux(order=i)
            pydgm.dgm.compute_xs_moments(order=i)
            pydgm.dgmsweeper.inner_solve(i, phi_m, psi_m)
            p1 = phi_m[0, :, 0]
            p2 = phi_m_test[:, i]
            f1 = p1[1] / p1[0]
            f2 = p2[1] / p2[0]
            np.testing.assert_almost_equal(f1, f2, 12, 'order {} failed'.format(i))

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()

if __name__ == '__main__':

    unittest.main()

