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
        s = 'test.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.Lambda = 1.0
        pydgm.control.store_psi = True
        pydgm.control.use_dgm = True
        pydgm.control.energy_group_map = [4]
        pydgm.control.max_inner_iters = 500
        pydgm.control.ignore_warnings = True
        pydgm.control.use_recondensation = False
        s = 'eigen'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        s = 'basis'
        pydgm.control.dgm_basis_name = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 0.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 0

        pydgm.dgmsolver.initialize_dgmsolver()

    def test_dgmsweeper_innersolver(self):
        ''' 
        Test the inner solver in the dgm sweeper routine
        '''

        keff_test = 0.9565644234498036
        phi_test = np.array([0.1020504930641289, 3.811547533557685, 2.864052634383366, 0.21526881958866095, 0.006948383972629901, 0.0001320419273584034, 9.350617154876036e-08])
        phi_m = np.reshape(np.zeros(20), (1, 2, 10), 'F')
        psi_m = np.reshape(np.zeros(80), (2, 4, 10), 'F')

        # Provide the values for the scalar and angular flux
        for c in range(pydgm.mesh.number_cells):
            pydgm.state.phi[0, :, c] = phi_test
            for a in range(pydgm.angle.number_angles * 2):
                pydgm.state.psi[:, a, c] = phi_test / 2

        # Solve the problem
        pydgm.dgm.compute_flux_moments()
        phi_m_test = np.array([3.4964597402969204, 0.13591713796652238, -3.1791404276441306, -0.6609152883071883, 0.004087939785148761, 0.00491318330648654, 0.0027288922698201013, 0.0]).reshape((2, -1))

        for i in range(4):
            incoming = self.getIncoming(i)
            pydgm.dgm.compute_xs_moments(order=i)
            pydgm.dgmsweeper.inner_solve(i, incoming, phi_m, psi_m)
            np.testing.assert_array_almost_equal(phi_m[0, :, 0], phi_m_test[:, i], 12, 'order {} failed'.format(i))

    def getIncoming(self, o):
        incoming = np.reshape(np.zeros(4), (2, 2), 'F')

        for a in range(pydgm.angle.number_angles):
            for g in range(pydgm.material.number_groups):
                cg = pydgm.dgm.energymesh[g]
                incoming[cg - 1, a] += pydgm.dgm.basis[g, o] * pydgm.state.psi[g, a, 0]
        return incoming

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()

if __name__ == '__main__':

    unittest.main()

