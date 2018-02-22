import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestMG_SOLVER(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/3gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = False
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-16
        pydgm.control.inner_tolerance = 1e-16
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.legendre_order = 0
        pydgm.control.max_inner_iters = 10
        pydgm.control.max_outer_iters = 2000

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

    def test_mg_solver_mg_solve_2_loop(self):
        ''' 
        Test convergence for the multigroup solver with limited iterations
        '''
        
        pydgm.control.max_outer_iters = 2
        pydgm.control.max_inner_iters = 10

        nG = pydgm.material.number_groups
        source = pydgm.state.d_source
        phi = pydgm.state.d_phi
        psi = pydgm.state.d_psi
        incident = pydgm.state.d_incoming

        pydgm.mg_solver.mg_solve(nG, source, phi, psi, incident)
        
        phi_test = [33.7869910712284636, 16.21623617487068, 6.7810024622273959]
        incident_test = np.array([[16.7848315072878371, 17.4223043071873391],
                                  [8.1244096150619072, 8.2065674159132111],
                                  [3.3905426593073713, 3.3937458743022715]])
        
        np.testing.assert_array_almost_equal(phi.flatten(), phi_test, 12)
        np.testing.assert_array_almost_equal(incident.flatten(), incident_test.flatten('F'), 12)

    def test_mg_solver_mg_solve_R(self):
        ''' 
        Test convergence for the multigroup solver with reflective conditions
        '''
        
        pydgm.control.max_inner_iters = 10

        nG = pydgm.material.number_groups
        source = pydgm.state.d_source
        phi = pydgm.state.d_phi
        psi = pydgm.state.d_psi
        incident = pydgm.state.d_incoming

        pydgm.mg_solver.mg_solve(nG, source, phi, psi, incident)
        
        phi_test = [161.534959460539, 25.4529297193052813, 6.9146161770064944]
        incident_test = np.array([[80.7674797302686329, 80.7674797302685761],
                                  [12.7264648596526406, 12.7264648596526424],
                                  [3.4573080885032477, 3.4573080885032477]])
        
        np.testing.assert_array_almost_equal(phi.flatten(), phi_test, 12)
        np.testing.assert_array_almost_equal(incident.flatten(), incident_test.flatten('F'), 12)

    def test_mg_solver_mg_solve_V(self):
        ''' 
        Test convergence for the multigroup solver with reflective conditions
        '''
        
        pydgm.control.max_inner_iters = 10
        pydgm.control.boundary_type = [0.0, 0.0]

        nG = pydgm.material.number_groups
        source = pydgm.state.d_source
        phi = pydgm.state.d_phi
        psi = pydgm.state.d_psi
        incident = pydgm.state.d_incoming

        pydgm.mg_solver.mg_solve(nG, source, phi, psi, incident)
        
        phi_test = [1.1128139420344907, 1.0469097961254414, 0.9493149657672653]
        incident_test = np.array([[0.6521165715780991, 1.3585503570955177],
                                  [0.6642068017193753, 1.2510439369070607],
                                  [0.589237898650032, 1.1413804154385336]])
        
        np.testing.assert_array_almost_equal(phi.flatten(), phi_test, 12)
        np.testing.assert_array_almost_equal(incident.flatten(), incident_test.flatten('F'), 12)

    def test_mg_solver_compute_source(self):
        ''' 
        Test the computation for the source
        '''

        phi = np.ones((1, 1, 3), order='F')
        source = np.ones((1, 4, 3), order='F') * 0.5
        
        source_test = np.array([0.5, 0.50058041595, 0.5007291251])

        for g in range(3):
            pydgm.mg_solver.compute_source(g + 1, 3, phi, source[:,:,g])
            np.testing.assert_array_almost_equal(source[:,:,g].flatten(), np.ones(4) * source_test[g], 12)

    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()

