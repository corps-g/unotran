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
        pydgm.control.boundary_type = [0.0, 0.0]
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
        pydgm.control.boundary_type = [1.0, 1.0]
        
        phi = np.ones((1, 1, 3), order='F')
        psi = np.ones((1, 4, 3), order='F') * 0.5
        source = np.ones((1, 4, 3), order='F') * 0.5
        incident = np.ones((2, 3), order='F') * 0.5
        
        phi_test = np.array([2.4702649838962234, 2.0959331638532706, 2.0499030092048951])
        incident_test = np.array([[1.3519854437737708, 1.9598760493672542],
                                  [1.1652460559871114, 1.4317589997573743],
                                  [1.1306722250125505, 1.4342562108392354]])
        psi_test = np.array([[0.74789724066688767, 1.0164427642376821, 1.1738899625537731, 1.7463807889213092],
                             [0.71457173895342729, 0.90414923839142525, 1.047194766946983, 1.3700287382701124],
                             [0.69725052556455225, 0.88208317443622986, 1.0125866380708275, 1.3492112798558475]])
        
        for g in range(3):

            pydgm.wg_solver.wg_solve(g + 1, source[:,:,g], phi[:,:,g], psi[:,:,g], incident[:,g])
            
            np.testing.assert_array_almost_equal(phi[:,:,g].flatten(), phi_test[g], 12, 'group {} is wrong'.format(g))
            np.testing.assert_array_almost_equal(psi[:,:,g].flatten(), psi_test[g], 12, 'group {} is wrong'.format(g))
            np.testing.assert_array_almost_equal(incident[:,g].flatten(), incident_test[g], 12, 'group {} is wrong'.format(g))

    def test_wg_solver_wg_solve_10_loop(self):
        ''' 
        Test convergence for the within group solver
        '''
        pydgm.control.max_inner_iters = 10
        pydgm.control.boundary_type = [1.0, 1.0]
        
        phi = np.ones((1, 1, 3), order='F')
        psi = np.ones((1, 4, 3), order='F') * 0.5
        source = np.ones((1, 4, 3), order='F') * 0.5
        incident = np.ones((2, 3), order='F') * 0.5
        
        phi_test = np.array([18.2411679339726867, 10.5756995784462422, 6.1925362263931918])
        incident_test = np.array([[8.9987469448852977, 9.7137175868117875],
                                  [5.3169120951712197, 5.4634759411925238],
                                  [3.096716697732711, 3.1314174651125835]])
        psi_test = np.array([[8.4110546157322599, 9.1833418653517462, 8.8254852462930931, 9.5936956365879666],
                             [5.0752325933222497, 5.2668991482206478, 5.2536056771005519, 5.4404764864419803],
                             [3.0503289650786818, 3.0973688798583501, 3.0840783765680735, 3.1261733488806658]])
        
        for g in range(3):

            pydgm.wg_solver.wg_solve(g + 1, source[:,:,g], phi[:,:,g], psi[:,:,g], incident[:,g])
            
            np.testing.assert_array_almost_equal(phi[:,:,g].flatten(), phi_test[g], 12, 'group {} is wrong'.format(g))
            np.testing.assert_array_almost_equal(psi[:,:,g].flatten(), psi_test[g], 12, 'group {} is wrong'.format(g))
            np.testing.assert_array_almost_equal(incident[:,g].flatten(), incident_test[g], 12, 'group {} is wrong'.format(g))

    def test_wg_solver_compute_in_scattering(self):
        ''' 
        Test the computation for the within group scattering source
        '''
        
        phi = np.ones((1, 10), order='F')
        
        source_test = [0.63800764465, 0.7266715137, 0.6411932185]
        
        for g in range(3):
            source = np.ones((1, 4), order='F') * 0.5
            pydgm.wg_solver.compute_within_scattering(g + 1, phi, source)
            
            np.testing.assert_array_almost_equal(source, np.ones((1,4), order='F') * source_test[g], 12)
            
    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()
        
if __name__ == '__main__':
    
    unittest.main()

