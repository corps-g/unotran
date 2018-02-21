import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np

class TestWG_SOLVER(unittest.TestCase):
    
    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = False
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.legendre_order = 0
        
        # Initialize the dependancies
        pydgm.solver.initialize_solver()
    
    def test_wg_solver_wg_solve(self):
        ''' 
        Test convergence for the within group solver
        '''
        
        raise NotImplementedError 

    def test_wg_solver_compute_in_scattering(self):
        ''' 
        Test the computation for the within group scattering source
        '''
        
        raise NotImplementedError 
        
    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()
        
if __name__ == '__main__':
    
    unittest.main()

