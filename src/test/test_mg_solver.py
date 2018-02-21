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
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.legendre_order = 0
        
        # Initialize the dependancies
        pydgm.solver.initialize_solver()
    
    def test_mg_solver_mg_solve(self):
        ''' 
        Test convergence for the multigroup solver
        '''
        
        nG = pydgm.material.number_groups
        source = pydgm.state.d_source
        phi = pydgm.state.d_phi
        psi = pydgm.state.d_psi
        incident = pydgm.state.d_incoming
        
        pydgm.mg_solver.mg_solve(nG, source, phi, psi, incident)
        
        print phi
        print psi
        
        
        

    def test_mg_solver_compute_source(self):
        ''' 
        Test the computation for the source
        '''
        
        raise NotImplementedError 
        
    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()
        
if __name__ == '__main__':
    
    unittest.main()

