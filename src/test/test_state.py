import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np

class TestSTATE(unittest.TestCase):
    
    def setUp(self):
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test.anlxs'
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = True
        pydgm.control.energy_group_map = [4]
        #pydgm.control.dgm_basis_name = 'basis'
        pydgm.control.outer_print = True
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-12
        pydgm.control.inner_tolerance = 1e-12
        pydgm.control.Lambda = 0.5
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = False
        pydgm.control.use_recondensation = False
        #pydgm.control.solver_type = 'fixed'
    
    def test_state_initialize(self):
        ''' 
        Test the function used to create the legendre polynomials
        '''
        
        pydgm.mesh.create_mesh()
        pydgm.material.create_material()
        
    def tearDown(self):
        pydgm.angle.finalize_angle()
        
if __name__ == '__main__':
    
    unittest.main()

