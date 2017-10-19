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
        s = 'test.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = True
        pydgm.control.energy_group_map = [4]
        s = 'basis'
        pydgm.control.dgm_basis_name = s + ' ' * (256 - len(s))
        pydgm.control.outer_print = True
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-12
        pydgm.control.inner_tolerance = 1e-12
        pydgm.control.Lambda = 0.5
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = False
        pydgm.control.use_recondensation = False
        s = 'fixed'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 0.0
        
        # Initialize the dependancies
        pydgm.mesh.create_mesh()
        pydgm.material.create_material()
        pydgm.angle.initialize_angle()
        pydgm.angle.initialize_polynomials(pydgm.material.number_legendre)
    
    def test_state_initialize(self):
        ''' 
        Test initializing of state arrays, no psi
        '''
        pydgm.state.initialize_state()
        
        phi_test = np.ones((7, 1))
        source_test = np.zeros((7, 4, 1))
        
        np.testing.assert_array_almost_equal(pydgm.state.phi[0], phi_test)
        np.testing.assert_array_almost_equal(pydgm.state.source, source_test)
        
    def test_state_initialize2(self):
        ''' 
        Test initializing of state arrays, with psi
        '''
        pydgm.control.store_psi = True
        pydgm.state.initialize_state()
        
        phi_test = np.ones((7, 1))
        psi_test = np.ones((7, 4, 1))
        source_test = np.zeros((7, 4, 1))
        
        np.testing.assert_array_almost_equal(pydgm.state.phi[0], phi_test)
        np.testing.assert_array_almost_equal(pydgm.state.psi, psi_test)
        np.testing.assert_array_almost_equal(pydgm.state.source, source_test)
        
    def tearDown(self):
        # Finalize the dependancies
        pydgm.mesh.finalize_mesh()
        pydgm.material.finalize_material()
        pydgm.angle.finalize_angle()
        pydgm.state.finalize_state()
        
if __name__ == '__main__':
    
    unittest.main()

