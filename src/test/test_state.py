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
        pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = True
        pydgm.control.energy_group_map = [4]
        pydgm.control.dgm_basis_name = 'test/7gbasis'.ljust(256)
        pydgm.control.outer_print = True
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 0.5
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = False
        pydgm.control.use_recondensation = False
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.legendre_order = 0
        
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
        
        np.testing.assert_array_almost_equal(pydgm.state.phi[0], phi_test, 12)
        np.testing.assert_array_almost_equal(pydgm.state.source, source_test, 12)
        
    def test_state_initialize2(self):
        ''' 
        Test initializing of state arrays, with psi
        '''
        pydgm.control.store_psi = True
        pydgm.state.initialize_state()
        
        phi_test = np.ones((7, 1))
        psi_test = np.ones((7, 4, 1)) / 2
        source_test = np.zeros((7, 4, 1))
        
        np.testing.assert_array_almost_equal(pydgm.state.phi[0], phi_test, 12)
        np.testing.assert_array_almost_equal(pydgm.state.psi, psi_test, 12)
        np.testing.assert_array_almost_equal(pydgm.state.source, source_test, 12)
        
    def tearDown(self):
        # Finalize the dependancies
        pydgm.mesh.finalize_mesh()
        pydgm.material.finalize_material()
        pydgm.angle.finalize_angle()
        pydgm.state.finalize_state()
        pydgm.control.finalize_control()
        
if __name__ == '__main__':
    
    unittest.main()

