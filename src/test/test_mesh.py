import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np

class TestMESH(unittest.TestCase):
    
    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [2, 4, 2]
        pydgm.control.coarse_mesh = [0.0, 1.0, 2.0, 3.0]
        pydgm.control.material_map = [1, 2, 3]
        s = 'test.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 10
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = False
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.Lambda = 1.0
        pydgm.control.store_psi = True
        s = 'fixed'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        
        # Initialize the dependancies
        pydgm.mesh.create_mesh()
    
    def test_angle_initialization_mesh(self):
        ''' 
        Test the mesh initialization
        '''
        
        # Test the number of cells
        self.assertEqual(pydgm.mesh.number_cells, 8)
        
        # Test the cell size
        dx_test = [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5]
        np.testing.assert_array_equal(pydgm.mesh.dx, dx_test)
        
        # Test the material assignment
        mMap_test = [1, 1, 2, 2, 2, 2, 3, 3]
        np.testing.assert_array_equal(pydgm.mesh.mmap, mMap_test)
        
        # Test the problem width
        self.assertEqual(pydgm.mesh.width, 3.0)
  
        
    def tearDown(self):
        pydgm.angle.finalize_angle()
        
if __name__ == '__main__':
    
    unittest.main()

