import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestMESH(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.spatial_dimension = 1
        pydgm.control.fine_mesh_x = [2, 4, 2]
        pydgm.control.coarse_mesh_x = [0.0, 1.0, 2.0, 3.0]
        pydgm.control.material_map = [1, 2, 3]

        # Initialize the dependancies
        pydgm.mesh.create_mesh()

    def test_mesh_create_mesh(self):
        ''' 
        Test the mesh initialization
        '''

        # Test the number of cells
        self.assertEqual(pydgm.control.number_cells, 8)

        # Test the cell size
        dx_test = [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5]
        np.testing.assert_array_equal(pydgm.mesh.dx, dx_test)

        # Test the material assignment
        mMap_test = [1, 1, 2, 2, 2, 2, 3, 3]
        np.testing.assert_array_equal(pydgm.mesh.mmap, mMap_test)

        # Test the problem width
        self.assertEqual(pydgm.mesh.width_x, 3.0)

    def tearDown(self):
        pydgm.mesh.finalize_mesh()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()
