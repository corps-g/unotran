import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestMESH(unittest.TestCase):

    def test_mesh_create_mesh_1D(self):
        ''' 
        Test the mesh initialization for 1D
        '''

        # Set the variables for the test
        pydgm.control.spatial_dimension = 1
        pydgm.control.fine_mesh_x = [2, 4, 2]
        pydgm.control.coarse_mesh_x = [0.0, 1.0, 2.0, 3.0]
        pydgm.control.material_map = [1, 2, 3]

        # Initialize the dependancies
        pydgm.mesh.create_mesh()

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

    def test_mesh_create_mesh_2D(self):
        ''' 
        Test the mesh initialization for 2D
        '''

        # Set the variables for the test
        pydgm.control.spatial_dimension = 2
        pydgm.control.fine_mesh_x = [5, 5, 3]
        pydgm.control.fine_mesh_y = [4, 4, 2]
        pydgm.control.coarse_mesh_x = [0.0, 21.42, 42.84, 64.26]
        pydgm.control.coarse_mesh_y = [0.0, 21.42, 42.84, 64.26]
        pydgm.control.material_map = [2, 4, 5,
                                      4, 2, 5,
                                      5, 5, 5]

        # Initialize the dependancies
        pydgm.mesh.create_mesh()

        # Test the number of cells
        self.assertEqual(pydgm.control.number_cells, 130)
        self.assertEqual(pydgm.control.number_cells_x, 13)
        self.assertEqual(pydgm.control.number_cells_y, 10)

        # Test the cell size
        dx_test = [4.284, 4.284, 4.284, 4.284, 4.284, 4.284, 4.284, 4.284, 4.284, 4.284, 7.14, 7.14, 7.14]
        dy_test = [5.355, 5.355, 5.355, 5.355, 5.355, 5.355, 5.355, 5.355, 10.71, 10.71, 10.71]
        np.testing.assert_array_almost_equal(pydgm.mesh.dx, dx_test, 12)
        np.testing.assert_array_almost_equal(pydgm.mesh.dx, dx_test, 12)

        # Test the material assignment
        mMap_test = [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 5, 5, 5,
                     2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 5, 5, 5,
                     2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 5, 5, 5,
                     2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 5, 5, 5,
                     4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 5, 5, 5,
                     4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 5, 5, 5,
                     4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 5, 5, 5,
                     4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 5, 5, 5,
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        np.testing.assert_array_equal(pydgm.mesh.mmap, mMap_test)

        # Test the problem width
        self.assertEqual(pydgm.mesh.width_x, 64.26)
        self.assertEqual(pydgm.mesh.width_y, 64.26)

    def tearDown(self):
        pydgm.mesh.finalize_mesh()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()
