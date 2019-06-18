import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestSOURCES(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.spatial_dimension = 1
        pydgm.control.fine_mesh_x = [1]
        pydgm.control.coarse_mesh_x = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/3gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.equation_type = 'DD'
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = False
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.scatter_leg_order = 0
        pydgm.control.use_dgm = False

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

    def test_compute_source(self):
        test = [0.652227008050471, 0.500051646579093, 0.5]
        pydgm.sources.compute_source()

        for g in range(3):
            with self.subTest(g=g):
                source = pydgm.state.mg_source[g]
                np.testing.assert_array_almost_equal(source, test[g], 12, 'Failed for g={}'.format(g + 1))

    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()


class TestSOURCES_2D(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.spatial_dimension = 2
        pydgm.control.fine_mesh_x = [1]
        pydgm.control.fine_mesh_y = [1]
        pydgm.control.coarse_mesh_x = [0.0, 1.0]
        pydgm.control.coarse_mesh_y = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 0.0
        pydgm.control.boundary_north = 0.0
        pydgm.control.boundary_south = 0.0
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.equation_type = 'DD'
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = False
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.scatter_leg_order = 1
        pydgm.control.use_dgm = False

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

    def test_compute_source(self):

        test = [(1.0 + 0.5) / (2 * np.pi)]
        pydgm.sources.compute_source()

        source = pydgm.state.mg_source[0]
        np.testing.assert_array_almost_equal(source, test, 12)

        test = np.array([0.3, 0.2, 0.2, 0.2]) / (2 * np.pi)
        np.testing.assert_array_almost_equal(pydgm.state.sigphi.flatten(), test, 12)

    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()


class TestSOURCESdgm(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.spatial_dimension = 1
        pydgm.control.fine_mesh_x = [1]
        pydgm.control.coarse_mesh_x = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.equation_type = 'DD'
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.scatter_leg_order = 0
        pydgm.control.use_dgm = True
        pydgm.control.xs_name = 'test/4gXS.anlxs'.ljust(256)
        pydgm.control.energy_group_map = [1, 1, 2, 2]
        pydgm.control.dgm_basis_name = 'test/4gbasis'.ljust(256)

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()
        pydgm.dgmsolver.compute_flux_moments()
        pydgm.state.mg_phi = pydgm.dgm.phi_m[0]
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(0)
        pydgm.state.update_fission_density()

    def test_compute_source(self):
        test = [0.727975095456, 0.707343815354]
        pydgm.sources.compute_source()
        for g in range(2):
            with self.subTest(g=g):
                source = pydgm.state.mg_source[g]
                np.testing.assert_array_almost_equal(source, test[g], 12, 'Failed for g={}'.format(g + 1))

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()
