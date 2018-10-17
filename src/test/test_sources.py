import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestSOURCES(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/3gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.equation_type = 'DD'
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = False
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.scatter_legendre_order = 0
        pydgm.control.use_dgm = False

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

    def test_compute_external(self):
        for g in range(3):
            source = pydgm.sources.compute_external(g + 1)
            np.testing.assert_array_almost_equal(source, 0.5, 12, 'Failed for g={}'.format(g + 1))

    def test_compute_scatter(self):
        test = [0.13800764465, 0.22725192965, 0.1419223436]
        for c in range(pydgm.control.number_cells):
            for a in range(pydgm.control.number_angles * 2):
                for g in range(3):
                    source = pydgm.sources.compute_scatter(g + 1, c + 1, a + 1)
                    np.testing.assert_array_almost_equal(source, test[g], 12, 'Failed for g={}'.format(g + 1))

    def test_compute_fission(self):
        test = [0.152227008050471, 0.0000516465790925616, 0.0]
        for c in range(pydgm.control.number_cells):
            for g in range(3):
                source = pydgm.sources.compute_fission(g + 1, c + 1)
                np.testing.assert_array_almost_equal(source, test[g], 12, 'Failed for g={} c={}'.format(g + 1, c + 1))

    def test_compute_source(self):
        test = [0.652227008050471, 0.500051646579093, 0.5]
        pydgm.sources.compute_source()

        for g in range(3):
            source = pydgm.state.mg_source[g]
            np.testing.assert_array_almost_equal(source, test[g], 12, 'Failed for g={}'.format(g + 1))

    def test_add_transport_sources(self):
        test = [0.7902346527, 0.727303576229, 0.6419223436]
        pydgm.sources.compute_source()
        for c in range(pydgm.control.number_cells):
            for a in range(pydgm.control.number_angles * 2):
                for g in range(3):
                    source = pydgm.sources.add_transport_sources(g + 1, c + 1, a + 1)
                    np.testing.assert_array_almost_equal(source, test[g], 12, 'Failed for g={}'.format(g + 1))

    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()


class TestSOURCESdgm(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
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
        pydgm.control.scatter_legendre_order = 0
        pydgm.control.use_dgm = True
        pydgm.control.xs_name = 'test/4gXS.anlxs'.ljust(256)
        pydgm.control.energy_group_map = [1, 1, 2, 2]
        pydgm.control.dgm_basis_name = 'test/4gbasis'.ljust(256)

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()
        pydgm.dgmsolver.compute_flux_moments()
        pydgm.state.mg_phi = pydgm.dgm.phi_m_zero
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(0)
        pydgm.state.update_fission_density()

    def test_compute_external(self):
        for g in range(2):
            source = pydgm.sources.compute_external(g + 1)
            np.testing.assert_array_almost_equal(source, 0.707106781187, 12, 'Failed for g={}'.format(g + 1))

    def test_compute_scatter(self):
        test = [0.109164720667, 0.24180894357]
        for c in range(pydgm.control.number_cells):
            for a in range(pydgm.control.number_angles * 2):
                for g in range(2):
                    source = pydgm.sources.compute_scatter(g + 1, c + 1, a + 1)
                    np.testing.assert_array_almost_equal(source, test[g], 12, 'Failed for g={}'.format(g + 1))

    def test_compute_fission(self):
        test = [0.02086831427, 0.000237034168]
        for c in range(pydgm.control.number_cells):
            for g in range(2):
                source = pydgm.sources.compute_fission(g + 1, c + 1)
                np.testing.assert_array_almost_equal(source, test[g], 12, 'Failed for g={} c={}'.format(g + 1, c + 1))

    def test_compute_delta(self):
        test = [0.0, 0.0]
        for c in range(pydgm.control.number_cells):
            for a in range(pydgm.control.number_angles * 2):
                for g in range(2):
                    source = pydgm.sources.compute_delta(g + 1, c + 1, a + 1)
                    np.testing.assert_array_almost_equal(source, test[g], 12, 'Failed for g={} c={}'.format(g + 1, c + 1))

    def test_compute_source(self):
        test = [0.727975095456, 0.707343815354]
        pydgm.sources.compute_source()
        for g in range(2):
            source = pydgm.state.mg_source[g]
            np.testing.assert_array_almost_equal(source, test[g], 12, 'Failed for g={}'.format(g + 1))

    def test_add_transport_sources(self):
        test = [0.837139816124, 0.949152758925]
        pydgm.sources.compute_source()
        for c in range(pydgm.control.number_cells):
            for a in range(pydgm.control.number_angles * 2):
                for g in range(2):
                    source = pydgm.sources.add_transport_sources(g + 1, c + 1, a + 1)
                    np.testing.assert_array_almost_equal(source, test[g], 12, 'Failed for g={}'.format(g + 1))

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()
