import sys
from numpy.ma.testutils import assert_almost_equal
from bokeh.tests.test_driving import phi
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestDGMSOLVER(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.spatial_dimension = 1
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.recon_print = False
        pydgm.control.eigen_print = False
        pydgm.control.outer_print = False
        pydgm.control.recon_tolerance = 1e-14
        pydgm.control.eigen_tolerance = 1e-14
        pydgm.control.outer_tolerance = 1e-15
        pydgm.control.lamb = 1.0
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = True
        pydgm.control.equation_type = 'DD'
        pydgm.control.scatter_leg_order = 0
        pydgm.control.ignore_warnings = True

    # Define methods to set various variables for the tests

    def setGroups(self, G):
        if G == 2:
            pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)
            pydgm.control.dgm_basis_name = 'test/2gbasis'.ljust(256)
            pydgm.control.energy_group_map = [1, 1]
        elif G == 4:
            pydgm.control.xs_name = 'test/4gXS.anlxs'.ljust(256)
            pydgm.control.energy_group_map = [1, 1, 2, 2]
            pydgm.control.dgm_basis_name = 'test/4gbasis'.ljust(256)
        elif G == 7:
            pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
            pydgm.control.dgm_basis_name = 'test/7gbasis'.ljust(256)
            pydgm.control.energy_group_map = [1, 1, 1, 1, 2, 2, 2]

    def setSolver(self, solver):
        if solver == 'fixed':
            pydgm.control.solver_type = 'fixed'.ljust(256)
            pydgm.control.source_value = 1.0
            pydgm.control.allow_fission = False
            pydgm.control.max_recon_iters = 10000
            pydgm.control.max_eigen_iters = 1
            pydgm.control.max_outer_iters = 100000
        elif solver == 'eigen':
            pydgm.control.solver_type = 'eigen'.ljust(256)
            pydgm.control.source_value = 0.0
            pydgm.control.allow_fission = True
            pydgm.control.max_recon_iters = 10000
            pydgm.control.max_eigen_iters = 10000
            pydgm.control.max_outer_iters = 1

    def setMesh(self, mesh):
        if mesh.isdigit():
            N = int(mesh)
            pydgm.control.fine_mesh_x = [N]
            pydgm.control.coarse_mesh_x = [0.0, float(N)]
        elif mesh == 'coarse_pin':
            pydgm.control.fine_mesh_x = [3, 10, 3]
            pydgm.control.coarse_mesh_x = [0.0, 0.09, 1.17, 1.26]
        elif mesh == 'fine_pin':
            pydgm.control.fine_mesh_x = [3, 22, 3]
            pydgm.control.coarse_mesh_x = [0.0, 0.09, 1.17, 1.26]

    def setBoundary(self, bounds):
        if bounds == 'reflect':
            pydgm.control.boundary_east = 1.0
            pydgm.control.boundary_west = 1.0
        elif bounds == 'vacuum':
            pydgm.control.boundary_east = 0.0
            pydgm.control.boundary_west = 0.0

    def angular_test(self):
        nAngles = pydgm.control.number_angles_per_octant
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]

        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_dgmsolver_intialize_using_mass(self):
        self.setGroups(2)
        self.setSolver('eigen')
        self.setMesh('10')
        self.setBoundary('vacuum')
        pydgm.control.material_map = [1]

        sig_t = np.array([[1.0, 2.0], [1.0, 3.0]])
        vsig_f = np.array([[0.5, 0.5], [0.0, 0.0]])
        sig_s = np.array([[[0.3, 0.3],
                           [0.0, 0.3]],
                          [[0.8, 1.2],
                           [0.0, 1.2]]])
        chi = np.array([[1.0, 0.0], [1.0, 0.0]]).T
        basis = np.loadtxt('test/2gbasis')

        expansion_order = len(basis)
        number_materials = len(sig_t)
        number_cells = pydgm.control.fine_mesh_x[0]
        number_fine_groups = len(basis[0])
        number_coarse_groups = 1
        scatter_leg_order = 1

        expanded_sig_t = np.zeros((expansion_order, number_coarse_groups, number_materials, expansion_order), order='F')
        expanded_nu_sig_f = np.zeros((expansion_order, number_coarse_groups, number_materials), order='F')
        expanded_sig_s = np.zeros((expansion_order, scatter_leg_order, number_coarse_groups, number_coarse_groups, number_materials, expansion_order), order='F')
        chi_m = np.zeros((number_coarse_groups, number_cells, expansion_order), order='F')

        for i in range(expansion_order):
            for m in range(number_materials):
                for g in range(number_fine_groups):
                    cg = 0
                    expanded_sig_t[:, cg, m, i] += basis[i, g] * sig_t[m, g] * basis[:, g]

        for m in range(number_materials):
            for g in range(number_fine_groups):
                cg = 0
                expanded_nu_sig_f[:, cg, m] += vsig_f[m, g] * basis[:, g]

        for i in range(expansion_order):
            for m in range(number_materials):
                for g in range(number_fine_groups):
                    cg = 0
                    for gp in range(number_fine_groups):
                        cgp = 0
                        for l in range(scatter_leg_order):
                            expanded_sig_s[:, l, cgp, cg, m, i] += basis[i, g] * sig_s[m, gp, g] * basis[:, gp]

        for i in range(expansion_order):
            for g in range(number_fine_groups):
                cg = 0
                chi_m[cg, :, i] += basis[i, g] * chi[g, 0]

        pydgm.dgmsolver.initialize_dgmsolver_with_moments(2, expanded_sig_t, expanded_nu_sig_f, expanded_sig_s, chi_m)

        # Set the test flux
        phi_test = np.array([0.7263080826036219, 0.12171194697729938, 1.357489062141697, 0.2388759408761157, 1.8494817499319578, 0.32318764022244134, 2.199278050699694, 0.38550684315075284, 2.3812063412628075, 0.4169543421336097, 2.381206341262808, 0.41695434213360977, 2.1992780506996943, 0.38550684315075295, 1.8494817499319585, 0.3231876402224415, 1.3574890621416973, 0.23887594087611572, 0.7263080826036221, 0.12171194697729937])

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.keff, 0.8099523232983424, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_solve_orders_fission(self):
        '''
        Test order 0 returns the same value when given the converged input for fixed problem
        '''

        # Set the variables for the test
        self.setSolver('fixed')
        self.setGroups(7)
        self.setMesh('1')
        self.setBoundary('vacuum')

        pydgm.control.material_map = [1]
        pydgm.control.allow_fission = False
        nA = 2

        pydgm.dgmsolver.initialize_dgmsolver()

        ########################################################################
        phi = np.array([1.0270690018072897, 1.1299037448361107, 1.031220528952085, 1.0309270835964415, 1.0404782471236467, 1.6703756546880606, 0.220435842109856])
        psi = np.array([[0.28670426208182, 0.3356992956691126, 0.3449054812807308, 0.3534008341488156, 0.3580544322663831, 0.6250475242024148, 0.0981878157679874],
                        [0.6345259657784981, 0.6872354146444389, 0.6066643580008859, 0.6019079440169605, 0.6067485919732419, 0.9472768646717264, 0.1166347906435061],
                        [0.6345259657784981, 0.6872354146444389, 0.6066643580008859, 0.6019079440169605, 0.6067485919732419, 0.9472768646717264, 0.1166347906435061],
                        [0.28670426208182, 0.3356992956691126, 0.3449054812807308, 0.3534008341488156, 0.3580544322663831, 0.6250475242024148, 0.0981878157679874]])

        phi_m_test = np.array([[2.1095601795959631, 0.0194781579525075, -0.0515640941922323, -0.0670614070008202],
                               [1.6923809227259041, 0.5798575454457766, -0.8490899895663372, 0.]])
        psi_m_test = np.array([[[0.6603549365902395, -0.0467999863865106, -0.0202498403596039, -0.0087381098484839],
                                [0.6242829410728972, 0.1837534467300195, -0.3240890486311904, 0.]],
                               [[1.2651668412203918, 0.0398970701525067, -0.0287329314249332, -0.0467550965071546],
                                [0.9645561434964076, 0.3465627924733725, -0.4781282918931471, 0.]],
                               [[1.2651668412203918, 0.0398970701525067, -0.0287329314249332, -0.0467550965071546],
                                [0.9645561434964076, 0.3465627924733725, -0.4781282918931471, 0.]],
                               [[0.6603549365902395, -0.0467999863865106, -0.0202498403596039, -0.0087381098484839],
                                [0.6242829410728972, 0.1837534467300195, -0.3240890486311904, 0.]]])

        # Set the converged fluxes
        pydgm.state.phi[0, :, 0] = phi
        for a in range(2 * nA):
            pydgm.state.psi[:, a, 0] = psi[a]
        pydgm.state.keff = 1.0
        old_psi = pydgm.state.psi

        # Get the moments from the fluxes
        pydgm.dgmsolver.compute_flux_moments()

        order = 0
        pydgm.dgm.dgm_order = order
        pydgm.state.mg_phi = pydgm.dgm.phi_m[0]
        pydgm.state.mg_psi = pydgm.dgm.psi_m[0]

        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.mg_phi.flatten(), phi_m, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(pydgm.state.mg_psi[:, a, 0].flatten(), psi_m[a], 12)

        ########################################################################
        order = 1
        pydgm.dgm.dgm_order = order
        pydgm.state.mg_phi = pydgm.dgm.phi_m[0]
        pydgm.state.mg_psi = pydgm.dgm.psi_m[0]

        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.mg_phi.flatten(), phi_m, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(pydgm.state.mg_psi[:, a, 0].flatten(), psi_m[a], 12)

        ########################################################################
        order = 2
        pydgm.dgm.dgm_order = order
        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.mg_phi.flatten(), phi_m, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(pydgm.state.mg_psi[:, a, 0].flatten(), psi_m[a], 12)

        ########################################################################
        order = 3
        pydgm.dgm.dgm_order = order
        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.mg_phi.flatten(), phi_m, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(pydgm.state.mg_psi[:, a, 0].flatten(), psi_m[a], 12)

    def test_dgmsolver_solve_orders_fixed(self):
        '''
        Test order 0 returns the same value when given the converged input for fixed problem
        '''

        # Set the variables for the test
        self.setSolver('fixed')
        self.setGroups(7)
        self.setMesh('1')
        self.setBoundary('reflect')

        pydgm.control.material_map = [1]
        pydgm.control.allow_fission = False
        nA = 2

        pydgm.state.initialize_state()
        pydgm.state.mg_mmap = pydgm.control.homogenization_map
        pydgm.dgmsolver.compute_source_moments()

        ########################################################################
        order = 0
        pydgm.dgm.dgm_order = order
        # Set the converged fluxes
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi = np.linalg.solve((T - S), np.ones(7))
        pydgm.state.phi[0, :, 0] = phi
        for a in range(2 * nA):
            pydgm.state.psi[:, a, 0] = phi / 2.0
        pydgm.state.keff = 1.0
        old_psi = pydgm.state.psi

        # Get the moments from the fluxes
        pydgm.dgmsolver.compute_flux_moments()

        pydgm.state.mg_phi = pydgm.dgm.phi_m[0]
        pydgm.state.mg_psi = pydgm.dgm.psi_m[0]

        phi_m_test = np.array([46.0567816728045685, 39.9620014433207302])

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 1
        pydgm.dgm.dgm_order = order
        pydgm.dgm.phi_m[0] = pydgm.state.mg_phi
        pydgm.dgm.psi_m[0] = pydgm.state.mg_psi

        phi_m_test = np.array([-7.7591835637013871, 18.2829496616545661])

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 2
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-10.382535949686881, -23.8247979105656675])

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 3
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-7.4878268473063185, 0.0])

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

    def test_dgmsolver_solve_orders(self):
        '''
        Test order 0 returns the same value when given the converged input for eigen problem
        '''

        # Set the variables for the test
        self.setSolver('eigen')
        self.setGroups(7)
        self.setMesh('1')
        self.setBoundary('reflect')
        pydgm.control.material_map = [1]
        nA = 2

        pydgm.dgmsolver.initialize_dgmsolver()

        ########################################################################
        order = 0
        pydgm.dgm.dgm_order = order
        # Set the converged fluxes
        phi = np.array([0.198933535568562, 2.7231683533646702, 1.3986600409998782, 1.010361903429942, 0.8149441787223116, 0.8510697418684054, 0.00286224604623])
        pydgm.state.phi[0, :, 0] = phi
        for a in range(2 * nA):
            with self.subTest(a=a):
                pydgm.state.psi[:, a, 0] = phi / 2.0
        pydgm.state.keff = 1.0674868709852505
        old_psi = pydgm.state.psi

        # Get the moments from the fluxes
        pydgm.dgmsolver.compute_flux_moments()

        pydgm.state.mg_phi = pydgm.dgm.phi_m[0]
        pydgm.state.mg_psi = pydgm.dgm.psi_m[0]

        phi_m_test = np.array([2.6655619166815265, 0.9635261040519922])
        norm_frac = 2 / sum(phi_m_test)
        phi_m_test *= norm_frac

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        assert_almost_equal(pydgm.state.keff, 1.0674868709852505, 12)

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 1
        pydgm.dgm.dgm_order = order
        pydgm.dgm.phi_m[0] = pydgm.state.mg_phi
        pydgm.dgm.psi_m[0] = pydgm.state.mg_psi

        phi_m_test = np.array([-0.2481536345018054, 0.5742286414743346])
        phi_m_test *= norm_frac

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 2
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-1.4562664776830221, -0.3610274595244746])
        phi_m_test *= norm_frac

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 3
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-1.0699480859043353, 0.0])
        phi_m_test *= norm_frac

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

    def test_dgmsolver_solve_orders_eigen(self):
        '''
        Test order 0 returns the same value when given the converged input for eigen problem
        '''
        # Set the variables for the test
        self.setSolver('eigen')
        self.setGroups(7)
        pydgm.control.fine_mesh_x = [2, 1, 2]
        pydgm.control.coarse_mesh_x = [0.0, 5.0, 6.0, 11.0]
        pydgm.control.material_map = [1, 5, 3]
        self.setBoundary('vacuum')
        pydgm.control.angle_order = 4
        pydgm.control.xs_name = 'test/alt_7gXS.anlxs'.ljust(256)

        pydgm.control.recon_print = 0
        pydgm.control.eigen_print = 0
        pydgm.control.outer_print = 0
        pydgm.control.inner_print = 0

        pydgm.control.max_recon_iters = 1
        pydgm.control.max_eigen_iters = 1
        pydgm.control.max_outer_iters = 1
        pydgm.control.max_inner_iters = 1

        pydgm.dgmsolver.initialize_dgmsolver()

        ########################################################################
        order = 0
        pydgm.dgm.dgm_order = order
        # Set the converged fluxes
        phi = np.array([[[1.6242342173603628, 1.6758532156636183, 0.8405795956015387, 0.1625775747329378, 0.1563516098298166, 0.1085336308306435, 0.0620903836758187],
                         [2.6801570555785044, 3.0677447677828793, 1.449823985768667, 0.2681250437495656, 0.223283519933589, 0.1450135178081014, 0.0801799450539108],
                         [3.2427808456629887, 3.6253236761323215, 1.5191384960457082, 0.2657137576543584, 0.1710313947880435, 0.0951334377954927, 0.0425020905708747],
                         [3.078740985037656, 3.2970204075028375, 1.4972548403087522, 0.2419744554249777, 0.1399415224162651, 0.0804404829573202, 0.0429420231808826],
                         [2.025799210421367, 1.9237182726014126, 0.8984917270995029, 0.1359184813858208, 0.0691590845219909, 0.0395429438436025, 0.0228838012778573]]])
        psi = np.array([[[0.222079350493803, 0.367893748426737, 0.2235158737950531, 0.0432849735160482, 0.0463727895022187, 0.0355187175888715, 0.0228020647902731],
                         [0.2550513848467887, 0.4086506508949974, 0.2437587461588053, 0.0472221628586482, 0.0500292057938681, 0.0378826318416084, 0.0239178952309458],
                         [0.3382749040753765, 0.5005376956437221, 0.286799618688869, 0.0556029704482913, 0.0575482625391845, 0.0425783497502043, 0.0260285263773904],
                         [0.5750074909228001, 0.6987878823386922, 0.3690096305377114, 0.0716465574529971, 0.0710147551998868, 0.050472013809047, 0.029289670734657],
                         [1.0537788563447283, 0.9734181137831549, 0.4957923512162933, 0.0961929513181893, 0.0917160223956373, 0.0619693328895713, 0.0340926844459002],
                         [1.3255267239562842, 1.1718393856235512, 0.5467264435687115, 0.106076277173514, 0.0986556746148473, 0.0659669292501365, 0.0361998578074773],
                         [1.344743505152009, 1.275694892110874, 0.5780321815222638, 0.1109128207385084, 0.1003555995219249, 0.0669432975064532, 0.0368743872958025],
                         [1.3169150639303029, 1.3131431962945315, 0.5934134444747582, 0.112983521080786, 0.1006890422223745, 0.0671233256092146, 0.0370834493694713]],
                        [[0.6885553482025526, 1.089335039975068, 0.6069001661447114, 0.1149124812343551, 0.1081129453446089, 0.0743840699002204, 0.0426285473144128],
                         [0.7725107502337766, 1.172471051868116, 0.6392311814456784, 0.1209778691436543, 0.112132068182718, 0.0761388322283501, 0.0430146562230251],
                         [0.9634060077282145, 1.3324269934504407, 0.6954805007449582, 0.1314528956036147, 0.118327409011439, 0.0784415022220629, 0.0433112324654103],
                         [1.3418218650264861, 1.5478745404418306, 0.7556888528998005, 0.1422593096610216, 0.1224628337686979, 0.0787659469902519, 0.0426536374845071],
                         [1.729477941003646, 1.6974160643540823, 0.7442585512313288, 0.1383544390474219, 0.1120623770468576, 0.0714367487802417, 0.0391794498708677],
                         [1.7051696722305358, 1.751812196905169, 0.7582694132424415, 0.1369370337610382, 0.1046797020301904, 0.0665917118269666, 0.0368374928147647],
                         [1.5387508951279656, 1.7186389175341759, 0.7620051265765969, 0.135317553737312, 0.1007482443032926, 0.0640130246749766, 0.03564770613761],
                         [1.4364950427473613, 1.6792136928011256, 0.7579125860583991, 0.1336344485909219, 0.098604182325654, 0.0626567122928802, 0.0350590921713212]],
                        [[1.0637855120993567, 1.536007235058809, 0.7512967906309741, 0.1404630994676566, 0.1148056366649365, 0.0698763787160453, 0.0333956120236536],
                         [1.1772365412120178, 1.623439996483817, 0.7691016739211279, 0.1437011859524629, 0.1139120708369216, 0.0675556535748025, 0.0312946245257323],
                         [1.4167593249106138, 1.7678192897287501, 0.7810108616578451, 0.1455407088176891, 0.107675105342892, 0.0605018605823794, 0.026386507283148],
                         [1.7904939585505957, 1.89352053932841, 0.7181697463659802, 0.1317868631490471, 0.082793577541892, 0.0420585740741061, 0.0164782671802563],
                         [2.1207386621923865, 1.9637650446071278, 0.7212071824348782, 0.12113922754426, 0.0658334448866402, 0.034958078599678, 0.0145491386909295],
                         [1.7517468797848386, 1.8958105952103985, 0.798043960365745, 0.1300892526694405, 0.0723937875679986, 0.0403303303934091, 0.0189198483739351],
                         [1.467070365430227, 1.7596888439540126, 0.7918720366437686, 0.1285518126305569, 0.0733074589055889, 0.041586719950917, 0.0205120079897094],
                         [1.3289690484789758, 1.6715864407056662, 0.7760283202736972, 0.1259707834617701, 0.07292702807355, 0.041730732082634, 0.0210874065276817]],
                        [[1.3742390593834137, 1.6804507107489854, 0.7616179736173394, 0.1299769203242264, 0.0826519306543626, 0.0473735675866383, 0.0231585138707784],
                         [1.49144934721379, 1.7347522148427128, 0.7689492552815056, 0.1302419543122088, 0.0802672835477712, 0.0454900456157048, 0.0223847678848062],
                         [1.7148976537965404, 1.8047767702734685, 0.7718681965783076, 0.1284953242691218, 0.0749924134827408, 0.042038811478865, 0.0212660172635992],
                         [1.9823521863053744, 1.824301962488276, 0.7671582509538742, 0.1232633853173764, 0.0683865153826584, 0.0389703997168544, 0.0208888756568808],
                         [1.766407427859224, 1.725997639499484, 0.7874221794566647, 0.1232544708010276, 0.0685805464349633, 0.0397831546607478, 0.0217985791844457],
                         [1.2599720527733733, 1.49648379942603, 0.7295310940574015, 0.1149891560923702, 0.0655143483522853, 0.0382708203491909, 0.0214214755581922],
                         [1.0082273816890912, 1.3201786458724547, 0.6726845915892845, 0.1067770556462346, 0.0623342020910217, 0.0366918535944347, 0.0209491303191844],
                         [0.8979420792671616, 1.2278028414312567, 0.6395952779081429, 0.1019260121956693, 0.0603514150340045, 0.0356908055351527, 0.0206234908394634]],
                        [[1.4648855138220889, 1.4418670068930663, 0.623563195498159, 0.0946670238578062, 0.0444506293512766, 0.0243640916897312, 0.0133250773869966],
                         [1.5309495376107136, 1.4197218176391437, 0.6102323454639805, 0.0916160580458327, 0.0427338663841867, 0.0237604956146663, 0.0133235079710548],
                         [1.6038006670473024, 1.339896091825747, 0.5814786019156777, 0.0859079287167494, 0.0407949773027919, 0.0233252278905371, 0.0133720031593783],
                         [1.4598645444698681, 1.1444682238971755, 0.5306971842723684, 0.0785437229491508, 0.039713913866885, 0.0228212778870397, 0.0128190133645811],
                         [0.7372082396801695, 0.8130863049868733, 0.3980411128911099, 0.0611423034295212, 0.0323171553205731, 0.0184328283005457, 0.0106824936928838],
                         [0.4329105987588648, 0.5823444079707099, 0.3100178712779538, 0.0484740605658099, 0.0272440970357683, 0.0158582531655901, 0.0096058269236477],
                         [0.3261965462563942, 0.4754151246434862, 0.2637847730312079, 0.0416367506767501, 0.0242295879053059, 0.0142777111873078, 0.0088952657992601],
                         [0.2839554762443687, 0.4279896897990576, 0.2420050678428891, 0.0383706034304901, 0.0227129595199266, 0.0134676747135594, 0.0085151463676417]]])
        # psi /= (np.linalg.norm(psi) * 10)
        phi_new = phi * 0
        for a in range(pydgm.control.number_angles_per_octant):
            phi_new[0] += psi[:, a, :] * pydgm.angle.wt[a]
            phi_new[0] += psi[:, 2 * pydgm.control.number_angles_per_octant - a - 1, :] * pydgm.angle.wt[a]

        for c in range(5):
            for g in range(7):
                pydgm.state.phi[0, :, c] = phi[0, c]
                for a in range(8):
                    pydgm.state.psi[g, a, c] = psi[c, a, g]
        pydgm.state.keff = 0.33973731848126831
        old_psi = pydgm.state.psi

        # Get the moments from the fluxes
        pydgm.dgmsolver.compute_flux_moments()

        pydgm.state.mg_phi = pydgm.dgm.phi_m[0]
        pydgm.state.mg_psi = pydgm.dgm.psi_m[0]

        phi_m_test = pydgm.dgm.phi_m[0].flatten('F')
        phi_m_test /= np.linalg.norm(phi_m_test, 1) / 10

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, pydgm.control.number_angles_per_octant:, 0]
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten('F'), phi_m_test, 12)

        ########################################################################
        order = 1
        pydgm.dgm.dgm_order = order
        pydgm.dgm.phi_m[0] = pydgm.state.mg_phi
        pydgm.dgm.psi_m[0] = pydgm.state.mg_psi

        phi_m_test = np.array([0.66268605409797898, 1.1239769588944581, 1.4011457517310117, 1.3088156391543195, 0.84988298005049157, 3.7839914869954847E-002, 5.7447025802385317E-002, 5.1596378790218486E-002, 3.8939158159247110E-002, 1.8576596655769165E-002])

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, pydgm.control.number_angles_per_octant:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.T.flatten('F'), phi_m_test, 12)

        ########################################################################
        order = 2
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-0.20710920655711104, -0.44545552454860282, -0.46438347612912256, -0.41828263508757896, -0.18748642683048020, 3.1862102568187112E-004, 3.1141556263365915E-003, 5.3924924332473369E-003, 5.0995287080187754E-003, 3.0030380436572414E-003])

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, pydgm.control.number_angles_per_octant:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.T.flatten('F'), phi_m_test, 12)

        ########################################################################
        order = 3
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-0.13255187402833862, -0.30996650357216082, -0.42418668341792881, -0.32530149073950271, -0.15053175043041164, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000])

        pydgm.state.mg_incident_x = pydgm.dgm.psi_m[order, :, pydgm.control.number_angles_per_octant:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.T.flatten('F'), phi_m_test, 12)

    def test_dgmsolver_unfold_flux_moments(self):
        '''
        Test unfolding flux moments into the scalar and angular fluxes
        '''
        self.setGroups(7)
        self.setSolver('fixed')
        self.setMesh('1')
        pydgm.control.material_map = [1]
        self.setBoundary('reflect')

        pydgm.state.initialize_state()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))
        # Compute the moments directly
        basis = np.loadtxt('test/7gbasis').T
        phi_m = basis.dot(phi_test)
        phi_m.resize(2, 4)
        for i in range(4):
            pydgm.dgm.phi_m[i, 0, :, 0] = phi_m[:, i]

        # Assume infinite homogeneous media (isotropic flux)
        for a in range(4):
            pydgm.dgm.psi_m[:, :, a, :] = 0.5 * pydgm.dgm.phi_m[:, 0, :, :]

        pydgm.dgmsolver.unfold_flux_moments()

        np.testing.assert_array_almost_equal(pydgm.state.phi.flatten(), phi_test, 12)
        for a in range(4):
            with self.subTest(a=a):
                np.testing.assert_array_almost_equal(pydgm.state.psi[:, a, 0].flatten(), phi_test * 0.5)

    def test_dgmsolver_vacuum1(self):
        '''
        Test the 7g->2G dgm fixed source problem with vacuum boundary conditions

        Using pin cell geometry with 3 material regions

        with fission
        '''
        # Set the variables for the test
        self.setGroups(7)
        self.setSolver('fixed')
        self.setMesh('fine_pin')
        self.setBoundary('vacuum')
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.angle_order = 10
        pydgm.control.allow_fission = True
        pydgm.control.lamb = 0.6
        pydgm.control.source_value = 1.0

        pydgm.dgmsolver.initialize_dgmsolver()

        phi_test = [1.6274528794638465, 2.71530879612549, 1.461745652768521, 1.3458703902580473, 1.3383852126342237, 1.9786760428590306, 0.24916735316863525, 1.6799175379390339, 2.8045999684695797, 1.516872017690622, 1.3885229934177148, 1.3782095743929001, 2.051131534663419, 0.26873064494111804, 1.728788120766425, 2.883502682394886, 1.5639999234445578, 1.4246328795261316, 1.4121166958899956, 2.1173467066121874, 0.2724292532553828, 1.7839749586964595, 2.990483236041222, 1.6474286521554664, 1.5039752034511047, 1.4924425499449177, 2.3127049909257686, 0.25496633574011124, 1.8436202405517381, 3.122355600505027, 1.7601872542791979, 1.61813693117119, 1.6099652659907275, 2.60256939853679, 0.24873883482629144, 1.896225857094417, 3.2380762891116794, 1.8534459525081792, 1.7117690484677541, 1.7061424886519436, 2.831599567019092, 0.26081315241625463, 1.9421441425092316, 3.338662519105913, 1.9310368092514267, 1.789369188781964, 1.7857603538028388, 3.0201767784594478, 0.2667363594339594, 1.9816803882995633, 3.424961908919033, 1.9955392685572624, 1.853808027881203, 1.851843446016314, 3.1773523146671065, 0.27189861962890616, 2.0150973757748596, 3.4976972455932094, 2.0486999251118014, 1.9069365316531377, 1.9063232414331912, 3.307833351001605, 0.2755922553419729, 2.042616213943685, 3.5574620224059217, 2.0917047787489365, 1.9499600832109016, 1.9504462746103806, 3.4141788518658602, 0.27833708525534473, 2.0644181962111365, 3.604732065595381, 2.1253588124042495, 1.9836690960190415, 1.985023407898914, 3.497921277464179, 0.28030660972118154, 2.080646338594525, 3.6398748475310785, 2.150203190885212, 2.0085809732608575, 2.0105818574623395, 3.5600286331289643, 0.2816665790912415, 2.0914067095511766, 3.663158139593214, 2.16659102830272, 2.0250269209204395, 2.0274573320958647, 3.6011228563902344, 0.2825198396790823, 2.0967694470675315, 3.6747566727970047, 2.174734975618102, 2.033203922754008, 2.0358487486465924, 3.621580567528384, 0.28293121918903963, 2.0967694470675315, 3.6747566727970042, 2.1747349756181023, 2.033203922754008, 2.0358487486465924, 3.6215805675283836, 0.2829312191890396, 2.0914067095511766, 3.6631581395932136, 2.1665910283027205, 2.02502692092044, 2.0274573320958647, 3.6011228563902358, 0.2825198396790823, 2.080646338594525, 3.639874847531079, 2.150203190885212, 2.008580973260857, 2.01058185746234, 3.5600286331289652, 0.2816665790912415, 2.0644181962111365, 3.6047320655953805, 2.125358812404249, 1.9836690960190408, 1.985023407898914, 3.4979212774641804, 0.2803066097211815, 2.042616213943685, 3.5574620224059217, 2.0917047787489365, 1.9499600832109014, 1.9504462746103808, 3.4141788518658616, 0.2783370852553448, 2.01509737577486, 3.49769724559321, 2.0486999251118005, 1.9069365316531375, 1.9063232414331914, 3.3078333510016056, 0.27559225534197296, 1.981680388299563, 3.424961908919033, 1.9955392685572624, 1.8538080278812032, 1.8518434460163142, 3.1773523146671074, 0.27189861962890616, 1.9421441425092318, 3.338662519105913, 1.931036809251427, 1.7893691887819645, 1.7857603538028393, 3.020176778459449, 0.2667363594339594, 1.896225857094417, 3.2380762891116777, 1.8534459525081792, 1.7117690484677544, 1.706142488651944, 2.831599567019092, 0.2608131524162547, 1.8436202405517386, 3.122355600505027, 1.7601872542791974, 1.6181369311711902, 1.6099652659907278, 2.6025693985367897, 0.24873883482629144, 1.783974958696459, 2.990483236041223, 1.6474286521554669, 1.5039752034511054, 1.4924425499449177, 2.312704990925769, 0.2549663357401113, 1.7287881207664255, 2.883502682394885, 1.5639999234445578, 1.4246328795261323, 1.412116695889996, 2.117346706612188, 0.27242925325538286, 1.6799175379390343, 2.8045999684695793, 1.516872017690622, 1.388522993417715, 1.3782095743929004, 2.05113153466342, 0.26873064494111826, 1.6274528794638465, 2.7153087961254894, 1.4617456527685213, 1.3458703902580476, 1.3383852126342235, 1.978676042859031, 0.24916735316863528]
        pydgm.state.phi[0, :, :] = np.reshape(phi_test, (7, 28), 'F')

        pydgm.dgmsolver.dgmsolve()

        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_reflect1(self):
        '''
        Test the 7g->2G, fixed source, infinite medium problem

        Uses 3 spatial regions with the same material in each

        no fission
        '''
        # Set the variables for the test
        self.setGroups(7)
        self.setSolver('fixed')
        self.setMesh('fine_pin')
        self.setBoundary('reflect')
        pydgm.control.material_map = [1, 1, 1]
        pydgm.control.angle_order = 10
        pydgm.control.lamb = 0.4

        pydgm.state.initialize_state()
        pydgm.state.mg_mmap = pydgm.control.homogenization_map
        pydgm.dgmsolver.compute_source_moments()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))
        phi_test = np.array([phi_test for i in range(28)]).flatten()

        pydgm.dgmsolver.dgmsolve()

        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_vacuum2(self):
        '''
        Test the 7g->2G, fixed source problem with vacuum conditions

        Uses one spatial cell, no fission
        '''
        # Set the variables for the test
        self.setGroups(7)
        self.setSolver('fixed')
        self.setMesh('1')
        self.setBoundary('vacuum')
        pydgm.control.material_map = [1]
        pydgm.control.lamb = 0.9
        pydgm.control.allow_fission = True
        phi_test = np.array([1.0781901438738859, 1.5439788126739036, 1.0686290157458673, 1.0348940034466163, 1.0409956199943164, 1.670442207080332, 0.2204360523334687])
        # pydgm.control.recon_print = 1
        # pydgm.control.eigen_print = 1

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        pydgm.state.phi[0, :, 0] = phi_test
        for a in range(4):
            pydgm.state.psi[:, a, 0] = 0.5 * phi_test

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_reflect2(self):
        '''
        Test the 7g->2G, fixed source problem with infinite medium and one spatial cell
        '''

        # Set the variables for the test
        self.setGroups(7)
        self.setSolver('fixed')
        self.setMesh('1')
        self.setBoundary('reflect')
        pydgm.control.material_map = [1]
        pydgm.control.equation_type = 'DD'
        pydgm.control.lamb = 0.45

        # Initialize the dependancies
        pydgm.state.initialize_state()
        pydgm.state.mg_mmap = pydgm.control.homogenization_map
        pydgm.dgmsolver.compute_source_moments()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_eigenV2g(self):
        '''
        Test the 2g->1G, eigenvalue problem with 1 medium and vacuum conditions
        '''

        # Set the variables for the test
        self.setGroups(2)
        self.setSolver('eigen')
        self.setMesh('10')
        self.setBoundary('vacuum')
        pydgm.control.material_map = [1]

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        phi_test = np.array([0.7263080826036219, 0.12171194697729938, 1.357489062141697, 0.2388759408761157, 1.8494817499319578, 0.32318764022244134, 2.199278050699694, 0.38550684315075284, 2.3812063412628075, 0.4169543421336097, 2.381206341262808, 0.41695434213360977, 2.1992780506996943, 0.38550684315075295, 1.8494817499319585, 0.3231876402224415, 1.3574890621416973, 0.23887594087611572, 0.7263080826036221, 0.12171194697729937])

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.keff, 0.8099523232983424, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_eigenV4g(self):
        '''
        Test the 4g->2G, eigenvalue problem with 1 medium and vacuum conditions
        '''
        # Set the variables for the test
        self.setGroups(4)
        self.setSolver('eigen')
        self.setMesh('10')
        self.setBoundary('vacuum')
        pydgm.control.material_map = [1]

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        phi_test = np.array([1.6283945282803138, 1.2139688020213637, 0.03501217302426163, 6.910819115000905e-17, 1.9641731545343404, 1.59852298044156, 0.05024813427412045, 1.0389359842780806e-16, 2.2277908375593736, 1.8910978193073922, 0.061518351747482505, 1.3055885402420332e-16, 2.40920191588961, 2.088554929299159, 0.06902375359471126, 1.487795240822353e-16, 2.5016194254000244, 2.188087672560707, 0.0727855220655801, 1.5805185521208351e-16, 2.501619425400025, 2.1880876725607075, 0.07278552206558009, 1.5805185521208351e-16, 2.40920191588961, 2.088554929299159, 0.06902375359471127, 1.487795240822353e-16, 2.2277908375593736, 1.891097819307392, 0.0615183517474825, 1.3055885402420332e-16, 1.9641731545343404, 1.59852298044156, 0.05024813427412045, 1.0389359842780806e-16, 1.6283945282803138, 1.2139688020213637, 0.03501217302426163, 6.910819115000904e-17])

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.keff, 0.185134666261, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_eigenV7g(self):
        '''
        Test the 7g->2G, eigenvalue problem with 1 medium and vacuum conditions
        '''
        # Set the variables for the test
        self.setGroups(7)
        self.setSolver('eigen')
        self.setMesh('10')
        self.setBoundary('vacuum')
        pydgm.control.material_map = [1]
        pydgm.control.lamb = 0.45

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        phi_test = np.array([0.19050251326520584, 1.9799335510805185, 0.69201814518126, 0.3927000245492841, 0.2622715078950253, 0.20936059119838546, 0.000683954269595958, 0.25253653423327665, 2.8930819653774895, 1.158606945184528, 0.6858113244922716, 0.4639601075261923, 0.4060114930207368, 0.0013808859451732852, 0.30559047625122115, 3.6329637815416556, 1.498034484581793, 0.9026484213739354, 0.6162114941108023, 0.5517562407150877, 0.0018540270157502057, 0.3439534785160265, 4.153277746375052, 1.7302149163096785, 1.0513217539517374, 0.7215915434720093, 0.653666204542615, 0.0022067618449436725, 0.36402899896324237, 4.421934793951583, 1.8489909842118943, 1.127291245982061, 0.7756443978822711, 0.705581398687358, 0.0023773065003326204, 0.36402899896324237, 4.421934793951582, 1.8489909842118946, 1.1272912459820612, 0.7756443978822711, 0.705581398687358, 0.0023773065003326204, 0.34395347851602653, 4.153277746375052, 1.7302149163096785, 1.0513217539517377, 0.7215915434720092, 0.653666204542615, 0.002206761844943672, 0.3055904762512212, 3.6329637815416564, 1.498034484581793, 0.9026484213739353, 0.6162114941108023, 0.5517562407150877, 0.0018540270157502063, 0.2525365342332767, 2.8930819653774895, 1.1586069451845278, 0.6858113244922716, 0.4639601075261923, 0.4060114930207368, 0.0013808859451732852, 0.19050251326520584, 1.9799335510805192, 0.6920181451812601, 0.3927000245492842, 0.26227150789502535, 0.20936059119838543, 0.0006839542695959579])

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.keff, 0.30413628310914226, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        self.angular_test()

    # Test the eigenvalue solver for infinite media

    def test_dgmsolver_eigenR2g(self):
        '''
        Test the 2g->1G, eigenvalue problem with infinite medium
        '''
        # Set the variables for the test
        self.setGroups(2)
        self.setSolver('eigen')
        self.setMesh('10')
        self.setBoundary('reflect')
        pydgm.control.lamb = 0.74
        pydgm.control.material_map = [1]

        # Initialize the dependancies
        pydgm.state.initialize_state()
        pydgm.state.mg_mmap = pydgm.control.homogenization_map
        pydgm.dgmsolver.compute_source_moments()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        X = np.outer(pydgm.material.chi[:, 0], pydgm.material.nu_sig_f[:, 0])

        keff, phi = np.linalg.eig(np.linalg.inv(T - S).dot(X))

        i = np.argmax(keff)
        keff_test = keff[i]
        phi_test = phi[:, i]

        phi_test = np.array([phi_test for i in range(10)]).flatten()

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_eigenR4g(self):
        '''
        Test the 4g->2G, eigenvalue problem with infinite medium
        '''
        # Set the variables for the test
        self.setGroups(4)
        self.setSolver('eigen')
        self.setMesh('10')
        self.setBoundary('reflect')
        pydgm.control.lamb = 0.95
        pydgm.control.material_map = [1]

        # Initialize the dependancies
        pydgm.state.initialize_state()
        pydgm.state.mg_mmap = pydgm.control.homogenization_map
        pydgm.dgmsolver.compute_source_moments()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        X = np.outer(pydgm.material.chi[:, 0], pydgm.material.nu_sig_f[:, 0])

        keff, phi = np.linalg.eig(np.linalg.inv(T - S).dot(X))
        i = np.argmax(keff)
        keff_test = keff[i]
        phi_test = phi[:, i]

        phi_test = np.array([phi_test for i in range(10)]).flatten()

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_eigenR7g(self):
        '''
        Test the 7g->2G, eigenvalue problem with infinite medium
        '''
        # Set the variables for the test
        self.setGroups(7)
        self.setSolver('eigen')
        self.setMesh('10')
        self.setBoundary('reflect')
        pydgm.control.material_map = [1]
        pydgm.control.lamb = 0.46

        # Initialize the dependancies
        pydgm.state.initialize_state()
        pydgm.state.mg_mmap = pydgm.control.homogenization_map
        pydgm.dgmsolver.compute_source_moments()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        X = np.outer(pydgm.material.chi[:, 0], pydgm.material.nu_sig_f[:, 0])

        keff, phi = np.linalg.eig(np.linalg.inv(T - S).dot(X))
        keff = np.real(keff)
        i = np.argmax(keff)
        keff_test = keff[i]
        phi_test = phi[:, i]

        phi_test = np.array([phi_test for i in range(10)]).flatten()

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        self.angular_test()

    # Test the eigenvalue solver for pin cell like problems

    def test_dgmsolver_eigenR2gPin(self):
        '''
        Test the 2g->1G, eigenvalue problem on a pin cell of
            water | fuel | water
        with reflective conditions
        '''
        # Set the variables for the test
        self.setGroups(2)
        self.setSolver('eigen')
        self.setMesh('coarse_pin')
        self.setBoundary('reflect')
        pydgm.control.material_map = [2, 1, 2]
        pydgm.control.lamb = 0.7

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # set the test flux
        keff_test = 0.8418546852484950
        phi_test = [0.13393183108467394, 0.04663240631432256, 0.13407552941360298, 0.04550808086281801, 0.13436333428621713, 0.043206841474147446, 0.1351651393398092, 0.0384434752119791, 0.13615737742196526, 0.03329929560434661, 0.13674284660888314, 0.030464508103354708, 0.13706978363298242, 0.028970199506203023, 0.13721638515632006, 0.028325674662651124, 0.13721638515632006, 0.028325674662651124, 0.1370697836329824, 0.028970199506203012, 0.13674284660888308, 0.0304645081033547, 0.13615737742196524, 0.03329929560434659, 0.13516513933980914, 0.03844347521197908, 0.13436333428621713, 0.043206841474147425, 0.13407552941360296, 0.045508080862818004, 0.1339318310846739, 0.046632406314322555]

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_eigenR4gPin(self):
        '''
        Test the 4g->2G, eigenvalue problem on a pin cell of
            water | fuel | water
        with reflective conditions
        '''
        # Set the variables for the test
        self.setGroups(4)
        self.setSolver('eigen')
        self.setMesh('coarse_pin')
        self.setBoundary('reflect')
        pydgm.control.material_map = [2, 1, 2]
        pydgm.control.lamb = 1.4

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # set the test flux
        keff_test = 0.759180925837
        phi_test = [2.22727714687889, 1.7369075872008062, 0.03381777446256108, 1.7036045485771946e-51, 2.227312836950116, 1.737071399085861, 0.033828681689533874, 1.698195884848375e-51, 2.2273842137861877, 1.737399076914814, 0.03385050700298498, 1.6965081411970257e-51, 2.2275366771427696, 1.738049905609514, 0.033892498696060584, 1.697491400636173e-51, 2.2277285082016354, 1.7388423494826295, 0.03394281751690741, 1.6984869893367e-51, 2.2278725176984917, 1.7394359801217965, 0.03398044173136341, 1.697888141875072e-51, 2.2279685888050587, 1.739831399863755, 0.03400546998960266, 1.6957049057554318e-51, 2.2280166437743327, 1.7400290096083806, 0.034017967785934036, 1.691942073409801e-51, 2.2280166437743327, 1.7400290096083808, 0.03401796778593402, 1.68659920584123e-51, 2.2279685888050587, 1.7398313998637547, 0.03400546998960263, 1.6796706200591657e-51, 2.2278725176984917, 1.7394359801217967, 0.03398044173136335, 1.6711453402288656e-51, 2.227728508201635, 1.73884234948263, 0.033942817516907337, 1.6610070122205585e-51, 2.2275366771427696, 1.7380499056095144, 0.0338924986960605, 1.6492337810058256e-51, 2.227384213786188, 1.7373990769148142, 0.03385050700298487, 1.6385765949262272e-51, 2.2273128369501163, 1.7370713990858613, 0.03382868168953376, 1.631610014066153e-51, 2.2272771468788894, 1.7369075872008064, 0.03381777446256096, 1.6281341640813905e-51]

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 11)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_eigenR7gPin(self):
        '''
        Test the 7g->2G, eigenvalue problem on a pin cell of
            water | fuel | water
        with reflective conditions
        '''
        # Set the variables for the test
        self.setGroups(7)
        self.setSolver('eigen')
        self.setMesh('coarse_pin')
        self.setBoundary('reflect')
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.lamb = 0.42

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # set the test flux
        keff_test = 1.0794314789325041
        phi_test = [0.18617101855192203, 2.858915338372074, 1.4772041911943246, 1.0299947729491368, 0.7782291112252604, 0.6601323950057741, 0.0018780861364841711, 0.18615912584783736, 2.8586649211715436, 1.4771766639520822, 1.030040359498237, 0.7782969794725844, 0.6603312122972236, 0.0018857945539742516, 0.18613533094381535, 2.858163988201594, 1.4771216026067822, 1.0301315410919691, 0.7784327301908717, 0.6607289506819793, 0.001901260571677254, 0.1860688679764812, 2.856246121648244, 1.4768054633757053, 1.0308126366153867, 0.7795349485104974, 0.6645605858759833, 0.0020390343478796447, 0.18597801819761287, 2.8534095558345967, 1.4763058823653368, 1.0319062100766097, 0.7813236329806805, 0.6707834262470258, 0.002202198406120643, 0.18591115233580913, 2.851306298580461, 1.4759330573010532, 1.0327026586727457, 0.7826354049117061, 0.6752310599415209, 0.002251216654318464, 0.18586715682184884, 2.8499152893733255, 1.4756853599772022, 1.0332225362074143, 0.7834960048959739, 0.6780933062272468, 0.0022716374303459433, 0.1858453299832966, 2.8492230801887986, 1.475561761882258, 1.0334791905008067, 0.7839221795374745, 0.6794942839316992, 0.002280077669362046, 0.18584532998329656, 2.8492230801887986, 1.475561761882258, 1.0334791905008065, 0.7839221795374745, 0.6794942839316991, 0.0022800776693620455, 0.18586715682184884, 2.8499152893733255, 1.4756853599772024, 1.0332225362074146, 0.7834960048959738, 0.6780933062272467, 0.002271637430345943, 0.18591115233580915, 2.851306298580461, 1.4759330573010532, 1.0327026586727457, 0.7826354049117062, 0.6752310599415207, 0.0022512166543184635, 0.1859780181976129, 2.853409555834596, 1.476305882365337, 1.0319062100766097, 0.7813236329806805, 0.6707834262470258, 0.002202198406120643, 0.1860688679764812, 2.856246121648244, 1.4768054633757055, 1.0308126366153867, 0.7795349485104973, 0.6645605858759831, 0.002039034347879644, 0.18613533094381537, 2.858163988201594, 1.4771216026067824, 1.0301315410919691, 0.7784327301908716, 0.6607289506819792, 0.0019012605716772534, 0.1861591258478374, 2.858664921171543, 1.4771766639520822, 1.0300403594982372, 0.7782969794725842, 0.6603312122972235, 0.0018857945539742511, 0.18617101855192209, 2.8589153383720736, 1.477204191194325, 1.0299947729491368, 0.7782291112252603, 0.660132395005774, 0.0018780861364841707]

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 11)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_homogenize_xs_moments(self):
        '''
        Make sure that the cross sections are being properly homogenized
        '''
        self.setGroups(2)
        pydgm.control.dgm_basis_name = 'test/2gdelta'.ljust(256)
        pydgm.control.energy_group_map = [1, 2]
        self.setSolver('fixed')
        pydgm.control.allow_fission = True
        self.setBoundary('reflect')
        pydgm.control.fine_mesh_x = [2, 3, 3, 2]
        pydgm.control.coarse_mesh_x = [0.0, 1.0, 2.0, 4.0, 4.5]
        pydgm.control.material_map = [1, 2, 1, 2]
        pydgm.control.homogenization_map = [1, 1, 2, 3, 3, 3, 4, 4, 4, 5]

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()
        pydgm.dgmsolver.compute_flux_moments()
        pydgm.dgmsolver.compute_xs_moments()

        # Check sig_t
        sig_t_test = np.array([[1.0, 2.0],
                               [1.0, 3.0],
                               [1.0, 2.5],
                               [1.0, 2.1578947368421],
                               [1.0, 3.0]])
        np.testing.assert_array_almost_equal(pydgm.state.mg_sig_t, sig_t_test.T, 12)

        # Check nu_sig_f
        nu_sig_f_test = np.array([[0.5, 0.5],
                                  [0.0, 0.0],
                                  [0.25, 0.25],
                                  [0.4210526315789470, 0.4210526315789470],
                                  [0.0, 0.0]])
        np.testing.assert_array_almost_equal(pydgm.state.mg_nu_sig_f, nu_sig_f_test.T, 12)

        # Check sig_s
        sig_s_test = np.array([[[0.3, 0.8, 0.55, 0.378947368421053, 0.8],
                                [0.3, 1.2, 0.75, 0.442105263157895, 1.2]],
                               [[0., 0., 0., 0., 0.],
                                [0.3, 1.2, 0.75, 0.442105263157895, 1.2]]])

        np.testing.assert_array_almost_equal(pydgm.dgm.sig_s_m[0, :, :, :, 0], sig_s_test, 12)

    def test_dgmsolver_homogenize_xs_moments_2(self):
        '''
        Make sure that the cross sections are being properly homogenized
        '''
        self.setGroups(2)
        pydgm.control.dgm_basis_name = 'test/2gdelta'.ljust(256)
        pydgm.control.energy_group_map = [1, 2]
        self.setSolver('fixed')
        pydgm.control.allow_fission = True
        self.setBoundary('reflect')
        pydgm.control.fine_mesh_x = [2, 3, 3, 2]
        pydgm.control.coarse_mesh_x = [0.0, 1.0, 2.0, 4.0, 4.5]
        pydgm.control.material_map = [1, 2, 1, 2]
        pydgm.control.homogenization_map = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()
        pydgm.dgmsolver.compute_flux_moments()
        pydgm.dgmsolver.compute_xs_moments()

        # Check sig_t
        sig_t_test = np.array([[1.0, 2.44],
                               [1.0, 2.24137931034483]])
        np.testing.assert_array_almost_equal(pydgm.state.mg_sig_t, sig_t_test.T, 12)

        # Check nu_sig_f
        nu_sig_f_test = np.array([[0.28, 0.28],
                                  [0.3793103448275860, 0.3793103448275860]])
        np.testing.assert_array_almost_equal(pydgm.state.mg_nu_sig_f, nu_sig_f_test.T, 12)

        # Check sig_s
        sig_s_test = np.array([[[0.52, 0.4206896551724140],
                                [0.696, 0.5172413793103450]],
                               [[0.0, 0.0],
                                [0.696, 0.5172413793103450]]])
        np.testing.assert_array_almost_equal(pydgm.dgm.sig_s_m[0, :, :, :, 0], sig_s_test, 12)

    def test_dgmsolver_homogenize_xs_moments_3(self):
        '''
        Make sure that the cross sections are being properly homogenized
        '''
        self.setGroups(2)
        pydgm.control.dgm_basis_name = 'test/2gdelta'.ljust(256)
        pydgm.control.energy_group_map = [1, 2]
        self.setSolver('fixed')
        pydgm.control.allow_fission = True
        self.setBoundary('reflect')
        pydgm.control.fine_mesh_x = [2, 3, 3, 2]
        pydgm.control.coarse_mesh_x = [0.0, 1.0, 2.0, 4.0, 4.5]
        pydgm.control.material_map = [1, 2, 1, 2]
        pydgm.control.homogenization_map = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        pydgm.state.phi[0, 0, :] = range(1, 11)
        pydgm.state.phi[0, 1, :] = range(10, 0, -1)

        pydgm.dgmsolver.compute_flux_moments()
        pydgm.dgmsolver.compute_xs_moments()

        # Check sig_t
        sig_t_test = np.array([[1.0, 2.4025974025974000],
                               [1.0, 2.2080536912751700]])
        np.testing.assert_array_almost_equal(pydgm.state.mg_sig_t, sig_t_test.T, 12)

        # Check nu_sig_f
        nu_sig_f_test = np.array([[0.2561983471074380, 0.2987012987012990],
                                  [0.3647058823529410, 0.3959731543624160]])
        np.testing.assert_array_almost_equal(pydgm.state.mg_nu_sig_f, nu_sig_f_test.T, 12)

        # Check sig_s
        sig_s_test = np.array([[[0.5438016528925620, 0.4352941176470590],
                                [0.7388429752066120, 0.5435294117647060]],
                               [[0.0000000000000000, 0.0000000000000000],
                                [0.6623376623376620, 0.4872483221476510]]])
        np.testing.assert_array_almost_equal(pydgm.dgm.sig_s_m[0, :, :, :, 0], sig_s_test, 12)

    def test_dgmsolver_non_contiguous(self):
        '''
        Test the 7g->2G, eigenvalue problem on a pin cell of
            water | fuel | water
        with reflective conditions
        '''
        # Set the variables for the test
        self.setGroups(7)
        pydgm.control.energy_group_map = [1, 2, 1, 2, 1, 2, 1]
        pydgm.control.dgm_basis_name = 'test/7g_non_contig'.ljust(256)
        self.setSolver('eigen')
        self.setMesh('coarse_pin')
        self.setBoundary('reflect')
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.lamb = 0.27

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # set the test flux
        keff_test = 1.0794314789325041
        phi_test = [0.18617101855192203, 2.858915338372074, 1.4772041911943246, 1.0299947729491368, 0.7782291112252604, 0.6601323950057741, 0.0018780861364841711, 0.18615912584783736, 2.8586649211715436, 1.4771766639520822, 1.030040359498237, 0.7782969794725844, 0.6603312122972236, 0.0018857945539742516, 0.18613533094381535, 2.858163988201594, 1.4771216026067822, 1.0301315410919691, 0.7784327301908717, 0.6607289506819793, 0.001901260571677254, 0.1860688679764812, 2.856246121648244, 1.4768054633757053, 1.0308126366153867, 0.7795349485104974, 0.6645605858759833, 0.0020390343478796447, 0.18597801819761287, 2.8534095558345967, 1.4763058823653368, 1.0319062100766097, 0.7813236329806805, 0.6707834262470258, 0.002202198406120643, 0.18591115233580913, 2.851306298580461, 1.4759330573010532, 1.0327026586727457, 0.7826354049117061, 0.6752310599415209, 0.002251216654318464, 0.18586715682184884, 2.8499152893733255, 1.4756853599772022, 1.0332225362074143, 0.7834960048959739, 0.6780933062272468, 0.0022716374303459433, 0.1858453299832966, 2.8492230801887986, 1.475561761882258, 1.0334791905008067, 0.7839221795374745, 0.6794942839316992, 0.002280077669362046, 0.18584532998329656, 2.8492230801887986, 1.475561761882258, 1.0334791905008065, 0.7839221795374745, 0.6794942839316991, 0.0022800776693620455, 0.18586715682184884, 2.8499152893733255, 1.4756853599772024, 1.0332225362074146, 0.7834960048959738, 0.6780933062272467, 0.002271637430345943, 0.18591115233580915, 2.851306298580461, 1.4759330573010532, 1.0327026586727457, 0.7826354049117062, 0.6752310599415207, 0.0022512166543184635, 0.1859780181976129, 2.853409555834596, 1.476305882365337, 1.0319062100766097, 0.7813236329806805, 0.6707834262470258, 0.002202198406120643, 0.1860688679764812, 2.856246121648244, 1.4768054633757055, 1.0308126366153867, 0.7795349485104973, 0.6645605858759831, 0.002039034347879644, 0.18613533094381537, 2.858163988201594, 1.4771216026067824, 1.0301315410919691, 0.7784327301908716, 0.6607289506819792, 0.0019012605716772534, 0.1861591258478374, 2.858664921171543, 1.4771766639520822, 1.0300403594982372, 0.7782969794725842, 0.6603312122972235, 0.0018857945539742511, 0.18617101855192209, 2.8589153383720736, 1.477204191194325, 1.0299947729491368, 0.7782291112252603, 0.660132395005774, 0.0018780861364841707]

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 10)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 11)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_partisn_eigen_2g_l0(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh_x = [10, 4]
        pydgm.control.coarse_mesh_x = [0.0, 1.5, 2.0]
        pydgm.control.material_map = [1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.dgm_basis_name = 'test/2gbasis'.ljust(256)
        pydgm.control.energy_group_map = [1, 1]
        pydgm.control.angle_order = 8
        self.setSolver('eigen')
        self.setBoundary('reflect')
        pydgm.control.scatter_leg_order = 0

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Partisn output flux
        phi_test = [[3.442525765957952, 3.44133409525864, 3.438914799438404, 3.435188915015736, 3.4300162013446567, 3.423154461297391, 3.4141703934131375, 3.4022425049312393, 3.3857186894563807, 3.3610957244311783, 3.332547468621578, 3.310828206454775, 3.2983691806875637, 3.292637618967479], [1.038826864565325, 1.0405199414437678, 1.0439340321802706, 1.0491279510472575, 1.0561979892073443, 1.065290252825513, 1.0766260147603826, 1.0905777908540444, 1.1080393189273399, 1.1327704173615665, 1.166565957603695, 1.195247115307628, 1.2108105235380155, 1.2184043154658892]]

        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_leg_order + 1, -1)  # Group, legendre, cell

        keff_test = 1.17455939

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        phi_zero = pydgm.state.phi[0, :, :].flatten()

        phi_zero_test = phi_test[:, 0].flatten() / np.linalg.norm(phi_test[:, 0]) * np.linalg.norm(phi_zero)

        np.testing.assert_array_almost_equal(phi_zero, phi_zero_test, 6)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_partisn_eigen_2g_l3(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh_x = [10, 4]
        pydgm.control.coarse_mesh_x = [0.0, 1.5, 2.0]
        pydgm.control.material_map = [1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.dgm_basis_name = 'test/2gbasis'.ljust(256)
        pydgm.control.energy_group_map = [1, 1]
        pydgm.control.angle_order = 8
        self.setSolver('eigen')
        self.setBoundary('reflect')
        pydgm.control.scatter_leg_order = 3
        pydgm.control.lamb = 0.85

        # Partisn output flux
        phi_test = [[[3.4348263649218573, 3.433888018868773, 3.4319780924466303, 3.42902287259914, 3.424889537793379, 3.419345135226939, 3.411967983491092, 3.401955097583459, 3.3876978314964576, 3.3658226138962055, 3.34060976759337, 3.3219476543135085, 3.311461944978981, 3.3067065381042795], [0.0020423376009602104, 0.0061405490727312745, 0.010279647696426843, 0.014487859380892587, 0.01879501657686898, 0.023233575719681245, 0.027840232201383777, 0.03265902220608759, 0.0377496380894604, 0.04322160597649044, 0.040276170715938614, 0.02871879421071569, 0.0172133108882554, 0.005734947202660587], [-0.014080757853872444, -0.013902043581305812, -0.013529929565041343, -0.012931463587873976, -0.012046199898157889, -0.01076583796993369, -0.008890167424784212, -0.006031191544566744, -0.0014017648096517898, 0.006661714787003198, 0.016668552081138488, 0.024174518049647653, 0.028202353769555294, 0.029973018707551896], [-0.0005793248806626183, -0.001756926314379438, -0.002992384144214974, -0.004327832838676431, -0.0058116663533987895, -0.007503402096014022, -0.009482521534408456, -0.011866075285174027, -0.014846832287363936, -0.018786937451990907, -0.018006190583241585, -0.0122930135244824, -0.007176905800153899, -0.0023616380722009875]], [[1.0468914012024944, 1.0481968358689602, 1.0508312357939356, 1.0548444570440296, 1.0603189615576314, 1.0673816455493632, 1.076228676326199, 1.0872017225096173, 1.1011537915124854, 1.1216747771114552, 1.1506436314870738, 1.1750453435123285, 1.1877668049794452, 1.1939304950579657], [-0.0018035802543606028, -0.005420473986754876, -0.009066766225751461, -0.01276274710378097, -0.016529853018089805, -0.020391389910122064, -0.02437367793449243, -0.02850824856871196, -0.032837766571938154, -0.03744058644378857, -0.03479493818294627, -0.02478421594673745, -0.014845697536314062, -0.004944617763983217], [0.009044528296445413, 0.0089941262937423, 0.00888482790544369, 0.008698315780443527, 0.008403457458358271, 0.00795064328003027, 0.0072597762570490235, 0.006182844624800746, 0.004322369521129958, -7.745469082057199e-05, -0.007283839517581971, -0.01236885988414086, -0.013996362056737995, -0.014714527559469726], [0.00015593171761653195, 0.00048010038877656716, 0.000842299907640745, 0.0012717736906895632, 0.0018042482177840942, 0.002486031735562782, 0.003380433007147932, 0.004580592143898934, 0.006250135781847204, 0.008825755375993198, 0.00828387207848599, 0.004886452046255631, 0.0026569024049473873, 0.0008431177342006317]]]

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_leg_order + 1, -1)  # Group, legendre, cell

        keff_test = 1.17563713

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            with self.subTest(l=l):
                phi = pydgm.state.phi[l, :, :].flatten()
                phi_zero_test = phi_test[:, l].flatten() / np.linalg.norm(phi_test[:, l]) * np.linalg.norm(phi)
                np.testing.assert_array_almost_equal(phi, phi_zero_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_partisn_eigen_2g_l7_zeroed(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh_x = [10, 4]
        pydgm.control.coarse_mesh_x = [0.0, 1.5, 2.0]
        pydgm.control.material_map = [1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g_zeroed'.ljust(256)
        pydgm.control.dgm_basis_name = 'test/2gbasis'.ljust(256)
        pydgm.control.energy_group_map = [1, 1]
        pydgm.control.angle_order = 8
        self.setSolver('eigen')
        self.setBoundary('reflect')
        pydgm.control.scatter_leg_order = 7
        pydgm.control.lamb = 1.0

        # Partisn output flux
        phi_test = [[[3.442525765957952, 3.44133409525864, 3.438914799438404, 3.435188915015736, 3.4300162013446567, 3.423154461297391, 3.4141703934131375, 3.4022425049312393, 3.3857186894563807, 3.3610957244311783, 3.332547468621578, 3.310828206454775, 3.2983691806875637, 3.292637618967479], [0.0019644320984535876, 0.0059108120945694995, 0.009910062792966128, 0.013998549920050937, 0.01821447539711394, 0.0225989946277805, 0.02719794108129725, 0.032065052304995886, 0.03727049568899598, 0.04293653815177072, 0.040124283218521156, 0.028601385958514365, 0.017139357954699563, 0.005709715174245851], [-0.012532132606398044, -0.012412411288681507, -0.012159246802683127, -0.011741748785570688, -0.011102896432281906, -0.010139488406617853, -0.008658030296501418, -0.006276951809857101, -0.002206866764278115, 0.005255535093060093, 0.014694305934498608, 0.021718007349751933, 0.02539449632364707, 0.026984954024896868], [-0.00046778165720109954, -0.0014230643907324342, -0.0024386426765394334, -0.0035587633331204543, -0.004834802697643303, -0.006330868937728024, -0.008134119319950651, -0.010375610180247068, -0.013276009498773167, -0.017258163911275617, -0.01664765863491359, -0.01122390121670383, -0.0065043145578594155, -0.0021329570728661276], [0.003969114299288534, 0.00401269078074043, 0.004095196069701726, 0.004205088944363153, 0.004317974960796611, 0.004382969771282358, 0.004292138460096337, 0.0038118805106395898, 0.0024275417388900794, -0.0010155859307936083, -0.005793334371634901, -0.009246181085985392, -0.010849128786682899, -0.01147953358250492], [8.627219578290907e-05, 0.0002714307160306753, 0.0004956786398214036, 0.0007896022697536365, 0.0011920986005699644, 0.0017577114517792342, 0.0025709460150363794, 0.003776204607134917, 0.005643863586461559, 0.008727151896997014, 0.0087004534724765, 0.005368765896173973, 0.002939042494471404, 0.0009374881666138965], [-0.0012338455399673876, -0.0012965906401851829, -0.001422028253820068, -0.0016085775620775788, -0.0018490809840636807, -0.00212121054303277, -0.002365370724816171, -0.002434159103351513, -0.001976610668184789, -0.00016923996766794736, 0.002573819362687621, 0.00449141131591789, 0.005260455272500297, 0.005521341697351342], [1.840769880720461e-05, 4.910755636990116e-05, 6.038798543175905e-05, 3.5299096483823456e-05, -5.088653346535521e-05, -0.00023851493223742137, -0.000599745406981475, -0.0012735717818884024, -0.0025442692239777687, -0.00502646745306673, -0.005168777456609798, -0.0028012837878743507, -0.0013925606361596884, -0.0004220122706306076]], [[1.038826864565325, 1.0405199414437678, 1.0439340321802706, 1.0491279510472575, 1.0561979892073443, 1.065290252825513, 1.0766260147603826, 1.0905777908540444, 1.1080393189273399, 1.1327704173615665, 1.166565957603695, 1.195247115307628, 1.2108105235380155, 1.2184043154658892], [-0.0017407245334155781, -0.005234760773456578, -0.008766789581401696, -0.012362938501971777, -0.016050645928225263, -0.01985945597951201, -0.02382224164936656, -0.027977489969826033, -0.03237537000839117, -0.03710334229483023, -0.034559133368562936, -0.024602951035812462, -0.01473186294651191, -0.004905829460754477], [0.007988002179783696, 0.007960765879538347, 0.007898265607441536, 0.007783087821116828, 0.007585372093529892, 0.007257043348967783, 0.006719774335684138, 0.005827997866319691, 0.004185948821983483, -1.90129809110387e-05, -0.007032887957149146, -0.01191356044176433, -0.013405701553960495, -0.014062567322648795], [0.00011800879780404609, 0.0003659723155822782, 0.0006509752833367376, 0.0010019128678555624, 0.0014547236241962795, 0.0020569208977303036, 0.0028747334190305732, 0.004007351954780396, 0.005632401867932851, 0.00824057743028571, 0.0077964145437174875, 0.004525072515655652, 0.002442683255565622, 0.0007721328099104426], [-0.0008906521221368827, -0.0009365449499154424, -0.0010290223295595993, -0.0011691346663959892, -0.001357481383370794, -0.001592436351699908, -0.0018652704156764746, -0.0021394919429477757, -0.002228893035960515, -0.0009761895052693495, 0.0017006042630191842, 0.003062562168807445, 0.0028818484057990187, 0.0027747808277108488], [3.112628137190779e-05, 9.029048934796821e-05, 0.00013932578157633754, 0.00016839583101848146, 0.0001622621357261038, 9.628165203490191e-05, -7.062050159533859e-05, -0.00041226249765456344, -0.001096598659970919, -0.0027313974690471587, -0.002564785859125975, -0.0008616578944919406, -0.0003363466125526522, -8.528754593967874e-05], [-2.6867653309309292e-05, -6.053675389576518e-06, 3.804812906652369e-05, 0.00011063327980663853, 0.00022012110754064326, 0.0003788015317576568, 0.0006024794524562074, 0.0009003179022477464, 0.0011914241310051858, 0.000697034420395138, -0.000640281151709915, -0.0011056126012183448, -0.0007208905299782367, -0.0005492671454416925], [-2.2772579441603102e-05, -6.869734279097567e-05, -0.00011538022215601754, -0.00016191863663079023, -0.00020475943928101314, -0.00023529998362117194, -0.00023493873533717707, -0.00016063931535125267, 0.00012175371040683974, 0.0012483904597068128, 0.0011309737504319323, 1.678280817822564e-05, -7.189281083687382e-05, -3.563431557253652e-05]]]

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # phi_test = np.array([[3.433326, 3.433288, 3.433213, 3.4331, 3.432949, 3.432759, 3.432531, 3.432263, 3.431955, 3.431607, 3.431217, 3.430785, 3.43031, 3.429791, 3.429226, 3.428615, 3.427955, 3.427246, 3.426485, 3.425671, 3.424802, 3.423875, 3.422888, 3.421839, 3.420725, 3.419543, 3.41829, 3.416962, 3.415555, 3.414067, 3.412491, 3.410824, 3.409061, 3.407196, 3.405224, 3.403138, 3.400931, 3.398597, 3.396128, 3.393515, 3.390749, 3.387821, 3.384719, 3.381434, 3.377952, 3.37426, 3.370345, 3.366191, 3.361781, 3.357098, 3.352582, 3.348515, 3.344729, 3.341211, 3.33795, 3.334938, 3.332164, 3.329621, 3.3273, 3.325196, 3.323303, 3.321613, 3.320124, 3.318829, 3.317727, 3.316813, 3.316085, 3.31554, 3.315178, 3.314998, 0.0004094004, 0.001228307, 0.00204753, 0.002867283, 0.003687776, 0.004509223, 0.005331837, 0.006155833, 0.006981427, 0.007808836, 0.00863828, 0.009469979, 0.01030416, 0.01114104, 0.01198085, 0.01282383, 0.01367021, 0.01452023, 0.01537413, 0.01623215, 0.01709456, 0.01796161, 0.01883356, 0.01971069, 0.02059327, 0.0214816, 0.02237596, 0.02327668, 0.02418406, 0.02509845, 0.02602018, 0.02694963, 0.02788717, 0.0288332, 0.02978816, 0.03075248, 0.03172665, 0.03271118, 0.03370661, 0.03471354, 0.0357326, 0.03676448, 0.03780992, 0.03886974, 0.03994483, 0.04103617, 0.04214482, 0.043272, 0.044419, 0.0455873, 0.04501365, 0.04268848, 0.04036613, 0.03804639, 0.03572907, 0.033414, 0.03110099, 0.02878989, 0.02648052, 0.02417273, 0.02186636, 0.01956128, 0.01725733, 0.01495437, 0.01265226, 0.01035088, 0.00805008, 0.005749734, 0.003449712, 0.001149882], [1.04734, 1.04739, 1.047492, 1.047644, 1.047847, 1.048102, 1.048407, 1.048764, 1.049173, 1.049633, 1.050146, 1.050712, 1.05133, 1.052002, 1.052729, 1.05351, 1.054346, 1.055239, 1.056188, 1.057196, 1.058262, 1.059389, 1.060577, 1.061828, 1.063143, 1.064525, 1.065976, 1.067497, 1.069091, 1.070762, 1.072512, 1.074346, 1.076267, 1.078281, 1.080392, 1.082607, 1.084933, 1.087377, 1.08995, 1.09266, 1.09552, 1.098544, 1.101747, 1.105145, 1.10876, 1.112615, 1.116735, 1.121151, 1.125898, 1.131015, 1.137325, 1.144257, 1.150486, 1.156095, 1.161153, 1.165716, 1.169832, 1.17354, 1.176872, 1.179856, 1.182513, 1.184862, 1.186919, 1.188695, 1.1902, 1.191444, 1.192431, 1.193168, 1.193657, 1.193901, -0.0003617221, -0.001085242, -0.00180899, -0.002533119, -0.003257779, -0.003983126, -0.004709312, -0.005436491, -0.00616482, -0.006894453, -0.007625549, -0.008358266, -0.009092764, -0.009829207, -0.01056776, -0.01130858, -0.01205185, -0.01279773, -0.0135464, -0.01429804, -0.01505283, -0.01581094, -0.01657259, -0.01733795, -0.01810722, -0.01888063, -0.01965837, -0.02044067, -0.02122776, -0.02201987, -0.02281726, -0.02362018, -0.02442892, -0.02524375, -0.02606498, -0.02689293, -0.02772795, -0.0285704, -0.02942068, -0.0302792, -0.03114642, -0.03202284, -0.03290899, -0.03380545, -0.03471287, -0.03563194, -0.03656344, -0.03750822, -0.03846724, -0.03944153, -0.03892121, -0.03690057, -0.0348842, -0.03287175, -0.03086291, -0.02885737, -0.02685485, -0.0248551, -0.02285785, -0.02086288, -0.01886996, -0.01687886, -0.01488938, -0.01290132, -0.01091447, -0.008928642, -0.006943643, -0.004959288, -0.00297539, -0.0009917663]])
        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_leg_order + 1, -1)  # Group, legendre, cell

        keff_test = 1.17455939

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            with self.subTest(l=l):
                phi = pydgm.state.phi[l, :, :].flatten()
                phi_zero_test = phi_test[:, l].flatten() / np.linalg.norm(phi_test[:, l]) * np.linalg.norm(phi)
                np.testing.assert_array_almost_equal(phi, phi_zero_test, 12)

        # Test the angular flux
        self.angular_test()

    def test_dgmsolver_partisn_eigen_2g_l7_symmetric(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh_x = [10, 4]
        pydgm.control.coarse_mesh_x = [0.0, 1.5, 2.0]
        pydgm.control.material_map = [1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g_symmetric'.ljust(256)
        pydgm.control.dgm_basis_name = 'test/2gbasis'.ljust(256)
        pydgm.control.energy_group_map = [1, 1]
        pydgm.control.angle_order = 8
        self.setSolver('eigen')
        self.setBoundary('reflect')
        pydgm.control.scatter_leg_order = 7
        pydgm.control.lamb = 1.0

        # Partisn output flux
        phi_test = [[[0.1538782092306766, 0.15382581211270707, 0.15371986228765858, 0.153557761213616, 0.15333472881184002, 0.15304218023034566, 0.15266423446037952, 0.1521701827407139, 0.15149814229420455, 0.1505194051927887, 0.1494249687053612, 0.1486226070725345, 0.14817047331310246, 0.14796502736009512], [0.0001026884291694685, 0.000308030321038727, 0.0005132665423330503, 0.0007183238836813465, 0.0009231242116952735, 0.0011275804740904216, 0.001331589291639088, 0.0015350161940947502, 0.0017376649215678002, 0.0019392120261786507, 0.0017831926104664695, 0.0012716369586346283, 0.0007622344671049473, 0.00025396061625605824], [-0.000682002429319776, -0.0006703722551940497, -0.0006465291239265841, -0.0006091636714206919, -0.0005558703969694842, -0.00048233555403088147, -0.0003805905629842574, -0.00023524453134053137, -1.530782750082567e-05, 0.000343640189869052, 0.0007749990633443837, 0.0010955352332986893, 0.00126790980764641, 0.0013438061833929496], [-3.058337499400269e-05, -9.237270549150144e-05, -0.000156061628751189, -0.00022303056882785367, -0.0002948629752353676, -0.0003735095105970188, -0.00046159206220763036, -0.0005630090779214299, -0.0006841935938615459, -0.0008367925438085822, -0.0007896375192952836, -0.0005397705453395623, -0.00031536355653434935, -0.00010380975887062981], [0.00023192773237918404, 0.00023170419605027595, 0.00023099437633635964, 0.00022917428036478126, 0.00022500268466479033, 0.00021605146985450782, 0.000197467460152566, 0.00015928145031182932, 8.053932401339784e-05, -8.35365504396621e-05, -0.0002985310711184743, -0.00045360492028924913, -0.0005279989261356239, -0.0005580164988029952], [7.399667935943047e-06, 2.2713517743269424e-05, 3.961572529958277e-05, 5.932811702596529e-05, 8.337219672099492e-05, 0.00011383325722981018, 0.00015386938501401595, 0.00020875066014553042, 0.0002880504099107323, 0.0004103501289148088, 0.00039931287995456297, 0.0002511676906294287, 0.00013917831523372736, 4.4655208369662885e-05], [-7.735314258768304e-05, -7.939037040513011e-05, -8.340114274570012e-05, -8.919024664814186e-05, -9.625067330982931e-05, -0.00010335265393286459, -0.00010763270188858098, -0.00010258629461190995, -7.365082975850298e-05, 1.1510318766527229e-05, 0.0001330732285196889, 0.00021783037521026132, 0.00025313880126033583, 0.0002655853959215326], [-6.052281039214784e-07, -2.105593440887652e-06, -4.518823271217054e-06, -8.613917736378408e-06, -1.546882765090553e-05, -2.6773569766345116e-05, -4.5441298246425184e-05, -7.689028022185053e-05, -0.0001317898459288181, -0.00023200916513577488, -0.0002324970071112759, -0.00012949718930364363, -6.569369965651493e-05, -2.012501435494406e-05]], [[0.15387820923067544, 0.15382581211270596, 0.1537198622876574, 0.15355776121361495, 0.1533347288118389, 0.15304218023034463, 0.15266423446037858, 0.152170182740713, 0.1514981422942038, 0.15051940519278806, 0.14942496870536068, 0.14862260707253408, 0.1481704733131023, 0.14796502736009523], [0.00010268842916943727, 0.000308030321038652, 0.0005132665423329395, 0.0007183238836811969, 0.0009231242116950705, 0.0011275804740901753, 0.0013315892916388105, 0.0015350161940944143, 0.0017376649215673982, 0.001939212026178192, 0.0017831926104659298, 0.0012716369586340077, 0.0007622344671042425, 0.0002539606162552429], [-0.0006820024293195696, -0.0006703722551938442, -0.0006465291239263751, -0.0006091636714204733, -0.0005558703969692639, -0.0004823355540306568, -0.0003805905629840232, -0.00023524453134028678, -1.5307827500579338e-05, 0.00034364018986930527, 0.0007749990633446344, 0.001095535233298939, 0.001267909807646645, 0.0013438061833931413], [-3.058337499400052e-05, -9.237270549149494e-05, -0.00015606162875118554, -0.00022303056882785324, -0.0002948629752353676, -0.0003735095105970119, -0.00046159206220760824, -0.0005630090779214117, -0.0006841935938615164, -0.0008367925438085419, -0.0007896375192952203, -0.0005397705453394808, -0.00031536355653423464, -0.00010380975887046675], [0.00023192773237917623, 0.00023170419605026988, 0.0002309943763363501, 0.00022917428036476999, 0.00022500268466477298, 0.00021605146985448354, 0.00019746746015253044, 0.00015928145031178682, 8.05393240133484e-05, -8.353655043972022e-05, -0.00029853107111854366, -0.00045360492028932026, -0.0005279989261356959, -0.0005580164988030507], [7.399667935945216e-06, 2.2713517743276363e-05, 3.961572529959361e-05, 5.932811702597917e-05, 8.337219672100403e-05, 0.00011383325722982016, 0.00015386938501402853, 0.00020875066014554213, 0.00028805040991074357, 0.0004103501289148127, 0.0003993128799545595, 0.0002511676906294157, 0.00013917831523369874, 4.46552083696e-05], [-7.735314258769258e-05, -7.939037040514269e-05, -8.340114274571009e-05, -8.91902466481514e-05, -9.625067330983582e-05, -0.00010335265393286676, -0.00010763270188857924, -0.00010258629461190822, -7.365082975849735e-05, 1.151031876654024e-05, 0.0001330732285197123, 0.00021783037521029254, 0.00025313880126037183, 0.0002655853959215638], [-6.052281039223457e-07, -2.105593440891989e-06, -4.518823271223559e-06, -8.613917736386215e-06, -1.5468827650912467e-05, -2.6773569766352055e-05, -4.544129824643039e-05, -7.689028022185747e-05, -0.00013178984592882633, -0.00023200916513578225, -0.00023249700711128326, -0.0001294971893036471, -6.569369965651146e-05, -2.0125014354917172e-05]]]

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # phi_test = np.array([[3.433326, 3.433288, 3.433213, 3.4331, 3.432949, 3.432759, 3.432531, 3.432263, 3.431955, 3.431607, 3.431217, 3.430785, 3.43031, 3.429791, 3.429226, 3.428615, 3.427955, 3.427246, 3.426485, 3.425671, 3.424802, 3.423875, 3.422888, 3.421839, 3.420725, 3.419543, 3.41829, 3.416962, 3.415555, 3.414067, 3.412491, 3.410824, 3.409061, 3.407196, 3.405224, 3.403138, 3.400931, 3.398597, 3.396128, 3.393515, 3.390749, 3.387821, 3.384719, 3.381434, 3.377952, 3.37426, 3.370345, 3.366191, 3.361781, 3.357098, 3.352582, 3.348515, 3.344729, 3.341211, 3.33795, 3.334938, 3.332164, 3.329621, 3.3273, 3.325196, 3.323303, 3.321613, 3.320124, 3.318829, 3.317727, 3.316813, 3.316085, 3.31554, 3.315178, 3.314998, 0.0004094004, 0.001228307, 0.00204753, 0.002867283, 0.003687776, 0.004509223, 0.005331837, 0.006155833, 0.006981427, 0.007808836, 0.00863828, 0.009469979, 0.01030416, 0.01114104, 0.01198085, 0.01282383, 0.01367021, 0.01452023, 0.01537413, 0.01623215, 0.01709456, 0.01796161, 0.01883356, 0.01971069, 0.02059327, 0.0214816, 0.02237596, 0.02327668, 0.02418406, 0.02509845, 0.02602018, 0.02694963, 0.02788717, 0.0288332, 0.02978816, 0.03075248, 0.03172665, 0.03271118, 0.03370661, 0.03471354, 0.0357326, 0.03676448, 0.03780992, 0.03886974, 0.03994483, 0.04103617, 0.04214482, 0.043272, 0.044419, 0.0455873, 0.04501365, 0.04268848, 0.04036613, 0.03804639, 0.03572907, 0.033414, 0.03110099, 0.02878989, 0.02648052, 0.02417273, 0.02186636, 0.01956128, 0.01725733, 0.01495437, 0.01265226, 0.01035088, 0.00805008, 0.005749734, 0.003449712, 0.001149882], [1.04734, 1.04739, 1.047492, 1.047644, 1.047847, 1.048102, 1.048407, 1.048764, 1.049173, 1.049633, 1.050146, 1.050712, 1.05133, 1.052002, 1.052729, 1.05351, 1.054346, 1.055239, 1.056188, 1.057196, 1.058262, 1.059389, 1.060577, 1.061828, 1.063143, 1.064525, 1.065976, 1.067497, 1.069091, 1.070762, 1.072512, 1.074346, 1.076267, 1.078281, 1.080392, 1.082607, 1.084933, 1.087377, 1.08995, 1.09266, 1.09552, 1.098544, 1.101747, 1.105145, 1.10876, 1.112615, 1.116735, 1.121151, 1.125898, 1.131015, 1.137325, 1.144257, 1.150486, 1.156095, 1.161153, 1.165716, 1.169832, 1.17354, 1.176872, 1.179856, 1.182513, 1.184862, 1.186919, 1.188695, 1.1902, 1.191444, 1.192431, 1.193168, 1.193657, 1.193901, -0.0003617221, -0.001085242, -0.00180899, -0.002533119, -0.003257779, -0.003983126, -0.004709312, -0.005436491, -0.00616482, -0.006894453, -0.007625549, -0.008358266, -0.009092764, -0.009829207, -0.01056776, -0.01130858, -0.01205185, -0.01279773, -0.0135464, -0.01429804, -0.01505283, -0.01581094, -0.01657259, -0.01733795, -0.01810722, -0.01888063, -0.01965837, -0.02044067, -0.02122776, -0.02201987, -0.02281726, -0.02362018, -0.02442892, -0.02524375, -0.02606498, -0.02689293, -0.02772795, -0.0285704, -0.02942068, -0.0302792, -0.03114642, -0.03202284, -0.03290899, -0.03380545, -0.03471287, -0.03563194, -0.03656344, -0.03750822, -0.03846724, -0.03944153, -0.03892121, -0.03690057, -0.0348842, -0.03287175, -0.03086291, -0.02885737, -0.02685485, -0.0248551, -0.02285785, -0.02086288, -0.01886996, -0.01687886, -0.01488938, -0.01290132, -0.01091447, -0.008928642, -0.006943643, -0.004959288, -0.00297539, -0.0009917663]])
        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_leg_order + 1, -1)  # Group, legendre, cell

        keff_test = 0.15282105

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            with self.subTest(l=l):
                phi = pydgm.state.phi[l, :, :].flatten()
                phi_zero_test = phi_test[:, l].flatten() / np.linalg.norm(phi_test[:, l]) * np.linalg.norm(phi)
                np.testing.assert_array_almost_equal(phi, phi_zero_test, 12)

        # Test the angular flux
        self.angular_test()

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()


class TestDGMSOLVER_2D(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.spatial_dimension = 2
        pydgm.control.angle_order = 8
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.recon_print = False
        pydgm.control.eigen_print = False
        pydgm.control.outer_print = False
        pydgm.control.recon_tolerance = 1e-12
        pydgm.control.eigen_tolerance = 1e-14
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.scatter_leg_order = 0
        pydgm.control.delta_leg_order = -1
        pydgm.control.equation_type = 'DD'
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = True
        pydgm.control.ignore_warnings = True
        pydgm.control.lamb = 1.0
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 0.0
        pydgm.control.boundary_north = 0.0
        pydgm.control.boundary_south = 0.0

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()

    def setGroups(self, G):
        if G == 1:
            pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)
            pydgm.control.dgm_basis_name = 'test/1gbasis'.ljust(256)
            pydgm.control.energy_group_map = [1]
        elif G == 2:
            pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
            pydgm.control.dgm_basis_name = 'test/2gbasis'.ljust(256)
            pydgm.control.energy_group_map = [1, 1]
        elif G == 4:
            pydgm.control.xs_name = 'test/4gXS.anlxs'.ljust(256)
            pydgm.control.energy_group_map = [1, 1, 2, 2]
            pydgm.control.dgm_basis_name = 'test/4gbasis'.ljust(256)
        elif G == 7:
            pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_7g'.ljust(256)
            pydgm.control.dgm_basis_name = 'test/7gbasis'.ljust(256)
            pydgm.control.energy_group_map = [1, 1, 1, 1, 2, 2, 2]

    def setSolver(self, solver):
        if solver == 'fixed':
            pydgm.control.solver_type = 'fixed'.ljust(256)
            pydgm.control.source_value = 1.0
            pydgm.control.allow_fission = False
            pydgm.control.max_recon_iters = 10000
            pydgm.control.max_eigen_iters = 1
            pydgm.control.max_outer_iters = 100000
        elif solver == 'eigen':
            pydgm.control.solver_type = 'eigen'.ljust(256)
            pydgm.control.source_value = 0.0
            pydgm.control.allow_fission = True
            pydgm.control.max_recon_iters = 10000
            pydgm.control.max_eigen_iters = 10000
            pydgm.control.max_outer_iters = 10

    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()

    def angular_test(self):
        # Test the angular flux
        nAngles = pydgm.control.number_angles_per_octant
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for o in range(4):
            for c in range(pydgm.control.number_cells):
                for a in range(nAngles):
                    phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, o * nAngles + a, c]
        np.testing.assert_array_almost_equal(phi_test, pydgm.state.phi[0, :, :], 12)

    def set_mesh(self, pType):
        '''Select the problem mesh'''

        if pType == 'homogeneous':
            pydgm.control.fine_mesh_x = [10]
            pydgm.control.fine_mesh_y = [10]
            pydgm.control.coarse_mesh_x = [0.0, 100000.0]
            pydgm.control.coarse_mesh_y = [0.0, 100000.0]
            pydgm.control.material_map = [2]
            pydgm.control.boundary_east = 1.0
            pydgm.control.boundary_west = 1.0
            pydgm.control.boundary_north = 1.0
            pydgm.control.boundary_south = 1.0
        elif pType == 'simple':
            pydgm.control.fine_mesh_x = [25]
            pydgm.control.fine_mesh_y = [25]
            pydgm.control.coarse_mesh_x = [0.0, 10.0]
            pydgm.control.coarse_mesh_y = [0.0, 10.0]
            pydgm.control.material_map = [1]
        elif pType == 'slab':
            pydgm.control.fine_mesh_x = [10]
            pydgm.control.fine_mesh_y = [20]
            pydgm.control.coarse_mesh_x = [0.0, 21.42]
            pydgm.control.coarse_mesh_y = [0.0, 21.42]
            pydgm.control.material_map = [2]
        elif pType == 'c5g7':
            pydgm.control.fine_mesh_x = [10, 10, 6]
            pydgm.control.fine_mesh_y = [10, 10, 6]
            pydgm.control.coarse_mesh_x = [0.0, 21.42, 42.84, 64.26]
            pydgm.control.coarse_mesh_y = [0.0, 21.42, 42.84, 64.26]
            pydgm.control.material_map = [2, 4, 5,
                                          4, 2, 5,
                                          5, 5, 5]
        elif pType == 'single':
            pydgm.control.fine_mesh_x = [2]
            pydgm.control.fine_mesh_y = [1]
            pydgm.control.coarse_mesh_x = [0.0, 1.0]
            pydgm.control.coarse_mesh_y = [0.0, 1.0]
            pydgm.control.material_map = [1]

    def test_dgmsolver_basic_2D_1g_reflect(self):
        '''
        Test for a basic 1 group problem
        '''
        self.setSolver('fixed')
        self.set_mesh('homogeneous')
        self.setGroups(1)
        pydgm.control.angle_order = 8

        pydgm.control.allow_fission = False

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        assert(pydgm.control.number_groups == 1)

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [5] * 100

        phi_test = np.array(phi_test)

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi, phi_test, 8)

        self.angular_test()

    def test_dgmsolver_basic_2D_1g_1a_vacuum(self):
        '''
        Test for a basic 1 group problem
        '''
        self.set_mesh('simple')
        self.setGroups(1)
        self.setSolver('fixed')
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 0.0
        pydgm.control.boundary_north = 0.0
        pydgm.control.boundary_south = 0.0
        pydgm.control.allow_fission = False

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        assert(pydgm.control.number_groups == 1)

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[0.5128609747568281, 0.6773167712139831, 0.7428920389440171, 0.7815532843645795, 0.8017050988344507, 0.8129183924456294, 0.8186192333777655, 0.8218872988327544, 0.8235294558562516, 0.8244499117457011, 0.8249136380974397, 0.8251380055566573, 0.8252036667060597, 0.8251380055566576, 0.8249136380974401, 0.8244499117457017, 0.8235294558562521, 0.8218872988327554, 0.8186192333777662, 0.81291839244563, 0.8017050988344515, 0.7815532843645803, 0.7428920389440175, 0.6773167712139835, 0.5128609747568282, 0.6773167712139831, 0.9191522109956556, 1.0101279836265808, 1.046811188269745, 1.0706509901493697, 1.082306179953408, 1.0890982284197688, 1.092393940905045, 1.0943710244790774, 1.095322191522249, 1.0958571613995292, 1.0961001761636988, 1.0961736115735345, 1.0961001761636997, 1.0958571613995298, 1.0953221915222502, 1.094371024479079, 1.092393940905046, 1.08909822841977, 1.082306179953409, 1.0706509901493706, 1.046811188269746, 1.0101279836265817, 0.9191522109956564, 0.6773167712139837, 0.742892038944017, 1.010127983626581, 1.1482625986461896, 1.1962723842001344, 1.2162573092331372, 1.2309663391416374, 1.2373535288720443, 1.2414786838078147, 1.2432965986450648, 1.2444791104681647, 1.24499985971512, 1.2452803731491178, 1.2453559925876914, 1.2452803731491184, 1.244999859715121, 1.244479110468166, 1.2432965986450666, 1.2414786838078167, 1.237353528872046, 1.230966339141639, 1.2162573092331381, 1.1962723842001353, 1.1482625986461907, 1.010127983626582, 0.7428920389440178, 0.7815532843645796, 1.046811188269745, 1.196272384200134, 1.2752422132283252, 1.2995505587401641, 1.310409682796424, 1.3196712848467558, 1.3229121459817077, 1.3255006837110048, 1.3264359863630988, 1.3271401701539467, 1.3273900240538614, 1.3274869234142137, 1.327390024053862, 1.3271401701539483, 1.3264359863631008, 1.3255006837110068, 1.3229121459817097, 1.3196712848467576, 1.3104096827964256, 1.2995505587401661, 1.275242213228327, 1.1962723842001357, 1.0468111882697462, 0.7815532843645802, 0.8017050988344504, 1.0706509901493695, 1.2162573092331366, 1.2995505587401643, 1.3452497209031955, 1.3567411771030033, 1.3627115460270347, 1.3687134641761085, 1.3700956107492908, 1.3718153695089819, 1.3722188033283726, 1.3726261825701194, 1.3726746849086748, 1.3726261825701203, 1.3722188033283742, 1.3718153695089843, 1.3700956107492925, 1.3687134641761105, 1.3627115460270367, 1.356741177103005, 1.3452497209031975, 1.2995505587401661, 1.2162573092331384, 1.0706509901493706, 0.8017050988344514, 0.8129183924456289, 1.0823061799534075, 1.2309663391416374, 1.3104096827964238, 1.3567411771030033, 1.3837407849220562, 1.3883940252403495, 1.391781180853413, 1.3958074539475693, 1.3960964469643635, 1.3973279212330587, 1.3973887886413996, 1.3976146226859119, 1.3973887886414005, 1.3973279212330607, 1.3960964469643657, 1.3958074539475707, 1.3917811808534148, 1.3883940252403517, 1.3837407849220587, 1.3567411771030051, 1.3104096827964258, 1.2309663391416386, 1.0823061799534086, 0.81291839244563, 0.8186192333777653, 1.0890982284197686, 1.2373535288720436, 1.319671284846756, 1.3627115460270351, 1.3883940252403497, 1.4048482432271674, 1.4059080588108543, 1.4079482979581446, 1.4107567935223844, 1.4103944840640301, 1.4113681192038354, 1.4110951509099934, 1.411368119203836, 1.4103944840640317, 1.4107567935223853, 1.4079482979581461, 1.4059080588108563, 1.4048482432271694, 1.3883940252403517, 1.362711546027037, 1.3196712848467576, 1.237353528872046, 1.0890982284197699, 0.8186192333777663, 0.8218872988327544, 1.092393940905045, 1.2414786838078145, 1.322912145981707, 1.3687134641761085, 1.3917811808534124, 1.4059080588108548, 1.4163858413766799, 1.4155936684368302, 1.4169325249341456, 1.4189994578626368, 1.4181484360902454, 1.4191873654776683, 1.4181484360902463, 1.4189994578626375, 1.4169325249341471, 1.415593668436832, 1.416385841376682, 1.4059080588108568, 1.3917811808534146, 1.3687134641761105, 1.3229121459817095, 1.2414786838078165, 1.092393940905047, 0.8218872988327552, 0.8235294558562515, 1.0943710244790772, 1.2432965986450644, 1.325500683711004, 1.3700956107492905, 1.3958074539475689, 1.4079482979581444, 1.4155936684368298, 1.4226595845322698, 1.4209588928818724, 1.421835104078357, 1.423671773688655, 1.4219795630324892, 1.423671773688656, 1.4218351040783586, 1.420958892881874, 1.4226595845322718, 1.4155936684368327, 1.4079482979581466, 1.3958074539475707, 1.3700956107492928, 1.3255006837110068, 1.2432965986450666, 1.0943710244790792, 0.8235294558562524, 0.8244499117457008, 1.0953221915222489, 1.2444791104681645, 1.3264359863630988, 1.3718153695089823, 1.396096446964363, 1.4107567935223833, 1.4169325249341451, 1.4209588928818722, 1.4259745842172793, 1.424082580661134, 1.4241542494391093, 1.4264775572959318, 1.42415424943911, 1.4240825806611355, 1.425974584217281, 1.420958892881874, 1.4169325249341471, 1.4107567935223857, 1.3960964469643653, 1.3718153695089845, 1.3264359863631006, 1.2444791104681667, 1.0953221915222502, 0.824449911745702, 0.8249136380974391, 1.0958571613995285, 1.2449998597151197, 1.3271401701539467, 1.3722188033283722, 1.3973279212330585, 1.4103944840640301, 1.4189994578626361, 1.4218351040783568, 1.4240825806611341, 1.427381093837987, 1.426071570295947, 1.4256483954611119, 1.4260715702959486, 1.4273810938379876, 1.424082580661136, 1.4218351040783581, 1.4189994578626381, 1.410394484064032, 1.3973279212330605, 1.3722188033283746, 1.3271401701539485, 1.2449998597151215, 1.0958571613995303, 0.8249136380974402, 0.8251380055566571, 1.0961001761636986, 1.2452803731491175, 1.3273900240538605, 1.3726261825701191, 1.3973887886413991, 1.4113681192038343, 1.4181484360902443, 1.4236717736886548, 1.4241542494391093, 1.4260715702959472, 1.4284214388119818, 1.425280452566786, 1.428421438811983, 1.426071570295949, 1.4241542494391097, 1.4236717736886555, 1.4181484360902465, 1.4113681192038365, 1.3973887886414016, 1.3726261825701216, 1.3273900240538627, 1.2452803731491189, 1.0961001761637001, 0.825138005556658, 0.8252036667060596, 1.0961736115735343, 1.2453559925876905, 1.3274869234142128, 1.3726746849086742, 1.3976146226859112, 1.4110951509099916, 1.4191873654776679, 1.421979563032488, 1.4264775572959316, 1.4256483954611114, 1.4252804525667855, 1.4309739984926175, 1.4252804525667866, 1.4256483954611128, 1.4264775572959327, 1.4219795630324894, 1.4191873654776692, 1.4110951509099945, 1.3976146226859132, 1.3726746849086764, 1.327486923414215, 1.2453559925876925, 1.0961736115735354, 0.8252036667060605, 0.825138005556657, 1.0961001761636986, 1.2452803731491173, 1.3273900240538608, 1.372626182570119, 1.3973887886413987, 1.4113681192038343, 1.4181484360902448, 1.4236717736886546, 1.4241542494391095, 1.426071570295948, 1.428421438811982, 1.4252804525667868, 1.4284214388119838, 1.426071570295949, 1.4241542494391108, 1.4236717736886564, 1.418148436090247, 1.411368119203837, 1.3973887886414011, 1.3726261825701218, 1.3273900240538627, 1.2452803731491195, 1.0961001761637001, 0.8251380055566581, 0.8249136380974397, 1.095857161399529, 1.24499985971512, 1.3271401701539465, 1.3722188033283722, 1.3973279212330587, 1.4103944840640301, 1.418999457862636, 1.4218351040783574, 1.4240825806611341, 1.427381093837987, 1.4260715702959483, 1.4256483954611125, 1.4260715702959488, 1.4273810938379878, 1.4240825806611361, 1.421835104078359, 1.4189994578626386, 1.4103944840640321, 1.3973279212330607, 1.3722188033283749, 1.3271401701539487, 1.2449998597151215, 1.0958571613995305, 0.8249136380974409, 0.8244499117457011, 1.0953221915222489, 1.2444791104681645, 1.3264359863630988, 1.3718153695089828, 1.3960964469643642, 1.410756793522384, 1.4169325249341456, 1.420958892881872, 1.4259745842172795, 1.4240825806611346, 1.4241542494391095, 1.426477557295932, 1.4241542494391102, 1.4240825806611361, 1.4259745842172817, 1.4209588928818744, 1.4169325249341482, 1.4107567935223861, 1.3960964469643657, 1.371815369508985, 1.326435986363101, 1.2444791104681663, 1.0953221915222509, 0.8244499117457021, 0.8235294558562516, 1.0943710244790774, 1.2432965986450648, 1.325500683711005, 1.3700956107492908, 1.3958074539475689, 1.4079482979581448, 1.4155936684368302, 1.4226595845322698, 1.4209588928818724, 1.421835104078357, 1.423671773688655, 1.4219795630324894, 1.4236717736886562, 1.4218351040783583, 1.420958892881874, 1.422659584532272, 1.4155936684368329, 1.4079482979581472, 1.3958074539475713, 1.3700956107492932, 1.3255006837110068, 1.2432965986450673, 1.0943710244790794, 0.8235294558562526, 0.8218872988327546, 1.0923939409050452, 1.241478683807815, 1.3229121459817077, 1.368713464176108, 1.3917811808534133, 1.4059080588108555, 1.4163858413766814, 1.4155936684368304, 1.4169325249341456, 1.4189994578626364, 1.4181484360902452, 1.4191873654776685, 1.4181484360902465, 1.4189994578626381, 1.4169325249341473, 1.4155936684368324, 1.4163858413766832, 1.4059080588108572, 1.3917811808534155, 1.3687134641761114, 1.32291214598171, 1.2414786838078167, 1.0923939409050465, 0.8218872988327557, 0.8186192333777654, 1.089098228419769, 1.2373535288720448, 1.3196712848467562, 1.3627115460270354, 1.3883940252403495, 1.4048482432271683, 1.4059080588108557, 1.4079482979581455, 1.410756793522384, 1.4103944840640303, 1.4113681192038354, 1.411095150909993, 1.411368119203836, 1.410394484064032, 1.410756793522386, 1.4079482979581472, 1.4059080588108572, 1.4048482432271705, 1.388394025240352, 1.3627115460270374, 1.3196712848467584, 1.237353528872046, 1.0890982284197708, 0.8186192333777667, 0.8129183924456292, 1.0823061799534082, 1.2309663391416377, 1.3104096827964242, 1.3567411771030033, 1.3837407849220569, 1.3883940252403497, 1.391781180853413, 1.3958074539475698, 1.3960964469643644, 1.3973279212330594, 1.3973887886413998, 1.397614622685912, 1.3973887886414005, 1.3973279212330607, 1.396096446964366, 1.3958074539475713, 1.391781180853415, 1.388394025240352, 1.3837407849220584, 1.3567411771030051, 1.310409682796426, 1.2309663391416392, 1.0823061799534093, 0.8129183924456299, 0.8017050988344508, 1.07065099014937, 1.2162573092331375, 1.2995505587401643, 1.3452497209031953, 1.3567411771030036, 1.3627115460270354, 1.3687134641761085, 1.370095610749291, 1.3718153695089825, 1.3722188033283735, 1.3726261825701203, 1.372674684908675, 1.3726261825701211, 1.3722188033283742, 1.371815369508985, 1.3700956107492934, 1.3687134641761116, 1.3627115460270374, 1.3567411771030051, 1.3452497209031975, 1.2995505587401661, 1.2162573092331384, 1.0706509901493708, 0.8017050988344514, 0.7815532843645797, 1.046811188269745, 1.1962723842001342, 1.2752422132283254, 1.2995505587401648, 1.3104096827964244, 1.3196712848467562, 1.322912145981708, 1.3255006837110048, 1.3264359863630992, 1.3271401701539476, 1.3273900240538619, 1.3274869234142137, 1.327390024053862, 1.3271401701539485, 1.3264359863631006, 1.3255006837110068, 1.3229121459817104, 1.3196712848467584, 1.3104096827964262, 1.2995505587401661, 1.2752422132283268, 1.1962723842001353, 1.0468111882697464, 0.7815532843645804, 0.742892038944017, 1.0101279836265813, 1.1482625986461896, 1.196272384200134, 1.2162573092331368, 1.2309663391416377, 1.2373535288720443, 1.241478683807815, 1.2432965986450653, 1.2444791104681654, 1.2449998597151208, 1.245280373149118, 1.2453559925876916, 1.245280373149119, 1.2449998597151217, 1.2444791104681663, 1.2432965986450666, 1.2414786838078173, 1.2373535288720463, 1.230966339141639, 1.2162573092331386, 1.1962723842001357, 1.1482625986461912, 1.0101279836265824, 0.7428920389440179, 0.677316771213983, 0.9191522109956558, 1.010127983626581, 1.0468111882697448, 1.0706509901493695, 1.0823061799534077, 1.0890982284197688, 1.0923939409050452, 1.0943710244790776, 1.0953221915222493, 1.0958571613995294, 1.096100176163699, 1.0961736115735348, 1.0961001761636995, 1.09585716139953, 1.0953221915222504, 1.0943710244790792, 1.092393940905047, 1.0890982284197708, 1.0823061799534093, 1.0706509901493708, 1.0468111882697464, 1.010127983626582, 0.9191522109956568, 0.6773167712139838, 0.5128609747568279, 0.6773167712139831, 0.7428920389440171, 0.7815532843645797, 0.8017050988344507, 0.8129183924456291, 0.8186192333777655, 0.8218872988327544, 0.8235294558562514, 0.8244499117457011, 0.8249136380974398, 0.8251380055566575, 0.82520366670606, 0.8251380055566576, 0.8249136380974401, 0.8244499117457018, 0.8235294558562525, 0.8218872988327557, 0.8186192333777665, 0.8129183924456304, 0.8017050988344514, 0.7815532843645805, 0.7428920389440181, 0.6773167712139839, 0.5128609747568283]]]

        phi_test = np.array(phi_test)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            with self.subTest(l=l):
                phi = pydgm.state.mg_phi[l, :, :].flatten()
                phi_zero_test = phi_test[:, l].flatten()
                np.testing.assert_array_almost_equal(phi, phi_zero_test, 8)

        self.angular_test()

    def test_dgmsolver_basic_2D_1g_1a_vacuum_l1(self):
        '''
        Test for a basic 1 group problem
        '''
        self.set_mesh('simple')
        self.setSolver('fixed')
        self.setGroups(1)
        pydgm.control.material_map = [2]
        pydgm.control.lamb = 0.95
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 0.0
        pydgm.control.boundary_north = 1.0
        pydgm.control.boundary_south = 1.0
        pydgm.control.allow_fission = False
        pydgm.control.scatter_leg_order = 1

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        assert(pydgm.control.number_groups == 1)

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[2.1155214619699514, 2.77746163813215, 3.2874952059661293, 3.6804820674435823, 3.9832821869498707, 4.216591422853343, 4.396356053214123, 4.534862675988926, 4.641577976503701, 4.723795758832229, 4.7871354645428115, 4.835926251697274, 4.873502885229927, 4.902433662296076, 4.924695950888916, 4.9418113394821415, 4.954949634939837, 4.965008816768869, 4.972676412447021, 4.978476488723142, 4.982805470662491, 4.9859592366063445, 4.988153340938925, 4.9895377468524105, 4.9902070760687325, 2.1155214619699456, 2.777461638132166, 3.287495205966074, 3.6804820674436494, 3.9832821869498685, 4.216591422853293, 4.396356053214151, 4.534862675988926, 4.641577976503699, 4.723795758832234, 4.787135464542815, 4.835926251697278, 4.87350288522993, 4.902433662296076, 4.924695950888913, 4.941811339482138, 4.954949634939832, 4.9650088167688695, 4.972676412447021, 4.978476488723141, 4.982805470662492, 4.985959236606346, 4.988153340938926, 4.989537746852409, 4.990207076068729, 2.115521461969962, 2.777461638132113, 3.2874952059661533, 3.6804820674435756, 3.983282186949894, 4.21659142285333, 4.3963560532140935, 4.534862675988948, 4.641577976503708, 4.723795758832238, 4.787135464542824, 4.835926251697284, 4.873502885229934, 4.902433662296074, 4.924695950888906, 4.941811339482129, 4.954949634939827, 4.965008816768864, 4.9726764124470195, 4.9784764887231425, 4.982805470662496, 4.98595923660635, 4.988153340938925, 4.989537746852403, 4.990207076068724, 2.1155214619699336, 2.777461638132193, 3.2874952059660827, 3.6804820674436236, 3.9832821869498463, 4.21659142285334, 4.396356053214133, 4.534862675988895, 4.641577976503733, 4.72379575883225, 4.787135464542832, 4.835926251697296, 4.873502885229939, 4.9024336622960725, 4.924695950888901, 4.94181133948212, 4.954949634939815, 4.9650088167688535, 4.972676412447014, 4.97847648872314, 4.9828054706624965, 4.985959236606352, 4.988153340938927, 4.9895377468524025, 4.990207076068718, 2.1155214619699465, 2.7774616381321464, 3.287495205966141, 3.680482067443583, 3.983282186949881, 4.216591422853283, 4.396356053214144, 4.534862675988945, 4.641577976503681, 4.723795758832279, 4.787135464542853, 4.835926251697308, 4.873502885229949, 4.902433662296076, 4.924695950888897, 4.9418113394821095, 4.9549496349398, 4.965008816768842, 4.972676412447004, 4.978476488723136, 4.982805470662497, 4.985959236606355, 4.988153340938934, 4.989537746852406, 4.990207076068718, 2.115521461969965, 2.7774616381321353, 3.2874952059661307, 3.6804820674436347, 3.983282186949838, 4.216591422853322, 4.3963560532140775, 4.534862675988952, 4.641577976503745, 4.723795758832235, 4.787135464542882, 4.835926251697332, 4.873502885229963, 4.902433662296084, 4.9246959508888954, 4.941811339482098, 4.954949634939785, 4.965008816768823, 4.972676412446988, 4.978476488723127, 4.982805470662496, 4.985959236606362, 4.988153340938939, 4.989537746852414, 4.990207076068727, 2.1155214619699616, 2.7774616381321873, 3.287495205966097, 3.680482067443629, 3.9832821869499013, 4.216591422853279, 4.39635605321412, 4.534862675988878, 4.641577976503744, 4.723795758832312, 4.787135464542844, 4.835926251697361, 4.873502885229992, 4.902433662296099, 4.9246959508889, 4.94181133948209, 4.9549496349397675, 4.965008816768801, 4.972676412446969, 4.978476488723113, 4.98280547066249, 4.985959236606364, 4.9881533409389505, 4.989537746852428, 4.990207076068748, 2.115521461969961, 2.777461638132167, 3.2874952059661524, 3.6804820674435925, 3.9832821869499035, 4.2165914228533525, 4.396356053214075, 4.5348626759889195, 4.641577976503658, 4.723795758832302, 4.787135464542938, 4.83592625169734, 4.8735028852300255, 4.902433662296131, 4.924695950888914, 4.941811339482091, 4.954949634939754, 4.965008816768776, 4.972676412446941, 4.978476488723093, 4.982805470662478, 4.985959236606366, 4.988153340938966, 4.989537746852457, 4.990207076068792, 2.1155214619699585, 2.7774616381321655, 3.287495205966137, 3.6804820674436463, 3.983282186949857, 4.216591422853364, 4.396356053214162, 4.534862675988877, 4.641577976503705, 4.723795758832208, 4.787135464542919, 4.83592625169745, 4.873502885230019, 4.90243366229617, 4.92469595088895, 4.941811339482107, 4.954949634939747, 4.965008816768752, 4.972676412446909, 4.978476488723062, 4.982805470662458, 4.985959236606364, 4.988153340938987, 4.989537746852505, 4.990207076068866, 2.115521461969955, 2.777461638132161, 3.287495205966126, 3.68048206744363, 3.9832821869499093, 4.216591422853312, 4.396356053214188, 4.53486267598898, 4.6415779765036635, 4.723795758832261, 4.787135464542813, 4.835926251697418, 4.873502885230147, 4.902433662296186, 4.924695950888996, 4.94181133948214, 4.954949634939753, 4.965008816768735, 4.972676412446874, 4.978476488723021, 4.982805470662428, 4.985959236606355, 4.98815334093901, 4.989537746852569, 4.990207076068971, 2.1155214619699465, 2.7774616381321517, 3.2874952059661178, 3.6804820674436143, 3.983282186949893, 4.21659142285336, 4.396356053214128, 4.53486267598902, 4.641577976503782, 4.723795758832218, 4.78713546454288, 4.835926251697309, 4.873502885230105, 4.902433662296334, 4.924695950889035, 4.94181133948219, 4.954949634939786, 4.9650088167687345, 4.972676412446843, 4.978476488722976, 4.982805470662388, 4.985959236606341, 4.988153340939038, 4.989537746852649, 4.9902070760691135, 2.1155214619699425, 2.7774616381321406, 3.287495205966106, 3.680482067443601, 3.9832821869498716, 4.216591422853341, 4.3963560532141654, 4.534862675988948, 4.6415779765038385, 4.723795758832351, 4.787135464542834, 4.835926251697391, 4.87350288523, 4.90243366229628, 4.924695950889204, 4.941811339482259, 4.954949634939842, 4.965008816768761, 4.972676412446831, 4.9784764887229365, 4.9828054706623455, 4.985959236606322, 4.988153340939068, 4.989537746852747, 4.990207076069285, 2.115521461969938, 2.7774616381321353, 3.2874952059660925, 3.680482067443586, 3.983282186949854, 4.216591422853312, 4.396356053214142, 4.5348626759889745, 4.641577976503753, 4.723795758832422, 4.787135464542972, 4.835926251697342, 4.87350288523011, 4.9024336622961915, 4.924695950889147, 4.941811339482447, 4.954949634939936, 4.965008816768823, 4.97267641244685, 4.978476488722916, 4.982805470662308, 4.9859592366063, 4.988153340939096, 4.989537746852852, 4.990207076069479, 2.115521461969938, 2.7774616381321326, 3.2874952059660885, 3.6804820674435716, 3.983282186949836, 4.2165914228532895, 4.396356053214106, 4.534862675988942, 4.641577976503759, 4.723795758832318, 4.7871354645430575, 4.83592625169748, 4.873502885230049, 4.902433662296341, 4.924695950889094, 4.941811339482397, 4.95494963494015, 4.965008816768947, 4.972676412446921, 4.978476488722933, 4.98280547066229, 4.985959236606281, 4.98815334093912, 4.989537746852953, 4.990207076069677, 2.1155214619699416, 2.7774616381321358, 3.2874952059660885, 3.680482067443571, 3.9832821869498236, 4.216591422853272, 4.396356053214076, 4.534862675988894, 4.6415779765037115, 4.723795758832293, 4.787135464542929, 4.835926251697569, 4.87350288523017, 4.902433662296263, 4.924695950889303, 4.941811339482415, 4.954949634940126, 4.965008816769193, 4.972676412447082, 4.978476488723019, 4.9828054706623135, 4.985959236606278, 4.988153340939131, 4.989537746853027, 4.9902070760698365, 2.1155214619699456, 2.7774616381321438, 3.2874952059660965, 3.680482067443577, 3.9832821869498267, 4.21659142285326, 4.396356053214055, 4.534862675988856, 4.641577976503651, 4.723795758832229, 4.787135464542869, 4.83592625169741, 4.8735028852302555, 4.902433662296349, 4.9246959508892, 4.941811339482711, 4.954949634940262, 4.96500881676923, 4.972676412447377, 4.978476488723223, 4.982805470662418, 4.985959236606305, 4.988153340939135, 4.989537746853045, 4.9902070760699075, 2.1155214619699527, 2.7774616381321526, 3.2874952059661084, 3.6804820674435894, 3.983282186949837, 4.216591422853266, 4.396356053214046, 4.534862675988836, 4.641577976503607, 4.723795758832156, 4.787135464542783, 4.835926251697306, 4.873502885230058, 4.902433662296415, 4.924695950889225, 4.941811339482574, 4.954949634940682, 4.965008816769538, 4.972676412447512, 4.9784764887235715, 4.982805470662651, 4.9859592366064, 4.988153340939122, 4.989537746852983, 4.990207076069819, 2.115521461969957, 2.7774616381321615, 3.28749520596612, 3.6804820674436036, 3.983282186949852, 4.2165914228532815, 4.396356053214059, 4.534862675988832, 4.641577976503589, 4.723795758832109, 4.787135464542699, 4.835926251697201, 4.8735028852299145, 4.902433662296182, 4.9246959508892605, 4.941811339482512, 4.954949634940508, 4.9650088167701325, 4.972676412448062, 4.9784764887238415, 4.98280547066304, 4.985959236606607, 4.988153340939116, 4.989537746852791, 4.990207076069488, 2.115521461969959, 2.777461638132165, 3.2874952059661293, 3.6804820674436165, 3.9832821869498702, 4.2165914228533, 4.3963560532140775, 4.53486267598885, 4.6415779765035925, 4.723795758832099, 4.787135464542659, 4.83592625169712, 4.873502885229795, 4.902433662296005, 4.9246959508889985, 4.941811339482505, 4.954949634940328, 4.9650088167699105, 4.972676412448865, 4.978476488724687, 4.982805470663438, 4.985959236606952, 4.988153340939141, 4.989537746852459, 4.990207076068828, 2.115521461969959, 2.7774616381321673, 3.287495205966133, 3.6804820674436236, 3.9832821869498813, 4.216591422853317, 4.396356053214099, 4.534862675988874, 4.641577976503622, 4.723795758832121, 4.787135464542672, 4.8359262516971, 4.873502885229737, 4.902433662295897, 4.92469595088881, 4.94181133948223, 4.954949634940263, 4.965008816769564, 4.972676412448564, 4.9784764887257005, 4.982805470664563, 4.9859592366073775, 4.9881533409392205, 4.989537746851977, 4.990207076067756, 2.1155214619699563, 2.7774616381321633, 3.287495205966129, 3.6804820674436227, 3.9832821869498836, 4.216591422853322, 4.396356053214114, 4.534862675988901, 4.641577976503661, 4.723795758832171, 4.7871354645427235, 4.835926251697157, 4.8735028852297715, 4.902433662295896, 4.9246959508887524, 4.941811339482067, 4.9549496349399895, 4.965008816769406, 4.972676412447958, 4.9784764887252075, 4.9828054706656815, 4.985959236608623, 4.988153340939394, 4.989537746851361, 4.990207076066229, 2.1155214619699514, 2.777461638132155, 3.28749520596612, 3.680482067443612, 3.9832821869498733, 4.216591422853318, 4.3963560532141175, 4.534862675988919, 4.641577976503696, 4.723795758832233, 4.787135464542814, 4.835926251697265, 4.873502885229904, 4.902433662296027, 4.9246959508888555, 4.941811339482099, 4.95494963493988, 4.96500881676912, 4.972676412447615, 4.978476488724162, 4.982805470664763, 4.985959236609568, 4.988153340940375, 4.989537746850758, 4.99020707606428, 2.115521461969946, 2.7774616381321473, 3.287495205966106, 3.6804820674435925, 3.9832821869498507, 4.216591422853299, 4.396356053214109, 4.534862675988926, 4.641577976503724, 4.723795758832294, 4.787135464542917, 4.835926251697422, 4.873502885230109, 4.902433662296279, 4.924695950889122, 4.941811339482341, 4.954949634940019, 4.96500881676904, 4.97267641244722, 4.978476488723418, 4.9828054706629485, 4.985959236607795, 4.988153340940623, 4.9895377468508535, 4.990207076062149, 2.115521461969941, 2.7774616381321344, 3.287495205966085, 3.680482067443565, 3.9832821869498223, 4.216591422853273, 4.396356053214089, 4.5348626759889195, 4.641577976503747, 4.723795758832354, 4.787135464543032, 4.835926251697607, 4.873502885230372, 4.902433662296615, 4.924695950889526, 4.941811339482757, 4.954949634940379, 4.96500881676922, 4.972676412447033, 4.978476488722675, 4.982805470661423, 4.985959236604697, 4.988153340937385, 4.989537746849712, 4.990207076060558, 2.115521461969936, 2.7774616381321215, 3.287495205966064, 3.68048206744354, 3.9832821869497947, 4.216591422853245, 4.396356053214069, 4.534862675988914, 4.641577976503769, 4.723795758832419, 4.787135464543158, 4.83592625169781, 4.873502885230665, 4.902433662297009, 4.92469595089, 4.941811339483287, 4.954949634940883, 4.965008816769582, 4.972676412447073, 4.97847648872212, 4.982805470659965, 4.985959236601895, 4.988153340932358, 4.989537746844231, 4.990207076057311]]]

        phi_test = np.array(phi_test)

        # Test the scalar flux
        l = 0
        phi = pydgm.state.mg_phi[l, :, :].flatten()
        phi_zero_test = phi_test[:, l].flatten()
        np.testing.assert_array_almost_equal(phi, phi_zero_test, 8)

        self.angular_test()

    def test_dgmsolver_basic_2D_2g_1a_vacuum(self):
        '''
        Test for a basic 2 group problem
        '''
        self.set_mesh('simple')
        self.setSolver('fixed')
        self.setGroups(2)
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 0.0
        pydgm.control.boundary_north = 0.0
        pydgm.control.boundary_south = 0.0
        pydgm.control.allow_fission = False

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        assert(pydgm.control.number_fine_groups == 2)

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[2.1137582766948073, 2.636581917488039, 3.026045587155106, 3.371616718556787, 3.6667806259373976, 3.918852544828727, 4.129290293281016, 4.301839388015012, 4.43913453975634, 4.543496011917352, 4.616741622074667, 4.660168812348085, 4.674557938961098, 4.660168812348085, 4.616741622074666, 4.543496011917352, 4.439134539756339, 4.301839388015011, 4.129290293281011, 3.918852544828722, 3.666780625937393, 3.3716167185567794, 3.0260455871551, 2.6365819174880345, 2.1137582766948024, 2.636581917488038, 3.4335823665029626, 3.9865333116002404, 4.413365681101105, 4.7889411365991865, 5.104453801345153, 5.36923080244542, 5.585594884403345, 5.7577177607817, 5.8884937048816415, 5.980248025842213, 6.034641916141866, 6.052663312839643, 6.034641916141865, 5.980248025842214, 5.888493704881639, 5.757717760781698, 5.585594884403341, 5.369230802445415, 5.104453801345149, 4.78894113659918, 4.413365681101101, 3.986533311600236, 3.433582366502959, 2.636581917488036, 3.0260455871551004, 3.986533311600237, 4.764985911789622, 5.3195594392634815, 5.755716685290208, 6.1360848502194845, 6.449966786612971, 6.708038228026866, 6.912828986796572, 7.068443611677266, 7.177617595002775, 7.242328637658107, 7.263768519691014, 7.242328637658108, 7.177617595002773, 7.068443611677264, 6.912828986796565, 6.708038228026858, 6.449966786612963, 6.136084850219477, 5.755716685290201, 5.319559439263479, 4.764985911789622, 3.9865333116002364, 3.026045587155101, 3.3716167185567794, 4.413365681101101, 5.319559439263478, 6.059734933640805, 6.594669919226913, 7.019009405830075, 7.385013603583526, 7.680284321463562, 7.916262866710092, 8.095197815005834, 8.220753908477315, 8.295185933912736, 8.319842223627575, 8.295185933912736, 8.220753908477313, 8.095197815005829, 7.9162628667100865, 7.680284321463553, 7.385013603583518, 7.019009405830066, 6.5946699192269085, 6.0597349336408035, 5.319559439263479, 4.413365681101103, 3.3716167185567816, 3.666780625937396, 4.788941136599183, 5.755716685290202, 6.594669919226909, 7.282837454229495, 7.7826243439997524, 8.17920924579081, 8.516051572939901, 8.77921200419981, 8.980435993926783, 9.1213414136694, 9.204876783614267, 9.232563239373745, 9.20487678361426, 9.121341413669393, 8.980435993926774, 8.779212004199799, 8.516051572939887, 8.179209245790801, 7.782624343999744, 7.28283745422949, 6.594669919226911, 5.755716685290203, 4.788941136599184, 3.6667806259373954, 3.918852544828727, 5.104453801345156, 6.136084850219482, 7.019009405830074, 7.7826243439997524, 8.40938822057789, 8.862542988092518, 9.219039481903943, 9.514904344894836, 9.734791360013814, 9.890382762980401, 9.982413174745984, 10.012890843410913, 9.982413174745975, 9.890382762980387, 9.734791360013801, 9.514904344894822, 9.219039481903929, 8.862542988092505, 8.40938822057788, 7.782624343999746, 7.019009405830071, 6.136084850219481, 5.104453801345154, 3.918852544828727, 4.129290293281017, 5.369230802445423, 6.449966786612975, 7.385013603583529, 8.179209245790814, 8.86254298809252, 9.421325551938462, 9.819043013362487, 10.125544719186678, 10.370617986609494, 10.537607630974078, 10.637814624227977, 10.67092412242758, 10.637814624227971, 10.537607630974074, 10.37061798660948, 10.125544719186667, 9.81904301336248, 9.421325551938448, 8.862542988092509, 8.179209245790803, 7.385013603583525, 6.449966786612973, 5.369230802445423, 4.129290293281019, 4.301839388015017, 5.585594884403349, 6.70803822802687, 7.680284321463569, 8.516051572939904, 9.219039481903947, 9.819043013362498, 10.305028855809763, 10.640260599269606, 10.888489126670626, 11.074282265530496, 11.179757933772853, 11.215539376713828, 11.179757933772844, 11.074282265530494, 10.888489126670617, 10.640260599269597, 10.305028855809747, 9.819043013362476, 9.219039481903932, 8.516051572939894, 7.680284321463562, 6.708038228026868, 5.585594884403352, 4.301839388015019, 4.439134539756346, 5.7577177607817065, 6.912828986796578, 7.916262866710104, 8.779212004199813, 9.51490434489484, 10.125544719186685, 10.64026059926961, 11.049730149466411, 11.31657882466728, 11.499364812828357, 11.618289856582024, 11.653913937623651, 11.61828985658202, 11.499364812828354, 11.316578824667275, 11.049730149466402, 10.640260599269599, 10.12554471918667, 9.514904344894827, 8.779212004199808, 7.916262866710098, 6.912828986796577, 5.757717760781709, 4.439134539756347, 4.543496011917359, 5.888493704881649, 7.068443611677271, 8.095197815005841, 8.98043599392679, 9.734791360013816, 10.370617986609494, 10.88848912667063, 11.316578824667287, 11.64652310519862, 11.839888723085174, 11.950458326084561, 11.997247515430729, 11.950458326084556, 11.839888723085167, 11.646523105198613, 11.316578824667275, 10.888489126670617, 10.370617986609481, 9.734791360013805, 8.980435993926779, 8.095197815005836, 7.068443611677272, 5.888493704881652, 4.543496011917362, 4.61674162207467, 5.980248025842219, 7.177617595002779, 8.22075390847732, 9.121341413669404, 9.890382762980396, 10.537607630974078, 11.074282265530497, 11.499364812828357, 11.839888723085176, 12.08742129738902, 12.20452583801236, 12.231939627403282, 12.204525838012362, 12.087421297389014, 11.839888723085165, 11.499364812828354, 11.074282265530488, 10.537607630974072, 9.890382762980394, 9.121341413669398, 8.220753908477317, 7.177617595002782, 5.980248025842224, 4.6167416220746755, 4.660168812348088, 6.03464191614187, 7.242328637658109, 8.295185933912737, 9.204876783614266, 9.982413174745979, 10.637814624227977, 11.17975793377285, 11.61828985658203, 11.95045832608456, 12.20452583801236, 12.36242521569222, 12.403495350196929, 12.36242521569222, 12.204525838012351, 11.950458326084558, 11.618289856582022, 11.179757933772846, 10.637814624227973, 9.982413174745977, 9.20487678361426, 8.295185933912736, 7.2423286376581135, 6.034641916141876, 4.660168812348094, 4.674557938961099, 6.052663312839645, 7.263768519691019, 8.319842223627575, 9.232563239373746, 10.012890843410915, 10.67092412242758, 11.215539376713833, 11.653913937623651, 11.99724751543073, 12.231939627403282, 12.403495350196932, 12.490278681424238, 12.403495350196927, 12.231939627403277, 11.997247515430733, 11.653913937623647, 11.215539376713828, 10.670924122427579, 10.012890843410915, 9.232563239373748, 8.31984222362758, 7.263768519691021, 6.05266331283965, 4.674557938961106, 4.660168812348089, 6.034641916141869, 7.24232863765811, 8.295185933912737, 9.204876783614264, 9.98241317474598, 10.63781462422798, 11.179757933772853, 11.61828985658203, 11.950458326084561, 12.204525838012364, 12.362425215692218, 12.40349535019693, 12.362425215692218, 12.204525838012357, 11.950458326084558, 11.618289856582027, 11.179757933772848, 10.637814624227977, 9.982413174745984, 9.204876783614267, 8.29518593391274, 7.2423286376581135, 6.034641916141875, 4.660168812348094, 4.616741622074671, 5.980248025842217, 7.177617595002775, 8.220753908477317, 9.121341413669398, 9.890382762980392, 10.537607630974076, 11.0742822655305, 11.499364812828361, 11.839888723085176, 12.087421297389016, 12.204525838012358, 12.231939627403277, 12.204525838012355, 12.087421297389016, 11.839888723085169, 11.499364812828361, 11.074282265530499, 10.537607630974078, 9.890382762980403, 9.121341413669404, 8.220753908477318, 7.177617595002784, 5.980248025842225, 4.616741622074678, 4.5434960119173535, 5.888493704881644, 7.068443611677267, 8.095197815005829, 8.980435993926775, 9.734791360013807, 10.370617986609489, 10.888489126670626, 11.316578824667278, 11.646523105198616, 11.839888723085169, 11.95045832608456, 11.99724751543073, 11.950458326084556, 11.83988872308517, 11.646523105198614, 11.31657882466728, 10.888489126670631, 10.370617986609496, 9.73479136001382, 8.980435993926786, 8.095197815005838, 7.068443611677275, 5.888493704881654, 4.543496011917363, 4.4391345397563375, 5.7577177607816985, 6.912828986796568, 7.916262866710087, 8.779212004199803, 9.514904344894827, 10.125544719186676, 10.640260599269608, 11.049730149466406, 11.316578824667276, 11.499364812828356, 11.61828985658202, 11.653913937623644, 11.618289856582027, 11.499364812828356, 11.316578824667276, 11.049730149466411, 10.640260599269613, 10.125544719186681, 9.51490434489484, 8.779212004199811, 7.916262866710099, 6.912828986796576, 5.757717760781707, 4.439134539756347, 4.301839388015011, 5.58559488440334, 6.708038228026859, 7.680284321463556, 8.51605157293989, 9.219039481903934, 9.819043013362487, 10.305028855809757, 10.640260599269604, 10.88848912667062, 11.074282265530497, 11.179757933772846, 11.215539376713828, 11.179757933772846, 11.074282265530496, 10.888489126670626, 10.640260599269608, 10.30502885580976, 9.819043013362489, 9.219039481903943, 8.516051572939903, 7.680284321463565, 6.708038228026868, 5.585594884403352, 4.3018393880150185, 4.12929029328101, 5.369230802445414, 6.449966786612966, 7.3850136035835225, 8.179209245790803, 8.862542988092512, 9.42132555193845, 9.819043013362489, 10.125544719186678, 10.37061798660949, 10.537607630974074, 10.637814624227977, 10.670924122427584, 10.637814624227978, 10.537607630974083, 10.370617986609492, 10.125544719186678, 9.81904301336249, 9.421325551938459, 8.862542988092521, 8.17920924579081, 7.38501360358353, 6.4499667866129755, 5.369230802445425, 4.129290293281018, 3.91885254482872, 5.104453801345149, 6.136084850219476, 7.019009405830068, 7.782624343999747, 8.409388220577886, 8.86254298809251, 9.21903948190394, 9.514904344894834, 9.734791360013814, 9.890382762980398, 9.982413174745982, 10.012890843410915, 9.982413174745984, 9.890382762980401, 9.734791360013817, 9.514904344894838, 9.219039481903938, 8.862542988092521, 8.409388220577892, 7.782624343999754, 7.019009405830075, 6.136084850219486, 5.104453801345158, 3.918852544828728, 3.6667806259373914, 4.78894113659918, 5.755716685290201, 6.594669919226909, 7.28283745422949, 7.782624343999746, 8.179209245790807, 8.5160515729399, 8.77921200419981, 8.980435993926783, 9.121341413669407, 9.204876783614267, 9.23256323937375, 9.204876783614273, 9.121341413669406, 8.980435993926788, 8.77921200419981, 8.5160515729399, 8.179209245790808, 7.7826243439997524, 7.282837454229498, 6.594669919226916, 5.755716685290205, 4.788941136599187, 3.6667806259373967, 3.37161671855678, 4.413365681101101, 5.319559439263477, 6.059734933640802, 6.5946699192269085, 7.0190094058300705, 7.385013603583527, 7.680284321463568, 7.916262866710099, 8.09519781500584, 8.220753908477324, 8.295185933912745, 8.319842223627582, 8.295185933912743, 8.220753908477322, 8.095197815005838, 7.916262866710101, 7.680284321463565, 7.385013603583527, 7.019009405830073, 6.5946699192269165, 6.059734933640807, 5.319559439263479, 4.413365681101102, 3.371616718556781, 3.026045587155099, 3.9865333116002355, 4.764985911789621, 5.319559439263474, 5.755716685290202, 6.136084850219482, 6.449966786612975, 6.708038228026869, 6.912828986796578, 7.068443611677276, 7.177617595002782, 7.242328637658115, 7.263768519691024, 7.242328637658114, 7.177617595002782, 7.068443611677276, 6.91282898679658, 6.708038228026866, 6.449966786612972, 6.136084850219481, 5.755716685290203, 5.319559439263478, 4.764985911789619, 3.9865333116002333, 3.0260455871550973, 2.6365819174880283, 3.433582366502954, 3.986533311600234, 4.413365681101106, 4.788941136599191, 5.104453801345162, 5.369230802445426, 5.585594884403354, 5.757717760781711, 5.888493704881652, 5.980248025842221, 6.034641916141872, 6.052663312839648, 6.034641916141873, 5.980248025842222, 5.88849370488165, 5.757717760781708, 5.58559488440335, 5.369230802445423, 5.104453801345157, 4.788941136599186, 4.4133656811011015, 3.986533311600232, 3.4335823665029483, 2.636581917488022, 2.113758276694795, 2.636581917488029, 3.0260455871551013, 3.3716167185567865, 3.666780625937402, 3.9188525448287326, 4.129290293281024, 4.30183938801502, 4.439134539756346, 4.543496011917359, 4.616741622074675, 4.6601688123480915, 4.674557938961102, 4.66016881234809, 4.616741622074674, 4.54349601191736, 4.4391345397563455, 4.301839388015018, 4.129290293281022, 3.91885254482873, 3.6667806259373985, 3.371616718556779, 3.0260455871550938, 2.6365819174880216, 2.11375827669479]], [[1.4767264367289454, 2.0536115658036045, 2.4501895371664633, 2.769756152635721, 3.0178250504254187, 3.2116431023155574, 3.360859644160609, 3.475513976480186, 3.5616249895027488, 3.6242461228006966, 3.666736503780116, 3.6913689043365716, 3.699440712242182, 3.6913689043365707, 3.666736503780114, 3.624246122800694, 3.561624989502746, 3.475513976480182, 3.3608596441606076, 3.211643102315556, 3.0178250504254156, 2.7697561526357193, 2.450189537166463, 2.0536115658036023, 1.476726436728944, 2.053611565803603, 2.9806508783667836, 3.594949481069355, 4.041192377459999, 4.399655593175661, 4.676062267018393, 4.891427824136701, 5.056008833791333, 5.18036209966473, 5.270748798231736, 5.332176789846988, 5.367815445130931, 5.379493675422171, 5.367815445130932, 5.3321767898469865, 5.2707487982317325, 5.180362099664724, 5.056008833791329, 4.891427824136698, 4.676062267018389, 4.399655593175656, 4.041192377459995, 3.5949494810693516, 2.9806508783667813, 2.053611565803603, 2.450189537166463, 3.594949481069355, 4.45557153192178, 5.048969549536415, 5.494034596665513, 5.85133278235618, 6.125118539119001, 6.337114918558057, 6.496329623276333, 6.612650566613521, 6.691642932786554, 6.737508894986761, 6.752551290663871, 6.737508894986762, 6.691642932786551, 6.6126505666135165, 6.496329623276329, 6.337114918558051, 6.125118539118993, 5.851332782356174, 5.494034596665506, 5.04896954953641, 4.455571531921775, 3.594949481069352, 2.4501895371664615, 2.769756152635722, 4.041192377459998, 5.048969549536414, 5.81893429279327, 6.364366237392115, 6.781718477169064, 7.115632644688468, 7.3685696691782505, 7.561517028803791, 7.701421995516999, 7.796908865110687, 7.85229656349987, 7.870455566936735, 7.852296563499866, 7.796908865110681, 7.701421995516996, 7.561517028803788, 7.368569669178243, 7.11563264468846, 6.781718477169059, 6.364366237392108, 5.818934292793264, 5.048969549536411, 4.041192377459995, 2.7697561526357193, 3.0178250504254174, 4.399655593175659, 5.49403459666551, 6.364366237392114, 7.038462665587848, 7.523991240371386, 7.899375123783939, 8.19699542212261, 8.41762546391737, 8.580784108132265, 8.69101588151382, 8.755320614100603, 8.776386094007659, 8.755320614100603, 8.691015881513819, 8.58078410813226, 8.417625463917357, 8.196995422122601, 7.8993751237839325, 7.523991240371379, 7.038462665587842, 6.364366237392106, 5.4940345966655055, 4.399655593175656, 3.017825050425416, 3.2116431023155574, 4.6760622670183905, 5.85133278235618, 6.781718477169063, 7.5239912403713864, 8.104277290662166, 8.525040454068225, 8.850655152516179, 9.104145446405983, 9.28471412881898, 9.409988549808515, 9.481974487228463, 9.505771074606805, 9.481974487228467, 9.409988549808512, 9.284714128818969, 9.10414544640597, 8.850655152516165, 8.525040454068211, 8.104277290662157, 7.523991240371374, 6.781718477169058, 5.851332782356172, 4.676062267018387, 3.2116431023155547, 3.360859644160608, 4.891427824136697, 6.125118539118997, 7.115632644688469, 7.899375123783939, 8.525040454068227, 9.016095450707015, 9.370641412825156, 9.642142018518644, 9.846394441040012, 9.981067128226861, 10.061616277791211, 10.087320592479987, 10.061616277791208, 9.981067128226854, 9.846394441040006, 9.64214201851863, 9.37064141282514, 9.016095450707004, 8.525040454068215, 7.899375123783932, 7.11563264468846, 6.125118539118992, 4.891427824136693, 3.3608596441606036, 3.4755139764801837, 5.056008833791331, 6.337114918558055, 7.36856966917825, 8.196995422122612, 8.850655152516179, 9.370641412825155, 9.777620689458715, 10.066035229392874, 10.28081219824272, 10.432034289079153, 10.515655792353899, 10.545275368624662, 10.515655792353899, 10.432034289079148, 10.280812198242712, 10.066035229392858, 9.777620689458702, 9.37064141282514, 8.850655152516167, 8.196995422122601, 7.368569669178241, 6.337114918558049, 5.056008833791327, 3.4755139764801806, 3.5616249895027474, 5.180362099664729, 6.496329623276332, 7.561517028803792, 8.41762546391737, 9.104145446405981, 9.642142018518644, 10.06603522939287, 10.393790492194215, 10.616656169989959, 10.772672174346603, 10.867813795448603, 10.894360150480162, 10.867813795448603, 10.772672174346596, 10.616656169989948, 10.3937904921942, 10.066035229392854, 9.64214201851863, 9.104145446405969, 8.41762546391736, 7.561517028803784, 6.496329623276324, 5.180362099664722, 3.5616249895027434, 3.6242461228006966, 5.270748798231736, 6.612650566613521, 7.701421995517, 8.580784108132269, 9.284714128818985, 9.846394441040017, 10.280812198242721, 10.616656169989957, 10.869209298799491, 11.027337059982218, 11.121638960398776, 11.158788722839008, 11.121638960398776, 11.02733705998221, 10.869209298799477, 10.616656169989946, 10.280812198242707, 9.846394441040001, 9.284714128818967, 8.580784108132256, 7.701421995516991, 6.612650566613514, 5.270748798231729, 3.6242461228006935, 3.666736503780114, 5.332176789846987, 6.6916429327865545, 7.796908865110688, 8.69101588151383, 9.409988549808519, 9.981067128226861, 10.432034289079157, 10.772672174346605, 11.02733705998222, 11.20686735617178, 11.30192763378054, 11.331229666432739, 11.301927633780531, 11.20686735617177, 11.027337059982205, 10.772672174346594, 10.432034289079148, 9.981067128226854, 9.409988549808507, 8.691015881513815, 7.796908865110677, 6.691642932786546, 5.332176789846984, 3.6667365037801107, 3.69136890433657, 5.367815445130929, 6.737508894986762, 7.852296563499869, 8.755320614100611, 9.481974487228475, 10.06161627779122, 10.515655792353908, 10.867813795448614, 11.12163896039878, 11.301927633780542, 11.409796082720524, 11.436653246156595, 11.409796082720513, 11.30192763378053, 11.12163896039877, 10.8678137954486, 10.515655792353895, 10.06161627779121, 9.481974487228463, 8.755320614100597, 7.852296563499859, 6.737508894986757, 5.367815445130929, 3.6913689043365685, 3.6994407122421826, 5.3794936754221725, 6.752551290663872, 7.870455566936736, 8.776386094007666, 9.505771074606814, 10.087320592479994, 10.545275368624665, 10.894360150480173, 11.158788722839018, 11.331229666432744, 11.436653246156599, 11.485906230578191, 11.436653246156594, 11.33122966643273, 11.158788722839, 10.89436015048016, 10.545275368624655, 10.087320592479983, 9.505771074606805, 8.776386094007654, 7.870455566936726, 6.752551290663868, 5.379493675422171, 3.6994407122421813, 3.691368904336571, 5.367815445130932, 6.737508894986764, 7.852296563499871, 8.75532061410061, 9.481974487228472, 10.061616277791215, 10.515655792353906, 10.867813795448614, 11.121638960398782, 11.301927633780544, 11.409796082720526, 11.436653246156595, 11.40979608272052, 11.301927633780533, 11.121638960398773, 10.867813795448596, 10.515655792353893, 10.06161627779121, 9.481974487228465, 8.7553206141006, 7.852296563499864, 6.73750889498676, 5.3678154451309315, 3.6913689043365703, 3.6667365037801134, 5.332176789846987, 6.6916429327865545, 7.79690886511069, 8.691015881513826, 9.409988549808523, 9.981067128226861, 10.432034289079157, 10.772672174346612, 11.027337059982218, 11.206867356171783, 11.301927633780545, 11.33122966643274, 11.301927633780535, 11.206867356171777, 11.02733705998221, 10.772672174346594, 10.432034289079148, 9.981067128226849, 9.409988549808507, 8.691015881513819, 7.796908865110685, 6.691642932786552, 5.332176789846987, 3.6667365037801125, 3.6242461228006957, 5.270748798231736, 6.612650566613523, 7.701421995517, 8.580784108132269, 9.284714128818983, 9.846394441040015, 10.280812198242723, 10.616656169989962, 10.869209298799493, 11.027337059982218, 11.121638960398778, 11.158788722839013, 11.121638960398782, 11.027337059982216, 10.869209298799483, 10.61665616998995, 10.280812198242712, 9.846394441040005, 9.284714128818973, 8.58078410813226, 7.701421995516997, 6.612650566613518, 5.270748798231733, 3.624246122800695, 3.561624989502747, 5.1803620996647295, 6.496329623276332, 7.561517028803793, 8.417625463917368, 9.104145446405983, 9.642142018518644, 10.06603522939287, 10.393790492194213, 10.616656169989959, 10.772672174346605, 10.867813795448605, 10.894360150480168, 10.86781379544861, 10.772672174346601, 10.616656169989954, 10.393790492194206, 10.06603522939286, 9.642142018518634, 9.104145446405973, 8.41762546391736, 7.561517028803788, 6.496329623276331, 5.1803620996647295, 3.5616249895027474, 3.475513976480184, 5.056008833791331, 6.337114918558059, 7.368569669178249, 8.19699542212261, 8.850655152516179, 9.370641412825155, 9.777620689458713, 10.066035229392867, 10.280812198242717, 10.432034289079153, 10.515655792353906, 10.545275368624662, 10.5156557923539, 10.432034289079152, 10.280812198242712, 10.066035229392861, 9.777620689458704, 9.370641412825144, 8.850655152516172, 8.196995422122603, 7.368569669178246, 6.337114918558055, 5.056008833791332, 3.4755139764801855, 3.3608596441606076, 4.891427824136698, 6.125118539118999, 7.115632644688468, 7.899375123783939, 8.525040454068225, 9.016095450707015, 9.370641412825153, 9.642142018518642, 9.846394441040012, 9.981067128226858, 10.061616277791218, 10.087320592479992, 10.061616277791213, 9.981067128226854, 9.846394441040008, 9.642142018518634, 9.370641412825144, 9.016095450707004, 8.525040454068215, 7.899375123783936, 7.115632644688464, 6.125118539118995, 4.891427824136698, 3.3608596441606085, 3.2116431023155565, 4.6760622670183905, 5.851332782356179, 6.781718477169064, 7.523991240371383, 8.104277290662164, 8.52504045406822, 8.850655152516177, 9.10414544640598, 9.28471412881898, 9.409988549808514, 9.48197448722847, 9.50577107460681, 9.481974487228467, 9.40998854980851, 9.284714128818974, 9.104145446405976, 8.850655152516172, 8.525040454068213, 8.104277290662159, 7.523991240371381, 6.781718477169062, 5.851332782356178, 4.6760622670183905, 3.2116431023155583, 3.0178250504254183, 4.39965559317566, 5.494034596665512, 6.364366237392111, 7.038462665587846, 7.52399124037138, 7.899375123783936, 8.196995422122605, 8.417625463917366, 8.580784108132264, 8.69101588151382, 8.755320614100606, 8.776386094007663, 8.7553206141006, 8.691015881513817, 8.58078410813226, 8.417625463917362, 8.196995422122601, 7.899375123783928, 7.523991240371378, 7.038462665587844, 6.364366237392112, 5.494034596665513, 4.399655593175662, 3.0178250504254196, 2.7697561526357215, 4.041192377460001, 5.048969549536414, 5.818934292793269, 6.364366237392113, 6.781718477169062, 7.115632644688465, 7.368569669178244, 7.561517028803786, 7.701421995516996, 7.796908865110683, 7.852296563499866, 7.87045556693673, 7.8522965634998645, 7.796908865110679, 7.701421995516995, 7.561517028803784, 7.36856966917824, 7.1156326446884615, 6.781718477169056, 6.364366237392108, 5.818934292793268, 5.048969549536415, 4.041192377460001, 2.769756152635721, 2.4501895371664633, 3.5949494810693534, 4.455571531921776, 5.048969549536413, 5.494034596665512, 5.851332782356177, 6.1251185391189935, 6.337114918558053, 6.496329623276328, 6.612650566613516, 6.6916429327865465, 6.737508894986757, 6.752551290663868, 6.7375088949867585, 6.691642932786549, 6.612650566613514, 6.496329623276323, 6.337114918558048, 6.125118539118991, 5.851332782356175, 5.494034596665507, 5.048969549536414, 4.455571531921776, 3.5949494810693525, 2.4501895371664637, 2.053611565803603, 2.9806508783667827, 3.594949481069354, 4.041192377460001, 4.399655593175661, 4.67606226701839, 4.891427824136697, 5.05600883379133, 5.180362099664725, 5.270748798231732, 5.332176789846985, 5.367815445130927, 5.37949367542217, 5.367815445130928, 5.332176789846985, 5.270748798231729, 5.1803620996647215, 5.056008833791325, 4.891427824136693, 4.676062267018388, 4.399655593175656, 4.041192377459996, 3.594949481069352, 2.9806508783667813, 2.0536115658036027, 1.476726436728944, 2.0536115658036027, 2.4501895371664637, 2.769756152635721, 3.0178250504254183, 3.2116431023155556, 3.3608596441606067, 3.4755139764801837, 3.5616249895027456, 3.6242461228006935, 3.666736503780112, 3.691368904336567, 3.6994407122421795, 3.691368904336569, 3.6667365037801116, 3.624246122800693, 3.5616249895027448, 3.47551397648018, 3.3608596441606036, 3.2116431023155547, 3.0178250504254165, 2.7697561526357197, 2.450189537166461, 2.0536115658036027, 1.4767264367289434]]]

        phi_test = np.array(phi_test)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            with self.subTest(l=l):
                phi = pydgm.state.phi[l, :, :].flatten()
                phi_zero_test = phi_test[:, l].flatten()
                np.testing.assert_array_almost_equal(phi, phi_zero_test, 7)

        self.angular_test()

    def test_dgmsolver_basic_2D_2g_1a_vacuum_eigen(self):
        '''
        Test for a basic 2 group problem
        '''
        self.set_mesh('simple')
        self.setGroups(2)
        self.setSolver('eigen')
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 0.0
        pydgm.control.boundary_north = 0.0
        pydgm.control.boundary_south = 0.0

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        assert(pydgm.control.number_fine_groups == 2)

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[0.2487746067595804, 0.30933535885463925, 0.36503957161582756, 0.4201277246266406, 0.4727429041064538, 0.5219789831678958, 0.5666972948641681, 0.6060354489662487, 0.6392379836284101, 0.6656917181028958, 0.6849238962362818, 0.6965974817500068, 0.7005110260557053, 0.6965974817500072, 0.6849238962362818, 0.6656917181028954, 0.6392379836284099, 0.6060354489662486, 0.5666972948641676, 0.5219789831678954, 0.47274290410645325, 0.4201277246266398, 0.3650395716158268, 0.30933535885463875, 0.24877460675958005, 0.3093353588546392, 0.40164987296136834, 0.48291526870614243, 0.5584914296565835, 0.6312485387266421, 0.6988077036867092, 0.7601167557294932, 0.813912097913677, 0.8592449558483541, 0.8953214292816684, 0.92152628635136, 0.9374235947881444, 0.9427517097445433, 0.9374235947881437, 0.9215262863513597, 0.8953214292816681, 0.8592449558483536, 0.8139120979136766, 0.7601167557294923, 0.698807703686708, 0.6312485387266414, 0.5584914296565827, 0.48291526870614165, 0.40164987296136784, 0.3093353588546388, 0.36503957161582756, 0.4829152687061422, 0.5944451912542015, 0.6940912164324666, 0.7860949620587045, 0.8723187135451134, 0.9500816973490145, 1.0183122596044265, 1.0757218667223183, 1.1213682123712332, 1.1545053937082992, 1.174599589924656, 1.1813330856367277, 1.1745995899246553, 1.1545053937082987, 1.121368212371233, 1.0757218667223176, 1.0183122596044252, 0.950081697349013, 0.8723187135451121, 0.7860949620587036, 0.694091216432466, 0.5944451912542007, 0.48291526870614176, 0.36503957161582723, 0.4201277246266406, 0.5584914296565837, 0.6940912164324667, 0.8215542114308743, 0.9352477382783926, 1.0386119631536603, 1.132791234434996, 1.2149670473854255, 1.284144202657454, 1.3390943477694461, 1.3789641487964708, 1.4031352581076166, 1.4112332698908574, 1.4031352581076166, 1.3789641487964701, 1.3390943477694455, 1.2841442026574532, 1.2149670473854237, 1.1327912344349942, 1.0386119631536588, 0.9352477382783915, 0.8215542114308731, 0.6940912164324659, 0.5584914296565832, 0.4201277246266401, 0.47274290410645387, 0.6312485387266424, 0.7860949620587048, 0.9352477382783928, 1.0736610472994341, 1.1957589812484655, 1.3044202797066318, 1.4002986998355933, 1.480566815593873, 1.5443844450713775, 1.590659651713007, 1.618703820457089, 1.6280998521423597, 1.6187038204570887, 1.5906596517130063, 1.5443844450713762, 1.4805668155938718, 1.400298699835592, 1.3044202797066304, 1.1957589812484641, 1.0736610472994326, 0.9352477382783914, 0.7860949620587037, 0.6312485387266413, 0.47274290410645303, 0.5219789831678961, 0.6988077036867091, 0.8723187135451133, 1.0386119631536603, 1.1957589812484652, 1.3391074138222738, 1.463235534944001, 1.5706511446747617, 1.661703820432259, 1.7336628995791248, 1.785911181959954, 1.8175629277589989, 1.8281624800473772, 1.8175629277589982, 1.7859111819599531, 1.7336628995791235, 1.6617038204322576, 1.5706511446747609, 1.463235534944001, 1.3391074138222734, 1.195758981248464, 1.0386119631536588, 0.872318713545112, 0.6988077036867077, 0.5219789831678953, 0.5666972948641688, 0.7601167557294939, 0.9500816973490143, 1.1327912344349955, 1.304420279706632, 1.4632355349440012, 1.6050104613189407, 1.7245450833554363, 1.8241342432960628, 1.9039815054710298, 1.9615421041215386, 1.9964802438076725, 2.0081817572886504, 1.9964802438076716, 1.9615421041215384, 1.9039815054710294, 1.8241342432960614, 1.724545083355435, 1.6050104613189402, 1.4632355349440005, 1.3044202797066307, 1.132791234434994, 0.9500816973490128, 0.760116755729492, 0.5666972948641678, 0.6060354489662493, 0.8139120979136776, 1.0183122596044265, 1.2149670473854246, 1.400298699835593, 1.5706511446747617, 1.724545083355436, 1.8581966015787008, 1.9666783752660877, 2.0522072216331457, 2.114975856603474, 2.152703108428814, 2.1653793244298796, 2.1527031084288137, 2.114975856603474, 2.0522072216331453, 1.9666783752660866, 1.8581966015786995, 1.7245450833554352, 1.570651144674761, 1.4002986998355924, 1.2149670473854235, 1.018312259604425, 0.8139120979136766, 0.6060354489662485, 0.6392379836284107, 0.859244955848355, 1.0757218667223187, 1.2841442026574539, 1.4805668155938723, 1.661703820432259, 1.8241342432960632, 1.9666783752660877, 2.0860050936007664, 2.1774982372953064, 2.243415389846528, 2.284049071070629, 2.297430887485442, 2.2840490710706294, 2.243415389846527, 2.1774982372953033, 2.086005093600764, 1.9666783752660864, 1.824134243296061, 1.6617038204322578, 1.4805668155938714, 1.2841442026574532, 1.0757218667223174, 0.8592449558483536, 0.6392379836284099, 0.6656917181028967, 0.8953214292816696, 1.1213682123712334, 1.3390943477694452, 1.5443844450713782, 1.7336628995791252, 1.9039815054710307, 2.0522072216331466, 2.177498237295307, 2.276984903871456, 2.3463849560806116, 2.3880835394753026, 2.402706436696479, 2.388083539475304, 2.3463849560806103, 2.276984903871452, 2.1774982372953047, 2.0522072216331444, 1.9039815054710287, 1.7336628995791237, 1.5443844450713762, 1.3390943477694448, 1.1213682123712327, 0.8953214292816684, 0.6656917181028957, 0.6849238962362824, 0.9215262863513608, 1.1545053937082996, 1.3789641487964708, 1.590659651713007, 1.7859111819599531, 1.9615421041215388, 2.1149758566034755, 2.2434153898465286, 2.3463849560806116, 2.4214652692362337, 2.4648625085214766, 2.4785988981976548, 2.464862508521476, 2.421465269236231, 2.34638495608061, 2.2434153898465263, 2.114975856603473, 1.9615421041215368, 1.7859111819599527, 1.590659651713006, 1.37896414879647, 1.1545053937082985, 0.9215262863513604, 0.684923896236282, 0.6965974817500075, 0.9374235947881446, 1.1745995899246564, 1.4031352581076177, 1.6187038204570894, 1.8175629277589984, 1.9964802438076719, 2.1527031084288146, 2.2840490710706303, 2.3880835394753044, 2.4648625085214766, 2.5118370812605506, 2.5268458313707605, 2.511837081260549, 2.4648625085214757, 2.3880835394753026, 2.2840490710706276, 2.152703108428813, 1.9964802438076716, 1.8175629277589982, 1.6187038204570894, 1.4031352581076162, 1.1745995899246557, 0.9374235947881441, 0.6965974817500072, 0.7005110260557053, 0.9427517097445437, 1.1813330856367281, 1.411233269890858, 1.6280998521423609, 1.8281624800473772, 2.008181757288651, 2.16537932442988, 2.2974308874854423, 2.4027064366964797, 2.478598898197655, 2.526845831370762, 2.544995566124512, 2.526845831370761, 2.478598898197654, 2.4027064366964774, 2.29743088748544, 2.1653793244298787, 2.008181757288651, 1.828162480047377, 1.6280998521423604, 1.4112332698908572, 1.1813330856367281, 0.942751709744544, 0.7005110260557053, 0.696597481750007, 0.9374235947881437, 1.1745995899246553, 1.4031352581076164, 1.6187038204570892, 1.817562927758998, 1.9964802438076719, 2.152703108428814, 2.284049071070629, 2.3880835394753035, 2.4648625085214775, 2.5118370812605497, 2.526845831370762, 2.511837081260549, 2.4648625085214744, 2.388083539475302, 2.284049071070627, 2.1527031084288137, 1.9964802438076714, 1.8175629277589977, 1.6187038204570898, 1.403135258107617, 1.1745995899246562, 0.9374235947881443, 0.6965974817500074, 0.6849238962362815, 0.9215262863513597, 1.1545053937082983, 1.37896414879647, 1.5906596517130056, 1.785911181959953, 1.9615421041215382, 2.1149758566034746, 2.2434153898465277, 2.346384956080611, 2.4214652692362324, 2.464862508521476, 2.478598898197654, 2.464862508521475, 2.4214652692362306, 2.34638495608061, 2.2434153898465268, 2.114975856603473, 1.9615421041215373, 1.7859111819599534, 1.5906596517130067, 1.3789641487964701, 1.154505393708299, 0.9215262863513605, 0.6849238962362822, 0.6656917181028956, 0.8953214292816684, 1.1213682123712325, 1.3390943477694441, 1.5443844450713762, 1.7336628995791235, 1.9039815054710296, 2.0522072216331453, 2.177498237295306, 2.2769849038714542, 2.3463849560806107, 2.388083539475303, 2.4027064366964783, 2.3880835394753013, 2.3463849560806094, 2.2769849038714542, 2.1774982372953042, 2.0522072216331435, 1.903981505471029, 1.733662899579124, 1.5443844450713766, 1.3390943477694452, 1.1213682123712334, 0.8953214292816688, 0.6656917181028961, 0.6392379836284101, 0.8592449558483539, 1.0757218667223172, 1.2841442026574528, 1.4805668155938716, 1.6617038204322572, 1.824134243296061, 1.966678375266086, 2.0860050936007646, 2.177498237295305, 2.243415389846527, 2.2840490710706276, 2.2974308874854406, 2.2840490710706276, 2.2434153898465268, 2.1774982372953056, 2.0860050936007637, 1.966678375266086, 1.8241342432960603, 1.6617038204322567, 1.4805668155938712, 1.2841442026574534, 1.0757218667223183, 0.8592449558483548, 0.6392379836284101, 0.6060354489662485, 0.8139120979136767, 1.0183122596044252, 1.214967047385424, 1.4002986998355924, 1.5706511446747604, 1.724545083355434, 1.8581966015786986, 1.966678375266085, 2.052207221633143, 2.114975856603472, 2.1527031084288124, 2.165379324429878, 2.152703108428813, 2.1149758566034733, 2.0522072216331444, 1.9666783752660861, 1.8581966015786993, 1.7245450833554339, 1.5706511446747597, 1.400298699835592, 1.2149670473854246, 1.018312259604426, 0.8139120979136778, 0.6060354489662493, 0.5666972948641678, 0.7601167557294926, 0.9500816973490129, 1.132791234434994, 1.3044202797066307, 1.4632355349439998, 1.6050104613189393, 1.7245450833554337, 1.8241342432960592, 1.9039815054710265, 1.9615421041215355, 1.99648024380767, 2.0081817572886496, 1.9964802438076705, 1.9615421041215368, 1.9039815054710283, 1.8241342432960603, 1.724545083355434, 1.6050104613189389, 1.4632355349439996, 1.3044202797066304, 1.1327912344349949, 0.9500816973490139, 0.7601167557294932, 0.5666972948641683, 0.5219789831678955, 0.6988077036867083, 0.8723187135451121, 1.0386119631536588, 1.1957589812484641, 1.3391074138222725, 1.4632355349439998, 1.5706511446747595, 1.6617038204322556, 1.733662899579122, 1.7859111819599514, 1.8175629277589964, 1.828162480047376, 1.8175629277589962, 1.785911181959952, 1.7336628995791221, 1.6617038204322565, 1.5706511446747602, 1.4632355349439996, 1.3391074138222723, 1.195758981248464, 1.0386119631536592, 0.8723187135451124, 0.6988077036867084, 0.5219789831678958, 0.47274290410645325, 0.6312485387266413, 0.7860949620587037, 0.9352477382783914, 1.0736610472994323, 1.1957589812484632, 1.3044202797066298, 1.4002986998355915, 1.4805668155938698, 1.5443844450713744, 1.5906596517130036, 1.618703820457087, 1.628099852142358, 1.6187038204570874, 1.5906596517130043, 1.5443844450713748, 1.48056681559387, 1.400298699835591, 1.3044202797066302, 1.1957589812484637, 1.0736610472994326, 0.9352477382783915, 0.7860949620587039, 0.6312485387266413, 0.4727429041064536, 0.42012772462664005, 0.558491429656583, 0.6940912164324656, 0.821554211430873, 0.9352477382783911, 1.0386119631536583, 1.1327912344349942, 1.214967047385423, 1.2841442026574517, 1.3390943477694435, 1.3789641487964681, 1.4031352581076144, 1.411233269890855, 1.4031352581076149, 1.378964148796469, 1.3390943477694435, 1.2841442026574519, 1.214967047385423, 1.1327912344349942, 1.0386119631536583, 0.9352477382783914, 0.8215542114308733, 0.6940912164324659, 0.5584914296565829, 0.4201277246266399, 0.3650395716158271, 0.4829152687061416, 0.5944451912542004, 0.6940912164324654, 0.7860949620587033, 0.872318713545112, 0.950081697349013, 1.0183122596044247, 1.0757218667223167, 1.1213682123712319, 1.1545053937082967, 1.1745995899246537, 1.181333085636726, 1.1745995899246535, 1.1545053937082974, 1.1213682123712316, 1.0757218667223165, 1.0183122596044245, 0.9500816973490126, 0.8723187135451116, 0.7860949620587037, 0.6940912164324659, 0.5944451912542006, 0.4829152687061415, 0.36503957161582695, 0.3093353588546387, 0.4016498729613677, 0.4829152687061415, 0.5584914296565826, 0.6312485387266412, 0.6988077036867079, 0.7601167557294921, 0.8139120979136765, 0.8592449558483536, 0.8953214292816674, 0.9215262863513591, 0.9374235947881425, 0.9427517097445415, 0.9374235947881424, 0.9215262863513588, 0.8953214292816672, 0.8592449558483534, 0.8139120979136764, 0.760116755729492, 0.6988077036867076, 0.6312485387266412, 0.5584914296565827, 0.4829152687061414, 0.4016498729613678, 0.3093353588546387, 0.24877460675957994, 0.3093353588546386, 0.36503957161582684, 0.42012772462663983, 0.47274290410645287, 0.5219789831678951, 0.5666972948641678, 0.6060354489662485, 0.6392379836284093, 0.6656917181028953, 0.6849238962362811, 0.6965974817500061, 0.7005110260557043, 0.6965974817500061, 0.684923896236281, 0.6656917181028952, 0.6392379836284097, 0.606035448966248, 0.5666972948641674, 0.5219789831678951, 0.47274290410645314, 0.42012772462663994, 0.3650395716158269, 0.3093353588546386, 0.24877460675958013]], [[0.01998505220506663, 0.028624451099833366, 0.03645151076496133, 0.04399260475136248, 0.05100072704478553, 0.05739396487095979, 0.06307779967625268, 0.067997245315577, 0.07209514838534635, 0.07532933795436803, 0.0776647807484596, 0.07907628103283944, 0.07954850885838675, 0.07907628103283944, 0.07766478074845957, 0.075329337954368, 0.07209514838534632, 0.06799724531557698, 0.06307779967625264, 0.057393964870959754, 0.05100072704478547, 0.043992604751362394, 0.036451510764961274, 0.028624451099833345, 0.019985052205066602, 0.028624451099833387, 0.04289219826344366, 0.05537627658493564, 0.06683400942917612, 0.07756460151772676, 0.08729947508349442, 0.09596646083873438, 0.10345581477898921, 0.10969794646838663, 0.11462311713123982, 0.1181799334309722, 0.12032968475616068, 0.12104887790598191, 0.12032968475616067, 0.1181799334309722, 0.11462311713123977, 0.1096979464683866, 0.10345581477898919, 0.09596646083873432, 0.08729947508349438, 0.07756460151772669, 0.06683400942917608, 0.055376276584935595, 0.04289219826344358, 0.02862445109983335, 0.036451510764961344, 0.05537627658493567, 0.07281240864873842, 0.08833946431758972, 0.10251035073030715, 0.11547966534412975, 0.12697133089098303, 0.13691796221646127, 0.1451970438988955, 0.1517319977604755, 0.1564501107528323, 0.15930170090015233, 0.16025573986215147, 0.15930170090015233, 0.15645011075283227, 0.1517319977604754, 0.1451970438988954, 0.13691796221646116, 0.126971330890983, 0.11547966534412962, 0.10251035073030706, 0.08833946431758967, 0.07281240864873835, 0.0553762765849356, 0.03645151076496128, 0.04399260475136246, 0.06683400942917615, 0.08833946431758974, 0.10812515410923546, 0.12576424655216117, 0.14166281810818404, 0.15587030482414796, 0.16810882465046229, 0.17831532245918838, 0.18636112398825838, 0.19217229897749483, 0.19568374851974488, 0.1968584242752685, 0.19568374851974477, 0.1921722989774947, 0.1863611239882583, 0.1783153224591883, 0.16810882465046217, 0.15587030482414788, 0.1416628181081839, 0.12576424655216112, 0.10812515410923533, 0.08833946431758966, 0.06683400942917607, 0.04399260475136238, 0.05100072704478551, 0.07756460151772678, 0.10251035073030718, 0.1257642465521612, 0.14699529578237652, 0.16577082703609106, 0.18238584389256482, 0.19681538872095647, 0.20878732915049025, 0.2182473780344291, 0.225070203331361, 0.22919492827762505, 0.23057461282062075, 0.22919492827762494, 0.22507020333136094, 0.21824737803442895, 0.20878732915049006, 0.19681538872095633, 0.18238584389256474, 0.16577082703609097, 0.14699529578237644, 0.1257642465521611, 0.10251035073030706, 0.07756460151772662, 0.05100072704478543, 0.057393964870959775, 0.08729947508349441, 0.11547966534412972, 0.14166281810818399, 0.16577082703609108, 0.18750266335421473, 0.20642199389721985, 0.22274385294262125, 0.23639580825188378, 0.24712057785504718, 0.2548797983308887, 0.2595620678850089, 0.26112952299654413, 0.25956206788500885, 0.25487979833088864, 0.24712057785504704, 0.23639580825188358, 0.22274385294262108, 0.20642199389721977, 0.1875026633542146, 0.16577082703609095, 0.14166281810818385, 0.11547966534412953, 0.08729947508349425, 0.05739396487095971, 0.06307779967625268, 0.0959664608387344, 0.126971330890983, 0.15587030482414793, 0.1823858438925648, 0.2064219938972199, 0.22769822236798304, 0.24578147642763298, 0.2608347764396368, 0.2727617209937381, 0.28132937120877516, 0.28652333758737614, 0.2882555932407505, 0.28652333758737614, 0.28132937120877494, 0.2727617209937379, 0.2608347764396366, 0.24578147642763265, 0.22769822236798287, 0.2064219938972197, 0.18238584389256465, 0.1558703048241477, 0.1269713308909828, 0.09596646083873425, 0.0630777996762526, 0.06799724531557702, 0.10345581477898926, 0.13691796221646121, 0.16810882465046226, 0.19681538872095627, 0.22274385294262114, 0.24578147642763293, 0.26566827591237996, 0.28198376832678235, 0.2948636707750291, 0.3042076650579847, 0.30981520997629597, 0.3117071758004514, 0.3098152099762959, 0.3042076650579845, 0.29486367077502895, 0.2819837683267821, 0.26566827591237957, 0.24578147642763262, 0.22274385294262097, 0.1968153887209561, 0.16810882465046204, 0.13691796221646108, 0.10345581477898912, 0.06799724531557694, 0.07209514838534634, 0.10969794646838667, 0.14519704389889548, 0.1783153224591884, 0.20878732915049014, 0.23639580825188367, 0.2608347764396368, 0.2819837683267825, 0.29960847209767005, 0.31331052795118874, 0.32321880946406617, 0.3292494667036456, 0.33122850555567945, 0.3292494667036456, 0.323218809464066, 0.31331052795118847, 0.29960847209766955, 0.28198376832678207, 0.2608347764396365, 0.2363958082518834, 0.20878732915049, 0.17831532245918816, 0.14519704389889518, 0.10969794646838649, 0.07209514838534624, 0.07532933795436803, 0.11462311713123982, 0.15173199776047552, 0.1863611239882584, 0.2182473780344291, 0.24712057785504715, 0.27276172099373797, 0.29486367077502923, 0.31331052795118886, 0.3278999450140935, 0.3382680210151431, 0.3445408145247518, 0.3466961419642437, 0.34454081452475166, 0.33826802101514275, 0.327899945014093, 0.3133105279511884, 0.29486367077502884, 0.2727617209937377, 0.2471205778550469, 0.21824737803442884, 0.18636112398825816, 0.15173199776047527, 0.11462311713123965, 0.07532933795436791, 0.07766478074845957, 0.11817993343097222, 0.1564501107528323, 0.19217229897749483, 0.22507020333136107, 0.25487979833088875, 0.28132937120877516, 0.3042076650579848, 0.32321880946406634, 0.3382680210151432, 0.3491782796062605, 0.3556589542210299, 0.357796817251322, 0.3556589542210296, 0.34917827960626024, 0.33826802101514264, 0.32321880946406584, 0.3042076650579843, 0.2813293712087748, 0.2548797983308885, 0.22507020333136074, 0.19217229897749455, 0.15645011075283208, 0.11817993343097205, 0.07766478074845946, 0.07907628103283938, 0.12032968475616064, 0.15930170090015233, 0.1956837485197449, 0.22919492827762514, 0.2595620678850089, 0.2865233375873762, 0.3098152099762961, 0.32924946670364585, 0.344540814524752, 0.35565895422102983, 0.36240803074961725, 0.36459041779744544, 0.36240803074961697, 0.35565895422102956, 0.3445408145247515, 0.3292494667036454, 0.30981520997629564, 0.2865233375873758, 0.2595620678850086, 0.2291949282776248, 0.19568374851974465, 0.1593017009001521, 0.12032968475616053, 0.07907628103283933, 0.0795485088583867, 0.12104887790598186, 0.16025573986215144, 0.19685842427526862, 0.23057461282062083, 0.2611295229965441, 0.2882555932407507, 0.3117071758004516, 0.3312285055556798, 0.3466961419642438, 0.3577968172513219, 0.36459041779744533, 0.36700813351718253, 0.3645904177974454, 0.3577968172513218, 0.3466961419642433, 0.3312285055556792, 0.31170717580045104, 0.28825559324075023, 0.2611295229965439, 0.23057461282062056, 0.19685842427526837, 0.16025573986215128, 0.12104887790598169, 0.07954850885838664, 0.07907628103283937, 0.1203296847561606, 0.15930170090015222, 0.19568374851974482, 0.229194928277625, 0.2595620678850089, 0.28652333758737614, 0.30981520997629614, 0.3292494667036457, 0.34454081452475166, 0.3556589542210297, 0.3624080307496171, 0.3645904177974454, 0.3624080307496171, 0.3556589542210296, 0.34454081452475144, 0.3292494667036452, 0.3098152099762958, 0.2865233375873758, 0.25956206788500863, 0.22919492827762492, 0.19568374851974468, 0.15930170090015214, 0.12032968475616056, 0.0790762810328393, 0.07766478074845959, 0.11817993343097216, 0.15645011075283216, 0.19217229897749472, 0.2250702033313609, 0.2548797983308887, 0.2813293712087751, 0.30420766505798474, 0.32321880946406617, 0.33826802101514303, 0.34917827960626036, 0.3556589542210297, 0.35779681725132184, 0.35565895422102967, 0.3491782796062603, 0.3382680210151428, 0.3232188094640659, 0.3042076650579844, 0.2813293712087748, 0.2548797983308885, 0.22507020333136085, 0.1921722989774947, 0.1564501107528321, 0.11817993343097208, 0.07766478074845945, 0.075329337954368, 0.11462311713123972, 0.15173199776047533, 0.18636112398825833, 0.2182473780344291, 0.2471205778550471, 0.2727617209937379, 0.2948636707750291, 0.3133105279511887, 0.3278999450140934, 0.33826802101514303, 0.34454081452475155, 0.34669614196424364, 0.3445408145247516, 0.33826802101514286, 0.32789994501409314, 0.3133105279511885, 0.2948636707750289, 0.2727617209937377, 0.24712057785504687, 0.21824737803442884, 0.18636112398825821, 0.15173199776047533, 0.11462311713123967, 0.07532933795436793, 0.0720951483853463, 0.10969794646838658, 0.14519704389889537, 0.17831532245918832, 0.20878732915049014, 0.2363958082518836, 0.2608347764396367, 0.28198376832678224, 0.29960847209766983, 0.3133105279511886, 0.32321880946406595, 0.3292494667036455, 0.3312285055556795, 0.32924946670364563, 0.32321880946406606, 0.3133105279511885, 0.2996084720976697, 0.28198376832678207, 0.2608347764396365, 0.2363958082518834, 0.20878732915049, 0.17831532245918827, 0.14519704389889534, 0.10969794646838664, 0.07209514838534628, 0.06799724531557692, 0.10345581477898914, 0.13691796221646116, 0.1681088246504622, 0.1968153887209563, 0.22274385294262114, 0.24578147642763282, 0.26566827591237974, 0.28198376832678224, 0.294863670775029, 0.30420766505798447, 0.30981520997629586, 0.3117071758004514, 0.309815209976296, 0.3042076650579846, 0.29486367077502895, 0.28198376832678207, 0.26566827591237946, 0.24578147642763268, 0.22274385294262103, 0.1968153887209562, 0.16810882465046217, 0.1369179622164611, 0.10345581477898914, 0.06799724531557695, 0.06307779967625264, 0.09596646083873434, 0.12697133089098292, 0.15587030482414788, 0.18238584389256474, 0.2064219938972198, 0.22769822236798282, 0.24578147642763276, 0.2608347764396366, 0.2727617209937378, 0.28132937120877494, 0.2865233375873759, 0.2882555932407505, 0.28652333758737614, 0.28132937120877494, 0.2727617209937377, 0.26083477643963643, 0.2457814764276326, 0.22769822236798273, 0.20642199389721966, 0.18238584389256474, 0.1558703048241478, 0.12697133089098286, 0.0959664608387343, 0.06307779967625263, 0.057393964870959734, 0.08729947508349435, 0.1154796653441296, 0.1416628181081839, 0.165770827036091, 0.1875026633542146, 0.20642199389721977, 0.22274385294262092, 0.23639580825188347, 0.24712057785504693, 0.25487979833088853, 0.25956206788500874, 0.2611295229965441, 0.2595620678850088, 0.25487979833088864, 0.24712057785504699, 0.2363958082518835, 0.22274385294262092, 0.2064219938972196, 0.18750266335421453, 0.16577082703609092, 0.14166281810818385, 0.11547966534412955, 0.08729947508349435, 0.05739396487095975, 0.05100072704478549, 0.07756460151772669, 0.10251035073030709, 0.12576424655216104, 0.14699529578237644, 0.16577082703609092, 0.18238584389256463, 0.19681538872095608, 0.20878732915048992, 0.2182473780344288, 0.22507020333136082, 0.22919492827762483, 0.23057461282062058, 0.2291949282776249, 0.22507020333136082, 0.21824737803442895, 0.20878732915049003, 0.19681538872095614, 0.18238584389256465, 0.16577082703609092, 0.14699529578237638, 0.125764246552161, 0.10251035073030706, 0.07756460151772666, 0.05100072704478548, 0.04399260475136242, 0.06683400942917608, 0.08833946431758964, 0.10812515410923533, 0.12576424655216106, 0.1416628181081838, 0.15587030482414774, 0.16810882465046206, 0.1783153224591881, 0.18636112398825808, 0.19217229897749458, 0.19568374851974465, 0.19685842427526834, 0.19568374851974465, 0.19217229897749458, 0.1863611239882582, 0.17831532245918824, 0.1681088246504621, 0.1558703048241478, 0.14166281810818382, 0.12576424655216106, 0.10812515410923529, 0.08833946431758967, 0.06683400942917608, 0.043992604751362414, 0.03645151076496128, 0.055376276584935595, 0.07281240864873831, 0.0883394643175896, 0.102510350730307, 0.11547966534412953, 0.12697133089098286, 0.13691796221646105, 0.14519704389889526, 0.15173199776047522, 0.15645011075283208, 0.1593017009001521, 0.16025573986215125, 0.1593017009001521, 0.1564501107528321, 0.15173199776047538, 0.14519704389889532, 0.1369179622164611, 0.12697133089098284, 0.11547966534412958, 0.10251035073030708, 0.08833946431758967, 0.07281240864873834, 0.055376276584935574, 0.036451510764961274, 0.028624451099833324, 0.04289219826344357, 0.055376276584935574, 0.06683400942917601, 0.07756460151772661, 0.08729947508349428, 0.09596646083873418, 0.10345581477898907, 0.10969794646838647, 0.11462311713123963, 0.11817993343097204, 0.12032968475616049, 0.12104887790598168, 0.12032968475616052, 0.11817993343097209, 0.11462311713123971, 0.10969794646838653, 0.10345581477898909, 0.09596646083873425, 0.0872994750834943, 0.07756460151772665, 0.06683400942917607, 0.0553762765849356, 0.04289219826344358, 0.028624451099833324, 0.01998505220506658, 0.028624451099833328, 0.03645151076496123, 0.04399260475136236, 0.05100072704478543, 0.05739396487095966, 0.06307779967625257, 0.06799724531557688, 0.07209514838534621, 0.07532933795436791, 0.07766478074845945, 0.07907628103283927, 0.07954850885838662, 0.07907628103283931, 0.07766478074845949, 0.0753293379543679, 0.07209514838534624, 0.0679972453155769, 0.06307779967625259, 0.05739396487095971, 0.05100072704478544, 0.0439926047513624, 0.036451510764961274, 0.028624451099833328, 0.019985052205066585]]]

        phi_test = np.array(phi_test)

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, 0.23385815, 8)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            with self.subTest(l=l):
                phi = pydgm.state.phi[l, :, :].flatten()
                phi_zero_test = phi_test[:, l].flatten() / np.linalg.norm(phi_test[:, l]) * np.linalg.norm(phi)
                np.testing.assert_array_almost_equal(phi, phi_zero_test, 7)

        self.angular_test()

    def test_dgmsolver_partisn_eigen_2g_l0_simple(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        self.setSolver('eigen')
        self.set_mesh('slab')
        self.setGroups(2)
        pydgm.control.scatter_leg_order = 0
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 0.0
        pydgm.control.boundary_north = 1.0
        pydgm.control.boundary_south = 1.0

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        assert(pydgm.control.number_fine_groups == 2)

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[2.0222116180374377, 3.906659286632303, 5.489546165510137, 6.624935332662817, 7.217123379989135, 7.217123379989135, 6.624935332662812, 5.4895461655101325, 3.9066592866322987, 2.022211618037434, 2.0222116180374723, 3.906659286632373, 5.489546165510238, 6.624935332662936, 7.217123379989267, 7.217123379989266, 6.624935332662933, 5.489546165510232, 3.906659286632369, 2.02221161803747, 2.0222116180375425, 3.906659286632511, 5.4895461655104345, 6.624935332663176, 7.2171233799895305, 7.217123379989527, 6.624935332663175, 5.489546165510432, 3.9066592866325083, 2.02221161803754, 2.0222116180376455, 3.9066592866327152, 5.489546165510723, 6.6249353326635285, 7.217123379989911, 7.21712337998991, 6.624935332663527, 5.48954616551072, 3.9066592866327117, 2.022211618037643, 2.022211618037778, 3.9066592866329786, 5.489546165511096, 6.62493533266398, 7.217123379990407, 7.217123379990406, 6.624935332663978, 5.489546165511094, 3.906659286632976, 2.022211618037777, 2.022211618037936, 3.906659286633293, 5.489546165511546, 6.624935332664524, 7.217123379991003, 7.217123379991001, 6.62493533266452, 5.489546165511541, 3.9066592866332917, 2.022211618037936, 2.02221161803812, 3.9066592866336483, 5.489546165512049, 6.624935332665141, 7.217123379991674, 7.217123379991672, 6.624935332665138, 5.489546165512053, 3.906659286633648, 2.0222116180381193, 2.02221161803832, 3.9066592866340404, 5.489546165512603, 6.624935332665814, 7.217123379992408, 7.217123379992409, 6.6249353326658165, 5.489546165512604, 3.906659286634039, 2.0222116180383196, 2.0222116180385177, 3.9066592866344605, 5.489546165513193, 6.624935332666522, 7.217123379993187, 7.217123379993187, 6.624935332666523, 5.489546165513196, 3.9066592866344605, 2.022211618038519, 2.0222116180387206, 3.9066592866348797, 5.489546165513811, 6.624935332667246, 7.217123379993982, 7.217123379993986, 6.624935332667246, 5.48954616551381, 3.9066592866348824, 2.0222116180387224, 2.0222116180389453, 3.906659286635285, 5.489546165514407, 6.6249353326679925, 7.217123379994771, 7.217123379994772, 6.624935332667999, 5.48954616551441, 3.906659286635288, 2.022211618038948, 2.0222116180391927, 3.906659286635688, 5.48954616551496, 6.624935332668722, 7.217123379995537, 7.217123379995538, 6.624935332668729, 5.489546165514966, 3.9066592866356924, 2.0222116180391954, 2.022211618039421, 3.906659286636115, 5.489546165515496, 6.624935332669343, 7.217123379996293, 7.217123379996299, 6.624935332669352, 5.489546165515503, 3.906659286636122, 2.022211618039423, 2.02221161803949, 3.9066592866365792, 5.489546165516017, 6.62493533266986, 7.217123379997003, 7.217123379997005, 6.624935332669865, 5.489546165516022, 3.9066592866365855, 2.0222116180394942, 2.0222116180396172, 3.9066592866368017, 5.489546165516571, 6.624935332670406, 7.217123379997539, 7.217123379997541, 6.624935332670408, 5.489546165516575, 3.9066592866368084, 2.022211618039622, 2.0222116180400502, 3.9066592866368364, 5.489546165516934, 6.624935332671166, 7.217123379997814, 7.217123379997816, 6.62493533267117, 5.489546165516938, 3.9066592866368413, 2.0222116180400542, 2.0222116180400302, 3.9066592866373284, 5.489546165517048, 6.624935332671695, 7.217123379998117, 7.217123379998122, 6.624935332671701, 5.489546165517053, 3.9066592866373346, 2.022211618040033, 2.0222116180394636, 3.9066592866377623, 5.4895461655177265, 6.624935332671153, 7.217123379998839, 7.217123379998841, 6.624935332671155, 5.489546165517731, 3.9066592866377667, 2.022211618039466, 2.022211618039759, 3.9066592866374505, 5.489546165518068, 6.624935332670921, 7.217123379999226, 7.217123379999228, 6.624935332670927, 5.489546165518074, 3.906659286637456, 2.0222116180397633, 2.0222116180410867, 3.9066592866369314, 5.489546165516315, 6.624935332671973, 7.217123379998163, 7.217123379998167, 6.6249353326719795, 5.489546165516322, 3.9066592866369385, 2.0222116180410916]], [[0.2601203722850905, 0.5582585327512433, 0.7906322218586104, 0.9549760069773039, 1.040451416793767, 1.0404514167937677, 0.9549760069773037, 0.7906322218586092, 0.5582585327512429, 0.26012037228509016, 0.260120372285095, 0.5582585327512531, 0.7906322218586241, 0.9549760069773205, 1.040451416793786, 1.0404514167937855, 0.9549760069773202, 0.7906322218586236, 0.5582585327512527, 0.2601203722850945, 0.260120372285104, 0.5582585327512725, 0.7906322218586517, 0.9549760069773544, 1.0404514167938226, 1.0404514167938221, 0.9549760069773543, 0.7906322218586515, 0.558258532751272, 0.2601203722851037, 0.26012037228511703, 0.5582585327513011, 0.7906322218586924, 0.9549760069774041, 1.040451416793876, 1.0404514167938763, 0.9549760069774037, 0.7906322218586922, 0.5582585327513006, 0.26012037228511686, 0.26012037228513407, 0.5582585327513375, 0.7906322218587449, 0.954976006977468, 1.040451416793946, 1.040451416793946, 0.9549760069774672, 0.7906322218587444, 0.5582585327513372, 0.26012037228513396, 0.26012037228515444, 0.5582585327513817, 0.7906322218588078, 0.954976006977544, 1.0404514167940295, 1.040451416794029, 0.9549760069775434, 0.7906322218588072, 0.5582585327513814, 0.2601203722851543, 0.2601203722851778, 0.5582585327514316, 0.790632221858879, 0.9549760069776305, 1.0404514167941241, 1.040451416794124, 0.9549760069776303, 0.7906322218588783, 0.5582585327514313, 0.26012037228517765, 0.26012037228520346, 0.558258532751486, 0.7906322218589572, 0.9549760069777256, 1.0404514167942274, 1.0404514167942271, 0.9549760069777252, 0.7906322218589572, 0.5582585327514857, 0.26012037228520346, 0.2601203722852295, 0.5582585327515446, 0.7906322218590393, 0.9549760069778253, 1.0404514167943364, 1.0404514167943366, 0.9549760069778253, 0.7906322218590391, 0.5582585327515445, 0.2601203722852295, 0.26012037228525536, 0.5582585327516034, 0.790632221859125, 0.9549760069779274, 1.0404514167944487, 1.0404514167944485, 0.9549760069779275, 0.7906322218591252, 0.5582585327516035, 0.26012037228525536, 0.2601203722852845, 0.5582585327516609, 0.7906322218592101, 0.9549760069780301, 1.0404514167945595, 1.0404514167945598, 0.9549760069780304, 0.7906322218592103, 0.5582585327516612, 0.2601203722852847, 0.26012037228531204, 0.5582585327517199, 0.7906322218592873, 0.9549760069781328, 1.0404514167946664, 1.0404514167946664, 0.9549760069781337, 0.7906322218592882, 0.5582585327517203, 0.26012037228531243, 0.2601203722853439, 0.5582585327517724, 0.7906322218593651, 0.9549760069782204, 1.0404514167947703, 1.0404514167947703, 0.9549760069782209, 0.7906322218593661, 0.5582585327517732, 0.2601203722853444, 0.26012037228535895, 0.5582585327518332, 0.7906322218594308, 0.954976006978293, 1.0404514167948657, 1.0404514167948657, 0.954976006978294, 0.7906322218594317, 0.5582585327518342, 0.26012037228535934, 0.26012037228535956, 0.5582585327518739, 0.7906322218594933, 0.9549760069783544, 1.0404514167949381, 1.0404514167949384, 0.954976006978355, 0.7906322218594941, 0.5582585327518749, 0.26012037228535995, 0.2601203722854187, 0.5582585327518456, 0.7906322218595377, 0.9549760069784322, 1.0404514167949561, 1.0404514167949563, 0.9549760069784329, 0.7906322218595386, 0.5582585327518462, 0.2601203722854192, 0.26012037228539797, 0.5582585327518799, 0.790632221859497, 0.9549760069785007, 1.0404514167949224, 1.040451416794923, 0.9549760069785015, 0.790632221859498, 0.5582585327518805, 0.2601203722853983, 0.26012037228518786, 0.5582585327519209, 0.7906322218595863, 0.9549760069782882, 1.0404514167949557, 1.040451416794956, 0.9549760069782894, 0.7906322218595871, 0.5582585327519218, 0.2601203722851882, 0.2601203722851841, 0.5582585327518561, 0.7906322218597195, 0.954976006978139, 1.0404514167950512, 1.0404514167950507, 0.9549760069781394, 0.7906322218597205, 0.5582585327518564, 0.26012037228518436, 0.2601203722858843, 0.5582585327521365, 0.7906322218596814, 0.9549760069791707, 1.0404514167954841, 1.0404514167954846, 0.9549760069791717, 0.7906322218596826, 0.5582585327521374, 0.2601203722858849]]]

        phi_test = np.array(phi_test)

        keff_test = 0.88662575

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            with self.subTest(l=l):
                phi = pydgm.state.phi[l, :, :].flatten()
                phi_zero_test = phi_test[:, l].flatten() / np.linalg.norm(phi_test[:, l]) * np.linalg.norm(phi)
                np.testing.assert_array_almost_equal(phi, phi_zero_test, 8)

        self.angular_test()


if __name__ == '__main__':

    unittest.main()
