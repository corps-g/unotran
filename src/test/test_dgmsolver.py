import sys
from numpy.ma.testutils import assert_almost_equal
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestDGMSOLVER(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.recon_print = False
        pydgm.control.eigen_print = False
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.recon_tolerance = 1e-14
        pydgm.control.eigen_tolerance = 1e-14
        pydgm.control.outer_tolerance = 1e-15
        pydgm.control.inner_tolerance = 1e-15
        pydgm.control.lamb = 1.0
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = True
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 0
        pydgm.control.ignore_warnings = True
        pydgm.control.max_recon_iters = 1000
        pydgm.control.max_eigen_iters = 1000
        pydgm.control.max_outer_iters = 100
        pydgm.control.max_inner_iters = 100

    # Define methods to set various variables for the tests

    def setGroups(self, G):
        if G == 2:
            pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)
            pydgm.control.dgm_basis_name = 'test/2gbasis'.ljust(256)
        elif G == 4:
            pydgm.control.xs_name = 'test/4gXS.anlxs'.ljust(256)
            pydgm.control.energy_group_map = [2]
            pydgm.control.dgm_basis_name = 'test/4gbasis'.ljust(256)
        elif G == 7:
            pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
            pydgm.control.dgm_basis_name = 'test/7gbasis'.ljust(256)
            pydgm.control.energy_group_map = [4]

    def setSolver(self, solver):
        if solver == 'fixed':
            pydgm.control.solver_type = 'fixed'.ljust(256)
            pydgm.control.source_value = 1.0
            pydgm.control.allow_fission = False
        elif solver == 'eigen':
            pydgm.control.solver_type = 'eigen'.ljust(256)
            pydgm.control.source_value = 0.0
            pydgm.control.allow_fission = True

    def setMesh(self, mesh):
        if mesh.isdigit():
            N = int(mesh)
            pydgm.control.fine_mesh = [N]
            pydgm.control.coarse_mesh = [0.0, float(N)]
        elif mesh == 'coarse_pin':
            pydgm.control.fine_mesh = [3, 10, 3]
            pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
        elif mesh == 'fine_pin':
            pydgm.control.fine_mesh = [3, 22, 3]
            pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]

    def setBoundary(self, bounds):
        if bounds == 'reflect':
            pydgm.control.boundary_type = [1.0, 1.0]
        elif bounds == 'vacuum':
            pydgm.control.boundary_type = [0.0, 0.0]

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

        pydgm.dgmsolver.initialize_dgmsolver()

        ########################################################################
        order = 0

        phi = np.array([[[1.0270690018072897, 1.1299037448361107, 1.031220528952085,
                          1.0309270835964415, 1.0404782471236467, 1.6703756546880606, 0.220435842109856]]])
        psi = np.array([[[0.28670426208182, 0.3356992956691126, 0.3449054812807308, 0.3534008341488156, 0.3580544322663831, 0.6250475242024148, 0.0981878157679874],
                         [0.6345259657784981, 0.6872354146444389, 0.6066643580008859, 0.6019079440169605,
                             0.6067485919732419, 0.9472768646717264, 0.1166347906435061],
                         [0.6345259657784981, 0.6872354146444389, 0.6066643580008859, 0.6019079440169605,
                             0.6067485919732419, 0.9472768646717264, 0.1166347906435061],
                         [0.28670426208182, 0.3356992956691126, 0.3449054812807308, 0.3534008341488156, 0.3580544322663831, 0.6250475242024148, 0.0981878157679874]]])
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
        pydgm.state.phi[0, 0, :] = phi
        pydgm.state.psi = psi
        pydgm.state.d_keff = 1.0
        old_psi = pydgm.state.psi

        # Get the moments from the fluxes
        pydgm.dgmsolver.compute_flux_moments()

        pydgm.state.d_phi = pydgm.dgm.phi_m_zero
        pydgm.state.d_psi = pydgm.dgm.psi_m_zero

        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.d_phi.flatten(), phi_m, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(pydgm.state.d_psi[0, a, :].flatten(), psi_m[a], 12)

        ########################################################################
        pydgm.control.solver_type = 'fixed'.ljust(256)
        order = 1

        pydgm.state.d_phi = pydgm.dgm.phi_m_zero
        pydgm.state.d_psi = pydgm.dgm.psi_m_zero

        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.d_phi.flatten(), phi_m, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(pydgm.state.d_psi[0, a, :].flatten(), psi_m[a], 12)

        ########################################################################
        order = 2

        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.d_phi.flatten(), phi_m, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(pydgm.state.d_psi[0, a, :].flatten(), psi_m[a], 12)

        ########################################################################
        order = 3

        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.d_phi.flatten(), phi_m, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(pydgm.state.d_psi[0, a, :].flatten(), psi_m[a], 12)

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

        pydgm.dgmsolver.initialize_dgmsolver()

        ########################################################################
        order = 0
        # Set the converged fluxes
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi = np.linalg.solve((T - S), np.ones(7))
        pydgm.state.phi[0, 0, :] = phi
        for a in range(4):
            pydgm.state.psi[0, a, :] = phi / 2.0
        pydgm.state.d_keff = 1.0
        old_psi = pydgm.state.psi

        # Get the moments from the fluxes
        pydgm.dgmsolver.compute_flux_moments()

        pydgm.state.d_phi = pydgm.dgm.phi_m_zero
        pydgm.state.d_psi = pydgm.dgm.psi_m_zero

        phi_m_test = np.array([46.0567816728045685, 39.9620014433207302])

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[0, a, :].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        pydgm.control.solver_type = 'fixed'.ljust(256)
        order = 1

        pydgm.dgm.phi_m_zero = pydgm.state.d_phi
        pydgm.dgm.psi_m_zero = pydgm.state.d_psi

        phi_m_test = np.array([-7.7591835637013871, 18.2829496616545661])

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[0, a, :].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 2
        phi_m_test = np.array([-10.382535949686881, -23.8247979105656675])

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[0, a, :].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 3
        phi_m_test = np.array([-7.4878268473063185, 0.0])

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[0, a, :].flatten(), 0.5 * phi_m_test, 12)

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

        pydgm.dgmsolver.initialize_dgmsolver()

        ########################################################################
        order = 0
        # Set the converged fluxes
        phi = np.array([0.198933535568562, 2.7231683533646702, 1.3986600409998782,
                        1.010361903429942, 0.8149441787223116, 0.8510697418684054, 0.00286224604623])
        pydgm.state.phi[0, 0, :] = phi
        for a in range(4):
            pydgm.state.psi[0, a, :] = phi / 2.0
        pydgm.state.d_keff = 1.0674868709852505
        old_psi = pydgm.state.psi

        # Get the moments from the fluxes
        pydgm.dgmsolver.compute_flux_moments()

        pydgm.state.d_phi = pydgm.dgm.phi_m_zero
        pydgm.state.d_psi = pydgm.dgm.psi_m_zero

        phi_m_test = np.array([2.6655619166815265, 0.9635261040519922])
        norm_frac = 2 / sum(phi_m_test)
        phi_m_test *= norm_frac

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        assert_almost_equal(pydgm.state.d_keff, 1.0674868709852505, 12)

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[0, a, :].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 1

        pydgm.dgm.phi_m_zero = pydgm.state.d_phi
        pydgm.dgm.psi_m_zero = pydgm.state.d_psi

        phi_m_test = np.array([-0.2481536345018054, 0.5742286414743346])
        phi_m_test *= norm_frac

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve(True)

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[0, a, :].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 2
        phi_m_test = np.array([-1.4562664776830221, -0.3610274595244746])
        phi_m_test *= norm_frac

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve(True)

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[0, a, :].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 3
        phi_m_test = np.array([-1.0699480859043353, 0.0])
        phi_m_test *= norm_frac

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve(True)

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[0, a, :].flatten(), 0.5 * phi_m_test, 12)

    def test_sanity(self):
        def set_problem():
            self.setSolver('eigen')
            self.setGroups(7)
            pydgm.control.fine_mesh = [2, 1, 2]
            pydgm.control.coarse_mesh = [0.0, 5.0, 6.0, 11.0]
            pydgm.control.material_map = [1, 5, 3]
            self.setBoundary('vacuum')
            pydgm.control.angle_order = 4
            pydgm.control.angle_option = pydgm.angle.gl
            pydgm.control.xs_name = 'pythonTools/makeXS/7g/7gXS.anlxs'.ljust(256)
            pydgm.control.angle_option = pydgm.angle.gl
            pydgm.control.recon_print = False
            pydgm.control.eigen_print = False
            pydgm.control.outer_print = False
            pydgm.control.inner_print = False
            pydgm.control.recon_tolerance = 1e-14
            pydgm.control.eigen_tolerance = 1e-14
            pydgm.control.outer_tolerance = 1e-15
            pydgm.control.inner_tolerance = 1e-15
            pydgm.control.lamb = 1.0
            pydgm.control.use_dgm = True
            pydgm.control.store_psi = True
            pydgm.control.equation_type = 'DD'
            pydgm.control.legendre_order = 0
            pydgm.control.ignore_warnings = True
            pydgm.control.max_recon_iters = 1000
            pydgm.control.max_eigen_iters = 1000
            pydgm.control.max_outer_iters = 100
            pydgm.control.max_inner_iters = 100

        # Get the reference solution
        set_problem()
        pydgm.control.use_dgm = False

        # Initialize the dependancies
        pydgm.solver.initialize_solver()
        pydgm.solver.solve()

        # Save reference values
        ref_keff = pydgm.state.d_keff * 1
        ref_phi = pydgm.state.d_phi * 1
        ref_psi = pydgm.state.d_psi * 1

        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()

#         set_problem()
#         pydgm.control.eigen_print = 1
#         pydgm.control.max_recon_iters = 1
#         pydgm.control.max_eigen_iters = 1
#         pydgm.control.max_outer_iters = 1
#         pydgm.control.max_inner_iters = 1
#         pydgm.control.use_dgm = False
#
#         # Initialize the dependancies
#         pydgm.solver.initialize_solver()
#         pydgm.state.d_keff = ref_keff
#         pydgm.state.d_phi = ref_phi[:]
#         pydgm.state.d_psi = ref_psi[:]
#         pydgm.solver.solve()
#         print pydgm.state.d_keff
#         print pydgm.state.d_phi / ref_phi
#         exit()

        # Solve the real problem
        order = 0
        set_problem()
        pydgm.control.recon_print = 0
        pydgm.control.eigen_print = 2
        pydgm.control.outer_print = 0
        pydgm.control.inner_print = 0

        pydgm.control.max_recon_iters = 1
        pydgm.control.max_eigen_iters = 1
        pydgm.control.max_outer_iters = 1
        pydgm.control.max_inner_iters = 1
        pydgm.control.dgm_basis_name = 'test/7gdelta'.ljust(256)
        pydgm.control.energy_group_map = range(1, 7)

        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the correct flux as initial
        pydgm.state.phi = ref_phi[:]
        pydgm.state.psi = ref_psi[:]
        pydgm.state.d_keff = 2.0

        # Get the moments from the fluxes
        pydgm.dgmsolver.compute_flux_moments()

        # Set the correct moments as the initial values
        pydgm.state.d_phi = pydgm.dgm.phi_m_zero
        pydgm.state.d_psi = pydgm.dgm.psi_m_zero

        phi_m_ref = pydgm.dgm.phi_m_zero * 1

        pydgm.dgmsolver.compute_incoming_flux(order, ref_psi)
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

        #pydgm.dgm.delta_m[:, :, :, 0] *= 0

        pydgm.solver.solve()

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        print phi_m / ref_phi

        np.testing.assert_array_almost_equal(phi_m, phi_m_ref, 12)

    def test_dgmsolver_solve_orders_eigen(self):
        '''
        Test order 0 returns the same value when given the converged input for eigen problem
        '''
        # Set the variables for the test
        self.setSolver('eigen')
        self.setGroups(7)
        pydgm.control.fine_mesh = [2, 1, 2]
        pydgm.control.coarse_mesh = [0.0, 5.0, 6.0, 11.0]
        pydgm.control.material_map = [3, 5, 1]
        self.setBoundary('vacuum')
        pydgm.control.angle_order = 4
        pydgm.control.xs_name = 'pythonTools/makeXS/7g/7gXS.anlxs'.ljust(256)
        pydgm.control.dgm_basis_name = 'test/7gdelta'.ljust(256)
        pydgm.control.energy_group_map = range(1, 7)

        pydgm.control.recon_print = 0
        pydgm.control.eigen_print = 2
        pydgm.control.outer_print = 0
        pydgm.control.inner_print = 0

        pydgm.control.max_recon_iters = 1
        pydgm.control.max_eigen_iters = 1
        pydgm.control.max_outer_iters = 1
        pydgm.control.max_inner_iters = 1

        pydgm.dgmsolver.initialize_dgmsolver()

        np.set_printoptions(precision=4)
        ########################################################################
        order = 0
        # Set the converged fluxes
        phi = np.array([[[2.0257992104213676, 1.923718272601414, 0.8984917270995034,
                          0.1359184813858209, 0.069159084521991, 0.0395429438436026,
                          0.0228838012778573],
                         [3.078740985037656, 3.297020407502839, 1.4972548403087529,
                          0.2419744554249779, 0.1399415224162652, 0.0804404829573202,
                          0.0429420231808826],
                         [3.242780845662988, 3.6253236761323233, 1.5191384960457086,
                          0.2657137576543585, 0.1710313947880435, 0.0951334377954927,
                          0.0425020905708747],
                         [2.680157055578504, 3.06774476778288, 1.4498239857686677,
                          0.2681250437495658, 0.2232835199335891, 0.1450135178081015,
                          0.0801799450539108],
                         [1.6242342173603628, 1.6758532156636186, 0.8405795956015388,
                          0.1625775747329379, 0.1563516098298167, 0.1085336308306435,
                          0.0620903836758187]]])

        psi = np.array([[[0.2839554762443688, 0.4279896897990578, 0.2420050678428892,
                          0.0383706034304901, 0.0227129595199267, 0.0134676747135594,
                          0.0085151463676417],
                         [0.3261965462563942, 0.4754151246434864, 0.2637847730312081,
                          0.0416367506767502, 0.0242295879053059, 0.0142777111873079,
                          0.0088952657992601],
                         [0.4329105987588649, 0.5823444079707101, 0.310017871277954,
                          0.0484740605658099, 0.0272440970357683, 0.0158582531655902,
                          0.0096058269236478],
                         [0.7372082396801698, 0.8130863049868736, 0.3980411128911101,
                          0.0611423034295212, 0.0323171553205731, 0.0184328283005457,
                          0.0106824936928838],
                         [1.4598645444698684, 1.1444682238971762, 0.5306971842723688,
                          0.0785437229491508, 0.039713913866885, 0.0228212778870398,
                          0.0128190133645812],
                         [1.603800667047302, 1.339896091825748, 0.5814786019156781,
                          0.0859079287167495, 0.0407949773027919, 0.0233252278905371,
                          0.0133720031593783],
                         [1.5309495376107136, 1.4197218176391444, 0.610232345463981,
                          0.0916160580458328, 0.0427338663841867, 0.0237604956146664,
                          0.0133235079710548],
                         [1.464885513822089, 1.441867006893067, 0.6235631954981594,
                          0.0946670238578062, 0.0444506293512767, 0.0243640916897313,
                          0.0133250773869967]],
                        [[0.8979420792671616, 1.2278028414312574, 0.6395952779081433,
                          0.1019260121956693, 0.0603514150340045, 0.0356908055351527,
                          0.0206234908394634],
                         [1.0082273816890912, 1.3201786458724556, 0.672684591589285,
                            0.1067770556462347, 0.0623342020910218, 0.0366918535944347,
                            0.0209491303191844],
                         [1.2599720527733733, 1.496483799426031, 0.7295310940574019,
                            0.1149891560923703, 0.0655143483522853, 0.038270820349191,
                            0.0214214755581922],
                         [1.766407427859224, 1.7259976394994847, 0.7874221794566651,
                            0.1232544708010277, 0.0685805464349633, 0.0397831546607479,
                            0.0217985791844457],
                         [1.982352186305374, 1.8243019624882768, 0.7671582509538747,
                            0.1232633853173765, 0.0683865153826585, 0.0389703997168544,
                            0.0208888756568809],
                         [1.7148976537965404, 1.8047767702734698, 0.771868196578308,
                            0.1284953242691219, 0.0749924134827409, 0.0420388114788651,
                            0.0212660172635992],
                         [1.4914493472137902, 1.7347522148427135, 0.768949255281506,
                            0.1302419543122088, 0.0802672835477712, 0.0454900456157049,
                            0.0223847678848062],
                         [1.374239059383414, 1.6804507107489859, 0.7616179736173397,
                            0.1299769203242264, 0.0826519306543626, 0.0473735675866383,
                            0.0231585138707784]],
                        [[1.3289690484789753, 1.671586440705667, 0.7760283202736976,
                          0.1259707834617702, 0.0729270280735501, 0.041730732082634,
                          0.0210874065276817],
                         [1.467070365430227, 1.7596888439540137, 0.791872036643769,
                            0.128551812630557, 0.0733074589055889, 0.041586719950917,
                            0.0205120079897094],
                         [1.7517468797848381, 1.8958105952103994, 0.7980439603657452,
                            0.1300892526694405, 0.0723937875679987, 0.0403303303934091,
                            0.0189198483739351],
                         [2.1207386621923865, 1.963765044607128, 0.7212071824348786,
                            0.1211392275442601, 0.0658334448866402, 0.034958078599678,
                            0.0145491386909295],
                         [1.7904939585505952, 1.89352053932841, 0.7181697463659803,
                            0.1317868631490471, 0.082793577541892, 0.0420585740741061,
                            0.0164782671802563],
                         [1.4167593249106136, 1.7678192897287506, 0.7810108616578453,
                            0.1455407088176891, 0.107675105342892, 0.0605018605823795,
                            0.026386507283148],
                         [1.177236541212018, 1.6234399964838175, 0.7691016739211285,
                            0.1437011859524629, 0.1139120708369217, 0.0675556535748025,
                            0.0312946245257324],
                         [1.0637855120993567, 1.5360072350588092, 0.7512967906309744,
                            0.1404630994676566, 0.1148056366649366, 0.0698763787160454,
                            0.0333956120236536]],
                        [[1.436495042747361, 1.6792136928011263, 0.7579125860583994,
                          0.1336344485909219, 0.098604182325654, 0.0626567122928802,
                          0.0350590921713212],
                         [1.5387508951279658, 1.7186389175341765, 0.7620051265765974,
                            0.1353175537373121, 0.1007482443032926, 0.0640130246749766,
                            0.03564770613761],
                         [1.7051696722305356, 1.7518121969051692, 0.7582694132424417,
                            0.1369370337610383, 0.1046797020301904, 0.0665917118269667,
                            0.0368374928147647],
                         [1.7294779410036458, 1.6974160643540825, 0.744258551231329,
                            0.138354439047422, 0.1120623770468577, 0.0714367487802417,
                            0.0391794498708677],
                         [1.341821865026486, 1.547874540441831, 0.7556888528998007,
                            0.1422593096610216, 0.1224628337686979, 0.0787659469902519,
                            0.0426536374845071],
                         [0.9634060077282144, 1.3324269934504411, 0.6954805007449583,
                            0.1314528956036147, 0.118327409011439, 0.0784415022220629,
                            0.0433112324654103],
                         [0.7725107502337766, 1.1724710518681165, 0.6392311814456787,
                            0.1209778691436543, 0.112132068182718, 0.0761388322283502,
                            0.0430146562230252],
                         [0.6885553482025525, 1.0893350399750683, 0.6069001661447118,
                            0.1149124812343552, 0.1081129453446089, 0.0743840699002204,
                            0.0426285473144128]],
                        [[1.3169150639303029, 1.313143196294532, 0.5934134444747584,
                          0.112983521080786, 0.1006890422223745, 0.0671233256092146,
                          0.0370834493694714],
                         [1.3447435051520091, 1.2756948921108744, 0.578032181522264,
                            0.1109128207385085, 0.1003555995219249, 0.0669432975064532,
                            0.0368743872958025],
                         [1.3255267239562842, 1.1718393856235514, 0.5467264435687116,
                            0.106076277173514, 0.0986556746148474, 0.0659669292501365,
                            0.0361998578074773],
                         [1.0537788563447281, 0.9734181137831552, 0.4957923512162933,
                            0.0961929513181893, 0.0917160223956373, 0.0619693328895713,
                            0.0340926844459002],
                         [0.5750074909228001, 0.6987878823386924, 0.3690096305377115,
                            0.0716465574529972, 0.0710147551998868, 0.0504720138090471,
                            0.0292896707346571],
                         [0.3382749040753765, 0.5005376956437223, 0.286799618688869,
                            0.0556029704482913, 0.0575482625391845, 0.0425783497502043,
                            0.0260285263773904],
                         [0.2550513848467887, 0.4086506508949976, 0.2437587461588054,
                            0.0472221628586482, 0.0500292057938681, 0.0378826318416084,
                            0.0239178952309458],
                         [0.2220793504938031, 0.3678937484267372, 0.2235158737950532,
                            0.0432849735160482, 0.0463727895022187, 0.0355187175888716,
                            0.0228020647902731]]])
        psi /= (np.linalg.norm(psi) * 1)
        phi_new = phi * 0
        for a in range(pydgm.control.number_angles):
            phi_new[0] += psi[:, a, :] * pydgm.angle.wt[a]
        print phi / phi_new
        exit()

        pydgm.state.phi = phi
        pydgm.state.psi = psi
        pydgm.state.d_keff = 0.33973731848126831
        old_psi = pydgm.state.psi

        # Get the moments from the fluxes
        pydgm.dgmsolver.compute_flux_moments()

#         print pydgm.dgm.phi_m_zero
#         print pydgm.dgm.psi_m_zero

        #pydgm.state.normalize_flux(pydgm.dgm.phi_m_zero, pydgm.dgm.psi_m_zero)

        pydgm.state.d_phi = pydgm.dgm.phi_m_zero
        pydgm.state.d_psi = pydgm.dgm.psi_m_zero
        print phi

        # print pydgm.state.d_phi

        phi_m_test = np.array([1.2215130211493985, 2.1192460274357612, 2.4562162616694869, 2.3035099572301396, 1.4147307747490998,
                               0.10717335199217316, 0.14699805722553888, 0.10117227811925951, 8.6310161063384294E-002, 4.3130109365754711E-002])
        phi_m_test = pydgm.dgm.phi_m_zero * 1
        norm_frac = 1 / sum(phi_m_test)
        phi_m_test *= norm_frac

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

#         pydgm.state.output_moments()
#         print pydgm.dgm.delta_m[:, :, :, 0]

#         print pydgm.dgm.delta_m[:, :, :, 0]
        pydgm.dgm.delta_m[:, :, :, 0] *= 0
        # exit()

        pydgm.solver.solve()

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        print phi_m
        print phi_m / phi

        np.testing.assert_array_almost_equal(phi_m, phi_m_test, 12)

        ########################################################################
        order = 1

        pydgm.dgm.phi_m_zero = pydgm.state.d_phi
        pydgm.dgm.psi_m_zero = pydgm.state.d_psi

        phi_m_test = np.array([0.66268605409797898, 1.1239769588944581, 1.4011457517310117, 1.3088156391543195, 0.84988298005049157,
                               3.7839914869954847E-002, 5.7447025802385317E-002, 5.1596378790218486E-002, 3.8939158159247110E-002, 1.8576596655769165E-002])
        phi_m_test *= norm_frac

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve(True)

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        np.testing.assert_array_almost_equal(phi_m.T.flatten(), phi_m_test, 12)

        ########################################################################
        order = 2
        phi_m_test = np.array([-0.20710920655711104, -0.44545552454860282, -0.46438347612912256, -0.41828263508757896, -0.18748642683048020,
                               3.1862102568187112E-004, 3.1141556263365915E-003, 5.3924924332473369E-003, 5.0995287080187754E-003, 3.0030380436572414E-003])
        phi_m_test *= norm_frac

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve(True)

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        np.testing.assert_array_almost_equal(phi_m.T.flatten(), phi_m_test, 12)

        ########################################################################
        order = 3
        phi_m_test = np.array([-0.13255187402833862, -0.30996650357216082, -0.42418668341792881, -0.32530149073950271, -
                               0.15053175043041164, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000])
        phi_m_test *= norm_frac

        pydgm.dgmsolver.compute_incoming_flux(order, old_psi)
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve(True)

        phi_m = pydgm.state.d_phi
        psi_m = pydgm.state.d_psi

        np.testing.assert_array_almost_equal(phi_m.T.flatten(), phi_m_test, 12)

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

        # Assume infinite homogeneous media (isotropic flux)
        psi_moments = 0.5 * phi_m
        psi = np.reshape(np.zeros(8), (1, 4, 2), 'F')
        phi_new = np.reshape(np.zeros(7), (1, 1, 7), 'F')
        psi_new = np.reshape(np.zeros(28), (1, 4, 7), 'F')

        for order in range(4):
            for a in range(4):
                psi[0, a, :] = psi_moments[:, order]

            pydgm.dgmsolver.unfold_flux_moments(order, psi, phi_new, psi_new)

        np.testing.assert_array_almost_equal(phi_new.flatten(), phi_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_new[0, a, :].flatten(), phi_test * 0.5)

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

        phi_test = [1.6274528794638465, 2.71530879612549, 1.461745652768521, 1.3458703902580473, 1.3383852126342237, 1.9786760428590306, 0.24916735316863525, 1.6799175379390339, 2.8045999684695797, 1.516872017690622, 1.3885229934177148, 1.3782095743929001, 2.051131534663419, 0.26873064494111804, 1.728788120766425, 2.883502682394886, 1.5639999234445578, 1.4246328795261316, 1.4121166958899956, 2.1173467066121874, 0.2724292532553828, 1.7839749586964595, 2.990483236041222, 1.6474286521554664, 1.5039752034511047, 1.4924425499449177, 2.3127049909257686, 0.25496633574011124, 1.8436202405517381, 3.122355600505027, 1.7601872542791979, 1.61813693117119, 1.6099652659907275, 2.60256939853679, 0.24873883482629144, 1.896225857094417, 3.2380762891116794, 1.8534459525081792, 1.7117690484677541, 1.7061424886519436, 2.831599567019092, 0.26081315241625463, 1.9421441425092316, 3.338662519105913, 1.9310368092514267, 1.789369188781964, 1.7857603538028388, 3.0201767784594478, 0.2667363594339594, 1.9816803882995633, 3.424961908919033, 1.9955392685572624, 1.853808027881203, 1.851843446016314, 3.1773523146671065, 0.27189861962890616, 2.0150973757748596, 3.4976972455932094, 2.0486999251118014, 1.9069365316531377, 1.9063232414331912, 3.307833351001605, 0.2755922553419729, 2.042616213943685, 3.5574620224059217, 2.0917047787489365, 1.9499600832109016, 1.9504462746103806, 3.4141788518658602, 0.27833708525534473, 2.0644181962111365, 3.604732065595381, 2.1253588124042495, 1.9836690960190415, 1.985023407898914, 3.497921277464179, 0.28030660972118154, 2.080646338594525, 3.6398748475310785, 2.150203190885212, 2.0085809732608575, 2.0105818574623395, 3.5600286331289643, 0.2816665790912415, 2.0914067095511766, 3.663158139593214, 2.16659102830272, 2.0250269209204395, 2.0274573320958647, 3.6011228563902344, 0.2825198396790823, 2.0967694470675315, 3.6747566727970047, 2.174734975618102, 2.033203922754008, 2.0358487486465924, 3.621580567528384, 0.28293121918903963,
                    2.0967694470675315, 3.6747566727970042, 2.1747349756181023, 2.033203922754008, 2.0358487486465924, 3.6215805675283836, 0.2829312191890396, 2.0914067095511766, 3.6631581395932136, 2.1665910283027205, 2.02502692092044, 2.0274573320958647, 3.6011228563902358, 0.2825198396790823, 2.080646338594525, 3.639874847531079, 2.150203190885212, 2.008580973260857, 2.01058185746234, 3.5600286331289652, 0.2816665790912415, 2.0644181962111365, 3.6047320655953805, 2.125358812404249, 1.9836690960190408, 1.985023407898914, 3.4979212774641804, 0.2803066097211815, 2.042616213943685, 3.5574620224059217, 2.0917047787489365, 1.9499600832109014, 1.9504462746103808, 3.4141788518658616, 0.2783370852553448, 2.01509737577486, 3.49769724559321, 2.0486999251118005, 1.9069365316531375, 1.9063232414331914, 3.3078333510016056, 0.27559225534197296, 1.981680388299563, 3.424961908919033, 1.9955392685572624, 1.8538080278812032, 1.8518434460163142, 3.1773523146671074, 0.27189861962890616, 1.9421441425092318, 3.338662519105913, 1.931036809251427, 1.7893691887819645, 1.7857603538028393, 3.020176778459449, 0.2667363594339594, 1.896225857094417, 3.2380762891116777, 1.8534459525081792, 1.7117690484677544, 1.706142488651944, 2.831599567019092, 0.2608131524162547, 1.8436202405517386, 3.122355600505027, 1.7601872542791974, 1.6181369311711902, 1.6099652659907278, 2.6025693985367897, 0.24873883482629144, 1.783974958696459, 2.990483236041223, 1.6474286521554669, 1.5039752034511054, 1.4924425499449177, 2.312704990925769, 0.2549663357401113, 1.7287881207664255, 2.883502682394885, 1.5639999234445578, 1.4246328795261323, 1.412116695889996, 2.117346706612188, 0.27242925325538286, 1.6799175379390343, 2.8045999684695793, 1.516872017690622, 1.388522993417715, 1.3782095743929004, 2.05113153466342, 0.26873064494111826, 1.6274528794638465, 2.7153087961254894, 1.4617456527685213, 1.3458703902580476, 1.3383852126342235, 1.978676042859031, 0.24916735316863528]
        pydgm.state.phi[0, :, :] = np.reshape(phi_test, (28, 7), 'F')

        pydgm.dgmsolver.dgmsolve()

        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten(), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]

        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        pydgm.control.lamb = 0.45

        pydgm.dgmsolver.initialize_dgmsolver()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))
        phi_test = np.array([phi_test for i in range(28)]).flatten()

        pydgm.dgmsolver.dgmsolve()

        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :].flatten(), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        pydgm.control.lamb = 0.8
        pydgm.control.allow_fission = True
        phi_test = np.array([1.0781901438738859, 1.5439788126739036, 1.0686290157458673,
                             1.0348940034466163, 1.0409956199943164, 1.670442207080332, 0.2204360523334687])

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        pydgm.state.phi[0, 0, :] = phi_test
        for a in range(4):
            pydgm.state.psi[0, a, :] = 0.5 * phi_test

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]

        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        pydgm.control.lamb = 0.4

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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

        pydgm.control.lamb = 0.5
        pydgm.control.recon_print = 1
        #pydgm.control.max_eigen_iters = 1

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        phi_test = np.array([0.7263080826036219, 0.12171194697729938, 1.357489062141697, 0.2388759408761157, 1.8494817499319578, 0.32318764022244134, 2.199278050699694, 0.38550684315075284, 2.3812063412628075, 0.4169543421336097,
                             2.381206341262808, 0.41695434213360977, 2.1992780506996943, 0.38550684315075295, 1.8494817499319585, 0.3231876402224415, 1.3574890621416973, 0.23887594087611572, 0.7263080826036221, 0.12171194697729937])

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.d_keff, 0.8099523232983424, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        pydgm.control.lamb = 0.8
        pydgm.control.max_eigen_iters = 1

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        phi_test = np.array([1.6283945282803138, 1.2139688020213637, 0.03501217302426163, 6.910819115000905e-17, 1.9641731545343404, 1.59852298044156, 0.05024813427412045, 1.0389359842780806e-16, 2.2277908375593736, 1.8910978193073922, 0.061518351747482505, 1.3055885402420332e-16, 2.40920191588961, 2.088554929299159, 0.06902375359471126, 1.487795240822353e-16, 2.5016194254000244, 2.188087672560707, 0.0727855220655801, 1.5805185521208351e-16,
                             2.501619425400025, 2.1880876725607075, 0.07278552206558009, 1.5805185521208351e-16, 2.40920191588961, 2.088554929299159, 0.06902375359471127, 1.487795240822353e-16, 2.2277908375593736, 1.891097819307392, 0.0615183517474825, 1.3055885402420332e-16, 1.9641731545343404, 1.59852298044156, 0.05024813427412045, 1.0389359842780806e-16, 1.6283945282803138, 1.2139688020213637, 0.03501217302426163, 6.910819115000904e-17])

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.d_keff, 0.185134666261, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        pydgm.control.lamb = 0.55
        pydgm.control.max_eigen_iters = 1

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        phi_test = np.array([0.19050251326520584, 1.9799335510805185, 0.69201814518126, 0.3927000245492841, 0.2622715078950253, 0.20936059119838546, 0.000683954269595958, 0.25253653423327665, 2.8930819653774895, 1.158606945184528, 0.6858113244922716, 0.4639601075261923, 0.4060114930207368, 0.0013808859451732852, 0.30559047625122115, 3.6329637815416556, 1.498034484581793, 0.9026484213739354, 0.6162114941108023, 0.5517562407150877, 0.0018540270157502057, 0.3439534785160265, 4.153277746375052, 1.7302149163096785, 1.0513217539517374, 0.7215915434720093, 0.653666204542615, 0.0022067618449436725, 0.36402899896324237, 4.421934793951583, 1.8489909842118943, 1.127291245982061, 0.7756443978822711, 0.705581398687358, 0.0023773065003326204,
                             0.36402899896324237, 4.421934793951582, 1.8489909842118946, 1.1272912459820612, 0.7756443978822711, 0.705581398687358, 0.0023773065003326204, 0.34395347851602653, 4.153277746375052, 1.7302149163096785, 1.0513217539517377, 0.7215915434720092, 0.653666204542615, 0.002206761844943672, 0.3055904762512212, 3.6329637815416564, 1.498034484581793, 0.9026484213739353, 0.6162114941108023, 0.5517562407150877, 0.0018540270157502063, 0.2525365342332767, 2.8930819653774895, 1.1586069451845278, 0.6858113244922716, 0.4639601075261923, 0.4060114930207368, 0.0013808859451732852, 0.19050251326520584, 1.9799335510805192, 0.6920181451812601, 0.3927000245492842, 0.26227150789502535, 0.20936059119838543, 0.0006839542695959579])

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.d_keff, 0.30413628310914226, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        pydgm.control.material_map = [1]

        pydgm.control.lamb = 0.8
        pydgm.control.max_eigen_iters = 1

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

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
        assert_almost_equal(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_dgmsolver_eigenR4g(self):
        '''
        Test the 4g->2G, eigenvalue problem with infinite medium
        '''
        # Set the variables for the test
        self.setGroups(4)
        self.setSolver('eigen')
        self.setMesh('10')
        self.setBoundary('reflect')
        pydgm.control.material_map = [1]
        # pydgm.control.max_inner_iters = 10
        pydgm.control.lamb = 0.8
        pydgm.control.max_eigen_iters = 1

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

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
        assert_almost_equal(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        pydgm.control.lamb = 0.55
        pydgm.control.max_eigen_iters = 1

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

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
        assert_almost_equal(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        pydgm.control.lamb = 0.8
        pydgm.control.max_eigen_iters = 1

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # set the test flux
        keff_test = 0.8418546852484950
        phi_test = [0.13393183108467394, 0.04663240631432256, 0.13407552941360298, 0.04550808086281801, 0.13436333428621713, 0.043206841474147446, 0.1351651393398092, 0.0384434752119791, 0.13615737742196526, 0.03329929560434661, 0.13674284660888314, 0.030464508103354708, 0.13706978363298242, 0.028970199506203023, 0.13721638515632006, 0.028325674662651124,
                    0.13721638515632006, 0.028325674662651124, 0.1370697836329824, 0.028970199506203012, 0.13674284660888308, 0.0304645081033547, 0.13615737742196524, 0.03329929560434659, 0.13516513933980914, 0.03844347521197908, 0.13436333428621713, 0.043206841474147425, 0.13407552941360296, 0.045508080862818004, 0.1339318310846739, 0.046632406314322555]

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        pydgm.control.lamb = 0.8
        pydgm.control.max_eigen_iters = 1

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # set the test flux
        keff_test = 0.759180925837
        phi_test = [2.22727714687889, 1.7369075872008062, 0.03381777446256108, 1.7036045485771946e-51, 2.227312836950116, 1.737071399085861, 0.033828681689533874, 1.698195884848375e-51, 2.2273842137861877, 1.737399076914814, 0.03385050700298498, 1.6965081411970257e-51, 2.2275366771427696, 1.738049905609514, 0.033892498696060584, 1.697491400636173e-51, 2.2277285082016354, 1.7388423494826295, 0.03394281751690741, 1.6984869893367e-51, 2.2278725176984917, 1.7394359801217965, 0.03398044173136341, 1.697888141875072e-51, 2.2279685888050587, 1.739831399863755, 0.03400546998960266, 1.6957049057554318e-51, 2.2280166437743327, 1.7400290096083806, 0.034017967785934036, 1.691942073409801e-51,
                    2.2280166437743327, 1.7400290096083808, 0.03401796778593402, 1.68659920584123e-51, 2.2279685888050587, 1.7398313998637547, 0.03400546998960263, 1.6796706200591657e-51, 2.2278725176984917, 1.7394359801217967, 0.03398044173136335, 1.6711453402288656e-51, 2.227728508201635, 1.73884234948263, 0.033942817516907337, 1.6610070122205585e-51, 2.2275366771427696, 1.7380499056095144, 0.0338924986960605, 1.6492337810058256e-51, 2.227384213786188, 1.7373990769148142, 0.03385050700298487, 1.6385765949262272e-51, 2.2273128369501163, 1.7370713990858613, 0.03382868168953376, 1.631610014066153e-51, 2.2272771468788894, 1.7369075872008064, 0.03381777446256096, 1.6281341640813905e-51]

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        pydgm.control.lamb = 0.55
        pydgm.control.max_eigen_iters = 1

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # set the test flux
        keff_test = 1.0794314789325041
        phi_test = [0.18617101855192203, 2.858915338372074, 1.4772041911943246, 1.0299947729491368, 0.7782291112252604, 0.6601323950057741, 0.0018780861364841711, 0.18615912584783736, 2.8586649211715436, 1.4771766639520822, 1.030040359498237, 0.7782969794725844, 0.6603312122972236, 0.0018857945539742516, 0.18613533094381535, 2.858163988201594, 1.4771216026067822, 1.0301315410919691, 0.7784327301908717, 0.6607289506819793, 0.001901260571677254, 0.1860688679764812, 2.856246121648244, 1.4768054633757053, 1.0308126366153867, 0.7795349485104974, 0.6645605858759833, 0.0020390343478796447, 0.18597801819761287, 2.8534095558345967, 1.4763058823653368, 1.0319062100766097, 0.7813236329806805, 0.6707834262470258, 0.002202198406120643, 0.18591115233580913, 2.851306298580461, 1.4759330573010532, 1.0327026586727457, 0.7826354049117061, 0.6752310599415209, 0.002251216654318464, 0.18586715682184884, 2.8499152893733255, 1.4756853599772022, 1.0332225362074143, 0.7834960048959739, 0.6780933062272468, 0.0022716374303459433, 0.1858453299832966, 2.8492230801887986, 1.475561761882258, 1.0334791905008067, 0.7839221795374745, 0.6794942839316992, 0.002280077669362046,
                    0.18584532998329656, 2.8492230801887986, 1.475561761882258, 1.0334791905008065, 0.7839221795374745, 0.6794942839316991, 0.0022800776693620455, 0.18586715682184884, 2.8499152893733255, 1.4756853599772024, 1.0332225362074146, 0.7834960048959738, 0.6780933062272467, 0.002271637430345943, 0.18591115233580915, 2.851306298580461, 1.4759330573010532, 1.0327026586727457, 0.7826354049117062, 0.6752310599415207, 0.0022512166543184635, 0.1859780181976129, 2.853409555834596, 1.476305882365337, 1.0319062100766097, 0.7813236329806805, 0.6707834262470258, 0.002202198406120643, 0.1860688679764812, 2.856246121648244, 1.4768054633757055, 1.0308126366153867, 0.7795349485104973, 0.6645605858759831, 0.002039034347879644, 0.18613533094381537, 2.858163988201594, 1.4771216026067824, 1.0301315410919691, 0.7784327301908716, 0.6607289506819792, 0.0019012605716772534, 0.1861591258478374, 2.858664921171543, 1.4771766639520822, 1.0300403594982372, 0.7782969794725842, 0.6603312122972235, 0.0018857945539742511, 0.18617101855192209, 2.8589153383720736, 1.477204191194325, 1.0299947729491368, 0.7782291112252603, 0.660132395005774, 0.0018780861364841707]

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_fine_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()
