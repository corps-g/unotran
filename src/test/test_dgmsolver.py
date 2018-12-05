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
        pydgm.control.recon_tolerance = 1e-14
        pydgm.control.eigen_tolerance = 1e-14
        pydgm.control.outer_tolerance = 1e-15
        pydgm.control.lamb = 1.0
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = True
        pydgm.control.equation_type = 'DD'
        pydgm.control.scatter_legendre_order = 0
        pydgm.control.ignore_warnings = True
        pydgm.control.max_recon_iters = 10000
        pydgm.control.max_eigen_iters = 1000
        pydgm.control.max_outer_iters = 100

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
        nA = 2

        pydgm.dgmsolver.initialize_dgmsolver()

        ########################################################################
        order = 0
        pydgm.dgm.dgm_order = order
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

        pydgm.state.mg_phi = pydgm.dgm.phi_m[0]
        pydgm.state.mg_psi = pydgm.dgm.psi_m[0]

        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.mg_phi.flatten(), phi_m, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(pydgm.state.mg_psi[:, a, 0].flatten(), psi_m[a], 12)

        ########################################################################
        order = 1
        pydgm.dgm.dgm_order = order
        pydgm.state.mg_phi = pydgm.dgm.phi_m[0]
        pydgm.state.mg_psi = pydgm.dgm.psi_m[0]

        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.mg_phi.flatten(), phi_m, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(pydgm.state.mg_psi[:, a, 0].flatten(), psi_m[a], 12)

        ########################################################################
        order = 2
        pydgm.dgm.dgm_order = order
        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.mg_phi.flatten(), phi_m, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(pydgm.state.mg_psi[:, a, 0].flatten(), psi_m[a], 12)

        ########################################################################
        order = 3
        pydgm.dgm.dgm_order = order
        phi_m = phi_m_test[:, order]
        psi_m = psi_m_test[:, :, order]

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.mg_phi.flatten(), phi_m, 12)
        for a in range(4):
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

        pydgm.dgmsolver.initialize_dgmsolver()

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

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 1
        pydgm.dgm.dgm_order = order
        pydgm.dgm.phi_m[0] = pydgm.state.mg_phi
        pydgm.dgm.psi_m[0] = pydgm.state.mg_psi

        phi_m_test = np.array([-7.7591835637013871, 18.2829496616545661])

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 2
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-10.382535949686881, -23.8247979105656675])

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 3
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-7.4878268473063185, 0.0])

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
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

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.compute_xs_moments()
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        assert_almost_equal(pydgm.state.keff, 1.0674868709852505, 12)

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 1
        pydgm.dgm.dgm_order = order
        pydgm.dgm.phi_m[0] = pydgm.state.mg_phi
        pydgm.dgm.psi_m[0] = pydgm.state.mg_psi

        phi_m_test = np.array([-0.2481536345018054, 0.5742286414743346])
        phi_m_test *= norm_frac

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 2
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-1.4562664776830221, -0.3610274595244746])
        phi_m_test *= norm_frac

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

        ########################################################################
        order = 3
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-1.0699480859043353, 0.0])
        phi_m_test *= norm_frac

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, nA:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.flatten(), phi_m_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_m[:, a, 0].flatten(), 0.5 * phi_m_test, 12)

    def test_dgmsolver_solve_orders_eigen(self):
        '''
        Test order 0 returns the same value when given the converged input for eigen problem
        '''
        # Set the variables for the test
        self.setSolver('eigen')
        self.setGroups(7)
        pydgm.control.fine_mesh = [2, 1, 2]
        pydgm.control.coarse_mesh = [0.0, 5.0, 6.0, 11.0]
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
        for a in range(pydgm.control.number_angles):
            phi_new[0] += psi[:, a, :] * pydgm.angle.wt[a]
            phi_new[0] += psi[:, 2 * pydgm.control.number_angles - a - 1, :] * pydgm.angle.wt[a]

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

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, pydgm.control.number_angles:, 0]
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

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, pydgm.control.number_angles:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.T.flatten('F'), phi_m_test, 12)

        ########################################################################
        order = 2
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-0.20710920655711104, -0.44545552454860282, -0.46438347612912256, -0.41828263508757896, -0.18748642683048020, 3.1862102568187112E-004, 3.1141556263365915E-003, 5.3924924332473369E-003, 5.0995287080187754E-003, 3.0030380436572414E-003])

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, pydgm.control.number_angles:, 0]
        pydgm.dgmsolver.slice_xs_moments(order)

        pydgm.solver.solve()

        phi_m = pydgm.state.mg_phi
        psi_m = pydgm.state.mg_psi

        np.testing.assert_array_almost_equal(phi_m.T.flatten('F'), phi_m_test, 12)

        ########################################################################
        order = 3
        pydgm.dgm.dgm_order = order
        phi_m_test = np.array([-0.13255187402833862, -0.30996650357216082, -0.42418668341792881, -0.32530149073950271, -0.15053175043041164, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000])

        pydgm.state.mg_incident = pydgm.dgm.psi_m[order, :, pydgm.control.number_angles:, 0]
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

        # Assume infinite homogeneous media (isotropic flux)
        psi_moments = 0.5 * phi_m
        psi = np.reshape(np.zeros(8), (2, 4, 1), 'F')
        phi_new = np.reshape(np.zeros(7), (1, 7, 1), 'F')
        psi_new = np.reshape(np.zeros(28), (7, 4, 1), 'F')

        for order in range(4):
            for a in range(4):
                psi[:, a, 0] = psi_moments[:, order]

            pydgm.dgmsolver.unfold_flux_moments(order, psi, phi_new, psi_new)

        np.testing.assert_array_almost_equal(phi_new.flatten(), phi_test, 12)
        for a in range(4):
            np.testing.assert_array_almost_equal(psi_new[:, a, 0].flatten(), phi_test * 0.5)

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
        pydgm.control.lamb = 0.55
        pydgm.control.source_value = 1.0

        pydgm.dgmsolver.initialize_dgmsolver()

        phi_test = [1.6274528794638465, 2.71530879612549, 1.461745652768521, 1.3458703902580473, 1.3383852126342237, 1.9786760428590306, 0.24916735316863525, 1.6799175379390339, 2.8045999684695797, 1.516872017690622, 1.3885229934177148, 1.3782095743929001, 2.051131534663419, 0.26873064494111804, 1.728788120766425, 2.883502682394886, 1.5639999234445578, 1.4246328795261316, 1.4121166958899956, 2.1173467066121874, 0.2724292532553828, 1.7839749586964595, 2.990483236041222, 1.6474286521554664, 1.5039752034511047, 1.4924425499449177, 2.3127049909257686, 0.25496633574011124, 1.8436202405517381, 3.122355600505027, 1.7601872542791979, 1.61813693117119, 1.6099652659907275, 2.60256939853679, 0.24873883482629144, 1.896225857094417, 3.2380762891116794, 1.8534459525081792, 1.7117690484677541, 1.7061424886519436, 2.831599567019092, 0.26081315241625463, 1.9421441425092316, 3.338662519105913, 1.9310368092514267, 1.789369188781964, 1.7857603538028388, 3.0201767784594478, 0.2667363594339594, 1.9816803882995633, 3.424961908919033, 1.9955392685572624, 1.853808027881203, 1.851843446016314, 3.1773523146671065, 0.27189861962890616, 2.0150973757748596, 3.4976972455932094, 2.0486999251118014, 1.9069365316531377, 1.9063232414331912, 3.307833351001605, 0.2755922553419729, 2.042616213943685, 3.5574620224059217, 2.0917047787489365, 1.9499600832109016, 1.9504462746103806, 3.4141788518658602, 0.27833708525534473, 2.0644181962111365, 3.604732065595381, 2.1253588124042495, 1.9836690960190415, 1.985023407898914, 3.497921277464179, 0.28030660972118154, 2.080646338594525, 3.6398748475310785, 2.150203190885212, 2.0085809732608575, 2.0105818574623395, 3.5600286331289643, 0.2816665790912415, 2.0914067095511766, 3.663158139593214, 2.16659102830272, 2.0250269209204395, 2.0274573320958647, 3.6011228563902344, 0.2825198396790823, 2.0967694470675315, 3.6747566727970047, 2.174734975618102, 2.033203922754008, 2.0358487486465924, 3.621580567528384, 0.28293121918903963, 2.0967694470675315, 3.6747566727970042, 2.1747349756181023, 2.033203922754008, 2.0358487486465924, 3.6215805675283836, 0.2829312191890396, 2.0914067095511766, 3.6631581395932136, 2.1665910283027205, 2.02502692092044, 2.0274573320958647, 3.6011228563902358, 0.2825198396790823, 2.080646338594525, 3.639874847531079, 2.150203190885212, 2.008580973260857, 2.01058185746234, 3.5600286331289652, 0.2816665790912415, 2.0644181962111365, 3.6047320655953805, 2.125358812404249, 1.9836690960190408, 1.985023407898914, 3.4979212774641804, 0.2803066097211815, 2.042616213943685, 3.5574620224059217, 2.0917047787489365, 1.9499600832109014, 1.9504462746103808, 3.4141788518658616, 0.2783370852553448, 2.01509737577486, 3.49769724559321, 2.0486999251118005, 1.9069365316531375, 1.9063232414331914, 3.3078333510016056, 0.27559225534197296, 1.981680388299563, 3.424961908919033, 1.9955392685572624, 1.8538080278812032, 1.8518434460163142, 3.1773523146671074, 0.27189861962890616, 1.9421441425092318, 3.338662519105913, 1.931036809251427, 1.7893691887819645, 1.7857603538028393, 3.020176778459449, 0.2667363594339594, 1.896225857094417, 3.2380762891116777, 1.8534459525081792, 1.7117690484677544, 1.706142488651944, 2.831599567019092, 0.2608131524162547, 1.8436202405517386, 3.122355600505027, 1.7601872542791974, 1.6181369311711902, 1.6099652659907278, 2.6025693985367897, 0.24873883482629144, 1.783974958696459, 2.990483236041223, 1.6474286521554669, 1.5039752034511054, 1.4924425499449177, 2.312704990925769, 0.2549663357401113, 1.7287881207664255, 2.883502682394885, 1.5639999234445578, 1.4246328795261323, 1.412116695889996, 2.117346706612188, 0.27242925325538286, 1.6799175379390343, 2.8045999684695793, 1.516872017690622, 1.388522993417715, 1.3782095743929004, 2.05113153466342, 0.26873064494111826, 1.6274528794638465, 2.7153087961254894, 1.4617456527685213, 1.3458703902580476, 1.3383852126342235, 1.978676042859031, 0.24916735316863528]
        pydgm.state.phi[0, :, :] = np.reshape(phi_test, (7, 28), 'F')

        pydgm.dgmsolver.dgmsolve()

        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]

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
        pydgm.control.lamb = 0.48

        pydgm.dgmsolver.initialize_dgmsolver()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))
        phi_test = np.array([phi_test for i in range(28)]).flatten()

        pydgm.dgmsolver.dgmsolve()

        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
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
        pydgm.control.lamb = 0.76
        pydgm.control.allow_fission = True
        phi_test = np.array([1.0781901438738859, 1.5439788126739036, 1.0686290157458673, 1.0348940034466163, 1.0409956199943164, 1.670442207080332, 0.2204360523334687])

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
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
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
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
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
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
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
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
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
        pydgm.control.lamb = 0.4

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
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
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
        pydgm.control.lamb = 0.7
        pydgm.control.material_map = [1]

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
        assert_almost_equal(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
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
        assert_almost_equal(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
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
        pydgm.control.lamb = 0.46

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
        assert_almost_equal(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
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
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
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
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
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
        pydgm.control.lamb = 0.43

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
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        pydgm.control.fine_mesh = [2, 3, 3, 2]
        pydgm.control.coarse_mesh = [0.0, 1.0, 2.0, 4.0, 4.5]
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
        pydgm.control.fine_mesh = [2, 3, 3, 2]
        pydgm.control.coarse_mesh = [0.0, 1.0, 2.0, 4.0, 4.5]
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
        pydgm.control.fine_mesh = [2, 3, 3, 2]
        pydgm.control.coarse_mesh = [0.0, 1.0, 2.0, 4.0, 4.5]
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
        pydgm.control.lamb = 0.25

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
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_fine_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 11)

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()
