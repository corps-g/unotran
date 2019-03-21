import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestSOLVER(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [3, 22, 3]
        pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 10
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = False
        pydgm.control.eigen_print = False
        pydgm.control.outer_print = False
        pydgm.control.eigen_tolerance = 1e-15
        pydgm.control.outer_tolerance = 1e-16
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.scatter_legendre_order = 0
        pydgm.control.use_DGM = False
        pydgm.control.max_eigen_iters = 10000
        pydgm.control.max_outer_iters = 1000

    def test_solver_vacuum1(self):
        ''' 
        Test fixed source problem with vacuum conditions
        '''
        # Activate fissioning
        pydgm.control.allow_fission = True

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 7)

        phi_test = [1.6274528794638465, 2.71530879612549, 1.461745652768521, 1.3458703902580473, 1.3383852126342237, 1.9786760428590306, 0.24916735316863525, 1.6799175379390339, 2.8045999684695797, 1.516872017690622, 1.3885229934177148, 1.3782095743929001, 2.051131534663419, 0.26873064494111804, 1.728788120766425, 2.883502682394886, 1.5639999234445578, 1.4246328795261316, 1.4121166958899956, 2.1173467066121874, 0.2724292532553828, 1.7839749586964595, 2.990483236041222, 1.6474286521554664, 1.5039752034511047, 1.4924425499449177, 2.3127049909257686, 0.25496633574011124, 1.8436202405517381, 3.122355600505027, 1.7601872542791979, 1.61813693117119, 1.6099652659907275, 2.60256939853679, 0.24873883482629144, 1.896225857094417, 3.2380762891116794, 1.8534459525081792, 1.7117690484677541, 1.7061424886519436, 2.831599567019092, 0.26081315241625463, 1.9421441425092316, 3.338662519105913, 1.9310368092514267, 1.789369188781964, 1.7857603538028388, 3.0201767784594478, 0.2667363594339594, 1.9816803882995633, 3.424961908919033, 1.9955392685572624, 1.853808027881203, 1.851843446016314, 3.1773523146671065, 0.27189861962890616, 2.0150973757748596, 3.4976972455932094, 2.0486999251118014, 1.9069365316531377, 1.9063232414331912, 3.307833351001605, 0.2755922553419729, 2.042616213943685, 3.5574620224059217, 2.0917047787489365, 1.9499600832109016, 1.9504462746103806, 3.4141788518658602, 0.27833708525534473, 2.0644181962111365, 3.604732065595381, 2.1253588124042495, 1.9836690960190415, 1.985023407898914, 3.497921277464179, 0.28030660972118154, 2.080646338594525, 3.6398748475310785, 2.150203190885212, 2.0085809732608575, 2.0105818574623395, 3.5600286331289643, 0.2816665790912415, 2.0914067095511766, 3.663158139593214, 2.16659102830272, 2.0250269209204395, 2.0274573320958647, 3.6011228563902344, 0.2825198396790823, 2.0967694470675315, 3.6747566727970047, 2.174734975618102, 2.033203922754008, 2.0358487486465924, 3.621580567528384, 0.28293121918903963, 2.0967694470675315, 3.6747566727970042, 2.1747349756181023, 2.033203922754008, 2.0358487486465924, 3.6215805675283836, 0.2829312191890396, 2.0914067095511766, 3.6631581395932136, 2.1665910283027205, 2.02502692092044, 2.0274573320958647, 3.6011228563902358, 0.2825198396790823, 2.080646338594525, 3.639874847531079, 2.150203190885212, 2.008580973260857, 2.01058185746234, 3.5600286331289652, 0.2816665790912415, 2.0644181962111365, 3.6047320655953805, 2.125358812404249, 1.9836690960190408, 1.985023407898914, 3.4979212774641804, 0.2803066097211815, 2.042616213943685, 3.5574620224059217, 2.0917047787489365, 1.9499600832109014, 1.9504462746103808, 3.4141788518658616, 0.2783370852553448, 2.01509737577486, 3.49769724559321, 2.0486999251118005, 1.9069365316531375, 1.9063232414331914, 3.3078333510016056, 0.27559225534197296, 1.981680388299563, 3.424961908919033, 1.9955392685572624, 1.8538080278812032, 1.8518434460163142, 3.1773523146671074, 0.27189861962890616, 1.9421441425092318, 3.338662519105913, 1.931036809251427, 1.7893691887819645, 1.7857603538028393, 3.020176778459449, 0.2667363594339594, 1.896225857094417, 3.2380762891116777, 1.8534459525081792, 1.7117690484677544, 1.706142488651944, 2.831599567019092, 0.2608131524162547, 1.8436202405517386, 3.122355600505027, 1.7601872542791974, 1.6181369311711902, 1.6099652659907278, 2.6025693985367897, 0.24873883482629144, 1.783974958696459, 2.990483236041223, 1.6474286521554669, 1.5039752034511054, 1.4924425499449177, 2.312704990925769, 0.2549663357401113, 1.7287881207664255, 2.883502682394885, 1.5639999234445578, 1.4246328795261323, 1.412116695889996, 2.117346706612188, 0.27242925325538286, 1.6799175379390343, 2.8045999684695793, 1.516872017690622, 1.388522993417715, 1.3782095743929004, 2.05113153466342, 0.26873064494111826, 1.6274528794638465, 2.7153087961254894, 1.4617456527685213, 1.3458703902580476, 1.3383852126342235, 1.978676042859031, 0.24916735316863528]

        # Solve the problem
        pydgm.solver.solve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.mg_phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_inf_med_1g(self):
        '''
        Test infinite medium fixed source problem with reflective conditions in 1g
        '''

        # Set problem conditions
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.material_map = [1, 1, 1]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 1)

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi_test = np.linalg.solve((T - S), np.ones(1))
        phi_test = np.array([phi_test for i in range(28)]).flatten()

        # Solve the problem
        pydgm.solver.solve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.mg_phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_inf_med_2g(self):
        '''
        Test infinite medium fixed source problem with reflective conditions in 2g
        '''

        # Set problem conditions
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.material_map = [1, 1, 1]
        pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi_test = np.linalg.solve((T - S), np.ones(2))
        phi_test = np.array([phi_test for i in range(28)]).flatten()

        # Solve the problem
        pydgm.solver.solve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.mg_phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_inf_med_7g(self):
        '''
        Test infinite medium fixed source problem with reflective conditions in 7g
        '''

        # Set problem conditions
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.material_map = [1, 1, 1]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 7)

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))
        phi_test = np.array([phi_test for i in range(28)]).flatten()

        # Solve the problem
        pydgm.solver.solve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.mg_phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_2med_ref_1g(self):
        '''
        Test fixed source problem with reflective conditions with 1g
        '''

        # Set problem conditions
        pydgm.control.fine_mesh = [5, 5]
        pydgm.control.coarse_mesh = [0.0, 1.0, 2.0]
        pydgm.control.material_map = [2, 1]
        pydgm.control.angle_order = 2
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 1)

        phi_test = [3.2453115765816385, 3.1911557006288493, 3.0781870772267665, 2.8963110347254495, 2.6282848825690057, 2.3200166254673196, 2.0780738596493524, 1.9316794281903034, 1.848396854623421, 1.8106188687176359]

        # Solve the problem
        pydgm.solver.solve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.mg_phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_2med_ref_2g(self):
        '''
        Test fixed source problem with reflective conditions with 2g
        '''

        # Set problem conditions
        pydgm.control.fine_mesh = [5, 5]
        pydgm.control.coarse_mesh = [0.0, 1.0, 2.0]
        pydgm.control.material_map = [2, 1]
        pydgm.control.angle_order = 2
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        phi_test = [3.2453115765816385, 2.637965384358304, 3.1911557006288493, 2.589995786667246, 3.0781870772267665, 2.4861268449530334, 2.8963110347254495, 2.298884153728786, 2.6282848825690057, 1.9010277088673162, 2.3200166254673196, 1.4356296946991254, 2.0780738596493524, 1.1705973629208932, 1.9316794281903034, 1.0564188975526514, 1.848396854623421, 1.0028739873869337, 1.8106188687176359, 0.9806778431098238]

        # Solve the problem
        pydgm.solver.solve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.mg_phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_2med_ref_7g(self):
        '''
        Test fixed source problem with reflective conditions with 7g
        '''

        # Set problem conditions
        pydgm.control.fine_mesh = [5, 5]
        pydgm.control.coarse_mesh = [0.0, 1.0, 2.0]
        pydgm.control.material_map = [1, 1]
        pydgm.control.angle_order = 4
        pydgm.control.boundary_type = [1.0, 1.0]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 7)

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))
        phi_test = np.array([phi_test for i in range(10)]).flatten()

        # Solve the problem
        pydgm.solver.solve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.mg_phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_eigenV1g(self):
        '''
        Test eigenvalue source problem with vacuum conditions and 1g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.max_outer_iters = 1

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 1)

        phi_test = [0.05917851752472814, 0.1101453392055481, 0.1497051827466689, 0.1778507990738045, 0.1924792729907672, 0.1924792729907672, 0.1778507990738046, 0.1497051827466690, 0.1101453392055482, 0.05917851752472817]

        # Solve the problem
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, 0.6893591115415211, 12)

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_eigenV2g(self):
        '''
        Test eigenvalue source problem with vacuum conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        phi_test = [0.05882749189351658, 0.00985808742274279, 0.1099501419733753, 0.019347812329268695, 0.14979920402552727, 0.026176658005947512, 0.1781310366743269, 0.03122421632573625, 0.19286636083585607, 0.03377131381204501, 0.19286636083585607, 0.03377131381204501, 0.17813103667432686, 0.031224216325736256, 0.14979920402552724, 0.026176658005947512, 0.10995014197337528, 0.01934781232926869, 0.058827491893516604, 0.009858087422742794]

        # Solve the problem
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, 0.8099523232983425, 12)

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_eigenV7g(self):
        '''
        Test eigenvalue source problem with vacuum conditions and 7g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.angle_order = 2
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 7)

        phi_test = [0.19050251326520584, 1.9799335510805185, 0.69201814518126, 0.3927000245492841, 0.2622715078950253, 0.20936059119838546, 0.000683954269595958, 0.25253653423327665, 2.8930819653774895, 1.158606945184528, 0.6858113244922716, 0.4639601075261923, 0.4060114930207368, 0.0013808859451732852, 0.30559047625122115, 3.6329637815416556, 1.498034484581793, 0.9026484213739354, 0.6162114941108023, 0.5517562407150877, 0.0018540270157502057, 0.3439534785160265, 4.153277746375052, 1.7302149163096785, 1.0513217539517374, 0.7215915434720093, 0.653666204542615, 0.0022067618449436725, 0.36402899896324237, 4.421934793951583, 1.8489909842118943, 1.127291245982061, 0.7756443978822711, 0.705581398687358, 0.0023773065003326204, 0.36402899896324237, 4.421934793951582, 1.8489909842118946, 1.1272912459820612, 0.7756443978822711, 0.705581398687358, 0.0023773065003326204, 0.34395347851602653, 4.153277746375052, 1.7302149163096785, 1.0513217539517377, 0.7215915434720092, 0.653666204542615, 0.002206761844943672, 0.3055904762512212, 3.6329637815416564, 1.498034484581793, 0.9026484213739353, 0.6162114941108023, 0.5517562407150877, 0.0018540270157502063, 0.2525365342332767, 2.8930819653774895, 1.1586069451845278, 0.6858113244922716, 0.4639601075261923, 0.4060114930207368, 0.0013808859451732852, 0.19050251326520584, 1.9799335510805192, 0.6920181451812601, 0.3927000245492842, 0.26227150789502535, 0.20936059119838543, 0.0006839542695959579]

        # Solve the problem
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, 0.30413628310914226, 12)

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_eigenR1g(self):
        '''
        Test eigenvalue source problem with reflective conditions and 1g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_type = [1.0, 1.0]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 1)

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        X = np.outer(pydgm.material.chi[:, 0], pydgm.material.nu_sig_f[:, 0])

        keff, phi = np.linalg.eig(np.linalg.inv(T - S).dot(X))
        i = np.argmax(keff)
        keff_test = keff[i]
        phi_test = phi[i]

        phi_test = np.array([phi_test for i in range(10)]).flatten()

        # Solve the problem
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_eigenR2g(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_type = [1.0, 1.0]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

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
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_eigenR7g(self):
        '''
        Test eigenvalue source problem with reflective conditions and 7g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.angle_order = 2
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_type = [1.0, 1.0]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 7)

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
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_eigenR1gPin(self):
        '''
        Test eigenvalue source problem with reflective conditions and 1g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh = [3, 10, 3]
        pydgm.control.material_map = [2, 1, 2]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 1)

        # set the test flux
        keff_test = 0.6824742039858390
        phi_test = [0.13341837652120125, 0.13356075080242463, 0.1338459024547955, 0.13470975320182185, 0.13591529539088204, 0.13679997220977602, 0.13738073107705168, 0.13766845210630752, 0.1376684521063075, 0.1373807310770516, 0.1367999722097759, 0.13591529539088193, 0.1347097532018217, 0.13384590245479533, 0.1335607508024245, 0.13341837652120114]

        # Solve the problem
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_eigenR2gPin(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh = [3, 10, 3]
        pydgm.control.material_map = [2, 1, 2]
        pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # set the test flux
        keff_test = 0.8418546852484950
        phi_test = [0.13393183108467394, 0.04663240631432256, 0.13407552941360298, 0.04550808086281801, 0.13436333428621713, 0.043206841474147446, 0.1351651393398092, 0.0384434752119791, 0.13615737742196526, 0.03329929560434661, 0.13674284660888314, 0.030464508103354708, 0.13706978363298242, 0.028970199506203023, 0.13721638515632006, 0.028325674662651124, 0.13721638515632006, 0.028325674662651124, 0.1370697836329824, 0.028970199506203012, 0.13674284660888308, 0.0304645081033547, 0.13615737742196524, 0.03329929560434659, 0.13516513933980914, 0.03844347521197908, 0.13436333428621713, 0.043206841474147425, 0.13407552941360296, 0.045508080862818004, 0.1339318310846739, 0.046632406314322555]

        # Solve the problem
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_eigenR7gPin(self):
        '''
        Test eigenvalue source problem with reflective conditions and 7g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh = [3, 10, 3]
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.angle_order = 2
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 7)

        # set the test flux
        keff_test = 1.0794314789325041
        phi_test = [0.18617101855192203, 2.858915338372074, 1.4772041911943246, 1.0299947729491368, 0.7782291112252604, 0.6601323950057741, 0.0018780861364841711, 0.18615912584783736, 2.8586649211715436, 1.4771766639520822, 1.030040359498237, 0.7782969794725844, 0.6603312122972236, 0.0018857945539742516, 0.18613533094381535, 2.858163988201594, 1.4771216026067822, 1.0301315410919691, 0.7784327301908717, 0.6607289506819793, 0.001901260571677254, 0.1860688679764812, 2.856246121648244, 1.4768054633757053, 1.0308126366153867, 0.7795349485104974, 0.6645605858759833, 0.0020390343478796447, 0.18597801819761287, 2.8534095558345967, 1.4763058823653368, 1.0319062100766097, 0.7813236329806805, 0.6707834262470258, 0.002202198406120643, 0.18591115233580913, 2.851306298580461, 1.4759330573010532, 1.0327026586727457, 0.7826354049117061, 0.6752310599415209, 0.002251216654318464, 0.18586715682184884, 2.8499152893733255, 1.4756853599772022, 1.0332225362074143, 0.7834960048959739, 0.6780933062272468, 0.0022716374303459433, 0.1858453299832966, 2.8492230801887986, 1.475561761882258, 1.0334791905008067, 0.7839221795374745, 0.6794942839316992, 0.002280077669362046, 0.18584532998329656, 2.8492230801887986, 1.475561761882258, 1.0334791905008065, 0.7839221795374745, 0.6794942839316991, 0.0022800776693620455, 0.18586715682184884, 2.8499152893733255, 1.4756853599772024, 1.0332225362074146, 0.7834960048959738, 0.6780933062272467, 0.002271637430345943, 0.18591115233580915, 2.851306298580461, 1.4759330573010532, 1.0327026586727457, 0.7826354049117062, 0.6752310599415207, 0.0022512166543184635, 0.1859780181976129, 2.853409555834596, 1.476305882365337, 1.0319062100766097, 0.7813236329806805, 0.6707834262470258, 0.002202198406120643, 0.1860688679764812, 2.856246121648244, 1.4768054633757055, 1.0308126366153867, 0.7795349485104973, 0.6645605858759831, 0.002039034347879644, 0.18613533094381537, 2.858163988201594, 1.4771216026067824, 1.0301315410919691, 0.7784327301908716, 0.6607289506819792, 0.0019012605716772534, 0.1861591258478374, 2.858664921171543, 1.4771766639520822, 1.0300403594982372, 0.7782969794725842, 0.6603312122972235, 0.0018857945539742511, 0.18617101855192209, 2.8589153383720736, 1.477204191194325, 1.0299947729491368, 0.7782291112252603, 0.660132395005774, 0.0018780861364841707]

        # Solve the problem
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_anisotropic_fixed_vacuum(self):
        # Set the variables for the test
        pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.scatter_legendre_order = 7

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        # set the test flux
        phi_test = [[1.402916, 1.480212, 1.549589, 1.626816, 1.708564, 1.777336, 1.834982, 1.882947, 1.922353, 1.954062, 1.97873, 1.99684, 2.008733, 2.014627, 2.014627, 2.008733, 1.99684, 1.97873, 1.954062, 1.922353, 1.882947, 1.834982, 1.777336, 1.708564, 1.626816, 1.549589, 1.480212, 1.402916, -0.5780989, -0.5492885, -0.5205376, -0.4829597, -0.4365932, -0.3903496, -0.3442092, -0.298155, -0.2521721, -0.2062473, -0.1603684, -0.1145245, -0.06870502, -0.02290007, 0.02290007, 0.06870502, 0.1145245, 0.1603684, 0.2062473, 0.2521721, 0.298155, 0.3442092, 0.3903496, 0.4365932, 0.4829597, 0.5205376, 0.5492885, 0.5780989, -0.1562649, -0.1877694, -0.2156792, -0.2453879, -0.2755873, -0.300373, -0.3206687, -0.3371943, -0.3505086, -0.3610415, -0.3691193, -0.3749832, -0.3788028, -0.3806863, -0.3806863, -0.3788028, -0.3749832, -0.3691193, -0.3610415, -0.3505086, -0.3371943, -0.3206687, -0.300373, -0.2755873, -0.2453879, -0.2156792, -0.1877694, -0.1562649, 0.2398292, 0.2246475, 0.2102004, 0.1921321, 0.1706907, 0.1503306, 0.1308587, 0.1121136, 0.09395839, 0.07627536, 0.05896129, 0.0419239, 0.02507885, 0.008347134, -0.008347134, -0.02507885, -0.0419239, -0.05896129, -0.07627536, -0.09395839, -0.1121136, -0.1308587, -0.1503306, -0.1706907, -0.1921321, -0.2102004, -0.2246475, -0.2398292, 0.03127032, 0.04774343, 0.06197699, 0.07667378, 0.09111516, 0.102442, 0.1113024, 0.1181998, 0.1235236, 0.127572, 0.130571, 0.1326865, 0.1340354, 0.1346918, 0.1346918, 0.1340354, 0.1326865, 0.130571, 0.127572, 0.1235236, 0.1181998, 0.1113024, 0.102442, 0.09111516, 0.07667378, 0.06197699, 0.04774343, 0.03127032, -0.125933, -0.1150924, -0.1053465, -0.09376179, -0.08064968, -0.06902323, -0.05859353, -0.04912475, -0.04042225, -0.03232321, -0.02468906, -0.01739944, -0.01034713, -0.003433788, 0.003433788, 0.01034713, 0.01739944, 0.02468906, 0.03232321, 0.04042225, 0.04912475, 0.05859353, 0.06902323, 0.08064968, 0.09376179, 0.1053465, 0.1150924, 0.125933, -0.007098792, -0.01687172, -0.02511613, -0.03336916, -0.04118888, -0.047011, -0.05131073, -0.0544555, -0.05672796, -0.05834376, -0.05946511, -0.06021096, -0.06066453, -0.06087854, -0.06087854, -0.06066453, -0.06021096, -0.05946511, -0.05834376, -0.05672796, -0.0544555, -0.05131073, -0.047011, -0.04118888, -0.03336916, -0.02511613, -0.01687172, -0.007098792, 0.07811477, 0.06921594, 0.06155096, 0.05284321, 0.04341409, 0.03558954, 0.02903647, 0.02348776, 0.01872736, 0.01457858, 0.01089474, 0.007551526, 0.004440838, 0.001465472, -0.001465472, -0.004440838, -0.007551526, -0.01089474, -0.01457858, -0.01872736, -0.02348776, -0.02903647, -0.03558954, -0.04341409, -0.05284321, -0.06155096, -0.06921594, -0.07811477], [1.22148, 1.373229, 1.502981, 1.614577, 1.706907, 1.782879, 1.845998, 1.898507, 1.941852, 1.976982, 2.004519, 2.024876, 2.038317, 2.045001, 2.045001, 2.038317, 2.024876, 2.004519, 1.976982, 1.941852, 1.898507, 1.845998, 1.782879, 1.706907, 1.614577, 1.502981, 1.373229, 1.22148, -0.5486675, -0.5178567, -0.4870263, -0.4496929, -0.4060223, -0.3626349, -0.3194807, -0.2765202, -0.2337202, -0.1910519, -0.148489, -0.1060067, -0.06358135, -0.02119002, 0.02119002, 0.06358135, 0.1060067, 0.148489, 0.1910519, 0.2337202, 0.2765202, 0.3194807, 0.3626349, 0.4060223, 0.4496929, 0.4870263, 0.5178567, 0.5486675, -0.04509524, -0.09083662, -0.1273229, -0.1573565, -0.1806772, -0.1981428, -0.211493, -0.2218365, -0.2298881, -0.2361134, -0.2408182, -0.244203, -0.2463965, -0.2474753, -0.2474753, -0.2463965, -0.244203, -0.2408182, -0.2361134, -0.2298881, -0.2218365, -0.211493, -0.1981428, -0.1806772, -0.1573565, -0.1273229, -0.09083662, -0.04509524, 0.1628073, 0.1477466, 0.1359778, 0.1221973, 0.105935, 0.09141182, 0.07821556, 0.06604919, 0.0546893, 0.04396007, 0.0337167, 0.02383472, 0.01420287, 0.004718104, -0.004718104, -0.01420287, -0.02383472, -0.0337167, -0.04396007, -0.0546893, -0.06604919, -0.07821556, -0.09141182, -0.105935, -0.1221973, -0.1359778, -0.1477466, -0.1628073, -0.00487541, 0.01573118, 0.03057912, 0.04184279, 0.04950789, 0.05410669, 0.05681218, 0.05835478, 0.05919161, 0.05961017, 0.0597918, 0.05985051, 0.05985653, 0.05985031, 0.05985031, 0.05985653, 0.05985051, 0.0597918, 0.05961017, 0.05919161, 0.05835478, 0.05681218, 0.05410669, 0.04950789, 0.04184279, 0.03057912, 0.01573118, -0.00487541, -0.07005798, -0.05851431, -0.05124847, -0.04353958, -0.03467055, -0.02774893, -0.02222036, -0.01770875, -0.01395069, -0.0107545, -0.007974574, -0.005494951, -0.003218577, -0.001059992, 0.001059992, 0.003218577, 0.005494951, 0.007974574, 0.0107545, 0.01395069, 0.01770875, 0.02222036, 0.02774893, 0.03467055, 0.04353958, 0.05124847, 0.05851431, 0.07005798, 0.005450864, -0.005721242, -0.01297235, -0.01794268, -0.02069357, -0.02162339, -0.02155434, -0.02098306, -0.02020816, -0.01940808, -0.01868846, -0.018111, -0.01771079, -0.01750663, -0.01750663, -0.01771079, -0.018111, -0.01868846, -0.01940808, -0.02020816, -0.02098306, -0.02155434, -0.02162339, -0.02069357, -0.01794268, -0.01297235, -0.005721242, 0.005450864, 0.03824504, 0.02836365, 0.02303511, 0.01789961, 0.01212374, 0.008191002, 0.00549294, 0.003636248, 0.00236177, 0.001494412, 0.0009119888, 0.0005256335, 0.0002672625, 8.127639e-05, -8.127639e-05, -0.0002672625, -0.0005256335, -0.0009119888, -0.001494412, -0.00236177, -0.003636248, -0.00549294, -0.008191002, -0.01212374, -0.01789961, -0.02303511, -0.02836365, -0.03824504]]

        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_legendre_order + 1, -1)

        # Solve the problem
        pydgm.solver.solve()

        for l in range(pydgm.control.scatter_legendre_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()

            test = phi_test[:, l].flatten()

            np.testing.assert_array_almost_equal(phi, test, 6)

    def test_solver_anisotropic_symmetric(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [12]
        pydgm.control.coarse_mesh = [0.0, 5.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/2g_symmetric.anlxs'.ljust(256)
        pydgm.control.angle_order = 4
        pydgm.control.scatter_legendre_order = 1
        pydgm.control.boundary_type = [0.0, 0.0]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        # Solve the problem
        pydgm.solver.solve()

        # Test that the scalar flux is symmetric
        phi0 = pydgm.state.mg_phi[0, 0, :].flatten()
        phi1 = pydgm.state.mg_phi[0, 1, :].flatten()
        np.testing.assert_array_almost_equal(phi0, phi1, 12)

    def test_solver_anisotropic_fixed_reflect(self):
        # Set the variables for the test
        pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.scatter_legendre_order = 7
        pydgm.control.boundary_type = [1.0, 1.0]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        # set the test flux
        phi_test = [[30.76119, 30.75972, 30.75674, 30.75181, 30.74625, 30.74176, 30.73813, 30.73522, 30.7329, 30.73109, 30.72971, 30.72871, 30.72807, 30.72775, 30.72775, 30.72807, 30.72871, 30.72971, 30.73109, 30.7329, 30.73522, 30.73813, 30.74176, 30.74625, 30.75181, 30.75674, 30.75972, 30.76119, 0.0022698, 0.00680983, 0.01135114, 0.01299083, 0.01173218, 0.01048102, 0.00923596, 0.007995856, 0.006759729, 0.005526751, 0.00429621, 0.003067479, 0.001839999, 0.0006132526, -0.0006132526, -0.001839999, -0.003067479, -0.00429621, -0.005526751, -0.006759729, -0.007995856, -0.00923596, -0.01048102, -0.01173218, -0.01299083, -0.01135114, -0.00680983, -0.0022698, -0.009846732, -0.009167684, -0.00779404, -0.005545983, -0.003055195, -0.001066293, 0.0005188634, 0.001776495, 0.002765512, 0.003531066, 0.004107268, 0.004519236, 0.004784601, 0.004914565, 0.004914565, 0.004784601, 0.004519236, 0.004107268, 0.003531066, 0.002765512, 0.001776495, 0.0005188634, -0.001066293, -0.003055195, -0.005545983, -0.00779404, -0.009167684, -0.009846732, -0.001398409, -0.004203415, -0.00703315, -0.007934677, -0.006939387, -0.006030363, -0.005190272, -0.004405083, -0.00366333, -0.002955532, -0.002273721, -0.00161106, -0.0009615313, -0.0003196701, 0.0003196701, 0.0009615313, 0.00161106, 0.002273721, 0.002955532, 0.00366333, 0.004405083, 0.005190272, 0.006030363, 0.006939387, 0.007934677, 0.00703315, 0.004203415, 0.001398409, 0.005900857, 0.005477797, 0.00462052, 0.003222081, 0.001690197, 0.000491691, -0.0004435364, -0.001169856, -0.001729228, -0.00215377, -0.00246771, -0.002688866, -0.002829734, -0.002898244, -0.002898244, -0.002829734, -0.002688866, -0.00246771, -0.00215377, -0.001729228, -0.001169856, -0.0004435364, 0.000491691, 0.001690197, 0.003222081, 0.00462052, 0.005477797, 0.005900857, 0.001021291, 0.00307741, 0.005174453, 0.005741801, 0.004827285, 0.00404751, 0.003374251, 0.002784917, 0.00226129, 0.001788519, 0.00135432, 0.0009483263, 0.0005615569, 0.0001859634, -0.0001859634, -0.0005615569, -0.0009483263, -0.00135432, -0.001788519, -0.00226129, -0.002784917, -0.003374251, -0.00404751, -0.004827285, -0.005741801, -0.005174453, -0.00307741, -0.001021291, -0.003882712, -0.003597428, -0.003018459, -0.002076746, -0.001055543, -0.0002712218, 0.0003288076, 0.0007852529, 0.001129467, 0.001385398, 0.001571074, 0.001699733, 0.001780639, 0.001819669, 0.001819669, 0.001780639, 0.001699733, 0.001571074, 0.001385398, 0.001129467, 0.0007852529, 0.0003288076, -0.0002712218, -0.001055543, -0.002076746, -0.003018459, -0.003597428, -0.003882712, -0.0008116857, -0.002451341, -0.004140259, -0.004526624, -0.003666721, -0.002966135, -0.002390452, -0.00191235, -0.001510003, -0.001165815, -0.0008654081, -0.0005968077, -0.0003497721, -0.0001152273, 0.0001152273, 0.0003497721, 0.0005968077, 0.0008654081, 0.001165815, 0.001510003, 0.00191235, 0.002390452, 0.002966135, 0.003666721, 0.004526624, 0.004140259, 0.002451341, 0.0008116857], [22.04346, 22.01602, 21.95719, 21.87895, 21.80985, 21.75958, 21.72206, 21.69348, 21.67148, 21.65459, 21.64186, 21.63271, 21.62679, 21.62387, 21.62387, 21.62679, 21.63271, 21.64186, 21.65459, 21.67148, 21.69348, 21.72206, 21.75958, 21.80985, 21.87895, 21.95719, 22.01602, 22.04346, 0.02446302, 0.07339242, 0.1223324, 0.1397673, 0.1258324, 0.1121438, 0.09863599, 0.08526417, 0.07199624, 0.05880815, 0.04568086, 0.03259845, 0.01954698, 0.006513665, -0.006513665, -0.01954698, -0.03259845, -0.04568086, -0.05880815, -0.07199624, -0.08526417, -0.09863599, -0.1121438, -0.1258324, -0.1397673, -0.1223324, -0.07339242, -0.02446302, -0.1047343, -0.09379987, -0.06994558, -0.03807634, -0.01068506, 0.008032781, 0.02111448, 0.03044583, 0.03720317, 0.0421237, 0.04567071, 0.04813446, 0.04969315, 0.05044915, 0.05044915, 0.04969315, 0.04813446, 0.04567071, 0.0421237, 0.03720317, 0.03044583, 0.02111448, 0.008032781, -0.01068506, -0.03807634, -0.06994558, -0.09379987, -0.1047343, -0.01210323, -0.03675097, -0.06280258, -0.07017218, -0.0590317, -0.04980513, -0.04190584, -0.03494773, -0.02866943, -0.02288753, -0.0174675, -0.01230551, -0.007316805, -0.002428041, 0.002428041, 0.007316805, 0.01230551, 0.0174675, 0.02288753, 0.02866943, 0.03494773, 0.04190584, 0.04980513, 0.0590317, 0.07017218, 0.06280258, 0.03675097, 0.01210323, 0.05511964, 0.04906855, 0.0355585, 0.01742902, 0.0024258, -0.00694562, -0.01282426, -0.01652721, -0.01886723, -0.02034678, -0.02127655, -0.02184764, -0.02217522, -0.02232434, -0.02232434, -0.02217522, -0.02184764, -0.02127655, -0.02034678, -0.01886723, -0.01652721, -0.01282426, -0.00694562, 0.0024258, 0.01742902, 0.0355585, 0.04906855, 0.05511964, 0.007127174, 0.02201507, 0.03893293, 0.04239183, 0.03258143, 0.0254044, 0.01997882, 0.01573951, 0.01231735, 0.009465068, 0.007011263, 0.004831781, 0.002831577, 0.0009328951, -0.0009328951, -0.002831577, -0.004831781, -0.007011263, -0.009465068, -0.01231735, -0.01573951, -0.01997882, -0.0254044, -0.03258143, -0.04239183, -0.03893293, -0.02201507, -0.007127174, -0.0317884, -0.02810615, -0.01969277, -0.008372873, 0.0006445864, 0.005758553, 0.008547714, 0.009973088, 0.01061884, 0.01083921, 0.01084809, 0.01077346, 0.01069019, 0.01063946, 0.01063946, 0.01069019, 0.01077346, 0.01084809, 0.01083921, 0.01061884, 0.009973088, 0.008547714, 0.005758553, 0.0006445864, -0.008372873, -0.01969277, -0.02810615, -0.0317884, -0.004580231, -0.01446082, -0.02666308, -0.02828494, -0.01951068, -0.01366167, -0.009687533, -0.006929441, -0.004969681, -0.003539329, -0.002461366, -0.001615426, -0.0009156867, -0.0002966308, 0.0002966308, 0.0009156867, 0.001615426, 0.002461366, 0.003539329, 0.004969681, 0.006929441, 0.009687533, 0.01366167, 0.01951068, 0.02828494, 0.02666308, 0.01446082, 0.004580231]]

        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_legendre_order + 1, -1)

        # Solve the problem
        pydgm.solver.solve()

        for l in range(pydgm.control.scatter_legendre_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()

            test = phi_test[:, l].flatten()

            np.testing.assert_array_almost_equal(phi, test, 6)

    def test_solver_partisn_eigen_2g_l0(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh = [2, 1]
        pydgm.control.coarse_mesh = [0.0, 1.5, 2.0]
        pydgm.control.material_map = [1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.scatter_legendre_order = 0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux
        phi_test = [[3.43974, 3.392601, 3.313021], [1.044395, 1.097075, 1.188267]]

        # phi_test = np.array([[3.433326, 3.433288, 3.433213, 3.4331, 3.432949, 3.432759, 3.432531, 3.432263, 3.431955, 3.431607, 3.431217, 3.430785, 3.43031, 3.429791, 3.429226, 3.428615, 3.427955, 3.427246, 3.426485, 3.425671, 3.424802, 3.423875, 3.422888, 3.421839, 3.420725, 3.419543, 3.41829, 3.416962, 3.415555, 3.414067, 3.412491, 3.410824, 3.409061, 3.407196, 3.405224, 3.403138, 3.400931, 3.398597, 3.396128, 3.393515, 3.390749, 3.387821, 3.384719, 3.381434, 3.377952, 3.37426, 3.370345, 3.366191, 3.361781, 3.357098, 3.352582, 3.348515, 3.344729, 3.341211, 3.33795, 3.334938, 3.332164, 3.329621, 3.3273, 3.325196, 3.323303, 3.321613, 3.320124, 3.318829, 3.317727, 3.316813, 3.316085, 3.31554, 3.315178, 3.314998, 0.0004094004, 0.001228307, 0.00204753, 0.002867283, 0.003687776, 0.004509223, 0.005331837, 0.006155833, 0.006981427, 0.007808836, 0.00863828, 0.009469979, 0.01030416, 0.01114104, 0.01198085, 0.01282383, 0.01367021, 0.01452023, 0.01537413, 0.01623215, 0.01709456, 0.01796161, 0.01883356, 0.01971069, 0.02059327, 0.0214816, 0.02237596, 0.02327668, 0.02418406, 0.02509845, 0.02602018, 0.02694963, 0.02788717, 0.0288332, 0.02978816, 0.03075248, 0.03172665, 0.03271118, 0.03370661, 0.03471354, 0.0357326, 0.03676448, 0.03780992, 0.03886974, 0.03994483, 0.04103617, 0.04214482, 0.043272, 0.044419, 0.0455873, 0.04501365, 0.04268848, 0.04036613, 0.03804639, 0.03572907, 0.033414, 0.03110099, 0.02878989, 0.02648052, 0.02417273, 0.02186636, 0.01956128, 0.01725733, 0.01495437, 0.01265226, 0.01035088, 0.00805008, 0.005749734, 0.003449712, 0.001149882], [1.04734, 1.04739, 1.047492, 1.047644, 1.047847, 1.048102, 1.048407, 1.048764, 1.049173, 1.049633, 1.050146, 1.050712, 1.05133, 1.052002, 1.052729, 1.05351, 1.054346, 1.055239, 1.056188, 1.057196, 1.058262, 1.059389, 1.060577, 1.061828, 1.063143, 1.064525, 1.065976, 1.067497, 1.069091, 1.070762, 1.072512, 1.074346, 1.076267, 1.078281, 1.080392, 1.082607, 1.084933, 1.087377, 1.08995, 1.09266, 1.09552, 1.098544, 1.101747, 1.105145, 1.10876, 1.112615, 1.116735, 1.121151, 1.125898, 1.131015, 1.137325, 1.144257, 1.150486, 1.156095, 1.161153, 1.165716, 1.169832, 1.17354, 1.176872, 1.179856, 1.182513, 1.184862, 1.186919, 1.188695, 1.1902, 1.191444, 1.192431, 1.193168, 1.193657, 1.193901, -0.0003617221, -0.001085242, -0.00180899, -0.002533119, -0.003257779, -0.003983126, -0.004709312, -0.005436491, -0.00616482, -0.006894453, -0.007625549, -0.008358266, -0.009092764, -0.009829207, -0.01056776, -0.01130858, -0.01205185, -0.01279773, -0.0135464, -0.01429804, -0.01505283, -0.01581094, -0.01657259, -0.01733795, -0.01810722, -0.01888063, -0.01965837, -0.02044067, -0.02122776, -0.02201987, -0.02281726, -0.02362018, -0.02442892, -0.02524375, -0.02606498, -0.02689293, -0.02772795, -0.0285704, -0.02942068, -0.0302792, -0.03114642, -0.03202284, -0.03290899, -0.03380545, -0.03471287, -0.03563194, -0.03656344, -0.03750822, -0.03846724, -0.03944153, -0.03892121, -0.03690057, -0.0348842, -0.03287175, -0.03086291, -0.02885737, -0.02685485, -0.0248551, -0.02285785, -0.02086288, -0.01886996, -0.01687886, -0.01488938, -0.01290132, -0.01091447, -0.008928642, -0.006943643, -0.004959288, -0.00297539, -0.0009917663]])
        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_legendre_order + 1, -1)  # Group, legendre, cell

        keff_test = 1.17502312

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        phi_zero = pydgm.state.mg_phi[0, :, :].flatten()
        # phi_one = pydgm.state.mg_phi[1, :, :].flatten()

        phi_zero_test = phi_test[:, 0].flatten() / np.linalg.norm(phi_test[:, 0]) * np.linalg.norm(phi_zero)
        # phi_one_test = phi_test[:, 1].flatten() / np.linalg.norm(phi_test[:, 0]) * np.linalg.norm(phi_zero)

        np.testing.assert_array_almost_equal(phi_zero, phi_zero_test, 6)
        # np.testing.assert_array_almost_equal(phi_one, phi_one_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 6)

    def test_solver_partisn_eigen_2g_l7(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh = [2, 1]
        pydgm.control.coarse_mesh = [0.0, 1.5, 2.0]
        pydgm.control.material_map = [1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.scatter_legendre_order = 7

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux
        phi_test = [[3.433248, 3.393178, 3.323417, 0.01039521, 0.03345433, 0.02305913, -0.01460567, -0.002353428, 0.0241502, -0.00287316, -0.01353507, -0.01066191, 0.005629042, 0.002111631, -0.0103497, 0.0002272388, 0.00590844, 0.005681201, -0.002213293, -0.001547421, 0.005102069, 0.000626036, -0.002848882, -0.003474918], [1.051019, 1.092742, 1.169623, -0.009161827, -0.02911314, -0.01995131, 0.009478481, 0.004575, -0.01219504, 0.0006244532, 0.005791658, 0.005167205, -0.001474019, -0.001966079, 0.003070435, 0.0005132227, -0.001251699, -0.001764921, 5.642222e-05, 0.000990359, -0.0009445576, -0.0004808684, 0.000275483, 0.0007563515]]

        # phi_test = np.array([[3.433326, 3.433288, 3.433213, 3.4331, 3.432949, 3.432759, 3.432531, 3.432263, 3.431955, 3.431607, 3.431217, 3.430785, 3.43031, 3.429791, 3.429226, 3.428615, 3.427955, 3.427246, 3.426485, 3.425671, 3.424802, 3.423875, 3.422888, 3.421839, 3.420725, 3.419543, 3.41829, 3.416962, 3.415555, 3.414067, 3.412491, 3.410824, 3.409061, 3.407196, 3.405224, 3.403138, 3.400931, 3.398597, 3.396128, 3.393515, 3.390749, 3.387821, 3.384719, 3.381434, 3.377952, 3.37426, 3.370345, 3.366191, 3.361781, 3.357098, 3.352582, 3.348515, 3.344729, 3.341211, 3.33795, 3.334938, 3.332164, 3.329621, 3.3273, 3.325196, 3.323303, 3.321613, 3.320124, 3.318829, 3.317727, 3.316813, 3.316085, 3.31554, 3.315178, 3.314998, 0.0004094004, 0.001228307, 0.00204753, 0.002867283, 0.003687776, 0.004509223, 0.005331837, 0.006155833, 0.006981427, 0.007808836, 0.00863828, 0.009469979, 0.01030416, 0.01114104, 0.01198085, 0.01282383, 0.01367021, 0.01452023, 0.01537413, 0.01623215, 0.01709456, 0.01796161, 0.01883356, 0.01971069, 0.02059327, 0.0214816, 0.02237596, 0.02327668, 0.02418406, 0.02509845, 0.02602018, 0.02694963, 0.02788717, 0.0288332, 0.02978816, 0.03075248, 0.03172665, 0.03271118, 0.03370661, 0.03471354, 0.0357326, 0.03676448, 0.03780992, 0.03886974, 0.03994483, 0.04103617, 0.04214482, 0.043272, 0.044419, 0.0455873, 0.04501365, 0.04268848, 0.04036613, 0.03804639, 0.03572907, 0.033414, 0.03110099, 0.02878989, 0.02648052, 0.02417273, 0.02186636, 0.01956128, 0.01725733, 0.01495437, 0.01265226, 0.01035088, 0.00805008, 0.005749734, 0.003449712, 0.001149882], [1.04734, 1.04739, 1.047492, 1.047644, 1.047847, 1.048102, 1.048407, 1.048764, 1.049173, 1.049633, 1.050146, 1.050712, 1.05133, 1.052002, 1.052729, 1.05351, 1.054346, 1.055239, 1.056188, 1.057196, 1.058262, 1.059389, 1.060577, 1.061828, 1.063143, 1.064525, 1.065976, 1.067497, 1.069091, 1.070762, 1.072512, 1.074346, 1.076267, 1.078281, 1.080392, 1.082607, 1.084933, 1.087377, 1.08995, 1.09266, 1.09552, 1.098544, 1.101747, 1.105145, 1.10876, 1.112615, 1.116735, 1.121151, 1.125898, 1.131015, 1.137325, 1.144257, 1.150486, 1.156095, 1.161153, 1.165716, 1.169832, 1.17354, 1.176872, 1.179856, 1.182513, 1.184862, 1.186919, 1.188695, 1.1902, 1.191444, 1.192431, 1.193168, 1.193657, 1.193901, -0.0003617221, -0.001085242, -0.00180899, -0.002533119, -0.003257779, -0.003983126, -0.004709312, -0.005436491, -0.00616482, -0.006894453, -0.007625549, -0.008358266, -0.009092764, -0.009829207, -0.01056776, -0.01130858, -0.01205185, -0.01279773, -0.0135464, -0.01429804, -0.01505283, -0.01581094, -0.01657259, -0.01733795, -0.01810722, -0.01888063, -0.01965837, -0.02044067, -0.02122776, -0.02201987, -0.02281726, -0.02362018, -0.02442892, -0.02524375, -0.02606498, -0.02689293, -0.02772795, -0.0285704, -0.02942068, -0.0302792, -0.03114642, -0.03202284, -0.03290899, -0.03380545, -0.03471287, -0.03563194, -0.03656344, -0.03750822, -0.03846724, -0.03944153, -0.03892121, -0.03690057, -0.0348842, -0.03287175, -0.03086291, -0.02885737, -0.02685485, -0.0248551, -0.02285785, -0.02086288, -0.01886996, -0.01687886, -0.01488938, -0.01290132, -0.01091447, -0.008928642, -0.006943643, -0.004959288, -0.00297539, -0.0009917663]])
        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_legendre_order + 1, -1)  # Group, legendre, cell

        keff_test = 1.17598606

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        phi_zero = pydgm.state.mg_phi[0, :, :].flatten()
        # phi_one = pydgm.state.mg_phi[1, :, :].flatten()

        phi_zero_test = phi_test[:, 0].flatten() / np.linalg.norm(phi_test[:, 0]) * np.linalg.norm(phi_zero)
        # phi_one_test = phi_test[:, 1].flatten() / np.linalg.norm(phi_test[:, 0]) * np.linalg.norm(phi_zero)

        np.testing.assert_array_almost_equal(phi_zero, phi_zero_test, 6)
        # np.testing.assert_array_almost_equal(phi_one, phi_one_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 6)

    def test_solver_1loop(self):

        def set_parameters():
            self.setUp()
            # define the nonstandard test variables
            pydgm.control.solver_type = 'eigen'.ljust(256)
            pydgm.control.source_value = 0.0
            pydgm.control.allow_fission = True
            pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
            pydgm.control.fine_mesh = [2, 1, 2]
            pydgm.control.coarse_mesh = [0.0, 5.0, 6.0, 11.0]
            pydgm.control.material_map = [1, 5, 3]  # UO2 | water | MOX
            pydgm.control.boundary_type = [0.0, 0.0]  # Vacuum | Vacuum

        #-----------------------------------------------------------------------
        # Solve reference problem
        #-----------------------------------------------------------------------

        # Get reference solution
        set_parameters()

        # Initialize the dependancies
        pydgm.solver.initialize_solver()
        pydgm.solver.solve()

        # Save reference values
        ref_keff = pydgm.state.keff * 1
        ref_phi = pydgm.state.mg_phi * 1
        ref_psi = pydgm.state.mg_psi * 1

        # Clean up the problem
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()

        #-----------------------------------------------------------------------
        # Solve test problem
        #-----------------------------------------------------------------------
        set_parameters()
        # limit to a single iteration
        pydgm.control.max_recon_iters = 1
        pydgm.control.max_eigen_iters = 1
        pydgm.control.max_outer_iters = 1
        pydgm.control.max_inner_iters = 1

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        # Set the initial values
        pydgm.state.keff = ref_keff
        pydgm.state.mg_phi = ref_phi
        pydgm.state.mg_psi = ref_psi

        # Solve the problem
        pydgm.solver.solve()

        np.testing.assert_array_almost_equal(pydgm.state.keff, ref_keff, 12)
        np.testing.assert_array_almost_equal(pydgm.state.mg_phi, ref_phi, 12)
        np.testing.assert_array_almost_equal(pydgm.state.mg_psi, ref_psi, 12)

    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()
