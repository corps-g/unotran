import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestSOLVER(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.spatial_dimension = 1
        pydgm.control.fine_mesh_x = [3, 22, 3]
        pydgm.control.coarse_mesh_x = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 10
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 0.0
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
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
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
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
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
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
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
        pydgm.control.fine_mesh_x = [5, 5]
        pydgm.control.coarse_mesh_x = [0.0, 1.0, 2.0]
        pydgm.control.material_map = [2, 1]
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
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
        pydgm.control.fine_mesh_x = [5, 5]
        pydgm.control.coarse_mesh_x = [0.0, 1.0, 2.0]
        pydgm.control.material_map = [2, 1]
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
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
        pydgm.control.fine_mesh_x = [5, 5]
        pydgm.control.coarse_mesh_x = [0.0, 1.0, 2.0]
        pydgm.control.material_map = [1, 1]
        pydgm.control.angle_order = 4
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0

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
        pydgm.control.fine_mesh_x = [10]
        pydgm.control.coarse_mesh_x = [0.0, 10.0]
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
        pydgm.control.fine_mesh_x = [10]
        pydgm.control.coarse_mesh_x = [0.0, 10.0]
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
        pydgm.control.fine_mesh_x = [10]
        pydgm.control.coarse_mesh_x = [0.0, 10.0]
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
        pydgm.control.fine_mesh_x = [10]
        pydgm.control.coarse_mesh_x = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0

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
        pydgm.control.fine_mesh_x = [10]
        pydgm.control.coarse_mesh_x = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0

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
        pydgm.control.fine_mesh_x = [10]
        pydgm.control.coarse_mesh_x = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.angle_order = 2
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0

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
        pydgm.control.fine_mesh_x = [3, 10, 3]
        pydgm.control.material_map = [2, 1, 2]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
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
        pydgm.control.fine_mesh_x = [3, 10, 3]
        pydgm.control.material_map = [2, 1, 2]
        pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
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
        pydgm.control.fine_mesh_x = [3, 10, 3]
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
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
        pydgm.control.coarse_mesh_x = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.scatter_legendre_order = 7

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        # set the test flux
        phi_test = [[[1.4029164436301702, 1.4802119704475423, 1.5495888971817153, 1.626815932451072, 1.7085644741288557, 1.7773357033976998, 1.8349818057576361, 1.8829471863949965, 1.9223530024630586, 1.9540623273187099, 1.9787299685145165, 1.9968400126457129, 2.008733411830428, 2.01462731175292, 2.01462731175292, 2.008733411830428, 1.9968400126457133, 1.978729968514517, 1.95406232731871, 1.922353002463059, 1.8829471863949974, 1.8349818057576368, 1.7773357033977002, 1.7085644741288561, 1.6268159324510723, 1.5495888971817156, 1.4802119704475425, 1.4029164436301704], [-0.5780989114905826, -0.5492884933838695, -0.520537582666649, -0.482959651628116, -0.4365932342866728, -0.39034964765757363, -0.34420923792593605, -0.2981550279181293, -0.2521721286908298, -0.20624726936044147, -0.16036841852669964, -0.11452447573844954, -0.06870501543022531, -0.022900068787467087, 0.022900068787466428, 0.06870501543022475, 0.11452447573844896, 0.16036841852669909, 0.20624726936044097, 0.25217212869082944, 0.298155027918129, 0.34420923792593566, 0.39034964765757324, 0.43659323428667257, 0.48295965162811566, 0.5205375826666488, 0.5492884933838693, 0.5780989114905825], [-0.15626492240297754, -0.1877693930174307, -0.2156792311760249, -0.2453879170735357, -0.27558726182945853, -0.30037303958839967, -0.3206687207708231, -0.33719431063729743, -0.35050856017019766, -0.36104150519823525, -0.3691193422376261, -0.3749831757292076, -0.3788027926419254, -0.3806863133601728, -0.3806863133601728, -0.3788027926419255, -0.3749831757292077, -0.36911934223762627, -0.36104150519823547, -0.3505085601701978, -0.33719431063729777, -0.3206687207708234, -0.3003730395883999, -0.27558726182945875, -0.2453879170735359, -0.21567923117602505, -0.1877693930174309, -0.15626492240297774], [0.23982918604971462, 0.22464749026295341, 0.21020037865558608, 0.19213205019607393, 0.17069065874065356, 0.1503305787366992, 0.13085870516614978, 0.11211358400144683, 0.09395839332823447, 0.07627536083165615, 0.058961285265606025, 0.04192389713220073, 0.02507884524433541, 0.008347134236776584, -0.008347134236776382, -0.0250788452443352, -0.04192389713220062, -0.058961285265605935, -0.07627536083165609, -0.09395839332823444, -0.11211358400144666, -0.13085870516614972, -0.15033057873669903, -0.1706906587406533, -0.19213205019607377, -0.21020037865558602, -0.2246474902629534, -0.23982918604971454], [0.03127032376366864, 0.04774343491417285, 0.0619769912505068, 0.07667377682595822, 0.09111515744575428, 0.10244202546348577, 0.11130243175898469, 0.11819984834749346, 0.12352355362024048, 0.1275720487039491, 0.13057095388100948, 0.13268649215310283, 0.1340353938018044, 0.13469183428532921, 0.13469183428532927, 0.1340353938018044, 0.13268649215310288, 0.13057095388100956, 0.12757204870394917, 0.12352355362024063, 0.11819984834749361, 0.11130243175898485, 0.10244202546348594, 0.09111515744575446, 0.07667377682595838, 0.06197699125050698, 0.04774343491417303, 0.03127032376366885], [-0.12593303200825925, -0.11509238928686001, -0.10534645626989476, -0.09376178850299745, -0.08064968274884363, -0.06902322779279649, -0.05859353076024055, -0.049124750961078205, -0.04042225332753143, -0.032323206361740915, -0.02468905804437354, -0.01739943874686583, -0.01034712805913645, -0.0034337879690522904, 0.0034337879690521586, 0.010347128059136304, 0.017399438746865767, 0.024689058044373477, 0.03232320636174092, 0.04042225332753144, 0.04912475096107815, 0.05859353076024054, 0.0690232277927964, 0.08064968274884354, 0.09376178850299738, 0.10534645626989475, 0.11509238928686001, 0.12593303200825928], [-0.007098791725427716, -0.016871717971064522, -0.02511613104016501, -0.03336915591039636, -0.041188881609202255, -0.04701099960337066, -0.051310731519584325, -0.054455496693975654, -0.056727955554454854, -0.058343759031099784, -0.05946510701311203, -0.060210958568287676, -0.06066452861646829, -0.06087853711493113, -0.060878537114931144, -0.060664528616468316, -0.06021095856828771, -0.05946510701311207, -0.05834375903109984, -0.05672795555445498, -0.05445549669397575, -0.051310731519584395, -0.04701099960337077, -0.041188881609202366, -0.033369155910396506, -0.025116131040165093, -0.01687171797106464, -0.007098791725427836], [0.0781147684604678, 0.06921594211975576, 0.06155096446775292, 0.05284320917399808, 0.043414092618247996, 0.03558953520026994, 0.029036469778367012, 0.023487756799620556, 0.01872735588992392, 0.014578582929671596, 0.0108947367708732, 0.007551526057898897, 0.004440837791539046, 0.0014654721585467823, -0.001465472158546699, -0.004440837791538976, -0.007551526057898814, -0.010894736770873187, -0.014578582929671638, -0.018727355889923955, -0.023487756799620563, -0.02903646977836702, -0.03558953520026989, -0.043414092618247954, -0.05284320917399807, -0.06155096446775295, -0.06921594211975582, -0.07811476846046791]], [[1.2214802332508068, 1.3732286030067224, 1.5029807674106739, 1.6145766476980894, 1.7069071589703357, 1.782878787228597, 1.845998423616293, 1.8985067334074022, 1.9418520981267107, 1.9769815098994288, 2.0045192154960283, 2.024876299851067, 2.0383174865056555, 2.0450010218879617, 2.0450010218879617, 2.038317486505656, 2.024876299851068, 2.004519215496029, 1.9769815098994301, 1.9418520981267122, 1.898506733407404, 1.8459984236162945, 1.7828787872285985, 1.7069071589703375, 1.6145766476980907, 1.502980767410675, 1.3732286030067233, 1.2214802332508077], [-0.5486674588771329, -0.5178567322876322, -0.4870263124021468, -0.44969294990432107, -0.4060222663292231, -0.3626349017854431, -0.3194807342599886, -0.27652018202694534, -0.23372021766783094, -0.19105192026819415, -0.14848896108322612, -0.10600665434434137, -0.06358134843606449, -0.021190019415994868, 0.021190019415994066, 0.06358134843606372, 0.10600665434434042, 0.14848896108322548, 0.19105192026819354, 0.23372021766783063, 0.2765201820269449, 0.31948073425998824, 0.36263490178544266, 0.4060222663292229, 0.4496929499043208, 0.4870263124021469, 0.5178567322876325, 0.5486674588771334], [-0.045095244894249924, -0.09083661540673255, -0.12732285831035817, -0.15735648013520642, -0.180677168275448, -0.19814280878034674, -0.21149297006958911, -0.22183651838448287, -0.2298881288442725, -0.2361134459423319, -0.24081822557674804, -0.24420301310962284, -0.24639647177849827, -0.24747528029677715, -0.24747528029677723, -0.24639647177849838, -0.24420301310962297, -0.2408182255767482, -0.23611344594233216, -0.22988812884427282, -0.22183651838448318, -0.2114929700695894, -0.19814280878034704, -0.1806771682754483, -0.1573564801352068, -0.1273228583103585, -0.09083661540673282, -0.04509524489425007], [0.16280726409180385, 0.14774656896241772, 0.1359778495146811, 0.12219729428662576, 0.10593499896485281, 0.09141182495877667, 0.07821556372007987, 0.06604919096984832, 0.054689304514015234, 0.043960072717659116, 0.03371669811575426, 0.023834716810508902, 0.01420286555996008, 0.004718104438157129, -0.004718104438156803, -0.01420286555995981, -0.023834716810508534, -0.03371669811575408, -0.043960072717659004, -0.05468930451401511, -0.06604919096984815, -0.07821556372007976, -0.09141182495877667, -0.10593499896485273, -0.12219729428662574, -0.1359778495146811, -0.14774656896241778, -0.16280726409180402], [-0.004875409579200303, 0.01573118158636161, 0.0305791169421098, 0.041842788646241996, 0.049507892944767234, 0.05410669075622939, 0.05681218018321313, 0.058354779455742745, 0.059191609811622165, 0.05961016533836615, 0.059791795789504706, 0.05985050997852803, 0.059856532224489875, 0.05985030551083388, 0.059850305510833875, 0.059856532224489896, 0.05985050997852806, 0.059791795789504804, 0.05961016533836626, 0.05919160981162232, 0.058354779455742925, 0.05681218018321321, 0.05410669075622953, 0.0495078929447674, 0.04184278864624212, 0.030579116942109914, 0.01573118158636165, -0.004875409579200296], [-0.0700579831255859, -0.05851430633957662, -0.05124847482168606, -0.04353958308519819, -0.034670552854779664, -0.027748929979118868, -0.022220356577399826, -0.017708754187513887, -0.013950690637646618, -0.010754499905124601, -0.007974574195270097, -0.005494951313376441, -0.0032185771504072286, -0.0010599918761467278, 0.001059991876146707, 0.00321857715040727, 0.0054949513133764685, 0.007974574195270145, 0.01075449990512467, 0.013950690637646673, 0.017708754187513935, 0.022220356577399798, 0.02774892997911884, 0.034670552854779636, 0.043539583085198175, 0.05124847482168603, 0.05851430633957661, 0.0700579831255859], [0.005450864354276734, -0.00572124228131158, -0.012972352119864566, -0.017942680717792345, -0.020693568775194265, -0.021623387981841276, -0.02155433781717929, -0.020983056129773053, -0.020208163825997966, -0.019408083469874188, -0.01868846417193156, -0.018110998367212652, -0.017710794956672563, -0.017506630556303623, -0.01750663055630365, -0.017710794956672556, -0.018110998367212652, -0.018688464171931603, -0.019408083469874243, -0.020208163825998042, -0.020983056129773102, -0.021554337817179373, -0.021623387981841366, -0.020693568775194418, -0.0179426807177925, -0.01297235211986468, -0.005721242281311691, 0.0054508643542766425], [0.03824504123571494, 0.028363645961749933, 0.02303510891269477, 0.017899609842738945, 0.01212373812650698, 0.008191001882634434, 0.005492940038089707, 0.0036362482688489864, 0.0023617698541363308, 0.0014944121425625526, 0.0009119887561440457, 0.0005256335399601852, 0.0002672625191109529, 8.127638914905838e-05, -8.127638914911389e-05, -0.0002672625191110223, -0.0005256335399602199, -0.0009119887561440942, -0.001494412142562622, -0.0023617698541363585, -0.0036362482688489864, -0.005492940038089748, -0.008191001882634455, -0.012123738126507078, -0.017899609842739042, -0.023035108912694863, -0.02836364596175002, -0.03824504123571507]]]

        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_legendre_order + 1, -1)

        # Solve the problem
        pydgm.solver.solve()

        for l in range(pydgm.control.scatter_legendre_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()

            test = phi_test[:, l].flatten()

            np.testing.assert_array_almost_equal(phi, test, 12)

    def test_solver_anisotropic_symmetric(self):
        # Set the variables for the test
        pydgm.control.fine_mesh_x = [12]
        pydgm.control.coarse_mesh_x = [0.0, 5.0]
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
        pydgm.control.coarse_mesh_x = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.scatter_legendre_order = 7
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        # set the test flux
        phi_test = [[[30.761191851691446, 30.759719924860693, 30.756744954639235, 30.751806043767765, 30.74625260146888, 30.741760978230626, 30.73813458430471, 30.735220722001802, 30.73290136586005, 30.73108604905141, 30.729706421979046, 30.728712150048292, 30.728067898992542, 30.727751222545916, 30.727751222545045, 30.728067898989927, 30.728712150043926, 30.7297064219729, 30.731086049043434, 30.73290136585019, 30.73522072198996, 30.738134584290773, 30.741760978214444, 30.746252601450294, 30.75180604374656, 30.756744954615645, 30.759719924835, 30.76119185166351], [0.0022697997032592454, 0.006809830073580381, 0.011351139536491917, 0.012990832563841592, 0.011732180376967494, 0.010481016154806067, 0.009235960301030688, 0.007995855972959875, 0.006759728870959436, 0.005526751112812478, 0.004296209719771404, 0.0030674794631515012, 0.0018399994430786104, 0.0006132526101852998, -0.0006132526111014003, -0.0018399994440538303, -0.003067479464224865, -0.004296209720974942, -0.0055267511141792736, -0.006759728872532067, -0.007995855974780641, -0.009235960303148993, -0.01048101615728153, -0.011732180379867674, -0.012990832567245425, -0.011351139540390964, -0.006809830077953161, -0.0022697997081666532], [-0.009846731728859526, -0.009167683658181547, -0.00779403998706929, -0.0055459830508499675, -0.0030551953074158256, -0.0010662927913507314, 0.000518863365048583, 0.0017764949679732744, 0.0027655121188572274, 0.0035310662829906025, 0.004107268013558318, 0.004519235748807526, 0.004784601405486644, 0.004914565300024432, 0.0049145653004571965, 0.004784601406788047, 0.004519235750980233, 0.004107268016618981, 0.0035310662869632026, 0.00276551212377929, 0.001776494973890319, 0.000518863372022782, -0.001066292783244549, -0.0030551952980857333, -0.005545983040186497, -0.007794039975194567, -0.009167683645252778, -0.009846731714805212], [-0.0013984089902102093, -0.004203415097014107, -0.007033150101170393, -0.007934677495059106, -0.006939386746524556, -0.006030362579446391, -0.005190271895189724, -0.004405082580074882, -0.003663330064153869, -0.0029555324260026605, -0.0022737210463770197, -0.0016110598895150985, -0.0009615313220033284, -0.00031967010118111316, 0.00031967009053063267, 0.0009615313113631174, 0.001611059878884269, 0.0022737210357494653, 0.002955532415370221, 0.0036633300535140467, 0.004405082569424845, 0.005190271884531084, 0.006030362568781866, 0.006939386735862529, 0.007934677484410069, 0.007033150090558271, 0.004203415086469375, 0.0013984089797519084], [0.00590085702895049, 0.005477796861363626, 0.004620520136698447, 0.0032220813079857358, 0.001690196698391322, 0.0004916910249364026, -0.00044353636559590903, -0.0011698556038717278, -0.0017292283283965038, -0.0021537700304845586, -0.002467709559390241, -0.002688866050494565, -0.0028297341004706045, -0.0028982440182981906, -0.002898244018384122, -0.0028297341007301746, -0.002688866050933103, -0.002467709560018738, -0.0021537700313214447, -0.0017292283294687572, -0.0011698556052142095, -0.00044353636725369405, 0.000491691022907137, 0.0016901966959216308, 0.0032220813049925745, 0.004620520133198469, 0.005477796857395356, 0.005900857024459638], [0.0010212914039941512, 0.003077410263137237, 0.005174453056863704, 0.005741801160114746, 0.004827285289393746, 0.004047509871701049, 0.0033742506011232853, 0.002784917081602778, 0.002261290163689922, 0.0017885191438957726, 0.001354319884291244, 0.0009483263395589914, 0.0005615568635398693, 0.0001859634031043944, -0.0001859634022277623, -0.0005615568626634038, -0.0009483263386739771, -0.0013543198833827486, -0.001788519142949474, -0.002261290162691554, -0.00278491708053874, -0.0033742505999792005, -0.004047509870462707, -0.004827285288046934, -0.00574180115864481, -0.0051744530552919055, -0.0030774102614932186, -0.0010212914022775799], [-0.003882712199050342, -0.0035974283747575164, -0.0030184588756753117, -0.0020767464423055992, -0.0010555432745139592, -0.0002712217899963898, 0.0003288076075016466, 0.0007852528819602966, 0.0011294672398021177, 0.001385397566575186, 0.0015710740386130073, 0.0016997325927502471, 0.0017806394759611877, 0.001819668798672347, 0.0018196687987235283, 0.0017806394761149535, 0.0016997325930108165, 0.0015710740389882627, 0.0013853975670795604, 0.001129467240455817, 0.0007852528827895222, 0.0003288076085404823, -0.00027122178870542246, -0.0010555432729179026, -0.0020767464403407265, -0.0030184588733489504, -0.003597428372093092, -0.0038827121960061106], [-0.0008116856673593409, -0.0024513412666662404, -0.0041402589428698455, -0.004526624272935487, -0.003666720521891098, -0.0029661348417961975, -0.002390451927915649, -0.0019123500162734342, -0.0015100032014355724, -0.0011658150905656806, -0.0008654081076832076, -0.0005968076821920576, -0.00034977207366093666, -0.00011522727271495725, 0.00011522727140078626, 0.00034977207234165864, 0.0005968076808545719, 0.0008654081063103058, 0.0011658150891398211, 0.0015100031999381036, 0.0019123500146831507, 0.0023904519262114565, 0.0029661348399547816, 0.0036667205198874786, 0.004526624270740687, 0.004140258940502517, 0.0024513412641566923, 0.0008116856646941395]], [[22.043457229072096, 22.01602384057805, 21.957185466861898, 21.87895465271431, 21.809851500663328, 21.759584672753707, 21.722056521913203, 21.69347835822571, 21.671484582479895, 21.654590894864175, 21.641863079615376, 21.63271496237606, 21.62678604129403, 21.623868923224904, 21.623868923225235, 21.626786041295006, 21.632714962377662, 21.64186307961754, 21.654590894866836, 21.671484582482922, 21.69347835822892, 21.72205652191632, 21.759584672756297, 21.809851500664752, 21.87895465271354, 21.957185466857315, 22.01602384056705, 22.04345722905058], [0.02446302454386312, 0.07339242330080217, 0.12233243078442418, 0.13976725490589476, 0.12583235070262444, 0.11214376128354264, 0.0986359905532195, 0.08526416885793944, 0.07199624159805665, 0.05880815244526075, 0.045680856543725834, 0.03259845063959005, 0.0195469820872044, 0.006513665380510486, -0.0065136653788192, -0.019546982085465153, -0.03259845063775205, -0.04568085654173637, -0.058808152443062534, -0.07199624159559609, -0.08526416885515012, -0.098635990550035, -0.11214376127988696, -0.12583235069840162, -0.1397672549010319, -0.12233243077880326, -0.07339242329428938, -0.0244630245364659], [-0.10473431107916853, -0.0937998675802213, -0.06994557900992138, -0.03807634420645223, -0.010685058978144202, 0.00803278145452202, 0.021114478553565963, 0.030445832618655855, 0.03720317374048743, 0.04212369591738829, 0.04567071374776166, 0.04813446380418063, 0.049693151716835304, 0.050449149972276675, 0.0504491499720795, 0.04969315171624866, 0.048134463803216176, 0.04567071374644116, 0.04212369591574516, 0.03720317373857818, 0.030445832616564306, 0.021114478551419347, 0.008032781452521065, -0.010685058979685969, -0.038076344207044754, -0.06994557900879328, -0.09379986757610759, -0.10473431107004605], [-0.012103233417161186, -0.036750966782180594, -0.06280258253308563, -0.07017217883915483, -0.05903169554269344, -0.04980513228462152, -0.04190584073298104, -0.03494772623490122, -0.028669428637315364, -0.022887528727887962, -0.017467502319431782, -0.012305514060192424, -0.007316804917043507, -0.002428041169053319, 0.002428041175926432, 0.007316804923909681, 0.01230551406704411, 0.017467502326257156, 0.02288752873467359, 0.028669428644043315, 0.0349477262415443, 0.04190584073950071, 0.04980513229096373, 0.05903169554877785, 0.07017217884488097, 0.0628025825382853, 0.03675096678658518, 0.012103233420478754], [0.05511963945991105, 0.049068549016530794, 0.03555849525475285, 0.01742902420311021, 0.0024258000326546902, -0.006945619526173896, -0.012824256762401176, -0.01652720881077141, -0.018867226613651145, -0.020346777228088864, -0.021276549064350614, -0.021847641209509994, -0.022175215303099738, -0.022324336255886812, -0.022324336256020483, -0.02217521530350275, -0.021847641210190782, -0.021276549065322503, -0.020346777229376167, -0.018867226615291166, -0.01652720881282055, -0.01282425676494825, -0.0069456195293562395, 0.0024258000286184744, 0.017429024197879173, 0.03555849524777277, 0.049068549006895834, 0.055119639446185364], [0.007127174492710275, 0.022015067876339944, 0.03893292850029434, 0.04239182935779695, 0.032581432921953146, 0.025404400905523428, 0.01997882494047981, 0.01573950980639449, 0.0123173512843619, 0.009465067840074393, 0.007011262899538961, 0.004831781041847172, 0.0028315768480500214, 0.0009328950601668051, -0.0009328950571452777, -0.0028315768449993506, -0.004831781038736049, -0.007011262896333581, -0.00946506783673673, -0.01231735128084771, -0.015739509802647766, -0.019978824936430162, -0.02540440090107765, -0.03258143291698046, -0.0423918293521256, -0.038932928493641994, -0.02201506786827817, -0.007127174482688681], [-0.031788396036350086, -0.028106153502096998, -0.019692773562300037, -0.008372872866156733, 0.0006445863845488953, 0.005758553074859507, 0.00854771412555555, 0.009973088343101622, 0.010618837582398788, 0.010839206940689627, 0.010848088565808878, 0.010773455830556666, 0.010690190660801235, 0.010639460523014344, 0.010639460522986033, 0.01069019066071808, 0.010773455830420997, 0.010848088565625358, 0.010839206940467805, 0.010618837582154428, 0.009973088342864034, 0.008547714125372918, 0.005758553074815098, 0.0006445863847825972, -0.008372872865417103, -0.019692773560672894, -0.028106153498941855, -0.03178839603062655], [-0.0045802311050726074, -0.014460817463747055, -0.02666307683067698, -0.028284940666706893, -0.019510676470427646, -0.013661666032456776, -0.009687533474548038, -0.006929441145179627, -0.004969680894637396, -0.0035393290075922956, -0.002461365746782296, -0.0016154260717305968, -0.0009156866804730379, -0.0002966308153283803, 0.00029663081482628195, 0.0009156866799511776, 0.0016154260711671586, 0.002461365746152855, 0.0035393290068685967, 0.004969680893783857, 0.006929441144149506, 0.009687533473278331, 0.01366166603085861, 0.019510676468368238, 0.028284940664003333, 0.026663076827019017, 0.014460817458647857, 0.004580231097861542]]]

        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_legendre_order + 1, -1)

        # Solve the problem
        pydgm.solver.solve()

        for l in range(pydgm.control.scatter_legendre_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()

            test = phi_test[:, l].flatten()

            np.testing.assert_array_almost_equal(phi, test, 9)

    def test_solver_partisn_eigen_2g_l0(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh_x = [10, 4]
        pydgm.control.coarse_mesh_x = [0.0, 1.5, 2.0]
        pydgm.control.material_map = [1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0

        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
        pydgm.control.scatter_legendre_order = 0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux
        phi_test = [[3.442525765957952, 3.44133409525864, 3.438914799438404, 3.435188915015736, 3.4300162013446567, 3.423154461297391, 3.4141703934131375, 3.4022425049312393, 3.3857186894563807, 3.3610957244311783, 3.332547468621578, 3.310828206454775, 3.2983691806875637, 3.292637618967479], [1.038826864565325, 1.0405199414437678, 1.0439340321802706, 1.0491279510472575, 1.0561979892073443, 1.065290252825513, 1.0766260147603826, 1.0905777908540444, 1.1080393189273399, 1.1327704173615665, 1.166565957603695, 1.195247115307628, 1.2108105235380155, 1.2184043154658892]]

        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_legendre_order + 1, -1)  # Group, legendre, cell

        keff_test = 1.17455939

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        phi_zero = pydgm.state.mg_phi[0, :, :].flatten()

        phi_zero_test = phi_test[:, 0].flatten() / np.linalg.norm(phi_test[:, 0]) * np.linalg.norm(phi_zero)

        np.testing.assert_array_almost_equal(phi_zero, phi_zero_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_partisn_eigen_2g_l7(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.fine_mesh_x = [10, 4]
        pydgm.control.coarse_mesh_x = [0.0, 1.5, 2.0]
        pydgm.control.material_map = [1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
        pydgm.control.scatter_legendre_order = 7

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[3.4347602852260457, 3.4338222458022396, 3.431913102564298, 3.4289595149478354, 3.42482931141838, 3.4192906043709104, 3.4119234234055424, 3.4019274881836847, 3.3876982772674196, 3.3658681618062154, 3.34070577019797, 3.322084670360727, 3.3116282200682177, 3.306887464734553], [0.0020425079990794015, 0.006141055824933565, 0.010280477283313556, 0.014488988641503553, 0.018796411845326454, 0.023235192019730004, 0.02784201432654461, 0.03266091540339621, 0.03775162510819405, 0.04322363524982223, 0.04027802436530039, 0.028720242100487507, 0.01721423055811079, 0.00573526224597129], [-0.014043175376659278, -0.013864638454379852, -0.013492963463993146, -0.012895383132327465, -0.012011774689980326, -0.010734367418404306, -0.008863802887779854, -0.006013432809048869, -0.0013981713918363725, 0.006642614571652411, 0.016624103448331484, 0.02410947535371022, 0.028122618919115827, 0.02988592542244567], [-0.0005815260582084716, -0.001763517031618471, -0.0030033211424644393, -0.004343027083168201, -0.00583095424501387, -0.007526497566595659, -0.009508940019425233, -0.011895018635082327, -0.014877024186556766, -0.0188162961923218, -0.01803220412213285, -0.012313418072773766, -0.007189899282343329, -0.0023660944740765757], [0.004680471538592718, 0.004708687275173026, 0.004759474154300947, 0.004819160392550578, 0.004859808008457478, 0.004825189996392545, 0.00459997623316151, 0.0039419212948576254, 0.0023313167418493597, -0.0013699293198698725, -0.006397633005789841, -0.010063641940086188, -0.01182150118349333, -0.012530543653381895], [0.00012077154565669213, 0.00037538908680516403, 0.0006704426262086824, 0.0010372867699327687, 0.0015153228292910034, 0.0021590064366188433, 0.003051328935256406, 0.004331559811147134, 0.006255486169653435, 0.009336140299455876, 0.009249849164873904, 0.005807390813692741, 0.003214319791852313, 0.0010307478448935911], [-0.0014951120924339978, -0.0015552615892510946, -0.0016748080008523458, -0.0018506450636592287, -0.0020730026909920873, -0.002315467093667617, -0.0025125211079179005, -0.0025092967947779043, -0.0019488784911843998, -1.5802018757685166e-05, 0.002852214264202485, 0.004877857988651882, 0.005725983991531153, 0.006026644208786527], [7.3756351610665405e-06, 1.5381594938093657e-05, 2.0563907207754584e-06, -5.089466446328572e-05, -0.00016961848279553604, -0.00039584947236004364, -0.0008025998344126095, -0.0015274663776118497, -0.0028458777322057927, -0.005340715351454056, -0.005460496519064682, -0.0030380827793034665, -0.0015401075725997374, -0.00047164106473560063]], [[1.0468988965851684, 1.0482039066792, 1.0508374761295303, 1.0548495077428226, 1.0603225626298973, 1.0673837551321859, 1.076229817119911, 1.0872041099591274, 1.1011637581627296, 1.1216888353486294, 1.1506561958511974, 1.1750591235939027, 1.1877839892260984, 1.1939512173711346], [-0.0018037231330366964, -0.005420899455129392, -0.009067464657264646, -0.012763702055788537, -0.016531040703842596, -0.02039277877911453, -0.02437522972242518, -0.0285099264728463, -0.0328395627942114, -0.03744247111536142, -0.03479669107222463, -0.024785585686068916, -0.01484656658040623, -0.004944915099087104], [0.009044222884081916, 0.008994080581571237, 0.008885291188975845, 0.00869951180576372, 0.008405557021954946, 0.007953703488126046, 0.007263566096093681, 0.006186280429767779, 0.004322305340770813, -7.92578902843108e-05, -0.007284529509716177, -0.012369785027157056, -0.013998722067242592, -0.014718517386448186], [0.00015605671357175194, 0.0004804431832614083, 0.0008427650570911344, 0.0012722064721627357, 0.0018044434542845312, 0.0024857529642005084, 0.0033794599712204353, 0.004578858830085896, 0.006248087239271316, 0.008823984631500687, 0.008282352937899702, 0.004884977468543283, 0.002655682997061523, 0.000842624820554333], [-0.0011062410074079362, -0.0011510824499980574, -0.001240932816833476, -0.0013757329042850541, -0.0015542513826867221, -0.0017720452059471134, -0.002015975985084427, -0.002241982547261802, -0.002251592717574219, -0.0009113132608139229, 0.001832014409231919, 0.003242312420659532, 0.003088201064543508, 0.0029899398270237243], [2.7084774482374202e-05, 7.76253875693772e-05, 0.00011641039362962971, 0.00013251436927942672, 0.00010964808141212395, 2.2273787440529186e-05, -0.00017118062055123187, -0.0005440963026852562, -0.0012598891406956324, -0.0028838311878326556, -0.0026776945146877937, -0.000938560587728697, -0.000376077262636252, -9.762375501407017e-05], [3.0909390321964825e-06, 2.5576618008702617e-05, 7.287174344039124e-05, 0.00014981621482667087, 0.0002641373710909681, 0.0004268254149958281, 0.000651154056289311, 0.0009409516756199963, 0.0012049977784007737, 0.0006763751835193246, -0.0006792042741425169, -0.0011506446002332686, -0.0007662541935604429, -0.0005921092886468325], [-2.369039114824556e-05, -7.12100424579698e-05, -0.00011873728900866493, -0.0001647767050291203, -0.0002050505121492345, -0.0002300568574770713, -0.0002200535280689267, -0.00013049976915558564, 0.00017295202230512008, 0.00129308500597922, 0.0011546971495171136, 3.0432463254098652e-05, -6.792236810924945e-05, -3.477901183772045e-05]]]

        phi_test = np.array(phi_test)

        keff_test = 1.17564099

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_legendre_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()
            phi_zero_test = phi_test[:, l].flatten() / np.linalg.norm(phi_test[:, l]) * np.linalg.norm(phi)
            np.testing.assert_array_almost_equal(phi, phi_zero_test, 12)
        # np.testing.assert_array_almost_equal(phi_one, phi_one_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    def test_solver_1loop(self):

        def set_parameters():
            self.setUp()
            # define the nonstandard test variables
            pydgm.control.solver_type = 'eigen'.ljust(256)
            pydgm.control.source_value = 0.0
            pydgm.control.allow_fission = True
            pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
            pydgm.control.fine_mesh_x = [2, 1, 2]
            pydgm.control.coarse_mesh_x = [0.0, 5.0, 6.0, 11.0]
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
