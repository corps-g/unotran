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
        pydgm.control.scatter_leg_order = 0
        pydgm.control.use_DGM = False
        pydgm.control.max_eigen_iters = 10000
        pydgm.control.max_outer_iters = 1000

    def angular_test(self):
        '''
        Generic tester for the angular flux
        '''
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

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
        self.angular_test()

    def test_solver_anisotropic_fixed_vacuum(self):
        # Set the variables for the test
        pydgm.control.coarse_mesh_x = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.angle_order = 8
        pydgm.control.scatter_leg_order = 7

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        # set the test flux
        phi_test = [[[1.4029164436301702, 1.4802119704475423, 1.5495888971817153, 1.626815932451072, 1.7085644741288557, 1.7773357033976998, 1.8349818057576361, 1.8829471863949965, 1.9223530024630586, 1.9540623273187099, 1.9787299685145165, 1.9968400126457129, 2.008733411830428, 2.01462731175292, 2.01462731175292, 2.008733411830428, 1.9968400126457133, 1.978729968514517, 1.95406232731871, 1.922353002463059, 1.8829471863949974, 1.8349818057576368, 1.7773357033977002, 1.7085644741288561, 1.6268159324510723, 1.5495888971817156, 1.4802119704475425, 1.4029164436301704], [-0.5780989114905826, -0.5492884933838695, -0.520537582666649, -0.482959651628116, -0.4365932342866728, -0.39034964765757363, -0.34420923792593605, -0.2981550279181293, -0.2521721286908298, -0.20624726936044147, -0.16036841852669964, -0.11452447573844954, -0.06870501543022531, -0.022900068787467087, 0.022900068787466428, 0.06870501543022475, 0.11452447573844896, 0.16036841852669909, 0.20624726936044097, 0.25217212869082944, 0.298155027918129, 0.34420923792593566, 0.39034964765757324, 0.43659323428667257, 0.48295965162811566, 0.5205375826666488, 0.5492884933838693, 0.5780989114905825], [-0.15626492240297754, -0.1877693930174307, -0.2156792311760249, -0.2453879170735357, -0.27558726182945853, -0.30037303958839967, -0.3206687207708231, -0.33719431063729743, -0.35050856017019766, -0.36104150519823525, -0.3691193422376261, -0.3749831757292076, -0.3788027926419254, -0.3806863133601728, -0.3806863133601728, -0.3788027926419255, -0.3749831757292077, -0.36911934223762627, -0.36104150519823547, -0.3505085601701978, -0.33719431063729777, -0.3206687207708234, -0.3003730395883999, -0.27558726182945875, -0.2453879170735359, -0.21567923117602505, -0.1877693930174309, -0.15626492240297774], [0.23982918604971462, 0.22464749026295341, 0.21020037865558608, 0.19213205019607393, 0.17069065874065356, 0.1503305787366992, 0.13085870516614978, 0.11211358400144683, 0.09395839332823447, 0.07627536083165615, 0.058961285265606025, 0.04192389713220073, 0.02507884524433541, 0.008347134236776584, -0.008347134236776382, -0.0250788452443352, -0.04192389713220062, -0.058961285265605935, -0.07627536083165609, -0.09395839332823444, -0.11211358400144666, -0.13085870516614972, -0.15033057873669903, -0.1706906587406533, -0.19213205019607377, -0.21020037865558602, -0.2246474902629534, -0.23982918604971454], [0.03127032376366864, 0.04774343491417285, 0.0619769912505068, 0.07667377682595822, 0.09111515744575428, 0.10244202546348577, 0.11130243175898469, 0.11819984834749346, 0.12352355362024048, 0.1275720487039491, 0.13057095388100948, 0.13268649215310283, 0.1340353938018044, 0.13469183428532921, 0.13469183428532927, 0.1340353938018044, 0.13268649215310288, 0.13057095388100956, 0.12757204870394917, 0.12352355362024063, 0.11819984834749361, 0.11130243175898485, 0.10244202546348594, 0.09111515744575446, 0.07667377682595838, 0.06197699125050698, 0.04774343491417303, 0.03127032376366885], [-0.12593303200825925, -0.11509238928686001, -0.10534645626989476, -0.09376178850299745, -0.08064968274884363, -0.06902322779279649, -0.05859353076024055, -0.049124750961078205, -0.04042225332753143, -0.032323206361740915, -0.02468905804437354, -0.01739943874686583, -0.01034712805913645, -0.0034337879690522904, 0.0034337879690521586, 0.010347128059136304, 0.017399438746865767, 0.024689058044373477, 0.03232320636174092, 0.04042225332753144, 0.04912475096107815, 0.05859353076024054, 0.0690232277927964, 0.08064968274884354, 0.09376178850299738, 0.10534645626989475, 0.11509238928686001, 0.12593303200825928], [-0.007098791725427716, -0.016871717971064522, -0.02511613104016501, -0.03336915591039636, -0.041188881609202255, -0.04701099960337066, -0.051310731519584325, -0.054455496693975654, -0.056727955554454854, -0.058343759031099784, -0.05946510701311203, -0.060210958568287676, -0.06066452861646829, -0.06087853711493113, -0.060878537114931144, -0.060664528616468316, -0.06021095856828771, -0.05946510701311207, -0.05834375903109984, -0.05672795555445498, -0.05445549669397575, -0.051310731519584395, -0.04701099960337077, -0.041188881609202366, -0.033369155910396506, -0.025116131040165093, -0.01687171797106464, -0.007098791725427836], [0.0781147684604678, 0.06921594211975576, 0.06155096446775292, 0.05284320917399808, 0.043414092618247996, 0.03558953520026994, 0.029036469778367012, 0.023487756799620556, 0.01872735588992392, 0.014578582929671596, 0.0108947367708732, 0.007551526057898897, 0.004440837791539046, 0.0014654721585467823, -0.001465472158546699, -0.004440837791538976, -0.007551526057898814, -0.010894736770873187, -0.014578582929671638, -0.018727355889923955, -0.023487756799620563, -0.02903646977836702, -0.03558953520026989, -0.043414092618247954, -0.05284320917399807, -0.06155096446775295, -0.06921594211975582, -0.07811476846046791]], [[1.2214802332508068, 1.3732286030067224, 1.5029807674106739, 1.6145766476980894, 1.7069071589703357, 1.782878787228597, 1.845998423616293, 1.8985067334074022, 1.9418520981267107, 1.9769815098994288, 2.0045192154960283, 2.024876299851067, 2.0383174865056555, 2.0450010218879617, 2.0450010218879617, 2.038317486505656, 2.024876299851068, 2.004519215496029, 1.9769815098994301, 1.9418520981267122, 1.898506733407404, 1.8459984236162945, 1.7828787872285985, 1.7069071589703375, 1.6145766476980907, 1.502980767410675, 1.3732286030067233, 1.2214802332508077], [-0.5486674588771329, -0.5178567322876322, -0.4870263124021468, -0.44969294990432107, -0.4060222663292231, -0.3626349017854431, -0.3194807342599886, -0.27652018202694534, -0.23372021766783094, -0.19105192026819415, -0.14848896108322612, -0.10600665434434137, -0.06358134843606449, -0.021190019415994868, 0.021190019415994066, 0.06358134843606372, 0.10600665434434042, 0.14848896108322548, 0.19105192026819354, 0.23372021766783063, 0.2765201820269449, 0.31948073425998824, 0.36263490178544266, 0.4060222663292229, 0.4496929499043208, 0.4870263124021469, 0.5178567322876325, 0.5486674588771334], [-0.045095244894249924, -0.09083661540673255, -0.12732285831035817, -0.15735648013520642, -0.180677168275448, -0.19814280878034674, -0.21149297006958911, -0.22183651838448287, -0.2298881288442725, -0.2361134459423319, -0.24081822557674804, -0.24420301310962284, -0.24639647177849827, -0.24747528029677715, -0.24747528029677723, -0.24639647177849838, -0.24420301310962297, -0.2408182255767482, -0.23611344594233216, -0.22988812884427282, -0.22183651838448318, -0.2114929700695894, -0.19814280878034704, -0.1806771682754483, -0.1573564801352068, -0.1273228583103585, -0.09083661540673282, -0.04509524489425007], [0.16280726409180385, 0.14774656896241772, 0.1359778495146811, 0.12219729428662576, 0.10593499896485281, 0.09141182495877667, 0.07821556372007987, 0.06604919096984832, 0.054689304514015234, 0.043960072717659116, 0.03371669811575426, 0.023834716810508902, 0.01420286555996008, 0.004718104438157129, -0.004718104438156803, -0.01420286555995981, -0.023834716810508534, -0.03371669811575408, -0.043960072717659004, -0.05468930451401511, -0.06604919096984815, -0.07821556372007976, -0.09141182495877667, -0.10593499896485273, -0.12219729428662574, -0.1359778495146811, -0.14774656896241778, -0.16280726409180402], [-0.004875409579200303, 0.01573118158636161, 0.0305791169421098, 0.041842788646241996, 0.049507892944767234, 0.05410669075622939, 0.05681218018321313, 0.058354779455742745, 0.059191609811622165, 0.05961016533836615, 0.059791795789504706, 0.05985050997852803, 0.059856532224489875, 0.05985030551083388, 0.059850305510833875, 0.059856532224489896, 0.05985050997852806, 0.059791795789504804, 0.05961016533836626, 0.05919160981162232, 0.058354779455742925, 0.05681218018321321, 0.05410669075622953, 0.0495078929447674, 0.04184278864624212, 0.030579116942109914, 0.01573118158636165, -0.004875409579200296], [-0.0700579831255859, -0.05851430633957662, -0.05124847482168606, -0.04353958308519819, -0.034670552854779664, -0.027748929979118868, -0.022220356577399826, -0.017708754187513887, -0.013950690637646618, -0.010754499905124601, -0.007974574195270097, -0.005494951313376441, -0.0032185771504072286, -0.0010599918761467278, 0.001059991876146707, 0.00321857715040727, 0.0054949513133764685, 0.007974574195270145, 0.01075449990512467, 0.013950690637646673, 0.017708754187513935, 0.022220356577399798, 0.02774892997911884, 0.034670552854779636, 0.043539583085198175, 0.05124847482168603, 0.05851430633957661, 0.0700579831255859], [0.005450864354276734, -0.00572124228131158, -0.012972352119864566, -0.017942680717792345, -0.020693568775194265, -0.021623387981841276, -0.02155433781717929, -0.020983056129773053, -0.020208163825997966, -0.019408083469874188, -0.01868846417193156, -0.018110998367212652, -0.017710794956672563, -0.017506630556303623, -0.01750663055630365, -0.017710794956672556, -0.018110998367212652, -0.018688464171931603, -0.019408083469874243, -0.020208163825998042, -0.020983056129773102, -0.021554337817179373, -0.021623387981841366, -0.020693568775194418, -0.0179426807177925, -0.01297235211986468, -0.005721242281311691, 0.0054508643542766425], [0.03824504123571494, 0.028363645961749933, 0.02303510891269477, 0.017899609842738945, 0.01212373812650698, 0.008191001882634434, 0.005492940038089707, 0.0036362482688489864, 0.0023617698541363308, 0.0014944121425625526, 0.0009119887561440457, 0.0005256335399601852, 0.0002672625191109529, 8.127638914905838e-05, -8.127638914911389e-05, -0.0002672625191110223, -0.0005256335399602199, -0.0009119887561440942, -0.001494412142562622, -0.0023617698541363585, -0.0036362482688489864, -0.005492940038089748, -0.008191001882634455, -0.012123738126507078, -0.017899609842739042, -0.023035108912694863, -0.02836364596175002, -0.03824504123571507]]]

        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_leg_order + 1, -1)

        # Solve the problem
        pydgm.solver.solve()

        for l in range(pydgm.control.scatter_leg_order + 1):
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
        pydgm.control.scatter_leg_order = 1
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
        pydgm.control.scatter_leg_order = 7
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        # set the test flux
        phi_test = [[[30.761191851691446, 30.759719924860693, 30.756744954639235, 30.751806043767765, 30.74625260146888, 30.741760978230626, 30.73813458430471, 30.735220722001802, 30.73290136586005, 30.73108604905141, 30.729706421979046, 30.728712150048292, 30.728067898992542, 30.727751222545916, 30.727751222545045, 30.728067898989927, 30.728712150043926, 30.7297064219729, 30.731086049043434, 30.73290136585019, 30.73522072198996, 30.738134584290773, 30.741760978214444, 30.746252601450294, 30.75180604374656, 30.756744954615645, 30.759719924835, 30.76119185166351], [0.0022697997032592454, 0.006809830073580381, 0.011351139536491917, 0.012990832563841592, 0.011732180376967494, 0.010481016154806067, 0.009235960301030688, 0.007995855972959875, 0.006759728870959436, 0.005526751112812478, 0.004296209719771404, 0.0030674794631515012, 0.0018399994430786104, 0.0006132526101852998, -0.0006132526111014003, -0.0018399994440538303, -0.003067479464224865, -0.004296209720974942, -0.0055267511141792736, -0.006759728872532067, -0.007995855974780641, -0.009235960303148993, -0.01048101615728153, -0.011732180379867674, -0.012990832567245425, -0.011351139540390964, -0.006809830077953161, -0.0022697997081666532], [-0.009846731728859526, -0.009167683658181547, -0.00779403998706929, -0.0055459830508499675, -0.0030551953074158256, -0.0010662927913507314, 0.000518863365048583, 0.0017764949679732744, 0.0027655121188572274, 0.0035310662829906025, 0.004107268013558318, 0.004519235748807526, 0.004784601405486644, 0.004914565300024432, 0.0049145653004571965, 0.004784601406788047, 0.004519235750980233, 0.004107268016618981, 0.0035310662869632026, 0.00276551212377929, 0.001776494973890319, 0.000518863372022782, -0.001066292783244549, -0.0030551952980857333, -0.005545983040186497, -0.007794039975194567, -0.009167683645252778, -0.009846731714805212], [-0.0013984089902102093, -0.004203415097014107, -0.007033150101170393, -0.007934677495059106, -0.006939386746524556, -0.006030362579446391, -0.005190271895189724, -0.004405082580074882, -0.003663330064153869, -0.0029555324260026605, -0.0022737210463770197, -0.0016110598895150985, -0.0009615313220033284, -0.00031967010118111316, 0.00031967009053063267, 0.0009615313113631174, 0.001611059878884269, 0.0022737210357494653, 0.002955532415370221, 0.0036633300535140467, 0.004405082569424845, 0.005190271884531084, 0.006030362568781866, 0.006939386735862529, 0.007934677484410069, 0.007033150090558271, 0.004203415086469375, 0.0013984089797519084], [0.00590085702895049, 0.005477796861363626, 0.004620520136698447, 0.0032220813079857358, 0.001690196698391322, 0.0004916910249364026, -0.00044353636559590903, -0.0011698556038717278, -0.0017292283283965038, -0.0021537700304845586, -0.002467709559390241, -0.002688866050494565, -0.0028297341004706045, -0.0028982440182981906, -0.002898244018384122, -0.0028297341007301746, -0.002688866050933103, -0.002467709560018738, -0.0021537700313214447, -0.0017292283294687572, -0.0011698556052142095, -0.00044353636725369405, 0.000491691022907137, 0.0016901966959216308, 0.0032220813049925745, 0.004620520133198469, 0.005477796857395356, 0.005900857024459638], [0.0010212914039941512, 0.003077410263137237, 0.005174453056863704, 0.005741801160114746, 0.004827285289393746, 0.004047509871701049, 0.0033742506011232853, 0.002784917081602778, 0.002261290163689922, 0.0017885191438957726, 0.001354319884291244, 0.0009483263395589914, 0.0005615568635398693, 0.0001859634031043944, -0.0001859634022277623, -0.0005615568626634038, -0.0009483263386739771, -0.0013543198833827486, -0.001788519142949474, -0.002261290162691554, -0.00278491708053874, -0.0033742505999792005, -0.004047509870462707, -0.004827285288046934, -0.00574180115864481, -0.0051744530552919055, -0.0030774102614932186, -0.0010212914022775799], [-0.003882712199050342, -0.0035974283747575164, -0.0030184588756753117, -0.0020767464423055992, -0.0010555432745139592, -0.0002712217899963898, 0.0003288076075016466, 0.0007852528819602966, 0.0011294672398021177, 0.001385397566575186, 0.0015710740386130073, 0.0016997325927502471, 0.0017806394759611877, 0.001819668798672347, 0.0018196687987235283, 0.0017806394761149535, 0.0016997325930108165, 0.0015710740389882627, 0.0013853975670795604, 0.001129467240455817, 0.0007852528827895222, 0.0003288076085404823, -0.00027122178870542246, -0.0010555432729179026, -0.0020767464403407265, -0.0030184588733489504, -0.003597428372093092, -0.0038827121960061106], [-0.0008116856673593409, -0.0024513412666662404, -0.0041402589428698455, -0.004526624272935487, -0.003666720521891098, -0.0029661348417961975, -0.002390451927915649, -0.0019123500162734342, -0.0015100032014355724, -0.0011658150905656806, -0.0008654081076832076, -0.0005968076821920576, -0.00034977207366093666, -0.00011522727271495725, 0.00011522727140078626, 0.00034977207234165864, 0.0005968076808545719, 0.0008654081063103058, 0.0011658150891398211, 0.0015100031999381036, 0.0019123500146831507, 0.0023904519262114565, 0.0029661348399547816, 0.0036667205198874786, 0.004526624270740687, 0.004140258940502517, 0.0024513412641566923, 0.0008116856646941395]], [[22.043457229072096, 22.01602384057805, 21.957185466861898, 21.87895465271431, 21.809851500663328, 21.759584672753707, 21.722056521913203, 21.69347835822571, 21.671484582479895, 21.654590894864175, 21.641863079615376, 21.63271496237606, 21.62678604129403, 21.623868923224904, 21.623868923225235, 21.626786041295006, 21.632714962377662, 21.64186307961754, 21.654590894866836, 21.671484582482922, 21.69347835822892, 21.72205652191632, 21.759584672756297, 21.809851500664752, 21.87895465271354, 21.957185466857315, 22.01602384056705, 22.04345722905058], [0.02446302454386312, 0.07339242330080217, 0.12233243078442418, 0.13976725490589476, 0.12583235070262444, 0.11214376128354264, 0.0986359905532195, 0.08526416885793944, 0.07199624159805665, 0.05880815244526075, 0.045680856543725834, 0.03259845063959005, 0.0195469820872044, 0.006513665380510486, -0.0065136653788192, -0.019546982085465153, -0.03259845063775205, -0.04568085654173637, -0.058808152443062534, -0.07199624159559609, -0.08526416885515012, -0.098635990550035, -0.11214376127988696, -0.12583235069840162, -0.1397672549010319, -0.12233243077880326, -0.07339242329428938, -0.0244630245364659], [-0.10473431107916853, -0.0937998675802213, -0.06994557900992138, -0.03807634420645223, -0.010685058978144202, 0.00803278145452202, 0.021114478553565963, 0.030445832618655855, 0.03720317374048743, 0.04212369591738829, 0.04567071374776166, 0.04813446380418063, 0.049693151716835304, 0.050449149972276675, 0.0504491499720795, 0.04969315171624866, 0.048134463803216176, 0.04567071374644116, 0.04212369591574516, 0.03720317373857818, 0.030445832616564306, 0.021114478551419347, 0.008032781452521065, -0.010685058979685969, -0.038076344207044754, -0.06994557900879328, -0.09379986757610759, -0.10473431107004605], [-0.012103233417161186, -0.036750966782180594, -0.06280258253308563, -0.07017217883915483, -0.05903169554269344, -0.04980513228462152, -0.04190584073298104, -0.03494772623490122, -0.028669428637315364, -0.022887528727887962, -0.017467502319431782, -0.012305514060192424, -0.007316804917043507, -0.002428041169053319, 0.002428041175926432, 0.007316804923909681, 0.01230551406704411, 0.017467502326257156, 0.02288752873467359, 0.028669428644043315, 0.0349477262415443, 0.04190584073950071, 0.04980513229096373, 0.05903169554877785, 0.07017217884488097, 0.0628025825382853, 0.03675096678658518, 0.012103233420478754], [0.05511963945991105, 0.049068549016530794, 0.03555849525475285, 0.01742902420311021, 0.0024258000326546902, -0.006945619526173896, -0.012824256762401176, -0.01652720881077141, -0.018867226613651145, -0.020346777228088864, -0.021276549064350614, -0.021847641209509994, -0.022175215303099738, -0.022324336255886812, -0.022324336256020483, -0.02217521530350275, -0.021847641210190782, -0.021276549065322503, -0.020346777229376167, -0.018867226615291166, -0.01652720881282055, -0.01282425676494825, -0.0069456195293562395, 0.0024258000286184744, 0.017429024197879173, 0.03555849524777277, 0.049068549006895834, 0.055119639446185364], [0.007127174492710275, 0.022015067876339944, 0.03893292850029434, 0.04239182935779695, 0.032581432921953146, 0.025404400905523428, 0.01997882494047981, 0.01573950980639449, 0.0123173512843619, 0.009465067840074393, 0.007011262899538961, 0.004831781041847172, 0.0028315768480500214, 0.0009328950601668051, -0.0009328950571452777, -0.0028315768449993506, -0.004831781038736049, -0.007011262896333581, -0.00946506783673673, -0.01231735128084771, -0.015739509802647766, -0.019978824936430162, -0.02540440090107765, -0.03258143291698046, -0.0423918293521256, -0.038932928493641994, -0.02201506786827817, -0.007127174482688681], [-0.031788396036350086, -0.028106153502096998, -0.019692773562300037, -0.008372872866156733, 0.0006445863845488953, 0.005758553074859507, 0.00854771412555555, 0.009973088343101622, 0.010618837582398788, 0.010839206940689627, 0.010848088565808878, 0.010773455830556666, 0.010690190660801235, 0.010639460523014344, 0.010639460522986033, 0.01069019066071808, 0.010773455830420997, 0.010848088565625358, 0.010839206940467805, 0.010618837582154428, 0.009973088342864034, 0.008547714125372918, 0.005758553074815098, 0.0006445863847825972, -0.008372872865417103, -0.019692773560672894, -0.028106153498941855, -0.03178839603062655], [-0.0045802311050726074, -0.014460817463747055, -0.02666307683067698, -0.028284940666706893, -0.019510676470427646, -0.013661666032456776, -0.009687533474548038, -0.006929441145179627, -0.004969680894637396, -0.0035393290075922956, -0.002461365746782296, -0.0016154260717305968, -0.0009156866804730379, -0.0002966308153283803, 0.00029663081482628195, 0.0009156866799511776, 0.0016154260711671586, 0.002461365746152855, 0.0035393290068685967, 0.004969680893783857, 0.006929441144149506, 0.009687533473278331, 0.01366166603085861, 0.019510676468368238, 0.028284940664003333, 0.026663076827019017, 0.014460817458647857, 0.004580231097861542]]]

        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_leg_order + 1, -1)

        # Solve the problem
        pydgm.solver.solve()

        for l in range(pydgm.control.scatter_leg_order + 1):
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
        pydgm.control.scatter_leg_order = 0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux
        phi_test = [[3.442525765957952, 3.44133409525864, 3.438914799438404, 3.435188915015736, 3.4300162013446567, 3.423154461297391, 3.4141703934131375, 3.4022425049312393, 3.3857186894563807, 3.3610957244311783, 3.332547468621578, 3.310828206454775, 3.2983691806875637, 3.292637618967479], [1.038826864565325, 1.0405199414437678, 1.0439340321802706, 1.0491279510472575, 1.0561979892073443, 1.065290252825513, 1.0766260147603826, 1.0905777908540444, 1.1080393189273399, 1.1327704173615665, 1.166565957603695, 1.195247115307628, 1.2108105235380155, 1.2184043154658892]]

        phi_test = np.array(phi_test)

        phi_test = phi_test.reshape(2, pydgm.control.scatter_leg_order + 1, -1)  # Group, legendre, cell

        keff_test = 1.17455939

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        phi_zero = pydgm.state.mg_phi[0, :, :].flatten()

        phi_zero_test = phi_test[:, 0].flatten() / np.linalg.norm(phi_test[:, 0]) * np.linalg.norm(phi_zero)

        np.testing.assert_array_almost_equal(phi_zero, phi_zero_test, 12)

        # Test the angular flux
        self.angular_test()

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
        pydgm.control.scatter_leg_order = 7

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
        for l in range(pydgm.control.scatter_leg_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()
            phi_zero_test = phi_test[:, l].flatten() / np.linalg.norm(phi_test[:, l]) * np.linalg.norm(phi)
            np.testing.assert_array_almost_equal(phi, phi_zero_test, 12)
        # np.testing.assert_array_almost_equal(phi_one, phi_one_test, 12)

        # Test the angular flux
        self.angular_test()

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


class TestSOLVER_2D(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.spatial_dimension = 2
        pydgm.control.fine_mesh_x = [10, 10, 6]
        pydgm.control.fine_mesh_y = [10, 10, 6]
        pydgm.control.coarse_mesh_x = [0.0, 21.42, 42.84, 64.26]
        pydgm.control.coarse_mesh_y = [0.0, 21.42, 42.84, 64.26]
        pydgm.control.material_map = [2, 4, 5,
                                      4, 2, 5,
                                      5, 5, 5]
        pydgm.control.xs_name = 'test/partisn_cross_sections/anisotropic_2g'.ljust(256)
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.number_angles_pol = 8
        pydgm.control.number_angles_azi = 4
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 1.0
        pydgm.control.boundary_north = 1.0
        pydgm.control.boundary_south = 0.0
        pydgm.control.allow_fission = True
        pydgm.control.eigen_print = False
        pydgm.control.outer_print = False
        pydgm.control.eigen_tolerance = 1e-12
        pydgm.control.outer_tolerance = 1e-12
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.scatter_leg_order = 0
        pydgm.control.use_DGM = False
        pydgm.control.max_eigen_iters = 10000
        pydgm.control.max_outer_iters = 1000

    def angular_test(self):
        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for o in range(4):
            for c in range(pydgm.control.number_cells):
                for a in range(nAngles):
                    phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, o * nAngles + a, c]
        np.testing.assert_array_almost_equal(phi_test, pydgm.state.phi[0, :, :], 12)

    def test_solver_basic_2D(self):
        '''
        Test for a basic 1 group problem
        '''
        pydgm.control.fine_mesh_x = [25]
        pydgm.control.fine_mesh_y = [25]
        pydgm.control.coarse_mesh_x = [0.0, 10.0]
        pydgm.control.coarse_mesh_y = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)
        pydgm.control.number_angles_pol = 4
        pydgm.control.number_angles_azi = 8
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 0.0
        pydgm.control.boundary_north = 0.0
        pydgm.control.boundary_south = 0.0
        pydgm.control.allow_fission = False
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.eigen_tolerance = 1e-12
        pydgm.control.outer_tolerance = 1e-12

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 1)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [1.5080753059284397, 1.946122872048338, 2.084822636465254, 2.186505970931235, 2.2334426638060125, 2.267684057813763, 2.28435820032064, 2.2973261453693783, 2.3032183971736853, 2.308598668857393, 2.3103058547318387, 2.312456055349828, 2.312090293490035, 2.3124560553498283, 2.3103058547318374, 2.308598668857394, 2.3032183971736857, 2.297326145369378, 2.284358200320639, 2.267684057813764, 2.2334426638060143, 2.1865059709312344, 2.084822636465254, 1.9461228720483368, 1.5080753059284395, 1.9461228720483372, 2.5240683580250036, 2.708858388030063, 2.837535212080506, 2.8968151093628944, 2.939148390513296, 2.960210439995761, 2.976281562292791, 2.983846727779507, 2.9898198939815304, 2.992394766617096, 2.994515726326748, 2.9942720361507607, 2.9945157263267466, 2.9923947666170956, 2.9898198939815313, 2.983846727779507, 2.976281562292791, 2.960210439995762, 2.9391483905132967, 2.8968151093628953, 2.8375352120805064, 2.708858388030062, 2.5240683580250036, 1.9461228720483374, 2.084822636465253, 2.708858388030061, 2.9291804805139705, 3.0745255698671574, 3.1437599955888547, 3.191948446629185, 3.217041906284302, 3.233971943545909, 3.2438530935530823, 3.2500726459289133, 3.2539101052606996, 3.255154134257649, 3.2571077715394496, 3.2551541342576513, 3.253910105260699, 3.250072645928915, 3.243853093553083, 3.233971943545908, 3.217041906284303, 3.1919484466291843, 3.1437599955888555, 3.0745255698671587, 2.92918048051397, 2.708858388030061, 2.0848226364652547, 2.1865059709312344, 2.837535212080507, 3.074525569867159, 3.2355824363615344, 3.311759788251185, 3.3642574785567696, 3.3913676034451683, 3.410211215656466, 3.4211146970854887, 3.4263398792287068, 3.4321189008243445, 3.4321598526394697, 3.4341650279824067, 3.4321598526394714, 3.4321189008243453, 3.426339879228707, 3.4211146970854873, 3.4102112156564655, 3.39136760344517, 3.3642574785567696, 3.311759788251186, 3.235582436361535, 3.0745255698671574, 2.837535212080506, 2.186505970931234, 2.2334426638060143, 2.896815109362893, 3.1437599955888555, 3.311759788251186, 3.393896912395787, 3.450877451621544, 3.4820763800098775, 3.5011882659129308, 3.5139983002755875, 3.5205560739225668, 3.525072634935892, 3.526752763819402, 3.5284690170819553, 3.526752763819404, 3.525072634935893, 3.5205560739225685, 3.513998300275589, 3.5011882659129303, 3.482076380009876, 3.450877451621545, 3.393896912395788, 3.3117597882511878, 3.143759995588856, 2.8968151093628953, 2.233442663806014, 2.267684057813764, 2.9391483905132945, 3.1919484466291848, 3.36425747855677, 3.4508774516215457, 3.506625385363502, 3.540020677366607, 3.5598161901141596, 3.5731592756818187, 3.580790908768418, 3.585093755319976, 3.588655555183777, 3.587701435586312, 3.5886555551837764, 3.585093755319975, 3.5807909087684173, 3.573159275681819, 3.5598161901141605, 3.540020677366606, 3.5066253853635034, 3.4508774516215452, 3.3642574785567683, 3.191948446629186, 2.9391483905132962, 2.2676840578137636, 2.2843582003206384, 2.9602104399957616, 3.2170419062843023, 3.39136760344517, 3.4820763800098784, 3.5400206773666056, 3.5761432577575474, 3.595507849679144, 3.6092236278837864, 3.6166306602415275, 3.620740116577421, 3.6243871474781972, 3.6232319704272657, 3.6243871474781972, 3.6207401165774193, 3.616630660241527, 3.6092236278837873, 3.595507849679142, 3.576143257757547, 3.5400206773666074, 3.4820763800098753, 3.3913676034451696, 3.2170419062843023, 2.9602104399957625, 2.2843582003206384, 2.297326145369379, 2.9762815622927907, 3.2339719435459084, 3.410211215656465, 3.5011882659129325, 3.5598161901141605, 3.595507849679144, 3.6169414545829968, 3.631455429629087, 3.6399476786272027, 3.6438692380860216, 3.6468408624482915, 3.6479090064850546, 3.6468408624482906, 3.643869238086022, 3.639947678627202, 3.63145542962909, 3.6169414545829985, 3.595507849679143, 3.55981619011416, 3.5011882659129308, 3.410211215656466, 3.2339719435459093, 2.9762815622927903, 2.297326145369378, 2.303218397173686, 2.983846727779506, 3.243853093553082, 3.4211146970854878, 3.5139983002755875, 3.573159275681821, 3.609223627883786, 3.6314554296290886, 3.6434694804427883, 3.6531454876553124, 3.6585847759054637, 3.660352190711248, 3.661803160512844, 3.66035219071125, 3.6585847759054633, 3.6531454876553164, 3.6434694804427883, 3.631455429629087, 3.609223627883788, 3.5731592756818182, 3.513998300275587, 3.4211146970854873, 3.243853093553083, 2.983846727779507, 2.3032183971736844, 2.308598668857394, 2.98981989398153, 3.250072645928914, 3.4263398792287068, 3.520556073922566, 3.5807909087684173, 3.6166306602415275, 3.639947678627202, 3.6531454876553138, 3.6618966237313337, 3.6666274010388324, 3.6691096984828104, 3.6703984612945266, 3.669109698482812, 3.66662740103883, 3.6618966237313324, 3.6531454876553155, 3.639947678627202, 3.616630660241527, 3.5807909087684178, 3.5205560739225676, 3.4263398792287076, 3.2500726459289146, 2.9898198939815304, 2.308598668857394, 2.3103058547318396, 2.9923947666170942, 3.2539101052606987, 3.432118900824345, 3.525072634935892, 3.585093755319976, 3.620740116577423, 3.6438692380860207, 3.6585847759054624, 3.6666274010388324, 3.6717399870305725, 3.674626022126563, 3.6752868352570567, 3.674626022126563, 3.671739987030571, 3.6666274010388307, 3.6585847759054646, 3.6438692380860225, 3.620740116577419, 3.585093755319974, 3.5250726349358925, 3.4321189008243436, 3.2539101052606982, 2.992394766617096, 2.3103058547318374, 2.312456055349828, 2.994515726326747, 3.255154134257649, 3.43215985263947, 3.5267527638194034, 3.5886555551837773, 3.6243871474781972, 3.646840862448291, 3.660352190711251, 3.669109698482811, 3.674626022126563, 3.6772166834260975, 3.6778445162262035, 3.677216683426096, 3.6746260221265645, 3.6691096984828127, 3.6603521907112473, 3.6468408624482906, 3.624387147478198, 3.588655555183777, 3.5267527638194016, 3.4321598526394728, 3.255154134257651, 2.9945157263267475, 2.312456055349828, 2.3120902934900345, 2.9942720361507584, 3.2571077715394496, 3.434165027982406, 3.5284690170819553, 3.587701435586313, 3.623231970427265, 3.647909006485054, 3.6618031605128434, 3.670398461294527, 3.675286835257055, 3.6778445162262052, 3.679217736376174, 3.6778445162262035, 3.6752868352570562, 3.6703984612945275, 3.6618031605128456, 3.647909006485053, 3.6232319704272657, 3.587701435586314, 3.5284690170819553, 3.434165027982406, 3.257107771539452, 2.9942720361507598, 2.312090293490034, 2.312456055349828, 2.9945157263267483, 3.2551541342576504, 3.4321598526394697, 3.5267527638194025, 3.588655555183776, 3.624387147478196, 3.6468408624482924, 3.6603521907112495, 3.66910969848281, 3.6746260221265636, 3.677216683426096, 3.6778445162262043, 3.677216683426096, 3.674626022126564, 3.669109698482811, 3.6603521907112464, 3.646840862448291, 3.624387147478198, 3.588655555183778, 3.5267527638194016, 3.432159852639472, 3.255154134257649, 2.9945157263267483, 2.312456055349829, 2.310305854731839, 2.992394766617095, 3.253910105260699, 3.4321189008243445, 3.525072634935892, 3.5850937553199755, 3.6207401165774207, 3.643869238086021, 3.658584775905463, 3.6666274010388307, 3.671739987030575, 3.674626022126562, 3.6752868352570554, 3.674626022126564, 3.6717399870305716, 3.666627401038832, 3.6585847759054646, 3.6438692380860216, 3.6207401165774207, 3.585093755319975, 3.525072634935893, 3.432118900824344, 3.2539101052607005, 2.9923947666170956, 2.310305854731838, 2.3085986688573947, 2.9898198939815295, 3.2500726459289146, 3.4263398792287063, 3.5205560739225668, 3.580790908768417, 3.616630660241526, 3.6399476786272027, 3.653145487655314, 3.6618966237313337, 3.666627401038832, 3.669109698482809, 3.6703984612945266, 3.6691096984828118, 3.6666274010388316, 3.6618966237313333, 3.6531454876553138, 3.639947678627202, 3.6166306602415275, 3.5807909087684178, 3.520556073922569, 3.426339879228707, 3.250072645928913, 2.9898198939815295, 2.308598668857394, 2.3032183971736857, 2.9838467277795044, 3.2438530935530805, 3.4211146970854873, 3.513998300275587, 3.573159275681821, 3.6092236278837864, 3.6314554296290886, 3.64346948044279, 3.6531454876553133, 3.658584775905463, 3.6603521907112486, 3.661803160512844, 3.66035219071125, 3.6585847759054624, 3.6531454876553164, 3.643469480442788, 3.6314554296290864, 3.6092236278837877, 3.5731592756818182, 3.513998300275587, 3.4211146970854887, 3.243853093553082, 2.983846727779505, 2.3032183971736844, 2.297326145369379, 2.9762815622927903, 3.2339719435459098, 3.410211215656464, 3.501188265912933, 3.5598161901141614, 3.595507849679145, 3.616941454582996, 3.631455429629087, 3.639947678627202, 3.643869238086022, 3.6468408624482924, 3.647909006485053, 3.6468408624482906, 3.643869238086022, 3.639947678627201, 3.631455429629088, 3.616941454582997, 3.595507849679144, 3.5598161901141605, 3.501188265912931, 3.410211215656467, 3.23397194354591, 2.976281562292791, 2.2973261453693787, 2.2843582003206375, 2.960210439995763, 3.2170419062843023, 3.391367603445169, 3.482076380009877, 3.5400206773666056, 3.5761432577575456, 3.5955078496791453, 3.6092236278837864, 3.616630660241529, 3.6207401165774216, 3.6243871474781963, 3.6232319704272657, 3.6243871474781986, 3.6207401165774207, 3.616630660241528, 3.6092236278837877, 3.5955078496791435, 3.576143257757547, 3.5400206773666065, 3.482076380009877, 3.391367603445169, 3.217041906284303, 2.960210439995762, 2.2843582003206397, 2.267684057813763, 2.9391483905132976, 3.1919484466291843, 3.36425747855677, 3.450877451621544, 3.5066253853635043, 3.540020677366608, 3.5598161901141605, 3.5731592756818196, 3.5807909087684164, 3.585093755319976, 3.588655555183776, 3.587701435586313, 3.588655555183778, 3.585093755319975, 3.5807909087684178, 3.57315927568182, 3.5598161901141627, 3.5400206773666065, 3.5066253853635017, 3.4508774516215435, 3.3642574785567687, 3.191948446629185, 2.9391483905132967, 2.2676840578137636, 2.233442663806013, 2.896815109362894, 3.143759995588855, 3.3117597882511847, 3.393896912395788, 3.450877451621545, 3.4820763800098766, 3.5011882659129316, 3.5139983002755857, 3.5205560739225676, 3.525072634935891, 3.526752763819402, 3.5284690170819553, 3.5267527638194034, 3.5250726349358916, 3.5205560739225694, 3.513998300275588, 3.501188265912931, 3.482076380009876, 3.4508774516215457, 3.3938969123957863, 3.311759788251185, 3.143759995588855, 2.8968151093628953, 2.2334426638060134, 2.186505970931234, 2.8375352120805077, 3.0745255698671587, 3.2355824363615358, 3.311759788251185, 3.364257478556769, 3.39136760344517, 3.410211215656467, 3.4211146970854878, 3.4263398792287063, 3.4321189008243436, 3.4321598526394705, 3.4341650279824054, 3.43215985263947, 3.4321189008243462, 3.426339879228707, 3.4211146970854873, 3.4102112156564655, 3.391367603445171, 3.364257478556769, 3.311759788251185, 3.235582436361535, 3.074525569867158, 2.8375352120805064, 2.1865059709312344, 2.0848226364652533, 2.7088583880300607, 2.9291804805139696, 3.0745255698671583, 3.1437599955888555, 3.191948446629186, 3.217041906284302, 3.2339719435459107, 3.243853093553082, 3.250072645928915, 3.2539101052606982, 3.2551541342576504, 3.257107771539451, 3.255154134257649, 3.2539101052606996, 3.2500726459289146, 3.243853093553082, 3.2339719435459098, 3.2170419062843023, 3.191948446629185, 3.1437599955888555, 3.0745255698671596, 2.929180480513972, 2.7088583880300625, 2.0848226364652533, 1.9461228720483374, 2.524068358025004, 2.708858388030061, 2.8375352120805073, 2.896815109362895, 2.9391483905132976, 2.960210439995762, 2.9762815622927907, 2.9838467277795035, 2.9898198939815295, 2.992394766617094, 2.994515726326748, 2.9942720361507593, 2.9945157263267466, 2.992394766617094, 2.9898198939815295, 2.9838467277795058, 2.9762815622927925, 2.9602104399957625, 2.9391483905132967, 2.8968151093628935, 2.8375352120805077, 2.708858388030063, 2.5240683580250027, 1.9461228720483377, 1.5080753059284393, 1.9461228720483381, 2.084822636465254, 2.186505970931234, 2.233442663806013, 2.2676840578137627, 2.284358200320637, 2.2973261453693787, 2.3032183971736857, 2.3085986688573947, 2.3103058547318382, 2.3124560553498275, 2.312090293490033, 2.3124560553498266, 2.3103058547318387, 2.3085986688573943, 2.303218397173686, 2.29732614536938, 2.2843582003206384, 2.2676840578137636, 2.233442663806014, 2.186505970931234, 2.0848226364652533, 1.9461228720483366, 1.50807530592844]

        phi_test = np.array(phi_test)
#         import matplotlib.pyplot as plt
#         plt.contourf(pydgm.state.phi.flatten().reshape((25, 25)), cmap='inferno')
#         plt.colorbar()
#         plt.show()

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi, phi_test, 12)

        self.angular_test()

    def test_solver_partisn_eigen_2g_l0(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.scatter_leg_order = 0
        pydgm.control.eigen_print = 1

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[48.42395003305526, 46.222210660008, 41.98980482747511, 35.59027550995211, 28.16111798108446, 20.76361351645976, 13.797884720009113, 9.231151174905873, 6.522951052295694, 4.702697099163015, 2.101222285568176, 0.24248043934926675, 0.07861501523905629, 46.222210660008024, 44.14474961910683, 40.15456791375139, 34.098552110745814, 27.066311848518303, 20.064745313412786, 13.437552812915197, 9.075288586611054, 6.478384576323626, 4.713246290450097, 2.1123171278709423, 0.24562460640777653, 0.07934399550600754, 41.98980482747515, 40.15456791375141, 36.62799204420789, 31.26224876483542, 25.002335715995997, 18.714185651966687, 12.751519345868404, 8.804870233766893, 6.419720425828961, 4.744658383761403, 2.141507338426806, 0.2556706039546343, 0.07566449422468555, 35.59027550995214, 34.09855211074582, 31.262248764835395, 26.92241557968437, 21.923895634092176, 16.921044898100757, 12.00106242142249, 8.645221732882503, 6.511451420066254, 4.912950734783291, 2.222992980816688, 0.251371949385381, 0.0769930147862221, 28.161117981084487, 27.06631184851831, 25.002335715995986, 21.923895634092162, 18.441264986702294, 14.988871979989852, 11.507900967259996, 8.842225611870933, 6.9336126798963145, 5.287265114004081, 2.2932556402831636, 0.2437900495853736, 0.0767213004939575, 20.763613516459756, 20.064745313412775, 18.714185651966677, 16.92104489810075, 14.988871979989828, 12.6235941821774, 10.595182399855716, 8.765562141164965, 7.127017417701973, 5.335693959312803, 2.2644845634080415, 0.263392741149507, 0.05388706748127458, 13.797884720009105, 13.437552812915172, 12.751519345868369, 12.001062421422462, 11.507900967259944, 10.59518239985566, 9.646708186617062, 8.477841280926441, 7.1138777406149005, 5.326452702821138, 2.207506617522488, 0.24966543989124923, 0.04916746663356068, 9.23115117490584, 9.075288586611034, 8.804870233766861, 8.64522173288246, 8.84222561187086, 8.76556214116486, 8.477841280926388, 7.747534239786249, 6.669417692085264, 5.047710509575884, 2.083391582317072, 0.21604746414913698, 0.04675007902079233, 6.522951052295656, 6.478384576323583, 6.419720425828921, 6.511451420066227, 6.93361267989623, 7.127017417701862, 7.113877740614872, 6.669417692085306, 5.84338218049466, 4.444452138715809, 1.8140255020687122, 0.17363900752533054, 0.0430122908631021, 4.70269709916297, 4.713246290450051, 4.74465838376136, 4.912950734783274, 5.287265114004025, 5.335693959312714, 5.326452702821129, 5.047710509575963, 4.444452138715863, 3.37305010240024, 1.3505489977585063, 0.13770861620581115, 0.038376330396076616, 2.1012222855681513, 2.1123171278709156, 2.1415073384267846, 2.2229929808166755, 2.2932556402831445, 2.264484563408015, 2.2075066175224864, 2.083391582317103, 1.814025502068742, 1.3505489977585179, 0.638125114252331, 0.1228954110342053, 0.014909693870152776, 0.24248043934926214, 0.245624606407773, 0.2556706039546311, 0.251371949385376, 0.2437900495853713, 0.26339274114951056, 0.24966543989125, 0.216047464149134, 0.17363900752533126, 0.1377086162058142, 0.12289541103420681, 0.06387850104567742, 0.008320684057974764, 0.07861501523905606, 0.07934399550600685, 0.0756644942246844, 0.07699301478622232, 0.07672130049395769, 0.05388706748127248, 0.049167466633560114, 0.04675007902079354, 0.043012290863101904, 0.03837633039607605, 0.014909693870152887, 0.00832068405797489, 0.007002594579192592]], [[7.202758549977769, 6.888974926372049, 6.21653126861651, 5.422404317767946, 3.4876802110873477, 0.8125196847688073, 0.2143018013783584, 0.34707263227943275, 0.0724300566717295, 0.26699836719596454, 1.6951217659107705, 1.7057627727352416, 0.32961611085074805, 6.888974926372051, 6.594433080472231, 5.955398360690706, 5.209803693018254, 3.3589334776216306, 0.7706944373255585, 0.19223684228354834, 0.3297655915333281, 0.06101583811142334, 0.2579949800250307, 1.7089495480784491, 1.7212451500055042, 0.3319476587298314, 6.216531268616516, 5.955398360690711, 5.397837998382603, 4.737578746082998, 3.068017769419533, 0.7493356287168504, 0.2263001074086465, 0.34497122283113935, 0.08772795888753225, 0.2836876508548885, 1.746680327694202, 1.746465304349609, 0.34127775840172353, 5.422404317767951, 5.209803693018256, 4.737578746082995, 4.227237374009562, 2.76465340654945, 0.6143230512308309, 0.13198548276598726, 0.2799491195603595, 0.03728791308360167, 0.24186663586165913, 1.796106651005342, 1.8003941737388036, 0.33535516604961324, 3.487680211087352, 3.3589334776216297, 3.0680177694195305, 2.7646534065494475, 1.9111366577698639, 0.6282318063295139, 0.3169798926627501, 0.3860581726834555, 0.15485716820471113, 0.3603654237487099, 1.9440348143888462, 1.8321386098590806, 0.3270787992526564, 0.8125196847688091, 0.7706944373255585, 0.7493356287168506, 0.614323051230833, 0.6282318063295139, 1.2171880938319657, 1.2834897070278657, 1.0452711573886047, 0.7824016703945608, 0.9763570559973967, 2.183520425088211, 1.78655330007802, 0.33864128519880005, 0.21430180137835728, 0.1922368422835475, 0.22630010740864415, 0.1319854827659895, 0.3169798926627478, 1.2834897070278348, 1.5584502377585168, 1.296279571566085, 1.0299500581971328, 1.21873159816236, 2.260629150134286, 1.7233089650210784, 0.31672167059841133, 0.34707263227942964, 0.3297655915333266, 0.3449712228311344, 0.27994911956035956, 0.38605817268345405, 1.0452711573885571, 1.2962795715660642, 1.1659761202233148, 0.9360728206292767, 1.1239178583304723, 2.117864224171333, 1.6074404212623532, 0.2794208291034941, 0.07243005667172955, 0.06101583811142412, 0.08772795888753106, 0.03728791308360328, 0.1548571682047063, 0.7824016703945267, 1.0299500581971404, 0.9360728206293184, 0.762944486429801, 0.9478348964629764, 1.8482800913894744, 1.4046684233335172, 0.2354538783553743, 0.2669983671959594, 0.25799498002502674, 0.28368765085488223, 0.24186663586166476, 0.3603654237487108, 0.976357055997341, 1.218731598162364, 1.12391785833054, 0.9478348964629971, 1.0260256329511004, 1.5807211380046415, 1.1082635060738162, 0.19132944645569003, 1.6951217659107425, 1.7089495480784223, 1.746680327694178, 1.7961066510053365, 1.944034814388841, 2.183520425088153, 2.260629150134279, 2.1178642241713814, 1.8482800913894926, 1.580721138004648, 1.2406251977431755, 0.6572061701243973, 0.1566576423587207, 1.705762772735221, 1.7212451500054857, 1.7464653043495926, 1.8003941737387814, 1.8321386098590553, 1.7865533000780094, 1.7233089650210702, 1.6074404212623463, 1.4046684233335254, 1.1082635060738235, 0.6572061701243997, 0.2755904177421591, 0.08483399995941882, 0.3296161108507485, 0.33194765872983034, 0.34127775840171964, 0.3353551660496, 0.3270787992526449, 0.33864128519879616, 0.3167216705983977, 0.2794208291034757, 0.23545387835536677, 0.19132944645568753, 0.15665764235871701, 0.08483399995941722, 0.019487885991977514]]]

        phi_test = np.array(phi_test)

        keff_test = 1.10942229

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()
            phi_zero_test = phi_test[:, l].flatten() / np.linalg.norm(phi_test[:, l]) * np.linalg.norm(phi)
            np.testing.assert_array_almost_equal(phi, phi_zero_test, 12)
        # np.testing.assert_array_almost_equal(phi_one, phi_one_test, 12)

        self.angular_test()

    def test_solver_partisn_eigen_2g_l2(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.scatter_leg_order = 2

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[36.59996679007787, 35.281067133764395, 32.76325982777605, 28.853747488886853, 24.36726994797399, 19.708049014163848, 14.742137253395782, 11.041823479878737, 8.501789544137143, 6.511440929014308, 3.337192853079299, 0.9441097098011693, 0.3306704829944207, 35.2810671337629, 34.028107850950015, 31.64129792066047, 27.9166602289788, 23.647353108213835, 19.219323607886178, 14.461823333923894, 10.896464277274791, 8.437154150021792, 6.4918615486649855, 3.330484299592281, 0.9424120316380205, 0.3297128130397371, 32.763259827771684, 31.64129792065766, 29.5015489932898, 26.154083774996245, 22.30131005666772, 18.25824253950943, 13.906084055362973, 10.620387541509178, 8.315545492443295, 6.443190027194012, 3.31774522754129, 0.9414419581553315, 0.323237230027973, 28.853747488878792, 27.91666022897243, 26.154083774992646, 23.367775350830684, 20.22453812595896, 16.95459119097506, 13.260093877878546, 10.379332188121417, 8.26493589746647, 6.463634725957252, 3.3232114580978727, 0.9206757679586796, 0.318531599168613, 24.367269947961915, 23.647353108203742, 22.301310056661347, 20.22453812595591, 17.895122491443473, 15.48951255390415, 12.744585368301152, 10.3577427111256, 8.437893822499452, 6.613630224530355, 3.2793660624703214, 0.8908775119995902, 0.30731900490325276, 19.708049014140975, 19.219323607866443, 18.258242539496575, 16.954591190965818, 15.489512553897855, 13.466702293549568, 11.67107236200359, 9.929636126140782, 8.258605759736408, 6.31170848112277, 3.1108426507157727, 0.880827380870822, 0.26818117535930414, 14.742137253340244, 14.461823333874278, 13.906084055324246, 13.26009387784886, 12.74458536828224, 11.671072361997073, 10.602307245771678, 9.36501364591254, 7.933101821464251, 6.038437658751138, 2.915701458866213, 0.8218504806423068, 0.24403422497960806, 11.041823479802368, 10.896464277203991, 10.620387541444785, 10.379332188079468, 10.357742711108889, 9.929636126135257, 9.365013645911814, 8.472744891785362, 7.293110883918011, 5.583887965343672, 2.671437120671394, 0.7260227459199274, 0.220618175280668, 8.501789544088636, 8.437154149975916, 8.315545492401391, 8.264935897445714, 8.437893822503243, 8.258605759746573, 7.933101821474083, 7.293110883925002, 6.3553026346653265, 4.876244784105041, 2.300170888104252, 0.6093663619841947, 0.19171299656156268, 6.511440928999419, 6.491861548651902, 6.443190027189248, 6.463634725960598, 6.613630224543946, 6.311708481143707, 6.038437658770625, 5.583887965358106, 4.876244784112908, 3.762893577995164, 1.7671575195368823, 0.4892718509941254, 0.15837333718494087, 3.3371928530759987, 3.3304842995904864, 3.3177452275428694, 3.323211458102908, 3.279366062478352, 3.1108426507262257, 2.9157014588765917, 2.671437120679552, 2.3001708881094665, 1.767157519539005, 0.9915944326335091, 0.3545096157325757, 0.10183666823108038, 0.9441097098007891, 0.942412031637993, 0.9414419581555757, 0.9206757679597427, 0.890877512001284, 0.8808273808725227, 0.8218504806440977, 0.7260227459216141, 0.6093663619854118, 0.4892718509949845, 0.3545096157330004, 0.18259957424496095, 0.05646354353831498, 0.3306704829944108, 0.3297128130397755, 0.32323723002833743, 0.31853159916901924, 0.3073190049037114, 0.268181175360009, 0.2440342249802865, 0.2206181752812435, 0.19171299656210786, 0.15837333718530036, 0.10183666823123522, 0.056463543538370256, 0.024430440990623588]], [[5.42473775923082, 5.241656694260922, 4.837296801759442, 4.349454857797112, 2.925614185125308, 0.7869711916352219, 0.24377476250338376, 0.39889415670175554, 0.09472563999661388, 0.4008872511970434, 2.7726744184404657, 3.2225340129235454, 1.032363848082147, 5.241656694260656, 5.0694178140836765, 4.681566563948332, 4.22158900085095, 2.8459213102346665, 0.7541038193010808, 0.22320225802099, 0.38277057101109807, 0.08221111505635927, 0.38893871488015425, 2.7678887449251848, 3.2171949090199266, 1.0286471272852813, 4.837296801758777, 4.681566563947923, 4.340071987696634, 3.923834464162148, 2.654776294388762, 0.7457484182358515, 0.2578525297684245, 0.39954899128082344, 0.11069027407587846, 0.41384764036627136, 2.7632244282686784, 3.199408401405134, 1.0260368424301705, 4.3494548577957595, 4.22158900084989, 3.923834464161518, 3.604128307617552, 2.4584522594532383, 0.6263211610749541, 0.1586352526323979, 0.3270580032612379, 0.047488492483449364, 0.3528205723546801, 2.747996276862788, 3.188418930942426, 1.0013330026434832, 2.9256141851236444, 2.845921310233186, 2.654776294387774, 2.4584522594527014, 1.7750840808154338, 0.665897060378661, 0.3658835536935602, 0.4543150915520184, 0.19063512544483185, 0.496293459752625, 2.826199370015811, 3.130104943096746, 0.9674542031272344, 0.7869711916363102, 0.7541038193016515, 0.7457484182375571, 0.6263211610759378, 0.6658970603787054, 1.2645947754267233, 1.3592465247731151, 1.1518804472097444, 0.8906407607312736, 1.212133285075135, 2.9929764429886894, 2.974200397105705, 0.9434188718781119, 0.24377476250141286, 0.22320225801901974, 0.2578525297680213, 0.15863525263122105, 0.3658835536922648, 1.3592465247748673, 1.652115993184192, 1.406708038797413, 1.1553787971002059, 1.4619499916027685, 2.9801086816891775, 2.787902371974283, 0.8755626994938561, 0.39889415669424033, 0.38277057100428424, 0.39954899127616866, 0.32705800325664625, 0.4543150915472816, 1.1518804472081405, 1.4067080387916882, 1.2690701892467762, 1.0440872034360746, 1.3293753149285632, 2.737824345602011, 2.5469584733483592, 0.7820751935356425, 0.09472563999549896, 0.08221111505498922, 0.11069027407557451, 0.047488492482223865, 0.19063512544461866, 0.890640760736852, 1.1553787971005303, 1.0440872034370596, 0.8711619597442211, 1.1377066794447168, 2.3967836736220858, 2.2264721681949946, 0.6707691280813843, 0.4008872511985479, 0.3889387148806831, 0.41384764036951877, 0.35282057235626446, 0.4962934597550338, 1.2121332850906874, 1.4619499916147438, 1.329375314936677, 1.1377066794526796, 1.2858865642631319, 2.1334971698472724, 1.8229996044404349, 0.5483792191519649, 2.7726744184382643, 2.7678887449236824, 2.7632244282713674, 2.74799627686653, 2.8261993700210692, 2.992976443001731, 2.9801086816996922, 2.73782434560823, 2.3967836736287285, 2.1334971698505822, 1.8602474007805279, 1.2223119794738697, 0.4095761102161706, 3.2225340129222864, 3.217194909020082, 3.1994084014069952, 3.188418930946922, 3.130104943103865, 2.9742003971131736, 2.787902371982046, 2.5469584733556365, 2.2264721681999027, 1.8229996044436139, 1.2223119794752164, 0.64052791732327, 0.22725955703558293, 1.03236384808243, 1.0286471272854654, 1.026036842430921, 1.0013330026452756, 0.9674542031294618, 0.9434188718806014, 0.8755626994970329, 0.7820751935391539, 0.6707691280838844, 0.5483792191534078, 0.4095761102169261, 0.22725955703575212, 0.07009276418834443]]]

        phi_test = np.array(phi_test)

        keff_test = 1.07747550

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, keff_test, 8)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()
            phi_zero_test = phi_test[:, l].flatten() / np.linalg.norm(phi_test[:, l]) * np.linalg.norm(phi)
            np.testing.assert_array_almost_equal(phi, phi_zero_test, 12)
        # np.testing.assert_array_almost_equal(phi_one, phi_one_test, 12)

        self.angular_test()

    def test_solver_partisn_fixed_2g_l0(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.scatter_leg_order = 0
        pydgm.control.outer_print = 1

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[28.97417201124405, 28.961876843910467, 28.935036155856867, 28.88389562825871, 28.799370284612856, 28.650481198281923, 28.412334994037195, 27.996226906136886, 27.338112888923956, 26.13286433032176, 24.504868260671024, 23.376248708196183, 22.840720429778344, 22.566279827711085, 22.505146310554792, 22.604718387425763, 22.920191323544344, 23.543483514729733, 24.711626191747946, 26.91215087389021, 31.17477661353444, 34.466896851814745, 35.25373748081983, 33.502071278183465, 28.77483151482767, 16.358683693487315, 28.961876843910478, 28.9497658542242, 28.923240637113572, 28.87266175756438, 28.78882352499325, 28.641283586318668, 28.404535310338, 27.989783116874957, 27.33336856079424, 26.130225050808836, 24.504903726226225, 23.3780815857872, 22.843783810937715, 22.570214121204597, 22.50977332981829, 22.610123910381624, 22.925639860561862, 23.549068590306188, 24.71736659351486, 26.91770266731768, 31.17951057934149, 34.470994231414174, 35.25664344986803, 33.5038794597178, 28.77585015379511, 16.359025754408616, 28.935036155856878, 28.92324063711357, 28.897462342426643, 28.848048957576133, 28.76619731746472, 28.620574161536183, 28.38701749944784, 27.97664122414954, 27.32490793140482, 26.125985856580506, 24.503528834364275, 23.380355285011554, 22.849641343712914, 22.579039602824377, 22.520617641686876, 22.621528250601116, 22.93788380616796, 23.561141401784134, 24.72864933516679, 26.92888243706488, 31.191090733670347, 34.480134526778656, 35.26305077259156, 33.50776653339367, 28.7777976554845, 16.35988382872866, 28.883895628258706, 28.87266175756437, 28.84804895757614, 28.800719440134053, 28.721535778178513, 28.58105342059804, 28.352934645297154, 27.94909563503854, 27.305128819020055, 26.11576198763565, 24.504554880551243, 23.389726285000556, 22.86498920341311, 22.598770193741966, 22.5433905774919, 22.64705649940848, 22.963785618991384, 23.587018807371972, 24.75437532313613, 26.953180383664687, 31.211540557117562, 34.496579675002025, 35.27393928148091, 33.51415697607108, 28.781199706354247, 16.36094048538063, 28.799370284612852, 28.788823524993244, 28.766197317464712, 28.721535778178517, 28.64785442737091, 28.51324079026957, 28.295720545130333, 27.904894061609124, 27.276708124742356, 26.1026261434843, 24.5037543651752, 23.40255999268973, 22.89013913224375, 22.633676683894826, 22.584916303892587, 22.691200363524818, 23.009143661466965, 23.631324168188108, 24.795382533512722, 26.99227818244248, 31.248635045660944, 34.52405488193136, 35.29143265888405, 33.52378055581662, 28.78551578180205, 16.36290463923812, 28.65048119828192, 28.641283586318636, 28.620574161536172, 28.581053420598032, 28.513240790269574, 28.391276038963685, 28.188474451444556, 27.819256273879276, 27.21722655224911, 26.074854124922737, 24.512159881216412, 23.43933643069502, 22.948833062000606, 22.707886713235784, 22.669272390221153, 22.781792738845834, 23.100552736381093, 23.72013135699426, 24.88005246049207, 27.06947892233491, 31.31111115236013, 34.568266937281194, 35.31724402383651, 33.53695571092247, 28.792056542061065, 16.364644583061033, 28.412334994037195, 28.404535310338, 28.387017499447836, 28.352934645297164, 28.295720545130333, 28.188474451444563, 28.01274468373793, 27.679721424521517, 27.126347383148584, 26.038146868902924, 24.523141342718493, 23.499889760017513, 23.05040591722023, 22.838411823533335, 22.8189425951415, 22.939252817085084, 23.25819656457745, 23.872320063159243, 25.018346690859016, 27.194990703758435, 31.417871916619784, 34.63789278233283, 35.354665016299066, 33.55460338479755, 28.79908181795781, 16.36797382057012, 27.9962269061369, 27.98978311687498, 27.97664122414955, 27.94909563503855, 27.90489406160914, 27.819256273879276, 27.67972142452151, 27.407960649993214, 26.939069907906404, 25.959198777088414, 24.571416666712818, 23.649707040587323, 23.274020854762234, 23.11371003658463, 23.121301215046394, 23.25497224447548, 23.574641321199408, 24.174545569531233, 25.297805330531318, 27.439350194467615, 31.597300907805963, 34.74289680169918, 35.40317993138437, 33.575318009234145, 28.808963774640166, 16.370657604501947, 27.338112888923952, 27.333368560794263, 27.32490793140481, 27.305128819020027, 27.276708124742353, 27.217226552249127, 27.126347383148573, 26.93906990790639, 26.625059858433293, 25.847954031060382, 24.652873253398695, 23.92402604295922, 23.68972148120761, 23.603607875799366, 23.651273946629626, 23.80211208447502, 24.123871836189416, 24.705526807478286, 25.773934871493957, 27.842007283090176, 31.894521341933253, 34.88915733282957, 35.46046456158269, 33.60156125971644, 28.818940142021024, 16.37493478394506, 26.13286433032176, 26.130225050808825, 26.125985856580503, 26.115761987635654, 26.102626143484322, 26.07485412492277, 26.038146868902906, 25.9591987770884, 25.84795403106038, 25.525206801940726, 24.965227444195083, 24.68248072182164, 24.64202197669439, 24.658140064698518, 24.756450601138223, 24.93430566571867, 25.262441339159146, 25.826904186796696, 26.834185464506973, 28.707191817425613, 32.36806298196781, 35.05219346486298, 35.527939542578984, 33.62899929551682, 28.831275132818508, 16.37741622615349, 24.504868260671007, 24.50490372622622, 24.50352883436428, 24.504554880551247, 24.503754365175194, 24.512159881216405, 24.523141342718496, 24.571416666712825, 24.652873253398706, 24.965227444195083, 25.552713738552235, 25.905572673226846, 26.060413004453373, 26.204535320833777, 26.354927375868787, 26.566848118461607, 26.908703044203666, 27.456577376277284, 28.416886575150883, 30.002599523023605, 32.95006478000992, 35.212367276243356, 35.60949292583251, 33.654566179337905, 28.847211951797867, 16.379974191235313, 23.376248708196172, 23.378081585787186, 23.380355285011582, 23.38972628500056, 23.402559992689728, 23.439336430695032, 23.499889760017524, 23.649707040587334, 23.924026042959223, 24.682480721821634, 25.905572673226843, 26.713754278105718, 27.0722265088909, 27.325213242662173, 27.529738052474293, 27.769636662107963, 28.116994068073137, 28.644681291592416, 29.53639922669708, 30.911664905274975, 33.44593544644141, 35.38392087668072, 35.68149959562092, 33.68446914206552, 28.86014635710013, 16.38217434542954, 22.84072042977834, 22.84378381093773, 22.84964134371293, 22.86498920341311, 22.890139132243746, 22.948833062000617, 23.05040591722024, 23.274020854762234, 23.689721481207627, 24.6420219766944, 26.060413004453384, 27.072226508890882, 27.585053606593107, 27.920350995588933, 28.169780534814358, 28.428368395472795, 28.775671943680155, 29.278536364196885, 30.104906514858598, 31.387469422448785, 33.78592425654772, 35.55127059786933, 35.75121548358776, 33.717058090811314, 28.874699473872493, 16.38894350198692, 22.56627982771107, 22.57021412120461, 22.579039602824373, 22.598770193741963, 22.633676683894816, 22.70788671323578, 22.838411823533335, 23.113710036584642, 23.603607875799376, 24.658140064698514, 26.20453532083378, 27.325213242662194, 27.92035099558896, 28.314769110483986, 28.594203293991374, 28.865197935823687, 29.20953137533107, 29.691091941177824, 30.482242426833903, 31.712636857278987, 34.01784991880725, 35.68300569651862, 35.81408623419237, 33.74504638975082, 28.885548286945305, 16.390288692914538, 22.50514631055478, 22.50977332981827, 22.520617641686883, 22.543390577491923, 22.58491630389258, 22.66927239022117, 22.81894259514152, 23.12130121504639, 23.651273946629644, 24.75645060113823, 26.35492737586881, 27.52973805247431, 28.16978053481439, 28.594203293991406, 28.892419003041915, 29.168410664769247, 29.50656459226938, 29.972236919546678, 30.73710269784988, 31.937602492170562, 34.20017076439942, 35.80202254234554, 35.88079845290569, 33.7783463461203, 28.903101425833167, 16.399490872946828, 22.604718387425766, 22.610123910381624, 22.621528250601123, 22.64705649940849, 22.69120036352484, 22.78179273884584, 22.939252817085087, 23.254972244475493, 23.802112084475038, 24.934305665718682, 26.566848118461653, 27.76963666210801, 28.42836839547288, 28.865197935823744, 29.168410664769294, 29.442846799629, 29.772197311142342, 30.222715565203945, 30.965939492257153, 32.13946691765992, 34.35370692641479, 35.899525138680346, 35.93315944345745, 33.80089761907073, 28.907114031410885, 16.396307580474762, 22.920191323544355, 22.925639860561873, 22.937883806167953, 22.963785618991373, 23.009143661466968, 23.100552736381104, 23.25819656457746, 23.574641321199437, 24.123871836189462, 25.262441339159174, 26.908703044203744, 28.11699406807323, 28.77567194368028, 29.20953137533125, 29.50656459226956, 29.7721973111425, 30.087123621112838, 30.516661565009503, 31.232631747095972, 32.37478274465676, 34.54727869268485, 36.02918900480205, 36.007475490889775, 33.8398636335996, 28.93025265211517, 16.409482831070832, 23.543483514729736, 23.5490685903062, 23.561141401784138, 23.587018807371972, 23.63132416818812, 23.72013135699427, 23.872320063159272, 24.17454556953126, 24.70552680747832, 25.826904186796767, 27.456577376277366, 28.644681291592555, 29.278536364197112, 29.691091941178147, 29.97223691954708, 30.222715565204403, 30.51666156500985, 30.916287578073156, 31.591310688236973, 32.68305994536827, 34.76617589167289, 36.146672080487704, 36.05153777930549, 33.842889939059226, 28.915398794438648, 16.394404040260824, 24.71162619174795, 24.71736659351487, 24.728649335166804, 24.754375323136152, 24.795382533512747, 24.880052460492085, 25.01834669085904, 25.29780533053135, 25.773934871494014, 26.834185464507097, 28.416886575151054, 29.536399226697323, 30.104906514858985, 30.482242426834475, 30.737102697850702, 30.965939492258208, 31.232631747097155, 31.59131068823793, 32.196554420878925, 33.19636369261336, 35.14581548009019, 36.35180337155593, 36.14530328415037, 33.892652618821444, 28.948388289276025, 16.41556490569323, 26.912150873890205, 26.917702667317695, 26.928882437064864, 26.953180383664712, 26.99227818244258, 27.06947892233493, 27.19499070375837, 27.439350194467654, 27.84200728309033, 28.707191817425773, 30.002599523023854, 30.911664905275376, 31.387469422449414, 31.71263685727993, 31.93760249217195, 32.13946691766199, 32.37478274465958, 32.683059945371504, 33.19636369261598, 34.007816043120805, 35.58442201580068, 36.46717282946086, 36.135041193632745, 33.83453788636227, 28.879129753467478, 16.36764294316941, 31.17477661353448, 31.179510579341525, 31.19109073367034, 31.211540557117583, 31.248635045661054, 31.31111115236017, 31.41787191661975, 31.597300907806044, 31.89452134193345, 32.36806298196806, 32.95006478001027, 33.445935446442014, 33.78592425654867, 34.0178499188088, 34.20017076440188, 34.35370692641869, 34.547278692690895, 34.766175891681264, 35.145815480100474, 35.58442201581211, 36.35688617795719, 36.67578185543338, 36.1944421718936, 33.82595739088896, 28.87768232978707, 16.373626146803225, 34.46689685181476, 34.470994231414195, 34.48013452677869, 34.49657967500205, 34.52405488193136, 34.568266937281216, 34.637892782332955, 34.74289680169932, 34.88915733282968, 35.05219346486328, 35.21236727624392, 35.383920876681564, 35.55127059787072, 35.68300569652105, 35.80202254234965, 35.899525138687174, 36.029189004813524, 36.14667208050698, 36.35180337158888, 36.46717282951777, 36.67578185543669, 36.518534541615146, 35.86196471029214, 33.474193524278476, 28.58841183837758, 16.21507315589668, 35.25373748081989, 35.25664344986805, 35.26305077259158, 35.273939281480914, 35.291432658884105, 35.31724402383658, 35.354665016299144, 35.403179931384486, 35.460464561582874, 35.52793954257933, 35.60949292583308, 35.68149959562198, 35.75121548358955, 35.814086234195344, 35.880798452910824, 35.93315944346669, 36.00747549090606, 36.05153777933559, 36.14530328420438, 36.13504119368948, 36.194442171873305, 35.86196471024651, 35.14063645463302, 32.828256207257425, 28.12924950394769, 16.016404920819383, 33.50207127818346, 33.50387945971782, 33.5077665333937, 33.51415697607108, 33.523780555816685, 33.53695571092251, 33.55460338479761, 33.57531800923423, 33.60156125971659, 33.62899929551711, 33.65456617933847, 33.68446914206642, 33.71705809081287, 33.74504638975356, 33.778346346124934, 33.800897619078405, 33.83986363361212, 33.84288993908234, 33.892652618857994, 33.83453788637352, 33.82595739086736, 33.474193524263164, 32.82825620725327, 30.797552522501167, 26.580793612216606, 15.220975950160131, 28.774831514827692, 28.775850153795137, 28.777797655484534, 28.781199706354272, 28.78551578180204, 28.792056542061083, 28.799081817957887, 28.80896377464023, 28.81894014202112, 28.831275132818742, 28.847211951798243, 28.8601463571007, 28.874699473873427, 28.885548286946975, 28.903101425836034, 28.90711403141541, 28.93025265212143, 28.91539879444661, 28.948388289284058, 28.879129753468753, 28.877682329782406, 28.588411838373773, 28.129249503945424, 26.580793612215714, 23.404465595547176, 13.699998167298649, 16.35868369348732, 16.359025754408616, 16.35988382872867, 16.360940485380624, 16.362904639238128, 16.364644583061033, 16.3679738205701, 16.370657604501968, 16.37493478394517, 16.377416226153585, 16.379974191235405, 16.38217434542974, 16.3889435019873, 16.39028869291511, 16.39949087294772, 16.396307580476854, 16.40948283107494, 16.394404040262412, 16.41556490569253, 16.367642943169596, 16.373626146802543, 16.21507315589558, 16.016404920818584, 15.220975950159787, 13.69999816729865, 8.297291167363069]], [[12.369890313586152, 12.364873144676157, 12.366156979132455, 12.347367531767423, 12.345887446310089, 12.29412728908165, 12.25930554542009, 12.024149386418994, 11.476692619352551, 8.568522442217283, 3.6284525132632117, 2.1527649649939247, 2.487620642058023, 2.3701359496690806, 2.407395498760185, 2.4078194683442016, 2.3759283661029835, 2.544698227413379, 1.9526801109073881, 9.135150744715533, 72.93637768384329, 141.1963598639054, 164.9460501435018, 160.83956167217394, 129.79965817653422, 60.455515119322804, 12.36487314467615, 12.360471460441131, 12.360606326646923, 12.34374182443163, 12.340083938316871, 12.290907267845549, 12.254484002309738, 12.020786598787724, 11.471994194320173, 8.567333321238936, 3.631510256514928, 2.156966111189041, 2.4920518231860282, 2.3740447589426874, 2.412599078794479, 2.411343016869597, 2.3812810761600285, 2.548679621305033, 1.9562634763412399, 9.147554188723754, 72.94336884191785, 141.2107601944793, 164.96079451577313, 160.85097592235314, 129.80605415699918, 60.45766366415208, 12.366156979132466, 12.360606326646918, 12.363348545241596, 12.342934897877953, 12.344359571876065, 12.290615304554354, 12.258722073244522, 12.023037318323238, 11.478465886823653, 8.567955152937456, 3.624355898737514, 2.149117904303227, 2.4825944907275233, 2.3666398545182097, 2.4018989403256903, 2.404925373344018, 2.370234108657392, 2.5414280292508584, 1.9487927717267315, 9.12588132965296, 72.9657416429445, 141.25003652098758, 165.000317643666, 160.8816217494715, 129.8251803653785, 60.46541257931184, 12.347367531767423, 12.343741824431625, 12.34293489787795, 12.328569680265552, 12.323243886250255, 12.27801967891149, 12.240701767917313, 12.010783641381119, 11.462388697296063, 8.56305299591122, 3.635269619854402, 2.163649784090161, 2.4978011216736813, 2.380468282291591, 2.4199874445146086, 2.4167941331091027, 2.389758025919645, 2.5541746298832364, 1.9624583736746501, 9.168568444108477, 72.99638904342174, 141.30974254307125, 165.05917336476122, 160.92383234765498, 129.84833820368465, 60.47204399166038, 12.345887446310092, 12.340083938316866, 12.344359571876067, 12.323243886250257, 12.32823875584781, 12.273912015255654, 12.247508743298306, 12.013190997659088, 11.474579636851166, 8.565946335481406, 3.6171280569164055, 2.1422482648383685, 2.476014360740304, 2.360596649805674, 2.3945709807562303, 2.4010343714902316, 2.3612198210585813, 2.5382851996698808, 1.9416620032101253, 9.11715430982443, 73.0650514308062, 141.42767250850147, 165.1702432377115, 161.00429046935335, 129.89192361425623, 60.490552567612845, 12.294127289081644, 12.290907267845549, 12.290615304554352, 12.278019678911493, 12.273912015255656, 12.23381366451701, 12.198869228066036, 11.978529154833197, 11.43677257641776, 8.548348704723463, 3.6442687284700157, 2.1789981284147477, 2.5106630147420055, 2.3969621389778837, 2.4360538755599053, 2.431142490956695, 2.4097476506638937, 2.5661499496449265, 1.9794162167265963, 9.2194902241609, 73.16299229754887, 141.60730366844544, 165.3306171096681, 161.10084650674133, 129.9418876322933, 60.49945730164334, 12.25930554542008, 12.25448400230974, 12.258722073244519, 12.240701767917313, 12.247508743298306, 12.198869228066043, 12.184072268804451, 11.958334315179131, 11.441976067895903, 8.554542563286164, 3.59454399381236, 2.117538596358821, 2.45926230876148, 2.3391064388548615, 2.377251309492445, 2.3843037772720335, 2.338209498256745, 2.5276936798695906, 1.9172351619415995, 9.09444955119432, 73.37707438333697, 141.94491988318737, 165.60230903321448, 161.2697025847123, 130.00696895917525, 60.53648106918524, 12.024149386418992, 12.020786598787723, 12.023037318323237, 12.010783641381117, 12.013190997659091, 11.978529154833195, 11.958334315179133, 11.783108384618634, 11.273552814078869, 8.456182639054179, 3.6811540270085037, 2.2574867813882147, 2.5780891612899115, 2.476567434041656, 2.509710216513742, 2.5104100688612117, 2.4915351549134632, 2.6324942041281085, 2.06228985242616, 9.437436533932322, 73.72774238092192, 142.5177517655005, 166.0185224080462, 161.40814207247425, 130.0957981361246, 60.54661094160695, 11.476692619352551, 11.471994194320166, 11.478465886823653, 11.462388697296054, 11.474579636851166, 11.436772576417763, 11.4419760678959, 11.273552814078869, 10.979070717888781, 8.2946232716996, 3.3793164851396194, 1.8981337036699217, 2.246915042458793, 2.120846373993022, 2.1714290212532976, 2.161774862592674, 2.1300997784639715, 2.328814675624548, 1.699515337318644, 8.765155882772456, 74.64778857785092, 143.7634848986063, 166.44252112877234, 161.67213874877555, 130.1912940551578, 60.59452965214015, 8.568522442217283, 8.567333321238927, 8.567955152937452, 8.563052995911226, 8.565946335481401, 8.548348704723464, 8.554542563286164, 8.456182639054184, 8.2946232716996, 6.904395703783522, 4.045083668051531, 3.3938710833530026, 3.675390156050995, 3.623733252413315, 3.630297964516144, 3.690248936726017, 3.5937647735922913, 3.975780034949604, 3.3100116247262767, 12.558457329457566, 77.965910716985, 145.10082309966444, 166.94496011654917, 161.92441335772273, 130.27067193696197, 60.613520826338004, 3.6284525132632126, 3.631510256514927, 3.624355898737514, 3.6352696198544012, 3.6171280569164055, 3.6442687284700166, 3.5945439938123607, 3.6811540270085032, 3.37931648513962, 4.045083668051527, 6.878508722843541, 8.315031662994077, 8.448719931550059, 8.584907479162807, 8.56048549342724, 8.650530961220754, 8.681750412815374, 9.113341804381136, 10.641508710864322, 24.3672881098951, 83.56364918947882, 146.19768410751803, 167.74098876768542, 162.13029461696038, 130.47683205035713, 60.58721792312208, 2.1527649649939247, 2.1569661111890417, 2.1491179043032274, 2.1636497840901616, 2.1422482648383685, 2.178998128414747, 2.1175385963588202, 2.2574867813882147, 1.8981337036699226, 3.3938710833530013, 8.315031662994079, 10.988155057269942, 11.305913480294379, 11.47289505263426, 11.50199055809085, 11.578538551218958, 11.709469549594383, 12.3293006733, 15.268856896402887, 30.64305885693586, 87.75156919700453, 147.5781434857874, 168.32297781378645, 162.40390368102734, 130.57553111390698, 60.602611350880636, 2.4876206420580216, 2.492051823186027, 2.482594490727524, 2.4978011216736795, 2.476014360740303, 2.510663014742006, 2.45926230876148, 2.5780891612899106, 2.246915042458793, 3.6753901560510003, 8.448719931550063, 11.30591348029439, 11.799171326748587, 12.008217308932233, 12.028399643114856, 12.146292195272714, 12.270844506702959, 13.092711163975517, 16.123382769831167, 31.53720045276773, 89.61134028862696, 149.07411832533487, 168.9106609649752, 162.76791683537584, 130.73584008907193, 60.669046123079085, 2.370135949669081, 2.3740447589426874, 2.3666398545182123, 2.3804682822915897, 2.360596649805672, 2.3969621389778837, 2.3391064388548606, 2.476567434041658, 2.1208463739930266, 3.6237332524133206, 8.584907479162819, 11.472895052634275, 12.008217308932242, 12.241710109458417, 12.281742939185087, 12.393902284847407, 12.553389050848368, 13.395205973824295, 16.46791145528106, 32.200076067995155, 90.44609985553016, 150.02941524125856, 169.51199562196874, 163.03031138891555, 130.86704206370226, 60.69187746481568, 2.4073954987601835, 2.4125990787944795, 2.4018989403256903, 2.4199874445146095, 2.39457098075623, 2.4360538755599057, 2.377251309492445, 2.509710216513742, 2.1714290212532994, 3.630297964516154, 8.56048549342725, 11.501990558090858, 12.028399643114863, 12.28174293918509, 12.32176646285103, 12.439984278513288, 12.60658713737368, 13.438487332534466, 16.552974685239096, 32.30890270967381, 91.10798616902208, 150.91782954534332, 170.20330741276226, 163.45363777443566, 131.07280034632242, 60.77715936459038, 2.407819468344201, 2.411343016869597, 2.404925373344019, 2.416794133109105, 2.4010343714902347, 2.431142490956696, 2.384303777272034, 2.5104100688612134, 2.1617748625926816, 3.690248936726026, 8.650530961220769, 11.578538551218971, 12.14629219527273, 12.393902284847423, 12.439984278513304, 12.569484450909345, 12.720928236250863, 13.59010534999974, 16.723468940407436, 32.68936151216527, 91.7303082598234, 151.7973268537439, 170.89883347095736, 163.8112831265997, 131.21060034543356, 60.7953153371495, 2.3759283661029844, 2.381281076160029, 2.370234108657393, 2.3897580259196474, 2.3612198210585813, 2.4097476506639, 2.338209498256745, 2.4915351549134748, 2.1300997784639826, 3.5937647735923073, 8.681750412815372, 11.709469549594425, 12.270844506702968, 12.553389050848438, 12.606587137373706, 12.720928236250954, 12.895097726296877, 13.751657524225092, 16.945831235305203, 33.06397752815856, 92.93416154727305, 153.33679407998224, 171.98259159462836, 164.39238956638093, 131.45136555723883, 60.898552290057786, 2.5446982274133787, 2.5486796213050313, 2.5414280292508598, 2.554174629883238, 2.5382851996698816, 2.5661499496449327, 2.5276936798695955, 2.6324942041281227, 2.328814675624572, 3.9757800349496315, 9.113341804381148, 12.329300673300072, 13.09271116397559, 13.395205973824464, 13.438487332534601, 13.590105349999945, 13.751657524225179, 14.653634752073948, 17.936063449101844, 34.54818485792992, 95.21730254127814, 155.7491962155883, 173.31082393917907, 164.7863585920998, 131.5718312319789, 60.8913442737557, 1.9526801109073857, 1.9562634763412374, 1.9487927717267293, 1.9624583736746475, 1.9416620032101215, 1.9794162167265918, 1.9172351619415964, 2.06228985242616, 1.699515337318653, 3.3100116247263025, 10.641508710864418, 15.268856896403026, 16.123382769831423, 16.46791145528135, 16.5529746852397, 16.723468940408214, 16.94583123530617, 17.936063449102623, 21.268237973685327, 38.478862469614384, 101.0283256319194, 160.3955012361822, 174.95509029790236, 165.46260758956126, 131.8227394213867, 60.97309986639041, 9.135150744715524, 9.147554188723753, 9.125881329652943, 9.168568444108473, 9.117154309824446, 9.219490224160893, 9.094449551194295, 9.437436533932326, 8.765155882772529, 12.558457329457651, 24.367288109895366, 30.643058856936328, 31.537200452768516, 32.200076067996285, 32.308902709675415, 32.6893615121673, 33.06397752816131, 34.548184857932945, 38.478862469616814, 56.61662225713624, 115.0745254250955, 165.8836551730889, 176.34079052512783, 165.86724196868593, 131.678732288018, 60.89904643882202, 72.93637768384332, 72.94336884191783, 72.96574164294451, 72.99638904342186, 73.06505143080636, 73.16299229754902, 73.37707438333719, 73.72774238092234, 74.6477885778515, 77.96591071698577, 83.56364918948006, 87.75156919700643, 89.61134028862983, 90.44609985553461, 91.10798616902841, 91.7303082598322, 92.93416154728536, 95.21730254129314, 101.02832563193483, 115.07452542510654, 145.19925875727307, 172.53272131166437, 178.69412552016072, 165.92833398991937, 131.59709949918985, 60.66254000768449, 141.19635986390549, 141.2107601944794, 141.25003652098783, 141.3097425430715, 141.4276725085017, 141.60730366844587, 141.9449198831879, 142.5177517655012, 143.76348489860737, 145.10082309966594, 146.19768410752042, 147.57814348579103, 149.0741183253405, 150.0294152412669, 150.91782954535608, 151.7973268537623, 153.3367940800065, 155.7491962156191, 160.3955012362172, 165.88365517311513, 172.5327213116724, 178.96077141017955, 178.79490902922774, 164.3160395577438, 129.97518997879894, 59.899989725500895, 164.94605014350185, 164.96079451577342, 165.0003176436663, 165.05917336476142, 165.17024323771187, 165.33061710966845, 165.60230903321496, 166.01852240804726, 166.4425211287736, 166.94496011655096, 167.74098876768852, 168.32297781379123, 168.91066096498224, 169.51199562197965, 170.2033074127775, 170.89883347097864, 171.98259159465863, 173.31082393921614, 174.95509029793882, 176.34079052515145, 178.69412552016658, 178.79490902922376, 174.13636418927734, 159.2259876673652, 126.08380656016006, 58.31803850118654, 160.8395616721742, 160.85097592235326, 160.88162174947158, 160.92383234765506, 161.00429046935344, 161.10084650674148, 161.2697025847127, 161.40814207247485, 161.67213874877672, 161.92441335772452, 162.13029461696317, 162.40390368103172, 162.7679168353825, 163.0303113889255, 163.45363777445013, 163.81128312661977, 164.39238956640506, 164.78635859212716, 165.4626075895887, 165.86724196870077, 165.92833398991806, 164.3160395577362, 159.22598766735965, 145.62765046460217, 116.31619112592662, 54.1738885392151, 129.79965817653436, 129.80605415699944, 129.8251803653786, 129.84833820368482, 129.8919236142564, 129.9418876322936, 130.00696895917557, 130.0957981361252, 130.19129405515883, 130.2706719369633, 130.47683205035898, 130.57553111390993, 130.73584008907662, 130.86704206370885, 131.07280034633283, 131.21060034544837, 131.45136555725645, 131.57183123199698, 131.8227394214, 131.6787322880223, 131.59709949918619, 129.97518997879297, 126.08380656015595, 116.31619112592485, 94.9332519106587, 45.38568746546574, 60.45551511932288, 60.45766366415219, 60.46541257931192, 60.47204399166042, 60.49055256761299, 60.499457301643446, 60.536481069185164, 60.546610941607156, 60.59452965214073, 60.61352082633858, 60.58721792312288, 60.60261135088185, 60.66904612308078, 60.6918774648187, 60.77715936459435, 60.79531533715496, 60.89855229007348, 60.89134427377049, 60.97309986638564, 60.89904643881365, 60.66254000768312, 59.89998972549875, 58.31803850118478, 54.173888539214545, 45.38568746546551, 22.480953350583764]]]

        import matplotlib.pyplot as plt
        plt.imshow(np.array(pydgm.state.phi).flatten().reshape(2, 26, 26)[0], interpolation=None)
        plt.colorbar()
        plt.show()

        phi_test = np.array(phi_test)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()
            phi_zero_test = phi_test[:, l].flatten()
            np.testing.assert_array_almost_equal(phi, phi_zero_test, 12)
        # np.testing.assert_array_almost_equal(phi_one, phi_one_test, 12)

        self.angular_test()

    def test_solver_partisn_fixed_2g_l2(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.scatter_leg_order = 2

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[28.784977585739572, 28.708220487947383, 28.42222476431022, 27.895515587479483, 26.421282987065076, 24.331430463680633, 23.027556932446792, 23.105019480938637, 23.656022133865218, 26.540594788731724, 30.673871262072065, 31.33324558882314, 19.758182663959243, 28.708220487947465, 28.639585865932887, 28.366725917647912, 27.863252738092747, 26.414758592809367, 24.341240899203843, 23.05670853891712, 23.149273857999184, 23.7037007456598, 26.590136525669116, 30.723591639630584, 31.36819962045508, 19.773810535897162, 28.422224764310396, 28.36672591764795, 28.134782215838978, 27.689980321779952, 26.324763862981726, 24.362139637321928, 23.149682345588896, 23.27626397189877, 23.846239234570557, 26.722890160253826, 30.81752847461152, 31.412880453391335, 19.780707524113026, 27.895515587479444, 27.863252738093163, 27.689980321780286, 27.402657145726412, 26.248694254746887, 24.463840134684446, 23.439729329862267, 23.68834114221326, 24.24522540854259, 27.06877985180523, 31.090461915536316, 31.549040524520283, 19.819862872520485, 26.421282987064195, 26.41475859281179, 26.324763862984067, 26.248694254746056, 25.72797910941834, 24.90239415175772, 24.49387103149588, 24.92146991132457, 25.491960001095784, 28.060250221485973, 31.54689138514408, 31.624803165031718, 19.82698429221748, 24.331430463679716, 24.341240899206262, 24.362139637324482, 24.463840134683302, 24.902394151755363, 25.986098649844077, 26.54273466979216, 27.123264832120615, 27.784000479447794, 29.917226666583346, 32.273822094018506, 31.780304082575746, 19.922780426132228, 23.02755693244642, 23.05670853891746, 23.14968234558926, 23.439729329862466, 24.493871031495942, 26.542734669793308, 27.72322226813913, 28.481595310359527, 29.13943836242923, 30.980070240945086, 32.754550664374094, 31.842351913185023, 19.910272630117717, 23.10501948093798, 23.149273857999056, 23.276263971898416, 23.688341142214014, 24.921469911325246, 27.12326483212618, 28.481595310362867, 29.336905200120903, 29.932890547363314, 31.654342311224465, 33.243012450824, 32.10527939266111, 20.014847845448617, 23.65602213386453, 23.703700745659976, 23.84623923457076, 24.24522540854502, 25.491960001098292, 27.78400047945804, 29.139438362451653, 29.932890547390294, 30.441208981960525, 32.02125722855956, 33.42545615816064, 32.07019451004885, 19.91205673952471, 26.54059478873172, 26.590136525669493, 26.722890160254234, 27.068779851808834, 28.060250221490435, 29.91722666659383, 30.98007024098355, 31.654342311299764, 32.02125722864927, 33.27511221448896, 34.10467049660892, 32.35159468603449, 20.06810654939191, 30.673871262073632, 30.723591639631252, 30.8175284746111, 31.090461915538057, 31.546891385147603, 32.273822094012885, 32.754550664354305, 33.24301245083822, 33.425456158206465, 34.10467049656568, 33.80899597539237, 31.540854976862576, 19.54719649306038, 31.333245588824642, 31.368199620456974, 31.412880453395683, 31.549040524526102, 31.624803165044803, 31.780304082587968, 31.84235191319066, 32.105279392611614, 32.07019450996933, 32.351594685925946, 31.54085497675706, 29.409673305312214, 18.496069607954336, 19.758182663959282, 19.773810535899003, 19.78070752411926, 19.819862872528667, 19.82698429223335, 19.922780426167122, 19.910272630216678, 20.014847845451406, 19.912056739475293, 20.068106549384208, 19.547196492995507, 18.496069607893638, 11.936146269156133]], [[12.323255317277246, 12.336489156875425, 12.223354682922217, 12.348986198899304, 9.329222006110882, 3.2876685534678605, 2.0154966156997483, 2.7500682166444315, 2.0287309150541026, 6.329890348202532, 87.39529975792667, 147.84219022821077, 78.8886285607443, 12.336489156888067, 12.359890230126775, 12.23224927876285, 12.379610677756354, 9.360535389521191, 3.222871824040389, 1.9610794479951181, 2.6769664699816693, 1.9748850152611162, 6.16991172046771, 87.47842947330756, 147.99790374345477, 78.92479255683237, 12.223354682974776, 12.232249278808139, 12.150650338690056, 12.253034740046823, 9.267922427196696, 3.3890073711076356, 2.1109517029655644, 2.8774196125634366, 2.1218856768328553, 6.615059231425866, 87.69404782655972, 148.1893126279388, 79.13254554417203, 12.348986198869225, 12.3796106777329, 12.253034739978274, 12.533884490028395, 9.453465581279644, 3.021700048993916, 1.7940055756045905, 2.4318597924803846, 1.8135844687519511, 5.66511587873998, 88.11052180503894, 149.20189620840017, 79.18470285895869, 9.32922200603733, 9.36053538945203, 9.267922427099176, 9.453465581219403, 7.701338125546113, 3.6876991065672, 2.824424830912647, 3.5530926856644247, 2.669437009486237, 8.235823885307008, 91.13347017354407, 150.0935141191865, 79.26059758347029, 3.2876685534811743, 3.222871824048407, 3.389007371120185, 3.0217000490009154, 3.6876991065686378, 8.038525490044723, 9.495481671355698, 9.880081399732939, 8.60323849374538, 21.840158925774663, 97.42650378094032, 150.4484911438512, 79.98842191813162, 2.015496615698944, 1.9610794479920168, 2.110951702972827, 1.7940055756057458, 2.8244248309039115, 9.495481671347086, 12.33817068664656, 12.414469395561763, 11.537223569348107, 27.321151639246633, 101.58087440398027, 151.42473712320276, 80.22848054758133, 2.7500682165946855, 2.6769664699385474, 2.87741961252823, 2.431859792447225, 3.5530926856031826, 9.8800813996211, 12.41446939545051, 12.853769119120162, 11.92623649386442, 27.752036399894497, 103.4262174810234, 153.37143086329803, 80.83187881947153, 2.0287309150384347, 1.974885015245942, 2.1218856768261043, 1.813584468742099, 2.669437009462587, 8.603238493753619, 11.537223569416934, 11.92623649401948, 10.762235618418876, 27.24596306149478, 106.07846492485392, 156.2195386543766, 80.81531798424577, 6.329890348233633, 6.169911720487799, 6.615059231472701, 5.665115878767688, 8.235823885322489, 21.8401589259183, 27.321151639459025, 27.752036400162282, 27.245963061649356, 45.30625965488264, 119.6617677226004, 159.15467056515024, 80.72924363856644, 87.3952997579439, 87.47842947331834, 87.69404782659797, 88.11052180507099, 91.13347017355257, 97.4265037809724, 101.5808744039733, 103.42621748104206, 106.07846492492108, 119.66176772256213, 149.4280435806511, 156.63037090355897, 80.02936060668367, 147.8421902282202, 147.9979037434727, 148.18931262795763, 149.20189620842802, 150.09351411922398, 150.44849114387247, 151.42473712316553, 153.3714308631397, 156.21953865400855, 159.15467056509058, 156.63037090371455, 143.40874961905052, 73.85922632215653, 78.88862856074876, 78.92479255684225, 79.13254554420025, 79.18470285897051, 79.26059758347976, 79.98842191816141, 80.22848054760227, 80.83187881931114, 80.81531798379939, 80.72924363858615, 80.02936060694185, 73.85922632210399, 38.37581180877973]]]

        phi_test = np.array(phi_test)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()
            phi_zero_test = phi_test[:, l].flatten()
            np.testing.assert_array_almost_equal(phi, phi_zero_test, 12)
        # np.testing.assert_array_almost_equal(phi_one, phi_one_test, 12)

        self.angular_test()


if __name__ == '__main__':

    unittest.main()
