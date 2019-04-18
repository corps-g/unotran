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
        pydgm.control.angle_order = 8
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
        pydgm.control.max_outer_iters = 1

    def angular_test(self):
        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_groups, pydgm.control.number_cells))
        for o in range(4):
            for c in range(pydgm.control.number_cells):
                for a in range(nAngles):
                    phi_test[:, c] += pydgm.angle.wt[a] * pydgm.state.psi[:, o * nAngles + a, c]
        np.testing.assert_array_almost_equal(phi_test, pydgm.state.phi[0, :, :], 12)

    def test_solver_basic_2D_1g_reflect(self):
        '''
        Test for a basic 1 group problem
        '''
        pydgm.control.fine_mesh_x = [10]
        pydgm.control.fine_mesh_y = [10]
        pydgm.control.coarse_mesh_x = [0.0, 100000.0]
        pydgm.control.coarse_mesh_y = [0.0, 100000.0]
        pydgm.control.material_map = [2]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 1.0
        pydgm.control.boundary_west = 1.0
        pydgm.control.boundary_north = 1.0
        pydgm.control.boundary_south = 1.0
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
        phi_test = [5] * 100

        phi_test = np.array(phi_test)

        # Test the scalar flux
        phi = pydgm.state.mg_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi, phi_test, 8)

        self.angular_test()

    def test_solver_basic_2D_1g_1a_vacuum(self):
        '''
        Test for a basic 1 group problem
        '''
        pydgm.control.fine_mesh_x = [25]
        pydgm.control.fine_mesh_y = [25]
        pydgm.control.coarse_mesh_x = [0.0, 10.0]
        pydgm.control.coarse_mesh_y = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/1gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
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
        phi_test = [[[0.5128609747568281, 0.6773167712139831, 0.7428920389440171, 0.7815532843645795, 0.8017050988344507, 0.8129183924456294, 0.8186192333777655, 0.8218872988327544, 0.8235294558562516, 0.8244499117457011, 0.8249136380974397, 0.8251380055566573, 0.8252036667060597, 0.8251380055566576, 0.8249136380974401, 0.8244499117457017, 0.8235294558562521, 0.8218872988327554, 0.8186192333777662, 0.81291839244563, 0.8017050988344515, 0.7815532843645803, 0.7428920389440175, 0.6773167712139835, 0.5128609747568282, 0.6773167712139831, 0.9191522109956556, 1.0101279836265808, 1.046811188269745, 1.0706509901493697, 1.082306179953408, 1.0890982284197688, 1.092393940905045, 1.0943710244790774, 1.095322191522249, 1.0958571613995292, 1.0961001761636988, 1.0961736115735345, 1.0961001761636997, 1.0958571613995298, 1.0953221915222502, 1.094371024479079, 1.092393940905046, 1.08909822841977, 1.082306179953409, 1.0706509901493706, 1.046811188269746, 1.0101279836265817, 0.9191522109956564, 0.6773167712139837, 0.742892038944017, 1.010127983626581, 1.1482625986461896, 1.1962723842001344, 1.2162573092331372, 1.2309663391416374, 1.2373535288720443, 1.2414786838078147, 1.2432965986450648, 1.2444791104681647, 1.24499985971512, 1.2452803731491178, 1.2453559925876914, 1.2452803731491184, 1.244999859715121, 1.244479110468166, 1.2432965986450666, 1.2414786838078167, 1.237353528872046, 1.230966339141639, 1.2162573092331381, 1.1962723842001353, 1.1482625986461907, 1.010127983626582, 0.7428920389440178, 0.7815532843645796, 1.046811188269745, 1.196272384200134, 1.2752422132283252, 1.2995505587401641, 1.310409682796424, 1.3196712848467558, 1.3229121459817077, 1.3255006837110048, 1.3264359863630988, 1.3271401701539467, 1.3273900240538614, 1.3274869234142137, 1.327390024053862, 1.3271401701539483, 1.3264359863631008, 1.3255006837110068, 1.3229121459817097, 1.3196712848467576, 1.3104096827964256, 1.2995505587401661, 1.275242213228327, 1.1962723842001357, 1.0468111882697462, 0.7815532843645802, 0.8017050988344504, 1.0706509901493695, 1.2162573092331366, 1.2995505587401643, 1.3452497209031955, 1.3567411771030033, 1.3627115460270347, 1.3687134641761085, 1.3700956107492908, 1.3718153695089819, 1.3722188033283726, 1.3726261825701194, 1.3726746849086748, 1.3726261825701203, 1.3722188033283742, 1.3718153695089843, 1.3700956107492925, 1.3687134641761105, 1.3627115460270367, 1.356741177103005, 1.3452497209031975, 1.2995505587401661, 1.2162573092331384, 1.0706509901493706, 0.8017050988344514, 0.8129183924456289, 1.0823061799534075, 1.2309663391416374, 1.3104096827964238, 1.3567411771030033, 1.3837407849220562, 1.3883940252403495, 1.391781180853413, 1.3958074539475693, 1.3960964469643635, 1.3973279212330587, 1.3973887886413996, 1.3976146226859119, 1.3973887886414005, 1.3973279212330607, 1.3960964469643657, 1.3958074539475707, 1.3917811808534148, 1.3883940252403517, 1.3837407849220587, 1.3567411771030051, 1.3104096827964258, 1.2309663391416386, 1.0823061799534086, 0.81291839244563, 0.8186192333777653, 1.0890982284197686, 1.2373535288720436, 1.319671284846756, 1.3627115460270351, 1.3883940252403497, 1.4048482432271674, 1.4059080588108543, 1.4079482979581446, 1.4107567935223844, 1.4103944840640301, 1.4113681192038354, 1.4110951509099934, 1.411368119203836, 1.4103944840640317, 1.4107567935223853, 1.4079482979581461, 1.4059080588108563, 1.4048482432271694, 1.3883940252403517, 1.362711546027037, 1.3196712848467576, 1.237353528872046, 1.0890982284197699, 0.8186192333777663, 0.8218872988327544, 1.092393940905045, 1.2414786838078145, 1.322912145981707, 1.3687134641761085, 1.3917811808534124, 1.4059080588108548, 1.4163858413766799, 1.4155936684368302, 1.4169325249341456, 1.4189994578626368, 1.4181484360902454, 1.4191873654776683, 1.4181484360902463, 1.4189994578626375, 1.4169325249341471, 1.415593668436832, 1.416385841376682, 1.4059080588108568, 1.3917811808534146, 1.3687134641761105, 1.3229121459817095, 1.2414786838078165, 1.092393940905047, 0.8218872988327552, 0.8235294558562515, 1.0943710244790772, 1.2432965986450644, 1.325500683711004, 1.3700956107492905, 1.3958074539475689, 1.4079482979581444, 1.4155936684368298, 1.4226595845322698, 1.4209588928818724, 1.421835104078357, 1.423671773688655, 1.4219795630324892, 1.423671773688656, 1.4218351040783586, 1.420958892881874, 1.4226595845322718, 1.4155936684368327, 1.4079482979581466, 1.3958074539475707, 1.3700956107492928, 1.3255006837110068, 1.2432965986450666, 1.0943710244790792, 0.8235294558562524, 0.8244499117457008, 1.0953221915222489, 1.2444791104681645, 1.3264359863630988, 1.3718153695089823, 1.396096446964363, 1.4107567935223833, 1.4169325249341451, 1.4209588928818722, 1.4259745842172793, 1.424082580661134, 1.4241542494391093, 1.4264775572959318, 1.42415424943911, 1.4240825806611355, 1.425974584217281, 1.420958892881874, 1.4169325249341471, 1.4107567935223857, 1.3960964469643653, 1.3718153695089845, 1.3264359863631006, 1.2444791104681667, 1.0953221915222502, 0.824449911745702, 0.8249136380974391, 1.0958571613995285, 1.2449998597151197, 1.3271401701539467, 1.3722188033283722, 1.3973279212330585, 1.4103944840640301, 1.4189994578626361, 1.4218351040783568, 1.4240825806611341, 1.427381093837987, 1.426071570295947, 1.4256483954611119, 1.4260715702959486, 1.4273810938379876, 1.424082580661136, 1.4218351040783581, 1.4189994578626381, 1.410394484064032, 1.3973279212330605, 1.3722188033283746, 1.3271401701539485, 1.2449998597151215, 1.0958571613995303, 0.8249136380974402, 0.8251380055566571, 1.0961001761636986, 1.2452803731491175, 1.3273900240538605, 1.3726261825701191, 1.3973887886413991, 1.4113681192038343, 1.4181484360902443, 1.4236717736886548, 1.4241542494391093, 1.4260715702959472, 1.4284214388119818, 1.425280452566786, 1.428421438811983, 1.426071570295949, 1.4241542494391097, 1.4236717736886555, 1.4181484360902465, 1.4113681192038365, 1.3973887886414016, 1.3726261825701216, 1.3273900240538627, 1.2452803731491189, 1.0961001761637001, 0.825138005556658, 0.8252036667060596, 1.0961736115735343, 1.2453559925876905, 1.3274869234142128, 1.3726746849086742, 1.3976146226859112, 1.4110951509099916, 1.4191873654776679, 1.421979563032488, 1.4264775572959316, 1.4256483954611114, 1.4252804525667855, 1.4309739984926175, 1.4252804525667866, 1.4256483954611128, 1.4264775572959327, 1.4219795630324894, 1.4191873654776692, 1.4110951509099945, 1.3976146226859132, 1.3726746849086764, 1.327486923414215, 1.2453559925876925, 1.0961736115735354, 0.8252036667060605, 0.825138005556657, 1.0961001761636986, 1.2452803731491173, 1.3273900240538608, 1.372626182570119, 1.3973887886413987, 1.4113681192038343, 1.4181484360902448, 1.4236717736886546, 1.4241542494391095, 1.426071570295948, 1.428421438811982, 1.4252804525667868, 1.4284214388119838, 1.426071570295949, 1.4241542494391108, 1.4236717736886564, 1.418148436090247, 1.411368119203837, 1.3973887886414011, 1.3726261825701218, 1.3273900240538627, 1.2452803731491195, 1.0961001761637001, 0.8251380055566581, 0.8249136380974397, 1.095857161399529, 1.24499985971512, 1.3271401701539465, 1.3722188033283722, 1.3973279212330587, 1.4103944840640301, 1.418999457862636, 1.4218351040783574, 1.4240825806611341, 1.427381093837987, 1.4260715702959483, 1.4256483954611125, 1.4260715702959488, 1.4273810938379878, 1.4240825806611361, 1.421835104078359, 1.4189994578626386, 1.4103944840640321, 1.3973279212330607, 1.3722188033283749, 1.3271401701539487, 1.2449998597151215, 1.0958571613995305, 0.8249136380974409, 0.8244499117457011, 1.0953221915222489, 1.2444791104681645, 1.3264359863630988, 1.3718153695089828, 1.3960964469643642, 1.410756793522384, 1.4169325249341456, 1.420958892881872, 1.4259745842172795, 1.4240825806611346, 1.4241542494391095, 1.426477557295932, 1.4241542494391102, 1.4240825806611361, 1.4259745842172817, 1.4209588928818744, 1.4169325249341482, 1.4107567935223861, 1.3960964469643657, 1.371815369508985, 1.326435986363101, 1.2444791104681663, 1.0953221915222509, 0.8244499117457021, 0.8235294558562516, 1.0943710244790774, 1.2432965986450648, 1.325500683711005, 1.3700956107492908, 1.3958074539475689, 1.4079482979581448, 1.4155936684368302, 1.4226595845322698, 1.4209588928818724, 1.421835104078357, 1.423671773688655, 1.4219795630324894, 1.4236717736886562, 1.4218351040783583, 1.420958892881874, 1.422659584532272, 1.4155936684368329, 1.4079482979581472, 1.3958074539475713, 1.3700956107492932, 1.3255006837110068, 1.2432965986450673, 1.0943710244790794, 0.8235294558562526, 0.8218872988327546, 1.0923939409050452, 1.241478683807815, 1.3229121459817077, 1.368713464176108, 1.3917811808534133, 1.4059080588108555, 1.4163858413766814, 1.4155936684368304, 1.4169325249341456, 1.4189994578626364, 1.4181484360902452, 1.4191873654776685, 1.4181484360902465, 1.4189994578626381, 1.4169325249341473, 1.4155936684368324, 1.4163858413766832, 1.4059080588108572, 1.3917811808534155, 1.3687134641761114, 1.32291214598171, 1.2414786838078167, 1.0923939409050465, 0.8218872988327557, 0.8186192333777654, 1.089098228419769, 1.2373535288720448, 1.3196712848467562, 1.3627115460270354, 1.3883940252403495, 1.4048482432271683, 1.4059080588108557, 1.4079482979581455, 1.410756793522384, 1.4103944840640303, 1.4113681192038354, 1.411095150909993, 1.411368119203836, 1.410394484064032, 1.410756793522386, 1.4079482979581472, 1.4059080588108572, 1.4048482432271705, 1.388394025240352, 1.3627115460270374, 1.3196712848467584, 1.237353528872046, 1.0890982284197708, 0.8186192333777667, 0.8129183924456292, 1.0823061799534082, 1.2309663391416377, 1.3104096827964242, 1.3567411771030033, 1.3837407849220569, 1.3883940252403497, 1.391781180853413, 1.3958074539475698, 1.3960964469643644, 1.3973279212330594, 1.3973887886413998, 1.397614622685912, 1.3973887886414005, 1.3973279212330607, 1.396096446964366, 1.3958074539475713, 1.391781180853415, 1.388394025240352, 1.3837407849220584, 1.3567411771030051, 1.310409682796426, 1.2309663391416392, 1.0823061799534093, 0.8129183924456299, 0.8017050988344508, 1.07065099014937, 1.2162573092331375, 1.2995505587401643, 1.3452497209031953, 1.3567411771030036, 1.3627115460270354, 1.3687134641761085, 1.370095610749291, 1.3718153695089825, 1.3722188033283735, 1.3726261825701203, 1.372674684908675, 1.3726261825701211, 1.3722188033283742, 1.371815369508985, 1.3700956107492934, 1.3687134641761116, 1.3627115460270374, 1.3567411771030051, 1.3452497209031975, 1.2995505587401661, 1.2162573092331384, 1.0706509901493708, 0.8017050988344514, 0.7815532843645797, 1.046811188269745, 1.1962723842001342, 1.2752422132283254, 1.2995505587401648, 1.3104096827964244, 1.3196712848467562, 1.322912145981708, 1.3255006837110048, 1.3264359863630992, 1.3271401701539476, 1.3273900240538619, 1.3274869234142137, 1.327390024053862, 1.3271401701539485, 1.3264359863631006, 1.3255006837110068, 1.3229121459817104, 1.3196712848467584, 1.3104096827964262, 1.2995505587401661, 1.2752422132283268, 1.1962723842001353, 1.0468111882697464, 0.7815532843645804, 0.742892038944017, 1.0101279836265813, 1.1482625986461896, 1.196272384200134, 1.2162573092331368, 1.2309663391416377, 1.2373535288720443, 1.241478683807815, 1.2432965986450653, 1.2444791104681654, 1.2449998597151208, 1.245280373149118, 1.2453559925876916, 1.245280373149119, 1.2449998597151217, 1.2444791104681663, 1.2432965986450666, 1.2414786838078173, 1.2373535288720463, 1.230966339141639, 1.2162573092331386, 1.1962723842001357, 1.1482625986461912, 1.0101279836265824, 0.7428920389440179, 0.677316771213983, 0.9191522109956558, 1.010127983626581, 1.0468111882697448, 1.0706509901493695, 1.0823061799534077, 1.0890982284197688, 1.0923939409050452, 1.0943710244790776, 1.0953221915222493, 1.0958571613995294, 1.096100176163699, 1.0961736115735348, 1.0961001761636995, 1.09585716139953, 1.0953221915222504, 1.0943710244790792, 1.092393940905047, 1.0890982284197708, 1.0823061799534093, 1.0706509901493708, 1.0468111882697464, 1.010127983626582, 0.9191522109956568, 0.6773167712139838, 0.5128609747568279, 0.6773167712139831, 0.7428920389440171, 0.7815532843645797, 0.8017050988344507, 0.8129183924456291, 0.8186192333777655, 0.8218872988327544, 0.8235294558562514, 0.8244499117457011, 0.8249136380974398, 0.8251380055566575, 0.82520366670606, 0.8251380055566576, 0.8249136380974401, 0.8244499117457018, 0.8235294558562525, 0.8218872988327557, 0.8186192333777665, 0.8129183924456304, 0.8017050988344514, 0.7815532843645805, 0.7428920389440181, 0.6773167712139839, 0.5128609747568283]]]

        phi_test = np.array(phi_test)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()
            phi_zero_test = phi_test[:, l].flatten()
            np.testing.assert_array_almost_equal(phi, phi_zero_test, 8)

        self.angular_test()

    def test_solver_basic_2D_2g_1a_vacuum(self):
        '''
        Test for a basic 2 group problem
        '''
        pydgm.control.fine_mesh_x = [25]
        pydgm.control.fine_mesh_y = [25]
        pydgm.control.coarse_mesh_x = [0.0, 10.0]
        pydgm.control.coarse_mesh_y = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.angle_order = 2
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

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[2.1137582766948073, 2.636581917488039, 3.026045587155106, 3.371616718556787, 3.6667806259373976, 3.918852544828727, 4.129290293281016, 4.301839388015012, 4.43913453975634, 4.543496011917352, 4.616741622074667, 4.660168812348085, 4.674557938961098, 4.660168812348085, 4.616741622074666, 4.543496011917352, 4.439134539756339, 4.301839388015011, 4.129290293281011, 3.918852544828722, 3.666780625937393, 3.3716167185567794, 3.0260455871551, 2.6365819174880345, 2.1137582766948024, 2.636581917488038, 3.4335823665029626, 3.9865333116002404, 4.413365681101105, 4.7889411365991865, 5.104453801345153, 5.36923080244542, 5.585594884403345, 5.7577177607817, 5.8884937048816415, 5.980248025842213, 6.034641916141866, 6.052663312839643, 6.034641916141865, 5.980248025842214, 5.888493704881639, 5.757717760781698, 5.585594884403341, 5.369230802445415, 5.104453801345149, 4.78894113659918, 4.413365681101101, 3.986533311600236, 3.433582366502959, 2.636581917488036, 3.0260455871551004, 3.986533311600237, 4.764985911789622, 5.3195594392634815, 5.755716685290208, 6.1360848502194845, 6.449966786612971, 6.708038228026866, 6.912828986796572, 7.068443611677266, 7.177617595002775, 7.242328637658107, 7.263768519691014, 7.242328637658108, 7.177617595002773, 7.068443611677264, 6.912828986796565, 6.708038228026858, 6.449966786612963, 6.136084850219477, 5.755716685290201, 5.319559439263479, 4.764985911789622, 3.9865333116002364, 3.026045587155101, 3.3716167185567794, 4.413365681101101, 5.319559439263478, 6.059734933640805, 6.594669919226913, 7.019009405830075, 7.385013603583526, 7.680284321463562, 7.916262866710092, 8.095197815005834, 8.220753908477315, 8.295185933912736, 8.319842223627575, 8.295185933912736, 8.220753908477313, 8.095197815005829, 7.9162628667100865, 7.680284321463553, 7.385013603583518, 7.019009405830066, 6.5946699192269085, 6.0597349336408035, 5.319559439263479, 4.413365681101103, 3.3716167185567816, 3.666780625937396, 4.788941136599183, 5.755716685290202, 6.594669919226909, 7.282837454229495, 7.7826243439997524, 8.17920924579081, 8.516051572939901, 8.77921200419981, 8.980435993926783, 9.1213414136694, 9.204876783614267, 9.232563239373745, 9.20487678361426, 9.121341413669393, 8.980435993926774, 8.779212004199799, 8.516051572939887, 8.179209245790801, 7.782624343999744, 7.28283745422949, 6.594669919226911, 5.755716685290203, 4.788941136599184, 3.6667806259373954, 3.918852544828727, 5.104453801345156, 6.136084850219482, 7.019009405830074, 7.7826243439997524, 8.40938822057789, 8.862542988092518, 9.219039481903943, 9.514904344894836, 9.734791360013814, 9.890382762980401, 9.982413174745984, 10.012890843410913, 9.982413174745975, 9.890382762980387, 9.734791360013801, 9.514904344894822, 9.219039481903929, 8.862542988092505, 8.40938822057788, 7.782624343999746, 7.019009405830071, 6.136084850219481, 5.104453801345154, 3.918852544828727, 4.129290293281017, 5.369230802445423, 6.449966786612975, 7.385013603583529, 8.179209245790814, 8.86254298809252, 9.421325551938462, 9.819043013362487, 10.125544719186678, 10.370617986609494, 10.537607630974078, 10.637814624227977, 10.67092412242758, 10.637814624227971, 10.537607630974074, 10.37061798660948, 10.125544719186667, 9.81904301336248, 9.421325551938448, 8.862542988092509, 8.179209245790803, 7.385013603583525, 6.449966786612973, 5.369230802445423, 4.129290293281019, 4.301839388015017, 5.585594884403349, 6.70803822802687, 7.680284321463569, 8.516051572939904, 9.219039481903947, 9.819043013362498, 10.305028855809763, 10.640260599269606, 10.888489126670626, 11.074282265530496, 11.179757933772853, 11.215539376713828, 11.179757933772844, 11.074282265530494, 10.888489126670617, 10.640260599269597, 10.305028855809747, 9.819043013362476, 9.219039481903932, 8.516051572939894, 7.680284321463562, 6.708038228026868, 5.585594884403352, 4.301839388015019, 4.439134539756346, 5.7577177607817065, 6.912828986796578, 7.916262866710104, 8.779212004199813, 9.51490434489484, 10.125544719186685, 10.64026059926961, 11.049730149466411, 11.31657882466728, 11.499364812828357, 11.618289856582024, 11.653913937623651, 11.61828985658202, 11.499364812828354, 11.316578824667275, 11.049730149466402, 10.640260599269599, 10.12554471918667, 9.514904344894827, 8.779212004199808, 7.916262866710098, 6.912828986796577, 5.757717760781709, 4.439134539756347, 4.543496011917359, 5.888493704881649, 7.068443611677271, 8.095197815005841, 8.98043599392679, 9.734791360013816, 10.370617986609494, 10.88848912667063, 11.316578824667287, 11.64652310519862, 11.839888723085174, 11.950458326084561, 11.997247515430729, 11.950458326084556, 11.839888723085167, 11.646523105198613, 11.316578824667275, 10.888489126670617, 10.370617986609481, 9.734791360013805, 8.980435993926779, 8.095197815005836, 7.068443611677272, 5.888493704881652, 4.543496011917362, 4.61674162207467, 5.980248025842219, 7.177617595002779, 8.22075390847732, 9.121341413669404, 9.890382762980396, 10.537607630974078, 11.074282265530497, 11.499364812828357, 11.839888723085176, 12.08742129738902, 12.20452583801236, 12.231939627403282, 12.204525838012362, 12.087421297389014, 11.839888723085165, 11.499364812828354, 11.074282265530488, 10.537607630974072, 9.890382762980394, 9.121341413669398, 8.220753908477317, 7.177617595002782, 5.980248025842224, 4.6167416220746755, 4.660168812348088, 6.03464191614187, 7.242328637658109, 8.295185933912737, 9.204876783614266, 9.982413174745979, 10.637814624227977, 11.17975793377285, 11.61828985658203, 11.95045832608456, 12.20452583801236, 12.36242521569222, 12.403495350196929, 12.36242521569222, 12.204525838012351, 11.950458326084558, 11.618289856582022, 11.179757933772846, 10.637814624227973, 9.982413174745977, 9.20487678361426, 8.295185933912736, 7.2423286376581135, 6.034641916141876, 4.660168812348094, 4.674557938961099, 6.052663312839645, 7.263768519691019, 8.319842223627575, 9.232563239373746, 10.012890843410915, 10.67092412242758, 11.215539376713833, 11.653913937623651, 11.99724751543073, 12.231939627403282, 12.403495350196932, 12.490278681424238, 12.403495350196927, 12.231939627403277, 11.997247515430733, 11.653913937623647, 11.215539376713828, 10.670924122427579, 10.012890843410915, 9.232563239373748, 8.31984222362758, 7.263768519691021, 6.05266331283965, 4.674557938961106, 4.660168812348089, 6.034641916141869, 7.24232863765811, 8.295185933912737, 9.204876783614264, 9.98241317474598, 10.63781462422798, 11.179757933772853, 11.61828985658203, 11.950458326084561, 12.204525838012364, 12.362425215692218, 12.40349535019693, 12.362425215692218, 12.204525838012357, 11.950458326084558, 11.618289856582027, 11.179757933772848, 10.637814624227977, 9.982413174745984, 9.204876783614267, 8.29518593391274, 7.2423286376581135, 6.034641916141875, 4.660168812348094, 4.616741622074671, 5.980248025842217, 7.177617595002775, 8.220753908477317, 9.121341413669398, 9.890382762980392, 10.537607630974076, 11.0742822655305, 11.499364812828361, 11.839888723085176, 12.087421297389016, 12.204525838012358, 12.231939627403277, 12.204525838012355, 12.087421297389016, 11.839888723085169, 11.499364812828361, 11.074282265530499, 10.537607630974078, 9.890382762980403, 9.121341413669404, 8.220753908477318, 7.177617595002784, 5.980248025842225, 4.616741622074678, 4.5434960119173535, 5.888493704881644, 7.068443611677267, 8.095197815005829, 8.980435993926775, 9.734791360013807, 10.370617986609489, 10.888489126670626, 11.316578824667278, 11.646523105198616, 11.839888723085169, 11.95045832608456, 11.99724751543073, 11.950458326084556, 11.83988872308517, 11.646523105198614, 11.31657882466728, 10.888489126670631, 10.370617986609496, 9.73479136001382, 8.980435993926786, 8.095197815005838, 7.068443611677275, 5.888493704881654, 4.543496011917363, 4.4391345397563375, 5.7577177607816985, 6.912828986796568, 7.916262866710087, 8.779212004199803, 9.514904344894827, 10.125544719186676, 10.640260599269608, 11.049730149466406, 11.316578824667276, 11.499364812828356, 11.61828985658202, 11.653913937623644, 11.618289856582027, 11.499364812828356, 11.316578824667276, 11.049730149466411, 10.640260599269613, 10.125544719186681, 9.51490434489484, 8.779212004199811, 7.916262866710099, 6.912828986796576, 5.757717760781707, 4.439134539756347, 4.301839388015011, 5.58559488440334, 6.708038228026859, 7.680284321463556, 8.51605157293989, 9.219039481903934, 9.819043013362487, 10.305028855809757, 10.640260599269604, 10.88848912667062, 11.074282265530497, 11.179757933772846, 11.215539376713828, 11.179757933772846, 11.074282265530496, 10.888489126670626, 10.640260599269608, 10.30502885580976, 9.819043013362489, 9.219039481903943, 8.516051572939903, 7.680284321463565, 6.708038228026868, 5.585594884403352, 4.3018393880150185, 4.12929029328101, 5.369230802445414, 6.449966786612966, 7.3850136035835225, 8.179209245790803, 8.862542988092512, 9.42132555193845, 9.819043013362489, 10.125544719186678, 10.37061798660949, 10.537607630974074, 10.637814624227977, 10.670924122427584, 10.637814624227978, 10.537607630974083, 10.370617986609492, 10.125544719186678, 9.81904301336249, 9.421325551938459, 8.862542988092521, 8.17920924579081, 7.38501360358353, 6.4499667866129755, 5.369230802445425, 4.129290293281018, 3.91885254482872, 5.104453801345149, 6.136084850219476, 7.019009405830068, 7.782624343999747, 8.409388220577886, 8.86254298809251, 9.21903948190394, 9.514904344894834, 9.734791360013814, 9.890382762980398, 9.982413174745982, 10.012890843410915, 9.982413174745984, 9.890382762980401, 9.734791360013817, 9.514904344894838, 9.219039481903938, 8.862542988092521, 8.409388220577892, 7.782624343999754, 7.019009405830075, 6.136084850219486, 5.104453801345158, 3.918852544828728, 3.6667806259373914, 4.78894113659918, 5.755716685290201, 6.594669919226909, 7.28283745422949, 7.782624343999746, 8.179209245790807, 8.5160515729399, 8.77921200419981, 8.980435993926783, 9.121341413669407, 9.204876783614267, 9.23256323937375, 9.204876783614273, 9.121341413669406, 8.980435993926788, 8.77921200419981, 8.5160515729399, 8.179209245790808, 7.7826243439997524, 7.282837454229498, 6.594669919226916, 5.755716685290205, 4.788941136599187, 3.6667806259373967, 3.37161671855678, 4.413365681101101, 5.319559439263477, 6.059734933640802, 6.5946699192269085, 7.0190094058300705, 7.385013603583527, 7.680284321463568, 7.916262866710099, 8.09519781500584, 8.220753908477324, 8.295185933912745, 8.319842223627582, 8.295185933912743, 8.220753908477322, 8.095197815005838, 7.916262866710101, 7.680284321463565, 7.385013603583527, 7.019009405830073, 6.5946699192269165, 6.059734933640807, 5.319559439263479, 4.413365681101102, 3.371616718556781, 3.026045587155099, 3.9865333116002355, 4.764985911789621, 5.319559439263474, 5.755716685290202, 6.136084850219482, 6.449966786612975, 6.708038228026869, 6.912828986796578, 7.068443611677276, 7.177617595002782, 7.242328637658115, 7.263768519691024, 7.242328637658114, 7.177617595002782, 7.068443611677276, 6.91282898679658, 6.708038228026866, 6.449966786612972, 6.136084850219481, 5.755716685290203, 5.319559439263478, 4.764985911789619, 3.9865333116002333, 3.0260455871550973, 2.6365819174880283, 3.433582366502954, 3.986533311600234, 4.413365681101106, 4.788941136599191, 5.104453801345162, 5.369230802445426, 5.585594884403354, 5.757717760781711, 5.888493704881652, 5.980248025842221, 6.034641916141872, 6.052663312839648, 6.034641916141873, 5.980248025842222, 5.88849370488165, 5.757717760781708, 5.58559488440335, 5.369230802445423, 5.104453801345157, 4.788941136599186, 4.4133656811011015, 3.986533311600232, 3.4335823665029483, 2.636581917488022, 2.113758276694795, 2.636581917488029, 3.0260455871551013, 3.3716167185567865, 3.666780625937402, 3.9188525448287326, 4.129290293281024, 4.30183938801502, 4.439134539756346, 4.543496011917359, 4.616741622074675, 4.6601688123480915, 4.674557938961102, 4.66016881234809, 4.616741622074674, 4.54349601191736, 4.4391345397563455, 4.301839388015018, 4.129290293281022, 3.91885254482873, 3.6667806259373985, 3.371616718556779, 3.0260455871550938, 2.6365819174880216, 2.11375827669479]], [[1.4767264367289454, 2.0536115658036045, 2.4501895371664633, 2.769756152635721, 3.0178250504254187, 3.2116431023155574, 3.360859644160609, 3.475513976480186, 3.5616249895027488, 3.6242461228006966, 3.666736503780116, 3.6913689043365716, 3.699440712242182, 3.6913689043365707, 3.666736503780114, 3.624246122800694, 3.561624989502746, 3.475513976480182, 3.3608596441606076, 3.211643102315556, 3.0178250504254156, 2.7697561526357193, 2.450189537166463, 2.0536115658036023, 1.476726436728944, 2.053611565803603, 2.9806508783667836, 3.594949481069355, 4.041192377459999, 4.399655593175661, 4.676062267018393, 4.891427824136701, 5.056008833791333, 5.18036209966473, 5.270748798231736, 5.332176789846988, 5.367815445130931, 5.379493675422171, 5.367815445130932, 5.3321767898469865, 5.2707487982317325, 5.180362099664724, 5.056008833791329, 4.891427824136698, 4.676062267018389, 4.399655593175656, 4.041192377459995, 3.5949494810693516, 2.9806508783667813, 2.053611565803603, 2.450189537166463, 3.594949481069355, 4.45557153192178, 5.048969549536415, 5.494034596665513, 5.85133278235618, 6.125118539119001, 6.337114918558057, 6.496329623276333, 6.612650566613521, 6.691642932786554, 6.737508894986761, 6.752551290663871, 6.737508894986762, 6.691642932786551, 6.6126505666135165, 6.496329623276329, 6.337114918558051, 6.125118539118993, 5.851332782356174, 5.494034596665506, 5.04896954953641, 4.455571531921775, 3.594949481069352, 2.4501895371664615, 2.769756152635722, 4.041192377459998, 5.048969549536414, 5.81893429279327, 6.364366237392115, 6.781718477169064, 7.115632644688468, 7.3685696691782505, 7.561517028803791, 7.701421995516999, 7.796908865110687, 7.85229656349987, 7.870455566936735, 7.852296563499866, 7.796908865110681, 7.701421995516996, 7.561517028803788, 7.368569669178243, 7.11563264468846, 6.781718477169059, 6.364366237392108, 5.818934292793264, 5.048969549536411, 4.041192377459995, 2.7697561526357193, 3.0178250504254174, 4.399655593175659, 5.49403459666551, 6.364366237392114, 7.038462665587848, 7.523991240371386, 7.899375123783939, 8.19699542212261, 8.41762546391737, 8.580784108132265, 8.69101588151382, 8.755320614100603, 8.776386094007659, 8.755320614100603, 8.691015881513819, 8.58078410813226, 8.417625463917357, 8.196995422122601, 7.8993751237839325, 7.523991240371379, 7.038462665587842, 6.364366237392106, 5.4940345966655055, 4.399655593175656, 3.017825050425416, 3.2116431023155574, 4.6760622670183905, 5.85133278235618, 6.781718477169063, 7.5239912403713864, 8.104277290662166, 8.525040454068225, 8.850655152516179, 9.104145446405983, 9.28471412881898, 9.409988549808515, 9.481974487228463, 9.505771074606805, 9.481974487228467, 9.409988549808512, 9.284714128818969, 9.10414544640597, 8.850655152516165, 8.525040454068211, 8.104277290662157, 7.523991240371374, 6.781718477169058, 5.851332782356172, 4.676062267018387, 3.2116431023155547, 3.360859644160608, 4.891427824136697, 6.125118539118997, 7.115632644688469, 7.899375123783939, 8.525040454068227, 9.016095450707015, 9.370641412825156, 9.642142018518644, 9.846394441040012, 9.981067128226861, 10.061616277791211, 10.087320592479987, 10.061616277791208, 9.981067128226854, 9.846394441040006, 9.64214201851863, 9.37064141282514, 9.016095450707004, 8.525040454068215, 7.899375123783932, 7.11563264468846, 6.125118539118992, 4.891427824136693, 3.3608596441606036, 3.4755139764801837, 5.056008833791331, 6.337114918558055, 7.36856966917825, 8.196995422122612, 8.850655152516179, 9.370641412825155, 9.777620689458715, 10.066035229392874, 10.28081219824272, 10.432034289079153, 10.515655792353899, 10.545275368624662, 10.515655792353899, 10.432034289079148, 10.280812198242712, 10.066035229392858, 9.777620689458702, 9.37064141282514, 8.850655152516167, 8.196995422122601, 7.368569669178241, 6.337114918558049, 5.056008833791327, 3.4755139764801806, 3.5616249895027474, 5.180362099664729, 6.496329623276332, 7.561517028803792, 8.41762546391737, 9.104145446405981, 9.642142018518644, 10.06603522939287, 10.393790492194215, 10.616656169989959, 10.772672174346603, 10.867813795448603, 10.894360150480162, 10.867813795448603, 10.772672174346596, 10.616656169989948, 10.3937904921942, 10.066035229392854, 9.64214201851863, 9.104145446405969, 8.41762546391736, 7.561517028803784, 6.496329623276324, 5.180362099664722, 3.5616249895027434, 3.6242461228006966, 5.270748798231736, 6.612650566613521, 7.701421995517, 8.580784108132269, 9.284714128818985, 9.846394441040017, 10.280812198242721, 10.616656169989957, 10.869209298799491, 11.027337059982218, 11.121638960398776, 11.158788722839008, 11.121638960398776, 11.02733705998221, 10.869209298799477, 10.616656169989946, 10.280812198242707, 9.846394441040001, 9.284714128818967, 8.580784108132256, 7.701421995516991, 6.612650566613514, 5.270748798231729, 3.6242461228006935, 3.666736503780114, 5.332176789846987, 6.6916429327865545, 7.796908865110688, 8.69101588151383, 9.409988549808519, 9.981067128226861, 10.432034289079157, 10.772672174346605, 11.02733705998222, 11.20686735617178, 11.30192763378054, 11.331229666432739, 11.301927633780531, 11.20686735617177, 11.027337059982205, 10.772672174346594, 10.432034289079148, 9.981067128226854, 9.409988549808507, 8.691015881513815, 7.796908865110677, 6.691642932786546, 5.332176789846984, 3.6667365037801107, 3.69136890433657, 5.367815445130929, 6.737508894986762, 7.852296563499869, 8.755320614100611, 9.481974487228475, 10.06161627779122, 10.515655792353908, 10.867813795448614, 11.12163896039878, 11.301927633780542, 11.409796082720524, 11.436653246156595, 11.409796082720513, 11.30192763378053, 11.12163896039877, 10.8678137954486, 10.515655792353895, 10.06161627779121, 9.481974487228463, 8.755320614100597, 7.852296563499859, 6.737508894986757, 5.367815445130929, 3.6913689043365685, 3.6994407122421826, 5.3794936754221725, 6.752551290663872, 7.870455566936736, 8.776386094007666, 9.505771074606814, 10.087320592479994, 10.545275368624665, 10.894360150480173, 11.158788722839018, 11.331229666432744, 11.436653246156599, 11.485906230578191, 11.436653246156594, 11.33122966643273, 11.158788722839, 10.89436015048016, 10.545275368624655, 10.087320592479983, 9.505771074606805, 8.776386094007654, 7.870455566936726, 6.752551290663868, 5.379493675422171, 3.6994407122421813, 3.691368904336571, 5.367815445130932, 6.737508894986764, 7.852296563499871, 8.75532061410061, 9.481974487228472, 10.061616277791215, 10.515655792353906, 10.867813795448614, 11.121638960398782, 11.301927633780544, 11.409796082720526, 11.436653246156595, 11.40979608272052, 11.301927633780533, 11.121638960398773, 10.867813795448596, 10.515655792353893, 10.06161627779121, 9.481974487228465, 8.7553206141006, 7.852296563499864, 6.73750889498676, 5.3678154451309315, 3.6913689043365703, 3.6667365037801134, 5.332176789846987, 6.6916429327865545, 7.79690886511069, 8.691015881513826, 9.409988549808523, 9.981067128226861, 10.432034289079157, 10.772672174346612, 11.027337059982218, 11.206867356171783, 11.301927633780545, 11.33122966643274, 11.301927633780535, 11.206867356171777, 11.02733705998221, 10.772672174346594, 10.432034289079148, 9.981067128226849, 9.409988549808507, 8.691015881513819, 7.796908865110685, 6.691642932786552, 5.332176789846987, 3.6667365037801125, 3.6242461228006957, 5.270748798231736, 6.612650566613523, 7.701421995517, 8.580784108132269, 9.284714128818983, 9.846394441040015, 10.280812198242723, 10.616656169989962, 10.869209298799493, 11.027337059982218, 11.121638960398778, 11.158788722839013, 11.121638960398782, 11.027337059982216, 10.869209298799483, 10.61665616998995, 10.280812198242712, 9.846394441040005, 9.284714128818973, 8.58078410813226, 7.701421995516997, 6.612650566613518, 5.270748798231733, 3.624246122800695, 3.561624989502747, 5.1803620996647295, 6.496329623276332, 7.561517028803793, 8.417625463917368, 9.104145446405983, 9.642142018518644, 10.06603522939287, 10.393790492194213, 10.616656169989959, 10.772672174346605, 10.867813795448605, 10.894360150480168, 10.86781379544861, 10.772672174346601, 10.616656169989954, 10.393790492194206, 10.06603522939286, 9.642142018518634, 9.104145446405973, 8.41762546391736, 7.561517028803788, 6.496329623276331, 5.1803620996647295, 3.5616249895027474, 3.475513976480184, 5.056008833791331, 6.337114918558059, 7.368569669178249, 8.19699542212261, 8.850655152516179, 9.370641412825155, 9.777620689458713, 10.066035229392867, 10.280812198242717, 10.432034289079153, 10.515655792353906, 10.545275368624662, 10.5156557923539, 10.432034289079152, 10.280812198242712, 10.066035229392861, 9.777620689458704, 9.370641412825144, 8.850655152516172, 8.196995422122603, 7.368569669178246, 6.337114918558055, 5.056008833791332, 3.4755139764801855, 3.3608596441606076, 4.891427824136698, 6.125118539118999, 7.115632644688468, 7.899375123783939, 8.525040454068225, 9.016095450707015, 9.370641412825153, 9.642142018518642, 9.846394441040012, 9.981067128226858, 10.061616277791218, 10.087320592479992, 10.061616277791213, 9.981067128226854, 9.846394441040008, 9.642142018518634, 9.370641412825144, 9.016095450707004, 8.525040454068215, 7.899375123783936, 7.115632644688464, 6.125118539118995, 4.891427824136698, 3.3608596441606085, 3.2116431023155565, 4.6760622670183905, 5.851332782356179, 6.781718477169064, 7.523991240371383, 8.104277290662164, 8.52504045406822, 8.850655152516177, 9.10414544640598, 9.28471412881898, 9.409988549808514, 9.48197448722847, 9.50577107460681, 9.481974487228467, 9.40998854980851, 9.284714128818974, 9.104145446405976, 8.850655152516172, 8.525040454068213, 8.104277290662159, 7.523991240371381, 6.781718477169062, 5.851332782356178, 4.6760622670183905, 3.2116431023155583, 3.0178250504254183, 4.39965559317566, 5.494034596665512, 6.364366237392111, 7.038462665587846, 7.52399124037138, 7.899375123783936, 8.196995422122605, 8.417625463917366, 8.580784108132264, 8.69101588151382, 8.755320614100606, 8.776386094007663, 8.7553206141006, 8.691015881513817, 8.58078410813226, 8.417625463917362, 8.196995422122601, 7.899375123783928, 7.523991240371378, 7.038462665587844, 6.364366237392112, 5.494034596665513, 4.399655593175662, 3.0178250504254196, 2.7697561526357215, 4.041192377460001, 5.048969549536414, 5.818934292793269, 6.364366237392113, 6.781718477169062, 7.115632644688465, 7.368569669178244, 7.561517028803786, 7.701421995516996, 7.796908865110683, 7.852296563499866, 7.87045556693673, 7.8522965634998645, 7.796908865110679, 7.701421995516995, 7.561517028803784, 7.36856966917824, 7.1156326446884615, 6.781718477169056, 6.364366237392108, 5.818934292793268, 5.048969549536415, 4.041192377460001, 2.769756152635721, 2.4501895371664633, 3.5949494810693534, 4.455571531921776, 5.048969549536413, 5.494034596665512, 5.851332782356177, 6.1251185391189935, 6.337114918558053, 6.496329623276328, 6.612650566613516, 6.6916429327865465, 6.737508894986757, 6.752551290663868, 6.7375088949867585, 6.691642932786549, 6.612650566613514, 6.496329623276323, 6.337114918558048, 6.125118539118991, 5.851332782356175, 5.494034596665507, 5.048969549536414, 4.455571531921776, 3.5949494810693525, 2.4501895371664637, 2.053611565803603, 2.9806508783667827, 3.594949481069354, 4.041192377460001, 4.399655593175661, 4.67606226701839, 4.891427824136697, 5.05600883379133, 5.180362099664725, 5.270748798231732, 5.332176789846985, 5.367815445130927, 5.37949367542217, 5.367815445130928, 5.332176789846985, 5.270748798231729, 5.1803620996647215, 5.056008833791325, 4.891427824136693, 4.676062267018388, 4.399655593175656, 4.041192377459996, 3.594949481069352, 2.9806508783667813, 2.0536115658036027, 1.476726436728944, 2.0536115658036027, 2.4501895371664637, 2.769756152635721, 3.0178250504254183, 3.2116431023155556, 3.3608596441606067, 3.4755139764801837, 3.5616249895027456, 3.6242461228006935, 3.666736503780112, 3.691368904336567, 3.6994407122421795, 3.691368904336569, 3.6667365037801116, 3.624246122800693, 3.5616249895027448, 3.47551397648018, 3.3608596441606036, 3.2116431023155547, 3.0178250504254165, 2.7697561526357197, 2.450189537166461, 2.0536115658036027, 1.4767264367289434]]]

        phi_test = np.array(phi_test)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()
            phi_zero_test = phi_test[:, l].flatten()
            np.testing.assert_array_almost_equal(phi, phi_zero_test, 7)

        self.angular_test()

    def test_solver_basic_2D_2g_1a_vacuum_eigen(self):
        '''
        Test for a basic 2 group problem
        '''
        pydgm.control.fine_mesh_x = [25]
        pydgm.control.fine_mesh_y = [25]
        pydgm.control.coarse_mesh_x = [0.0, 10.0]
        pydgm.control.coarse_mesh_y = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 0.0
        pydgm.control.boundary_north = 0.0
        pydgm.control.boundary_south = 0.0
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.eigen_tolerance = 1e-12
        pydgm.control.outer_tolerance = 1e-12

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[0.2487746067595804, 0.30933535885463925, 0.36503957161582756, 0.4201277246266406, 0.4727429041064538, 0.5219789831678958, 0.5666972948641681, 0.6060354489662487, 0.6392379836284101, 0.6656917181028958, 0.6849238962362818, 0.6965974817500068, 0.7005110260557053, 0.6965974817500072, 0.6849238962362818, 0.6656917181028954, 0.6392379836284099, 0.6060354489662486, 0.5666972948641676, 0.5219789831678954, 0.47274290410645325, 0.4201277246266398, 0.3650395716158268, 0.30933535885463875, 0.24877460675958005, 0.3093353588546392, 0.40164987296136834, 0.48291526870614243, 0.5584914296565835, 0.6312485387266421, 0.6988077036867092, 0.7601167557294932, 0.813912097913677, 0.8592449558483541, 0.8953214292816684, 0.92152628635136, 0.9374235947881444, 0.9427517097445433, 0.9374235947881437, 0.9215262863513597, 0.8953214292816681, 0.8592449558483536, 0.8139120979136766, 0.7601167557294923, 0.698807703686708, 0.6312485387266414, 0.5584914296565827, 0.48291526870614165, 0.40164987296136784, 0.3093353588546388, 0.36503957161582756, 0.4829152687061422, 0.5944451912542015, 0.6940912164324666, 0.7860949620587045, 0.8723187135451134, 0.9500816973490145, 1.0183122596044265, 1.0757218667223183, 1.1213682123712332, 1.1545053937082992, 1.174599589924656, 1.1813330856367277, 1.1745995899246553, 1.1545053937082987, 1.121368212371233, 1.0757218667223176, 1.0183122596044252, 0.950081697349013, 0.8723187135451121, 0.7860949620587036, 0.694091216432466, 0.5944451912542007, 0.48291526870614176, 0.36503957161582723, 0.4201277246266406, 0.5584914296565837, 0.6940912164324667, 0.8215542114308743, 0.9352477382783926, 1.0386119631536603, 1.132791234434996, 1.2149670473854255, 1.284144202657454, 1.3390943477694461, 1.3789641487964708, 1.4031352581076166, 1.4112332698908574, 1.4031352581076166, 1.3789641487964701, 1.3390943477694455, 1.2841442026574532, 1.2149670473854237, 1.1327912344349942, 1.0386119631536588, 0.9352477382783915, 0.8215542114308731, 0.6940912164324659, 0.5584914296565832, 0.4201277246266401, 0.47274290410645387, 0.6312485387266424, 0.7860949620587048, 0.9352477382783928, 1.0736610472994341, 1.1957589812484655, 1.3044202797066318, 1.4002986998355933, 1.480566815593873, 1.5443844450713775, 1.590659651713007, 1.618703820457089, 1.6280998521423597, 1.6187038204570887, 1.5906596517130063, 1.5443844450713762, 1.4805668155938718, 1.400298699835592, 1.3044202797066304, 1.1957589812484641, 1.0736610472994326, 0.9352477382783914, 0.7860949620587037, 0.6312485387266413, 0.47274290410645303, 0.5219789831678961, 0.6988077036867091, 0.8723187135451133, 1.0386119631536603, 1.1957589812484652, 1.3391074138222738, 1.463235534944001, 1.5706511446747617, 1.661703820432259, 1.7336628995791248, 1.785911181959954, 1.8175629277589989, 1.8281624800473772, 1.8175629277589982, 1.7859111819599531, 1.7336628995791235, 1.6617038204322576, 1.5706511446747609, 1.463235534944001, 1.3391074138222734, 1.195758981248464, 1.0386119631536588, 0.872318713545112, 0.6988077036867077, 0.5219789831678953, 0.5666972948641688, 0.7601167557294939, 0.9500816973490143, 1.1327912344349955, 1.304420279706632, 1.4632355349440012, 1.6050104613189407, 1.7245450833554363, 1.8241342432960628, 1.9039815054710298, 1.9615421041215386, 1.9964802438076725, 2.0081817572886504, 1.9964802438076716, 1.9615421041215384, 1.9039815054710294, 1.8241342432960614, 1.724545083355435, 1.6050104613189402, 1.4632355349440005, 1.3044202797066307, 1.132791234434994, 0.9500816973490128, 0.760116755729492, 0.5666972948641678, 0.6060354489662493, 0.8139120979136776, 1.0183122596044265, 1.2149670473854246, 1.400298699835593, 1.5706511446747617, 1.724545083355436, 1.8581966015787008, 1.9666783752660877, 2.0522072216331457, 2.114975856603474, 2.152703108428814, 2.1653793244298796, 2.1527031084288137, 2.114975856603474, 2.0522072216331453, 1.9666783752660866, 1.8581966015786995, 1.7245450833554352, 1.570651144674761, 1.4002986998355924, 1.2149670473854235, 1.018312259604425, 0.8139120979136766, 0.6060354489662485, 0.6392379836284107, 0.859244955848355, 1.0757218667223187, 1.2841442026574539, 1.4805668155938723, 1.661703820432259, 1.8241342432960632, 1.9666783752660877, 2.0860050936007664, 2.1774982372953064, 2.243415389846528, 2.284049071070629, 2.297430887485442, 2.2840490710706294, 2.243415389846527, 2.1774982372953033, 2.086005093600764, 1.9666783752660864, 1.824134243296061, 1.6617038204322578, 1.4805668155938714, 1.2841442026574532, 1.0757218667223174, 0.8592449558483536, 0.6392379836284099, 0.6656917181028967, 0.8953214292816696, 1.1213682123712334, 1.3390943477694452, 1.5443844450713782, 1.7336628995791252, 1.9039815054710307, 2.0522072216331466, 2.177498237295307, 2.276984903871456, 2.3463849560806116, 2.3880835394753026, 2.402706436696479, 2.388083539475304, 2.3463849560806103, 2.276984903871452, 2.1774982372953047, 2.0522072216331444, 1.9039815054710287, 1.7336628995791237, 1.5443844450713762, 1.3390943477694448, 1.1213682123712327, 0.8953214292816684, 0.6656917181028957, 0.6849238962362824, 0.9215262863513608, 1.1545053937082996, 1.3789641487964708, 1.590659651713007, 1.7859111819599531, 1.9615421041215388, 2.1149758566034755, 2.2434153898465286, 2.3463849560806116, 2.4214652692362337, 2.4648625085214766, 2.4785988981976548, 2.464862508521476, 2.421465269236231, 2.34638495608061, 2.2434153898465263, 2.114975856603473, 1.9615421041215368, 1.7859111819599527, 1.590659651713006, 1.37896414879647, 1.1545053937082985, 0.9215262863513604, 0.684923896236282, 0.6965974817500075, 0.9374235947881446, 1.1745995899246564, 1.4031352581076177, 1.6187038204570894, 1.8175629277589984, 1.9964802438076719, 2.1527031084288146, 2.2840490710706303, 2.3880835394753044, 2.4648625085214766, 2.5118370812605506, 2.5268458313707605, 2.511837081260549, 2.4648625085214757, 2.3880835394753026, 2.2840490710706276, 2.152703108428813, 1.9964802438076716, 1.8175629277589982, 1.6187038204570894, 1.4031352581076162, 1.1745995899246557, 0.9374235947881441, 0.6965974817500072, 0.7005110260557053, 0.9427517097445437, 1.1813330856367281, 1.411233269890858, 1.6280998521423609, 1.8281624800473772, 2.008181757288651, 2.16537932442988, 2.2974308874854423, 2.4027064366964797, 2.478598898197655, 2.526845831370762, 2.544995566124512, 2.526845831370761, 2.478598898197654, 2.4027064366964774, 2.29743088748544, 2.1653793244298787, 2.008181757288651, 1.828162480047377, 1.6280998521423604, 1.4112332698908572, 1.1813330856367281, 0.942751709744544, 0.7005110260557053, 0.696597481750007, 0.9374235947881437, 1.1745995899246553, 1.4031352581076164, 1.6187038204570892, 1.817562927758998, 1.9964802438076719, 2.152703108428814, 2.284049071070629, 2.3880835394753035, 2.4648625085214775, 2.5118370812605497, 2.526845831370762, 2.511837081260549, 2.4648625085214744, 2.388083539475302, 2.284049071070627, 2.1527031084288137, 1.9964802438076714, 1.8175629277589977, 1.6187038204570898, 1.403135258107617, 1.1745995899246562, 0.9374235947881443, 0.6965974817500074, 0.6849238962362815, 0.9215262863513597, 1.1545053937082983, 1.37896414879647, 1.5906596517130056, 1.785911181959953, 1.9615421041215382, 2.1149758566034746, 2.2434153898465277, 2.346384956080611, 2.4214652692362324, 2.464862508521476, 2.478598898197654, 2.464862508521475, 2.4214652692362306, 2.34638495608061, 2.2434153898465268, 2.114975856603473, 1.9615421041215373, 1.7859111819599534, 1.5906596517130067, 1.3789641487964701, 1.154505393708299, 0.9215262863513605, 0.6849238962362822, 0.6656917181028956, 0.8953214292816684, 1.1213682123712325, 1.3390943477694441, 1.5443844450713762, 1.7336628995791235, 1.9039815054710296, 2.0522072216331453, 2.177498237295306, 2.2769849038714542, 2.3463849560806107, 2.388083539475303, 2.4027064366964783, 2.3880835394753013, 2.3463849560806094, 2.2769849038714542, 2.1774982372953042, 2.0522072216331435, 1.903981505471029, 1.733662899579124, 1.5443844450713766, 1.3390943477694452, 1.1213682123712334, 0.8953214292816688, 0.6656917181028961, 0.6392379836284101, 0.8592449558483539, 1.0757218667223172, 1.2841442026574528, 1.4805668155938716, 1.6617038204322572, 1.824134243296061, 1.966678375266086, 2.0860050936007646, 2.177498237295305, 2.243415389846527, 2.2840490710706276, 2.2974308874854406, 2.2840490710706276, 2.2434153898465268, 2.1774982372953056, 2.0860050936007637, 1.966678375266086, 1.8241342432960603, 1.6617038204322567, 1.4805668155938712, 1.2841442026574534, 1.0757218667223183, 0.8592449558483548, 0.6392379836284101, 0.6060354489662485, 0.8139120979136767, 1.0183122596044252, 1.214967047385424, 1.4002986998355924, 1.5706511446747604, 1.724545083355434, 1.8581966015786986, 1.966678375266085, 2.052207221633143, 2.114975856603472, 2.1527031084288124, 2.165379324429878, 2.152703108428813, 2.1149758566034733, 2.0522072216331444, 1.9666783752660861, 1.8581966015786993, 1.7245450833554339, 1.5706511446747597, 1.400298699835592, 1.2149670473854246, 1.018312259604426, 0.8139120979136778, 0.6060354489662493, 0.5666972948641678, 0.7601167557294926, 0.9500816973490129, 1.132791234434994, 1.3044202797066307, 1.4632355349439998, 1.6050104613189393, 1.7245450833554337, 1.8241342432960592, 1.9039815054710265, 1.9615421041215355, 1.99648024380767, 2.0081817572886496, 1.9964802438076705, 1.9615421041215368, 1.9039815054710283, 1.8241342432960603, 1.724545083355434, 1.6050104613189389, 1.4632355349439996, 1.3044202797066304, 1.1327912344349949, 0.9500816973490139, 0.7601167557294932, 0.5666972948641683, 0.5219789831678955, 0.6988077036867083, 0.8723187135451121, 1.0386119631536588, 1.1957589812484641, 1.3391074138222725, 1.4632355349439998, 1.5706511446747595, 1.6617038204322556, 1.733662899579122, 1.7859111819599514, 1.8175629277589964, 1.828162480047376, 1.8175629277589962, 1.785911181959952, 1.7336628995791221, 1.6617038204322565, 1.5706511446747602, 1.4632355349439996, 1.3391074138222723, 1.195758981248464, 1.0386119631536592, 0.8723187135451124, 0.6988077036867084, 0.5219789831678958, 0.47274290410645325, 0.6312485387266413, 0.7860949620587037, 0.9352477382783914, 1.0736610472994323, 1.1957589812484632, 1.3044202797066298, 1.4002986998355915, 1.4805668155938698, 1.5443844450713744, 1.5906596517130036, 1.618703820457087, 1.628099852142358, 1.6187038204570874, 1.5906596517130043, 1.5443844450713748, 1.48056681559387, 1.400298699835591, 1.3044202797066302, 1.1957589812484637, 1.0736610472994326, 0.9352477382783915, 0.7860949620587039, 0.6312485387266413, 0.4727429041064536, 0.42012772462664005, 0.558491429656583, 0.6940912164324656, 0.821554211430873, 0.9352477382783911, 1.0386119631536583, 1.1327912344349942, 1.214967047385423, 1.2841442026574517, 1.3390943477694435, 1.3789641487964681, 1.4031352581076144, 1.411233269890855, 1.4031352581076149, 1.378964148796469, 1.3390943477694435, 1.2841442026574519, 1.214967047385423, 1.1327912344349942, 1.0386119631536583, 0.9352477382783914, 0.8215542114308733, 0.6940912164324659, 0.5584914296565829, 0.4201277246266399, 0.3650395716158271, 0.4829152687061416, 0.5944451912542004, 0.6940912164324654, 0.7860949620587033, 0.872318713545112, 0.950081697349013, 1.0183122596044247, 1.0757218667223167, 1.1213682123712319, 1.1545053937082967, 1.1745995899246537, 1.181333085636726, 1.1745995899246535, 1.1545053937082974, 1.1213682123712316, 1.0757218667223165, 1.0183122596044245, 0.9500816973490126, 0.8723187135451116, 0.7860949620587037, 0.6940912164324659, 0.5944451912542006, 0.4829152687061415, 0.36503957161582695, 0.3093353588546387, 0.4016498729613677, 0.4829152687061415, 0.5584914296565826, 0.6312485387266412, 0.6988077036867079, 0.7601167557294921, 0.8139120979136765, 0.8592449558483536, 0.8953214292816674, 0.9215262863513591, 0.9374235947881425, 0.9427517097445415, 0.9374235947881424, 0.9215262863513588, 0.8953214292816672, 0.8592449558483534, 0.8139120979136764, 0.760116755729492, 0.6988077036867076, 0.6312485387266412, 0.5584914296565827, 0.4829152687061414, 0.4016498729613678, 0.3093353588546387, 0.24877460675957994, 0.3093353588546386, 0.36503957161582684, 0.42012772462663983, 0.47274290410645287, 0.5219789831678951, 0.5666972948641678, 0.6060354489662485, 0.6392379836284093, 0.6656917181028953, 0.6849238962362811, 0.6965974817500061, 0.7005110260557043, 0.6965974817500061, 0.684923896236281, 0.6656917181028952, 0.6392379836284097, 0.606035448966248, 0.5666972948641674, 0.5219789831678951, 0.47274290410645314, 0.42012772462663994, 0.3650395716158269, 0.3093353588546386, 0.24877460675958013]], [[0.01998505220506663, 0.028624451099833366, 0.03645151076496133, 0.04399260475136248, 0.05100072704478553, 0.05739396487095979, 0.06307779967625268, 0.067997245315577, 0.07209514838534635, 0.07532933795436803, 0.0776647807484596, 0.07907628103283944, 0.07954850885838675, 0.07907628103283944, 0.07766478074845957, 0.075329337954368, 0.07209514838534632, 0.06799724531557698, 0.06307779967625264, 0.057393964870959754, 0.05100072704478547, 0.043992604751362394, 0.036451510764961274, 0.028624451099833345, 0.019985052205066602, 0.028624451099833387, 0.04289219826344366, 0.05537627658493564, 0.06683400942917612, 0.07756460151772676, 0.08729947508349442, 0.09596646083873438, 0.10345581477898921, 0.10969794646838663, 0.11462311713123982, 0.1181799334309722, 0.12032968475616068, 0.12104887790598191, 0.12032968475616067, 0.1181799334309722, 0.11462311713123977, 0.1096979464683866, 0.10345581477898919, 0.09596646083873432, 0.08729947508349438, 0.07756460151772669, 0.06683400942917608, 0.055376276584935595, 0.04289219826344358, 0.02862445109983335, 0.036451510764961344, 0.05537627658493567, 0.07281240864873842, 0.08833946431758972, 0.10251035073030715, 0.11547966534412975, 0.12697133089098303, 0.13691796221646127, 0.1451970438988955, 0.1517319977604755, 0.1564501107528323, 0.15930170090015233, 0.16025573986215147, 0.15930170090015233, 0.15645011075283227, 0.1517319977604754, 0.1451970438988954, 0.13691796221646116, 0.126971330890983, 0.11547966534412962, 0.10251035073030706, 0.08833946431758967, 0.07281240864873835, 0.0553762765849356, 0.03645151076496128, 0.04399260475136246, 0.06683400942917615, 0.08833946431758974, 0.10812515410923546, 0.12576424655216117, 0.14166281810818404, 0.15587030482414796, 0.16810882465046229, 0.17831532245918838, 0.18636112398825838, 0.19217229897749483, 0.19568374851974488, 0.1968584242752685, 0.19568374851974477, 0.1921722989774947, 0.1863611239882583, 0.1783153224591883, 0.16810882465046217, 0.15587030482414788, 0.1416628181081839, 0.12576424655216112, 0.10812515410923533, 0.08833946431758966, 0.06683400942917607, 0.04399260475136238, 0.05100072704478551, 0.07756460151772678, 0.10251035073030718, 0.1257642465521612, 0.14699529578237652, 0.16577082703609106, 0.18238584389256482, 0.19681538872095647, 0.20878732915049025, 0.2182473780344291, 0.225070203331361, 0.22919492827762505, 0.23057461282062075, 0.22919492827762494, 0.22507020333136094, 0.21824737803442895, 0.20878732915049006, 0.19681538872095633, 0.18238584389256474, 0.16577082703609097, 0.14699529578237644, 0.1257642465521611, 0.10251035073030706, 0.07756460151772662, 0.05100072704478543, 0.057393964870959775, 0.08729947508349441, 0.11547966534412972, 0.14166281810818399, 0.16577082703609108, 0.18750266335421473, 0.20642199389721985, 0.22274385294262125, 0.23639580825188378, 0.24712057785504718, 0.2548797983308887, 0.2595620678850089, 0.26112952299654413, 0.25956206788500885, 0.25487979833088864, 0.24712057785504704, 0.23639580825188358, 0.22274385294262108, 0.20642199389721977, 0.1875026633542146, 0.16577082703609095, 0.14166281810818385, 0.11547966534412953, 0.08729947508349425, 0.05739396487095971, 0.06307779967625268, 0.0959664608387344, 0.126971330890983, 0.15587030482414793, 0.1823858438925648, 0.2064219938972199, 0.22769822236798304, 0.24578147642763298, 0.2608347764396368, 0.2727617209937381, 0.28132937120877516, 0.28652333758737614, 0.2882555932407505, 0.28652333758737614, 0.28132937120877494, 0.2727617209937379, 0.2608347764396366, 0.24578147642763265, 0.22769822236798287, 0.2064219938972197, 0.18238584389256465, 0.1558703048241477, 0.1269713308909828, 0.09596646083873425, 0.0630777996762526, 0.06799724531557702, 0.10345581477898926, 0.13691796221646121, 0.16810882465046226, 0.19681538872095627, 0.22274385294262114, 0.24578147642763293, 0.26566827591237996, 0.28198376832678235, 0.2948636707750291, 0.3042076650579847, 0.30981520997629597, 0.3117071758004514, 0.3098152099762959, 0.3042076650579845, 0.29486367077502895, 0.2819837683267821, 0.26566827591237957, 0.24578147642763262, 0.22274385294262097, 0.1968153887209561, 0.16810882465046204, 0.13691796221646108, 0.10345581477898912, 0.06799724531557694, 0.07209514838534634, 0.10969794646838667, 0.14519704389889548, 0.1783153224591884, 0.20878732915049014, 0.23639580825188367, 0.2608347764396368, 0.2819837683267825, 0.29960847209767005, 0.31331052795118874, 0.32321880946406617, 0.3292494667036456, 0.33122850555567945, 0.3292494667036456, 0.323218809464066, 0.31331052795118847, 0.29960847209766955, 0.28198376832678207, 0.2608347764396365, 0.2363958082518834, 0.20878732915049, 0.17831532245918816, 0.14519704389889518, 0.10969794646838649, 0.07209514838534624, 0.07532933795436803, 0.11462311713123982, 0.15173199776047552, 0.1863611239882584, 0.2182473780344291, 0.24712057785504715, 0.27276172099373797, 0.29486367077502923, 0.31331052795118886, 0.3278999450140935, 0.3382680210151431, 0.3445408145247518, 0.3466961419642437, 0.34454081452475166, 0.33826802101514275, 0.327899945014093, 0.3133105279511884, 0.29486367077502884, 0.2727617209937377, 0.2471205778550469, 0.21824737803442884, 0.18636112398825816, 0.15173199776047527, 0.11462311713123965, 0.07532933795436791, 0.07766478074845957, 0.11817993343097222, 0.1564501107528323, 0.19217229897749483, 0.22507020333136107, 0.25487979833088875, 0.28132937120877516, 0.3042076650579848, 0.32321880946406634, 0.3382680210151432, 0.3491782796062605, 0.3556589542210299, 0.357796817251322, 0.3556589542210296, 0.34917827960626024, 0.33826802101514264, 0.32321880946406584, 0.3042076650579843, 0.2813293712087748, 0.2548797983308885, 0.22507020333136074, 0.19217229897749455, 0.15645011075283208, 0.11817993343097205, 0.07766478074845946, 0.07907628103283938, 0.12032968475616064, 0.15930170090015233, 0.1956837485197449, 0.22919492827762514, 0.2595620678850089, 0.2865233375873762, 0.3098152099762961, 0.32924946670364585, 0.344540814524752, 0.35565895422102983, 0.36240803074961725, 0.36459041779744544, 0.36240803074961697, 0.35565895422102956, 0.3445408145247515, 0.3292494667036454, 0.30981520997629564, 0.2865233375873758, 0.2595620678850086, 0.2291949282776248, 0.19568374851974465, 0.1593017009001521, 0.12032968475616053, 0.07907628103283933, 0.0795485088583867, 0.12104887790598186, 0.16025573986215144, 0.19685842427526862, 0.23057461282062083, 0.2611295229965441, 0.2882555932407507, 0.3117071758004516, 0.3312285055556798, 0.3466961419642438, 0.3577968172513219, 0.36459041779744533, 0.36700813351718253, 0.3645904177974454, 0.3577968172513218, 0.3466961419642433, 0.3312285055556792, 0.31170717580045104, 0.28825559324075023, 0.2611295229965439, 0.23057461282062056, 0.19685842427526837, 0.16025573986215128, 0.12104887790598169, 0.07954850885838664, 0.07907628103283937, 0.1203296847561606, 0.15930170090015222, 0.19568374851974482, 0.229194928277625, 0.2595620678850089, 0.28652333758737614, 0.30981520997629614, 0.3292494667036457, 0.34454081452475166, 0.3556589542210297, 0.3624080307496171, 0.3645904177974454, 0.3624080307496171, 0.3556589542210296, 0.34454081452475144, 0.3292494667036452, 0.3098152099762958, 0.2865233375873758, 0.25956206788500863, 0.22919492827762492, 0.19568374851974468, 0.15930170090015214, 0.12032968475616056, 0.0790762810328393, 0.07766478074845959, 0.11817993343097216, 0.15645011075283216, 0.19217229897749472, 0.2250702033313609, 0.2548797983308887, 0.2813293712087751, 0.30420766505798474, 0.32321880946406617, 0.33826802101514303, 0.34917827960626036, 0.3556589542210297, 0.35779681725132184, 0.35565895422102967, 0.3491782796062603, 0.3382680210151428, 0.3232188094640659, 0.3042076650579844, 0.2813293712087748, 0.2548797983308885, 0.22507020333136085, 0.1921722989774947, 0.1564501107528321, 0.11817993343097208, 0.07766478074845945, 0.075329337954368, 0.11462311713123972, 0.15173199776047533, 0.18636112398825833, 0.2182473780344291, 0.2471205778550471, 0.2727617209937379, 0.2948636707750291, 0.3133105279511887, 0.3278999450140934, 0.33826802101514303, 0.34454081452475155, 0.34669614196424364, 0.3445408145247516, 0.33826802101514286, 0.32789994501409314, 0.3133105279511885, 0.2948636707750289, 0.2727617209937377, 0.24712057785504687, 0.21824737803442884, 0.18636112398825821, 0.15173199776047533, 0.11462311713123967, 0.07532933795436793, 0.0720951483853463, 0.10969794646838658, 0.14519704389889537, 0.17831532245918832, 0.20878732915049014, 0.2363958082518836, 0.2608347764396367, 0.28198376832678224, 0.29960847209766983, 0.3133105279511886, 0.32321880946406595, 0.3292494667036455, 0.3312285055556795, 0.32924946670364563, 0.32321880946406606, 0.3133105279511885, 0.2996084720976697, 0.28198376832678207, 0.2608347764396365, 0.2363958082518834, 0.20878732915049, 0.17831532245918827, 0.14519704389889534, 0.10969794646838664, 0.07209514838534628, 0.06799724531557692, 0.10345581477898914, 0.13691796221646116, 0.1681088246504622, 0.1968153887209563, 0.22274385294262114, 0.24578147642763282, 0.26566827591237974, 0.28198376832678224, 0.294863670775029, 0.30420766505798447, 0.30981520997629586, 0.3117071758004514, 0.309815209976296, 0.3042076650579846, 0.29486367077502895, 0.28198376832678207, 0.26566827591237946, 0.24578147642763268, 0.22274385294262103, 0.1968153887209562, 0.16810882465046217, 0.1369179622164611, 0.10345581477898914, 0.06799724531557695, 0.06307779967625264, 0.09596646083873434, 0.12697133089098292, 0.15587030482414788, 0.18238584389256474, 0.2064219938972198, 0.22769822236798282, 0.24578147642763276, 0.2608347764396366, 0.2727617209937378, 0.28132937120877494, 0.2865233375873759, 0.2882555932407505, 0.28652333758737614, 0.28132937120877494, 0.2727617209937377, 0.26083477643963643, 0.2457814764276326, 0.22769822236798273, 0.20642199389721966, 0.18238584389256474, 0.1558703048241478, 0.12697133089098286, 0.0959664608387343, 0.06307779967625263, 0.057393964870959734, 0.08729947508349435, 0.1154796653441296, 0.1416628181081839, 0.165770827036091, 0.1875026633542146, 0.20642199389721977, 0.22274385294262092, 0.23639580825188347, 0.24712057785504693, 0.25487979833088853, 0.25956206788500874, 0.2611295229965441, 0.2595620678850088, 0.25487979833088864, 0.24712057785504699, 0.2363958082518835, 0.22274385294262092, 0.2064219938972196, 0.18750266335421453, 0.16577082703609092, 0.14166281810818385, 0.11547966534412955, 0.08729947508349435, 0.05739396487095975, 0.05100072704478549, 0.07756460151772669, 0.10251035073030709, 0.12576424655216104, 0.14699529578237644, 0.16577082703609092, 0.18238584389256463, 0.19681538872095608, 0.20878732915048992, 0.2182473780344288, 0.22507020333136082, 0.22919492827762483, 0.23057461282062058, 0.2291949282776249, 0.22507020333136082, 0.21824737803442895, 0.20878732915049003, 0.19681538872095614, 0.18238584389256465, 0.16577082703609092, 0.14699529578237638, 0.125764246552161, 0.10251035073030706, 0.07756460151772666, 0.05100072704478548, 0.04399260475136242, 0.06683400942917608, 0.08833946431758964, 0.10812515410923533, 0.12576424655216106, 0.1416628181081838, 0.15587030482414774, 0.16810882465046206, 0.1783153224591881, 0.18636112398825808, 0.19217229897749458, 0.19568374851974465, 0.19685842427526834, 0.19568374851974465, 0.19217229897749458, 0.1863611239882582, 0.17831532245918824, 0.1681088246504621, 0.1558703048241478, 0.14166281810818382, 0.12576424655216106, 0.10812515410923529, 0.08833946431758967, 0.06683400942917608, 0.043992604751362414, 0.03645151076496128, 0.055376276584935595, 0.07281240864873831, 0.0883394643175896, 0.102510350730307, 0.11547966534412953, 0.12697133089098286, 0.13691796221646105, 0.14519704389889526, 0.15173199776047522, 0.15645011075283208, 0.1593017009001521, 0.16025573986215125, 0.1593017009001521, 0.1564501107528321, 0.15173199776047538, 0.14519704389889532, 0.1369179622164611, 0.12697133089098284, 0.11547966534412958, 0.10251035073030708, 0.08833946431758967, 0.07281240864873834, 0.055376276584935574, 0.036451510764961274, 0.028624451099833324, 0.04289219826344357, 0.055376276584935574, 0.06683400942917601, 0.07756460151772661, 0.08729947508349428, 0.09596646083873418, 0.10345581477898907, 0.10969794646838647, 0.11462311713123963, 0.11817993343097204, 0.12032968475616049, 0.12104887790598168, 0.12032968475616052, 0.11817993343097209, 0.11462311713123971, 0.10969794646838653, 0.10345581477898909, 0.09596646083873425, 0.0872994750834943, 0.07756460151772665, 0.06683400942917607, 0.0553762765849356, 0.04289219826344358, 0.028624451099833324, 0.01998505220506658, 0.028624451099833328, 0.03645151076496123, 0.04399260475136236, 0.05100072704478543, 0.05739396487095966, 0.06307779967625257, 0.06799724531557688, 0.07209514838534621, 0.07532933795436791, 0.07766478074845945, 0.07907628103283927, 0.07954850885838662, 0.07907628103283931, 0.07766478074845949, 0.0753293379543679, 0.07209514838534624, 0.0679972453155769, 0.06307779967625259, 0.05739396487095971, 0.05100072704478544, 0.0439926047513624, 0.036451510764961274, 0.028624451099833328, 0.019985052205066585]]]

        phi_test = np.array(phi_test)

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, 0.23385815, 8)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()
            phi_zero_test = phi_test[:, l].flatten() / np.linalg.norm(phi_test[:, l]) * np.linalg.norm(phi)
            np.testing.assert_array_almost_equal(phi, phi_zero_test, 7)

        self.angular_test()

    def test_solver_basic_2D_2g_1a_vacuum_eigen(self):
        '''
        Test for a basic 2 group problem
        '''
        pydgm.control.fine_mesh_x = [25, 25]
        pydgm.control.fine_mesh_y = [25]
        pydgm.control.coarse_mesh_x = [0.0, 5.0, 10.0]
        pydgm.control.coarse_mesh_y = [0.0, 10.0]
        pydgm.control.material_map = [1, 5]
        pydgm.control.angle_order = 2
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 0.0
        pydgm.control.boundary_north = 0.0
        pydgm.control.boundary_south = 0.0
        pydgm.control.allow_fission = True
        pydgm.control.solver_type = 'eigen'.ljust(256)
        pydgm.control.source_value = 0.0
        pydgm.control.eigen_tolerance = 1e-12
        pydgm.control.outer_tolerance = 1e-12

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[0.2487746067595804, 0.30933535885463925, 0.36503957161582756, 0.4201277246266406, 0.4727429041064538, 0.5219789831678958, 0.5666972948641681, 0.6060354489662487, 0.6392379836284101, 0.6656917181028958, 0.6849238962362818, 0.6965974817500068, 0.7005110260557053, 0.6965974817500072, 0.6849238962362818, 0.6656917181028954, 0.6392379836284099, 0.6060354489662486, 0.5666972948641676, 0.5219789831678954, 0.47274290410645325, 0.4201277246266398, 0.3650395716158268, 0.30933535885463875, 0.24877460675958005, 0.3093353588546392, 0.40164987296136834, 0.48291526870614243, 0.5584914296565835, 0.6312485387266421, 0.6988077036867092, 0.7601167557294932, 0.813912097913677, 0.8592449558483541, 0.8953214292816684, 0.92152628635136, 0.9374235947881444, 0.9427517097445433, 0.9374235947881437, 0.9215262863513597, 0.8953214292816681, 0.8592449558483536, 0.8139120979136766, 0.7601167557294923, 0.698807703686708, 0.6312485387266414, 0.5584914296565827, 0.48291526870614165, 0.40164987296136784, 0.3093353588546388, 0.36503957161582756, 0.4829152687061422, 0.5944451912542015, 0.6940912164324666, 0.7860949620587045, 0.8723187135451134, 0.9500816973490145, 1.0183122596044265, 1.0757218667223183, 1.1213682123712332, 1.1545053937082992, 1.174599589924656, 1.1813330856367277, 1.1745995899246553, 1.1545053937082987, 1.121368212371233, 1.0757218667223176, 1.0183122596044252, 0.950081697349013, 0.8723187135451121, 0.7860949620587036, 0.694091216432466, 0.5944451912542007, 0.48291526870614176, 0.36503957161582723, 0.4201277246266406, 0.5584914296565837, 0.6940912164324667, 0.8215542114308743, 0.9352477382783926, 1.0386119631536603, 1.132791234434996, 1.2149670473854255, 1.284144202657454, 1.3390943477694461, 1.3789641487964708, 1.4031352581076166, 1.4112332698908574, 1.4031352581076166, 1.3789641487964701, 1.3390943477694455, 1.2841442026574532, 1.2149670473854237, 1.1327912344349942, 1.0386119631536588, 0.9352477382783915, 0.8215542114308731, 0.6940912164324659, 0.5584914296565832, 0.4201277246266401, 0.47274290410645387, 0.6312485387266424, 0.7860949620587048, 0.9352477382783928, 1.0736610472994341, 1.1957589812484655, 1.3044202797066318, 1.4002986998355933, 1.480566815593873, 1.5443844450713775, 1.590659651713007, 1.618703820457089, 1.6280998521423597, 1.6187038204570887, 1.5906596517130063, 1.5443844450713762, 1.4805668155938718, 1.400298699835592, 1.3044202797066304, 1.1957589812484641, 1.0736610472994326, 0.9352477382783914, 0.7860949620587037, 0.6312485387266413, 0.47274290410645303, 0.5219789831678961, 0.6988077036867091, 0.8723187135451133, 1.0386119631536603, 1.1957589812484652, 1.3391074138222738, 1.463235534944001, 1.5706511446747617, 1.661703820432259, 1.7336628995791248, 1.785911181959954, 1.8175629277589989, 1.8281624800473772, 1.8175629277589982, 1.7859111819599531, 1.7336628995791235, 1.6617038204322576, 1.5706511446747609, 1.463235534944001, 1.3391074138222734, 1.195758981248464, 1.0386119631536588, 0.872318713545112, 0.6988077036867077, 0.5219789831678953, 0.5666972948641688, 0.7601167557294939, 0.9500816973490143, 1.1327912344349955, 1.304420279706632, 1.4632355349440012, 1.6050104613189407, 1.7245450833554363, 1.8241342432960628, 1.9039815054710298, 1.9615421041215386, 1.9964802438076725, 2.0081817572886504, 1.9964802438076716, 1.9615421041215384, 1.9039815054710294, 1.8241342432960614, 1.724545083355435, 1.6050104613189402, 1.4632355349440005, 1.3044202797066307, 1.132791234434994, 0.9500816973490128, 0.760116755729492, 0.5666972948641678, 0.6060354489662493, 0.8139120979136776, 1.0183122596044265, 1.2149670473854246, 1.400298699835593, 1.5706511446747617, 1.724545083355436, 1.8581966015787008, 1.9666783752660877, 2.0522072216331457, 2.114975856603474, 2.152703108428814, 2.1653793244298796, 2.1527031084288137, 2.114975856603474, 2.0522072216331453, 1.9666783752660866, 1.8581966015786995, 1.7245450833554352, 1.570651144674761, 1.4002986998355924, 1.2149670473854235, 1.018312259604425, 0.8139120979136766, 0.6060354489662485, 0.6392379836284107, 0.859244955848355, 1.0757218667223187, 1.2841442026574539, 1.4805668155938723, 1.661703820432259, 1.8241342432960632, 1.9666783752660877, 2.0860050936007664, 2.1774982372953064, 2.243415389846528, 2.284049071070629, 2.297430887485442, 2.2840490710706294, 2.243415389846527, 2.1774982372953033, 2.086005093600764, 1.9666783752660864, 1.824134243296061, 1.6617038204322578, 1.4805668155938714, 1.2841442026574532, 1.0757218667223174, 0.8592449558483536, 0.6392379836284099, 0.6656917181028967, 0.8953214292816696, 1.1213682123712334, 1.3390943477694452, 1.5443844450713782, 1.7336628995791252, 1.9039815054710307, 2.0522072216331466, 2.177498237295307, 2.276984903871456, 2.3463849560806116, 2.3880835394753026, 2.402706436696479, 2.388083539475304, 2.3463849560806103, 2.276984903871452, 2.1774982372953047, 2.0522072216331444, 1.9039815054710287, 1.7336628995791237, 1.5443844450713762, 1.3390943477694448, 1.1213682123712327, 0.8953214292816684, 0.6656917181028957, 0.6849238962362824, 0.9215262863513608, 1.1545053937082996, 1.3789641487964708, 1.590659651713007, 1.7859111819599531, 1.9615421041215388, 2.1149758566034755, 2.2434153898465286, 2.3463849560806116, 2.4214652692362337, 2.4648625085214766, 2.4785988981976548, 2.464862508521476, 2.421465269236231, 2.34638495608061, 2.2434153898465263, 2.114975856603473, 1.9615421041215368, 1.7859111819599527, 1.590659651713006, 1.37896414879647, 1.1545053937082985, 0.9215262863513604, 0.684923896236282, 0.6965974817500075, 0.9374235947881446, 1.1745995899246564, 1.4031352581076177, 1.6187038204570894, 1.8175629277589984, 1.9964802438076719, 2.1527031084288146, 2.2840490710706303, 2.3880835394753044, 2.4648625085214766, 2.5118370812605506, 2.5268458313707605, 2.511837081260549, 2.4648625085214757, 2.3880835394753026, 2.2840490710706276, 2.152703108428813, 1.9964802438076716, 1.8175629277589982, 1.6187038204570894, 1.4031352581076162, 1.1745995899246557, 0.9374235947881441, 0.6965974817500072, 0.7005110260557053, 0.9427517097445437, 1.1813330856367281, 1.411233269890858, 1.6280998521423609, 1.8281624800473772, 2.008181757288651, 2.16537932442988, 2.2974308874854423, 2.4027064366964797, 2.478598898197655, 2.526845831370762, 2.544995566124512, 2.526845831370761, 2.478598898197654, 2.4027064366964774, 2.29743088748544, 2.1653793244298787, 2.008181757288651, 1.828162480047377, 1.6280998521423604, 1.4112332698908572, 1.1813330856367281, 0.942751709744544, 0.7005110260557053, 0.696597481750007, 0.9374235947881437, 1.1745995899246553, 1.4031352581076164, 1.6187038204570892, 1.817562927758998, 1.9964802438076719, 2.152703108428814, 2.284049071070629, 2.3880835394753035, 2.4648625085214775, 2.5118370812605497, 2.526845831370762, 2.511837081260549, 2.4648625085214744, 2.388083539475302, 2.284049071070627, 2.1527031084288137, 1.9964802438076714, 1.8175629277589977, 1.6187038204570898, 1.403135258107617, 1.1745995899246562, 0.9374235947881443, 0.6965974817500074, 0.6849238962362815, 0.9215262863513597, 1.1545053937082983, 1.37896414879647, 1.5906596517130056, 1.785911181959953, 1.9615421041215382, 2.1149758566034746, 2.2434153898465277, 2.346384956080611, 2.4214652692362324, 2.464862508521476, 2.478598898197654, 2.464862508521475, 2.4214652692362306, 2.34638495608061, 2.2434153898465268, 2.114975856603473, 1.9615421041215373, 1.7859111819599534, 1.5906596517130067, 1.3789641487964701, 1.154505393708299, 0.9215262863513605, 0.6849238962362822, 0.6656917181028956, 0.8953214292816684, 1.1213682123712325, 1.3390943477694441, 1.5443844450713762, 1.7336628995791235, 1.9039815054710296, 2.0522072216331453, 2.177498237295306, 2.2769849038714542, 2.3463849560806107, 2.388083539475303, 2.4027064366964783, 2.3880835394753013, 2.3463849560806094, 2.2769849038714542, 2.1774982372953042, 2.0522072216331435, 1.903981505471029, 1.733662899579124, 1.5443844450713766, 1.3390943477694452, 1.1213682123712334, 0.8953214292816688, 0.6656917181028961, 0.6392379836284101, 0.8592449558483539, 1.0757218667223172, 1.2841442026574528, 1.4805668155938716, 1.6617038204322572, 1.824134243296061, 1.966678375266086, 2.0860050936007646, 2.177498237295305, 2.243415389846527, 2.2840490710706276, 2.2974308874854406, 2.2840490710706276, 2.2434153898465268, 2.1774982372953056, 2.0860050936007637, 1.966678375266086, 1.8241342432960603, 1.6617038204322567, 1.4805668155938712, 1.2841442026574534, 1.0757218667223183, 0.8592449558483548, 0.6392379836284101, 0.6060354489662485, 0.8139120979136767, 1.0183122596044252, 1.214967047385424, 1.4002986998355924, 1.5706511446747604, 1.724545083355434, 1.8581966015786986, 1.966678375266085, 2.052207221633143, 2.114975856603472, 2.1527031084288124, 2.165379324429878, 2.152703108428813, 2.1149758566034733, 2.0522072216331444, 1.9666783752660861, 1.8581966015786993, 1.7245450833554339, 1.5706511446747597, 1.400298699835592, 1.2149670473854246, 1.018312259604426, 0.8139120979136778, 0.6060354489662493, 0.5666972948641678, 0.7601167557294926, 0.9500816973490129, 1.132791234434994, 1.3044202797066307, 1.4632355349439998, 1.6050104613189393, 1.7245450833554337, 1.8241342432960592, 1.9039815054710265, 1.9615421041215355, 1.99648024380767, 2.0081817572886496, 1.9964802438076705, 1.9615421041215368, 1.9039815054710283, 1.8241342432960603, 1.724545083355434, 1.6050104613189389, 1.4632355349439996, 1.3044202797066304, 1.1327912344349949, 0.9500816973490139, 0.7601167557294932, 0.5666972948641683, 0.5219789831678955, 0.6988077036867083, 0.8723187135451121, 1.0386119631536588, 1.1957589812484641, 1.3391074138222725, 1.4632355349439998, 1.5706511446747595, 1.6617038204322556, 1.733662899579122, 1.7859111819599514, 1.8175629277589964, 1.828162480047376, 1.8175629277589962, 1.785911181959952, 1.7336628995791221, 1.6617038204322565, 1.5706511446747602, 1.4632355349439996, 1.3391074138222723, 1.195758981248464, 1.0386119631536592, 0.8723187135451124, 0.6988077036867084, 0.5219789831678958, 0.47274290410645325, 0.6312485387266413, 0.7860949620587037, 0.9352477382783914, 1.0736610472994323, 1.1957589812484632, 1.3044202797066298, 1.4002986998355915, 1.4805668155938698, 1.5443844450713744, 1.5906596517130036, 1.618703820457087, 1.628099852142358, 1.6187038204570874, 1.5906596517130043, 1.5443844450713748, 1.48056681559387, 1.400298699835591, 1.3044202797066302, 1.1957589812484637, 1.0736610472994326, 0.9352477382783915, 0.7860949620587039, 0.6312485387266413, 0.4727429041064536, 0.42012772462664005, 0.558491429656583, 0.6940912164324656, 0.821554211430873, 0.9352477382783911, 1.0386119631536583, 1.1327912344349942, 1.214967047385423, 1.2841442026574517, 1.3390943477694435, 1.3789641487964681, 1.4031352581076144, 1.411233269890855, 1.4031352581076149, 1.378964148796469, 1.3390943477694435, 1.2841442026574519, 1.214967047385423, 1.1327912344349942, 1.0386119631536583, 0.9352477382783914, 0.8215542114308733, 0.6940912164324659, 0.5584914296565829, 0.4201277246266399, 0.3650395716158271, 0.4829152687061416, 0.5944451912542004, 0.6940912164324654, 0.7860949620587033, 0.872318713545112, 0.950081697349013, 1.0183122596044247, 1.0757218667223167, 1.1213682123712319, 1.1545053937082967, 1.1745995899246537, 1.181333085636726, 1.1745995899246535, 1.1545053937082974, 1.1213682123712316, 1.0757218667223165, 1.0183122596044245, 0.9500816973490126, 0.8723187135451116, 0.7860949620587037, 0.6940912164324659, 0.5944451912542006, 0.4829152687061415, 0.36503957161582695, 0.3093353588546387, 0.4016498729613677, 0.4829152687061415, 0.5584914296565826, 0.6312485387266412, 0.6988077036867079, 0.7601167557294921, 0.8139120979136765, 0.8592449558483536, 0.8953214292816674, 0.9215262863513591, 0.9374235947881425, 0.9427517097445415, 0.9374235947881424, 0.9215262863513588, 0.8953214292816672, 0.8592449558483534, 0.8139120979136764, 0.760116755729492, 0.6988077036867076, 0.6312485387266412, 0.5584914296565827, 0.4829152687061414, 0.4016498729613678, 0.3093353588546387, 0.24877460675957994, 0.3093353588546386, 0.36503957161582684, 0.42012772462663983, 0.47274290410645287, 0.5219789831678951, 0.5666972948641678, 0.6060354489662485, 0.6392379836284093, 0.6656917181028953, 0.6849238962362811, 0.6965974817500061, 0.7005110260557043, 0.6965974817500061, 0.684923896236281, 0.6656917181028952, 0.6392379836284097, 0.606035448966248, 0.5666972948641674, 0.5219789831678951, 0.47274290410645314, 0.42012772462663994, 0.3650395716158269, 0.3093353588546386, 0.24877460675958013]], [[0.01998505220506663, 0.028624451099833366, 0.03645151076496133, 0.04399260475136248, 0.05100072704478553, 0.05739396487095979, 0.06307779967625268, 0.067997245315577, 0.07209514838534635, 0.07532933795436803, 0.0776647807484596, 0.07907628103283944, 0.07954850885838675, 0.07907628103283944, 0.07766478074845957, 0.075329337954368, 0.07209514838534632, 0.06799724531557698, 0.06307779967625264, 0.057393964870959754, 0.05100072704478547, 0.043992604751362394, 0.036451510764961274, 0.028624451099833345, 0.019985052205066602, 0.028624451099833387, 0.04289219826344366, 0.05537627658493564, 0.06683400942917612, 0.07756460151772676, 0.08729947508349442, 0.09596646083873438, 0.10345581477898921, 0.10969794646838663, 0.11462311713123982, 0.1181799334309722, 0.12032968475616068, 0.12104887790598191, 0.12032968475616067, 0.1181799334309722, 0.11462311713123977, 0.1096979464683866, 0.10345581477898919, 0.09596646083873432, 0.08729947508349438, 0.07756460151772669, 0.06683400942917608, 0.055376276584935595, 0.04289219826344358, 0.02862445109983335, 0.036451510764961344, 0.05537627658493567, 0.07281240864873842, 0.08833946431758972, 0.10251035073030715, 0.11547966534412975, 0.12697133089098303, 0.13691796221646127, 0.1451970438988955, 0.1517319977604755, 0.1564501107528323, 0.15930170090015233, 0.16025573986215147, 0.15930170090015233, 0.15645011075283227, 0.1517319977604754, 0.1451970438988954, 0.13691796221646116, 0.126971330890983, 0.11547966534412962, 0.10251035073030706, 0.08833946431758967, 0.07281240864873835, 0.0553762765849356, 0.03645151076496128, 0.04399260475136246, 0.06683400942917615, 0.08833946431758974, 0.10812515410923546, 0.12576424655216117, 0.14166281810818404, 0.15587030482414796, 0.16810882465046229, 0.17831532245918838, 0.18636112398825838, 0.19217229897749483, 0.19568374851974488, 0.1968584242752685, 0.19568374851974477, 0.1921722989774947, 0.1863611239882583, 0.1783153224591883, 0.16810882465046217, 0.15587030482414788, 0.1416628181081839, 0.12576424655216112, 0.10812515410923533, 0.08833946431758966, 0.06683400942917607, 0.04399260475136238, 0.05100072704478551, 0.07756460151772678, 0.10251035073030718, 0.1257642465521612, 0.14699529578237652, 0.16577082703609106, 0.18238584389256482, 0.19681538872095647, 0.20878732915049025, 0.2182473780344291, 0.225070203331361, 0.22919492827762505, 0.23057461282062075, 0.22919492827762494, 0.22507020333136094, 0.21824737803442895, 0.20878732915049006, 0.19681538872095633, 0.18238584389256474, 0.16577082703609097, 0.14699529578237644, 0.1257642465521611, 0.10251035073030706, 0.07756460151772662, 0.05100072704478543, 0.057393964870959775, 0.08729947508349441, 0.11547966534412972, 0.14166281810818399, 0.16577082703609108, 0.18750266335421473, 0.20642199389721985, 0.22274385294262125, 0.23639580825188378, 0.24712057785504718, 0.2548797983308887, 0.2595620678850089, 0.26112952299654413, 0.25956206788500885, 0.25487979833088864, 0.24712057785504704, 0.23639580825188358, 0.22274385294262108, 0.20642199389721977, 0.1875026633542146, 0.16577082703609095, 0.14166281810818385, 0.11547966534412953, 0.08729947508349425, 0.05739396487095971, 0.06307779967625268, 0.0959664608387344, 0.126971330890983, 0.15587030482414793, 0.1823858438925648, 0.2064219938972199, 0.22769822236798304, 0.24578147642763298, 0.2608347764396368, 0.2727617209937381, 0.28132937120877516, 0.28652333758737614, 0.2882555932407505, 0.28652333758737614, 0.28132937120877494, 0.2727617209937379, 0.2608347764396366, 0.24578147642763265, 0.22769822236798287, 0.2064219938972197, 0.18238584389256465, 0.1558703048241477, 0.1269713308909828, 0.09596646083873425, 0.0630777996762526, 0.06799724531557702, 0.10345581477898926, 0.13691796221646121, 0.16810882465046226, 0.19681538872095627, 0.22274385294262114, 0.24578147642763293, 0.26566827591237996, 0.28198376832678235, 0.2948636707750291, 0.3042076650579847, 0.30981520997629597, 0.3117071758004514, 0.3098152099762959, 0.3042076650579845, 0.29486367077502895, 0.2819837683267821, 0.26566827591237957, 0.24578147642763262, 0.22274385294262097, 0.1968153887209561, 0.16810882465046204, 0.13691796221646108, 0.10345581477898912, 0.06799724531557694, 0.07209514838534634, 0.10969794646838667, 0.14519704389889548, 0.1783153224591884, 0.20878732915049014, 0.23639580825188367, 0.2608347764396368, 0.2819837683267825, 0.29960847209767005, 0.31331052795118874, 0.32321880946406617, 0.3292494667036456, 0.33122850555567945, 0.3292494667036456, 0.323218809464066, 0.31331052795118847, 0.29960847209766955, 0.28198376832678207, 0.2608347764396365, 0.2363958082518834, 0.20878732915049, 0.17831532245918816, 0.14519704389889518, 0.10969794646838649, 0.07209514838534624, 0.07532933795436803, 0.11462311713123982, 0.15173199776047552, 0.1863611239882584, 0.2182473780344291, 0.24712057785504715, 0.27276172099373797, 0.29486367077502923, 0.31331052795118886, 0.3278999450140935, 0.3382680210151431, 0.3445408145247518, 0.3466961419642437, 0.34454081452475166, 0.33826802101514275, 0.327899945014093, 0.3133105279511884, 0.29486367077502884, 0.2727617209937377, 0.2471205778550469, 0.21824737803442884, 0.18636112398825816, 0.15173199776047527, 0.11462311713123965, 0.07532933795436791, 0.07766478074845957, 0.11817993343097222, 0.1564501107528323, 0.19217229897749483, 0.22507020333136107, 0.25487979833088875, 0.28132937120877516, 0.3042076650579848, 0.32321880946406634, 0.3382680210151432, 0.3491782796062605, 0.3556589542210299, 0.357796817251322, 0.3556589542210296, 0.34917827960626024, 0.33826802101514264, 0.32321880946406584, 0.3042076650579843, 0.2813293712087748, 0.2548797983308885, 0.22507020333136074, 0.19217229897749455, 0.15645011075283208, 0.11817993343097205, 0.07766478074845946, 0.07907628103283938, 0.12032968475616064, 0.15930170090015233, 0.1956837485197449, 0.22919492827762514, 0.2595620678850089, 0.2865233375873762, 0.3098152099762961, 0.32924946670364585, 0.344540814524752, 0.35565895422102983, 0.36240803074961725, 0.36459041779744544, 0.36240803074961697, 0.35565895422102956, 0.3445408145247515, 0.3292494667036454, 0.30981520997629564, 0.2865233375873758, 0.2595620678850086, 0.2291949282776248, 0.19568374851974465, 0.1593017009001521, 0.12032968475616053, 0.07907628103283933, 0.0795485088583867, 0.12104887790598186, 0.16025573986215144, 0.19685842427526862, 0.23057461282062083, 0.2611295229965441, 0.2882555932407507, 0.3117071758004516, 0.3312285055556798, 0.3466961419642438, 0.3577968172513219, 0.36459041779744533, 0.36700813351718253, 0.3645904177974454, 0.3577968172513218, 0.3466961419642433, 0.3312285055556792, 0.31170717580045104, 0.28825559324075023, 0.2611295229965439, 0.23057461282062056, 0.19685842427526837, 0.16025573986215128, 0.12104887790598169, 0.07954850885838664, 0.07907628103283937, 0.1203296847561606, 0.15930170090015222, 0.19568374851974482, 0.229194928277625, 0.2595620678850089, 0.28652333758737614, 0.30981520997629614, 0.3292494667036457, 0.34454081452475166, 0.3556589542210297, 0.3624080307496171, 0.3645904177974454, 0.3624080307496171, 0.3556589542210296, 0.34454081452475144, 0.3292494667036452, 0.3098152099762958, 0.2865233375873758, 0.25956206788500863, 0.22919492827762492, 0.19568374851974468, 0.15930170090015214, 0.12032968475616056, 0.0790762810328393, 0.07766478074845959, 0.11817993343097216, 0.15645011075283216, 0.19217229897749472, 0.2250702033313609, 0.2548797983308887, 0.2813293712087751, 0.30420766505798474, 0.32321880946406617, 0.33826802101514303, 0.34917827960626036, 0.3556589542210297, 0.35779681725132184, 0.35565895422102967, 0.3491782796062603, 0.3382680210151428, 0.3232188094640659, 0.3042076650579844, 0.2813293712087748, 0.2548797983308885, 0.22507020333136085, 0.1921722989774947, 0.1564501107528321, 0.11817993343097208, 0.07766478074845945, 0.075329337954368, 0.11462311713123972, 0.15173199776047533, 0.18636112398825833, 0.2182473780344291, 0.2471205778550471, 0.2727617209937379, 0.2948636707750291, 0.3133105279511887, 0.3278999450140934, 0.33826802101514303, 0.34454081452475155, 0.34669614196424364, 0.3445408145247516, 0.33826802101514286, 0.32789994501409314, 0.3133105279511885, 0.2948636707750289, 0.2727617209937377, 0.24712057785504687, 0.21824737803442884, 0.18636112398825821, 0.15173199776047533, 0.11462311713123967, 0.07532933795436793, 0.0720951483853463, 0.10969794646838658, 0.14519704389889537, 0.17831532245918832, 0.20878732915049014, 0.2363958082518836, 0.2608347764396367, 0.28198376832678224, 0.29960847209766983, 0.3133105279511886, 0.32321880946406595, 0.3292494667036455, 0.3312285055556795, 0.32924946670364563, 0.32321880946406606, 0.3133105279511885, 0.2996084720976697, 0.28198376832678207, 0.2608347764396365, 0.2363958082518834, 0.20878732915049, 0.17831532245918827, 0.14519704389889534, 0.10969794646838664, 0.07209514838534628, 0.06799724531557692, 0.10345581477898914, 0.13691796221646116, 0.1681088246504622, 0.1968153887209563, 0.22274385294262114, 0.24578147642763282, 0.26566827591237974, 0.28198376832678224, 0.294863670775029, 0.30420766505798447, 0.30981520997629586, 0.3117071758004514, 0.309815209976296, 0.3042076650579846, 0.29486367077502895, 0.28198376832678207, 0.26566827591237946, 0.24578147642763268, 0.22274385294262103, 0.1968153887209562, 0.16810882465046217, 0.1369179622164611, 0.10345581477898914, 0.06799724531557695, 0.06307779967625264, 0.09596646083873434, 0.12697133089098292, 0.15587030482414788, 0.18238584389256474, 0.2064219938972198, 0.22769822236798282, 0.24578147642763276, 0.2608347764396366, 0.2727617209937378, 0.28132937120877494, 0.2865233375873759, 0.2882555932407505, 0.28652333758737614, 0.28132937120877494, 0.2727617209937377, 0.26083477643963643, 0.2457814764276326, 0.22769822236798273, 0.20642199389721966, 0.18238584389256474, 0.1558703048241478, 0.12697133089098286, 0.0959664608387343, 0.06307779967625263, 0.057393964870959734, 0.08729947508349435, 0.1154796653441296, 0.1416628181081839, 0.165770827036091, 0.1875026633542146, 0.20642199389721977, 0.22274385294262092, 0.23639580825188347, 0.24712057785504693, 0.25487979833088853, 0.25956206788500874, 0.2611295229965441, 0.2595620678850088, 0.25487979833088864, 0.24712057785504699, 0.2363958082518835, 0.22274385294262092, 0.2064219938972196, 0.18750266335421453, 0.16577082703609092, 0.14166281810818385, 0.11547966534412955, 0.08729947508349435, 0.05739396487095975, 0.05100072704478549, 0.07756460151772669, 0.10251035073030709, 0.12576424655216104, 0.14699529578237644, 0.16577082703609092, 0.18238584389256463, 0.19681538872095608, 0.20878732915048992, 0.2182473780344288, 0.22507020333136082, 0.22919492827762483, 0.23057461282062058, 0.2291949282776249, 0.22507020333136082, 0.21824737803442895, 0.20878732915049003, 0.19681538872095614, 0.18238584389256465, 0.16577082703609092, 0.14699529578237638, 0.125764246552161, 0.10251035073030706, 0.07756460151772666, 0.05100072704478548, 0.04399260475136242, 0.06683400942917608, 0.08833946431758964, 0.10812515410923533, 0.12576424655216106, 0.1416628181081838, 0.15587030482414774, 0.16810882465046206, 0.1783153224591881, 0.18636112398825808, 0.19217229897749458, 0.19568374851974465, 0.19685842427526834, 0.19568374851974465, 0.19217229897749458, 0.1863611239882582, 0.17831532245918824, 0.1681088246504621, 0.1558703048241478, 0.14166281810818382, 0.12576424655216106, 0.10812515410923529, 0.08833946431758967, 0.06683400942917608, 0.043992604751362414, 0.03645151076496128, 0.055376276584935595, 0.07281240864873831, 0.0883394643175896, 0.102510350730307, 0.11547966534412953, 0.12697133089098286, 0.13691796221646105, 0.14519704389889526, 0.15173199776047522, 0.15645011075283208, 0.1593017009001521, 0.16025573986215125, 0.1593017009001521, 0.1564501107528321, 0.15173199776047538, 0.14519704389889532, 0.1369179622164611, 0.12697133089098284, 0.11547966534412958, 0.10251035073030708, 0.08833946431758967, 0.07281240864873834, 0.055376276584935574, 0.036451510764961274, 0.028624451099833324, 0.04289219826344357, 0.055376276584935574, 0.06683400942917601, 0.07756460151772661, 0.08729947508349428, 0.09596646083873418, 0.10345581477898907, 0.10969794646838647, 0.11462311713123963, 0.11817993343097204, 0.12032968475616049, 0.12104887790598168, 0.12032968475616052, 0.11817993343097209, 0.11462311713123971, 0.10969794646838653, 0.10345581477898909, 0.09596646083873425, 0.0872994750834943, 0.07756460151772665, 0.06683400942917607, 0.0553762765849356, 0.04289219826344358, 0.028624451099833324, 0.01998505220506658, 0.028624451099833328, 0.03645151076496123, 0.04399260475136236, 0.05100072704478543, 0.05739396487095966, 0.06307779967625257, 0.06799724531557688, 0.07209514838534621, 0.07532933795436791, 0.07766478074845945, 0.07907628103283927, 0.07954850885838662, 0.07907628103283931, 0.07766478074845949, 0.0753293379543679, 0.07209514838534624, 0.0679972453155769, 0.06307779967625259, 0.05739396487095971, 0.05100072704478544, 0.0439926047513624, 0.036451510764961274, 0.028624451099833328, 0.019985052205066585]]]

        phi_test = np.array(phi_test)

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.keff, 0.23385815, 8)

        # Test the scalar flux
        for l in range(pydgm.control.scatter_leg_order + 1):
            phi = pydgm.state.mg_phi[l, :, :].flatten()
            phi_zero_test = phi_test[:, l].flatten() / np.linalg.norm(phi_test[:, l]) * np.linalg.norm(phi)
            np.testing.assert_array_almost_equal(phi, phi_zero_test, 7)

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
        pydgm.control.angle_order = 2

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[49.09993934183404, 48.54663511219223, 47.43509578792542, 45.787200636481046, 43.60877816499753, 40.9212915335, 37.75182058961189, 34.140082497261, 30.191877880419955, 26.386444641690034, 22.66129067106905, 18.673843108107757, 15.14199701364031, 12.40586283195162, 10.20035679307175, 8.44569932921349, 7.05147890638834, 5.936750093300124, 5.107981178146741, 4.2807106570945885, 2.7110599066516543, 1.184191681919207, 0.5229537127950984, 0.23095828441376742, 0.09905825469791465, 0.033289751219464224, 48.54663511219221, 47.99149555027915, 46.90881865514066, 45.275560058246135, 43.129868366272916, 40.48138749698717, 37.35375194753773, 33.78959075857193, 29.894107149893987, 26.137075917150106, 22.45808316535814, 18.522308807994843, 15.02837006104685, 12.32839254237998, 10.146011463073062, 8.411531164056427, 7.033073310639177, 5.924391503768822, 5.108013275278842, 4.280548947242698, 2.713615139701458, 1.186913486616884, 0.5234442803371963, 0.23226366394768425, 0.09879722469315545, 0.03352592026974592, 47.435095787925356, 46.90881865514063, 45.83770519415298, 44.26802848965667, 42.17610660412243, 39.6014235966295, 36.56278788548342, 33.09458289139056, 29.299927477074913, 25.642581486317948, 22.060324145471967, 18.216207138689487, 14.814023829685036, 12.171270833817069, 10.045147499112177, 8.348062407768406, 6.993353090037047, 5.913090209356012, 5.1008143428819706, 4.286828094976633, 2.720210037249857, 1.1907807849063605, 0.5270761095985067, 0.2325681223076977, 0.09963700337204157, 0.03348191708777894, 45.78720063648094, 45.275560058246086, 44.26802848965665, 42.74664340159537, 40.762330798731256, 38.293541420090854, 35.38220635173517, 32.06137482386306, 28.420531145584317, 24.911292183349172, 21.470703058200435, 17.777385349524263, 14.492332909450125, 11.958224394136348, 9.899526345238872, 8.259070312570698, 6.954278734425946, 5.8902257198830625, 5.106032086913043, 4.297356441689237, 2.732071955908085, 1.1993696018465303, 0.5305558150247467, 0.2343673170580201, 0.10024976622189348, 0.03381133295381938, 43.60877816499742, 43.12986836627284, 42.17610660412241, 40.76233079873125, 38.874810539944384, 36.57185197127519, 33.82972009372563, 30.6962886917141, 27.26717350059559, 23.955637149242733, 20.709513207404758, 17.206565360000102, 14.099886812712352, 11.683409477666034, 9.73223266041598, 8.168069410298143, 6.90499986579879, 5.8870874413346765, 5.117936006818702, 4.322067452112758, 2.753385938539983, 1.2116428620097441, 0.535199951385026, 0.23724761564865565, 0.10061163554208913, 0.03218267574381387, 40.92129153349989, 40.481387496987054, 39.60142359662942, 38.2935414200908, 36.571851971275166, 34.423048214117, 31.91485689306736, 29.02687187068569, 25.84653744200675, 22.79489007493857, 19.789057556058577, 16.538450265000595, 13.635961713797265, 11.387219680687986, 9.560038003084944, 8.07996700115779, 6.8819698306301795, 5.898322140494753, 5.155286847467774, 4.365067085549986, 2.788196783128879, 1.2263640097273532, 0.544207457198811, 0.2379894340499582, 0.09850304829498313, 0.03714011993862657, 37.751820589611754, 37.35375194753762, 36.5627878854833, 35.382206351735086, 33.829720093725584, 31.914856893067334, 29.62714031697097, 27.049140650216344, 24.19966261885582, 21.44146683834892, 18.748383186065887, 15.790258875927595, 13.149960370327417, 11.104223171873555, 9.407176446561152, 8.029440492492316, 6.904092250457263, 5.9502701252670835, 5.225355620295312, 4.443157258228229, 2.8355128974809936, 1.2522105725367927, 0.54838340739446, 0.23478623182451436, 0.10553901382649333, 0.03185859397975482, 34.14008249726086, 33.78959075857182, 33.09458289139045, 32.061374823862955, 30.69628869171403, 29.02687187068565, 27.049140650216323, 24.76104648465996, 22.30313895584178, 19.95223305543921, 17.593574154288, 15.008486389259861, 12.68502100198477, 10.838174796045001, 9.316503488545457, 8.04698824851633, 6.971385444970257, 6.065926853947715, 5.34369823236733, 4.540398973236171, 2.9131125796188484, 1.2719499318903045, 0.5423792614487217, 0.24490763451203293, 0.10109705356954395, 0.0335110721939195, 30.191877880419813, 29.894107149893856, 29.2999274770748, 28.420531145584235, 27.267173500595515, 25.846537442006714, 24.199662618855804, 22.30313895584177, 20.19267471120996, 18.286176312343585, 16.464406482758655, 14.327588305449222, 12.318705019131334, 10.7311708949958, 9.370457181732927, 8.198227102417176, 7.1837956914454315, 6.281301808196931, 5.567425630210867, 4.741146364372493, 2.991913815863172, 1.2645123946644383, 0.5550341972955489, 0.24361389277319892, 0.09978522807815479, 0.03584653268235925, 26.386444641689895, 26.137075917149986, 25.64258148631782, 24.91129218334907, 23.955637149242666, 22.79489007493852, 21.441466838348877, 19.95223305543919, 18.28617631234357, 16.68159846232033, 15.223673569626577, 13.560594743696548, 11.974071855985844, 10.647203050803192, 9.44633997385734, 8.378341248594928, 7.4257575554937265, 6.536468700516306, 5.796567195205049, 4.89787935817689, 3.011373543597911, 1.2702135948653976, 0.5627299473132181, 0.2376566103878746, 0.1037277688482486, 0.03268373814745784, 22.66129067106892, 22.458083165358005, 22.060324145471846, 21.470703058200314, 20.709513207404672, 19.789057556058523, 18.748383186065848, 17.593574154287985, 16.46440648275864, 15.223673569626579, 13.760693100247268, 12.5175749828345, 11.43282665443081, 10.34298320467553, 9.335178744966802, 8.390752709349671, 7.497094857005979, 6.677619622585354, 5.880875875728797, 4.80580838706752, 2.9921544960050075, 1.2902137395472655, 0.5501977788401452, 0.24288515047112336, 0.09877097917198174, 0.03466237608727068, 18.673843108107622, 18.522308807994705, 18.216207138689356, 17.77738534952416, 17.206565360000017, 16.53845026500055, 15.79025887592757, 15.008486389259835, 14.32758830544924, 13.56059474369658, 12.517574982834512, 11.574125404692227, 10.837770590084496, 10.02717739983471, 9.18285036326072, 8.350421360973245, 7.539024990121856, 6.757137065849334, 5.906152124489949, 4.784426223573521, 2.9592363267744486, 1.288586554025184, 0.5457860652479967, 0.2368584996042072, 0.10015990451233166, 0.031339806495897425, 15.141997013640204, 15.028370061046717, 14.814023829684894, 14.49233290945004, 14.099886812712324, 13.635961713797201, 13.149960370327372, 12.685021001984818, 12.318705019131407, 11.974071855985892, 11.432826654430853, 10.837770590084526, 10.292169923028748, 9.712982785290015, 9.024818177877629, 8.29294023470508, 7.560325357943972, 6.774911929600311, 5.954714610944191, 4.835707369022152, 2.9383580519181853, 1.255063422576773, 0.5453206749947198, 0.22766758367041173, 0.09795029835334162, 0.03193616321845377, 12.40586283195152, 12.328392542379888, 12.171270833817001, 11.95822439413627, 11.683409477665965, 11.387219680687972, 11.104223171873606, 10.838174796045124, 10.731170894995913, 10.647203050803263, 10.342983204675622, 10.027177399834812, 9.712982785290064, 9.253184152561552, 8.723081996763767, 8.106226397263484, 7.411048353526639, 6.695066258601357, 5.900046861407118, 4.779159051690977, 2.9026734040360016, 1.2197767397318953, 0.5249186372292098, 0.22609271763940778, 0.09035819989320822, 0.032288164352766414, 10.200356793071677, 10.146011463073105, 10.045147499112213, 9.89952634523875, 9.732232660415814, 9.560038003085053, 9.407176446561463, 9.31650348854564, 9.370457181733052, 9.446339973857528, 9.335178744967017, 9.182850363260854, 9.024818177877686, 8.723081996763844, 8.290152690145566, 7.7577929203516405, 7.152640657737033, 6.485594022029388, 5.723219049937945, 4.6446552411314, 2.816765761988777, 1.1807708642241268, 0.4990358352536463, 0.2149913428811649, 0.08854242874260548, 0.027601473146190138, 8.4456993292137, 8.411531164056473, 8.348062407768289, 8.25907031257063, 8.168069410298305, 8.079967001158014, 8.029440492492556, 8.046988248516708, 8.198227102417565, 8.378341248595273, 8.390752709349881, 8.350421360973325, 8.29294023470529, 8.106226397263757, 7.757792920351778, 7.307244318603004, 6.779861780453598, 6.158457459364645, 5.453103215384106, 4.42918301761571, 2.6842048986734026, 1.1206151126347137, 0.47220124395950486, 0.19795231974068822, 0.08357600561670327, 0.028953599023684367, 7.05147890638869, 7.033073310639015, 6.9933530900369005, 6.954278734426504, 6.904999865799605, 6.881969830630519, 6.904092250457314, 6.971385444970683, 7.1837956914462096, 7.425757555494122, 7.497094857006032, 7.539024990122229, 7.560325357944564, 7.411048353527067, 7.152640657737273, 6.779861780453708, 6.299117567003224, 5.7528925841476815, 5.091489500803526, 4.1388241545553734, 2.5057508163519238, 1.0402253605121479, 0.43158413111816674, 0.18310120281810394, 0.0789006381278021, 0.024592448150141305, 5.936750093299041, 5.924391503768792, 5.913090209357093, 5.890225719884435, 5.887087441335714, 5.898322140495484, 5.95027012526754, 6.065926853947769, 6.281301808196872, 6.536468700516588, 6.677619622586152, 6.757137065850372, 6.774911929601086, 6.695066258601825, 6.485594022029701, 6.158457459364841, 5.7528925841477685, 5.257882657928412, 4.663911376344299, 3.7767807417994286, 2.2772070366459785, 0.928252375690603, 0.3845407394538162, 0.17122964583008105, 0.06920810480799187, 0.023025319268545556, 5.1079811781455495, 5.108013275279651, 5.1008143428838615, 5.106032086914417, 5.117936006819654, 5.155286847468908, 5.225355620296355, 5.343698232367499, 5.567425630210529, 5.7965671952056175, 5.88087587573032, 5.906152124491176, 5.954714610944805, 5.900046861407538, 5.7232190499382805, 5.453103215384339, 5.091489500803664, 4.663911376344356, 4.144642083842681, 3.3524796378195894, 1.9705546359864206, 0.789301111941797, 0.3462836099848901, 0.1497343783213631, 0.061815220414540094, 0.02188842607622848, 4.280710657096065, 4.280548947243905, 4.286828094977387, 4.297356441689973, 4.32206745211371, 4.365067085550926, 4.443157258229118, 4.5403989732372265, 4.7411463643738605, 4.897879358177937, 4.805808387068175, 4.7844262235741954, 4.835707369022625, 4.779159051691285, 4.644655241131692, 4.42918301761592, 4.138824154555512, 3.7767807417995054, 3.352479637819618, 2.6959686623217234, 1.5827879479977298, 0.658406242872678, 0.3006389953430647, 0.12708617734573982, 0.05622998173887868, 0.01824063439177462, 2.711059906653039, 2.7136151397021715, 2.720210037250142, 2.7320719559084887, 2.753385938540545, 2.7881967831293237, 2.8355128974815007, 2.9131125796198782, 2.9919138158644083, 3.011373543598608, 2.992154496005157, 2.9592363267745743, 2.938358051918466, 2.9026734040362623, 2.816765761988955, 2.684204898673544, 2.505750816352023, 2.277207036646042, 1.9705546359864556, 1.5827879479977451, 1.0421640436852402, 0.5091698882125201, 0.22553027141482965, 0.10589914697443548, 0.04416858585709379, 0.015604417852888504, 1.1841916819188734, 1.1869134866172222, 1.1907807849070513, 1.1993696018470084, 1.211642862009986, 1.226364009727661, 1.2522105725372108, 1.2719499318905032, 1.2645123946644088, 1.2702135948655793, 1.290213739547634, 1.2885865540253774, 1.2550634225768662, 1.2197767397320407, 1.1807708642242478, 1.1206151126347859, 1.0402253605122076, 0.9282523756906457, 0.7893011119418263, 0.6584062428726961, 0.5091698882125285, 0.3167780047369661, 0.1561238806234175, 0.07179309791717657, 0.03458499540351396, 0.010540588821198018, 0.52295371279494, 0.5234442803371853, 0.5270761095986926, 0.5305558150250334, 0.5351999513852941, 0.5442074571990141, 0.5483834073945194, 0.5423792614486684, 0.5550341972955474, 0.5627299473132666, 0.5501977788402237, 0.5457860652481648, 0.5453206749948404, 0.5249186372292406, 0.4990358352537005, 0.47220124395955715, 0.43158413111819477, 0.3845407394538409, 0.34628360998490537, 0.30063899534307575, 0.2255302714148362, 0.15612388062341997, 0.09848171960293524, 0.04782481107637846, 0.021678110106317486, 0.009121244947562293, 0.23095828441386493, 0.23226366394763473, 0.23256812230761534, 0.23436731705806424, 0.23724761564878802, 0.23798943404999415, 0.23478623182448458, 0.24490763451208664, 0.2436138927732697, 0.23765661038789518, 0.24288515047110967, 0.23685849960419048, 0.22766758367046958, 0.2260927176394751, 0.21499134288117472, 0.19795231974070795, 0.18310120281812692, 0.17122964583008962, 0.14973437832137299, 0.1270861773457455, 0.1058991469744394, 0.07179309791717828, 0.04782481107637915, 0.030672181291983423, 0.013734581342563522, 0.004726434325871723, 0.0990582546978637, 0.09879722469319847, 0.09963700337208549, 0.1002497662218776, 0.10061163554203677, 0.09850304829499396, 0.10553901382656257, 0.10109705356954811, 0.09978522807813188, 0.10372776884826257, 0.09877097917200496, 0.10015990451234044, 0.09795029835332095, 0.09035819989321946, 0.08854242874264077, 0.08357600561670371, 0.07890063812780682, 0.06920810480800284, 0.06181522041454192, 0.05622998173888288, 0.044168585857095655, 0.03458499540351495, 0.021678110106317837, 0.013734581342563782, 0.00888082742611655, 0.002627999900615463, 0.033289751219463, 0.03352592026973042, 0.03348191708779058, 0.03381133295382149, 0.032182675743816044, 0.0371401199386327, 0.03185859397974054, 0.03351107219392612, 0.03584653268237254, 0.03268373814745207, 0.03466237608726785, 0.03133980649590689, 0.03193616321846797, 0.032288164352753286, 0.027601473146184018, 0.028953599023703203, 0.0245924481501405, 0.023025319268544377, 0.02188842607623342, 0.018240634391774157, 0.015604417852889782, 0.010540588821198165, 0.00912124494756264, 0.004726434325871682, 0.00262799990061559, 0.001899277079397805]], [[7.309769036486057, 7.226438417248532, 7.062433962220487, 6.815719363773781, 6.4922399965688005, 6.091205096757742, 5.615710583787702, 5.055283954370296, 4.325064572345992, 2.8773262001689983, 1.0436262673263423, 0.37320281575612757, 0.4250998448167392, 0.31706132065568304, 0.2683109181739075, 0.22122123344100084, 0.18097865883694308, 0.16925775992922662, 0.07787011757255473, 0.3310483779725841, 1.5972388842691654, 2.281171178639088, 1.6814682337690419, 1.01662406085961, 0.5254818980799104, 0.17596710662225967, 7.226438417248527, 7.146262617892135, 6.981126038959686, 6.741950479482706, 6.419610893993733, 6.026062305773969, 5.5566566844495915, 5.003063549563617, 4.282424108219066, 2.8493472552442003, 1.034389662165542, 0.36972590289961443, 0.4225328130556834, 0.3145848418211853, 0.267095773300834, 0.220469717893347, 0.18007183540502264, 0.16950580132358228, 0.07732966475167301, 0.3316066189217274, 1.5998321875905097, 2.285589831389215, 1.6851220372499038, 1.0189136861827057, 0.5266105289545483, 0.17620437206810133, 7.0624339622204815, 6.981126038959677, 6.826838384048093, 6.587220814001854, 6.280613215992072, 5.894014847808456, 5.438678393606841, 4.9007331014734525, 4.196351223158117, 2.7947559523361956, 1.0146640258467994, 0.36508206840002294, 0.41509932668072963, 0.3115459240313512, 0.2640284221777349, 0.21856165712958922, 0.17977710547363443, 0.16823441975423534, 0.07783289935787267, 0.3320252514343944, 1.6056352320435157, 2.2947275955832036, 1.6927255524262041, 1.0234101846134425, 0.5286216021202269, 0.17708200597338997, 6.815719363773769, 6.741950479482698, 6.587220814001849, 6.367199525365988, 6.0649432840158894, 5.701776934884206, 5.263010657681159, 4.746294603776082, 4.07098447309376, 2.712406087489948, 0.9877830083121009, 0.35527729684781423, 0.4074971275322047, 0.30516057824029935, 0.2606245936084991, 0.2166093969969617, 0.1778659885580273, 0.16901084841986266, 0.07650614208801135, 0.3342694127255556, 1.6151377503522286, 2.309802469518037, 1.7042058802438063, 1.0299501025271138, 0.5318328519753555, 0.17746581540631845, 6.492239996568784, 6.4196108939937195, 6.280613215992061, 6.064943284015885, 5.791041373434996, 5.440254794249354, 5.033196217307199, 4.545536660233041, 3.902439485023111, 2.606988304981598, 0.9493639528746729, 0.34621103456330454, 0.3944950495135689, 0.29879031033933356, 0.2557569104097139, 0.21384603193730706, 0.17703377125874625, 0.16779248412174624, 0.07755158146799299, 0.33567942772368775, 1.6300454201961212, 2.331196875351803, 1.719945601931235, 1.0384990999492354, 0.534789934813632, 0.17922819552087121, 6.091205096757724, 6.026062305773952, 5.894014847808445, 5.701776934884197, 5.44025479424935, 5.128068594110752, 4.742935746928021, 4.296742824427316, 3.7013295206252503, 2.4737237270020884, 0.9086936413882397, 0.33327395858889414, 0.38251260381884455, 0.2924781975203672, 0.2517710949468563, 0.21222866508522414, 0.1771977927860013, 0.16917887224010564, 0.07708729475946467, 0.34197433017452006, 1.6503147286454203, 2.360800990943623, 1.7394247531129385, 1.0471691676886168, 0.5396123935549739, 0.17927827680523709, 5.615710583787684, 5.556656684449573, 5.438678393606824, 5.26301065768115, 5.0331962173071885, 4.742935746928016, 4.410288602229765, 3.9970794879629565, 3.4570211980950507, 2.3262177843879868, 0.8496072455376493, 0.3161023050282503, 0.3658975250757887, 0.2802753343647119, 0.24523070163185, 0.20865115782155688, 0.17362099843660253, 0.1698411062701858, 0.076284885912267, 0.34286833148482576, 1.6797896240723413, 2.3972882626482983, 1.7607600375515164, 1.0582853854110377, 0.5416867815410312, 0.18194607466534227, 5.055283954370277, 5.0030635495636, 4.9007331014734365, 4.746294603776068, 4.54553666023303, 4.296742824427309, 3.997079487962954, 3.6594583545766017, 3.172391635588508, 2.1385521202575295, 0.8121835754149548, 0.3219221346926457, 0.3628548416307096, 0.29058123657495033, 0.25417720649059417, 0.21948070932834934, 0.18821328478112795, 0.17836156234789838, 0.08833618832468261, 0.3641359221432755, 1.7136441869390582, 2.4417010621912967, 1.7868789785974948, 1.0643382245598434, 0.5470051982849155, 0.18101844289652688, 4.325064572345973, 4.282424108219047, 4.1963512231581, 4.070984473093746, 3.9024394850231023, 3.7013295206252455, 3.457021198095049, 3.1723916355885065, 2.8223140563511655, 1.9262731614334276, 0.6855823533911768, 0.23921162133286805, 0.2980947949651301, 0.23137081965603618, 0.2070649018824753, 0.17985538055277525, 0.15116404895977145, 0.1550474806645078, 0.05094762492278536, 0.33828111899477914, 1.7657765194458237, 2.505354472018149, 1.8026665833541218, 1.0735637867233, 0.5477085884196797, 0.18155002491642833, 2.877326200168984, 2.8493472552441865, 2.794755952336183, 2.7124060874899376, 2.6069883049815914, 2.4737237270020844, 2.326217784387983, 2.1385521202575286, 1.9262731614334268, 1.4463247586037329, 0.7337589442717611, 0.4956874339299923, 0.5155269720083108, 0.4421441128320411, 0.40214151786304814, 0.3595891933074649, 0.3128884972983561, 0.29982798745663863, 0.20696241639125532, 0.5045464787927506, 1.9047315635692612, 2.5567206780231504, 1.816464543499182, 1.078545952172481, 0.5457044167508935, 0.18243891923002536, 1.043626267326337, 1.034389662165537, 1.0146640258467945, 0.9877830083120968, 0.9493639528746702, 0.9086936413882378, 0.8496072455376485, 0.8121835754149541, 0.6855823533911766, 0.7337589442717609, 1.107434155300288, 1.2283638462672386, 1.137385127913529, 1.0542112044178507, 0.9538425740710491, 0.8623771141311893, 0.7775245976094107, 0.6933875490536463, 0.639724522574525, 0.9837802054042264, 2.1074205140887883, 2.5819188478001553, 1.8298381815669928, 1.0723494387393127, 0.544942232336065, 0.17914582931514345, 0.373202815756124, 0.36972590289961044, 0.3650820684000196, 0.3552772968478112, 0.34621103456330243, 0.33327395858889275, 0.31610230502824954, 0.3219221346926454, 0.23921162133286755, 0.4956874339299938, 1.22836384626724, 1.5942088864085342, 1.5107737869536206, 1.4079063954534334, 1.2962761905130964, 1.1783174745385965, 1.0692299446593023, 0.9556301118802858, 0.9266203564891949, 1.2627036446505773, 2.257885505806025, 2.599532011052691, 1.823558156401497, 1.0633612573342162, 0.5370410432361233, 0.17720765097558733, 0.4250998448167357, 0.42253281305567986, 0.41509932668072547, 0.4074971275322026, 0.39449504951356745, 0.3825126038188431, 0.365897525075788, 0.36285484163071063, 0.2980947949651319, 0.5155269720083125, 1.1373851279135332, 1.5107737869536255, 1.5208387706214765, 1.4253524114561011, 1.3305731051129752, 1.227088689462791, 1.111394990681897, 1.0189688286792276, 0.9738755693281377, 1.30056561970669, 2.3004448950918484, 2.601131153785095, 1.7968250916329347, 1.0463364731729659, 0.5246582406535724, 0.17349414281593037, 0.31706132065567993, 0.3145848418211829, 0.31154592403134973, 0.3051605782402966, 0.2987903103393317, 0.2924781975203667, 0.2802753343647141, 0.2905812365749537, 0.23137081965603734, 0.44214411283204524, 1.054211204417861, 1.4079063954534454, 1.4253524114561076, 1.3797649745205218, 1.2945029517219298, 1.2018564335914224, 1.107369388999945, 1.0087641047494786, 0.9692774621676635, 1.2970395451236678, 2.2741663715843807, 2.560161123248149, 1.7554464754902928, 1.0154748652574257, 0.5093704709033793, 0.16685390070379155, 0.26831091817390623, 0.2670957733008345, 0.26402842217773426, 0.26062459360849655, 0.2557569104097107, 0.25177109494685945, 0.24523070163185617, 0.2541772064905999, 0.20706490188247764, 0.4021415178630556, 0.953842574071067, 1.2962761905131133, 1.3305731051129905, 1.2945029517219417, 1.2340734060964202, 1.154913923010187, 1.066629705228951, 0.9785156042688903, 0.9438376082996086, 1.2599866250366583, 2.2074618236164554, 2.4768874087685524, 1.6916052942160127, 0.9745960711666967, 0.4870740546103883, 0.1603659963916872, 0.22122123344100936, 0.22046971789334982, 0.2185616571295888, 0.21660939699696277, 0.21384603193731308, 0.21222866508523094, 0.20865115782156532, 0.21948070932835995, 0.17985538055278488, 0.3595891933074801, 0.86237711413121, 1.1783174745386258, 1.2270886894628235, 1.201856433591454, 1.1549139230102057, 1.0896978413334626, 1.009235671083428, 0.9330215612718462, 0.8975903523381149, 1.202772361880732, 2.103396117471077, 2.356637413876885, 1.6042912627125343, 0.9216088027739663, 0.4607221645126956, 0.15099371152122207, 0.18097865883692088, 0.18007183540500452, 0.17977710547362283, 0.17786598855802926, 0.177033771258754, 0.17719779278600145, 0.17362099843659518, 0.18821328478112756, 0.1511640489597674, 0.31288849729836155, 0.7775245976094466, 1.0692299446593574, 1.111394990681963, 1.1073693890000047, 1.0666297052289866, 1.0092356710834445, 0.944263521541095, 0.8689161076022858, 0.8435019219298419, 1.1250171501923105, 1.9682203073572033, 2.2008909042380407, 1.4956058827188088, 0.8584554591144624, 0.4283773400821471, 0.14139649354274617, 0.1692577599292134, 0.16950580132357732, 0.16823441975424747, 0.16901084841989053, 0.1677924841217708, 0.16917887224012085, 0.169841106270192, 0.17836156234790002, 0.155047480664512, 0.29982798745665395, 0.6933875490536984, 0.955630111880394, 1.0189688286793415, 1.008764104749557, 0.9785156042689408, 0.9330215612718789, 0.8689161076023006, 0.8071882431862726, 0.7789888696640842, 1.0429111898630015, 1.8094713057773426, 2.0195825142408474, 1.3688934499586567, 0.7844254107474886, 0.39351028126560894, 0.1289608437749651, 0.07787011757213617, 0.07732966475143377, 0.07783289935777168, 0.0765061420879256, 0.0775515814678963, 0.07708729475937455, 0.07628488591215266, 0.08833618832448972, 0.050947624922549874, 0.20696241639113458, 0.6397245225745857, 0.9266203564893254, 0.9738755693282344, 0.9692774621677274, 0.9438376082996571, 0.8975903523381464, 0.8435019219298611, 0.7789888696640935, 0.747612214654247, 0.9754138410883076, 1.6651657615624624, 1.824723045932108, 1.2232279300067268, 0.7054355665077922, 0.3541416252950136, 0.1165191175767711, 0.33104837797300607, 0.33160661892202214, 0.3320252514345557, 0.33426941272568517, 0.33567942772382975, 0.34197433017467405, 0.34286833148500584, 0.3641359221434962, 0.33828111899503127, 0.5045464787929674, 0.9837802054044521, 1.2627036446507647, 1.300565619706818, 1.2970395451237715, 1.2599866250367295, 1.2027723618807848, 1.1250171501923487, 1.0429111898630294, 0.9754138410883226, 1.1246653440251455, 1.5949211077248042, 1.608165824620242, 1.0686150687697613, 0.6228456319836801, 0.3118055398293142, 0.10425211793119596, 1.5972388842715004, 1.5998321875918857, 1.60563523204417, 1.615137750352772, 1.6300454201967225, 1.6503147286460438, 1.6797896240731238, 1.7136441869402534, 1.7657765194472599, 1.904731563570276, 2.1074205140892537, 2.2578855058063065, 2.300444895092088, 2.274166371584562, 2.2074618236165975, 2.1033961174711884, 1.9682203073572877, 1.8094713057774041, 1.6651657615625, 1.5949211077248227, 1.5399976772883843, 1.2911692568470448, 0.870636873304581, 0.5056298921339625, 0.25897229113957965, 0.08522719148749969, 2.281171178639691, 2.285589831389859, 2.2947275955838373, 2.309802469518611, 2.3311968753523384, 2.360800990944174, 2.3972882626488863, 2.441701062191887, 2.5053544720187046, 2.5567206780236744, 2.581918847800622, 2.599532011053046, 2.601131153785348, 2.5601611232483634, 2.476887408768739, 2.3566374138770363, 2.2008909042381584, 2.019582514240933, 1.8247230459321644, 1.608165824620279, 1.2911692568470639, 0.9324132389272014, 0.626684926286034, 0.37246902947846244, 0.19198842583975329, 0.06516495006416623, 1.681468233767619, 1.6851220372492997, 1.6927255524262543, 1.7042058802440165, 1.7199456019314106, 1.7394247531130702, 1.7607600375514851, 1.7868789785970964, 1.8026665833535065, 1.816464543498889, 1.8298381815671618, 1.8235581564017618, 1.7968250916331334, 1.7554464754904755, 1.6916052942161766, 1.6042912627126704, 1.4956058827189158, 1.368893449958737, 1.223227930006785, 1.0686150687698026, 0.870636873304605, 0.6266849262860413, 0.4158733322599212, 0.25489539216334406, 0.13425629437911016, 0.045519436184127536, 1.0166240608592991, 1.0189136861823667, 1.023410184613232, 1.0299501025271152, 1.0384990999493229, 1.0471691676886408, 1.0582853854109635, 1.064338224559763, 1.0735637867232557, 1.078545952172427, 1.0723494387392702, 1.0633612573342608, 1.0463364731730764, 1.0154748652575374, 0.9745960711667963, 0.9216088027740531, 0.8584554591145335, 0.784425410747544, 0.705435566507833, 0.6228456319837103, 0.5056298921339818, 0.37246902947847055, 0.25489539216334633, 0.15855057521958535, 0.08648225688481606, 0.029600574905313072, 0.525481898079795, 0.5266105289544191, 0.5286216021201019, 0.531832851975278, 0.5347899348135988, 0.5396123935549505, 0.5416867815410115, 0.5470051982849069, 0.547708588419683, 0.5457044167508828, 0.5449422323360312, 0.537041043236105, 0.5246582406535968, 0.5093704709034286, 0.48707405461043846, 0.4607221645127406, 0.42837734008218536, 0.3935102812656394, 0.35414162529503695, 0.31180553982933185, 0.2589722911395915, 0.19198842583975875, 0.13425629437911252, 0.0864822568848167, 0.04768605297119097, 0.016972858607210844, 0.17596710662208576, 0.17620437206803607, 0.1770820059733713, 0.17746581540627598, 0.17922819552082195, 0.1792782768052198, 0.18194607466534057, 0.18101844289647523, 0.18155002491634026, 0.18243891922998054, 0.17914582931514422, 0.17720765097559266, 0.17349414281592979, 0.16685390070379832, 0.16036599639170368, 0.15099371152123792, 0.1413964935427588, 0.1289608437749755, 0.11651911757677941, 0.10425211793120236, 0.08522719148750384, 0.06516495006416831, 0.04551943618412841, 0.02960057490531348, 0.016972858607210924, 0.005889699211803961]]]

        phi_test = np.array(phi_test)

        keff_test = 1.10968367

        import matplotlib.pyplot as plt
        plt.imshow(pydgm.state.mg_phi[0, 0].reshape(int(np.sqrt(pydgm.state.mg_phi.shape[-1])), -1), interpolation=None)
        plt.colorbar()
        plt.show()

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
        pydgm.control.angle_order = 2
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

    def test_solver_partisn_fixed_2g_l0_no_fiss(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''

        # Set the variables for the test
        pydgm.control.allow_fission = False
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.scatter_leg_order = 0
        pydgm.control.outer_print = 1
        pydgm.control.number_angles_pol = 1
        pydgm.control.number_angles_azi = 1
        pydgm.control.boundary_east = 0.0
        pydgm.control.boundary_west = 0.0
        pydgm.control.boundary_north = 0.0
        pydgm.control.boundary_south = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 2)

        print(pydgm.angle.mu.tolist())
        print(pydgm.angle.eta.tolist())
        print(pydgm.angle.wt.tolist())

        # Solve the problem
        pydgm.solver.solve()

        # Partisn output flux indexed as group, Legendre, cell
        phi_test = [[[4.78760994961062, 7.6688359619348105, 8.97572601422468, 9.685828656182395, 10.03913784765901, 10.217145490104501, 10.278032927166151, 10.247805306210049, 10.111225821872695, 9.81205747807781, 9.404099426123576, 9.10473022538095, 8.960297870819263, 8.895692763417312, 8.87959809284213, 8.901191456291727, 8.969037720909254, 9.123784939602428, 9.4236730041138, 9.989310501817403, 11.01034126756837, 11.834109519385867, 12.037026500397431, 11.664120377696387, 10.475885142772233, 6.287265270988457, 7.66883596193481, 12.770293472325443, 15.181313830093151, 16.405144431978787, 17.07789584447075, 17.39354339153685, 17.511698959309605, 17.457003986706244, 17.211949317380043, 16.651313481568636, 15.891123470997272, 15.331552182871, 15.07378261185753, 14.958066936594255, 14.928632627999614, 14.967204623225086, 15.096482814905164, 15.369853463477329, 15.91351383517496, 17.09567377610736, 19.166256799207666, 20.866992633642774, 21.18378389721551, 20.51906277767043, 18.230961527331782, 10.541285818133748, 8.975726014224687, 15.181313830093153, 18.551938438707346, 20.23215101016599, 21.1286722917256, 21.593766302776935, 21.747208091432494, 21.6831067483271, 21.340171783304587, 20.58753622794334, 19.58022956842832, 18.83077820157052, 18.468506714247944, 18.315783012309748, 18.272979147119933, 18.330240942627963, 18.512353643889117, 18.874153848261045, 19.681483832763547, 21.246954849279067, 24.00014842148124, 26.326485654105266, 26.774546948666494, 25.90086502644445, 22.48041878899708, 12.659810767797161, 9.685828656182393, 16.405144431978794, 20.23215101016598, 22.362614029775308, 23.439716557895697, 23.996738508490566, 24.213716202292016, 24.12357819414354, 23.720696735515574, 22.84331820231393, 21.685798209067585, 20.81128035448769, 20.380167427934555, 20.1849121014193, 20.143055512083404, 20.219633733684656, 20.423162777617375, 20.912917641461163, 21.86223295140187, 23.678723364678405, 26.895721657468254, 29.58197400261394, 30.211864130809335, 29.04543537815441, 24.817126755099764, 13.847837116066113, 10.039137847659015, 17.077895844470742, 21.128672291725614, 23.439716557895697, 24.7351000131335, 25.352108300003344, 25.59699872237207, 25.522066629221523, 25.073594750951496, 24.134968813494535, 22.886070018210216, 21.943988565530443, 21.46780281361942, 21.258952218707027, 21.21285384298507, 21.288970323504092, 21.559695624107455, 22.096768993611573, 23.142657651774023, 25.156759257825463, 28.608067904519118, 31.587628641625816, 32.26167369060193, 30.864939530254503, 26.18861344570175, 14.508108666925251, 10.217145490104505, 17.39354339153686, 21.59376630277693, 23.996738508490576, 25.352108300003337, 26.085181873648484, 26.33848506285249, 26.271865142275725, 25.83332382817968, 24.85774105327173, 23.579058728341693, 22.603316200211903, 22.11975266343947, 21.910582807579452, 21.855943770722693, 21.967836093534988, 22.255426916940237, 22.82601893890679, 23.962030539266422, 26.0384397829092, 29.683174080220773, 32.81420323034064, 33.47926131829918, 31.954962868361196, 26.980983763412066, 14.891888552767265, 10.278032927166157, 17.511698959309612, 21.747208091432505, 24.213716202292016, 25.596998722372078, 26.3384850628525, 26.68060232554865, 26.623708750013897, 26.202683270424505, 25.25774804269251, 23.98106938083665, 23.02890888344029, 22.5586650552769, 22.35421070842912, 22.351531315188545, 22.47220259385705, 22.75111305176694, 23.373205536532165, 24.514043257257534, 26.659858872323444, 30.39692759421344, 33.57919399212519, 34.237836108213614, 32.59900703459655, 27.446225085531886, 15.120851985335841, 10.24780530621006, 17.45700398670625, 21.68310674832709, 24.12357819414354, 25.522066629221516, 26.271865142275736, 26.623708750013897, 26.683939279978198, 26.32114369437849, 25.43984354190478, 24.247803192550602, 23.351820307081955, 22.942166145666732, 22.828576617163748, 22.84779974151465, 22.973756018886903, 23.29227043934592, 23.866757103589016, 25.02588188023034, 27.194393073075414, 30.935139340954514, 34.11978952794865, 34.70670394025322, 32.984690540859525, 27.730750736413604, 15.251010483587073, 10.111225821872695, 17.21194931738005, 21.34017178330458, 23.720696735515563, 25.073594750951486, 25.83332382817966, 26.2026832704245, 26.321143694378495, 26.185774807023027, 25.487677427988892, 24.451775027232877, 23.742564454994227, 23.531777318269818, 23.496943020634202, 23.53312122365342, 23.705594856173118, 24.011835188956454, 24.5953056004634, 25.681281536921645, 27.783234405935513, 31.4794221694259, 34.521183136722634, 34.999719122990854, 33.232060238398795, 27.89959768304926, 15.329984943642037, 9.812057478077817, 16.65131348156863, 20.587536227943342, 22.843318202313938, 24.13496881349453, 24.85774105327174, 25.257748042692498, 25.439843541904782, 25.48767742798888, 25.287857643807225, 24.818653141957704, 24.58960807000084, 24.58290726023828, 24.608240905575546, 24.723505244556794, 24.89520955295218, 25.22832638224697, 25.81267099216201, 26.8335091149347, 28.747261042007132, 32.079409304768234, 34.81568764358228, 35.207336828866794, 33.39120916889378, 28.00203801193863, 15.380677437426753, 9.404099426123576, 15.891123470997275, 19.580229568428308, 21.685798209067585, 22.886070018210212, 23.579058728341693, 23.98106938083666, 24.247803192550585, 24.451775027232873, 24.8186531419577, 25.412527468875336, 25.767374220874252, 25.91664185186867, 26.070581222861712, 26.21144315247769, 26.433320258830893, 26.773826708491935, 27.34634104528654, 28.355780499236804, 29.997811127319107, 32.68711235537881, 35.04268559451088, 35.372768621029785, 33.490020748237356, 28.072680702000373, 15.409976597704695, 9.104730225380948, 15.331552182871006, 18.83077820157052, 20.811280354487685, 21.943988565530454, 22.603316200211903, 23.028908883440288, 23.351820307081965, 23.74256445499422, 24.58960807000084, 25.76737422087425, 26.612598999795253, 26.983592284900688, 27.206233086743868, 27.418994838501902, 27.657502769649167, 28.005599996397713, 28.5787555232995, 29.515230940901915, 30.935034064309495, 33.22864279864799, 35.252365291751644, 35.49055236214536, 33.5682782854793, 28.115450470668197, 15.43214892166452, 8.960297870819266, 15.073782611857533, 18.468506714247944, 20.38016742793457, 21.467802813619436, 22.11975266343949, 22.5586650552769, 22.94216614566672, 23.531777318269842, 24.582907260238283, 25.916641851868683, 26.983592284900688, 27.57043409264726, 27.881912274772667, 28.133941386253102, 28.391021321510813, 28.74580171067537, 29.299354381231332, 30.132921244110022, 31.45598547634411, 33.62868496412728, 35.45955927143133, 35.5776917847033, 33.629772353064524, 28.148078770129686, 15.444590330170369, 8.895692763417316, 14.958066936594264, 18.315783012309744, 20.184912101419314, 21.258952218707037, 21.910582807579445, 22.354210708429143, 22.828576617163755, 23.496943020634184, 24.608240905575567, 26.070581222861733, 27.206233086743886, 27.881912274772652, 28.307325815248074, 28.569186503725234, 28.852284722209777, 29.207508275300587, 29.708540763627884, 30.511204942835693, 31.79408096335409, 33.89253540392404, 35.63406667029969, 35.66538727851348, 33.671307704492264, 28.177412030209815, 15.453233310528583, 8.879598092842128, 14.928632627999617, 18.272979147119944, 20.14305551208341, 21.21285384298508, 21.855943770722703, 22.35153131518854, 22.847799741514685, 23.533121223653417, 24.72350524455678, 26.211443152477745, 27.41899483850195, 28.13394138625312, 28.569186503725206, 28.899014349190725, 29.169238086519716, 29.50903754994702, 29.99961368442992, 30.77593255992854, 32.03537155714674, 34.087070613348175, 35.769474385230886, 35.749603495823074, 33.71156236743473, 28.196482076015986, 15.463368364063424, 8.901191456291732, 14.967204623225085, 18.330240942627963, 20.219633733684667, 21.288970323504095, 21.96783609353501, 22.47220259385707, 22.9737560188869, 23.705594856173153, 24.8952095529522, 26.433320258830896, 27.657502769649252, 28.391021321510912, 28.852284722209813, 29.169238086519666, 29.454419077073013, 29.76785006982839, 30.254897724339244, 31.009584196531677, 32.238331043754805, 34.26275693405519, 35.89137595090416, 35.82489779354687, 33.75349413686238, 28.21165786316311, 15.467700082683853, 8.969037720909256, 15.096482814905169, 18.512353643889103, 20.42316277761737, 21.559695624107473, 22.255426916940237, 22.751113051766957, 23.29227043934594, 24.011835188956454, 25.228326382247033, 26.773826708491985, 28.005599996397763, 28.74580171067554, 29.207508275300796, 29.50903754994713, 29.767850069828313, 30.09320204661445, 30.532476480411074, 31.2741130377024, 32.480431849757906, 34.450766828467316, 36.02541781741663, 35.901111359648, 33.78768985756788, 28.223925239209517, 15.473527130546719, 9.123784939602428, 15.36985346347733, 18.874153848261045, 20.912917641461156, 22.096768993611576, 22.826018938906795, 23.373205536532176, 23.86675710358904, 24.595305600463412, 25.812670992162033, 27.346341045286668, 28.578755523299623, 29.299354381231456, 29.708540763628246, 29.999613684430408, 30.254897724339532, 30.53247648041101, 30.960583631156588, 31.635632400460057, 32.80670470550234, 34.71121743650234, 36.18319752678803, 35.978334548841005, 33.80931530887556, 28.236637683997788, 15.474255179196332, 9.4236730041138, 15.913513835174957, 19.681483832763544, 21.86223295140187, 23.142657651774023, 23.96203053926643, 24.51404325725756, 25.025881880230365, 25.681281536921688, 26.83350911493477, 28.355780499236896, 29.51523094090216, 30.13292124411031, 30.51120494283605, 30.77593255992937, 31.009584196532796, 31.274113037703156, 31.635632400460164, 32.250345144579235, 33.30142043129268, 35.08642623900617, 36.37600820792546, 36.027964465298005, 33.83426107431121, 28.2363151933819, 15.472084379371971, 9.989310501817414, 17.09567377610737, 21.246954849279053, 23.67872336467841, 25.156759257825474, 26.03843978290921, 26.65985887232345, 27.194393073075453, 27.78323440593559, 28.74726104200726, 29.99781112731926, 30.93503406430974, 31.455985476344658, 31.794080963354798, 32.03537155714773, 32.23833104375668, 32.48043184976064, 32.80670470550468, 33.301420431293565, 34.187882297627944, 35.584525761225144, 36.541539392767675, 36.08092722345373, 33.82975879284915, 28.228071914449274, 15.46401427393425, 11.010341267568377, 19.166256799207677, 24.000148421481246, 26.895721657468247, 28.608067904519128, 29.68317408022079, 30.396927594213466, 30.935139340954592, 31.479422169426027, 32.07940930476842, 32.68711235537911, 33.22864279864845, 33.628684964128, 33.89253540392531, 34.087070613350065, 34.26275693405815, 34.450766828472446, 34.71121743650957, 35.08642623901423, 35.584525761233884, 36.26277689449282, 36.65901280417588, 36.06752691377266, 33.7690017986686, 28.159074848805208, 15.433736969257609, 11.834109519385871, 20.866992633642766, 26.32648565410527, 29.581974002613943, 31.587628641625813, 32.81420323034065, 33.579193992125255, 34.11978952794872, 34.521183136722776, 34.81568764358254, 35.042685594511305, 35.25236529175242, 35.45955927143251, 35.63406667030148, 35.7694743852341, 35.89137595090945, 36.025417817425016, 36.18319752680289, 36.376008207951784, 36.541539392807785, 36.65901280416871, 36.59297626915535, 35.80531363135874, 33.49960436905014, 27.96610981238056, 15.331267042332492, 12.037026500397431, 21.183783897215502, 26.774546948666494, 30.21186413080934, 32.26167369060193, 33.4792613182992, 34.23783610821367, 34.7067039402533, 34.999719122990996, 35.207336828867064, 35.372768621030254, 35.490552362146175, 35.57769178470479, 35.66538727851597, 35.74960349582698, 35.824897793553525, 35.90111135965984, 35.97833454886269, 36.027964465338364, 36.08092722349167, 36.06752691374513, 35.805313631316125, 34.98938537693847, 32.75686217628355, 27.413605601630586, 15.085140233182349, 11.664120377696385, 20.51906277767043, 25.900865026444453, 29.04543537815441, 30.86493953025451, 31.954962868361203, 32.59900703459656, 32.98469054085957, 33.232060238398915, 33.39120916889399, 33.49002074823772, 33.56827828547995, 33.629772353065704, 33.67130770449447, 33.71156236743848, 33.75349413686827, 33.78768985757804, 33.809315308892884, 33.834261074336894, 33.82975879285436, 33.76900179864858, 33.49960436903593, 32.756862176278474, 30.91053138257915, 26.0867917283471, 14.416608288868316, 10.475885142772237, 18.230961527331782, 22.480418788997092, 24.81712675509978, 26.18861344570175, 26.98098376341207, 27.446225085531907, 27.73075073641364, 27.899597683049315, 28.00203801193873, 28.07268070200055, 28.115450470668517, 28.14807877013027, 28.177412030210885, 28.19648207601833, 28.211657863167527, 28.223925239215866, 28.236637684004165, 28.236315193385586, 28.22807191444715, 28.159074848799598, 27.966109812377372, 27.41360560162894, 26.0867917283456, 22.61310353191734, 12.783015490750685, 6.287265270988456, 10.541285818133755, 12.659810767797172, 13.84783711606611, 14.508108666925253, 14.891888552767274, 15.120851985335852, 15.251010483587091, 15.329984943642062, 15.380677437426797, 15.409976597704762, 15.43214892166459, 15.444590330170525, 15.453233310528992, 15.463368364064264, 15.467700082686315, 15.473527130551474, 15.474255179197412, 15.472084379369216, 15.464014273932984, 15.433736969256344, 15.331267042331453, 15.085140233182445, 14.416608288868126, 12.78301549075019, 7.52212599441335]], [[3.1700949679479287, 5.0680120652526535, 5.408787809419276, 5.5401729090641805, 5.602131382049184, 5.625288959823299, 5.633617448422104, 5.600482192260722, 5.445572145558473, 4.230104757134074, 2.150838902040412, 1.4568790460547314, 1.634162040922643, 1.593574190034226, 1.5903320423316898, 1.6115378016864157, 1.5650607566688806, 1.7045316312415475, 1.2734991086719356, 3.5766031634507627, 18.86074103681332, 33.7279447207271, 37.57285678733443, 36.862856605545446, 31.674379674546007, 15.775251582103303, 5.0680120652526535, 8.502815540739721, 9.227896028543697, 9.507810014705115, 9.625676017596003, 9.684860690542665, 9.694918174311372, 9.634870310373309, 9.302425749438669, 7.020487344972587, 3.3095715290633345, 2.045155562040758, 2.3900634366533517, 2.2775031380960065, 2.3168126353514427, 2.3086599658350773, 2.2743165329423793, 2.462905428720908, 1.7963006177351588, 6.278077675723094, 39.67212671357732, 73.60585007288256, 83.00604251706535, 81.38231798872513, 68.61884375355329, 32.99898103710819, 5.4087878094192785, 9.227896028543695, 10.344651672436248, 10.681590422518399, 10.874816309393042, 10.946640753920978, 10.96095178574601, 10.890777204133753, 10.382637705514599, 7.666907665458051, 3.446715518387931, 1.9817262734019385, 2.3303748078677793, 2.24996564759287, 2.2524596574037434, 2.2719823580275107, 2.2300129197535483, 2.4084361151851343, 1.7480793281311438, 7.4205814579384, 51.65549968663786, 98.67450898225306, 113.00599457070844, 110.57874435250697, 91.33102896837674, 42.736228428847305, 5.54017290906418, 9.50781001470512, 10.681590422518395, 11.180873740749, 11.357999157083817, 11.475365775664718, 11.493168065030606, 11.388764752943597, 10.810441577624715, 7.971214877482966, 3.5838325337889714, 2.05922783409128, 2.427654774485325, 2.3211108662241284, 2.351787083695504, 2.34641897913062, 2.321328051913352, 2.4992419730504003, 1.838778991515657, 8.155622749596555, 59.048981739830644, 114.35247778891649, 132.3473945780123, 129.23837649886946, 105.32244583288339, 48.642662704414704, 5.602131382049188, 9.625676017596003, 10.874816309393045, 11.357999157083814, 11.628944212120306, 11.711127474775118, 11.758864144107553, 11.637236064255378, 11.043140044749961, 8.129085796008763, 3.6343630771413022, 2.0700879891254447, 2.4490601625157407, 2.342880466151972, 2.3612338391451893, 2.3793404825515547, 2.3241111792236073, 2.539717075837886, 1.8345932377132979, 8.59992884252315, 63.651349469053585, 124.2749342608274, 144.60509998909782, 141.00518576155386, 114.03424560241699, 52.25031828724946, 5.625288959823298, 9.684860690542669, 10.946640753920978, 11.475365775664729, 11.711127474775116, 11.87134488713883, 11.860284575814717, 11.776009969456036, 11.170695804533187, 8.210701811318819, 3.67589614370766, 2.1008717325762962, 2.463081313923029, 2.3780480521011236, 2.3883752001158447, 2.3905168132448185, 2.3788664342463504, 2.5380868575735778, 1.8902529382989306, 8.895222632933935, 66.57180380971808, 130.58741966224528, 152.3688260816392, 148.3933769353735, 119.46721052833169, 54.45228725274596, 5.633617448422105, 9.69491817431137, 10.960951785746015, 11.493168065030604, 11.758864144107557, 11.860284575814717, 11.951471748666998, 11.794717238701303, 11.210793676618302, 8.267307254726616, 3.6652185221689275, 2.067978897531124, 2.477790284242649, 2.3393545200307115, 2.386233058134567, 2.390815129399359, 2.3295618182395, 2.575843804093818, 1.825783967162394, 9.021867772479776, 68.5526349980143, 134.6910481690175, 157.35564642297257, 153.09007747183193, 122.80125322905752, 55.8698172593384, 5.600482192260724, 9.63487031037331, 10.890777204133752, 11.3887647529436, 11.637236064255372, 11.77600996945604, 11.794717238701303, 11.795494719375577, 11.158233802468944, 8.206652402407121, 3.7620546975209863, 2.2099898323939917, 2.54370682195024, 2.4911137873837905, 2.4756895071175125, 2.503525703826109, 2.483235610633664, 2.6223541213274597, 2.007153846057474, 9.404187120327567, 69.93316120605957, 137.6572466821596, 160.75943598821112, 155.97865438600448, 124.96991844343933, 56.70091159759273, 5.445572145558474, 9.302425749438667, 10.382637705514602, 10.810441577624708, 11.043140044749965, 11.170695804533182, 11.210793676618309, 11.158233802468947, 10.894860543980785, 8.080520867802726, 3.414526712522348, 1.7769667752188432, 2.2140391897832723, 2.0697927328610177, 2.1302745007409136, 2.1017411865512625, 2.079754178030337, 2.3419558608898035, 1.5846620039841366, 8.760099592414846, 71.57346462624919, 140.4802380278692, 162.91089240896815, 157.95537473602948, 126.30694393851532, 57.235247556894755, 4.2301047571340735, 7.020487344972588, 7.666907665458049, 7.971214877482973, 8.129085796008757, 8.210701811318826, 8.26730725472661, 8.20665240240713, 8.080520867802727, 6.726880128341121, 4.105081216032344, 3.4238580159059593, 3.780583986495765, 3.7043243721942885, 3.7178010831258304, 3.7805469975541683, 3.660985777732072, 4.095833303109532, 3.261197827190279, 12.887111346700426, 75.64780411219276, 142.74143212084138, 164.5113091520649, 159.31291300062034, 127.13281906894147, 57.62640382738917, 2.150838902040411, 3.309571529063336, 3.446715518387931, 3.58383253378897, 3.634363077141305, 3.675896143707655, 3.665218522168932, 3.7620546975209854, 3.414526712522352, 4.105081216032345, 6.742960316063268, 8.120618672364218, 8.267106571468295, 8.374259580796988, 8.376316239085835, 8.417539637928913, 8.497563289843415, 8.736736587476571, 10.046925060425039, 24.422370403537176, 81.55887565618339, 144.3693817307372, 165.97325784782421, 160.1265051496547, 127.79487897572793, 57.80404292148022, 1.4568790460547332, 2.045155562040758, 1.981726273401938, 2.0592278340912795, 2.070087989125443, 2.1008717325762998, 2.0679788975311184, 2.209989832393998, 1.776966775218839, 3.4238580159059553, 8.120618672364222, 10.964934518564627, 11.27033702749808, 11.36961329419692, 11.451886839187685, 11.444632236620244, 11.668676334360047, 11.868498713679532, 15.050432792378812, 31.09933795681387, 86.14751723850155, 146.1579044161917, 166.96935792076488, 160.86072995121833, 128.17413363924157, 57.994348955497884, 1.6341620409226412, 2.390063436653354, 2.3303748078677766, 2.4276547744853256, 2.449060162515741, 2.4630813139230283, 2.477790284242655, 2.5437068219502312, 2.2140391897832816, 3.7805839864957598, 8.267106571468288, 11.270337027498089, 11.93965083396728, 12.038755381740856, 12.053730311077915, 12.227292630889142, 12.19566123537657, 12.983304780642573, 16.17743135143111, 32.0489145834344, 88.27452120927204, 147.90819327732208, 167.76642724431596, 161.43674258546125, 128.49341443893323, 58.10446454427029, 1.5935741900342273, 2.2775031380960047, 2.2499656475928718, 2.321110866224127, 2.3428804661519744, 2.378048052101123, 2.3393545200307084, 2.4911137873837994, 2.0697927328610057, 3.7043243721943058, 8.374259580796993, 11.369613294196904, 12.038755381740874, 12.18559332536684, 12.328010491509835, 12.277300920104178, 12.546900678464555, 13.214840190656266, 16.284936244378848, 32.66634644541683, 89.16906313467246, 149.14506962932634, 168.58223413721174, 161.8918131433966, 128.77911452656986, 58.18752227755243, 1.5903320423316885, 2.3168126353514436, 2.252459657403741, 2.351787083695508, 2.3612338391451866, 2.3883752001158474, 2.3862330581345663, 2.475689507117511, 2.1302745007409265, 3.7178010831258197, 8.376316239085869, 11.451886839187692, 12.053730311077882, 12.328010491509872, 12.302960738973951, 12.452060660632718, 12.620958229551729, 13.211919635173706, 16.540109764073172, 32.84503176725284, 89.86607210182973, 150.03742913118867, 169.38486953715204, 162.35863503415837, 129.00978267065744, 58.28673812074753, 1.6115378016864161, 2.308659965835076, 2.2719823580275125, 2.346418979130619, 2.3793404825515596, 2.390516813244815, 2.390815129399367, 2.503525703826107, 2.1017411865512665, 3.7805469975541928, 8.417539637928886, 11.44463223662028, 12.22729263088917, 12.277300920104128, 12.45206066063277, 12.518526607135566, 12.628768827108859, 13.41791751997671, 16.56232730544406, 33.169848586646005, 90.53392732443393, 151.0938675443119, 170.20894294019223, 162.8738118004407, 129.25227136496602, 58.33330958077015, 1.5650607566688806, 2.2743165329423793, 2.2300129197535474, 2.3213280519133535, 2.3241111792236064, 2.3788664342463552, 2.3295618182394953, 2.48323561063368, 2.0797541780303326, 3.660985777732084, 8.497563289843447, 11.668676334360006, 12.195661235376646, 12.54690067846462, 12.620958229551677, 12.628768827108942, 12.890700752434107, 13.501436570410709, 16.892906913870814, 33.59005170561663, 91.71440639431836, 152.56717815491226, 171.3324921455309, 163.4408840634369, 129.43542433878358, 58.43341499126569, 1.7045316312415477, 2.4629054287209082, 2.408436115185135, 2.499241973050402, 2.539717075837889, 2.538086857573576, 2.575843804093822, 2.6223541213274597, 2.3419558608898394, 4.095833303109524, 8.736736587476603, 11.86849871367961, 12.983304780642532, 13.21484019065644, 13.211919635173878, 13.417917519976692, 13.501436570410945, 14.244993122812925, 17.63356266605567, 34.94908066569448, 94.00107210418497, 155.26186180928326, 172.86739063071997, 163.91656352913643, 129.68918921173054, 58.465801516505394, 1.2734991086719345, 1.7963006177351577, 1.7480793281311433, 1.8387789915156565, 1.8345932377132952, 1.8902529382989257, 1.8257839671623863, 2.0071538460574696, 1.5846620039841461, 3.261197827190305, 10.046925060425014, 15.050432792378947, 16.17743135143139, 16.28493624437889, 16.54010976407368, 16.562327305444878, 16.892906913871148, 17.633562666056214, 20.994619864879642, 38.87773467163522, 100.02402238074468, 160.0150529424735, 174.34504531873017, 164.51387430087055, 129.8507947405487, 58.48176739405738, 3.576603163450759, 6.278077675723091, 7.4205814579383995, 8.155622749596555, 8.599928842523145, 8.895222632933915, 9.021867772479741, 9.404187120327574, 8.76009959241488, 12.887111346700513, 24.422370403537446, 31.099337956814182, 32.04891458343504, 32.66634644541787, 32.84503176725384, 33.1698485866476, 33.590051705618926, 34.949080665696826, 38.87773467163675, 57.666594214928985, 114.65156502346781, 165.67708662609118, 175.85801255860508, 165.12128214127915, 129.83900752132863, 58.513818176360466, 18.860741036813298, 39.6721267135773, 51.65549968663788, 59.04898173983067, 63.65134946905359, 66.57180380971812, 68.55263499801436, 69.93316120605974, 71.57346462624955, 75.64780411219334, 81.55887565618431, 86.14751723850307, 88.27452120927414, 89.16906313467588, 89.86607210183462, 90.53392732444013, 91.71440639432734, 94.001072104196, 100.02402238075543, 114.65156502347496, 144.25290014907858, 172.02495259432217, 178.02559229521844, 165.01939698112238, 129.61865909778925, 58.27221066041721, 33.72794472072706, 73.60585007288253, 98.67450898225307, 114.35247778891654, 124.27493426082745, 130.58741966224534, 134.69104816901768, 137.65724668215992, 140.48023802786986, 142.74143212084246, 144.36938173073895, 146.1579044161946, 147.9081932773267, 149.14506962933254, 150.0374291311982, 151.09386754432603, 152.5671781549294, 155.261861809305, 160.01505294249836, 165.67708662610727, 172.02495259432473, 178.4899061893612, 178.18065594082856, 163.60611889451187, 128.11301973411224, 57.68224875306563, 37.57285678733437, 83.00604251706527, 113.00599457070841, 132.34739457801234, 144.605099989098, 152.3688260816393, 157.35564642297282, 160.75943598821164, 162.91089240896883, 164.5113091520661, 165.97325784782652, 166.96935792076837, 167.76642724432133, 168.58223413722035, 169.38486953716333, 170.2089429402073, 171.33249214555363, 172.86739063074685, 174.3450453187532, 175.85801255861773, 178.02559229521694, 178.18065594082088, 173.13904464444994, 158.26625716453353, 124.08124697039888, 56.022865377986555, 36.86285660554542, 81.38231798872509, 110.578744352507, 129.23837649886954, 141.0051857615539, 148.39337693537357, 153.09007747183213, 155.97865438600482, 157.95537473603008, 159.3129130006215, 160.12650514965657, 160.86072995122154, 161.4367425854664, 161.8918131434043, 162.35863503417022, 162.87381180045645, 163.44088406345406, 163.91656352915575, 164.51387430088963, 165.12128214128543, 165.01939698111394, 163.60611889450033, 158.26625716452702, 144.70539203782346, 114.56615313119765, 52.07938446883457, 31.67437967454599, 68.61884375355326, 91.33102896837671, 105.32244583288339, 114.034245602417, 119.46721052833172, 122.80125322905764, 124.96991844343952, 126.30694393851572, 127.13281906894224, 127.79487897572912, 128.17413363924362, 128.493414438937, 128.77911452657546, 129.00978267066668, 129.25227136498017, 129.43542433879975, 129.6891892117457, 129.8507947405605, 129.83900752132632, 129.61865909777674, 128.11301973410386, 124.08124697039416, 114.56615313119568, 92.78403937930885, 43.36748841208011, 15.77525158210328, 32.998981037108145, 42.73622842884725, 48.64266270441468, 52.250318287249456, 54.45228725274596, 55.86981725933842, 56.70091159759281, 57.235247556894855, 57.626403827389375, 57.80404292148062, 57.9943489554982, 58.10446454427148, 58.18752227755579, 58.286738120751174, 58.33330958077595, 58.43341499128481, 58.465801516521054, 58.48176739405066, 58.51381817634844, 58.272210660411005, 57.68224875306193, 56.02286537798462, 52.07938446883402, 43.367488412079716, 20.97956013167017]]]

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
