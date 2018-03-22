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
        pydgm.control.inner_print = False
        pydgm.control.eigen_tolerance = 1e-15
        pydgm.control.outer_tolerance = 1e-16
        pydgm.control.inner_tolerance = 1e-16
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 0
        pydgm.control.use_DGM = False

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
        np.testing.assert_array_almost_equal(pydgm.state.d_phi[0, :, :].flatten(), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        np.testing.assert_array_almost_equal(pydgm.state.d_phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        np.testing.assert_array_almost_equal(pydgm.state.d_phi[0, :, :].flatten(), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        phi = pydgm.state.d_phi[0, :, :].flatten()

        np.testing.assert_array_almost_equal(phi / phi_test, np.ones(28 * 7), 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        np.testing.assert_array_almost_equal(pydgm.state.d_phi[0, :, :].flatten(), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        np.testing.assert_array_almost_equal(pydgm.state.d_phi[0, :, :].flatten(), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        phi = pydgm.state.d_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi_test, np.ones(70), 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        assert(pydgm.control.number_groups == 1)
        
        phi_test = [0.05917851752472814, 0.1101453392055481, 0.1497051827466689, 0.1778507990738045, 0.1924792729907672, 0.1924792729907672, 0.1778507990738046, 0.1497051827466690, 0.1101453392055482, 0.05917851752472817]

        # Solve the problem
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.d_keff, 0.6893591115415211, 12)

        # Test the scalar flux
        phi = pydgm.state.d_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        self.assertAlmostEqual(pydgm.state.d_keff, 0.8099523232983425, 12)

        # Test the scalar flux
        phi = pydgm.state.d_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        self.assertAlmostEqual(pydgm.state.d_keff, 0.30413628310914226, 12)

        # Test the scalar flux
        phi = pydgm.state.d_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.d_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        pydgm.control.angle_order = 2
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
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.d_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.d_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.d_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.d_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
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
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.d_phi[0, :, :].flatten()
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.control.number_angles
        phi_test = np.zeros((pydgm.control.number_cells, pydgm.control.number_groups))
        for c in range(pydgm.control.number_cells):
            for a in range(nAngles):
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, a, :]
                phi_test[c] += pydgm.angle.wt[a] * pydgm.state.psi[c, 2 * nAngles - a - 1, :]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test, 12)

    # TODO: Add anisotropic tests

    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()

