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
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 1.0
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = True
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 0
        pydgm.control.ignore_warnings = True
        
    def setGroups(self, G):
        if G == 2:
            pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)
            pydgm.control.dgm_basis_name = 'test/2gbasis'.ljust(256)
            pydgm.control.energy_group_map = [1]
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
    
    def test_dgmsolver_2gtest(self):
        '''
        Test the 2g-1G dgm fixed source problem with vacuum conditions
        
        Using one spatial region
        
        no fission
        '''
        # Set the variables for the test
        self.setGroups(2)
        self.setSolver('fixed')
        self.setMesh('1')
        self.setBoundary('vacuum')
        pydgm.control.material_map = [1]
        
        phi_test = np.reshape([1.1420149990909008, 0.37464706668551212], (1, 2, 1), 'F')
        psi_test = np.reshape([0.81304488744042813, 0.29884810509581583, 1.31748796916478740, 0.41507830480599001, 1.31748796916478740, 0.41507830480599001, 0.81304488744042813, 0.29884810509581583], (2, 4, 1))

        pydgm.dgmsolver.initialize_dgmsolver()

        pydgm.state.phi = phi_test
        pydgm.state.psi = psi_test

        phi_new = np.reshape(np.zeros(2), phi_test.shape, 'F')
        psi_new = np.reshape(np.zeros(8), psi_test.shape, 'F')

        pydgm.dgmsweeper.dgmsweep(phi_new, psi_new, pydgm.dgmsolver.incoming)

        np.testing.assert_array_almost_equal(pydgm.state.phi, phi_test, 12)
        np.testing.assert_array_almost_equal(pydgm.state.psi, psi_test, 12)

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
        pydgm.control.material_map = [6, 1, 6]
        pydgm.control.angle_order = 10
        pydgm.control.allow_fission = True
        pydgm.control.lamb = 0.75

        pydgm.dgmsolver.initialize_dgmsolver()

        phi_test = [1.8461980363287278, 4.16904423296633, 1.6371065827443676, 1.3260798327554102, 1.1507280372172657, 1.221454815786205, 0.21787346591975096, 1.8779676212462644, 4.346293616739397, 1.7911796347394968, 1.4696141054340566, 1.2762011622747407, 1.410548033592605, 0.2805431110268466, 1.9076720822947586, 4.5235324366607985, 1.9348875590099663, 1.5995201255730067, 1.3850154736028073, 1.5361742894430894, 0.3503620127705863, 1.9653381411150332, 4.755704786297735, 2.0602776849402016, 1.7086687870124164, 1.473314427190565, 1.6032397558933036, 0.41859495632314897, 2.046831950358654, 5.024852817609021, 2.1627268750309767, 1.794379906316901, 1.5415868503767243, 1.6402668270670566, 0.45783584286439, 2.1183287442900616, 5.259894345939496, 2.249750972825359, 1.8662129456617995, 1.598567531294177, 1.6740654516371898, 0.48316274993701736, 2.180431853934209, 5.463189536952539, 2.323379278194582, 1.9263940668089787, 1.6461554034562276, 1.7042440374045473, 0.5017303824642151, 2.233671544963958, 5.636838362725401, 2.3851794675942126, 1.976556103614348, 1.6857277137465003, 1.7306376835631267, 0.5160940364316613, 2.278501420880871, 5.782620523285358, 2.43635508075605, 2.0178957585753343, 1.7182838252262809, 1.753193622571626, 0.5273657105739336, 2.3153010315366376, 5.902002999445916, 2.4778225682576234, 2.051286995516379, 1.7445473309392545, 1.7719149818882431, 0.5361455617094784, 2.344379734887513, 5.996163165441185, 2.5102702850685095, 2.0773624527330834, 1.7650382804142186, 1.7868302159372544, 0.5428111210518882, 2.365980375010593, 6.066012532308792, 2.534202681742982, 2.096571354470798, 1.7801238463024747, 1.7979755323610467, 0.5476165734363021, 2.380282353001199, 6.112216861028091, 2.54997251963054, 2.1092199413577197, 1.790053258345466, 1.8053845815737255, 0.5507325908285529, 2.3874039642095606, 6.135211472153911, 2.557803276430926, 2.115498615897139, 1.7949810125894576, 1.809082468016729, 0.5522651544630954, 2.3874039642095597, 6.1352114721539115, 2.5578032764309264, 2.1154986158971383, 1.7949810125894576, 1.809082468016729, 0.5522651544630954, 2.380282353001199, 6.112216861028088, 2.5499725196305403, 2.1092199413577197, 1.790053258345467, 1.8053845815737257, 0.5507325908285529, 2.365980375010593, 6.0660125323087914, 2.534202681742981, 2.096571354470798, 1.7801238463024753, 1.797975532361047, 0.547616573436302, 2.3443797348875135, 5.996163165441185, 2.5102702850685086, 2.0773624527330825, 1.7650382804142186, 1.7868302159372544, 0.5428111210518881, 2.315301031536638, 5.9020029994459176, 2.477822568257624, 2.0512869955163793, 1.7445473309392545, 1.7719149818882434, 0.5361455617094785, 2.278501420880872, 5.782620523285358, 2.436355080756051, 2.0178957585753348, 1.718283825226282, 1.753193622571626, 0.5273657105739337, 2.233671544963958, 5.636838362725398, 2.3851794675942113, 1.976556103614348, 1.6857277137465008, 1.7306376835631265, 0.516094036431661, 2.1804318539342096, 5.463189536952541, 2.323379278194582, 1.9263940668089785, 1.646155403456228, 1.704244037404548, 0.5017303824642152, 2.1183287442900616, 5.259894345939497, 2.2497509728253595, 1.8662129456617995, 1.5985675312941772, 1.6740654516371902, 0.48316274993701747, 2.046831950358654, 5.024852817609021, 2.1627268750309767, 1.7943799063169015, 1.5415868503767245, 1.640266827067057, 0.45783584286439005, 1.9653381411150337, 4.755704786297734, 2.060277684940202, 1.708668787012417, 1.4733144271905656, 1.6032397558933043, 0.4185949563231489, 1.9076720822947586, 4.5235324366608, 1.9348875590099668, 1.5995201255730072, 1.385015473602808, 1.53617428944309, 0.35036201277058626, 1.8779676212462644, 4.346293616739396, 1.791179634739497, 1.469614105434057, 1.2762011622747407, 1.4105480335926057, 0.2805431110268464, 1.8461980363287276, 4.169044232966329, 1.6371065827443676, 1.3260798327554109, 1.1507280372172666, 1.2214548157862055, 0.21787346591975099]
        pydgm.state.phi[0, :, :] = np.reshape(phi_test, (7, 28), 'F')

        pydgm.dgmsolver.dgmsolve()

        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)

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
        pydgm.control.lamb = 0.73
        pydgm.control.max_inner_iters = 10

        pydgm.dgmsolver.initialize_dgmsolver()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:, 0])
        S = pydgm.material.sig_s[0, :, :, 0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))
        phi_test = np.array([phi_test for i in range(28)]).flatten()

        pydgm.dgmsolver.dgmsolve()

        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :].flatten('F') / phi_test, np.ones(28 * 7), 12)

        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)

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

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        phi_test = np.array([1.0396071685339883, 1.1598847130734862, 1.1042907473694785, 1.035767541702943, 0.9363281958603344, 0.8907827126246091, 0.3974913066660486])

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)

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
        pydgm.control.lamb = 0.82
        pydgm.control.equation_type = 'DD'
        pydgm.control.max_inner_iters = 10

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        phi_test = np.array([13.200797328158604, 104.5813972488177, 147.3613380521834, 28.447902090302442, 7.138235800921474, 3.6663426589179435, 0.64574230078707])

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F') / phi_test, np.ones(pydgm.mesh.number_cells * pydgm.material.number_groups), 12)

        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)
        
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
        assert_almost_equal(pydgm.state.d_keff, 0.809952323298, 12)

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F') / phi_test, np.ones(pydgm.mesh.number_cells * pydgm.material.number_groups), 12)

        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)
        
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
        pydgm.control.lamb = 0.14
        pydgm.control.max_inner_iters = 100
        pydgm.control.max_outer_iters = 5000
        
        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        phi_test = np.array([1.6283945282803138, 1.2139688020213637, 0.03501217302426163, 6.910819115000905e-17, 1.9641731545343404, 1.59852298044156, 0.05024813427412045, 1.0389359842780806e-16, 2.2277908375593736, 1.8910978193073922, 0.061518351747482505, 1.3055885402420332e-16, 2.40920191588961, 2.088554929299159, 0.06902375359471126, 1.487795240822353e-16, 2.5016194254000244, 2.188087672560707, 0.0727855220655801, 1.5805185521208351e-16, 2.501619425400025, 2.1880876725607075, 0.07278552206558009, 1.5805185521208351e-16, 2.40920191588961, 2.088554929299159, 0.06902375359471127, 1.487795240822353e-16, 2.2277908375593736, 1.891097819307392, 0.0615183517474825, 1.3055885402420332e-16, 1.9641731545343404, 1.59852298044156, 0.05024813427412045, 1.0389359842780806e-16, 1.6283945282803138, 1.2139688020213637, 0.03501217302426163, 6.910819115000904e-17])

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.d_keff, 0.185134666261, 12)
        
        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)
        
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
        pydgm.control.lamb = 0.14
        pydgm.control.max_inner_iters = 10
        
        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set the test flux
        phi_test = np.array([0.22714604491018248, 3.2561614797025626, 0.6702436453773765, 0.028830522373756134, 0.0007442365449145364, 1.2302158377300607e-05, 8.046715731304513e-09, 0.30497847644838055, 4.650755985628613, 1.0317922178126702, 0.0462715482473371, 0.0012069810230059736, 2.0794329933149104e-05, 1.4674487232071664e-08, 0.3665461901653671, 5.721763243090282, 1.30389832484028, 0.05938705865400071, 0.0015570897974455599, 2.702796443933454e-05, 1.8912150015627453e-08, 0.409131837670784, 6.4514644737257125, 1.4884299834739563, 0.0683248044739793, 0.0017966497489318545, 3.1296360712972986e-05, 2.20048002048578e-08, 0.43089325136800743, 6.821714908229378, 1.5820419375564987, 0.072875235926122, 0.00191888973591009, 3.3475512155773155e-05, 2.3510783444432865e-08, 0.4308932513680075, 6.821714908229377, 1.582041937556499, 0.07287523592612198, 0.0019188897359100898, 3.347551215577316e-05, 2.351078344443286e-08, 0.409131837670784, 6.451464473725713, 1.4884299834739563, 0.06832480447397932, 0.0017966497489318538, 3.1296360712972986e-05, 2.200480020485779e-08, 0.36654619016536716, 5.721763243090282, 1.3038983248402798, 0.05938705865400071, 0.0015570897974455599, 2.7027964439334537e-05, 1.8912150015627453e-08, 0.3049784764483806, 4.650755985628613, 1.03179221781267, 0.046271548247337105, 0.0012069810230059738, 2.0794329933149104e-05, 1.4674487232071663e-08, 0.2271460449101824, 3.256161479702563, 0.6702436453773764, 0.02883052237375614, 0.0007442365449145365, 1.2302158377300605e-05, 8.046715731304513e-09])

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.d_keff, 0.2186665578996815, 12)

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)

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
        
        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:,0])
        S = pydgm.material.sig_s[0,:,:,0].T
        X = np.outer(pydgm.material.chi[:,0], pydgm.material.nu_sig_f[:,0])
        
        keff, phi = np.linalg.eig(np.linalg.inv(T - S).dot(X))
        i = np.argmax(keff)
        keff_test = keff[i]
        phi_test = phi[:,i]
        
        phi_test = np.array([phi_test for i in range(10)]).flatten()
        
        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)
        
        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)
        
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
        T = np.diag(pydgm.material.sig_t[:,0])
        S = pydgm.material.sig_s[0,:,:,0].T
        X = np.outer(pydgm.material.chi[:,0], pydgm.material.nu_sig_f[:,0])
        
        keff, phi = np.linalg.eig(np.linalg.inv(T - S).dot(X))
        i = np.argmax(keff)
        keff_test = keff[i]
        phi_test = phi[:,i]
        
        phi_test = np.array([phi_test for i in range(10)]).flatten()
        
        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)
        
        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)
    
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
        pydgm.control.lamb = 0.1
        pydgm.control.max_inner_iters = 2
        pydgm.control.max_outer_iters = 5000
        
        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:,0])
        S = pydgm.material.sig_s[0,:,:,0].T
        X = np.outer(pydgm.material.chi[:,0], pydgm.material.nu_sig_f[:,0])
        
        keff, phi = np.linalg.eig(np.linalg.inv(T - S).dot(X))
        i = np.argmax(keff)
        keff_test = keff[i]
        phi_test = phi[:,i]
        
        phi_test = np.array([phi_test for i in range(10)]).flatten()
        
        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        assert_almost_equal(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)
        
        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)
        
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
        
        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()
        
        # set the test flux
        keff_test = 0.8418546852484950
        phi_test = [0.13393183108467394, 0.04663240631432256, 0.13407552941360298, 0.04550808086281801, 0.13436333428621713, 0.043206841474147446, 0.1351651393398092, 0.0384434752119791, 0.13615737742196526, 0.03329929560434661, 0.13674284660888314, 0.030464508103354708, 0.13706978363298242, 0.028970199506203023, 0.13721638515632006, 0.028325674662651124, 0.13721638515632006, 0.028325674662651124, 0.1370697836329824, 0.028970199506203012, 0.13674284660888308, 0.0304645081033547, 0.13615737742196524, 0.03329929560434659, 0.13516513933980914, 0.03844347521197908, 0.13436333428621713, 0.043206841474147425, 0.13407552941360296, 0.045508080862818004, 0.1339318310846739, 0.046632406314322555]

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)
        
        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)

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
        pydgm.control.max_inner_iters = 1
        pydgm.control.max_outer_iters = 5000
        
        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()
        
        # set the test flux
        keff_test = 0.759180925837
        phi_test = [2.22727714687889, 1.7369075872008062, 0.03381777446256108, 1.7036045485771946e-51, 2.227312836950116, 1.737071399085861, 0.033828681689533874, 1.698195884848375e-51, 2.2273842137861877, 1.737399076914814, 0.03385050700298498, 1.6965081411970257e-51, 2.2275366771427696, 1.738049905609514, 0.033892498696060584, 1.697491400636173e-51, 2.2277285082016354, 1.7388423494826295, 0.03394281751690741, 1.6984869893367e-51, 2.2278725176984917, 1.7394359801217965, 0.03398044173136341, 1.697888141875072e-51, 2.2279685888050587, 1.739831399863755, 0.03400546998960266, 1.6957049057554318e-51, 2.2280166437743327, 1.7400290096083806, 0.034017967785934036, 1.691942073409801e-51, 2.2280166437743327, 1.7400290096083808, 0.03401796778593402, 1.68659920584123e-51, 2.2279685888050587, 1.7398313998637547, 0.03400546998960263, 1.6796706200591657e-51, 2.2278725176984917, 1.7394359801217967, 0.03398044173136335, 1.6711453402288656e-51, 2.227728508201635, 1.73884234948263, 0.033942817516907337, 1.6610070122205585e-51, 2.2275366771427696, 1.7380499056095144, 0.0338924986960605, 1.6492337810058256e-51, 2.227384213786188, 1.7373990769148142, 0.03385050700298487, 1.6385765949262272e-51, 2.2273128369501163, 1.7370713990858613, 0.03382868168953376, 1.631610014066153e-51, 2.2272771468788894, 1.7369075872008064, 0.03381777446256096, 1.6281341640813905e-51]

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)
        
        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)
    
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
        pydgm.control.material_map = [6, 1, 6]
        pydgm.control.angle_order = 2
        pydgm.control.lamb = 0.001
        pydgm.control.max_inner_iters = 100
        pydgm.control.max_outer_iters = 5000
        
        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()
        
        # set the test flux
        keff_test = 0.7735620332498745
        phi_test = [0.13187762273035553, 3.0560002021766643, 1.7745035683301031, 0.6497468427915719, 0.14448308078823344, 0.019895928705131523, 8.31379552547061e-05, 0.13188288634468276, 3.056403228279507, 1.774758068899994, 0.6487626069789563, 0.1438285999231679, 0.019608234905321438, 7.806909778328154e-05, 0.13189341388973588, 3.057209396485762, 1.7752674363208387, 0.6467907571461935, 0.14251875998107638, 0.019032507623511552, 6.727347241536639e-05, 0.1319513233456591, 3.0597550377239013, 1.7762411002132281, 0.6431245521476735, 0.14007992707737454, 0.018155959093001727, 5.245693945483657e-05, 0.13204157473703287, 3.063370328686325, 1.7774026907915714, 0.6388212800497931, 0.1372214690255107, 0.017251311003616352, 3.986781991875587e-05, 0.13210777149287783, 3.066060181391546, 1.7782690984803549, 0.6356261228532062, 0.13510955968997732, 0.01659332772257151, 3.238107030308677e-05, 0.1321512038138374, 3.0678432530705977, 1.7788444419391838, 0.6335112726127647, 0.13371667329125442, 0.0161642530798633, 2.8181571408842145e-05, 0.13217271256078084, 3.0687317651283674, 1.7791314385277586, 0.6324583990047713, 0.13302471551183112, 0.015952553852704478, 2.6294217726345094e-05, 0.13217271256078084, 3.068731765128368, 1.7791314385277583, 0.6324583990047713, 0.13302471551183115, 0.015952553852704478, 2.62942177263451e-05, 0.13215120381383738, 3.0678432530705986, 1.7788444419391842, 0.6335112726127649, 0.1337166732912544, 0.016164253079863303, 2.818157140884216e-05, 0.1321077714928778, 3.0660601813915465, 1.7782690984803553, 0.6356261228532063, 0.13510955968997732, 0.016593327722571518, 3.238107030308679e-05, 0.13204157473703285, 3.063370328686326, 1.7774026907915719, 0.6388212800497933, 0.1372214690255107, 0.017251311003616356, 3.986781991875591e-05, 0.13195132334565907, 3.0597550377239027, 1.7762411002132288, 0.6431245521476738, 0.14007992707737454, 0.018155959093001734, 5.2456939454836604e-05, 0.13189341388973586, 3.0572093964857636, 1.775267436320839, 0.6467907571461937, 0.14251875998107638, 0.019032507623511562, 6.727347241536645e-05, 0.1318828863446827, 3.0564032282795073, 1.7747580688999944, 0.6487626069789565, 0.14382859992316793, 0.019608234905321452, 7.806909778328162e-05, 0.1318776227303555, 3.0560002021766643, 1.774503568330104, 0.6497468427915722, 0.14448308078823346, 0.01989592870513153, 8.313795525470619e-05]

        # Solve the problem
        pydgm.dgmsolver.dgmsolve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.d_keff, keff_test, 12)

        # Test the scalar flux
        phi = pydgm.state.phi[0, :, :].flatten('F')
        np.testing.assert_array_almost_equal(phi / phi[0] * phi_test[0], phi_test, 12)
        
        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)
        
    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()

