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
        pydgm.control.material_map = [6, 1, 6]
        s = 'test.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 10
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = False
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.Lambda = 1.0
        pydgm.control.store_psi = True
        s = 'fixed'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 1.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 0

    def test_solver_vacuum1(self):
        ''' 
        Test fixed source problem with vacuum conditions
        '''

        # Activate fissioning
        pydgm.control.allow_fission = True

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        phi_test = [1.8461980363287278, 4.16904423296633, 1.6371065827443676, 1.3260798327554102, 1.1507280372172657, 1.221454815786205, 0.21787346591975096, 1.8779676212462644, 4.346293616739397, 1.7911796347394968, 1.4696141054340566, 1.2762011622747407, 1.410548033592605, 0.2805431110268466, 1.9076720822947586, 4.5235324366607985, 1.9348875590099663, 1.5995201255730067, 1.3850154736028073, 1.5361742894430894, 0.3503620127705863, 1.9653381411150332, 4.755704786297735, 2.0602776849402016, 1.7086687870124164, 1.473314427190565, 1.6032397558933036, 0.41859495632314897, 2.046831950358654, 5.024852817609021, 2.1627268750309767, 1.794379906316901, 1.5415868503767243, 1.6402668270670566, 0.45783584286439, 2.1183287442900616, 5.259894345939496, 2.249750972825359, 1.8662129456617995, 1.598567531294177, 1.6740654516371898, 0.48316274993701736, 2.180431853934209, 5.463189536952539, 2.323379278194582, 1.9263940668089787, 1.6461554034562276, 1.7042440374045473, 0.5017303824642151, 2.233671544963958, 5.636838362725401, 2.3851794675942126, 1.976556103614348, 1.6857277137465003, 1.7306376835631267, 0.5160940364316613, 2.278501420880871, 5.782620523285358, 2.43635508075605, 2.0178957585753343, 1.7182838252262809, 1.753193622571626, 0.5273657105739336, 2.3153010315366376, 5.902002999445916, 2.4778225682576234, 2.051286995516379, 1.7445473309392545, 1.7719149818882431, 0.5361455617094784, 2.344379734887513, 5.996163165441185, 2.5102702850685095, 2.0773624527330834, 1.7650382804142186, 1.7868302159372544, 0.5428111210518882, 2.365980375010593, 6.066012532308792, 2.534202681742982, 2.096571354470798, 1.7801238463024747, 1.7979755323610467, 0.5476165734363021, 2.380282353001199, 6.112216861028091, 2.54997251963054, 2.1092199413577197, 1.790053258345466, 1.8053845815737255, 0.5507325908285529, 2.3874039642095606, 6.135211472153911, 2.557803276430926, 2.115498615897139, 1.7949810125894576, 1.809082468016729, 0.5522651544630954, 2.3874039642095597, 6.1352114721539115, 2.5578032764309264, 2.1154986158971383, 1.7949810125894576, 1.809082468016729, 0.5522651544630954, 2.380282353001199, 6.112216861028088, 2.5499725196305403, 2.1092199413577197, 1.790053258345467, 1.8053845815737257, 0.5507325908285529, 2.365980375010593, 6.0660125323087914, 2.534202681742981, 2.096571354470798, 1.7801238463024753, 1.797975532361047, 0.547616573436302, 2.3443797348875135, 5.996163165441185, 2.5102702850685086, 2.0773624527330825, 1.7650382804142186, 1.7868302159372544, 0.5428111210518881, 2.315301031536638, 5.9020029994459176, 2.477822568257624, 2.0512869955163793, 1.7445473309392545, 1.7719149818882434, 0.5361455617094785, 2.278501420880872, 5.782620523285358, 2.436355080756051, 2.0178957585753348, 1.718283825226282, 1.753193622571626, 0.5273657105739337, 2.233671544963958, 5.636838362725398, 2.3851794675942113, 1.976556103614348, 1.6857277137465008, 1.7306376835631265, 0.516094036431661, 2.1804318539342096, 5.463189536952541, 2.323379278194582, 1.9263940668089785, 1.646155403456228, 1.704244037404548, 0.5017303824642152, 2.1183287442900616, 5.259894345939497, 2.2497509728253595, 1.8662129456617995, 1.5985675312941772, 1.6740654516371902, 0.48316274993701747, 2.046831950358654, 5.024852817609021, 2.1627268750309767, 1.7943799063169015, 1.5415868503767245, 1.640266827067057, 0.45783584286439005, 1.9653381411150337, 4.755704786297734, 2.060277684940202, 1.708668787012417, 1.4733144271905656, 1.6032397558933043, 0.4185949563231489, 1.9076720822947586, 4.5235324366608, 1.9348875590099668, 1.5995201255730072, 1.385015473602808, 1.53617428944309, 0.35036201277058626, 1.8779676212462644, 4.346293616739396, 1.791179634739497, 1.469614105434057, 1.2762011622747407, 1.4105480335926057, 0.2805431110268464, 1.8461980363287276, 4.169044232966329, 1.6371065827443676, 1.3260798327554109, 1.1507280372172666, 1.2214548157862055, 0.21787346591975099]

        # Solve the problem
        pydgm.solver.solve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)

    def test_solver_reflect1(self):
        '''
        Test fixed source problem with reflective conditions
        '''
        # Set problem conditions
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.material_map = [1, 1, 1]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()
        
        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:,0])
        S = pydgm.material.sig_s[0,:,:,0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))
        phi_test = np.array([phi_test for i in range(28)]).flatten()

        # Solve the problem
        pydgm.solver.solve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)

    def test_solver_reflect2(self):
        '''
        Test fixed source problem with reflective conditions
        '''
        # Set problem conditions
        pydgm.control.fine_mesh = [5, 5]
        pydgm.control.coarse_mesh = [0.0, 1.0, 2.0]
        pydgm.control.material_map = [2, 1]
        pydgm.control.angle_order = 2
        pydgm.control.boundary_type = [1.0, 1.0]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        phi_test = [12.672544648860312, 99.03671817311842, 130.2026866195674, 26.18559409855547, 6.400759203994376, 2.2337860609134186, 0.5196203316023417, 12.67448693202096, 99.04405976288407, 130.2324279776323, 26.190965696177237, 6.4088876008405276, 2.2595181875357944, 0.5209530580408911, 12.678388073294773, 99.05293427296532, 130.260637716364, 26.201843332427238, 6.425520342588559, 2.315655310875237, 0.5244779835956537, 12.684281397852965, 99.06338579650257, 130.28761750801002, 26.21851193557012, 6.4514324393938445, 2.412634999681313, 0.5334157340996336, 12.692217333538476, 99.07546520784048, 130.31365794340428, 26.241425884913447, 6.487845367812591, 2.5689673440912353, 0.5599294161703391, 12.700953021960547, 99.08787298541745, 130.33733158501667, 26.267144568832023, 6.528791987404324, 2.7375849590876493, 0.5972361793074424, 12.708475057710633, 99.09843071210251, 130.35651191444853, 26.289282550132317, 6.56345322945365, 2.861617320600131, 0.6215389354912455, 12.71406726930836, 99.10630761898258, 130.37072943890948, 26.305411950605524, 6.5883033624105725, 2.946942044099785, 0.6317018311707815, 12.71777205870059, 99.11153934332636, 130.38012940316244, 26.31594523771697, 6.60434224992093, 3.0003079242325135, 0.6361656468391427, 12.719617474155664, 99.1141493736049, 130.38480616286316, 26.321147187894013, 6.612206809491813, 3.0259691800610304, 0.6379141464275061]

        # Solve the problem
        pydgm.solver.solve()

        # Test the scalar flux
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :].flatten('F'), phi_test, 12)

        # Test the angular flux
        nAngles = pydgm.angle.number_angles
        phi_test = np.zeros((pydgm.mesh.number_cells, pydgm.material.number_groups))
        for c in range(pydgm.mesh.number_cells):
            for a in range(nAngles):
                phi_test[c] += 0.5 * pydgm.angle.wt[a] * pydgm.state.psi[:, a, c]
                phi_test[c] += 0.5 * pydgm.angle.wt[nAngles - a - 1] * pydgm.state.psi[:, 2 * nAngles - a - 1, c]
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :], phi_test.T, 12)

    def test_solver_eigenV1g(self):
        '''
        Test eigenvalue source problem with vacuum conditions and 1g
        '''
        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        s = 'test/1gXS.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 2
        pydgm.control.allow_fission = True
        s = 'eigen'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        phi_test = [0.05917851752472814, 0.1101453392055481, 0.1497051827466689, 0.1778507990738045, 0.1924792729907672, 0.1924792729907672, 0.1778507990738046, 0.1497051827466690, 0.1101453392055482, 0.05917851752472817]

        # Solve the problem
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.d_keff, 0.6893591115415211, 12)

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

    def test_solver_eigenV2g(self):
        '''
        Test eigenvalue source problem with vacuum conditions and 2g
        '''
        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        s = 'test/2gXS.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 2
        pydgm.control.allow_fission = True
        s = 'eigen'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        phi_test = [0.05882749189351658, 0.00985808742274279, 0.1099501419733753, 0.019347812329268695, 0.14979920402552727, 0.026176658005947512, 0.1781310366743269, 0.03122421632573625, 0.19286636083585607, 0.03377131381204501, 0.19286636083585607, 0.03377131381204501, 0.17813103667432686, 0.031224216325736256, 0.14979920402552724, 0.026176658005947512, 0.10995014197337528, 0.01934781232926869, 0.058827491893516604, 0.009858087422742794]

        # Solve the problem
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.d_keff, 0.8099523232983425, 12)

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
        s = 'eigen'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()

        phi_test = [0.05697980427410064, 0.8168112452571323, 0.16813126437954906, 0.0072321643224131415, 0.00018669245453791495, 3.086008284954765e-06, 2.0185263960916437e-09, 0.07650414473522332, 1.1666466210869162, 0.25882607221161863, 0.011607262471311624, 0.0003027723528297643, 5.21627770393424e-06, 3.6811092644681244e-09, 0.09194846505603246, 1.4353098237873552, 0.32708415139741975, 0.014897300896687289, 0.00039059747630968333, 6.779990927376679e-06, 4.744131057702992e-09, 0.1026311157740055, 1.6183560807301869, 0.37337409580512326, 0.017139343049238996, 0.0004506913210121037, 7.850722246217705e-06, 5.5199253381637e-09, 0.10808998737216971, 1.7112337590483877, 0.3968567447037582, 0.018280823164414765, 0.00048135533955243233, 8.39736448576657e-06, 5.897702685178424e-09, 0.1080899873721697, 1.7112337590483881, 0.3968567447037582, 0.018280823164414765, 0.00048135533955243233, 8.39736448576657e-06, 5.897702685178426e-09, 0.10263111577400548, 1.618356080730187, 0.3733740958051233, 0.017139343049238996, 0.00045069132101210364, 7.850722246217705e-06, 5.5199253381637e-09, 0.09194846505603246, 1.4353098237873554, 0.32708415139741975, 0.01489730089668729, 0.00039059747630968333, 6.7799909273766765e-06, 4.744131057702991e-09, 0.07650414473522334, 1.1666466210869164, 0.2588260722116186, 0.011607262471311622, 0.0003027723528297643, 5.216277703934239e-06, 3.6811092644681235e-09, 0.05697980427410064, 0.8168112452571323, 0.16813126437954903, 0.0072321643224131415, 0.00018669245453791487, 3.0860082849547647e-06, 2.0185263960916433e-09]

        # Solve the problem
        pydgm.solver.solve()

        # Test the eigenvalue
        self.assertAlmostEqual(pydgm.state.d_keff, 0.2186665578996815, 12)

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
        
    def test_solver_eigenR1g(self):
        '''
        Test eigenvalue source problem with reflective conditions and 1g
        '''
        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        s = 'test/1gXS.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 2
        pydgm.control.allow_fission = True
        s = 'eigen'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_type = [1.0, 1.0]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()
        
        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:,0])
        S = pydgm.material.sig_s[0,:,:,0].T
        X = np.outer(pydgm.material.chi[:,0], pydgm.material.nu_sig_f[:,0])
        
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
        
    def test_solver_eigenR2g(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''
        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        s = 'test/2gXS.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 2
        pydgm.control.allow_fission = True
        s = 'eigen'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_type = [1.0, 1.0]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()
        
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
        pydgm.solver.solve()

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
        s = 'eigen'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 0.0
        pydgm.control.boundary_type = [1.0, 1.0]

        # Initialize the dependancies
        pydgm.solver.initialize_solver()
        
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
        pydgm.solver.solve()

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

    def test_solver_eigenR1gPin(self):
        '''
        Test eigenvalue source problem with reflective conditions and 1g
        '''
        # Set the variables for the test
        pydgm.control.fine_mesh = [3, 10, 3]
        pydgm.control.material_map = [2, 1, 2]
        s = 'test/1gXS.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 2
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        s = 'eigen'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()
        
        # set the test flux
        keff_test = 0.6824742039858390
        phi_test = [0.13341837652120125, 0.13356075080242463, 0.1338459024547955, 0.13470975320182185, 0.13591529539088204, 0.13679997220977602, 0.13738073107705168, 0.13766845210630752, 0.1376684521063075, 0.1373807310770516, 0.1367999722097759, 0.13591529539088193, 0.1347097532018217, 0.13384590245479533, 0.1335607508024245, 0.13341837652120114]

        # Solve the problem
        pydgm.solver.solve()

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
        
    def test_solver_eigenR2gPin(self):
        '''
        Test eigenvalue source problem with reflective conditions and 2g
        '''
        # Set the variables for the test
        pydgm.control.fine_mesh = [3, 10, 3]
        pydgm.control.material_map = [2, 1, 2]
        s = 'test/2gXS.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 2
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        s = 'eigen'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()
        
        # set the test flux
        keff_test = 0.8418546852484950
        phi_test = [0.13393183108467394, 0.04663240631432256, 0.13407552941360298, 0.04550808086281801, 0.13436333428621713, 0.043206841474147446, 0.1351651393398092, 0.0384434752119791, 0.13615737742196526, 0.03329929560434661, 0.13674284660888314, 0.030464508103354708, 0.13706978363298242, 0.028970199506203023, 0.13721638515632006, 0.028325674662651124, 0.13721638515632006, 0.028325674662651124, 0.1370697836329824, 0.028970199506203012, 0.13674284660888308, 0.0304645081033547, 0.13615737742196524, 0.03329929560434659, 0.13516513933980914, 0.03844347521197908, 0.13436333428621713, 0.043206841474147425, 0.13407552941360296, 0.045508080862818004, 0.1339318310846739, 0.046632406314322555]

        # Solve the problem
        pydgm.solver.solve()

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
        
    def test_solver_eigenR7gPin(self):
        '''
        Test eigenvalue source problem with reflective conditions and 7g
        '''
        # Set the variables for the test
        pydgm.control.fine_mesh = [3, 10, 3]
        pydgm.control.material_map = [6, 1, 6]
        pydgm.control.angle_order = 2
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        s = 'eigen'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 0.0

        # Initialize the dependancies
        pydgm.solver.initialize_solver()
        
        # set the test flux
        keff_test = 0.7735620332498745
        phi_test = [0.13187762273035553, 3.0560002021766643, 1.7745035683301031, 0.6497468427915719, 0.14448308078823344, 0.019895928705131523, 8.31379552547061e-05, 0.13188288634468276, 3.056403228279507, 1.774758068899994, 0.6487626069789563, 0.1438285999231679, 0.019608234905321438, 7.806909778328154e-05, 0.13189341388973588, 3.057209396485762, 1.7752674363208387, 0.6467907571461935, 0.14251875998107638, 0.019032507623511552, 6.727347241536639e-05, 0.1319513233456591, 3.0597550377239013, 1.7762411002132281, 0.6431245521476735, 0.14007992707737454, 0.018155959093001727, 5.245693945483657e-05, 0.13204157473703287, 3.063370328686325, 1.7774026907915714, 0.6388212800497931, 0.1372214690255107, 0.017251311003616352, 3.986781991875587e-05, 0.13210777149287783, 3.066060181391546, 1.7782690984803549, 0.6356261228532062, 0.13510955968997732, 0.01659332772257151, 3.238107030308677e-05, 0.1321512038138374, 3.0678432530705977, 1.7788444419391838, 0.6335112726127647, 0.13371667329125442, 0.0161642530798633, 2.8181571408842145e-05, 0.13217271256078084, 3.0687317651283674, 1.7791314385277586, 0.6324583990047713, 0.13302471551183112, 0.015952553852704478, 2.6294217726345094e-05, 0.13217271256078084, 3.068731765128368, 1.7791314385277583, 0.6324583990047713, 0.13302471551183115, 0.015952553852704478, 2.62942177263451e-05, 0.13215120381383738, 3.0678432530705986, 1.7788444419391842, 0.6335112726127649, 0.1337166732912544, 0.016164253079863303, 2.818157140884216e-05, 0.1321077714928778, 3.0660601813915465, 1.7782690984803553, 0.6356261228532063, 0.13510955968997732, 0.016593327722571518, 3.238107030308679e-05, 0.13204157473703285, 3.063370328686326, 1.7774026907915719, 0.6388212800497933, 0.1372214690255107, 0.017251311003616356, 3.986781991875591e-05, 0.13195132334565907, 3.0597550377239027, 1.7762411002132288, 0.6431245521476738, 0.14007992707737454, 0.018155959093001734, 5.2456939454836604e-05, 0.13189341388973586, 3.0572093964857636, 1.775267436320839, 0.6467907571461937, 0.14251875998107638, 0.019032507623511562, 6.727347241536645e-05, 0.1318828863446827, 3.0564032282795073, 1.7747580688999944, 0.6487626069789565, 0.14382859992316793, 0.019608234905321452, 7.806909778328162e-05, 0.1318776227303555, 3.0560002021766643, 1.774503568330104, 0.6497468427915722, 0.14448308078823346, 0.01989592870513153, 8.313795525470619e-05]

        # Solve the problem
        pydgm.solver.solve()

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
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()

if __name__ == '__main__':

    unittest.main()

