import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np

class TestDGMSOLVER(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        s = 'test/2gXS.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = False
        pydgm.control.energy_group_map = [1]
        s = '2gbasis'
        pydgm.control.dgm_basis_name = s + ' ' * (256 - len(s))
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 1.0
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = True
        s = 'fixed'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 1.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 0
        pydgm.control.ignore_warnings = True

    def test_dgmsolver_2gtest(self):
        '''
        Test the 2g dgm fixed source problem
        '''
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
        Test the 7g dgm fixed source problem with vacuum boundary conditions
        '''
        # Set the variables for the test
        pydgm.control.fine_mesh = [3, 22, 3]
        pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [6, 1, 6]
        s = 'test.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 10
        pydgm.control.allow_fission = True
        pydgm.control.energy_group_map = [4]
        s = 'basis'
        pydgm.control.dgm_basis_name = s + ' ' * (256 - len(s))
        pydgm.control.lamb = 0.75
        
        pydgm.dgmsolver.initialize_dgmsolver()

        phi_test = [1.8461980363287278, 4.16904423296633, 1.6371065827443676, 1.3260798327554102, 1.1507280372172657, 1.221454815786205, 0.21787346591975096, 1.8779676212462644, 4.346293616739397, 1.7911796347394968, 1.4696141054340566, 1.2762011622747407, 1.410548033592605, 0.2805431110268466, 1.9076720822947586, 4.5235324366607985, 1.9348875590099663, 1.5995201255730067, 1.3850154736028073, 1.5361742894430894, 0.3503620127705863, 1.9653381411150332, 4.755704786297735, 2.0602776849402016, 1.7086687870124164, 1.473314427190565, 1.6032397558933036, 0.41859495632314897, 2.046831950358654, 5.024852817609021, 2.1627268750309767, 1.794379906316901, 1.5415868503767243, 1.6402668270670566, 0.45783584286439, 2.1183287442900616, 5.259894345939496, 2.249750972825359, 1.8662129456617995, 1.598567531294177, 1.6740654516371898, 0.48316274993701736, 2.180431853934209, 5.463189536952539, 2.323379278194582, 1.9263940668089787, 1.6461554034562276, 1.7042440374045473, 0.5017303824642151, 2.233671544963958, 5.636838362725401, 2.3851794675942126, 1.976556103614348, 1.6857277137465003, 1.7306376835631267, 0.5160940364316613, 2.278501420880871, 5.782620523285358, 2.43635508075605, 2.0178957585753343, 1.7182838252262809, 1.753193622571626, 0.5273657105739336, 2.3153010315366376, 5.902002999445916, 2.4778225682576234, 2.051286995516379, 1.7445473309392545, 1.7719149818882431, 0.5361455617094784, 2.344379734887513, 5.996163165441185, 2.5102702850685095, 2.0773624527330834, 1.7650382804142186, 1.7868302159372544, 0.5428111210518882, 2.365980375010593, 6.066012532308792, 2.534202681742982, 2.096571354470798, 1.7801238463024747, 1.7979755323610467, 0.5476165734363021, 2.380282353001199, 6.112216861028091, 2.54997251963054, 2.1092199413577197, 1.790053258345466, 1.8053845815737255, 0.5507325908285529, 2.3874039642095606, 6.135211472153911, 2.557803276430926, 2.115498615897139, 1.7949810125894576, 1.809082468016729, 0.5522651544630954, 2.3874039642095597, 6.1352114721539115, 2.5578032764309264, 2.1154986158971383, 1.7949810125894576, 1.809082468016729, 0.5522651544630954, 2.380282353001199, 6.112216861028088, 2.5499725196305403, 2.1092199413577197, 1.790053258345467, 1.8053845815737257, 0.5507325908285529, 2.365980375010593, 6.0660125323087914, 2.534202681742981, 2.096571354470798, 1.7801238463024753, 1.797975532361047, 0.547616573436302, 2.3443797348875135, 5.996163165441185, 2.5102702850685086, 2.0773624527330825, 1.7650382804142186, 1.7868302159372544, 0.5428111210518881, 2.315301031536638, 5.9020029994459176, 2.477822568257624, 2.0512869955163793, 1.7445473309392545, 1.7719149818882434, 0.5361455617094785, 2.278501420880872, 5.782620523285358, 2.436355080756051, 2.0178957585753348, 1.718283825226282, 1.753193622571626, 0.5273657105739337, 2.233671544963958, 5.636838362725398, 2.3851794675942113, 1.976556103614348, 1.6857277137465008, 1.7306376835631265, 0.516094036431661, 2.1804318539342096, 5.463189536952541, 2.323379278194582, 1.9263940668089785, 1.646155403456228, 1.704244037404548, 0.5017303824642152, 2.1183287442900616, 5.259894345939497, 2.2497509728253595, 1.8662129456617995, 1.5985675312941772, 1.6740654516371902, 0.48316274993701747, 2.046831950358654, 5.024852817609021, 2.1627268750309767, 1.7943799063169015, 1.5415868503767245, 1.640266827067057, 0.45783584286439005, 1.9653381411150337, 4.755704786297734, 2.060277684940202, 1.708668787012417, 1.4733144271905656, 1.6032397558933043, 0.4185949563231489, 1.9076720822947586, 4.5235324366608, 1.9348875590099668, 1.5995201255730072, 1.385015473602808, 1.53617428944309, 0.35036201277058626, 1.8779676212462644, 4.346293616739396, 1.791179634739497, 1.469614105434057, 1.2762011622747407, 1.4105480335926057, 0.2805431110268464, 1.8461980363287276, 4.169044232966329, 1.6371065827443676, 1.3260798327554109, 1.1507280372172666, 1.2214548157862055, 0.21787346591975099]
        pydgm.state.phi[0,:,:] = np.reshape(phi_test, (7, 28), 'F')

        pydgm.dgmsolver.dgmsolve()

        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

    def test_dgmsolver_reflect1(self):
        pydgm.control.fine_mesh = [3, 22, 3]
        pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [1, 1, 1]
        s = 'test.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 10
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = False
        pydgm.control.energy_group_map = [4]
        s = 'basis'
        pydgm.control.dgm_basis_name = s + ' ' * (256 - len(s))
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 0.3
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = True
        s = 'fixed'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 1.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 0
        pydgm.control.ignore_warnings = True
        pydgm.control.max_inner_iters = 10
        pydgm.control.max_outer_iters = 5000
        
        pydgm.dgmsolver.initialize_dgmsolver()
        
        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:,0])
        S = pydgm.material.sig_s[0,:,:,0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))
        phi_test = np.array([phi_test for i in range(28)]).flatten()
        
        pydgm.dgmsolver.dgmsolve()
        
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :].flatten('F') / phi_test, np.ones(28 * 7), 12)

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()

if __name__ == '__main__':

    unittest.main()

