import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np

class TestDGM(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        s = 'test.anlxs'
        pydgm.control.xs_name = s + ' ' * (256 - len(s))
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = False
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.energy_group_map = [4]
        s = 'basis'
        pydgm.control.dgm_basis_name = s + ' ' * (256 - len(s))
        pydgm.control.Lambda = 1.0
        pydgm.control.store_psi = True
        s = 'fixed'
        pydgm.control.solver_type = s + ' ' * (256 - len(s))
        pydgm.control.source_value = 1.0
        pydgm.control.equation_type = 'DD'
        
    def test_dgm_test1(self):
        ''' 
        Test moment computation
        '''
        # Activate fissioning
        pydgm.control.allow_fission = True
        
        # Initialize the dependancies
        pydgm.mesh.create_mesh()
        pydgm.material.create_material()
        pydgm.angle.initialize_angle()
        pydgm.angle.initialize_polynomials(pydgm.material.number_legendre)
        pydgm.state.initialize_state()
        pydgm.dgm.initialize_moments()
        
        # Test the basic definitions
        np.testing.assert_array_equal(pydgm.dgm.order, [3, 2])
        np.testing.assert_array_equal(pydgm.dgm.energymesh, [1, 1, 1, 1, 2, 2, 2])
        self.assertEqual(pydgm.dgm.expansion_order, 3)
        self.assertEqual(pydgm.dgm.number_coarse_groups, 2)
        
        # Test the basis
        pydgm.dgm.initialize_basis()
        basis_test = [0.5, 0.5, 0.5, 0.5, 0.5773502691896258421, 0.5773502691896258421, 0.5773502691896258421, 0.6708203932499369193, 0.2236067977499789639, -0.2236067977499789639, -0.6708203932499369193, 0.7071067811865474617, 0.0, -0.7071067811865474617, 0.5, -0.5, -0.5, 0.5, 0.4082482904638630727, -0.8164965809277261455, 0.4082482904638630727, 0.2236067977499789361, -0.6708203932499369193, 0.6708203932499369193, -0.2236067977499789361, 0.0, 0.0, 0.0]
        np.testing.assert_array_almost_equal(pydgm.dgm.basis.flatten('F'), basis_test, 12)
        
        # Test the moments
        pydgm.dgm.compute_source_moments()
        
        pydgm.dgm.compute_flux_moments()
        
        phi_m_test = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776]
        np.testing.assert_array_almost_equal(pydgm.state.d_phi.flatten('F'), phi_m_test, 12)

        psi_m_test = [2.0, 1.7320508075688776, 2.0, 1.7320508075688776, 2.0, 1.7320508075688776, 2.0, 1.7320508075688776]
        np.testing.assert_array_almost_equal(pydgm.state.d_psi.flatten('F'), psi_m_test, 12)

        source_m_test = np.array([2.0, 0.0, 0.0, 0.0, 1.7320508075688776, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.7320508075688776, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.7320508075688776, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.7320508075688776, 0.0, 0.0, 0.0]).reshape((4, 2, -1))
        sig_t_m_test = [0.3760865, 1.0070863333333333]
        delta_m_test = np.array([0.0, -0.12284241926631045, 0.00018900000000000167, 1.0668056713853770e-02, 0.0, -4.8916473462696236e-01, 2.0934839066069319e-01, 0.0, 0.0, -1.2284241926631045e-01, 1.8900000000000167e-04, 1.0668056713853770e-02, 0.0, -4.8916473462696236e-01, 2.0934839066069319e-01, 0.0, 0.0, -1.2284241926631045e-01, 1.8900000000000167e-04, 1.0668056713853770e-02, 0.0, -4.8916473462696236e-01, 2.0934839066069319e-01, 0.0, 0.0, -1.2284241926631045e-01, 1.8900000000000167e-04, 1.0668056713853770e-02, 0.0, -4.8916473462696236e-01, 2.0934839066069319e-01, 0.0]).reshape((4, 2, -1))
        sig_s_m_test = np.array([0.35342781806, 0.04743636186125, 0.0289331339485425, 0.02013445119055, 0.01450924725765, 0.0093014956238, 0.005620442104, 0.0030043367622, -0.1087492753068697, 0.032934103959394, 0.0289449758029991, 0.0224441910315558, 0.0167896813253091, 0.0112993763443412, 0.0069750093143139, 0.0039106927748581, -0.01426407321, 0.00452756650875, 0.0076277391555075, 0.00999316946045, 0.00843282744865, 0.0066059506217, 0.0043978711625, 0.0026803333838, 0.0140474443693441, -0.0065186049755285, -0.0052873421588672, -0.0000093035117265, 0.0010863232979017, 0.0017467451521609, 0.0014185324207123, 0.0010077531638459, 0.0015106138853239, -0.0004682252948101, -0.0000210589954063, -0.000000649273679, 0.0000009154696808, 0.0000030289238497, -0.0000021934200679, 0.0000012245339402, -0.0020267012012036, 0.0006281901527882, 0.0000282536071598, 0.0000008710920493, -0.0000012282314626, -0.000004063727776, 0.0000029427818251, -0.0000016428846786, 0.0015106138853239, -0.0004682252948101, -0.0000210589954063, -0.000000649273679, 0.0000009154696808, 0.0000030289238497, -0.0000021934200679, 0.0000012245339402, -0.0006755670670679, 0.0002093967175961, 0.0000094178690533, 0.0000002903640164, -0.0000004094104875, -0.0000013545759253, 0.000000980927275, -0.0000005476282262, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4236901533333333, 0.0053083109, 0.0016464802333333, 0.0012283232933333, 0.0007583007633333, -0.0006374102666667, -0.000077602452, 0.0004478038633333, -0.0379884015739014, 0.0038739129355235, -0.0015822948479042, -0.0014793513753941, -0.0010802063604453, 0.0007826396112285, 0.0001042419486558, -0.0005035107714306, 0.0326467618199471, -0.0018823973060567, 0.0009853706472698, 0.0007925234371622, 0.0005038871725552, -0.0003595438342519, -0.0000560788795342, 0.0002638059860685, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((4, 4, -1))
        nu_sig_f_m_test = [0.039245915, 1.504587272273979]
        chi_m_test = np.array([0.50000031545, 0.2595979010884317, -0.3848771845500001, -0.5216663031659152, 0.0, 0.0, 0.0, 0.0]).reshape((2,-1))
        
        for i in range(pydgm.dgm.expansion_order):
            pydgm.dgm.compute_xs_moments(i)
            np.testing.assert_array_almost_equal(pydgm.state.d_source.flatten('F'), source_m_test[:, :, i].flatten(), 12)
            np.testing.assert_array_almost_equal(pydgm.state.d_delta.flatten('F'), delta_m_test[:, :, i].flatten(), 12)
            np.testing.assert_array_almost_equal(pydgm.state.d_chi.flatten('F'), chi_m_test[:,i])
            #np.testing.assert_array_almost_equal(pydgm.state.d_sig_s.flatten(), sig_s_m_test[:,i].flatten('F'), 12)
            np.testing.assert_array_almost_equal(pydgm.state.d_sig_t.flatten('F'), sig_t_m_test)
            np.testing.assert_array_almost_equal(pydgm.state.d_nu_sig_f.flatten(), nu_sig_f_m_test)

    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()

if __name__ == '__main__':

    unittest.main()

