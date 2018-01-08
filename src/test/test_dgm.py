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
        pydgm.control.xs_name = 'test.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.energy_group_map = [4]
        pydgm.control.use_dgm = True
        pydgm.control.dgm_basis_name = 'basis'.ljust(256)
        pydgm.control.Lambda = 1.0
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 7

    def test_dgm_test1(self):
        ''' 
        Test moment computation with phi == 1
        '''
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
        sig_s_m_test = [[0.35342781806, 0.04743636186124999, 0.028933133948542498, 0.020134451190550004, 0.014509247257650001, 0.009301495623800001, 0.005620442104, 0.0030043367622, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0015106138853238885, -0.0004682252948100947, -2.10589954062589e-05, -6.492736789739235e-07, 9.154696808378383e-07, 3.0289238497360745e-06, -2.1934200679323454e-06, 1.224533940189083e-06, 0.42369015333333326, 0.005308310899999998, 0.0016464802333333328, 0.001228323293333333, 0.0007583007633333331, -0.0006374102666666665, -7.760245200000006e-05, 0.00044780386333333324], [-0.12616159348403644, 0.03383513776305, 0.02883351588732215, 0.022366592333776532, 0.016881896022033414, 0.011362358747254313, 0.006986249848033285, 0.003924474790883601, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0018501166087033542, -0.0005734565284744781, -2.5791896620474617e-05, -7.951946084528616e-07, 1.1212167965206372e-06, 3.7096589507999246e-06, -2.6863799790075303e-06, 1.4997416630915137e-06, 0.07992011421022244, -0.00496940668392808, -0.0036350055052213155, -0.0026867446329954044, -0.001559092362501901, -0.00042067344965861773, -0.00032993401682169713, -0.0005203939039775896], [-0.04163098694, 0.004344929711249999, 0.0073627300485425, 0.00983078873555, 0.008627426390350003, 0.006743598208799998, 0.004427864874, 0.0027046093172, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0010681653220670792, -0.0003310852810832883, -1.4890958456742022e-05, -4.591058212483988e-07, 6.473348192911198e-07, 2.1417725938460415e-06, -1.5509822040256192e-06, 8.658762529007827e-07, -0.17726364202447836, 0.01515281812461261, 0.004629729488908694, 0.002862815278436165, 0.0013721899683078955, 0.0017345523325466316, 0.0006840553221513953, 0.0002677613009861893], [-0.004584591418129613, -0.00662701769819133, -0.005421313113252008, -0.0001475188965045822, 0.0012658069904094703, 0.0018727119427210672, 0.0014459651667724411, 0.0010194647035502366, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        nu_sig_f_m_test = [0.039245915, 1.504587272273979]
        chi_m_test = np.array([0.50000031545, 0.2595979010884317, -0.3848771845500001, -0.5216663031659152, 0.0, 0.0, 0.0, 0.0]).reshape((2, -1))

        for i in range(pydgm.dgm.expansion_order):
            pydgm.dgm.compute_xs_moments(i)
            np.testing.assert_array_almost_equal(pydgm.state.d_source.flatten('F'), source_m_test[:, :, i].flatten(), 12)
            np.testing.assert_array_almost_equal(pydgm.state.d_delta.flatten('F'), delta_m_test[:, :, i].flatten(), 12)
            np.testing.assert_array_almost_equal(pydgm.state.d_chi.flatten('F'), chi_m_test[:, i])
            np.testing.assert_array_almost_equal(pydgm.state.d_sig_s.flatten('F'), sig_s_m_test[i], 12)
            np.testing.assert_array_almost_equal(pydgm.state.d_sig_t.flatten('F'), sig_t_m_test)
            np.testing.assert_array_almost_equal(pydgm.state.d_nu_sig_f.flatten(), nu_sig_f_m_test)

    def test_dgm_test2(self):
        '''
        Test moment calculation with flux != 1
        '''
        # Set the variables for the test
        pydgm.control.energy_group_map = [1, 2, 3, 4, 5, 6]
        s = 'deltaBasis'
        pydgm.control.dgm_basis_name = s + ' ' * (256 - len(s))
        pydgm.control.Lambda = 0.1

        # Initialize the dependancies
        pydgm.mesh.create_mesh()
        pydgm.material.create_material()
        pydgm.angle.initialize_angle()
        pydgm.angle.initialize_polynomials(pydgm.material.number_legendre)
        pydgm.state.initialize_state()
        pydgm.dgm.initialize_moments()

        # Test the basic definitions
        np.testing.assert_array_equal(pydgm.dgm.order, [0, 0, 0, 0, 0, 0, 0, ])
        np.testing.assert_array_equal(pydgm.dgm.energymesh, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(pydgm.dgm.expansion_order, 0)
        self.assertEqual(pydgm.dgm.number_coarse_groups, 7)

        phi_m_test = [1.05033726e+00, -1.38777878e-17, -1.57267763e-01, 1.00000000e-08, 1.43689742e-17, 1.00000000e-08, 9.88540222e-02, 6.93889390e-18, 6.32597614e-02, -8.67361738e-19, -8.69286487e-03, 1.00000000e-08, 6.81181057e-19, 2.16840434e-19, 5.46408649e-03, 1.00000000e-08, 5.78458741e-03, 1.00000000e-08, -7.15385990e-04, 1.00000000e-08, 4.34869208e-20, 5.42101086e-20, 4.49671194e-04, -5.42101086e-20, 3.20772663e-05, -4.23516474e-22, -3.67904369e-06, -4.23516474e-22, 1.73044862e-22, 1.05879118e-22, 2.31254175e-06, 1.00000000e-08, 1.55685757e-07, -3.30872245e-24, -1.78930684e-08, 8.27180613e-25, 8.48610071e-25, 8.27180613e-25, 1.12470715e-08, 8.27180613e-25, 7.86105345e-10, 1.00000000e-08, -8.28643069e-11, 6.46234854e-27, 2.51523310e-27, 3.23117427e-27, 5.20861358e-11, 6.46234854e-27, 3.40490800e-13, -6.31088724e-30, -2.37767941e-14, -1.57772181e-30, -1.77547007e-30, 1.57772181e-30, 1.49454134e-14, 3.15544362e-30]
        psi_m_test = [0.66243843, 1.46237369, 0.73124892, 0.66469264, 0.60404633, 0.59856738, 0.31047097, 1.52226789, 3.07942675, 1.4095816 , 1.23433893, 1.11360585, 1.04656294, 0.44397067, 1.52226789, 3.07942675, 1.4095816 , 1.23433893, 1.11360585, 1.04656294, 0.44397067, 0.66243843, 1.46237369, 0.73124892, 0.66469264, 0.60404633, 0.59856738, 0.31047097]
        psi_m_test = np.reshape(psi_m_test, (7,4,1), 'F')

        pydgm.state.phi = np.reshape(phi_m_test, (8, 7, 1), 'F')
        for a in range(4):
            pydgm.state.psi[:,a,:] = psi_m_test[:,a,:]
            
        pydgm.dgm.initialize_basis()
        pydgm.dgm.compute_source_moments()
        
        # Test the basis
        np.testing.assert_array_almost_equal(pydgm.dgm.basis, np.ones((7,1)), 12, 'basis failure')
        
        pydgm.dgm.compute_flux_moments()
        
        # test scalar flux moments
        np.testing.assert_array_equal(pydgm.state.d_phi.flatten('F'), phi_m_test, 12, 'scalar flux failure')
        
        # test angular flux moments
        np.testing.assert_array_almost_equal(pydgm.state.d_psi, psi_m_test, 12, 'angular flux failure')
        
        # Get the cross section moments for order 0
        pydgm.dgm.compute_xs_moments(0)
        
        # test source moments
        np.testing.assert_array_almost_equal(pydgm.state.d_source.flatten('F'), np.ones(28), 12, 'source failure')
        
        # test total cross section moments
        np.testing.assert_array_almost_equal(pydgm.state.d_sig_t[:,0], pydgm.material.sig_t[:,0], 12, 'sig_t failure')
        
        # test angular cross section moments (delta)
        np.testing.assert_array_almost_equal(pydgm.state.d_delta.flatten('F'), np.zeros(28), 12, 'delta failure')
        
        # test scattering cross section moments
        np.testing.assert_array_almost_equal(pydgm.state.d_sig_s[:,:,:,0], pydgm.material.sig_s[:,:,:,0], 12, 'sig_s failure')
        
        # test fission cross section moments
        np.testing.assert_array_almost_equal(pydgm.state.d_nu_sig_f[:,0], pydgm.material.nu_sig_f[:,0] * pydgm.state.phi[0,:,0] / pydgm.state.d_phi[0,:,0], 12, 'nu_sig_f failure')
        
        # test chi moments
        np.testing.assert_array_almost_equal(pydgm.state.d_chi[:,0], pydgm.material.chi[:,0], 12, 'chi failure')

    def test_dgm_test3(self):
        '''
        Test the truncation for moments
        '''
        # Set the variables for the test
        pydgm.control.Lambda = 0.1
        pydgm.control.truncation_map = [2, 1]
        
        # Initialize the dependancies
        pydgm.mesh.create_mesh()
        pydgm.material.create_material()
        pydgm.angle.initialize_angle()
        pydgm.angle.initialize_polynomials(pydgm.material.number_legendre)
        pydgm.state.initialize_state()
        pydgm.dgm.initialize_moments()

        # Test the basic definitions
        np.testing.assert_array_equal(pydgm.dgm.order, [2, 1])
        np.testing.assert_array_equal(pydgm.dgm.energymesh, [1, 1, 1, 1, 2, 2, 2])
        self.assertEqual(pydgm.dgm.expansion_order, 2)
        self.assertEqual(pydgm.dgm.number_coarse_groups, 2)

    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()

if __name__ == '__main__':

    unittest.main()

