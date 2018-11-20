import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np
np.set_printoptions(precision=16, linewidth=132)


class TestDGM(unittest.TestCase):
    '''
    Test the DGM routines when the fluxes are initialized to unity
    '''

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/7gXSaniso.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.energy_group_map = [1, 1, 1, 1, 2, 2, 2]
        pydgm.control.use_dgm = True
        pydgm.control.dgm_basis_name = 'test/7gbasis'.ljust(256)
        pydgm.control.Lambda = 1.0
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 2.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.scatter_legendre_order = 7

    def test_dgm_initialize_moments(self):
        '''
        Test if the variables are being initialized properly
        '''

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Test the basic definitions
        np.testing.assert_array_equal(pydgm.dgm.order, [3, 2])
        self.assertEqual(pydgm.dgm.expansion_order, 3)
        self.assertEqual(pydgm.control.number_coarse_groups, 2)

        nG = 2
        nA = 2 * 2
        nL = 8
        nC = 1

        # Check that arrays were properly resized
        assert(pydgm.state.mg_phi.shape == (nL, nG, nC))
        assert(pydgm.state.mg_source.shape == (nG, nC))
        assert(pydgm.state.mg_nu_sig_f.shape == (nG, nC))
        assert(pydgm.state.mg_sig_t.shape == (nG, nC))
        assert(pydgm.state.mg_chi.shape == (nG, nC))
        assert(pydgm.state.mg_sig_s.shape == (nL, nG, nG, nC))
        assert(pydgm.state.mg_psi.shape == (nG, nA, nC))
        assert(pydgm.state.mg_incident.shape == (nG, nA / 2))

    def test_dgm_test3(self):
        '''
        Test the truncation for moments
        '''

        # Set the variables for the test
        pydgm.control.Lambda = 0.1
        pydgm.control.truncation_map = [2, 1]

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Test the basic definitions
        np.testing.assert_array_equal(pydgm.dgm.order, [2, 1])
        self.assertEqual(pydgm.dgm.expansion_order, 2)
        self.assertEqual(pydgm.control.number_coarse_groups, 2)

    def test_dgm_initialize_basis(self):
        '''
        Test that the energy basis is properly initialized
        '''

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        np.set_printoptions(linewidth=132)

        assert(pydgm.dgm.basis.shape == (7, 4))
        basis_test = np.array([[0.5, 0.6708203932499369, 0.5, 0.2236067977499789],
                               [0.5, 0.223606797749979, -0.5, -0.6708203932499369],
                               [0.5, -0.223606797749979, -0.5, 0.6708203932499369],
                               [0.5, -0.6708203932499369, 0.5, -0.2236067977499789],
                               [0.5773502691896258, 0.7071067811865475, 0.4082482904638631, 0.],
                               [0.5773502691896258, 0., -0.8164965809277261, 0.],
                               [0.5773502691896258, -0.7071067811865475, 0.4082482904638631, 0.]])
        np.testing.assert_array_almost_equal(pydgm.dgm.basis, basis_test, 12)

    def test_dgm_finalize_moments(self):
        '''
        No good way to test this
        '''

    def test_dgm_compute_flux_moments(self):
        ''' 
        Check that the flux moments are properly computed
        '''

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Compute the flux moments using phi/psi = 1.0
        pydgm.dgmsolver.compute_flux_moments()

        phi_m_test = np.array([[[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                                [1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776, 1.7320508075688776]]]).T
        np.testing.assert_array_almost_equal(pydgm.dgm.phi_m_zero, phi_m_test, 12)

        psi_m_test = 0.5 * np.array([[[2.0],
                                      [2.0],
                                      [2.0],
                                      [2.0]],
                                     [[1.7320508075688776],
                                      [1.7320508075688776],
                                      [1.7320508075688776],
                                      [1.7320508075688776]]])

        np.testing.assert_array_almost_equal(pydgm.dgm.psi_m_zero, psi_m_test, 12)

    def test_dgm_compute_incoming_flux(self):
        '''
        Check that the higher order angular flux moments are computed correctly
        '''

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        phi = np.array([0.198933535568562, 2.7231683533646702, 1.3986600409998782, 1.010361903429942, 0.8149441787223116, 0.8510697418684054, 0.00286224604623])
        for a in range(4):
            pydgm.state.psi[:, a, 0] = phi / 2
        basis = np.loadtxt('test/7gbasis').T
        test = basis.dot(phi) * 0.5
        test.resize(2, 4)

        for i in range(4):
            pydgm.dgmsolver.compute_incoming_flux(i, pydgm.state.psi)
            for a in range(2):
                np.testing.assert_array_almost_equal(pydgm.state.mg_incident[:, a], test[:, i], 12)

    def test_dgm_compute_xs_moments(self):
        '''
        Check that the XS moments are properly computed
        '''

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        sig_t_m_test = np.array([[0.3760865],
                                 [1.0070863333333333]])
        delta_m_test = np.array([[[[0.0, -0.12284241926631045, 0.00018900000000000167, 1.0668056713853770e-02]],
                                  [[0.0, -0.12284241926631045, 0.00018900000000000167, 1.0668056713853770e-02]],
                                  [[0.0, -0.12284241926631045, 0.00018900000000000167, 1.0668056713853770e-02]],
                                  [[0.0, -0.12284241926631045, 0.00018900000000000167, 1.0668056713853770e-02]]],
                                 [[[0.0, -4.8916473462696236e-01, 2.0934839066069319e-01, 0.0]],
                                  [[0.0, -4.8916473462696236e-01, 2.0934839066069319e-01, 0.0]],
                                  [[0.0, -4.8916473462696236e-01, 2.0934839066069319e-01, 0.0]],
                                  [[0.0, -4.8916473462696236e-01, 2.0934839066069319e-01, 0.0]]]])

        sig_s_m_test = np.array([[[[[0.35342781806, -0.12616159348403644, -0.04163098694, -0.004584591418129613]],
                                   [[0.0015106138853238885, 0.0018501166087033542, 0.0010681653220670792, 0.0]]],
                                  [[[0.0, 0.0, 0.0, 0.0]],
                                   [[0.42369015333333326, 0.07992011421022244, -0.17726364202447836, 0.0]]]],
                                 [[[[0.04743636186124999, 0.03383513776305, 0.004344929711249999, -0.00662701769819133]],
                                   [[-0.0004682252948100947, -0.0005734565284744781, -0.0003310852810832883, 0.0]]],
                                  [[[0.0, 0.0, 0.0, 0.0]],
                                   [[0.005308310899999998, -0.00496940668392808, 0.01515281812461261, 0.0]]]],
                                 [[[[0.028933133948542498, 0.02883351588732215, 0.0073627300485425, -0.005421313113252008]],
                                   [[-2.10589954062589e-05, -2.5791896620474617e-05, -1.4890958456742022e-05, 0.0]]],
                                  [[[0.0, 0.0, 0.0, 0.0]],
                                   [[0.0016464802333333328, -0.0036350055052213155, 0.004629729488908694, 0.0]]]],
                                 [[[[0.020134451190550004, 0.022366592333776532, 0.00983078873555, -0.0001475188965045822]],
                                   [[-6.492736789739235e-07, -7.951946084528616e-07, -4.591058212483988e-07, 0.0]]],
                                  [[[0.0, 0.0, 0.0, 0.0]],
                                   [[0.001228323293333333, -0.0026867446329954044, 0.002862815278436165, 0.0]]]],
                                 [[[[0.014509247257650001, 0.016881896022033414, 0.008627426390350003, 0.0012658069904094703]],
                                   [[9.154696808378383e-07, 1.1212167965206372e-06, 6.473348192911198e-07, 0.0]]],
                                  [[[0.0, 0.0, 0.0, 0.0]],
                                   [[0.0007583007633333331, -0.001559092362501901, 0.0013721899683078955, 0.0]]]],
                                 [[[[0.009301495623800001, 0.011362358747254313, 0.006743598208799998, 0.0018727119427210672]],
                                   [[3.0289238497360745e-06, 3.7096589507999246e-06, 2.1417725938460415e-06, 0.0]]],
                                  [[[0.0, 0.0, 0.0, 0.0]],
                                   [[-0.0006374102666666665, -0.00042067344965861773, 0.0017345523325466316, 0.0]]]],
                                 [[[[0.005620442104, 0.006986249848033285, 0.004427864874, 0.0014459651667724411]],
                                   [[-2.1934200679323454e-06, -2.6863799790075303e-06, -1.5509822040256192e-06, 0.0]]],
                                  [[[0.0, 0.0, 0.0, 0.0]],
                                   [[-7.760245200000006e-05, -0.00032993401682169713, 0.0006840553221513953, 0.0]]]],
                                 [[[[0.0030043367622, 0.003924474790883601, 0.0027046093172, 0.0010194647035502366]],
                                   [[1.224533940189083e-06, 1.4997416630915137e-06, 8.658762529007827e-07, 0.0]]],
                                  [[[0.0, 0.0, 0.0, 0.0]],
                                   [[0.00044780386333333324, -0.0005203939039775896, 0.0002677613009861893, 0.0]]]]])

        nu_sig_f_m_test = np.array([[0.039245915],
                                    [1.504587272273979]])

        pydgm.dgmsolver.compute_flux_moments()

        pydgm.dgmsolver.compute_xs_moments()
        np.set_printoptions(precision=5, linewidth=132, suppress=True)

        for i in range(pydgm.dgm.expansion_order + 1):
            pydgm.dgmsolver.slice_xs_moments(i)
            np.testing.assert_array_almost_equal(pydgm.dgm.delta_m, delta_m_test, 12)
            np.testing.assert_array_almost_equal(pydgm.dgm.sig_s_m[i], sig_s_m_test[i], 12)
            np.testing.assert_array_almost_equal(pydgm.state.mg_sig_t, sig_t_m_test)
            np.testing.assert_array_almost_equal(pydgm.state.mg_nu_sig_f, nu_sig_f_m_test)

    def test_dgm_compute_source_moments(self):
        '''
        Check that the source moments are properly computed
        '''

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        source_m_test = np.array([[2.0, 0.0, 0.0, 0.0],
                                  [1.7320508075688776, 0.0, 0.0, 0.0]])
        chi_m_test = np.array([[[0.50000031545, 0.2595979010884317, -0.3848771845500001, -0.5216663031659152]],
                               [[0.0, 0.0, 0.0, 0.0]]])

        pydgm.dgmsolver.compute_flux_moments()

        for i in range(pydgm.dgm.expansion_order + 1):
            np.testing.assert_array_almost_equal(pydgm.dgm.source_m[:, i], source_m_test[:, i], 12)
            np.testing.assert_array_almost_equal(pydgm.dgm.chi_m[:, :, i], chi_m_test[:, :, i])

    def tearDown(self):
        '''
        Deallocate the arrays to prep for the next initialization
        '''
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()


class TestDGM2(unittest.TestCase):
    '''
    Test the DGM routines when the fluxes are initialized to a reasonable answer
    '''

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/7gXSaniso.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.energy_group_map = [1, 1, 1, 1, 2, 2, 2]
        pydgm.control.use_dgm = True
        pydgm.control.dgm_basis_name = 'test/7gbasis'.ljust(256)
        pydgm.control.Lambda = 1.0
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 2.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.scatter_legendre_order = 0

        # Initialize the dependancies
        pydgm.dgmsolver.initialize_dgmsolver()

        # Set a value for the flux
        phi = np.array([0.021377987105421, 0.7984597778757521, 0.5999743700269914,
                        0.0450954611897237, 0.0014555781016859, 0.0000276607249577, 0.000000019588085])

        pydgm.state.phi = np.reshape(phi, (1, 7, 1), 'F')
        for a in range(4):
            pydgm.state.psi[:, a, 0] = phi * 0.5

    def test_dgm_compute_flux_moments(self):
        ''' 
        Check that the flux moments are properly computed
        '''
        # Compute the flux moments using phi/psi = 1.0
        pydgm.dgmsolver.compute_flux_moments()

        phi_m_test = np.array([0.7324537980989441, 0.0008563596450213])

        np.testing.assert_array_almost_equal(pydgm.dgm.phi_m_zero.flatten(), phi_m_test, 12)

        psi_m_test = 0.5 * phi_m_test
        for a in range(4):
            np.testing.assert_array_almost_equal(pydgm.dgm.psi_m_zero[:, a, 0].flatten(), psi_m_test, 12)

    def test_dgm_compute_xs_moments(self):
        '''
        Check that the XS moments are properly computed
        '''
        sig_t_m_test = [0.3691229050566669, 0.558922490516065]
        delta_m_test = np.array([[0., -0.0390520749300116, 0.0058313443864203, 0.0841610173711961],
                                 [0., -0.003512611561523, -0.0060170505928347, 0.]])
        sig_s_m_test = np.array([[[0.3595146694424192, -0.0296917909741192, -0.322396034386523, 0.024412457624233],
                                  [0., 0., 0., 0.]],
                                 [[0.0001860099026453, 0.0002278146742929, 0.0001315288635283, 0.],
                                  [0.3998261803750482, 0.4733886921690485, 0.2545919465490534, 0.]]])
        nu_sig_f_m_test = [0.018256680188824, 0.1070002024403058]

        pydgm.dgmsolver.compute_flux_moments()
        pydgm.dgmsolver.compute_xs_moments()

        for i in range(pydgm.dgm.expansion_order):
            pydgm.dgmsolver.slice_xs_moments(i)
            for a in range(4):
                np.testing.assert_array_almost_equal(pydgm.dgm.delta_m[:, a, 0, i].flatten(), delta_m_test[:, i].flatten(), 12)
            np.testing.assert_array_almost_equal(pydgm.dgm.sig_s_m[:, :, :, :, i].flatten('F'), sig_s_m_test[:, :, i].flatten(), 12)
            np.testing.assert_array_almost_equal(pydgm.state.mg_sig_t.flatten(), sig_t_m_test)
            np.testing.assert_array_almost_equal(pydgm.state.mg_nu_sig_f.flatten(), nu_sig_f_m_test)

    def test_dgm_compute_source_moments(self):
        '''
        Check that the source/chi moments are computed correctly
        '''
        source_m_test = np.array([[2.0, 0.0, 0.0, 0.0],
                                  [1.7320508075688776, 0.0, 0.0, 0.0]])
        chi_m_test = np.array([[[0.50000031545, 0.2595979010884317, -0.3848771845500001, -0.5216663031659152]],
                               [[0.0, 0.0, 0.0, 0.0]]])

        pydgm.dgmsolver.compute_flux_moments()

        for i in range(pydgm.dgm.expansion_order + 1):
            np.testing.assert_array_almost_equal(pydgm.dgm.source_m[:, i], source_m_test[:, i], 12)
            np.testing.assert_array_almost_equal(pydgm.dgm.chi_m[:, :, i], chi_m_test[:, :, i])

    def tearDown(self):
        '''
        Deallocate the arrays to prep for the next initialization
        '''
        pydgm.state.finalize_state()
        pydgm.control.finalize_control()


if __name__ == '__main__':
    unittest.main()
