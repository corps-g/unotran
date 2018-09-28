import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


class TestMATERIAL(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [3, 22, 3]
        pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [5, 1, 5]
        pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 10
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = False
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.scatter_legendre_order = 7
        pydgm.control.use_DGM = False

        # Initialize the dependancies
        pydgm.material.create_material()

    def test_material_create_material(self):
        ''' 
        Test the material initialization
        '''

        # Test the number of materials
        self.assertEqual(pydgm.material.number_materials, 6)

        # Test the number of groups
        self.assertEqual(pydgm.control.number_fine_groups, 7)
        self.assertEqual(pydgm.control.number_coarse_groups, 7)

        # Test the number of legendre moments
        self.assertEqual(pydgm.control.number_legendre, 7)

        # Test the energy bounds
        ebounds_test = [1.0e+37, 4.0, 2.0e-01, 3.0e-03, 4.0e-05, 5.0e-07, 6.0e-09, 0.0]
        np.testing.assert_array_equal(pydgm.material.ebounds, ebounds_test)

        # Test the neutron velocity
        velocity_test = np.array([3.143250498205e+09, 1.249230161913e+09, 2.045236541832e+08, 2.203220667972e+07, 2.395737503833e+06, 4.242969399705e+05, 8.141200989970e+04])
        np.testing.assert_array_almost_equal(pydgm.material.velocity, velocity_test, 12)

        # Test the total cross section
        sig_t_test = [0.179199, 0.315392, 0.693434, 0.802303, 0.820692, 1.34187, 4.86795, 0.178195, 0.314747, 0.693385, 0.80507, 0.855642, 1.51102, 5.04796, 0.177849, 0.314152, 0.693524, 0.809918, 0.872608, 1.61102, 5.3949, 0.177513, 0.313868, 0.693338, 0.812779, 0.882472, 1.66547, 5.58623, 0.19876, 0.23771, 0.328415, 0.280221, 0.26279, 0.352763, 0.995686]
        np.testing.assert_array_equal(pydgm.material.sig_t[:, :-1].flatten('F'), sig_t_test)

        # Test the fission cross section times nu
        nu_sig_f_test = [0.0143281, 0.00408432, 0.00141357, 0.00996843, 0.023784, 0.153426, 0.644569, 0.015354, 0.00471237, 0.0011944, 0.00995565, 0.0282717, 0.334958, 0.69482, 0.0161717, 0.00543855, 0.00181553, 0.0147668, 0.0408908, 0.462188, 0.859447, 0.0166499, 0.00589924, 0.00221469, 0.0177065, 0.0484992, 0.528432, 0.931411, 0.0245729, 0.00699095, 0.002469, 0.017622, 0.0423755, 0.280273, 1.42345]
        np.testing.assert_array_equal(pydgm.material.nu_sig_f[:, :-1].flatten('F'), nu_sig_f_test)

        # Test the fission cross section
        sig_f_test = [0.004489414168126, 0.00153966087894, 0.0005807147346756, 0.004094971470355, 0.00976130282037, 0.06296466532606, 0.2645253826897, 0.004706841688994, 0.001695432549605, 0.0004164037414978, 0.003486946257951, 0.009875127492211, 0.1168583150756, 0.2422790592256, 0.004916396602358, 0.001928639060389, 0.0006301035636444, 0.005152353604114, 0.01423838822788, 0.161025405187, 0.2987728525789, 0.005037958903322, 0.002076297941385, 0.000767404147681, 0.006169168269365, 0.01686611905938, 0.184013650451, 0.3234380425875, 0.007699410627505, 0.002635369467042, 0.001014300444912, 0.007239012286849, 0.01739152739928, 0.1150215455329, 0.5841712151681]
        np.testing.assert_array_equal(pydgm.material.sig_f[:, :-1].flatten('F'), sig_f_test)

        # Test the chi spectrum
        chi_test = [0.108982, 0.855844, 0.0350962, 7.71068e-05, 1.98271e-07, 0.0, 0.0, 0.119944, 0.845961, 0.0340205, 7.43642e-05, 9.88194e-08, 0.0, 0.0, 0.120273, 0.845615, 0.0340446, 6.72337e-05, 9.89285e-08, 0.0, 0.0, 0.120513, 0.8454, 0.0340214, 6.61128e-05, 3.98991e-07, 0.0, 0.0, 0.108982, 0.855844, 0.0350962, 7.71068e-05, 1.98271e-07, 0.0, 0.0]
        np.testing.assert_array_equal(pydgm.material.chi[:, :-1].flatten('F'), chi_test)

        # Test the isotropic scattering cross section
        sig_s_test = [0.0879396, 0.0606981, 0.0464856, 0.0342624, 0.0249764, 0.016218, 0.00928166, 0.00463684, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0750404, 0.0210804, 0.00562259, -0.00512356, -0.00605188, -0.0027002, -5.57741e-05, 3.05752e-05, 0.257556, 0.109193, 0.0623677, 0.0222987, 0.00551627, -0.000785712, -0.000217445, 0.000805376, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00355822, 0.00022365, -0.000806328, -0.000346468, 0.000518394, 0.000407935, -0.000317138, -0.000421643, 0.0541365, 0.0232098, -0.0037677, -0.0136007, -0.00631385, 0.00260433, 0.00337346, -0.000483995, 0.583345, 0.265722, 0.127938, 0.0284192, -0.0103672, -0.00810046, 0.00207444, 0.00357066, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.77986e-05, 1.11871e-06, -1.30906e-05, -1.77316e-06, 9.84462e-06, 2.17839e-06, -7.74945e-06, -2.39328e-06, 0.000740866, 5.10797e-05, -0.000363919, -7.55964e-05, 0.0002629, 9.27512e-05, -0.000206332, -0.000105234, 0.104898, 0.0464105, -0.0119437, -0.0281867, -0.00914859, 0.00796552, 0.00551195, -0.00336738, 0.655076, 0.338951, 0.163434, 0.0347732, -0.014664, -0.0103887, 0.00317343, 0.00465138, 4.09472e-08, 4.08948e-08, 4.07903e-08, 4.06338e-08, 4.04257e-08, 4.01666e-08, 3.9857e-08, 3.94977e-08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5065e-07, 7.90244e-08, -8.74569e-08, -9.81195e-08, 1.10188e-08, 8.36764e-08, 3.89832e-08, -4.77198e-08, 9.61778e-06, 8.4521e-07, -4.61924e-06, -1.18157e-06, 3.2012e-06, 1.36076e-06, -2.37655e-06, -1.42839e-06, 0.00137468, 8.81482e-05, -0.000675138, -0.000130248, 0.000485488, 0.00015849, -0.000378326, -0.000177931, 0.12866, 0.056289, -0.0152533, -0.0343655, -0.0108021, 0.0097357, 0.00671799, -0.00381995, 0.656076, 0.350086, 0.159275, 0.0306974, -0.0120426, -0.00668616, 0.00323594, 0.00275299, 0.00259756, 0.00147068, 0.000651605, 0.000139609, -7.90709e-05, -0.000146264, -0.000141253, -0.000100376, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.14102e-08, 4.05994e-08, -9.73696e-09, -2.35129e-08, -1.1876e-08, 4.92028e-10, 5.64279e-09, 4.92937e-09, 1.64171e-05, 1.52651e-06, -6.08385e-06, -1.37522e-06, 2.81771e-06, 9.09906e-07, -1.32883e-06, -5.07491e-07, 0.00156415, 8.72393e-05, -0.000596066, -9.67173e-05, 0.000287099, 7.86088e-05, -0.000135358, -5.36397e-05, 0.124816, 0.0341514, -0.0114264, -0.0118027, -0.00341444, 0.00061962, -0.00137936, -0.00167068, 1.21013, 0.335178, 0.0831993, 0.0253386, -0.00589629, 0.00508266, -0.0124162, -0.000132203, 3.07391, -0.0918581, -0.0831578, -0.0340589, -0.0397422, -0.0123302, -0.0511599, -0.00604631, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.56495e-08, 2.54202e-08, 1.08365e-08, -3.6197e-10, -3.65092e-09, -2.416e-10, 4.55678e-09, 5.60963e-09, 6.63537e-06, -2.27065e-08, -3.44519e-07, 5.91797e-08, -5.36598e-08, -1.3637e-09, 7.15057e-08, 4.65949e-08, 0.000296306, 2.23088e-07, -1.75888e-06, 3.55369e-07, -1.3927e-06, -7.30074e-07, -3.19381e-06, 6.3792e-07, 0.0119, -0.000375452, -0.000318901, -0.000122848, -0.000145178, -4.32521e-05, -0.000193124, -2.22286e-05, 1.24284, 0.273954, 0.100243, 0.0519744, 0.0199418, 0.0242043, -0.00441744, 0.0190722]
        np.testing.assert_array_equal(pydgm.material.sig_s[:, :, :, 0].flatten('F'), sig_s_test)

    def tearDown(self):
        pydgm.material.finalize_material()


if __name__ == '__main__':

    unittest.main()
