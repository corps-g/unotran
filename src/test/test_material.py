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
        pydgm.control.material_map = [6, 1, 6]
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
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.legendre_order = 1

        # Initialize the dependancies
        pydgm.material.create_material()

    def test_material_create_material(self):
        ''' 
        Test the material initialization
        '''

        # Test the number of materials
        self.assertEqual(pydgm.material.number_materials, 7)

        # Test the number of groups
        self.assertEqual(pydgm.material.number_groups, 7)

        # Test the number of legendre moments
        self.assertEqual(pydgm.material.number_legendre, 1)

        # Test the energy bounds
        ebounds_test = [1.0e+37, 4.0, 2.0e-01, 3.0e-03, 4.0e-05, 5.0e-07, 6.0e-09, 0.0]
        np.testing.assert_array_equal(pydgm.material.ebounds, ebounds_test)

        # Test the neutron velocity
        velocity_test = np.array([3.158699373946e+09, 1.265531232045e+09, 2.062621178994e+08, 2.214246016018e+07, 2.447057902284e+06, 4.107855863553e+05, 8.296620786354e+04])
        np.testing.assert_array_almost_equal(pydgm.material.velocity, velocity_test, 12)

        # Test the total cross section
        sig_t_test = [0.21623600, 0.30664800, 0.44514700, 0.53631500, 0.55601600, 0.71102300, 1.75422000, 0.22840500, 0.31768300, 0.46417600, 0.55616200, 0.58850800, 0.98602300, 2.18612000, 0.22818900, 0.31742800, 0.46480700, 0.56672900, 0.62346000, 1.25597000, 2.87089000, 0.22807300, 0.31782600, 0.46535500, 0.57498500, 0.64967100, 1.48074000, 3.44862000, 0.15358000, 0.25602200, 0.38234300, 0.28413800, 0.25509500, 0.26036200, 0.29965500, 0.10642700, 0.29292300, 0.83901200, 1.05242000, 1.08411000, 1.89404000, 5.59617000]
        np.testing.assert_array_equal(pydgm.material.sig_t[:, :-1].flatten('F'), sig_t_test)

        # Test the fission cross section times nu
        nu_sig_f_test = [0.04377570, 0.01259030, 0.00249733, 0.01962850, 0.05301660, 0.52135500, 2.03165000, 0.04867930, 0.01496700, 0.00294972, 0.02593640, 0.07556850, 1.02075000, 3.02363000, 0.05036610, 0.01643930, 0.00432797, 0.03766070, 0.10852500, 1.50444000, 4.39402000, 0.05165750, 0.01756960, 0.00544935, 0.04686690, 0.13377400, 1.90256000, 5.53922000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000]
        np.testing.assert_array_equal(pydgm.material.nu_sig_f[:, :-1].flatten('F'), nu_sig_f_test)

        # Test the fission cross section
        sig_f_test = [1.361112752124e-02, 4.671881435744e-03, 9.717615471419e-04, 7.594611012447e-03, 2.030750491249e-02, 1.942752059741e-01, 7.793565364830e-01, 1.502549555834e-02, 5.437147849589e-03, 1.034669996633e-03, 9.129866975497e-03, 2.654889175412e-02, 3.570916316543e-01, 1.059268860901, 1.544621772972e-02, 5.907063985138e-03, 1.508818351374e-03, 1.319072253414e-02, 3.796531783819e-02, 5.251539395970e-01, 1.532870753141, 1.576692752845e-02, 6.264431339271e-03, 1.895637079605e-03, 1.638571024009e-02, 4.672674560760e-02, 6.637061017174e-01, 1.929551020124, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        np.testing.assert_array_equal(pydgm.material.sig_f[:, :-1].flatten('F'), sig_f_test)

        # Test the chi spectrum
        chi_test = [0.11505700, 0.85043100, 0.03444650, 0.0000661309, 0.00000000, 0.00000000, 0.00000000, 0.11924300, 0.84636000, 0.03434360, 0.0000532432, 0.00000000, 0.00000000, 0.00000000, 0.12066700, 0.84533500, 0.03392080, 0.0000778717, 0.00000000, 0.00000000, 0.00000000, 0.12019500, 0.84612500, 0.03359890, 0.0000807813, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000]
        np.testing.assert_array_equal(pydgm.material.chi[:, :-1].flatten('F'), chi_test)

        # Test the isotropic scattering cross section
        sig_s_test = [1.404830e-01, 0, 0, 0, 0, 0, 0, 0.0545935, 0.290195, 0, 0, 0, 0, 0, 0.00446131, 0.0100118, 0.430856, 0, 0, 0, 0, 5.6797e-06, 4.31254e-06, 0.00431667, 0.478784, 0, 0, 0, 0, 0, 0, 0.00523292, 0.393849, 0.00237858, 0, 0, 0, 0, 0, 0.0063998, 0.374041, 0.293938, 0, 0, 0, 0, 0, 0.00110108, 0.199363]
        np.testing.assert_array_equal(pydgm.material.sig_s[:, :, :, 0].flatten('F'), sig_s_test)

    def tearDown(self):
        pydgm.material.finalize_material()


if __name__ == '__main__':

    unittest.main()

