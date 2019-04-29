import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np
from scipy.special import sph_harm


class TestANGLE_1D(unittest.TestCase):

    def setUp(self):
        pydgm.control.spatial_dimension = 1

    def test_angle_legendre_p(self):
        ''' 
        Test the function used to create the legendre polynomials
        '''

        angle = pydgm.angle

        self.assertEqual(angle.legendre_p(0, 0.5), 1)
        self.assertEqual(angle.legendre_p(1, 0.5), 0.5)
        self.assertEqual(angle.legendre_p(2, 0.5), -0.125)

    def test_angle_d_legendre_p(self):
        ''' 
        Test the function used to create the double legendre polynomials
        '''

        angle = pydgm.angle

        self.assertEqual(angle.d_legendre_p(0, 0.5), 0.0)
        self.assertEqual(angle.d_legendre_p(1, 0.5), 1.0)
        self.assertEqual(angle.d_legendre_p(2, 0.5), 1.5)

    def test_angle_initialize_polynomials(self):
        '''
        Test the the legendre polynomial basis is correct
        '''
        nAngle = 8
        pydgm.control.angle_order = nAngle
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.number_legendre = 7

        pydgm.angle.initialize_angle()
        pydgm.angle.initialize_polynomials()

        # Get the basis set
        basis = pydgm.angle.p_leg

        # Use numpy to get the test basis
        x, wt = np.polynomial.legendre.leggauss(nAngle * 2)
        I = np.eye(nAngle * 2)
        test_basis = np.array([np.polynomial.legendre.legval(x, I[i]) * (-1) ** i for i in range(nAngle)])

        # Test the basis
        np.testing.assert_array_almost_equal(basis, test_basis, 12)

    def test_angle_quadrature(self):
        '''
        Test for correct gauss legendre points and weights
        '''
        nAngle = 8
        # setup control parameters for quadrature
        pydgm.control.angle_order = nAngle
        pydgm.control.angle_option = pydgm.angle.gl

        # build quadrature, test it, and deallocate
        pydgm.angle.initialize_angle()

        # Get the test quadrature weights from numpy
        mu_test, wt_test = np.polynomial.legendre.leggauss(nAngle * 2)
        mu_test = mu_test[:nAngle - 1:-1]
        wt_test = wt_test[:nAngle - 1:-1]

        # Test for equivalance
        np.testing.assert_array_almost_equal(pydgm.angle.mu, mu_test, 12)
        np.testing.assert_array_almost_equal(pydgm.angle.wt, wt_test, 12)

    def tearDown(self):
        pydgm.angle.finalize_angle()
        pydgm.control.finalize_control()


class TestANGLE_2D(unittest.TestCase):

    def setUp(self):
        pydgm.control.spatial_dimension = 2
        pydgm.control.angle_order = 10

    def test_angle_quadrature(self):
        '''
        Test for correct gauss legendre points and weights
        '''
        # setup control parameters for quadrature
        pydgm.control.angle_option = pydgm.angle.gl

        # build quadrature, test it, and deallocate
        pydgm.angle.initialize_angle()

        mu = [0.189321326478, 0.189321326478, 0.189321326478, 0.189321326478, 0.189321326478, 0.508881755583, 0.508881755583, 0.508881755583, 0.508881755583, 0.694318887594, 0.694318887594, 0.694318887594, 0.839759962237, 0.839759962237, 0.96349098111]
        eta = [0.189321326478, 0.508881755583, 0.694318887594, 0.839759962237, 0.96349098111 , 0.189321326478, 0.508881755583, 0.694318887594, 0.839759962237, 0.189321326478, 0.508881755583, 0.694318887594, 0.189321326478, 0.508881755583, 0.189321326478]
        wt = [0.089303147984, 0.072529151712, 0.045043767436, 0.072529151712, 0.089303147984, 0.072529151712, 0.053928114488, 0.053928114488, 0.072529151712, 0.045043767436, 0.053928114488, 0.045043767436, 0.072529151712, 0.072529151712, 0.089303147984]

        # Test for equivalance
        np.testing.assert_array_almost_equal(pydgm.angle.mu, mu, 12)
        np.testing.assert_array_almost_equal(pydgm.angle.eta, eta, 12)
        np.testing.assert_array_almost_equal(pydgm.angle.wt, wt, 12)

    def test_spherical_harmonics(self):

        # build quadrature, test it, and deallocate
        pydgm.control.angle_order = 16
        pydgm.angle.initialize_angle()
        pydgm.angle.initialize_polynomials()

        basis = pydgm.angle.p_leg

        print(basis.T)

        np.set_printoptions(precision=3, suppress=True, linewidth=132, threshold=10000)

        D = basis[:9] @ basis.T[:,:9]
        
        print(pydgm.control.number_angles)
        print(D)
        print(sum(pydgm.angle.wt))
        
#         mu = pydgm.angle.mu
#         eta = pydgm.angle.eta
#         xi = np.sqrt(1 - mu ** 2 - eta ** 2)
#
#         theta = np.arccos(xi)
#         phi = np.arccos(mu / np.sqrt(1 - xi ** 2))
#
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D
#
#         x = []
#         y = []
#         ref = []
#
#         l = 3
#         m = 0
#
#         for o in range(4):
#             for a in range(pydgm.control.number_angles):
#                 x.append(pydgm.angle.mu[a] * (1 if o < 2 else -1))
#                 y.append(pydgm.angle.eta[a] * (1 if o in [0, 3] else -1))
#
#                 xi = np.sqrt(1 - x[-1] ** 2 - y[-1] ** 2)
#                 theta = np.arccos(xi) * (1 if o in [0, 3] else -1)
#                 phi = np.arccos(x[-1] / np.sqrt(1 - xi ** 2))
#
#                 if m < 0:
#                     test = 1j / np.sqrt(2) * (sph_harm(m, l, phi, theta) - (-1) ** m * sph_harm(-m, l, phi, theta))
#                 elif m > 0:
#                     test = 1 / np.sqrt(2) * (sph_harm(-m, l, phi, theta) + (-1) ** m * sph_harm(m, l, phi, theta))
#                 else:
#                     test = sph_harm(m, l, phi, theta)
#
#                 ref.append(test)
#
#         x = np.array(x)
#         y = np.array(y)
#         z = pydgm.angle.p_leg[3]
#
#         print(len(z))
#
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         print(l ** 2 + l + 1 + m)
#         ax.plot(x, y, pydgm.angle.p_leg[(l ** 2 + l) + m], 'bo')
#         ax.plot(x, y, ref, 'r^')
#         plt.xlabel('$\mu$')
#         plt.ylabel('$\eta$')
# #         ax.plot(x, y, pydgm.angle.p_leg[2], 'go')
# #         ax.plot(x, y, pydgm.angle.p_leg[6], 'ro')
#         # ax.plot(x, y, pydgm.angle.p_leg[7], 'mo')
#         # ax.plot(x, y, pydgm.angle.p_leg[8], 'co')
#         plt.show()
#
#         for a in range(pydgm.control.number_angles):
#             ll = 0
#             for l in range(0, 2):
#                 for m in range(-l, l + 1):
#                     if m < 0:
#                         test = 1j / np.sqrt(2) * (sph_harm(m, l, phi[a], theta[a]) - (-1) ** m * sph_harm(-m, l, phi[a], theta[a]))
#                     elif m > 0:
#                         test = 1 / np.sqrt(2) * (sph_harm(-m, l, phi[a], theta[a]) + (-1) ** m * sph_harm(m, l, phi[a], theta[a]))
#                     else:
#                         test = sph_harm(m, l, phi[a], theta[a])
#
#                     norm = np.sqrt((2 * l + 1) / (4 * np.pi))
#                     with self.subTest(a=a, l=l, m=m):
#                         self.assertAlmostEqual(pydgm.angle.p_leg[ll, a], np.real(test) / norm, 8)
#                     ll += 1

    def tearDown(self):
        pydgm.angle.finalize_angle()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()

