import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np


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
        pydgm.control.angle_order = 8

    def test_angle_quadrature(self):
        '''
        Test for correct gauss legendre points and weights
        '''
        # setup control parameters for quadrature
        pydgm.control.angle_option = pydgm.angle.gl

        # build quadrature, test it, and deallocate
        pydgm.angle.initialize_angle()

        print(pydgm.angle.mu)
        print(pydgm.angle.eta)
        print(pydgm.angle.wt)
        stop()

        # Get the test quadrature weights from numpy
        N_polar, N_azi = 4, 8
        phia, step = np.linspace(0, np.pi / 2, N_azi, endpoint=False, retstep=True)
        phia += 0.5 * step
        w_azi = np.ones(N_azi) * (np.pi / N_azi) / 2
        xi, w_pol = np.polynomial.legendre.leggauss(2 * N_polar)
        xi = xi / 2 + 0.5
        mu = np.zeros(N_polar * N_azi)
        eta = 1 * mu
        wt = 1 * mu
        for p in range(N_polar):
            for a in range(N_azi):
                wt[a + p * N_azi] = w_pol[p] * w_azi[a]
                mu[a + p * N_azi] = np.cos(phia[a]) * np.sqrt(1 - xi[p] ** 2)
                eta[a + p * N_azi] = np.sin(phia[a]) * np.sqrt(1 - xi[p] ** 2)
        N_angle = N_polar * N_azi

        # Test for equivalance
        np.testing.assert_array_almost_equal(pydgm.angle.mu, mu, 12)
        np.testing.assert_array_almost_equal(pydgm.angle.eta, eta, 12)
        np.testing.assert_array_almost_equal(pydgm.angle.wt, wt, 12)

    def tearDown(self):
        pydgm.angle.finalize_angle()
        pydgm.control.finalize_control()


if __name__ == '__main__':

    unittest.main()

