import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np

class TestANGLE(unittest.TestCase):
    
    def setUp(self):
        pass
    
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

        pydgm.angle.initialize_angle()        
        pydgm.angle.initialize_polynomials(7)
        
        # Get the basis set
        basis = pydgm.angle.p_leg
        
        # Use numpy to get the test basis
        x, wt = np.polynomial.legendre.leggauss(nAngle * 2)
        I = np.eye(nAngle * 2)
        test_basis = np.array([np.polynomial.legendre.legval(x,I[i]) * (-1) ** i for i in range(nAngle)])
      
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
        
if __name__ == '__main__':
    
    unittest.main()

