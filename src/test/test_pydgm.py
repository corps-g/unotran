import sys
sys.path.append('../')

import unittest

import pydgm

class TestDGM(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_angle_legendre_p(self):
        """ Some useful description.
        """
        
        angle = pydgm.angle
        
        self.assertEqual(angle.legendre_p(0, 0.5), 1)
        self.assertEqual(angle.legendre_p(1, 0.5), 0.5)
        
        # do more!
        
    def test_angle_quadrature(self):
        """ lala
        """
        
        # setup control parameters for quadrature
        pydgm.control.angle_order = 1
        pydgm.control.angle_option = pydgm.angle.gl
        
        # build quadrature, test it, and deallocate
        pydgm.angle.initialize_angle()
        self.assertAlmostEqual(pydgm.angle.mu[0], 0.57735026918962573)
        pydgm.angle.finalize_angle()
        
if __name__ == '__main__':
    
    unittest.main()

