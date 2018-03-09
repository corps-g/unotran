import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np

class TestSWEEPER(unittest.TestCase):
    
    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [10]
        pydgm.control.coarse_mesh = [0.0, 10.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/7gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
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
        pydgm.control.legendre_order = 0
        
        # Initialize the dependancies
        pydgm.solver.initialize_solver()
    
    def test_sweeper_computeEQ_dd(self):
        ''' 
        Test diamond difference equation
        '''
        pydgm.control.equation_type = 'DD'
        
        Ps = 0.0
        inc = np.array([0.0])
        
        Ps = pydgm.sweeper.computeeq(1.0, inc, 1.0, 1.0, 0.8611363115940526)
        
        self.assertAlmostEqual(inc[0], 0.734680275209795978, 12)
        self.assertAlmostEqual(Ps, 0.367340137604897989, 12)

    def test_sweeper_computeEQ_sc(self):
        ''' 
        Test the step characteristics equation
        '''
        pydgm.control.equation_type = 'SC'
        
        Ps = 0.0
        inc = np.array([0.0])
        
        Ps = pydgm.sweeper.computeeq(1.0, inc, 1.0, 1.0, 0.8611363115940526)
        
        self.assertAlmostEqual(inc[0], 0.686907416523104323, 12)
        self.assertAlmostEqual(Ps, 0.408479080928661801, 12)
        
    def test_sweeper_computeEQ_sd(self):
        ''' 
        Test the step difference equation
        '''
        pydgm.control.equation_type = 'SD'
        
        Ps = 0.0
        inc = np.array([0.0])
        
        Ps = pydgm.sweeper.computeeq(1.0, inc, 1.0, 1.0, 0.8611363115940526)
        
        self.assertAlmostEqual(inc[0], 0.5373061574106336, 12)
        self.assertAlmostEqual(Ps, 0.5373061574106336, 12)
    
    def test_sweeper_updateRHS(self):
        '''
        Check for correct source assuming a flat flux and fission allowed
        '''
        
        # Set scalar flux
        pydgm.state.d_phi[:] = 1.0
        phi = pydgm.state.d_phi[0]
        chi = pydgm.material.chi[:,0]
        f = pydgm.material.nu_sig_f[:,0]
        s = pydgm.material.sig_s[0,:,:,0].T
        Q = np.reshape(np.zeros(7*4*10), (7, 4, 10), 'F')
        
        source = np.ones(Q.shape)
        for c in range(10):
            F = chi * f.dot(phi[:,c])
            S = s.dot(phi[:,c])
            for a in range(4):
                source[:,a,c] += F + S
                
        pydgm.sweeper.updaterhs(Q, 7)
        
        np.testing.assert_array_almost_equal(Q, 0.5 * source, 12)
        
    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()
        
if __name__ == '__main__':
    
    unittest.main()

