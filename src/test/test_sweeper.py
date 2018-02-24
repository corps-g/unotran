import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np

class TestSWEEPER(unittest.TestCase):
    
    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/3gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = True
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.equation_type = 'DD'
        pydgm.control.lamb = 1.0
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.legendre_order = 0
        pydgm.control.use_dgm = False
        
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
    
    def test_sweeper_sweep_R(self):
        '''
        Test the sweep through cells and angles with reflecting conditions and DD
        '''
        g = 1
        source = np.ones((1, 4), order='F') * 1.2760152893 * 0.5
        phi_g = np.array([1.0])
        psi_g = np.ones((1, 4), order='F') * 0.5
        incident = np.ones((2), order='F') * 0.5

        pydgm.sweeper.sweep(g, source, phi_g, psi_g, incident)
        
        np.testing.assert_array_almost_equal(phi_g, 2.4702649838962234, 12)
        psi_test = np.array([0.74789724066688767, 1.0164427642376821, 1.1738899625537731, 1.7463807889213092])
        np.testing.assert_array_almost_equal(psi_g.flatten(), psi_test, 12)
        np.testing.assert_array_almost_equal(incident, [1.3519854437737708, 1.9598760493672542])
        
    def test_sweeper_sweep_V(self):
        '''
        Test the sweep through cells and angles with vacuum conditions and DD
        '''
        
        pydgm.control.boundary_type = [0.0, 0.0]
        g = 1
        source = np.ones((1, 4), order='F') * 1.2760152893 * 0.5
        phi_g = np.array([1.0])
        psi_g = np.ones((1, 4), order='F') * 0.5
        incident = np.ones((2), order='F') * 0.5

        pydgm.sweeper.sweep(g, source, phi_g, psi_g, incident)
        
        np.testing.assert_array_almost_equal(phi_g, 1.0863050345964158, 12)
        psi_test = np.array([0.31829108536954637, 0.6630938187058939, 0.31829108536954637, 0.6630938187058939])
        np.testing.assert_array_almost_equal(psi_g.flatten(), psi_test, 12)
        np.testing.assert_array_almost_equal(incident, [0.6365821707390927, 1.3261876374117878])
        
    def tearDown(self):
        pydgm.solver.finalize_solver()
        pydgm.control.finalize_control()
        
if __name__ == '__main__':
    
    unittest.main()

