import unittest
from test_angle import TestANGLE
from test_state import TestSTATE

def AllSuite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestANGLE))
    suite.addTests(unittest.makeSuite(TestSTATE))
    
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()

    test_suite = AllSuite()

    runner.run(test_suite)