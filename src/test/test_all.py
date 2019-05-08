import unittest
import argparse
from test_angle import TestANGLE_1D, TestANGLE_2D
from test_mesh import TestMESH
from test_material import TestMATERIAL
from test_state import TestSTATE
from test_sources import TestSOURCES, TestSOURCESdgm
from test_sweeper_1D import TestSWEEPER_1D
from test_sweeper_2D import TestSWEEPER_2D
from test_mg_solver import TestMG_SOLVER
from test_solver import TestSOLVER, TestSOLVER_2D
from test_dgm import TestDGM, TestDGM2, TestDGM_2D
from test_dgmsolver import TestDGMSOLVER, TestDGMSOLVER_2D


def AllSuite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestANGLE_1D))
    suite.addTests(unittest.makeSuite(TestANGLE_2D))
    suite.addTests(unittest.makeSuite(TestMESH))
    suite.addTests(unittest.makeSuite(TestMATERIAL))
    suite.addTests(unittest.makeSuite(TestSTATE))
    suite.addTests(unittest.makeSuite(TestSOURCES))
    suite.addTests(unittest.makeSuite(TestSOURCESdgm))
    suite.addTests(unittest.makeSuite(TestSWEEPER_1D))
    suite.addTests(unittest.makeSuite(TestSWEEPER_2D))
    suite.addTests(unittest.makeSuite(TestMG_SOLVER))
    suite.addTests(unittest.makeSuite(TestSOLVER))
    suite.addTests(unittest.makeSuite(TestSOLVER_2D))
    suite.addTests(unittest.makeSuite(TestDGM))
    suite.addTests(unittest.makeSuite(TestDGM2))
    suite.addTests(unittest.makeSuite(TestDGM_2D))
    suite.addTests(unittest.makeSuite(TestDGMSOLVER))
    suite.addTests(unittest.makeSuite(TestDGMSOLVER_2D))

    return suite


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='store_true')
    options = parser.parse_args()
    v = options.v + 1

    runner = unittest.TextTestRunner(verbosity=v)

    test_suite = AllSuite()

    runner.run(test_suite)
